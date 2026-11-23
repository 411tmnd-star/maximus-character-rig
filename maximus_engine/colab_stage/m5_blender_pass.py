"""M5 Blender passes (run by m5_anthro_regressor_and_walk.py). bpy.data / pose-bone data and keyframe_insert only; operators only through
io_ops_gateway (save_copy, render_still).

modes
  corners  the 32 corners of the 5 anthropometric keys (2^5), evaluated at bind (rig REST)            -> m5_corners.npz
  walk     solve the runway walk on maximus_v2_master_psd.blend (m5_gait), keyframe it, save a walk scene that the UNCHANGED harness
           capture script (avatar_engine/harness/capture_walk_blender.py) can read                   -> m5_walk_scene.blend, m5_walk_solve.npz/json
  breath   90-frame 24 fps Living Statue loop (+0.8 deg spine02), GPU Cycles stills                    -> breath/frame_NNN.png, breath.json
"""
import itertools, json, math, sys, time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
A = dict(x.split("=", 1) for x in sys.argv[sys.argv.index("--") + 1:] if "=" in x)
ROOT = Path(A["root"]).resolve(); sys.path.insert(0, str(ROOT / "pipeline")); sys.path.insert(0, str(HERE))
for p in (HERE / "runtime_v3", HERE.parent / "runtime_v3"):
    if (p / "pose_space_engine.py").exists():
        sys.path.insert(0, str(p))
import bl_common as C                     # noqa: E402
import io_ops_gateway as io               # noqa: E402
import m5_gait as G                       # noqa: E402
from corpus_bridge import contracts       # noqa: E402

OUT = Path(A["out_dir"]); OUT.mkdir(parents=True, exist_ok=True)
ANTHRO = ("gt_lean", "gt_stocky_nat", "gt_stocky_ext", "leg_len_m5cm", "leg_len_p5cm")


def invariants(body, rig):
    arm = [m for m in body.modifiers if m.type == "ARMATURE"]; cs = [m for m in body.modifiers if m.type == "CORRECTIVE_SMOOTH"]
    assert len(body.data.vertices) == 13380 and len(rig.data.bones) == 152, "13,380 verts / 152 bones"
    assert len(arm) == 1 and not arm[0].use_deform_preserve_volume, "pure LBS (no DQS)"
    assert len(cs) == 1 and abs(cs[0].factor - 0.5) < 1e-9 and cs[0].iterations == 5, "Corrective Smooth 0.50 x 5"
    return {"verts": 13380, "bones": 152, "lbs_only": True, "cs": [cs[0].factor, cs[0].iterations]}


def eval_world(body):
    return C.evaluated_positions(body)


# ------------------------------------------------------------------------------------------------------------------------------------
def mode_corners():
    body, rig = C.human_and_rig(); inv = invariants(body, rig); me = body.data; rig.data.pose_position = "REST"
    kbs = me.shape_keys.key_blocks
    for k in kbs:
        k.value = 0.0
    corners = np.array(list(itertools.product([0.0, 1.0], repeat=5)))
    Vc = np.zeros((32, len(me.vertices), 3))
    for ci, w in enumerate(corners):
        for name, val in zip(ANTHRO, w):
            kbs[name].value = float(val)
        Vc[ci] = eval_world(body)
    for k in kbs:
        k.value = 0.0
    arr = C.mesh_arrays(me)
    W, gnames = C.weights_matrix(body)
    np.savez_compressed(OUT / "m5_corners.npz", corners=corners, V=Vc, T=arr["T"], key_names=np.array(ANTHRO),
                        W=W.astype(np.float32), group_names=np.array(gnames))
    return {"npz": str(OUT / "m5_corners.npz")}, {"corners": 32, "invariants": inv}


# ------------------------------------------------------------------------------------------------------------------------------------
def foot_sets(body):
    vg = {g.name: g.index for g in body.vertex_groups}; out = {}
    for s in "LR":
        ids = {vg[n] for n in vg if (n.startswith(f"foot.{s}") or n.startswith("toe")) and n.endswith("." + s)}
        out[s] = np.array([v.index for v in body.data.vertices if any(g.group in ids and g.weight > 0.5 for g in v.groups)], np.int64)
    return out


def ensure_volume_psd_keys(body, rig, hip_mm=6.5, axilla_mm=5.5, crease_mm=3.8, smooth_iters=3):
    """Ensure the 4 named volume-preserving PSD keys (PSD_hip_L/R, PSD_axilla_L/R) exist on body and attach Blender Simple-Expression
    quaternion drivers linked to upperleg01/upperarm01 LOCAL_SPACE ROT_W (ZERO bpy.ops; evaluates to 0.0 automatically at bind pose!)."""
    me = body.data; kbs = me.shape_keys.key_blocks
    names_req = ("PSD_hip_L", "PSD_hip_R", "PSD_axilla_L", "PSD_axilla_R")
    synthesized = not all(n in kbs for n in names_req)
    arr = C.mesh_arrays(me); V = arr["V"].astype(np.float64); T = arr["T"].astype(np.int64)
    W, gnames = C.weights_matrix(body); W = W.astype(np.float64)
    fn = np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]])
    vn = np.zeros_like(V)
    for c in range(3):
        np.add.at(vn, T[:, c], fn)
    vn /= np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-12)
    adj = [set() for _ in range(len(V))]
    for a, b, c in T:
        adj[a].update((b, c)); adj[b].update((a, c)); adj[c].update((a, b))

    def blend(ga, gb, guard):
        ca = [i for i, n in enumerate(gnames) if n in ga]
        cb = [i for i, n in enumerate(gnames) if n in gb]
        w = 4.0 * W[:, ca].sum(1) * W[:, cb].sum(1) * guard
        for _ in range(smooth_iters):
            w = np.array([0.5 * w[i] + 0.5 * (sum(w[j] for j in adj[i]) / max(len(adj[i]), 1)) for i in range(len(V))])
        return np.clip(w * guard, 0.0, 1.0)

    c_ul = [i for i, n in enumerate(gnames) if n.startswith("upperleg0")]
    c_sp = [i for i, n in enumerate(gnames) if n in ("spine05", "pelvis.L", "pelvis.R", "root")]
    w_ul = W[:, c_ul].sum(1); w_sp = W[:, c_sp].sum(1)
    crease_base = ((V[:, 2] > 0.795) & (V[:, 2] < 0.855) & (np.abs(V[:, 0]) < 0.038)
                   & (V[:, 1] > -0.09) & (V[:, 1] < 0.01) & (w_ul > 0.10) & (w_sp > 0.15)).astype(float)
    for _ in range(2):
        crease_base = np.array([0.6 * crease_base[i] + 0.4 * (sum(crease_base[j] for j in adj[i]) / max(len(adj[i]), 1)) for i in range(len(V))])
    crease_L = np.where(V[:, 0] >= -0.002, crease_base, 0.0)
    crease_R = np.where(V[:, 0] <= 0.002, crease_base, 0.0)

    hip_L_guard = np.where((V[:, 2] < 0.88) & ((V[:, 0] < 0.055) | (vn[:, 0] < -0.25)), 0.0, 1.0)
    hip_R_guard = np.where((V[:, 2] < 0.88) & ((V[:, 0] > -0.055) | (vn[:, 0] > 0.25)), 0.0, 1.0)
    ax_L_guard  = np.where((V[:, 2] < 1.34) & (V[:, 0] > 0.11) & ((vn[:, 2] < -0.20) | (vn[:, 0] < -0.25)), 0.0, 1.0)
    ax_R_guard  = np.where((V[:, 2] < 1.34) & (V[:, 0] < -0.11) & ((vn[:, 2] < -0.20) | (vn[:, 0] > 0.25)), 0.0, 1.0)

    specs = {
        "PSD_hip_L":    (hip_mm * 1e-3,    ("upperleg01.L", "upperleg02.L"), ("root", "pelvis.L", "spine05", "spine04"), hip_L_guard, crease_L),
        "PSD_hip_R":    (hip_mm * 1e-3,    ("upperleg01.R", "upperleg02.R"), ("root", "pelvis.R", "spine05", "spine04"), hip_R_guard, crease_R),
        "PSD_axilla_L": (axilla_mm * 1e-3, ("upperarm01.L", "upperarm02.L"), ("spine01", "spine02", "spine03", "clavicle.L", "shoulder01.L"), ax_L_guard, None),
        "PSD_axilla_R": (axilla_mm * 1e-3, ("upperarm01.R", "upperarm02.R"), ("spine01", "spine02", "spine03", "clavicle.R", "shoulder01.R"), ax_R_guard, None),
    }
    for kname, (amp, ga, gb, gd, cr) in specs.items():
        kb = kbs[kname] if kname in kbs else body.shape_key_add(name=kname, from_mix=False)
        kb.slider_min = 0.0; kb.slider_max = 1.0; kb.value = 0.0; kb.relative_key = kbs["Basis"]
        w_out = amp * blend(ga, gb, gd)
        if cr is not None:
            w_out = w_out - (crease_mm * 1e-3) * cr
        delta = w_out[:, None] * vn
        kb.data.foreach_set("co", (V + delta).astype(np.float32).ravel())
    me.update()

    # Attach Simple-Expression bone-rotation drivers (automatically 0.0 when pose bones are reset to identity at bind!)
    driver_map = {
        "PSD_hip_L":    ("upperleg01.L", 0.006),
        "PSD_hip_R":    ("upperleg01.R", 0.006),
        "PSD_axilla_L": ("upperarm01.L", 0.080),
        "PSD_axilla_R": ("upperarm01.R", 0.080),
    }
    for kname, (bname, denom) in driver_map.items():
        kb = kbs[kname]
        kb.driver_remove("value")
        fc = kb.driver_add("value")
        d = fc.driver; d.type = "SCRIPTED"
        v = d.variables.new(); v.name = "w"; v.type = "TRANSFORMS"
        t = v.targets[0]; t.id = rig; t.bone_target = bname
        t.transform_type = "ROT_W"; t.rotation_mode = "QUATERNION"; t.transform_space = "LOCAL_SPACE"
        d.expression = f"min(1.0, max(0.0, (1.0 - abs(w)) / {denom}))"
    return list(names_req), synthesized


def mode_walk():
    import pose_space_engine as PSE
    from maximus_core import geometry
    body, rig = C.human_and_rig(); inv = invariants(body, rig); me = body.data; sc = C.bpy.context.scene
    rig.data.pose_position = "POSE"
    preset = A.get("preset", "BASELINE_M5")
    fps = 24.0; n = int(A.get("frames", "250"))
    plan_kw = {"fps": fps, "n_frames": n}
    if "stride_s" in A:
        plan_kw["T"] = float(A["stride_s"])
    if "step_m" in A:
        plan_kw["step"] = float(A["step_m"])
    if "width_m" in A:
        plan_kw["width"] = float(A["width_m"])
    plan = G.GaitPlan.from_preset(preset, **plan_kw)
    T = plan.T
    bones = [b.name for b in rig.data.bones]; par = [bones.index(b.parent.name) if b.parent else -1 for b in rig.data.bones]
    rest = np.stack([np.array(b.matrix_local, float) for b in rig.data.bones])
    S = G.Skeleton(bones, par, rest); fit = json.loads(Path(A["theta_fit"]).read_text())["bones"]
    for name, v in fit.items():
        if name in S.idx:
            S.set_basis_qt(name, v["q_wxyz"], v.get("t", (0, 0, 0)))
    for c in ("clavicle.L", "clavicle.R"):                      # theta_fit is a static pose: its clavicles are already 0; keep that base
        S.set_basis_qt(c, (1, 0, 0, 0))
    S.fk()
    # Armpit & thigh clearance guard
    arm_guard_deg = float(A.get("arm_guard_deg", str(plan.arm_guard_deg)))
    for s, sgn in (("L", 1), ("R", -1)):
        S.rot_world(f"upperarm01.{s}", G.rot(G.FWD, sgn * math.radians(arm_guard_deg)))
    base_basis = S.basis.copy()
    psd = PSE.PSDModel.from_json(me["maximus_psd"]) if "maximus_psd" in me.keys() else None
    psd_joints = sorted({j for f in (psd.families.values() if psd else []) for j in f["joints"]})
    vol_psd_names, vol_psd_synth = ensure_volume_psd_keys(body, rig)
    kbs = me.shape_keys.key_blocks
    arr_base = C.mesh_arrays(me); V_basis = arr_base["V"].astype(np.float64); T_mesh = arr_base["T"].astype(np.int64)
    w_psd_fit = {}
    if psd:
        lq_fit = {j: S.local_quat(j) for j in psd_joints}
        w_psd_fit = {k_: float(val) for k_, val in psd.key_weights(lq_fit).items() if k_ in kbs and k_ not in vol_psd_names}
    # rig geometry
    a = {s: np.linalg.norm(S.head(f"lowerleg01.{s}") - S.head(f"upperleg01.{s}")) for s in "LR"}
    b = {s: np.linalg.norm(S.head(f"foot.{s}") - S.head(f"lowerleg01.{s}")) for s in "LR"}
    feet = foot_sets(body)
    vg_map = {g.name: g.index for g in body.vertex_groups}
    hand_ids = {vg_map[nm] for nm in vg_map if nm.startswith(("wrist", "finger", "metacarpal"))}
    thigh_ids = {vg_map[nm] for nm in vg_map if nm.startswith(("upperleg", "pelvis", "root"))}
    hand_verts = np.array([v.index for v in me.vertices if any(g.group in hand_ids and g.weight > 0.35 for g in v.groups)], dtype=np.int64)
    thigh_verts = np.array([v.index for v in me.vertices if any(g.group in thigh_ids and g.weight > 0.35 for g in v.groups)], dtype=np.int64)
    # reference standing state at theta_fit (write once, evaluate the mesh once)
    def write_pose():
        pbs = rig.pose.bones
        for i, nm in enumerate(bones):
            pb = pbs[nm]; pb.rotation_mode = "QUATERNION"; pb.rotation_quaternion = G.mat_to_quat(S.basis[i][:3, :3]); pb.location = S.basis[i][:3, 3]
        C.bpy.context.view_layer.update()
    write_pose(); Vfit = eval_world(body)
    sole0 = {s: float(Vfit[feet[s], 2].min()) for s in "LR"}
    ankle_ref = {s: S.head(f"foot.{s}") for s in "LR"}; Rf_ref = {s: S.world_rot(f"foot.{s}") for s in "LR"}
    toe_roots = {s: [f"toe{i}-1.{s}" for i in range(1, 6)] for s in "LR"}
    p_toe = {s: Rf_ref[s].T @ (np.mean([S.head(t) for t in toe_roots[s]], axis=0) - ankle_ref[s]) for s in "LR"}
    z_ankle = {s: ankle_ref[s][2] - sole0[s] for s in "LR"}
    root_fit = S.head("root"); hip_fit = {s: S.head(f"upperleg01.{s}") for s in "LR"}
    head_bind_R = rest[S.idx["head"]][:3, :3]
    y_start = 0.0
    def plant(side, k):
        x = (1 if side == "L" else -1) * plan.width / 2
        return np.array([x, y_start - plan.plant_y(side, k), z_ankle[side]])
    delta_hip_y = float(np.mean([hip_fit[s][1] - ankle_ref[s][1] for s in "LR"]))
    v = plan.v; fr = 1.0 / fps
    A_sway = plan.sway_m
    A_tilt = math.radians(plan.pelvis_tilt_deg)
    A_yaw = math.radians(plan.pelvis_yaw_deg)
    A_chest = math.radians(plan.chest_world_yaw_deg)
    A_arm = math.radians(plan.arm_swing_deg)
    chest_lift_rad = math.radians(plan.chest_lift_deg)
    if preset == "RUNWAY_CONFIDENT_V2":
        # Exact stride fractions on 27-frame (1.125s) cycle for S3 phase overlap
        d_c = 0.050 * T
        d_sh = d_c + 0.040 * T
        d_el = d_sh + 0.090 * T
        d_wr = d_el + 0.050 * T
        d_fg = d_wr + 0.040 * T
        d_hd = d_c + 0.050 * T
    else:
        d_c = 3 * fr; d_sh = d_c + 2 * fr; d_el = d_sh + 5 * fr; d_wr = d_el + 3 * fr; d_fg = d_wr + 2 * fr; d_hd = d_c + 3 * fr
    w = 2 * math.pi / T

    def foot_target(side, t):
        st = plan.foot(side, t); sgn = 1 if side == "L" else -1
        lat = Rf_ref[side] @ np.array([1.0, 0, 0]); lat[2] = 0.0; lat /= np.linalg.norm(lat)
        if st["mode"] == "stance":
            Ap = plant(side, st["k"])
            if st["pivot"] == "flat":
                return Ap.copy(), Rf_ref[side].copy(), st, 0.0
            P = Ap + Rf_ref[side] @ p_toe[side]; Rp = G.rot(lat, -st["pitch"])
            A_ = P + Rp @ (Ap - P); A_[2] += st["z_lift"]
            return A_, Rp @ Rf_ref[side], st, st["pitch"]
        Ap0 = plant(side, st["k"]); P0 = Ap0 + Rf_ref[side] @ p_toe[side]; Rto = G.rot(lat, plan.toe_off)
        A_to = P0 + Rto @ (Ap0 - P0); A_land = plant(side, st["k"] + 1); s = st["s"]
        s_drv = st.get("s_drive", s)
        q = G.smoothstep5(s_drv)
        A_ = A_to + (A_land - A_to) * q; A_[0] += sgn * plan.bulge * math.sin(math.pi * s); A_[2] += plan.lift_side[side] * math.sin(math.pi * s_drv)
        toe_rel = -plan.toe_off * (1 - G.smoothstep5(min(1.0, s / 0.35)))
        return A_, G.rot(lat, -st["pitch"]) @ Rf_ref[side], st, toe_rel

    def pelvis(t):
        x = A_sway * math.cos(w * (t - 0.3 * T)); y = y_start + delta_hip_y - v * (t - 0.3 * T)
        yaw = -A_yaw * math.cos(w * t); tilt = A_tilt * math.cos(w * (t - 0.3 * T)); return x, y, yaw, tilt

    # ---- pass 1: root height (bob) from reachability of the contact legs, then periodic low-pass
    def hip_offsets(yaw, tilt):
        R = G.rot(G.UP, yaw) @ G.rot(G.FWD, tilt); return {s: R @ (hip_fit[s] - root_fit) for s in "LR"}, R
    reach_ratio = 0.996 if preset == "RUNWAY_CONFIDENT_V2" else 0.982
    zmax = np.zeros(n)
    for f in range(n):
        t = f * fr; x, y, yaw, tilt = pelvis(t); off, _ = hip_offsets(yaw, tilt); lim = []
        for s in "LR":
            A_, _, st, _ = foot_target(s, t)
            if st["mode"] == "stance":
                hx, hy = x + off[s][0], y + off[s][1]; r = reach_ratio * (a[s] + b[s]); dxy2 = (A_[0] - hx) ** 2 + (A_[1] - hy) ** 2
                lim.append(A_[2] + math.sqrt(max(r * r - dxy2, 0.0)) - off[s][2])
        zmax[f] = min(lim)
    zs = G.periodic_lowpass(zmax, T * fps, 6)
    for _ in range(50):
        over = zs - zmax
        if over.max() <= 0:
            break
        zs -= over.max() + 1e-5
    if preset == "RUNWAY_CONFIDENT_V2":
        # Damp excessive vertical bob from longer 0.68m stride while keeping strict reach <= zmax (2-4 deg heel-strike knee extension)
        z_hi = float(zs.max())
        zs_damped = z_hi - 0.52 * (z_hi - zs)
        zs = np.minimum(zs_damped, zmax - 1e-5)
        zs = G.periodic_lowpass(zs, T * fps, 4)
        for _ in range(50):
            over = zs - zmax
            if over.max() <= 0:
                break
            zs -= over.max() + 1e-5
    root_z = zs

    # ---- pass 2: pose every frame
    def apply_frame(f, foot_bias, lat_corr=None):
        t = f * fr; S.basis[:] = base_basis; S.fk()
        x, y, yaw, tilt = pelvis(t)
        S.rot_world("root", G.rot(G.UP, yaw) @ G.rot(G.FWD, tilt)); S.translate_world("root", np.array([x, y, root_z[f]]) - S.head("root"))
        chest = A_chest * math.cos(w * (t - d_c)); rel = chest - yaw; roll = -0.8 * tilt
        for nm, fshare in (("spine05", 0.15), ("spine04", 0.20), ("spine03", 0.25), ("spine02", 0.20), ("spine01", 0.20)):
            lift_add = (-chest_lift_rad if nm == "spine02" else 0.0)
            S.rot_world(nm, G.rot(G.UP, fshare * rel) @ G.rot(G.FWD, fshare * roll) @ G.rot(G.LAT, lift_add + fshare * math.radians(0.4) * math.sin(2 * w * t)))
        info = {"t": t, "reach_clamped": []}
        for s in "LR":
            A_, Rf, st, toe_rel = foot_target(s, t); A_ = A_ + np.array([0, 0, foot_bias.get(s, 0.0)])
            sgn = 1 if s == "L" else -1
            pole_lat = 0.28 if preset == "RUNWAY_CONFIDENT_V2" else 0.25
            pole = G.FWD + (pole_lat * sgn * G.LAT if st["mode"] == "swing" else 0.05 * sgn * G.LAT)
            Sh = S.head(f"upperleg01.{s}"); K, cl = G.two_bone_ik(Sh, A_, a[s], b[s], pole)
            if cl:
                info["reach_clamped"].append(s)
            S.rot_world(f"upperleg01.{s}", G.align_rot(S.head(f"lowerleg01.{s}") - Sh, K - Sh))
            S.rot_world(f"lowerleg01.{s}", G.align_rot(S.head(f"foot.{s}") - K, A_ - K))
            S.rot_world(f"foot.{s}", Rf @ S.world_rot(f"foot.{s}").T)
            lat = Rf_ref[s] @ np.array([1.0, 0, 0]); lat[2] = 0; lat /= np.linalg.norm(lat)
            for tr in toe_roots[s]:
                S.rot_world(tr, G.rot(lat, toe_rel))
            info[f"mode_{s}"] = st["mode"]; info[f"lower_{s}"] = bool(st["lower"]); info[f"contact_{s}"] = st["mode"] == "stance"
        for s, sgn, amp in (("L", 1, 1.0), ("R", -1, 1.03)):
            ph = math.pi if s == "L" else 0.0
            if preset == "RUNWAY_CONFIDENT_V2":
                arm = math.radians(plan.arm_pendulum_deg(w * (t - d_sh) + ph, amp_scale=amp))
                elb_dyn = math.radians(0.5 * (plan.elbow_flex_range_deg[1] - plan.elbow_flex_range_deg[0]) * amp * (1.0 + math.cos(w * (t - d_el) + ph)) / 2.0)
                wr_dyn = math.radians(1.2) * amp * math.cos(w * (t - d_wr) + ph)
            else:
                arm = A_arm * amp * (math.cos(w * (t - d_sh) + ph) + 0.30 * math.cos(2 * w * (t - d_sh) + ph + 0.6))
                elb_dyn = math.radians(4.0) * amp * (1 + math.cos(w * (t - d_el) + ph)) / 2
                wr_dyn = math.radians(2.5) * amp * math.cos(w * (t - d_wr) + ph)
            if not plan.clavicle_lock:
                S.rot_world(f"clavicle.{s}", G.rot(G.FWD, sgn * math.radians(0.8) * amp * math.cos(w * (t - d_sh) + ph)))
            else:
                S.set_basis_qt(f"clavicle.{s}", (1.0, 0.0, 0.0, 0.0))
                S.fk()
            S.rot_world(f"upperarm01.{s}", G.rot(G.LAT, -arm))
            S.rot_world(f"lowerarm01.{s}", G.rot(G.LAT, -elb_dyn))
            S.rot_world(f"wrist.{s}", G.rot(G.LAT, -wr_dyn))
            for i in range(1, 6):
                fb = f"finger{i}-1.{s}"; M = S.basis[S.idx[fb]].copy(); M[:3, :3] = M[:3, :3] @ G.rot([1, 0, 0], math.radians(1.4) * amp * math.cos(w * (t - d_fg) + ph))
                S.basis[S.idx[fb]] = M
            S.fk()
        corr_used = 0.0
        lat_keep = (1.0 - plan.head_stabilization) if preset == "RUNWAY_CONFIDENT_V2" else 0.35
        for it2 in range(2):                                   # head: lateral position (spine03 roll), then orientation (S7)
            if lat_corr is None:                               # pass 1: feedback, recorded only
                hx_target = lat_keep * x; hx = S.head("head")[0]; lever = max(S.head("head")[2] - S.head("spine03")[2], 0.2)
                dr = -(hx_target - hx) / lever; S.rot_world("spine03", G.rot(G.FWD, dr)); corr_used += dr
            elif it2 == 0:                                     # pass 2: the cycle-smoothed correction (feedback alone jittered: S9 on spine03)
                S.rot_world("spine03", G.rot(G.FWD, lat_corr)); corr_used = lat_corr
            yaw_h = math.radians(0.95) * math.cos(w * (t - d_hd)); nod = math.radians(0.55) * math.sin(2 * w * t)   # S5: head and neck must move > 0.5 deg while keeping level 0 deg mean chin pitch
            Rt = G.rot(G.UP, yaw_h) @ G.rot(G.LAT, nod) @ head_bind_R; Rc = Rt @ S.world_rot("head").T
            ang = math.acos(max(-1.0, min(1.0, (np.trace(Rc) - 1) / 2)))
            if ang > 1e-9:
                axv = np.array([Rc[2, 1] - Rc[1, 2], Rc[0, 2] - Rc[2, 0], Rc[1, 0] - Rc[0, 1]]); axv /= np.linalg.norm(axv)
                S.rot_world("neck01", G.rot(axv, 0.5 * ang) @ G.rot(G.LAT, 0.5 * nod))
            S.rot_world("head", (Rt @ S.world_rot("head").T))
        info["lat_corr"] = corr_used
        if psd:
            lq = {j: S.local_quat(j) for j in psd_joints}
            for k_, val in psd.key_weights(lq).items():
                if k_ in kbs and k_ not in vol_psd_names:
                    if "__theta_fit" in k_:
                        val = max(float(val), w_psd_fit.get(k_, 0.0))
                    kbs[k_].value = float(val)
        return info

    t0 = time.time()
    lat_raw = np.array([apply_frame(f, {})["lat_corr"] for f in range(n)])          # pass 1: FK only, no mesh evaluation
    lat_s = G.periodic_lowpass(lat_raw, T * fps, 3)
    rows = []; rig_z = np.zeros(n); KEYED = None
    max_g5 = 0
    min_thigh_clear_mm = 999.0
    clav_locked_all = True
    for f in range(n):
        bias = {}; rig.location = (0.0, 0.0, 0.0)                 # measure every frame from an unshifted rig
        for it in range(3):
            info = apply_frame(f, bias, float(lat_s[f])); write_pose(); V = eval_world(body)
            zf = {s: float(V[feet[s], 2].min()) for s in "LR"}
            lower = "L" if info["lower_L"] else "R"; other = "R" if lower == "L" else "L"
            shift = -zf[lower]; zo = zf[other] + shift
            if ((info[f"contact_{other}"] and not (0.0003 <= zo <= 0.0015)) or zo < 0.0003) and it < 2:   # swing foot may not dip under the floor either
                bias[other] = bias.get(other, 0.0) + (plan.contact_lift - zo); continue
            break
        rig_z[f] = shift; rig.location = (0.0, 0.0, shift)
        col = geometry.collapsed_triangles(V_basis, V, T_mesh, ratio=0.25)
        g5_f = int(col["count"])
        max_g5 = max(max_g5, g5_f)
        if len(hand_verts) and len(thigh_verts):
            # Fast pairwise distance between hand/wrist and thigh/hip vertices
            dh = V[hand_verts][:, None, :] - V[thigh_verts][None, :, :]
            d_mm = float(np.sqrt((dh * dh).sum(axis=-1)).min() * 1000.0)
            min_thigh_clear_mm = min(min_thigh_clear_mm, d_mm)
        for cn in ("clavicle.L", "clavicle.R"):
            qc = rig.pose.bones[cn].rotation_quaternion
            if abs(float(qc.w) - 1.0) > 1e-6 or max(abs(float(qc.x)), abs(float(qc.y)), abs(float(qc.z))) > 1e-6:
                clav_locked_all = False
        rows.append({"frame": f + 1, "lower": lower, "z_lower_mm": 0.0, "z_other_mm": round(zo * 1e3, 4), "contact_other": info[f"contact_{other}"],
                     "reach_clamped": info["reach_clamped"], "iterations": it + 1, "min_body_z_mm": round((float(V[:, 2].min()) + shift) * 1e3, 4),
                     "g5": g5_f})
        # keyframes (data API)
        fnum = f + 1
        if KEYED is None:
            KEYED = [nm for i, nm in enumerate(bones) if not np.allclose(S.basis[i], base_basis[i], atol=1e-12) or nm in ("root",)]
            KEYED = sorted(set(KEYED) | {nm for nm in bones if nm.startswith(("upperleg", "lowerleg", "foot", "toe", "upperarm", "lowerarm", "wrist", "finger",
                                                                                  "clavicle", "spine", "neck", "head", "root"))})
        for nm in KEYED:
            pb = rig.pose.bones[nm]; pb.keyframe_insert("rotation_quaternion", frame=fnum)
            if nm == "root":
                pb.keyframe_insert("location", frame=fnum)
        rig.keyframe_insert("location", frame=fnum)
        if psd:
            for k_ in psd.key_weights({j: S.local_quat(j) for j in psd_joints}):
                if k_ in kbs and k_ not in vol_psd_names:
                    kbs[k_].keyframe_insert("value", frame=fnum)
        if f % 25 == 0:
            print(f"PROGRESS walk[{preset}] frame {fnum}/{n} lower={lower} z_other={zo * 1e3:.3f}mm G5={g5_f} clamped={info['reach_clamped']} {time.time() - t0:.0f}s", flush=True)
    sc.frame_start, sc.frame_end = 1, n; sc.render.fps = int(fps)
    io.save_copy(OUT / "m5_walk_scene.blend")
    clamped = sum(1 for r in rows if r["reach_clamped"])
    rep = {"preset": preset,
           "plan": {"preset": preset, "stride_s": T, "cadence_bpm": round(plan.cadence_steps_per_min, 2), "fps": fps, "frames": n,
                    "step_m": plan.step, "width_m": plan.width, "width_cm": round(plan.width * 100.0, 2), "stance": plan.stance,
                    "speed_m_s": round(v, 4), "sway_mm_each_way": round(A_sway * 1e3, 2),
                    "pelvis_tilt_deg": round(math.degrees(A_tilt), 2), "pelvis_yaw_deg": round(math.degrees(A_yaw), 2),
                    "spine_counter_yaw_deg": round(plan.spine_counter_yaw_deg, 2), "chest_world_yaw_deg": round(math.degrees(A_chest), 2),
                    "chest_lift_deg": round(plan.chest_lift_deg, 2), "chin_pitch_deg": round(plan.chin_pitch_deg, 2),
                    "head_stabilization_damping": round(plan.head_stabilization, 2),
                    "clavicle_lock": bool(plan.clavicle_lock), "clavicle_locked_all_frames": bool(clav_locked_all),
                    "arm_swing_deg": round(math.degrees(A_arm), 2),
                    "arm_backswing_deg": round(plan.arm_backswing_deg, 2), "arm_fwdswing_deg": round(plan.arm_fwdswing_deg, 2),
                    "elbow_flex_range_deg": list(plan.elbow_flex_range_deg),
                    "heel_strike_dorsiflex_deg": round(plan.heel_strike_dorsiflex_deg, 2),
                    "toe_off_plantarflex_deg": round(-plan.toe_off_deg, 2),
                    "heel_strike_knee_flex_deg": round(plan.heel_strike_knee_flex_deg, 2),
                    "arm_guard_deg": arm_guard_deg,
                    "delays_frames": {"chest<-pelvis": round(d_c / fr, 2), "shoulder<-chest": round((d_sh - d_c) / fr, 2),
                                      "elbow<-shoulder": round((d_el - d_sh) / fr, 2), "wrist<-elbow": round((d_wr - d_el) / fr, 2),
                                      "fingers<-wrist": round((d_fg - d_wr) / fr, 2), "head<-chest": round((d_hd - d_c) / fr, 2)}},
           "root_bob_mm_ptp": round(float(np.ptp(root_z + rig_z)) * 1e3, 2), "rig_shift_mm_range": [round(float(rig_z.min()) * 1e3, 3), round(float(rig_z.max()) * 1e3, 3)],
           "frames_with_reach_clamp": clamped, "min_body_z_mm_over_clip": min(r["min_body_z_mm"] for r in rows),
           "max_G5": max_g5, "min_thigh_clearance_mm": round(min_thigh_clear_mm, 2),
           "keyed_bones": len(KEYED), "psd_keys_animated": bool(psd), "volume_psd_keys_animated": vol_psd_names,
           "volume_psd_synthesized": vol_psd_synth, "invariants": inv, "seconds": round(time.time() - t0, 1),
           "spine03_lateral_correction_deg": {"raw_ptp": round(float(np.degrees(np.ptp(lat_raw))), 4), "smoothed_ptp": round(float(np.degrees(np.ptp(lat_s))), 4)}}
    (OUT / "m5_walk_solve.json").write_text(json.dumps({"report": rep, "frames": rows}, indent=1))
    return {"scene": str(OUT / "m5_walk_scene.blend")}, rep


# ------------------------------------------------------------------------------------------------------------------------------------
def mode_breath():
    import m_render as R
    import posing
    import m_pose_library as PL
    import pose_space_engine as PSE
    body, rig = C.human_and_rig(); inv = invariants(body, rig); me = body.data; sc = C.bpy.context.scene; rig.data.pose_position = "POSE"
    P = posing.Poser(C.bpy, rig); ev = lambda: eval_world(body)  # noqa: E731
    lib = PL.PoseLibrary(rig, P, ev, json.loads(Path(A["theta_fit"]).read_text())["bones"], ROOT)
    psd = PSE.PSDModel.from_json(me["maximus_psd"]) if "maximus_psd" in me.keys() else None
    lib.apply("theta_fit"); base = {pb.name: pb.matrix_basis.copy() for pb in rig.pose.bones}
    dev = "CPU"
    if A.get("device", "gpu") == "gpu":
        prefs = C.bpy.context.preferences.addons["cycles"].preferences
        for dt in ("OPTIX", "CUDA"):
            try:
                prefs.compute_device_type = dt; prefs.get_devices()
                if any(d.type == dt for d in prefs.devices):
                    for d in prefs.devices:
                        d.use = d.type == dt
                    dev = dt; break
            except Exception:  # noqa: BLE001
                pass
        if dev == "CPU" and A.get("require_gpu", "1") == "1":
            raise RuntimeError("no OPTIX/CUDA device: refusing to render the loop on CPU")
    sc.render.engine = "CYCLES"; sc.cycles.device = "GPU" if dev != "CPU" else "CPU"
    res = (int(A.get("w", "960")), int(A.get("h", "1200")))
    cam, lights = R.studio(C.bpy, sc, samples=int(A.get("samples", "96")), res=res, exposure=-1.0)
    camera = R.frame_camera(cam, ev(), view_deg=0.0)
    body.material_slots[0].material = C.bpy.data.materials["MX_M2_AlabasterChalk"]
    nfr = int(A.get("frames", "90")); spec = {"spine02": 0.8, "spine03": -0.35, "neck01": -0.35, "head": -0.10}
    from mathutils import Matrix
    outd = OUT / "breath"; outd.mkdir(exist_ok=True); t0 = time.time(); zmins = []
    for f in range(nfr):
        wave = 0.5 * (1 - math.cos(2 * math.pi * f / nfr))       # seamless: frame nfr == frame 0
        for pb in rig.pose.bones:
            pb.matrix_basis = base[pb.name]
        for nm, deg in spec.items():
            pb = rig.pose.bones[nm]; pb.matrix_basis = pb.matrix_basis @ Matrix.Rotation(math.radians(deg * wave), 4, "X")
        C.bpy.context.view_layer.update()
        if psd:
            lq = {j: np.array(rig.pose.bones[j].matrix_basis.to_quaternion(), float) for f_ in psd.families.values() for j in f_["joints"]}
            for k_, val in psd.key_weights(lq).items():
                me.shape_keys.key_blocks[k_].value = float(val)
        zmins.append(float(ev()[:, 2].min()))
        io.render_still(sc, outd / f"frame_{f:03d}.png")
        if f % 15 == 0:
            print(f"PROGRESS breath frame {f}/{nfr} on {dev} {time.time() - t0:.0f}s", flush=True)
    rep = {"frames": nfr, "fps": 24, "spine02_peak_deg": 0.8, "compensation_deg": spec, "device": dev, "resolution": res, "camera": camera,
           "z_sole_min_mm": round(min(zmins) * 1e3, 4), "seconds": round(time.time() - t0, 1), "invariants": inv}
    (OUT / "breath.json").write_text(json.dumps(rep, indent=1))
    return {"dir": str(outd)}, rep


C.run("m5_blender_" + A.get("mode", "walk"), {"corners": mode_corners, "walk": mode_walk, "breath": mode_breath}[A.get("mode", "walk")])
