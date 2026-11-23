"""Stage 02 (Blender): author P1/P2/P3 editorial poses on the NATIVE rig, on top of theta_fit.json (411chrone angles).

Rules: clavicle.L/.R locked to rotation (0,0,0) on all three axes; LBS + Corrective Smooth; feet grounded so the
lowest vertex sits on z = 0; POSE_01..10 (canonical_poses.json) are not touched, only referenced.
Outputs: poses/theta_monolith.json, theta_contrapposto.json, theta_cropped_editorial.json, pose_library_mapping.json,
         out/cache/pose_<P>.npz (evaluated posed vertices) for the gates. The master .blend is not modified.
"""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # noqa: E401,E702
import json, math

import numpy as np

import bl_common as C
import posing

from corpus_bridge import contracts
from maximus_core import clearance, geometry

from pose_specs import POSES  # noqa: E402

BACKDROP = {k: v["backdrop"] for k, v in C.FINISHES.items()}
FOOT = ("foot", "toe")
TORSO = ("spine", "breast", "pelvis", "root", "upperleg")
HAND = ("wrist", "finger", "metacarpal")


def dominant(W, names):
    """Dominant BONE per vertex (helper groups such as _cs_mask_shoulders are ignored)."""
    W = W.copy(); W[:, [i for i, n in enumerate(names) if n.startswith("_")]] = -1
    return np.array(names)[W.argmax(1)]


DISTAL = ("upperarm02", "lowerarm", "wrist", "finger", "metacarpal")
G5_RATIO, G5_MAX = 0.25, 6     # mirrors gate G5 in s03_gates.py (measurement only; the gate stays in s03)
GUARD ={"P1": {"L": "abduct", "R": "abduct"}, "P2": {"L": "abduct", "R": "abduct"}, "P3": {"L": "flex", "R": "abduct"}}


def skin_ring(E, seed, rings=4):
    """Vertices within `rings` edge hops of the seed set (the armpit skin that is connected to the torso)."""
    m = seed.copy()
    for _ in range(rings):
        nb = np.zeros_like(m); nb[E[m[E[:, 0]], 1]] = True; nb[E[m[E[:, 1]], 0]] = True; m |= nb
    return m


def arm_guard(P, ev, torso_tris, dom, free_side, edges_side, side, mode, step=0.5, max_steps=30, eps=clearance.EPS_BODY,
              extra_ok=lambda X: True):
    """Clean-room L_body rule applied to the pose. Until the arm's edges cross no torso triangle AND every free
    distal-arm vertex (not skin-connected to the torso) is >= eps (3 mm) off the torso: abduct the humerus outward when
    the upper arm is involved, otherwise (or when mode == 'flex') swing it forward. Stops at the first passing step."""
    sx = 1 if side == "L" else -1; added = {"abduct": 0.0, "flex": 0.0}
    for k in range(max_steps + 1):
        X = ev()
        n_cross, pairs = clearance.segment_triangle_hits(X[edges_side[:, 0]], X[edges_side[:, 1]], X, torso_tris)
        sd = clearance.point_mesh_distance(X[free_side], X, torso_tris, k=12)["signed"]; dmin = float(sd.min())
        ok = n_cross == 0 and dmin >= eps and extra_ok(X)
        if ok or k == max_steps:
            break
        involved = set(dom[edges_side[pairs[:, 0]]].ravel().tolist()) | set(dom[free_side[sd < eps]].tolist())
        act = "abduct" if (mode == "abduct" or any(b_.startswith("upperarm02") for b_ in involved)) else "flex"
        if act == "abduct":
            P.rot_about_head(f"upperarm01.{side}", posing.rot(posing.F, sx * step))
        else:
            h = P.J(f"lowerarm01.{side}") - P.J(f"upperarm01.{side}"); p = posing.F - (posing.F @ h) / (h @ h) * h
            P.rot_about_head(f"upperarm01.{side}", posing.rot(np.cross(h, p), step))
        added[act] += step
    return {"side": side, "mode": mode, "added_abduction_deg": added["abduct"], "added_forward_flex_deg": added["flex"],
            "edge_crossings": int(n_cross), "min_free_arm_torso_mm": round(dmin * 1000, 3), "converged": bool(ok)}


def main():
    body, rig = C.human_and_rig()
    if contracts.sha256_file(C.MASTER) != contracts.sha256_file(C.bpy.data.filepath):
        raise RuntimeError("stage 02 must run on out/maximus_v2_master.blend")
    W, names = C.weights_matrix(body); dom = dominant(W, names)
    T = C.mesh_arrays(body.data)["T"]
    side = np.array([n[-1] if n[-2:] in (".L", ".R") else "" for n in dom])
    foot = np.array([n.startswith(FOOT) for n in dom])
    torso_tri = np.all(np.isin(dom[T], [n for n in names if n.startswith(TORSO)]), axis=1)
    fit = json.loads(C.THETA_FIT.read_text())
    if fit["rig"]["sha256"] != C.V1_SHA256:
        raise RuntimeError("theta_fit was authored on a different rig")
    P = posing.Poser(C.bpy, rig)
    ev = lambda: C.evaluated_positions(body)  # noqa: E731
    results = {}
    poses_dir = C.ROOT / "poses"; poses_dir.mkdir(exist_ok=True)
    # baseline for the gates report: theta_fit alone (no editorial ops), grounded
    P.reset(); P.apply_local(fit["bones"]); P.ground(ev)
    np.savez_compressed(C.CACHE / "pose_BASE_theta_fit.npz", V=ev())
    rest = np.load(C.CACHE / "v2_rest.npz"); V_rest, T_rest = rest["V"], rest["T"]   # the same reference s03 uses for G5

    def build(pid, ops):
        """Author one pose from an op list: theta_fit + ops + knee balance (P2) + arm guards + grounding.
        Leaves the rig in that pose and returns (evaluated vertices, checks, min z)."""
        P.reset(); P.log = []
        P.apply_local(fit["bones"])
        P.lock_clavicles()
        for o in ops:
            P.op(dict(o))
        P.lock_clavicles()
        extra = {}
        if pid == "P2":   # contrapposto: bend the right knee until both soles sit at the same height
            snap = {b.name: b.matrix_basis.copy() for b in rig.pose.bones}

            def soles():
                X = ev(); return X[foot & (side == "L"), 2].min(), X[foot & (side == "R"), 2].min()

            def knee(phi):
                for b in rig.pose.bones:
                    b.matrix_basis = snap[b.name]
                P.update()
                if phi:
                    P.rot_about_head("upperleg01.R", posing.rot(posing.L, -phi / 2)); P.rot_about_head("lowerleg01.R", posing.rot(posing.L, phi))
                    P.rot_about_head("foot.R", posing.rot(posing.L, -phi / 2))
                zl, zr = soles(); return zl - zr
            lo, hi = 0.0, 35.0
            d0 = knee(0.0)
            if d0 > 0.0005:
                for _ in range(30):
                    mid = 0.5 * (lo + hi)
                    if knee(mid) > 0: lo = mid
                    else: hi = mid
                phi = 0.5 * (lo + hi); knee(phi)
            else:
                phi = 0.0
            extra["right_knee_bend_deg"] = round(phi, 3); extra["sole_height_diff_before_mm"] = round(d0 * 1000, 3)
            P.log.append({"op": "knee_balance", "side": "R", "deg": round(phi, 3), "why": "both soles on the floor (contrapposto free leg)"})
        distal = np.isin(dom, [n for n in names if n.startswith(DISTAL)])
        E = geometry.unique_edges(T)
        hand_L = np.nonzero(np.isin(dom, [n for n in names if n.startswith(HAND) and n.endswith(".L")]))[0]
        tt = T[torso_tri]
        guards = []
        ring = skin_ring(E, np.isin(dom, [n for n in names if n.startswith(TORSO)]))
        for s_ in ("L", "R"):
            ds = distal & (side == s_); es = E[ds[E[:, 0]] & ds[E[:, 1]]]
            free = np.nonzero(ds & ~ring)[0]
            ok = (lambda X: float(clearance.point_mesh_distance(X[hand_L], X, tt, k=12)["signed"].min()) >= 0.015) if (pid == "P3" and s_ == "L") else (lambda X: True)
            guards.append(arm_guard(P, ev, tt, dom, free, es, s_, GUARD[pid][s_], extra_ok=ok))
        P.log.append({"op": "clearance_guard", "rule": "L_body eps 3 mm: zero arm/torso edge crossings and every free distal-arm vertex (outside the 4-ring skin-connected to the torso) >= 3 mm off the torso", "result": guards})
        extra["clearance_guard"] = guards
        if pid == "P3":
            X = ev(); extra["hand_torso_clearance_mm"] = round(float(clearance.point_mesh_distance(X[hand_L], X, tt, k=12)["signed"].min()) * 1000, 3)
        P.lock_clavicles()
        z0 = P.ground(ev)
        return ev(), extra, z0

    for pid, spec in POSES.items():
        # G5 guard (closed loop, same measurement as gate G5: triangles under 0.25 x rest area, limit 6).
        # The spec ops are tried first. If they collapse more than the limit, the declared relief variants in
        # pose_specs are tried in order and the first that passes is kept; if none passes, the variant with the
        # fewest collapsed triangles is kept and s03 reports the failure. The gate itself is not touched.
        variants = [{"label": "spec", "ops": spec["ops"]}] + list(spec.get("g5_relief", []))
        trials = []; best = None; built = None
        for i, var in enumerate(variants):
            X, extra, z0 = build(pid, var["ops"]); built = i
            col = geometry.collapsed_triangles(V_rest, X, T_rest, ratio=G5_RATIO)
            trials.append({"variant": var["label"], "collapsed": col["count"], "min_ratio": round(float(col["min_ratio"]), 4),
                           "indices": col["indices"][:24]})
            if best is None or col["count"] < trials[best]["collapsed"]:
                best = i
            if col["count"] <= G5_MAX:
                break
        if built != best:
            X, extra, z0 = build(pid, variants[best]["ops"])
        extra["g5_guard"] = {"limit": G5_MAX, "ratio": G5_RATIO, "chosen": variants[best]["label"],
                             "why": variants[best].get("why", "ops as authored from 411chrone"), "trials": trials}
        P.log.append({"op": "g5_guard", "chosen": variants[best]["label"], "trials": [{k: t[k] for k in ("variant", "collapsed")} for t in trials]})
        clav = {n: list(rig.pose.bones[n].rotation_quaternion) for n in posing.CLAVICLES}
        doc = {"schema": "betterstyle.avatar_pose/1", "kind": spec["name"], "pose_id": pid, "title": spec["title"], "version": "2.0.0",
               "maps_to_canonical_pose": spec["maps_to"], "finish": spec["finish"], "backdrop_radial": BACKDROP[spec["finish"]],
               "rig": {"file": str(C.V1_BLEND), "sha256": C.V1_SHA256, "bones": len(rig.data.bones), "frame": "native 03_full_rigged (not the GLB re-import)"},
               "base": {"file": str(C.THETA_FIT), "sha256": contracts.sha256_file(C.THETA_FIT)},
               "skinning": "linear blend (no dual quaternion) + Corrective Smooth factor 0.50, repeat 5, group _cs_mask_shoulders",
               "clavicle_lock_xyz_deg": [0.0, 0.0, 0.0], "clavicle_quaternions": clav,
               "source_angles": "411chrone.pdf (P1/P2/P3 editorial poses)", "declarative_ops": P.log,
               "grounding": {"min_z_m": z0}, "checks": extra, "bones": P.local_quats()}
        contracts.atomic_write_json(poses_dir / spec["file"], doc)
        np.savez_compressed(C.CACHE / f"pose_{pid}.npz", V=X)
        results[pid] = {"file": str(poses_dir / spec["file"]), "min_z_m": z0, **extra,
                        "clavicle_identity": all(abs(q[0] - 1) < 1e-12 and max(map(abs, q[1:])) < 1e-12 for q in clav.values())}
    canon = C.CANONICAL_POSES
    contracts.atomic_write_json(poses_dir / "pose_library_mapping.json", {
        "editorial_layer": {"P1": "theta_monolith.json", "P2": "theta_contrapposto.json", "P3": "theta_cropped_editorial.json"},
        "maps_to_canonical": {"P1": "POSE_02", "P2": "POSE_06", "P3": "POSE_05"},
        "finish_pairing": {"P1": "M1", "P2": "M2", "P3": "M3"},
        "canonical_library": {"file": str(canon), "sha256": contracts.sha256_file(canon), "poses": "POSE_01..POSE_10", "status": "untouched (read only)"},
        "note": "The P layer is authored from 411chrone.pdf angles on theta_fit. The mapped POSE_xx entries keep their own (non-zero) clavicle values in the canonical library; the P layer locks clavicles at 0."})
    return {"poses": results}, {"n_poses": len(results)}


C.run("s02_build_poses", main)
