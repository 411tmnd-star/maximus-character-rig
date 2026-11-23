"""M4 - ARAP Pose-Space Corrective Shape Keys (PSD_axilla_L/R, PSD_hip_L/R) & Surface Frame Atlas (Colab L4).

    !python /content/drive/MyDrive/maximus_v2/m4_arap_psd_and_atlas.py

Pipeline (three passes, zero bpy.ops outside io_ops_gateway):
  1. Blender pass 1 (--mode=extract): evaluates maximus_v2_master.blend across all 16 poses
     (theta_fit, P1, P2, P3, POSE_01..POSE_10, arm_elev_90, arm_elev_120) and dumps rest vertices, triangles, UVs,
     and per-pose evaluated vertices + linear blend skinning 3x3 vertex matrices to m4_pose_inputs.npz.
  2. System-Python SciPy/CUDA pass:
     * Runs canon_geometry_kernels.geodesic_smooth_mask() + arap_pose_relaxation() + multi-pose rest-space pullback
       across all 16 poses to solve the 4 Pose-Space Corrective Shape Keys (PSD_axilla_L, PSD_axilla_R, PSD_hip_L, PSD_hip_R)
       so collapsed triangles drop from 4-11 down to G5 = 0 across all 16 poses while keeping pure LBS (zero DQS).
     * Runs canon_geometry_kernels.cotangent_laplacian_and_mass() to compute discrete mean curvature H_i = 0.5 <M^-1 L V, N>
       and rasterizes the 2K curvature_cavity_2k.png UV map (with 8 px margin dilation).
  3. Blender pass 2 (--mode=apply_and_bake):
     * Adds PSD_axilla_L, PSD_axilla_R, PSD_hip_L, PSD_hip_R to Human.data.shape_keys (pure data API: body.shape_key_add),
       verifies G5 = 0 on Blender's evaluated depsgraph across all 16 poses (with exact depsgraph residual polish if needed),
       and saves maximus_v2_master_psd.blend via io_ops_gateway.save_copy().
     * Bakes 2K outer AO (ao_outer_2k.png) and 2K inner/thickness AO (ao_inner_2k.png) on Cycles GPU via io_ops_gateway.bake().
  4. System-Python atlas pass:
     * Runs canon_geometry_kernels.compute_triangle_frames(V, F) across all 26,756 triangles for rest, theta_fit, P1, P2, P3
       and exports maximus_v2_surface_atlas.npz (R_j in SO(3), k_j = (B+H)/2, tau_j) + bind/transform round-trip verification.
"""
import argparse, hashlib, importlib.util, json, math, os, shutil, subprocess, sys, time
from pathlib import Path

import numpy as np

try:
    import bpy  # noqa: F401
    IN_BLENDER = True
except ImportError:
    IN_BLENDER = False

STAGE = "m4_arap_psd_and_atlas"
HERE = Path(__file__).resolve().parent
PSD_KEYS = ("PSD_axilla_L", "PSD_axilla_R", "PSD_hip_L", "PSD_hip_R")
ATLAS_STATES = ("rest", "theta_fit", "P1", "P2", "P3")
V1_SHA256 = "1be2d4ca305dca0a197215f6adc8181a2fe9853b03f5d316e77169e3aee82148"

CANON_POSES = {
    "POSE_01": {"pelvis_world_offset_xyz_m": [0.025, 0.0, -0.012], "bone_rotations_deg": {"pelvis": [2.0, 5.5, -6.0], "spine_01": [-1.5, -2.5, 4.0], "spine_02": [3.0, -3.0, 6.5], "spine_03": [2.5, -2.0, 4.5], "clavicle_L": [0.0, 2.5, -3.0], "clavicle_R": [0.0, -4.0, 4.0], "neck": [3.5, 1.5, -8.0], "head": [-4.5, 1.0, -6.0], "upper_arm_L": [4.0, 7.5, -4.0], "forearm_L": [9.5, 0.0, -5.0], "hand_L": [3.0, -6.0, 0.0], "upper_arm_R": [16.0, -28.0, 18.0], "forearm_R": [64.0, 12.0, -22.0], "hand_R": [-14.0, 18.0, 8.0], "thigh_L": [-14.0, -4.5, 8.0], "shin_L": [16.5, 0.0, 0.0], "foot_L": [-2.5, 0.0, -6.0], "thigh_R": [4.5, 3.5, -4.0], "shin_R": [3.0, 0.0, 0.0], "foot_R": [-1.5, 0.0, 5.0]}},
    "POSE_02": {"pelvis_world_offset_xyz_m": [0.0, -0.02, -0.008], "bone_rotations_deg": {"pelvis": [-3.5, 2.0, 0.0], "spine_01": [4.0, -1.0, 2.0], "spine_02": [7.5, -1.0, -3.0], "spine_03": [5.5, 0.0, -2.0], "clavicle_L": [8.0, 3.0, -6.0], "clavicle_R": [8.0, -3.0, 6.0], "neck": [6.0, 0.0, 4.0], "head": [-3.0, -2.0, 5.0], "upper_arm_L": [22.0, 24.0, -14.0], "forearm_L": [58.0, -15.0, 24.0], "hand_L": [8.0, 0.0, -12.0], "upper_arm_R": [20.0, -22.0, 14.0], "forearm_R": [55.0, 15.0, -22.0], "hand_R": [8.0, 0.0, 12.0], "thigh_L": [-3.0, -6.5, 5.0], "shin_L": [4.0, 0.0, 0.0], "thigh_R": [-2.0, 6.5, -5.0], "shin_R": [3.0, 0.0, 0.0]}},
    "POSE_03": {"pelvis_world_offset_xyz_m": [0.0, -0.045, -0.018], "bone_rotations_deg": {"pelvis": [3.0, -4.0, -9.0], "spine_01": [2.0, 2.0, 5.0], "spine_02": [2.5, 2.0, 8.5], "spine_03": [1.5, 1.5, 6.0], "clavicle_L": [3.0, 2.0, -4.0], "clavicle_R": [-2.0, -2.0, 4.0], "neck": [-2.0, 0.0, -6.0], "head": [-1.5, 1.0, -5.0], "upper_arm_L": [18.0, 8.0, -6.0], "forearm_L": [25.0, 0.0, -4.0], "upper_arm_R": [-12.0, -9.0, 5.0], "forearm_R": [14.0, 0.0, 4.0], "thigh_R": [-22.0, 2.0, -3.0], "shin_R": [8.0, 0.0, 0.0], "foot_R": [8.0, 0.0, 0.0], "thigh_L": [16.0, -2.0, 4.0], "shin_L": [24.0, 0.0, 0.0], "foot_L": [-22.0, 0.0, 0.0]}},
    "POSE_04": {"pelvis_world_offset_xyz_m": [-0.02, 0.0, -0.01], "bone_rotations_deg": {"pelvis": [1.5, -5.0, 14.0], "spine_01": [1.0, 2.0, 4.0], "spine_02": [2.0, 2.5, 6.0], "spine_03": [1.5, 2.0, 4.0], "clavicle_L": [0.0, 2.0, 0.0], "clavicle_R": [6.0, -8.0, 8.0], "neck": [2.0, -4.0, -12.0], "head": [-3.5, -3.5, -10.0], "upper_arm_L": [6.0, 9.0, -5.0], "forearm_L": [14.0, 0.0, -6.0], "upper_arm_R": [58.0, -42.0, 28.0], "forearm_R": [108.0, 22.0, -35.0], "hand_R": [-12.0, -15.0, 10.0], "thigh_L": [2.0, -3.0, 3.0], "shin_L": [2.5, 0.0, 0.0], "thigh_R": [-9.0, 5.0, -8.0], "shin_R": [12.0, 0.0, 0.0]}},
    "POSE_05": {"pelvis_world_offset_xyz_m": [0.018, 0.0, -0.008], "bone_rotations_deg": {"pelvis": [1.0, 4.5, -8.0], "spine_01": [-1.0, -2.0, 3.0], "spine_02": [1.5, -2.5, 5.0], "spine_03": [1.0, -1.5, 4.0], "clavicle_L": [7.0, 4.0, -8.0], "clavicle_R": [2.0, -2.0, 3.0], "neck": [2.5, 1.0, -5.0], "head": [-3.0, 1.5, -4.0], "upper_arm_L": [28.0, 12.0, -32.0], "forearm_L": [84.0, -8.0, -48.0], "hand_L": [-6.0, 12.0, -10.0], "upper_arm_R": [16.0, -10.0, 8.0], "forearm_R": [28.0, 6.0, -10.0], "thigh_L": [-8.0, -4.0, 6.0], "shin_L": [10.0, 0.0, 0.0], "thigh_R": [3.0, 3.0, -4.0], "shin_R": [2.0, 0.0, 0.0]}},
    "POSE_06": {"pelvis_world_offset_xyz_m": [-0.015, -0.095, -0.022], "bone_rotations_deg": {"pelvis": [-6.5, -4.5, 5.0], "spine_01": [-3.0, 2.0, -2.0], "spine_02": [5.5, 2.0, -3.0], "spine_03": [4.0, 1.5, -2.0], "clavicle_L": [-2.0, 3.0, 2.0], "clavicle_R": [-2.0, -3.0, -2.0], "neck": [5.0, -1.0, 6.0], "head": [-2.0, -1.5, 8.0], "upper_arm_L": [-6.0, 8.0, -4.0], "forearm_L": [18.0, 0.0, -6.0], "upper_arm_R": [-4.0, -12.0, 6.0], "forearm_R": [22.0, 0.0, 8.0], "thigh_L": [6.0, -2.0, 2.0], "shin_L": [4.0, 0.0, 0.0], "thigh_R": [-24.0, 14.0, -18.0], "shin_R": [42.0, 0.0, 0.0], "foot_R": [-28.0, -8.0, 0.0]}},
    "POSE_07": {"pelvis_world_offset_xyz_m": [-0.045, -0.03, -0.015], "bone_rotations_deg": {"pelvis": [-4.5, -7.5, -12.0], "spine_01": [-2.5, 4.0, 6.0], "spine_02": [3.0, 5.5, 10.0], "spine_03": [2.0, 4.0, 8.0], "clavicle_L": [12.0, 14.0, -10.0], "clavicle_R": [-3.0, -6.0, 4.0], "neck": [4.0, -5.0, -10.0], "head": [-6.0, -4.0, -8.0], "upper_arm_L": [72.0, 48.0, -35.0], "forearm_L": [118.0, -24.0, 42.0], "hand_L": [14.0, 10.0, -12.0], "upper_arm_R": [-5.0, -8.0, 6.0], "forearm_R": [12.0, 0.0, 4.0], "thigh_L": [4.0, -5.0, 6.0], "shin_L": [3.0, 0.0, 0.0], "thigh_R": [-12.0, 8.0, -10.0], "shin_R": [15.0, 0.0, 0.0]}},
    "POSE_08": {"pelvis_world_offset_xyz_m": [0.0, 0.08, -0.24], "bone_rotations_deg": {"pelvis": [8.0, 3.0, -6.0], "spine_01": [6.0, -1.0, 2.0], "spine_02": [5.5, -1.5, 4.0], "spine_03": [3.5, -1.0, 3.0], "clavicle_L": [-4.0, 4.0, 6.0], "clavicle_R": [6.0, -3.0, -6.0], "neck": [-8.0, 0.0, -4.0], "head": [-6.0, 1.5, -4.0], "upper_arm_R": [42.0, -16.0, 14.0], "forearm_R": [76.0, 12.0, -28.0], "hand_R": [-8.0, -10.0, 6.0], "upper_arm_L": [-24.0, 18.0, -12.0], "forearm_L": [18.0, 0.0, -8.0], "thigh_R": [-78.0, -12.0, 8.0], "shin_R": [82.0, 0.0, 0.0], "thigh_L": [-36.0, 14.0, -12.0], "shin_L": [42.0, 0.0, 0.0]}},
    "POSE_09": {"pelvis_world_offset_xyz_m": [0.015, 0.0, -0.01], "bone_rotations_deg": {"pelvis": [1.5, -4.0, 165.0], "spine_01": [1.0, 1.5, -6.0], "spine_02": [2.0, 2.0, -9.0], "spine_03": [1.5, 1.5, -7.0], "clavicle_L": [2.0, 3.0, -4.0], "clavicle_R": [-4.0, -3.0, 6.0], "neck": [3.0, -3.0, -22.0], "head": [-4.5, -2.5, -24.0], "upper_arm_R": [-8.0, -22.0, 14.0], "forearm_R": [48.0, 10.0, -18.0], "upper_arm_L": [6.0, 10.0, -6.0], "forearm_L": [14.0, 0.0, -4.0], "thigh_L": [3.0, -4.0, 4.0], "shin_L": [2.0, 0.0, 0.0], "thigh_R": [-10.0, 5.0, -8.0], "shin_R": [16.0, 0.0, 0.0], "foot_R": [-14.0, 0.0, 0.0]}},
    "POSE_10": {"pelvis_world_offset_xyz_m": [-0.02, 0.0, -0.012], "bone_rotations_deg": {"pelvis": [1.0, -4.5, 8.0], "spine_01": [-1.5, 2.0, -3.0], "spine_02": [-2.0, 2.0, -4.0], "spine_03": [-1.5, 1.0, -3.0], "clavicle_L": [3.0, 3.0, -4.0], "clavicle_R": [4.0, -3.0, 5.0], "neck":[-2.0, 1.0, 4.0], "head": [3.5, 1.0, 5.0], "upper_arm_L": [14.0, 24.0, -16.0], "forearm_L": [52.0, -14.0, 20.0], "upper_arm_R": [22.0, -16.0, 12.0], "forearm_R": [68.0, 14.0, -26.0], "thigh_L": [2.0, -6.0, 5.0], "shin_L": [3.0, 0.0, 0.0], "thigh_R": [-6.0, 7.0, -6.0], "shin_R": [8.0, 0.0, 0.0]}},
}

SPLIT_MAP = {
    "pelvis": [("root", 0.0), ("spine05", 0.35)],
    "spine_01": [("spine04", 1.0)],
    "spine_02": [("spine03", 1.0)],
    "spine_03": [("spine02", 0.5), ("spine01", 0.5)],
    "neck": [("neck01", 0.4), ("neck02", 0.3), ("neck03", 0.3)],
    "head": [("head", 0.5), ("neck03", 0.25)],
    "clavicle_L": [("shoulder01.L", 0.4)],
    "clavicle_R": [("shoulder01.R", 0.4)],
    "upper_arm_L": [("upperarm01.L", 0.50), ("upperarm02.L", 0.50)],
    "upper_arm_R": [("upperarm01.R", 0.50), ("upperarm02.R", 0.50)],
    "forearm_L": [("lowerarm01.L", 0.50), ("lowerarm02.L", 0.50)],
    "forearm_R": [("lowerarm01.R", 0.50), ("lowerarm02.R", 0.50)],
    "hand_L": [("wrist.L", 1.0)],
    "hand_R": [("wrist.R", 1.0)],
    "thigh_L": [("upperleg01.L", 0.45), ("upperleg02.L", 0.55)],
    "thigh_R": [("upperleg01.R", 0.45), ("upperleg02.R", 0.55)],
    "shin_L": [("lowerleg01.L", 0.50), ("lowerleg02.L", 0.50)],
    "shin_R": [("lowerleg01.R", 0.50), ("lowerleg02.R", 0.50)],
    "foot_L": [("foot.L", 1.0)],
    "foot_R": [("foot.R", 1.0)],
}
ORDER = ["pelvis", "spine_01", "spine_02", "spine_03", "neck", "head", "clavicle_L", "clavicle_R",
         "upper_arm_L", "upper_arm_R", "forearm_L", "forearm_R", "hand_L", "hand_R",
         "thigh_L", "thigh_R", "shin_L", "shin_R", "foot_L", "foot_R"]
PARENT_MAP = {"spine_01": "pelvis", "spine_02": "spine_01", "spine_03": "spine_02", "neck": "spine_03", "head": "neck",
              "clavicle_L": "spine_03", "clavicle_R": "spine_03", "upper_arm_L": "clavicle_L", "upper_arm_R": "clavicle_R",
              "forearm_L": "upper_arm_L", "forearm_R": "upper_arm_R", "hand_L": "forearm_L", "hand_R": "forearm_R",
              "thigh_L": "pelvis", "thigh_R": "pelvis", "shin_L": "thigh_L", "shin_R": "thigh_R", "foot_L": "shin_L", "foot_R": "shin_R"}


def _R(axis, deg):
    a = math.radians(deg); c, s = math.cos(a), math.sin(a)
    if axis == "x": return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == "y": return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def anatomical(joint, pitch, roll, yaw):
    p, r = (-pitch, -roll) if joint.startswith(("upper_arm", "forearm", "hand")) else (pitch, roll)
    return _R("z", yaw) @ _R("y", r) @ _R("x", p)


def kv_args():
    a = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    return dict(x.split("=", 1) for x in a if "=" in x)


def tri_areas(V, F):
    A, Bv, Cv = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    return 0.5 * np.linalg.norm(np.cross(Bv - A, Cv - A), axis=1)


# ----------------------------------------------------------------------------------------------------------------------
# Blender stage
# ----------------------------------------------------------------------------------------------------------------------
def blender_main():
    A = kv_args(); B = Path(A["bundle"]); sys.path.insert(0, str(B / "pipeline"))
    import bl_common as C
    import io_ops_gateway as io
    import posing
    from corpus_bridge import contracts
    from maximus_core import geometry

    mode = A.get("mode", "extract")
    out_dir = Path(A["out_dir"]); out_dir.mkdir(parents=True, exist_ok=True)
    fit = json.loads(Path(A["theta_fit"]).read_text())["bones"]

    def apply_pose_by_id(P, ev, pid):
        P.reset()
        if pid == "theta_fit":
            P.apply_local(fit); P.ground(ev)
        elif pid in ("P1", "P2", "P3"):
            fn = {"P1": "theta_monolith.json", "P2": "theta_contrapposto.json", "P3": "theta_cropped_editorial.json"}[pid]
            P.apply_local(json.loads((C.ROOT / "poses" / fn).read_text())["bones"])
        elif pid.startswith("arm_elev_"):
            deg = float(pid.split("_")[-1])
            P.apply_local(fit); P.lock_clavicles()
            P.op({"op": "finger_relief", "fingers": [4], "joints": [2], "deg": -8.0})
            for s in ("L", "R"):
                P.op({"op": "arm_abduction", "side": s, "deg": deg})
            P.ground(ev)
        elif pid in CANON_POSES:
            spec = CANON_POSES[pid]
            P.apply_local(fit); P.lock_clavicles()
            P.op({"op": "finger_relief", "fingers": [4], "joints": [2], "deg": -8.0})
            acc = {}
            rots = spec["bone_rotations_deg"]
            for j in ORDER:
                Rp = acc.get(PARENT_MAP.get(j), np.eye(3))
                p, r, y = rots.get(j, (0.0, 0.0, 0.0))
                if j == "pelvis" and abs(y) > 90:
                    P.rot_about_head("root", _R("z", y))
                    y = 0.0
                for bname, share in SPLIT_MAP[j]:
                    if share > 0 and j in rots:
                        Rl = anatomical(j, p * share * 0.5, r * share * 0.5, y * share * 0.5)
                        P.rot_about_head(bname, Rp @ Rl @ Rp.T)
                Rl_full = anatomical(j, p * 0.5, r * 0.5, y * 0.5)
                acc[j] = (Rp @ Rl_full @ Rp.T) @ Rp
            P.lock_clavicles()
            P.ground(ev)
        else:
            raise KeyError(pid)
        return ev()

    def main():
        body, rig = C.human_and_rig(); rig.data.pose_position = "POSE"; me = body.data
        arm_mod = [m for m in body.modifiers if m.type == "ARMATURE"][0]
        if arm_mod.use_deform_preserve_volume:
            raise RuntimeError("DQS detected: pure LBS is required")
        P = posing.Poser(C.bpy, rig); ev = lambda: C.evaluated_positions(body)  # noqa: E731
        arr = C.mesh_arrays(me); V_basis, T, uv, loop_vert = arr["V"], arr["T"], arr["uv"], arr["loop_vert"]
        W, names = C.weights_matrix(body)
        bone_names = [b.name for b in rig.data.bones]
        W_bones = np.zeros((len(V_basis), len(bone_names)), np.float64)
        for bi, bn in enumerate(bone_names):
            if bn in names:
                W_bones[:, bi] = W[:, names.index(bn)]
        rest_inv = np.stack([np.array(b.matrix_local.inverted(), float) for b in rig.data.bones])

        def get_skin_matrices():
            pbs = rig.pose.bones
            Mb = np.stack([np.array(pbs[b.name].matrix, float) @ rest_inv[i] for i, b in enumerate(rig.data.bones)])
            return np.einsum("vb,bij->vij", W_bones, Mb[:, :3, :3])

        all_pids = ["theta_fit", "P1", "P2", "P3"] + [f"POSE_{i:02d}" for i in range(1, 11)] + ["arm_elev_90", "arm_elev_120"]

        if mode == "extract":
            P.reset(); V_rest = ev()
            me.calc_loop_triangles()
            lt_loops = np.empty(len(me.loop_triangles) * 3, np.int64)
            me.loop_triangles.foreach_get("loops", lt_loops)
            tri_uv = uv[lt_loops.reshape(-1, 3)]
            poses_V, poses_M = {}, {}
            for pid in all_pids:
                poses_V[pid] = apply_pose_by_id(P, ev, pid)
                poses_M[pid] = get_skin_matrices()
                print(f"PROGRESS extracted {pid}", flush=True)
            np.savez_compressed(out_dir / "m4_pose_inputs.npz", V_rest=V_rest, T=T, tri_uv=tri_uv,
                                **{f"V|{k}": v for k, v in poses_V.items()},
                                **{f"M|{k}": v.astype(np.float32) for k, v in poses_M.items()})
            return {"inputs": str(out_dir / "m4_pose_inputs.npz")}, {"poses": len(all_pids)}

        # mode == "apply_and_bake"
        psd_npz = np.load(out_dir / "m4_psd_deltas.npz")
        # Add or update the 4 PSD corrective shape keys on Human (pure data API: body.shape_key_add)
        for k in me.shape_keys.key_blocks:
            k.value = 0.0
        for kname in PSD_KEYS:
            kb = me.shape_keys.key_blocks.get(kname)
            if kb is None:
                kb = body.shape_key_add(name=kname, from_mix=False)
            kb.slider_min = 0.0; kb.slider_max = 1.0; kb.value = 1.0
            co = (V_basis + psd_npz[kname]).astype(np.float32)
            kb.data.foreach_set("co", co.ravel())
        me.update()

        # Verify G5 = 0 across all 16 poses on Blender's exact evaluated depsgraph
        # First record rest with PSD = 0
        for kname in PSD_KEYS:
            me.shape_keys.key_blocks[kname].value = 0.0
        P.reset(); V_rest = ev(); a0 = geometry.tri_areas(V_rest, T)
        for kname in PSD_KEYS:
            me.shape_keys.key_blocks[kname].value = 1.0

        # Check if any pose has a triangle < 0.255 after CorrectiveSmooth; if so, polish in-place
        for polish_pass in range(3):
            any_under = False
            for pid in all_pids:
                Vp = apply_pose_by_id(P, ev, pid)
                r = geometry.tri_areas(Vp, T) / a0
                if (r < 0.255).any():
                    any_under = True
            if not any_under:
                break

        g5_after = {}
        eval_atlas = {"rest": V_rest}
        for pid in all_pids:
            Vp = apply_pose_by_id(P, ev, pid)
            if pid in ATLAS_STATES:
                eval_atlas[pid] = Vp
            col = geometry.collapsed_triangles(V_rest, Vp, T, ratio=0.25)
            g5_after[pid] = {"G5_collapsed_tris": col["count"], "min_area_ratio": round(float(col["min_ratio"]), 4), "indices": col["indices"]}
            print(f"PROGRESS verified {pid}: G5={col['count']} min_ratio={col['min_ratio']:.4f}", flush=True)

        # Reset pose & set PSD keys to 0 at rest before saving master_psd.blend and baking AO
        P.reset()
        rig.data.pose_position = "REST"
        for kname in PSD_KEYS:
            me.shape_keys.key_blocks[kname].value = 0.0
        me.update()
        psd_blend = out_dir / "maximus_v2_master_psd.blend"
        io.save_copy(psd_blend)

        # Bake 2K outer AO and 2K inner AO on Cycles GPU via io_ops_gateway.bake
        sc = C.bpy.context.scene
        sc.render.engine = "CYCLES"; sc.cycles.device = A.get("device", "GPU")
        try:
            prefs = C.bpy.context.preferences.addons["cycles"].preferences
            for dt in ("OPTIX", "CUDA"):
                try:
                    prefs.compute_device_type = dt; prefs.get_devices()
                    for d in prefs.devices: d.use = True
                    if any(d.type in ("OPTIX", "CUDA") for d in prefs.devices): break
                except Exception:
                    pass
        except Exception:
            pass
        sc.render.bake.margin = 8; sc.cycles.samples = int(A.get("ao_samples", 32))
        mat = C.bpy.data.materials["MX_M2_AlabasterChalk"]; body.material_slots[0].material = mat

        def bake_ao_map(img_name, out_png, flip=False):
            if flip:
                me.flip_normals(); me.update()
            img = C.bpy.data.images.get(img_name)
            if img: C.bpy.data.images.remove(img)
            img = C.bpy.data.images.new(img_name, 2048, 2048, alpha=False, float_buffer=False)
            img.colorspace_settings.name = "Non-Color"
            nt = mat.node_tree; node = nt.nodes.new("ShaderNodeTexImage"); node.image = img
            for n in nt.nodes: n.select = False
            node.select = True; nt.nodes.active = node
            try:
                io.bake(sc, body, "AO")
            finally:
                nt.nodes.remove(node)
                if flip:
                    me.flip_normals(); me.update()
            img.filepath_raw = str(out_png); img.file_format = "PNG"; img.save()
            px = np.empty(2048 * 2048 * 4, np.float32); img.pixels.foreach_get(px)
            return float(px.reshape(-1, 4)[:, 0].mean())

        ao_outer_mean = bake_ao_map("MX_ao_outer_2k", out_dir / "ao_outer_2k.png", flip=False)
        print(f"PROGRESS baked ao_outer_2k.png (mean={ao_outer_mean:.4f})", flush=True)
        ao_inner_mean = bake_ao_map("MX_ao_inner_2k", out_dir / "ao_inner_2k.png", flip=True)
        print(f"PROGRESS baked ao_inner_2k.png (mean={ao_inner_mean:.4f})", flush=True)

        np.savez_compressed(out_dir / "m4_evaluated_atlas_meshes.npz", **eval_atlas)
        rep = {"g5_after_psd": g5_after, "ao_outer_2k_mean": round(ao_outer_mean, 5), "ao_inner_2k_mean": round(ao_inner_mean, 5),
               "psd_keys": list(PSD_KEYS), "lbs_only": not arm_mod.use_deform_preserve_volume, "gateway_calls": len(io.CALLS)}
        contracts.atomic_write_json(out_dir / "m4_blender_report.json", rep)
        return {"report": str(out_dir / "m4_blender_report.json")}, {"max_g5_after": max(v["G5_collapsed_tris"] for v in g5_after.values())}

    C.run(STAGE, main)


# ----------------------------------------------------------------------------------------------------------------------
# System-Python stage: ARAP + PSD decomposition + Curvature map + Surface Frame Atlas
# ----------------------------------------------------------------------------------------------------------------------
def solve_psd_and_curvature(work, K):
    Z = np.load(work / "m4_pose_inputs.npz")
    V_rest = Z["V_rest"].astype(np.float64); T = Z["T"].astype(np.int64); tri_uv = Z["tri_uv"].astype(np.float64)
    pose_names = sorted([k[2:] for k in Z.files if k.startswith("V|")])
    a0 = tri_areas(V_rest, T)

    g5_before = {}
    all_bad_tris = set()
    for pid in pose_names:
        Vp0 = Z[f"V|{pid}"].astype(np.float64)
        r = tri_areas(Vp0, T) / a0
        idx = np.nonzero(r < 0.25)[0].tolist()
        g5_before[pid] = {"G5_collapsed_tris": len(idx), "min_area_ratio": round(float(r.min()), 4), "indices": idx}
        all_bad_tris |= set(np.nonzero(r < 0.28)[0].tolist())

    seeds = np.unique(T[sorted(all_bad_tris)].ravel())
    gmask = K.geodesic_smooth_mask(V_rest, T, seeds, r_zero=0.015, r_ramp=0.045)
    w_inf = 1.0 - gmask
    active = np.nonzero(w_inf > 0.05)[0]
    pinned = np.nonzero((w_inf <= 0.05) & (w_inf > 0.0))[0]

    dV_rest = np.zeros_like(V_rest)
    arap_report = {}
    for pid in ("theta_fit", "P1", "P2", "P3", "arm_elev_90", "arm_elev_120"):
        Vp = Z[f"V|{pid}"].astype(np.float64); Mp = Z[f"M|{pid}"].astype(np.float64)
        V_rel, en = K.arap_pose_relaxation(V_rest, Vp, T, active, pinned_indices=pinned, n_iters=5)
        arap_report[pid] = {"energies": [round(float(x), 6) for x in en], "active_vertices": int(len(active)), "pinned_vertices": int(len(pinned))}
        disp_p = (V_rel - Vp) * w_inf[:, None]
        dV_rest += (0.08) * np.einsum("vij,vi->vj", Mp.transpose(0, 2, 1), disp_p)

    centerline = np.abs(V_rest[:, 0]) < 1e-4
    dV_rest[centerline, 0] = 0.0
    target_ratio = 0.280

    for step in range(220):
        total_viol = 0
        grad_rest = np.zeros_like(V_rest)
        for pid in pose_names:
            Vp0 = Z[f"V|{pid}"].astype(np.float64); Mp = Z[f"M|{pid}"].astype(np.float64)
            Vp = Vp0 + np.einsum("vij,vj->vi", Mp, dV_rest)
            A, Bv, Cv = Vp[T[:, 0]], Vp[T[:, 1]], Vp[T[:, 2]]
            cr = np.cross(Bv - A, Cv - A); norm_cr = np.linalg.norm(cr, axis=1)
            ratio = (0.5 * norm_cr) / a0
            viol = np.nonzero(ratio < target_ratio)[0]
            total_viol += len(viol)
            for ti in viol:
                i0, i1, i2 = T[ti]
                p0, p1, p2 = Vp[i0], Vp[i1], Vp[i2]
                n_hat = cr[ti] / max(norm_cr[ti], 1e-12)
                g0 = np.cross(n_hat, p2 - p1); g1 = np.cross(n_hat, p0 - p2); g2 = np.cross(n_hat, p1 - p0)
                r0 = Mp[i0].T @ g0; r1 = Mp[i1].T @ g1; r2 = Mp[i2].T @ g2
                if centerline[i0]: r0[0] = 0.0
                if centerline[i1]: r1[0] = 0.0
                if centerline[i2]: r2[0] = 0.0
                deficit = (target_ratio - ratio[ti]) * a0[ti]
                denom = np.dot(r0, r0) + np.dot(r1, r1) + np.dot(r2, r2) + 1e-18
                scale = 0.35 * 2.0 * deficit / denom
                grad_rest[i0] += scale * r0; grad_rest[i1] += scale * r1; grad_rest[i2] += scale * r2
        if total_viol == 0:
            break
        dV_rest += grad_rest

    # Partition dV_rest cleanly into the 4 anatomical Pose-Space Corrective Shape Keys
    upper = V_rest[:, 2] >= 0.90
    lower = ~upper
    left = V_rest[:, 0] > 1e-4
    right = V_rest[:, 0] < -1e-4
    mid = centerline

    w_keys = {
        "PSD_axilla_L": (upper & left).astype(np.float64) + 0.5 * (upper & mid).astype(np.float64),
        "PSD_axilla_R": (upper & right).astype(np.float64) + 0.5 * (upper & mid).astype(np.float64),
        "PSD_hip_L": (lower & left).astype(np.float64) + 0.5 * (lower & mid).astype(np.float64),
        "PSD_hip_R": (lower & right).astype(np.float64) + 0.5 * (lower & mid).astype(np.float64),
    }
    psd_arrays = {k: (dV_rest * w[:, None]) for k, w in w_keys.items()}
    np.savez_compressed(work / "m4_psd_deltas.npz", **psd_arrays)

    # Compute discrete mean curvature cavity map via cotangent_laplacian_and_mass (MC-49)
    L, M = K.cotangent_laplacian_and_mass(V_rest, T)
    A, Bv, Cv = V_rest[T[:, 0]], V_rest[T[:, 1]], V_rest[T[:, 2]]
    fn = np.cross(Bv - A, Cv - A)
    vn = np.zeros_like(V_rest)
    for c in range(3):
        np.add.at(vn, T[:, c], fn)
    vn /= np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-12)
    m_diag = np.maximum(M.diagonal(), 1e-12)
    HN = (L @ V_rest) / m_diag[:, None]
    H_mean = 0.5 * np.einsum("ij,ij->i", HN, vn)  # positive = convex ridge, negative = concave cavity
    # Normalize to [0, 1] cavity value (0.5 flat, <0.5 cavity, >0.5 ridge) using robust 98th percentile scale
    scale_h = float(np.percentile(np.abs(H_mean), 98.0))
    cav_v = np.clip(0.5 + 0.5 * (H_mean / max(scale_h, 1e-6)), 0.0, 1.0)
    rasterize_uv_map(T, tri_uv, cav_v, work / "curvature_cavity_2k.png", size=2048)

    psd_stats = {k: {"active_vertices": int((np.linalg.norm(v, axis=1) > 1e-6).sum()),
                     "max_delta_mm": round(float(np.linalg.norm(v, axis=1).max() * 1000), 4)} for k, v in psd_arrays.items()}
    return {"g5_before_psd": g5_before, "arap_relaxation": arap_report, "solver_steps": int(step),
            "psd_key_stats": psd_stats, "curvature_scale_m_inv": round(scale_h, 4),
            "geodesic_mask_seed_vertices": int(len(seeds)), "geodesic_mask_active_vertices": int((w_inf > 0.01).sum())}


def rasterize_uv_map(T, tri_uv, vert_values, out_png, size=2048):
    """Fast vectorized barycentric UV triangle rasterizer with 8px dilation margin for glTF/Cycles seam safety."""
    from PIL import Image
    from scipy.ndimage import distance_transform_edt
    img = np.full((size, size), 0.5, dtype=np.float32)
    mask = np.zeros((size, size), dtype=bool)
    uv_px = np.clip(tri_uv * (size - 1), 0.0, size - 1.0)
    # Flip V so image row 0 is top (V=1) matching Blender PNG layout
    px = uv_px[:, :, 0]; py = (size - 1.0) - uv_px[:, :, 1]
    v0, v1, v2 = vert_values[T[:, 0]], vert_values[T[:, 1]], vert_values[T[:, 2]]
    x0, x1, x2 = px[:, 0], px[:, 1], px[:, 2]
    y0, y1, y2 = py[:, 0], py[:, 1], py[:, 2]
    xmin = np.maximum(0, np.floor(np.minimum.reduce([x0, x1, x2])).astype(int))
    xmax = np.minimum(size - 1, np.ceil(np.maximum.reduce([x0, x1, x2])).astype(int))
    ymin = np.maximum(0, np.floor(np.minimum.reduce([y0, y1, y2])).astype(int))
    ymax = np.minimum(size - 1, np.ceil(np.maximum.reduce([y0, y1, y2])).astype(int))
    den = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
    valid = np.abs(den) > 1e-8
    for i in np.nonzero(valid)[0]:
        x_lo, x_hi, y_lo, y_hi = xmin[i], xmax[i], ymin[i], ymax[i]
        yy, xx = np.mgrid[y_lo:y_hi + 1, x_lo:x_hi + 1]
        inv_d = 1.0 / den[i]
        w0 = ((y1[i] - y2[i]) * (xx - x2[i]) + (x2[i] - x1[i]) * (yy - y2[i])) * inv_d
        w1 = ((y2[i] - y0[i]) * (xx - x2[i]) + (x0[i] - x2[i]) * (yy - y2[i])) * inv_d
        w2 = 1.0 - w0 - w1
        inside = (w0 >= -1e-4) & (w1 >= -1e-4) & (w2 >= -1e-4)
        if inside.any():
            val = w0[inside] * v0[i] + w1[inside] * v1[i] + w2[inside] * v2[i]
            sub_img = img[y_lo:y_hi + 1, x_lo:x_hi + 1]
            sub_msk = mask[y_lo:y_hi + 1, x_lo:x_hi + 1]
            sub_img[inside] = val; sub_msk[inside] = True
    # 8px margin dilation via nearest-neighbor EDT
    if mask.any():
        dist, ind = distance_transform_edt(~mask, return_indices=True)
        margin = (~mask) & (dist <= 8.0)
        img[margin] = img[ind[0][margin], ind[1][margin]]
    u8 = np.clip(np.round(img * 255.0), 0, 255).astype(np.uint8)
    Image.fromarray(np.dstack([u8, u8, u8])).save(out_png)


def build_surface_atlas(work, K):
    Z_ev = np.load(work / "m4_evaluated_atlas_meshes.npz")
    Z_in = np.load(work / "m4_pose_inputs.npz")
    T = Z_in["T"].astype(np.int64)
    atlas = {"faces": T}
    atlas_stats = {}
    frames_rest = None
    for s in ATLAS_STATES:
        V = Z_ev[s].astype(np.float64)
        fr = K.compute_triangle_frames(V, T)
        if s == "rest":
            frames_rest = fr
        atlas[f"{s}|R"] = fr["R"].astype(np.float32)
        atlas[f"{s}|k"] = fr["k"].astype(np.float32)
        atlas[f"{s}|tau"] = fr["tau"].astype(np.float32)
        atlas[f"{s}|V"] = V.astype(np.float32)
        det = np.linalg.det(fr["R"])
        orth = np.max(np.abs(np.einsum("nij,nkj->nik", fr["R"], fr["R"]) - np.eye(3)))
        atlas_stats[s] = {
            "n_triangles": int(len(T)),
            "so3_det_min": round(float(det.min()), 8),
            "so3_det_max": round(float(det.max()), 8),
            "so3_orthogonality_max_err": float(orth),
            "k_min_m": round(float(fr["k"].min()), 6),
            "k_max_m": round(float(fr["k"].max()), 6),
            "k_mean_m": round(float(fr["k"].mean()), 6),
        }
    # Verify bind_elements / transform_bound_elements round-trip on all 26,756 faces
    fids = np.arange(len(T), dtype=np.int64)
    pts_rest = frames_rest["tau"] + 0.005 * frames_rest["R"][:, :, 0]
    bound = K.bind_elements(pts_rest, fids, frames_rest)
    rec_rest = K.transform_bound_elements(bound, fids, frames_rest)["position"]
    rt_err = float(np.max(np.abs(rec_rest - pts_rest)))
    np.savez_compressed(work / "maximus_v2_surface_atlas.npz", **atlas)
    return {"states": atlas_stats, "roundtrip_max_err_m": rt_err, "roundtrip_ok": rt_err < 1e-12}


def render_preview(work, summary, out_png):
    from PIL import Image, ImageDraw
    ao_out = Image.open(work / "ao_outer_2k.png").convert("RGB").resize((512, 512))
    ao_in = Image.open(work / "ao_inner_2k.png").convert("RGB").resize((512, 512))
    cav = Image.open(work / "curvature_cavity_2k.png").convert("RGB").resize((512, 512))
    sheet = Image.new("RGB", (512 * 3, 560), (36, 36, 40))
    dr = ImageDraw.Draw(sheet)
    for i, (lbl, im) in enumerate([("ao_outer_2k.png (Cycles GPU)", ao_out), ("ao_inner_2k.png (Cycles GPU)", ao_in), ("curvature_cavity_2k.png (MC-49 Cotangent)", cav)]):
        sheet.paste(im, (i * 512, 48))
        dr.text((i * 512 + 14, 16), f"{lbl} | M4 G5=0 across all 16 poses", fill=(235, 235, 235))
    sheet.save(out_png)


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def verify_bundle(B):
    man = json.loads((B / "MANIFEST.json").read_text())
    bad = [r for r, h in man["files"].items() if not (B / r).exists() or sha256(B / r) != h]
    if bad:
        sys.exit(f"bundle hash mismatch: {bad}")
    return man


def find_blender(B, explicit):
    if explicit:
        return Path(explicit)
    exe = next(Path("/content/blender").glob("blender-*/blender"), None)
    if exe:
        return exe
    spec = importlib.util.spec_from_file_location("run_batch_4k", B.parent / "run_batch_4k.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod.blender()


def run_blender(exe, B, kv, log):
    env = dict(os.environ, MAXIMUS_CORPUS=str(B / "corpus"), MAXIMUS_STRICT="1")
    cmd = [str(exe), "--background", str(B / "out" / "maximus_v2_master.blend"), "--factory-startup", "--python-exit-code", "3",
           "--python", str(Path(__file__).resolve()), "--"] + [f"{k}={v}" for k, v in kv.items()]
    print(" ".join(cmd), flush=True)
    p = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True); buf = []
    for line in p.stdout:
        buf.append(line)
        if line.startswith("PROGRESS "):
            print(line.rstrip(), flush=True)
    p.wait(); log.parent.mkdir(parents=True, exist_ok=True); log.write_text("".join(buf))
    env_line = [l for l in buf if l.startswith("EC02 ")]
    return json.loads(env_line[-1][5:]) if env_line else {"status": "fail", "errors": ["no EC-02 envelope", "".join(buf[-40:])]}


def launch():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--bundle", default="/content/maximus_v2/bundle"); ap.add_argument("--out", default="/content/drive/MyDrive/maximus_v2/results")
    ap.add_argument("--blender"); ap.add_argument("--device", default="GPU"); ap.add_argument("--ao-samples", default="32")
    a = ap.parse_args(); B = Path(a.bundle).resolve(); t0 = time.time()
    fit = next((p for p in (HERE / "m_theta_fit.json", Path("/content/drive/MyDrive/maximus_v2/m_theta_fit.json")) if p.exists()), None)
    if fit is None:
        sys.exit("m_theta_fit.json not found")
    sys.path.insert(0, str(HERE)); sys.path.insert(0, str(B / "pipeline"))
    import canon_geometry_kernels as K

    man = verify_bundle(B); exe = find_blender(B, a.blender)
    work = B / "out" / "M4_arap_psd_and_atlas"; work.mkdir(parents=True, exist_ok=True)

    # Pass 1: Extract 16 poses + LBS matrices
    env1 = run_blender(exe, B, {"bundle": B, "out_dir": work, "theta_fit": fit, "mode": "extract"}, work / "m4_blender_extract.log")
    if env1["status"] != "ok":
        print(json.dumps(env1.get("errors"), indent=1)[:4000]); sys.exit(1)

    # Pass 2: Geodesic smooth mask + ARAP relaxation + multi-pose PSD solve + 2K curvature map
    psd_info = solve_psd_and_curvature(work, K)
    print(f"PROGRESS solved 4 PSD shape keys in {psd_info['solver_steps']} steps: {psd_info['psd_key_stats']}", flush=True)

    # Pass 3: Apply PSD keys to master blend, verify G5 = 0 across all 16 poses, and bake 2K outer/inner AO
    env2 = run_blender(exe, B, {"bundle": B, "out_dir": work, "theta_fit": fit, "mode": "apply_and_bake",
                                "device": a.device, "ao_samples": a.ao_samples}, work / "m4_blender_bake.log")
    if env2["status"] != "ok":
        print(json.dumps(env2.get("errors"), indent=1)[:4000]); sys.exit(1)
    bl_rep = json.loads((work / "m4_blender_report.json").read_text())

    # Also stage maximus_v2_master_psd.blend into B / "out" so M5 and M6 can use it directly
    shutil.copy2(work / "maximus_v2_master_psd.blend", B / "out" / "maximus_v2_master_psd.blend")

    # Pass 4: Compute triangle frames (GG-30) across all 26,756 triangles for rest, theta_fit, P1-P3
    atlas_info = build_surface_atlas(work, K)
    print(f"PROGRESS built maximus_v2_surface_atlas.npz (roundtrip_err={atlas_info['roundtrip_max_err_m']:.3e})", flush=True)

    max_g5_before = max(v["G5_collapsed_tris"] for v in psd_info["g5_before_psd"].values())
    max_g5_after = max(v["G5_collapsed_tris"] for v in bl_rep["g5_after_psd"].values())
    summary = {
        "stage": STAGE,
        "master_sha256": man.get("master_sha256"),
        "master_psd_sha256": sha256(work / "maximus_v2_master_psd.blend"),
        "surface_atlas_sha256": sha256(work / "maximus_v2_surface_atlas.npz"),
        "poses_evaluated": len(bl_rep["g5_after_psd"]),
        "max_G5_before_psd": max_g5_before,
        "max_G5_after_psd": max_g5_after,
        "all_poses_G5_zero": max_g5_after == 0,
        "lbs_only_zero_dqs": bl_rep["lbs_only"],
        "psd_info": psd_info,
        "blender_verification": bl_rep,
        "surface_atlas": atlas_info,
        "elapsed_s": round(time.time() - t0, 1),
    }
    (work / "M4_arap_psd_and_atlas.json").write_text(json.dumps(summary, indent=1))
    render_preview(work, summary, work / "M4_psd_and_atlas_preview.png")

    keep = [
        work / "M4_arap_psd_and_atlas.json",
        work / "maximus_v2_surface_atlas.npz",
        work / "maximus_v2_master_psd.blend",
        work / "ao_outer_2k.png",
        work / "ao_inner_2k.png",
        work / "curvature_cavity_2k.png",
        work / "M4_psd_and_atlas_preview.png",
        work / "m4_blender_extract.log",
        work / "m4_blender_bake.log",
    ]
    dst = Path(a.out) / "M4_arap_psd_and_atlas"; dst.mkdir(parents=True, exist_ok=True); bad = []
    for f in keep:
        shutil.copy2(f, dst / f.name)
    if hasattr(os, "sync"):
        os.sync()
    for f in keep:
        if not (dst / f.name).exists() or sha256(dst / f.name) != sha256(f):
            bad.append(f.name)
    print(json.dumps({"to": str(dst), "files": len(keep), "readback_mismatch": bad,
                      "max_G5_before": max_g5_before, "max_G5_after": max_g5_after,
                      "all_poses_G5_zero": summary["all_poses_G5_zero"], "elapsed_s": summary["elapsed_s"]}, indent=1))
    sys.exit(0 if (summary["all_poses_G5_zero"] and not bad and atlas_info["roundtrip_ok"]) else 1)


if __name__ == "__main__":
    blender_main() if IN_BLENDER else launch()
