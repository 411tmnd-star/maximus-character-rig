"""M9 Body Measurement & 5-Key Anthropometric Ridge Solver (ISO 8559-1 / GG-32).

Couples:
  - GG-32 horizontal cross-sectional tape circumference slicer (m5_anthro_regressor_and_walk.measure)
  - Maximus v2's 5 anthropometric shape keys (gt_lean, gt_stocky_nat, gt_stocky_ext, leg_len_m5cm, leg_len_p5cm)
  - M5 32-corner forward/inverse ridge regressor M(beta)
  - Constrained solver min_{beta in [0,1]^5} 0.5 ||M(beta) - m_target||_W^2 + lambda ||beta||^2

Pure NumPy / SciPy at query time using a lightweight cached basis NPZ (out/m9/maximus_anthro_basis.npz).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import lsq_linear

HERE = Path(__file__).resolve().parent
COLAB_STAGE = HERE / "colab_stage"
for _p in (COLAB_STAGE, HERE / "pipeline"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import m5_anthro_regressor_and_walk as M5  # noqa: E402

ANTHRO_KEYS = ("gt_lean", "gt_stocky_nat", "gt_stocky_ext", "leg_len_m5cm", "leg_len_p5cm")
METRICS = ("stature_cm", "neck_girth_cm", "chest_girth_cm", "waist_girth_cm", "hip_girth_cm", "thigh_girth_cm")

DEFAULT_REGRESSOR_JSON = COLAB_STAGE / "results" / "M5_kinematics" / "MAXIMUS_V2_ANTHRO_REGRESSOR.json"
DEFAULT_PSD_BLEND = COLAB_STAGE / "results" / "M4_canon_atlas" / "maximus_v2_master_psd.blend"
DEFAULT_THETA_FIT = COLAB_STAGE / "m_theta_fit.json"
DEFAULT_BASIS_NPZ = HERE / "out" / "m9" / "maximus_anthro_basis.npz"


def build_anthro_basis_cache(
    blend_path: Path = DEFAULT_PSD_BLEND,
    theta_fit_path: Path = DEFAULT_THETA_FIT,
    out_npz: Path = DEFAULT_BASIS_NPZ,
) -> Path:
    """Extract Maximus v2's 5 anthropometric shape-key deltas at bind & theta_fit (zero bpy.ops)."""
    import bpy
    import bl_common as C
    import posing

    for col in (bpy.data.objects, bpy.data.meshes, bpy.data.materials, bpy.data.armatures, bpy.data.images):
        for item in list(col):
            col.remove(item, do_unlink=True)

    with bpy.data.libraries.load(str(blend_path), link=False) as (df, dt):
        dt.objects = ["Human", "Human.rig"]
    sc = bpy.context.scene
    for o in list(bpy.data.objects):
        if o.name not in sc.collection.objects:
            sc.collection.objects.link(o)

    body, rig = C.human_and_rig()
    me = body.data
    me.calc_loop_triangles()
    kbs = me.shape_keys.key_blocks

    # 1. Bind (REST) deltas
    rig.data.pose_position = "REST"
    for k in kbs:
        k.value = 0.0
    V_rest_0 = C.evaluated_positions(body)
    deltas_rest = np.zeros((len(ANTHRO_KEYS), len(V_rest_0), 3), dtype=np.float32)
    for ki, kname in enumerate(ANTHRO_KEYS):
        kbs[kname].value = 1.0
        V_k = C.evaluated_positions(body)
        deltas_rest[ki] = (V_k - V_rest_0).astype(np.float32)
        kbs[kname].value = 0.0

    # 2. theta_fit deltas (grounded so Z_sole_min >= 0)
    rig.data.pose_position = "POSE"
    P = posing.Poser(bpy, rig)
    fit_bones = json.loads(Path(theta_fit_path).read_text(encoding="utf-8"))["bones"]
    P.reset()
    P.apply_local(fit_bones)
    P.lock_clavicles()
    ev = lambda: C.evaluated_positions(body)  # noqa: E731
    P.ground(ev)
    V_fit_0 = ev()
    deltas_fit = np.zeros((len(ANTHRO_KEYS), len(V_fit_0), 3), dtype=np.float32)
    for ki, kname in enumerate(ANTHRO_KEYS):
        kbs[kname].value = 1.0
        V_k = ev()
        deltas_fit[ki] = (V_k - V_fit_0).astype(np.float32)
        kbs[kname].value = 0.0

    arr = C.mesh_arrays(me)
    W, gnames = C.weights_matrix(body)
    arm_mask = M5.arm_hand_vertex_mask(W, gnames)
    pelvis_mask = M5.pelvis_vertex_mask(W, gnames)

    # Shoulder & crotch landmark vertex sets for breadth / inseam
    vg_idx = {n: i for i, n in enumerate(gnames)}
    sh_cols = [vg_idx[n] for n in ("shoulder01.L", "shoulder01.R") if n in vg_idx]
    sh_mask = (W[:, sh_cols].sum(axis=1) > 0.25) if sh_cols else np.zeros(len(V_rest_0), dtype=bool)

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz,
        key_names=np.array(ANTHRO_KEYS),
        V_rest_0=V_rest_0.astype(np.float64),
        deltas_rest=deltas_rest,
        V_fit_0=V_fit_0.astype(np.float64),
        deltas_fit=deltas_fit,
        faces=arr["T"].astype(np.int64),
        W=W.astype(np.float32),
        group_names=np.array(gnames),
        arm_mask=arm_mask.astype(bool),
        pelvis_mask=pelvis_mask.astype(bool),
        shoulder_mask=sh_mask.astype(bool),
    )

    for col in (bpy.data.objects, bpy.data.meshes, bpy.data.materials, bpy.data.armatures, bpy.data.images):
        for item in list(col):
            col.remove(item, do_unlink=True)
    return out_npz


class BodyMeasureEngine:
    """Deterministic ISO 8559-1 body measurement & 5-key anthropometric solver for Maximus v2."""

    def __init__(self, regressor_json: Path = DEFAULT_REGRESSOR_JSON, basis_npz: Path = DEFAULT_BASIS_NPZ):
        self.reg = json.loads(Path(regressor_json).read_text(encoding="utf-8"))
        if not Path(basis_npz).exists():
            build_anthro_basis_cache(out_npz=Path(basis_npz))
        with np.load(basis_npz, allow_pickle=False) as d:
            self.key_names = [str(x) for x in d["key_names"]]
            self.V_rest_0 = d["V_rest_0"].astype(np.float64)
            self.deltas_rest = d["deltas_rest"].astype(np.float64)
            self.V_fit_0 = d["V_fit_0"].astype(np.float64)
            self.deltas_fit = d["deltas_fit"].astype(np.float64)
            self.F = d["faces"].astype(np.int64)
            self.arm_mask = d["arm_mask"].astype(bool)
            self.pelvis_mask = d["pelvis_mask"].astype(bool)
            self.shoulder_mask = d["shoulder_mask"].astype(bool)

        self.W_fwd = np.asarray(self.reg["forward_weights_cm"], dtype=np.float64)  # (5, 6)
        self.b_fwd = np.asarray(self.reg["forward_bias_cm"], dtype=np.float64)     # (6,)
        self.metric_names = list(self.reg["metric_names"])

    def morph_vertices(self, beta: np.ndarray, pose: str = "theta_fit") -> np.ndarray:
        """Compute deformed Maximus v2 body vertices V(beta) in metres (grounded so Z_sole_min == 0)."""
        b = np.clip(np.asarray(beta, dtype=np.float64), 0.0, 1.0)
        if pose == "rest":
            V = self.V_rest_0 + np.tensordot(b, self.deltas_rest, axes=(0, 0))
        else:
            V = self.V_fit_0 + np.tensordot(b, self.deltas_fit, axes=(0, 0))
        # Ensure grounded sole Z_sole_min >= 0.0
        z_min = float(V[:, 2].min())
        if z_min != 0.0:
            V = V.copy()
            V[:, 2] -= z_min
        return V

    def forward_regressor_cm(self, beta: np.ndarray) -> dict[str, float]:
        """Fast linear M(beta) forward prediction in cm."""
        b = np.clip(np.asarray(beta, dtype=np.float64), 0.0, 1.0)
        pred = b @ self.W_fwd + self.b_fwd
        return {m: round(float(v), 2) for m, v in zip(self.metric_names, pred)}

    def measure_mesh_gg32(self, V: np.ndarray, n_samples: int = 13) -> dict:
        """Measure exact ISO 8559-1 cross-sectional tape circumferences on deformed 3D mesh via GG-32."""
        girths, detail = M5.measure(V, self.F, n_samples=n_samples, arm_mask=self.arm_mask, pelvis_mask=self.pelvis_mask)

        z0, z1 = float(V[:, 2].min()), float(V[:, 2].max())
        H = z1 - z0
        # Biacromial shoulder breadth: torso + deltoid/acromion width at 80%-83% stature
        sh_band = (V[:, 2] >= z0 + 0.805 * H) & (V[:, 2] <= z0 + 0.835 * H) & self.shoulder_mask
        if sh_band.any():
            shoulder_breadth_cm = float(np.ptp(V[sh_band, 0])) * 100.0
        else:
            # Fallback to slice x-span at chest top
            shoulder_breadth_cm = float(detail["chest_girth_cm"]["x_span_cm"]) * 1.15

        # Approximate crotch / inseam landmark (lowest torso-pelvis slice below hip where thigh loops separate)
        inseam_cm = float(detail["thigh_girth_cm"]["height_frac"] * H * 100.0)

        out = {
            "standard": "ISO 8559-1 (GG-32 cross-sectional tape girths)",
            "stature_cm": round(float(girths["stature_cm"]), 2),
            "neck_girth_cm": round(float(girths["neck_girth_cm"]), 2),
            "shoulder_breadth_cm": round(shoulder_breadth_cm, 2),
            "chest_girth_cm": round(float(girths["chest_girth_cm"]), 2),
            "waist_girth_cm": round(float(girths["waist_girth_cm"]), 2),
            "hip_girth_cm": round(float(girths["hip_girth_cm"]), 2),
            "thigh_girth_cm": round(float(girths["thigh_girth_cm"]), 2),
            "inseam_cm": round(inseam_cm, 2),
            "landmarks": detail,
        }
        return out

    def measure_for_beta(self, beta: np.ndarray, pose: str = "rest") -> dict:
        """Evaluate morphed body V(beta) and emit full GG-32 body_measurements.json dict."""
        b = np.clip(np.asarray(beta, dtype=np.float64), 0.0, 1.0)
        V = self.morph_vertices(b, pose=pose)
        meas = self.measure_mesh_gg32(V)
        meas["beta"] = {k: round(float(v), 4) for k, v in zip(self.key_names, b)}
        meas["pose_measured"] = pose
        meas["regressor_forward_cm"] = self.forward_regressor_cm(b)
        return meas

    def solve_beta_from_targets(
        self,
        targets_cm: dict[str, float],
        metric_weights: dict[str, float] | None = None,
        lam: float = 0.02,
    ) -> np.ndarray:
        """Bounded ridge solver: min_{beta in [0,1]^5} 0.5 ||W^{1/2}(M(beta) - m_target)||^2 + lam ||beta||^2."""
        w_map = {"stature_cm": 1.6, "neck_girth_cm": 0.9, "chest_girth_cm": 1.8, "waist_girth_cm": 1.6, "hip_girth_cm": 1.2, "thigh_girth_cm": 0.8}
        if metric_weights:
            w_map.update(metric_weights)

        rows_A = []
        rows_b = []
        sd = np.asarray(self.reg["input_std_cm"], dtype=np.float64)
        for mi, mname in enumerate(self.metric_names):
            if mname not in targets_cm or targets_cm[mname] is None:
                continue
            wt = float(w_map.get(mname, 1.0)) / max(float(sd[mi]), 1.0)
            rows_A.append(wt * self.W_fwd[:, mi])
            rows_b.append(wt * (float(targets_cm[mname]) - float(self.b_fwd[mi])))

        # Ridge regularization row
        reg_s = float(np.sqrt(lam))
        for ki in range(len(self.key_names)):
            r = np.zeros(len(self.key_names), dtype=np.float64)
            r[ki] = reg_s
            rows_A.append(r)
            rows_b.append(0.0)

        A = np.vstack(rows_A)
        b_vec = np.asarray(rows_b, dtype=np.float64)
        res = lsq_linear(A, b_vec, bounds=(0.0, 1.0))
        return np.clip(res.x, 0.0, 1.0)

    def estimate_targets_from_profile(self, height_cm: float, weight_kg: float, build: str = "average") -> dict[str, float]:
        """Map (height_cm, weight_kg, build) to physically coherent ISO 8559-1 target measurements (cm)."""
        h = float(np.clip(height_cm, 158.0, 195.0))
        w = float(np.clip(weight_kg, 50.0, 130.0))
        bmi = w / ((h / 100.0) ** 2)
        build_offsets = {
            "lean":      {"chest": -5.5, "waist": -8.0, "hip": -4.5, "neck": -1.2, "thigh": -3.0},
            "average":   {"chest":  0.0, "waist":  0.0, "hip":  0.0, "neck":  0.0, "thigh":  0.0},
            "athletic":  {"chest": +4.5, "waist": -3.5, "hip": +1.0, "neck": +1.0, "thigh": +2.0},
            "stocky":    {"chest": +7.5, "waist": +9.0, "hip": +6.5, "neck": +1.8, "thigh": +4.5},
            "broad":     {"chest": +11.0, "waist": +12.0, "hip": +9.5, "neck": +2.4, "thigh": +6.0},
        }.get(build.lower(), {"chest": 0.0, "waist": 0.0, "hip": 0.0, "neck": 0.0, "thigh": 0.0})

        # BMI=22.5 neutral anchor calibrated against Maximus neutral basis (93.2 chest, 74.2 waist, 93.0 hip)
        dbmi = bmi - 22.5
        return {
            "stature_cm": round(h, 2),
            "neck_girth_cm": round(35.0 + 0.45 * dbmi + build_offsets["neck"], 2),
            "chest_girth_cm": round(96.5 + 2.45 * dbmi + build_offsets["chest"], 2),
            "waist_girth_cm": round(78.5 + 3.05 * dbmi + build_offsets["waist"], 2),
            "hip_girth_cm": round(96.0 + 2.35 * dbmi + build_offsets["hip"], 2),
            "thigh_girth_cm": round(55.5 + 1.45 * dbmi + build_offsets["thigh"], 2),
        }

    def measure_from_profile(self, height_cm: float, weight_kg: float, build: str = "average", pose: str = "rest") -> dict:
        targets = self.estimate_targets_from_profile(height_cm, weight_kg, build)
        beta = self.solve_beta_from_targets(targets)
        meas = self.measure_for_beta(beta, pose=pose)
        meas["profile_input"] = {"height_cm": height_cm, "weight_kg": weight_kg, "build": build}
        meas["target_cm"] = targets
        return meas


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--build-cache", action="store_true")
    ap.add_argument("--height-cm", type=float, default=173.1)
    ap.add_argument("--weight-kg", type=float, default=72.0)
    ap.add_argument("--build", default="average", choices=["lean", "average", "athletic", "stocky", "broad"])
    ap.add_argument("--beta", nargs=5, type=float, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.build_cache:
        p = build_anthro_basis_cache()
        print("Built basis cache:", p)

    eng = BodyMeasureEngine()
    if args.beta is not None:
        res = eng.measure_for_beta(np.asarray(args.beta, dtype=np.float64), pose="rest")
    else:
        res = eng.measure_from_profile(args.height_cm, args.weight_kg, args.build, pose="rest")

    txt = json.dumps(res, indent=2)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
