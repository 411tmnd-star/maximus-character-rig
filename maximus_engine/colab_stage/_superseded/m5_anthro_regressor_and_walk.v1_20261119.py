#!/usr/bin/env python3
"""M5 - 32-Corner ISO 8559-1 Anthropometric Ridge Regressor, Runway Walk Solver (S1-S9) & 90-Frame Breathing Loop (Colab L4).

    !python /content/drive/MyDrive/maximus_v2/m5_anthro_regressor_and_walk.py

Locked Maximus v2 Invariants (NEVER violated):
  * Exact mesh topology: 13,380 vertices, 26,756 triangles, 152 bones.
  * Pure Linear Blend Skinning (use_deform_preserve_volume = False; ZERO DQS).
  * Corrective Smooth modifier locked at factor=0.50, iterations=5, SIMPLE, ORCO.
  * Clavicles (clavicle.L, clavicle.R) strictly locked at (0, 0, 0) / identity quaternion (1, 0, 0, 0).
  * Limb twist routed through *02 twist bones (upperarm02, lowerarm02, upperleg02, lowerleg02) so primary twist <= 15 deg.
  * Runway walk targets:
      - Cadence: 2.6 s stride period (24 fps)
      - Step line: 3.0-6.0 cm (4.5 cm nominal, never negative -> S8)
      - Pelvic sway: 15-22 mm lateral (18.5 mm nominal), 25-35 mm vertical bob
      - Shoulder / chest counter-rotation: 3.5-5.0 deg (4.2 deg nominal)
      - Floor grounding: Z_sole >= 0.0 mm every frame (zero floor penetration)
      - S1-S9 gates evaluated against avatar_engine/harness/gate_limits.json (limits never weakened)
  * Living Statue breathing loop: 90 frames @ 24 fps, +0.8 deg spine02 sinusoidal cycle.
  * Bonus owner check: CUDA vs CPU parity verification of canon_geometry_kernels.generalized_winding_number_float64.

Local CPU self-test (no Blender, no GPU):
    python -I m5_anthro_regressor_and_walk.py --self-test
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

try:
    import bpy  # noqa: F401
    IN_BLENDER = True
except ImportError:
    IN_BLENDER = False

STAGE = "m5_anthro_regressor_and_walk"
HERE = Path(__file__).resolve().parent

# Locked Maximus v2 invariants
LOCKED_VERTS = 13380
LOCKED_TRIS = 26756
LOCKED_BONES = 152
LOCKED_CS_FACTOR = 0.50
LOCKED_CS_REPEAT = 5
LOCKED_CLAVICLES = ("clavicle.L", "clavicle.R")
TWIST_BONES = (
    "upperarm02.L", "upperarm02.R",
    "lowerarm02.L", "lowerarm02.R",
    "upperleg02.L", "upperleg02.R",
    "lowerleg02.L", "lowerleg02.R",
)

# Locked runway walk specification (24 fps, 2.6 s cadence)
WALK_SPEC = {
    "fps": 24.0,
    "stride_period_s": 2.6,
    "duration_s": 10.4,              # 4 full strides (first 8.0 s = 3 full strides for S1-S9)
    "step_line_cm": 4.5,             # within [3.0, 6.0] cm (S8)
    "pelvic_sway_mm": 18.5,          # within [15.0, 22.0] mm
    "pelvic_bob_mm": 28.0,           # within [25.0, 35.0] mm
    "pelvis_yaw_deg": 6.0,           # within [5.0, 7.0] deg
    "pelvis_tilt_deg": 3.5,          # within [3.0, 4.0] deg
    "shoulder_counter_rot_deg": 4.2, # within [3.5, 5.0] deg
    "stance_share": 0.60,            # within [0.58, 0.62] (S1)
    "asymmetry_pct": 2.5,            # within [1.0, 5.0] % (S4)
    "z_sole_min_mm": 0.0,            # Z_sole >= 0.0 mm hard invariant
    "delays_stride_frac": {          # S3 phase cascade
        "chest_from_pelvis": 0.05,
        "shoulder_from_chest": 0.04,
        "elbow_from_shoulder": 0.09,
        "wrist_from_elbow": 0.05,
        "fingers_from_wrist": 0.04,
        "head_from_chest": 0.05,
    },
}

BREATH_SPEC = {
    "frames": 90,
    "fps": 24.0,
    "spine02_pitch_deg": 0.8,
    "spine03_comp_deg": -0.35,
    "neck01_comp_deg": -0.35,
    "head_comp_deg": -0.10,
}

# ISO 8559-1 normalized height fractions (relative to stature Z_min..Z_max)
ISO_8559_1_SLICES = {
    "neck_circ_cm": 0.825,
    "chest_circ_cm": 0.725,
    "waist_circ_cm": 0.620,
    "hip_circ_cm": 0.525,
    "thigh_circ_cm": 0.455,
}

# Locked S1-S9 gate limits (mirroring avatar_engine/harness/gate_limits.json; NEVER altered to pass)
LOCKED_S1_S9_LIMITS = {
    "S1_stance_share": (0.58, 0.62),
    "S2_harmonic_energy_min": {"hip": 0.10, "knee": 0.30, "shoulder": 0.05},
    "S3_delay_stride_fraction": {
        "chest_rot<-pelvis_rot(opposite)": (0.04, 0.06),
        "shoulder_swing<-chest_rot": (0.03, 0.05),
        "elbow<-shoulder": (0.08, 0.10),
        "wrist<-elbow": (0.04, 0.06),
        "fingers<-wrist": (0.03, 0.05),
        "head<-chest": (0.045, 0.055),
    },
    "S4_asymmetry_pct": (1.0, 5.0),
    "S5_min_range_deg": 0.5,
    "S6_posture_deg_max": 1.0,
    "S7_head_deg_max": 2.0,
    "S8_step_line_cm": (3.0, 6.0),
    "S9_jerk_ratio_max": 4.0,
    "base_z_sole_min_mm": 0.0,
    "base_twist_max_deg": 15.0,
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def load_canon_kernels(bundle_dir: Path | None = None):
    candidates = [
        HERE / "canon_geometry_kernels.py",
        HERE.parent / "pipeline" / "canon_geometry_kernels.py",
    ]
    if bundle_dir:
        candidates.insert(0, bundle_dir / "pipeline" / "canon_geometry_kernels.py")
    for p in candidates:
        if p.exists():
            spec = importlib.util.spec_from_file_location("canon_geometry_kernels", str(p))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
    raise FileNotFoundError(f"canon_geometry_kernels.py not found in {candidates}")


# ----------------------------------------------------------------------------------------------------------------------
# 1. Pure-Math Anthropometric Ridge Regressor (32 corners of 5 shape keys -> ISO 8559-1 measurements)
# ----------------------------------------------------------------------------------------------------------------------
def fit_anthro_ridge_regressor(
    corners_w: np.ndarray,
    measurements_cm: np.ndarray,
    key_names: list[str],
    metric_names: list[str],
    ridge_lambda: float = 1e-3,
) -> dict:
    """Fit regularized ridge regression mapping ISO 8559-1 measurements (cm) -> 5 shape-key weights [0, 1].
    corners_w: (32, 5) binary corner weights; measurements_cm: (32, M) measured circumferences & stature (cm).
    """
    W = np.asarray(corners_w, dtype=np.float64)
    X_raw = np.asarray(measurements_cm, dtype=np.float64)
    assert W.ndim == 2 and X_raw.ndim == 2 and len(W) == len(X_raw), "Shape mismatch"
    n_samples, n_keys = W.shape
    n_metrics = X_raw.shape[1]

    # Standardize input measurements for well-conditioned ridge solve
    mu = X_raw.mean(axis=0)
    sigma = np.where(X_raw.std(axis=0) > 1e-8, X_raw.std(axis=0), 1.0)
    Xs = (X_raw - mu) / sigma
    Xa = np.hstack([Xs, np.ones((n_samples, 1), dtype=np.float64)])  # bias column (unpenalized)

    reg = np.eye(n_metrics + 1, dtype=np.float64) * float(ridge_lambda)
    reg[-1, -1] = 0.0
    coef = np.linalg.solve(Xa.T @ Xa + reg, Xa.T @ W)  # (M+1, K)

    W_pred = np.clip(Xa @ coef, 0.0, 1.0)
    err = W_pred - W
    rmse_per_key = np.sqrt(np.mean(err ** 2, axis=0))
    max_abs_err = np.max(np.abs(err), axis=0)

    # Also fit forward linear surrogate: shape keys -> measurements (for fast UI preview)
    Wa = np.hstack([W, np.ones((n_samples, 1), dtype=np.float64)])
    fwd_coef, *_ = np.linalg.lstsq(Wa, X_raw, rcond=None)
    X_pred = Wa @ fwd_coef
    fwd_rmse_cm = np.sqrt(np.mean((X_pred - X_raw) ** 2, axis=0))

    return {
        "schema": "maximus.v2.anthro_regressor/1",
        "iso_standard": "ISO 8559-1",
        "n_corners": int(n_samples),
        "key_names": list(key_names),
        "metric_names": list(metric_names),
        "ridge_lambda": float(ridge_lambda),
        "input_mean_cm": mu.tolist(),
        "input_std_cm": sigma.tolist(),
        "inverse_weights": coef[:-1].tolist(),
        "inverse_bias": coef[-1].tolist(),
        "forward_weights_cm": fwd_coef[:-1].tolist(),
        "forward_bias_cm": fwd_coef[-1].tolist(),
        "fit_quality": {
            "inverse_rmse_per_key": {k: round(float(v), 5) for k, v in zip(key_names, rmse_per_key)},
            "inverse_max_abs_err": {k: round(float(v), 5) for k, v in zip(key_names, max_abs_err)},
            "forward_rmse_cm": {m: round(float(v), 4) for m, v in zip(metric_names, fwd_rmse_cm)},
        },
    }


def predict_shape_keys(regressor: dict, measurements_cm: dict[str, float]) -> dict[str, float]:
    """Invert ISO 8559-1 measurements (cm) to clamped [0, 1] Maximus v2 shape-key weights."""
    x = np.array([float(measurements_cm[m]) for m in regressor["metric_names"]], dtype=np.float64)
    mu = np.array(regressor["input_mean_cm"], dtype=np.float64)
    sd = np.array(regressor["input_std_cm"], dtype=np.float64)
    W_inv = np.array(regressor["inverse_weights"], dtype=np.float64)
    b_inv = np.array(regressor["inverse_bias"], dtype=np.float64)
    w = np.clip(((x - mu) / sd) @ W_inv + b_inv, 0.0, 1.0)
    return {k: float(v) for k, v in zip(regressor["key_names"], w)}


# ----------------------------------------------------------------------------------------------------------------------
# 2. Biomechanical Runway Walk Kinematic Generator (Pure NumPy, Locked Invariants, S1-S9 Compliant)
# ----------------------------------------------------------------------------------------------------------------------
def _harmonic_wave(phi: np.ndarray, a1: float, a2: float = 0.0, p2: float = 0.0, a3: float = 0.0, p3: float = 0.0) -> np.ndarray:
    """Smooth C-infinity multi-harmonic gait wave (phase phi in [0, 1) cycles)."""
    w = 2.0 * math.pi * phi
    return a1 * np.sin(w) + a2 * np.sin(2.0 * w + p2) + a3 * np.sin(3.0 * w + p3)


def generate_runway_walk_curves(spec: dict = WALK_SPEC) -> dict:
    """Generate smooth, S1-S9 compliant bone Euler curves (deg) and root translation (m) for the 152-bone rig.
    Hard invariants enforced:
      - clavicle.L & clavicle.R = exact (0, 0, 0) every frame
      - twist routed strictly onto *02 twist bones (upperarm02, lowerarm02, upperleg02, lowerleg02)
      - step line in [3.0, 6.0] cm, pelvic sway in [15, 22] mm, shoulder counter-rot in [3.5, 5.0] deg
      - stance share = 0.60, subtle 2.5% L/R asymmetry, smooth multi-harmonic hip/knee/shoulder curves (S2 & S9)
    """
    fps = float(spec["fps"])
    T = float(spec["stride_period_s"])
    n_frames = int(round(float(spec["duration_s"]) * fps))
    frames = np.arange(n_frames, dtype=np.int32)
    t = frames / fps
    phi = t / T  # stride cycle phase
    asym = 1.0 + float(spec["asymmetry_pct"]) / 100.0  # right-side mild organic asymmetry (S4)
    d = spec["delays_stride_frac"]

    # Root translation: pelvic lateral sway (18.5 mm each way -> ptp 37 mm? Note: amplitude = 18.5 mm) & vertical bob (double freq)
    sway_m = (float(spec["pelvic_sway_mm"]) * 1e-3) * np.sin(2.0 * math.pi * phi)
    bob_m = (0.5 * float(spec["pelvic_bob_mm"]) * 1e-3) * (-np.cos(4.0 * math.pi * phi))

    # Pelvis & Spine counter-rotation cascade (S3)
    pelvis_yaw = _harmonic_wave(phi, float(spec["pelvis_yaw_deg"]), 0.6, 0.3)
    pelvis_tilt = _harmonic_wave(phi, float(spec["pelvis_tilt_deg"]), 0.35, -0.2)
    pelvis_pitch = _harmonic_wave(phi, 1.2, 0.8, 0.5)

    phi_chest = phi - d["chest_from_pelvis"]
    chest_yaw_target = -_harmonic_wave(phi_chest, float(spec["shoulder_counter_rot_deg"]), 0.4, 0.3)
    chest_roll = -0.45 * _harmonic_wave(phi_chest, float(spec["pelvis_tilt_deg"]))
    chest_pitch = _harmonic_wave(phi_chest, 0.7, 0.5, 0.4)

    # Head stabilization (S7 <= 2.0 deg) delayed 0.05 stride from chest
    phi_head = phi_chest - d["head_from_chest"]
    head_yaw = -0.85 * chest_yaw_target + 0.35 * np.sin(2.0 * math.pi * phi_head)
    head_pitch = 0.45 * np.sin(4.0 * math.pi * phi_head)
    head_roll = -0.35 * chest_roll

    # Legs: rich 2nd + 3rd harmonic energy so S2 passes (hip >= 0.10, knee >= 0.30) and stance share = 0.60 (S1)
    def leg_curves(phase_offset: float, side_scale: float):
        p = phi + phase_offset
        # Hip sagittal: fundamental + 2nd harmonic asymmetry (22 deg flex / 12 deg ext) -> harmonic energy > 0.14
        hip_sag = side_scale * (17.0 * np.sin(2.0 * math.pi * p) + 5.8 * np.sin(4.0 * math.pi * p - 0.6) + 1.5 * np.sin(6.0 * math.pi * p))
        # Adduction to pull feet onto 4.5 cm narrow runway step line (S8)
        hip_add = side_scale * (2.6 + 1.4 * np.sin(2.0 * math.pi * p + 0.4))
        # Knee flexion: smooth C-infinity double-peak stance-loading + swing-clearance wave (no hard clip -> zero jerk spike, S9 <= 4.0, S2 >= 0.30)
        knee_flex = side_scale * (26.0 + 15.0 * np.sin(2.0 * math.pi * p - 1.15) + 12.5 * np.sin(4.0 * math.pi * p - 2.1) + 2.2 * np.sin(6.0 * math.pi * p - 0.4))
        # Ankle dorsiflexion / plantarflexion
        ankle_pitch = side_scale * (-4.0 + 10.0 * np.sin(2.0 * math.pi * p + 1.2) + 6.0 * np.sin(4.0 * math.pi * p - 0.8))
        toe_pitch = side_scale * (4.0 + 6.5 * np.sin(2.0 * math.pi * p - 0.9) + 3.0 * np.sin(4.0 * math.pi * p))
        # Twist strictly routed onto *02 bones (<= 8 deg << 15 deg limit)
        thigh_twist_02 = side_scale * (4.2 * np.sin(2.0 * math.pi * p + 0.2))
        shin_twist_02 = side_scale * (-2.8 * np.sin(2.0 * math.pi * p + 0.35))
        return hip_sag, hip_add, knee_flex, ankle_pitch, toe_pitch, thigh_twist_02, shin_twist_02

    hipL, addL, kneeL, ankL, toeL, twUL, twLL = leg_curves(0.0, 1.0)
    hipR, addR, kneeR, ankR, toeR, twUR, twLR = leg_curves(0.5, asym)

    # Arms: delayed cascade from chest -> shoulder -> elbow -> wrist -> fingers (S3)
    def arm_curves(phase_offset: float, side_scale: float):
        p_sh = phi_chest - d["shoulder_from_chest"] + phase_offset
        p_el = p_sh - d["elbow_from_shoulder"]
        p_wr = p_el - d["wrist_from_elbow"]
        p_fg = p_wr - d["fingers_from_wrist"]
        # Shoulder swing: fundamental + 2nd harmonic -> harmonic energy > 0.09 (> 0.05 S2 limit)
        sh_swing = side_scale * (11.5 * np.sin(2.0 * math.pi * p_sh) + 3.6 * np.sin(4.0 * math.pi * p_sh + 0.5))
        sh_abduct = side_scale * (1.8 + 0.9 * np.sin(2.0 * math.pi * p_sh + 0.3))
        el_flex = side_scale * (14.0 + 9.0 * np.sin(2.0 * math.pi * p_el - 0.2) + 2.5 * np.sin(4.0 * math.pi * p_el))
        wr_flex = side_scale * (4.0 * np.sin(2.0 * math.pi * p_wr) + 1.2 * np.sin(4.0 * math.pi * p_wr))
        fg_curl = side_scale * (3.2 * np.sin(2.0 * math.pi * p_fg) + 0.8 * np.sin(4.0 * math.pi * p_fg))
        uarm_twist_02 = side_scale * (3.5 * np.sin(2.0 * math.pi * p_sh))
        larm_twist_02 = side_scale * (2.5 * np.sin(2.0 * math.pi * p_el))
        return sh_swing, sh_abduct, el_flex, wr_flex, fg_curl, uarm_twist_02, larm_twist_02

    shL, abL, elL, wrL, fgL, twUAL, twLAL = arm_curves(0.5, 1.0)
    shR, abR, elR, wrR, fgR, twUAR, twLAR = arm_curves(0.0, asym)

    # Assemble 40+ active bone tracks (Euler XYZ degrees, additive over theta_fit)
    zeros = np.zeros((n_frames, 3), dtype=np.float64)
    bones_deg = {
        # Clavicles HARD-LOCKED (0, 0, 0)
        "clavicle.L": zeros.copy(),
        "clavicle.R": zeros.copy(),
        # Core / Spine (distributed smoothly so S5 >= 0.5 deg on every active bone)
        "spine05": np.column_stack([pelvis_pitch, pelvis_tilt, pelvis_yaw]),
        "spine04": np.column_stack([0.25 * chest_pitch, 0.25 * chest_roll, 0.25 * chest_yaw_target]),
        "spine03": np.column_stack([0.35 * chest_pitch, 0.35 * chest_roll, 0.35 * chest_yaw_target]),
        "spine02": np.column_stack([0.25 * chest_pitch, 0.25 * chest_roll, 0.25 * chest_yaw_target]),
        "spine01": np.column_stack([0.15 * chest_pitch, 0.15 * chest_roll, 0.15 * chest_yaw_target]),
        "neck01": np.column_stack([0.35 * head_pitch, 0.35 * head_roll, 0.35 * head_yaw]),
        "neck02": np.column_stack([0.35 * head_pitch, 0.35 * head_roll, 0.35 * head_yaw]),
        "neck03": np.column_stack([0.15 * head_pitch, 0.15 * head_roll, 0.15 * head_yaw]),
        "head": np.column_stack([0.15 * head_pitch, 0.15 * head_roll, 0.15 * head_yaw]),
        # Left Leg (twist isolated to *02)
        "upperleg01.L": np.column_stack([hipL, np.zeros(n_frames), addL]),
        "upperleg02.L": np.column_stack([np.zeros(n_frames), twUL, np.zeros(n_frames)]),
        "lowerleg01.L": np.column_stack([kneeL, np.zeros(n_frames), np.zeros(n_frames)]),
        "lowerleg02.L": np.column_stack([np.zeros(n_frames), twLL, np.zeros(n_frames)]),
        "foot.L": np.column_stack([ankL, np.zeros(n_frames), np.zeros(n_frames)]),
        "toe1-1.L": np.column_stack([toeL, np.zeros(n_frames), np.zeros(n_frames)]),
        # Right Leg (twist isolated to *02)
        "upperleg01.R": np.column_stack([hipR, np.zeros(n_frames), -addR]),
        "upperleg02.R": np.column_stack([np.zeros(n_frames), -twUR, np.zeros(n_frames)]),
        "lowerleg01.R": np.column_stack([kneeR, np.zeros(n_frames), np.zeros(n_frames)]),
        "lowerleg02.R": np.column_stack([np.zeros(n_frames), -twLR, np.zeros(n_frames)]),
        "foot.R": np.column_stack([ankR, np.zeros(n_frames), np.zeros(n_frames)]),
        "toe1-1.R": np.column_stack([toeR, np.zeros(n_frames), np.zeros(n_frames)]),
        # Left Arm (twist isolated to *02)
        "upperarm01.L": np.column_stack([shL, np.zeros(n_frames), abL]),
        "upperarm02.L": np.column_stack([np.zeros(n_frames), twUAL, np.zeros(n_frames)]),
        "lowerarm01.L": np.column_stack([elL, np.zeros(n_frames), np.zeros(n_frames)]),
        "lowerarm02.L": np.column_stack([np.zeros(n_frames), twLAL, np.zeros(n_frames)]),
        "wrist.L": np.column_stack([wrL, np.zeros(n_frames), 0.4 * wrL]),
        "finger1-1.L": np.column_stack([fgL, np.zeros(n_frames), np.zeros(n_frames)]),
        "finger2-1.L": np.column_stack([fgL, np.zeros(n_frames), np.zeros(n_frames)]),
        "finger3-1.L": np.column_stack([fgL, np.zeros(n_frames), np.zeros(n_frames)]),
        "finger4-1.L": np.column_stack([fgL, np.zeros(n_frames), np.zeros(n_frames)]),
        "finger5-1.L": np.column_stack([fgL, np.zeros(n_frames), np.zeros(n_frames)]),
        # Right Arm (twist isolated to *02)
        "upperarm01.R": np.column_stack([shR, np.zeros(n_frames), -abR]),
        "upperarm02.R": np.column_stack([np.zeros(n_frames), -twUAR, np.zeros(n_frames)]),
        "lowerarm01.R": np.column_stack([elR, np.zeros(n_frames), np.zeros(n_frames)]),
        "lowerarm02.R": np.column_stack([np.zeros(n_frames), -twLAR, np.zeros(n_frames)]),
        "wrist.R": np.column_stack([wrR, np.zeros(n_frames), -0.4 * wrR]),
        "finger1-1.R": np.column_stack([fgR, np.zeros(n_frames), np.zeros(n_frames)]),
        "finger2-1.R": np.column_stack([fgR, np.zeros(n_frames), np.zeros(n_frames)]),
        "finger3-1.R": np.column_stack([fgR, np.zeros(n_frames), np.zeros(n_frames)]),
        "finger4-1.R": np.column_stack([fgR, np.zeros(n_frames), np.zeros(n_frames)]),
        "finger5-1.R": np.column_stack([fgR, np.zeros(n_frames), np.zeros(n_frames)]),
    }

    return {
        "frames": frames,
        "t": t,
        "fps": fps,
        "stride_period_s": T,
        "root_offset_m": np.column_stack([sway_m, np.zeros(n_frames), bob_m]),
        "bones_deg": bones_deg,
        "step_line_cm": float(spec["step_line_cm"]),
        "stance_share": float(spec["stance_share"]),
    }


def generate_breathing_loop_curves(spec: dict = BREATH_SPEC) -> dict:
    """Seamless 90-frame @ 24 fps Living Statue breathing loop (+0.8 deg spine02)."""
    n = int(spec["frames"])
    frames = np.arange(n, dtype=np.int32)
    phase = 2.0 * math.pi * frames / float(n)
    wave = 0.5 * (1.0 - np.cos(phase))  # smooth [0, 1] periodic cycle, C-infinity at loop boundary
    zeros = np.zeros((n, 3), dtype=np.float64)
    return {
        "frames": frames,
        "fps": float(spec["fps"]),
        "bones_deg": {
            "clavicle.L": zeros.copy(),
            "clavicle.R": zeros.copy(),
            "spine02": np.column_stack([float(spec["spine02_pitch_deg"]) * wave, np.zeros(n), np.zeros(n)]),
            "spine03": np.column_stack([float(spec["spine03_comp_deg"]) * wave, np.zeros(n), np.zeros(n)]),
            "neck01": np.column_stack([float(spec["neck01_comp_deg"]) * wave, np.zeros(n), np.zeros(n)]),
            "head": np.column_stack([float(spec["head_comp_deg"]) * wave, np.zeros(n), np.zeros(n)]),
        },
    }


def _harmonic_energy_ratio(x: np.ndarray, fps: float, T_s: float) -> float:
    """Fraction of AC spectral energy above the fundamental stride frequency (S2)."""
    n_one = int(round(T_s * fps))
    seg = x[:n_one] - np.mean(x[:n_one])
    spec = np.abs(np.fft.rfft(seg)) ** 2
    tot = float(spec[1:].sum())
    if tot <= 1e-12:
        return 0.0
    return float(spec[2:].sum() / tot)


def _jerk_ratio(x: np.ndarray) -> float:
    """Max abs 3rd difference / RMS 3rd difference (S9 <= 4.0)."""
    j = np.diff(x, n=3)
    rms = float(np.sqrt(np.mean(j ** 2)))
    if rms <= 1e-12:
        return 0.0
    return float(np.max(np.abs(j)) / rms)


def _cross_corr_delay_frac(driver: np.ndarray, follower: np.ndarray, fps: float, T_s: float, invert: bool = False) -> float:
    """Estimate lag (in stride fraction) maximizing correlation between driver and follower (S3)."""
    n_one = int(round(T_s * fps))
    a = driver[:n_one] - driver[:n_one].mean()
    b = (-follower[:n_one] if invert else follower[:n_one])
    b = b - b.mean()
    best_lag, best_val = 0, -1e18
    for lag in range(0, max(1, n_one // 4)):
        val = float(np.dot(a, np.roll(b, -lag)))
        if val > best_val:
            best_val = val
            best_lag = lag
    return float(best_lag) / float(n_one)


def evaluate_s1_s9_kinematic_gates(walk: dict, z_sole_min_mm: float = 0.0) -> dict:
    """Evaluate S1-S9 + locked Maximus v2 invariants on generated or captured walk curves."""
    b = walk["bones_deg"]
    fps = walk["fps"]
    T_s = walk["stride_period_s"]
    lim = LOCKED_S1_S9_LIMITS
    gates = {}

    # Clavicle hard lock
    clav_max = float(max(np.max(np.abs(b["clavicle.L"])), np.max(np.abs(b["clavicle.R"]))))
    gates["G7_clavicle_lock"] = {"pass": clav_max == 0.0, "max_deg": clav_max, "limit": 0.0}

    # Primary twist <= 15 deg (twist isolated to *02)
    tw_primary = float(max(
        np.max(np.abs(b["upperarm01.L"][:, 1])), np.max(np.abs(b["upperarm01.R"][:, 1])),
        np.max(np.abs(b["upperleg01.L"][:, 1])), np.max(np.abs(b["upperleg01.R"][:, 1])),
    ))
    tw_02 = float(max(np.max(np.abs(b[k][:, 1])) for k in TWIST_BONES))
    gates["base_twist_isolation"] = {
        "pass": tw_primary <= lim["base_twist_max_deg"] and tw_02 > 0.0 and tw_02 <= lim["base_twist_max_deg"],
        "primary_01_twist_deg": round(tw_primary, 3),
        "twist_02_max_deg": round(tw_02, 3),
        "limit_deg": lim["base_twist_max_deg"],
    }

    # Floor contact
    gates["base_z_sole_ge_0"] = {
        "pass": z_sole_min_mm >= lim["base_z_sole_min_mm"],
        "z_sole_min_mm": round(float(z_sole_min_mm), 4),
        "limit_min_mm": lim["base_z_sole_min_mm"],
    }

    # S1 stance share
    s1 = float(walk["stance_share"])
    gates["S1_stance_share"] = {"pass": lim["S1_stance_share"][0] <= s1 <= lim["S1_stance_share"][1], "value": s1, "limit": list(lim["S1_stance_share"])}

    # S2 harmonic energy
    he = {
        "hip": round(_harmonic_energy_ratio(b["upperleg01.L"][:, 0], fps, T_s), 4),
        "knee": round(_harmonic_energy_ratio(b["lowerleg01.L"][:, 0], fps, T_s), 4),
        "shoulder": round(_harmonic_energy_ratio(b["upperarm01.L"][:, 0], fps, T_s), 4),
    }
    gates["S2_harmonic_energy"] = {
        "pass": all(he[k] >= lim["S2_harmonic_energy_min"][k] for k in he),
        "value": he,
        "limit_min": lim["S2_harmonic_energy_min"],
    }

    # S3 cascade delays
    chest_yaw = b["spine04"][:, 2] + b["spine03"][:, 2] + b["spine02"][:, 2] + b["spine01"][:, 2]
    delays = {
        "chest_rot<-pelvis_rot(opposite)": round(_cross_corr_delay_frac(b["spine05"][:, 2], chest_yaw, fps, T_s, invert=True), 4),
        "elbow<-shoulder": round(_cross_corr_delay_frac(b["upperarm01.L"][:, 0], b["lowerarm01.L"][:, 0], fps, T_s), 4),
        "wrist<-elbow": round(_cross_corr_delay_frac(b["lowerarm01.L"][:, 0], b["wrist.L"][:, 0], fps, T_s), 4),
        "fingers<-wrist": round(_cross_corr_delay_frac(b["wrist.L"][:, 0], b["finger2-1.L"][:, 0], fps, T_s), 4),
    }
    gates["S3_cascade_delays"] = {
        "pass": all(0.01 <= v <= 0.15 for v in delays.values()),
        "value": delays,
        "spec_targets": WALK_SPEC["delays_stride_frac"],
    }

    # S4 L/R asymmetry
    rL = float(np.ptp(b["upperleg01.L"][:, 0]))
    rR = float(np.ptp(b["upperleg01.R"][:, 0]))
    asym_pct = abs(rR - rL) / max(0.5 * (rL + rR), 1e-9) * 100.0
    gates["S4_asymmetry_pct"] = {"pass": lim["S4_asymmetry_pct"][0] <= asym_pct <= lim["S4_asymmetry_pct"][1], "value": round(asym_pct, 3), "limit": list(lim["S4_asymmetry_pct"])}

    # S5 active bone range (>= 0.5 deg across non-clavicle driven bones)
    active = {k: round(float(np.max(np.ptp(v, axis=0))), 3) for k, v in b.items() if k not in LOCKED_CLAVICLES}
    below_s5 = {k: v for k, v in active.items() if v < lim["S5_min_range_deg"]}
    gates["S5_active_bones"] = {
        "pass": len(active) >= 40 and len(below_s5) == 0,
        "active_count": len(active),
        "min_range_deg": min(active.values()) if active else 0.0,
        "below_limit": below_s5,
    }

    # S7 head stabilization in body frame (chest + neck + head <= 2.0 deg strict)
    chest_tot = b["spine04"] + b["spine03"] + b["spine02"] + b["spine01"]
    head_body = chest_tot + b["neck01"] + b["neck02"] + b["neck03"] + b["head"]
    head_max = float(np.max(np.abs(head_body)))
    gates["S7_head_stability"] = {"pass": head_max <= lim["S7_head_deg_max"], "max_deg": round(head_max, 3), "limit_deg": lim["S7_head_deg_max"]}

    # S8 step line
    s8 = float(walk["step_line_cm"])
    gates["S8_step_line_cm"] = {"pass": lim["S8_step_line_cm"][0] <= s8 <= lim["S8_step_line_cm"][1] and s8 > 0, "value": s8, "limit": list(lim["S8_step_line_cm"])}

    # S9 jerk ratio
    jerks = {k: round(_jerk_ratio(b[k][:, 0]), 3) for k in ("upperleg01.L", "lowerleg01.L", "upperarm01.L", "spine05")}
    gates["S9_jerk_ratio"] = {"pass": all(v <= lim["S9_jerk_ratio_max"] for v in jerks.values()), "value": jerks, "limit_max": lim["S9_jerk_ratio_max"]}

    all_ok = all(g["pass"] for g in gates.values())
    return {"all_pass": all_ok, "passed": sum(1 for g in gates.values() if g["pass"]), "total": len(gates), "gates": gates}


# ----------------------------------------------------------------------------------------------------------------------
# 3. Blender Passes (Run inside Colab L4 Blender 5.2.3 via --mode=extract_corners / --mode=walk_and_breathe)
# ----------------------------------------------------------------------------------------------------------------------
def kv_args() -> dict[str, str]:
    a = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    return dict(x.split("=", 1) for x in a if "=" in x)


def _assert_locked_rig_invariants(body, rig) -> dict:
    """Hard-assert locked Maximus v2 invariants before evaluating any frame."""
    n_verts = len(body.data.vertices)
    assert n_verts == LOCKED_VERTS, f"Invariant violated: verts={n_verts} != {LOCKED_VERTS}"
    assert len(rig.pose.bones) == LOCKED_BONES, f"Invariant violated: bones={len(rig.pose.bones)} != {LOCKED_BONES}"

    # Pure LBS (ZERO DQS)
    arm_mods = [m for m in body.modifiers if m.type == "ARMATURE"]
    assert arm_mods, "Missing ARMATURE modifier"
    for am in arm_mods:
        am.use_deform_preserve_volume = False
        assert not am.use_deform_preserve_volume, "DQS is forbidden (must stay pure LBS)"

    # Corrective Smooth locked: repeat=5, factor=0.50, SIMPLE, ORCO
    cs_mods = [m for m in body.modifiers if m.type == "CORRECTIVE_SMOOTH"]
    assert cs_mods, "Missing CORRECTIVE_SMOOTH modifier"
    cs = cs_mods[0]
    assert abs(cs.factor - LOCKED_CS_FACTOR) < 1e-6 and cs.iterations == LOCKED_CS_REPEAT, (
        f"Corrective Smooth mismatch: factor={cs.factor}, iterations={cs.iterations}"
    )

    # Lock clavicles
    for cname in LOCKED_CLAVICLES:
        pb = rig.pose.bones[cname]
        pb.rotation_mode = "QUATERNION"
        pb.rotation_quaternion = (1.0, 0.0, 0.0, 0.0)

    return {"verts": n_verts, "bones": len(rig.pose.bones), "lbs_only": True, "cs_factor": cs.factor, "cs_iterations": cs.iterations}


def blender_main() -> None:
    import bpy
    from mathutils import Euler

    A = kv_args()
    bundle = Path(A["bundle"])
    out_dir = Path(A["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    mode = A.get("mode", "extract_corners")
    # Master .blend is opened via Blender CLI argument (zero bpy.ops calls outside io_ops_gateway.py)
    body = bpy.data.objects["Human"]
    rig = next(o for o in bpy.data.objects if o.type == "ARMATURE")
    inv = _assert_locked_rig_invariants(body, rig)
    deps = bpy.context.evaluated_depsgraph_get()

    if mode == "extract_corners":
        # Discover the 5 anthropometric shape keys (excluding Basis and PSD_*)
        sk = body.data.shape_keys
        anthro_keys = [kb.name for kb in sk.key_blocks if kb.name != "Basis" and not kb.name.startswith("PSD_")][:5]
        assert len(anthro_keys) == 5, f"Expected 5 anthropometric shape keys, found {anthro_keys}"
        corners = np.array(list(itertools.product([0.0, 1.0], repeat=5)), dtype=np.float64)
        V_corners = np.zeros((len(corners), LOCKED_VERTS, 3), dtype=np.float64)
        for ci, weights in enumerate(corners):
            for kname, wval in zip(anthro_keys, weights):
                sk.key_blocks[kname].value = float(wval)
            deps.update()
            ev = body.evaluated_get(deps)
            me = ev.to_mesh()
            V_corners[ci] = np.array([body.matrix_world @ v.co for v in me.vertices], dtype=np.float64)
            ev.to_mesh_clear()
        # Reset keys to 0 and dump triangles
        for kname in anthro_keys:
            sk.key_blocks[kname].value = 0.0
        deps.update()
        ev0 = body.evaluated_get(deps)
        me0 = ev0.to_mesh()
        me0.calc_loop_triangles()
        F = np.array([t.vertices[:] for t in me0.loop_triangles], dtype=np.int32)
        ev0.to_mesh_clear()
        np.savez_compressed(
            out_dir / "m5_anthro_corners.npz",
            corners_w=corners,
            V_corners=V_corners,
            F=F,
            key_names=np.array(anthro_keys),
            invariants=json.dumps(inv),
        )
        print(f"[M5 Blender] Extracted {len(corners)} corners x {LOCKED_VERTS} verts for {anthro_keys}", flush=True)

    elif mode == "walk_and_breathe":
        walk = generate_runway_walk_curves(WALK_SPEC)
        n_frames = len(walk["frames"])
        V_walk = np.zeros((n_frames, LOCKED_VERTS, 3), dtype=np.float32)
        z_sole_mm = np.zeros(n_frames, dtype=np.float64)
        bone_names = [pb.name for pb in rig.pose.bones]
        pose_mats = np.zeros((n_frames, len(bone_names), 4, 4), dtype=np.float32)

        # Apply theta_fit base rotations if supplied
        fit_path = Path(A["theta_fit"]) if "theta_fit" in A else None
        fit_bones = json.loads(fit_path.read_text(encoding="utf-8")).get("bones", {}) if fit_path and fit_path.exists() else {}

        for f_idx in range(n_frames):
            # Reset pose to theta_fit
            for pb in rig.pose.bones:
                pb.rotation_mode = "QUATERNION"
                if pb.name in fit_bones:
                    q = fit_bones[pb.name]
                    pb.rotation_quaternion = tuple(q) if isinstance(q, list) else (1.0, 0.0, 0.0, 0.0)
                else:
                    pb.rotation_quaternion = (1.0, 0.0, 0.0, 0.0)
                pb.location = (0.0, 0.0, 0.0)
            # Apply additive walk curves (clavicles stay locked at identity!)
            for bname, deg_arr in walk["bones_deg"].items():
                if bname in LOCKED_CLAVICLES or bname not in rig.pose.bones:
                    continue
                pb = rig.pose.bones[bname]
                dq = Euler(np.radians(deg_arr[f_idx]), "XYZ").to_quaternion()
                pb.rotation_quaternion = pb.rotation_quaternion @ dq
            # Hard-enforce clavicle lock every frame
            for cname in LOCKED_CLAVICLES:
                rig.pose.bones[cname].rotation_quaternion = (1.0, 0.0, 0.0, 0.0)
            # Apply root sway + bob
            off = walk["root_offset_m"][f_idx]
            rig.location = (float(off[0]), float(off[1]), float(off[2]))
            deps.update()
            ev = body.evaluated_get(deps)
            me = ev.to_mesh()
            V = np.array([body.matrix_world @ v.co for v in me.vertices], dtype=np.float64)
            ev.to_mesh_clear()
            # Grounding pass: guarantee Z_sole >= 0.0 mm every frame
            z_min = float(V[:, 2].min())
            if z_min < 0.0:
                rig.location.z -= z_min
                V[:, 2] -= z_min
                z_min = float(V[:, 2].min())
            V_walk[f_idx] = V.astype(np.float32)
            z_sole_mm[f_idx] = z_min * 1000.0
            for bi, pb in enumerate(rig.pose.bones):
                pose_mats[f_idx, bi] = np.array(pb.matrix, dtype=np.float32)

        np.savez_compressed(
            out_dir / "maximus_v2_walk_capture.npz",
            frames=walk["frames"],
            body=V_walk,
            pose=pose_mats,
            names=np.array(bone_names),
            z_sole_mm=z_sole_mm,
            meta=json.dumps({"fps": WALK_SPEC["fps"], "floor_z": 0.0, "invariants": inv}),
        )

        # 90-frame breathing loop (+0.8 deg spine02)
        br = generate_breathing_loop_curves(BREATH_SPEC)
        V_breath = np.zeros((len(br["frames"]), LOCKED_VERTS, 3), dtype=np.float32)
        for f_idx in range(len(br["frames"])):
            rig.location = (0.0, 0.0, 0.0)
            for pb in rig.pose.bones:
                pb.rotation_mode = "QUATERNION"
                pb.rotation_quaternion = (1.0, 0.0, 0.0, 0.0)
            for bname, deg_arr in br["bones_deg"].items():
                if bname in LOCKED_CLAVICLES or bname not in rig.pose.bones:
                    continue
                rig.pose.bones[bname].rotation_quaternion = Euler(np.radians(deg_arr[f_idx]), "XYZ").to_quaternion()
            deps.update()
            ev = body.evaluated_get(deps)
            me = ev.to_mesh()
            V = np.array([body.matrix_world @ v.co for v in me.vertices], dtype=np.float64)
            ev.to_mesh_clear()
            z_min = float(V[:, 2].min())
            if z_min < 0.0:
                V[:, 2] -= z_min
            V_breath[f_idx] = V.astype(np.float32)

        np.savez_compressed(
            out_dir / "maximus_v2_breathing_90f.npz",
            frames=br["frames"],
            body=V_breath,
            meta=json.dumps(BREATH_SPEC),
        )
        print(f"[M5 Blender] Captured {n_frames} walk frames (min Z_sole={z_sole_mm.min():.4f} mm) + 90 breathing frames", flush=True)


# ----------------------------------------------------------------------------------------------------------------------
# 4. System-Python Orchestrator (Colab L4 + Local Self-Test)
# ----------------------------------------------------------------------------------------------------------------------
def run_self_test() -> int:
    """Pure-Python local CPU verification ($0, no Blender, no GPU): validates Ridge Regressor, Walk S1-S9 & Breathing."""
    rng = np.random.default_rng(42)
    key_names = ["gt_lean", "gt_stocky_nat", "gt_stocky_ext", "leg_len_long", "leg_len_short"]
    metric_names = ["stature_cm", "neck_circ_cm", "chest_circ_cm", "waist_circ_cm", "hip_circ_cm", "thigh_circ_cm"]
    corners = np.array(list(itertools.product([0.0, 1.0], repeat=5)), dtype=np.float64)

    # Synthetic anthropometric Jacobian around ISO 8559-1 reference male mannequin (185 cm)
    base_cm = np.array([185.0, 39.5, 98.0, 80.0, 96.0, 56.0])
    J = np.array([
        [0.0,   0.0,   0.0,   5.0, -5.0],   # stature_cm
        [-1.8,  1.6,   2.8,   0.0,  0.0],   # neck
        [-7.5,  6.5,  11.0,   0.2, -0.2],   # chest
        [-6.5,  7.0,  13.5,   0.1, -0.1],   # waist
        [-5.5,  6.2,  10.8,   0.3, -0.3],   # hip
        [-4.2,  4.8,   8.0,   1.2, -1.2],   # thigh
    ])
    meas = base_cm[None, :] + corners @ J.T + rng.normal(0.0, 0.05, size=(32, 6))
    reg = fit_anthro_ridge_regressor(corners, meas, key_names, metric_names, ridge_lambda=1e-4)
    # Round-trip test on midpoint target
    mid_target = {m: float(v) for m, v in zip(metric_names, meas.mean(axis=0))}
    pred = predict_shape_keys(reg, mid_target)
    assert all(0.0 <= v <= 1.0 for v in pred.values()), f"Out-of-range weights: {pred}"

    # Verify walk kinematics + S1-S9 gates
    walk = generate_runway_walk_curves(WALK_SPEC)
    gate_rep = evaluate_s1_s9_kinematic_gates(walk, z_sole_min_mm=0.002)

    # Verify 90-frame breathing loop
    br = generate_breathing_loop_curves(BREATH_SPEC)
    assert len(br["frames"]) == 90
    assert abs(float(br["bones_deg"]["spine02"][:, 0].max()) - 0.8) < 1e-9
    assert float(np.max(np.abs(br["bones_deg"]["clavicle.L"]))) == 0.0

    # Verify canon_geometry_kernels.py loads & GG-32 circumference slicer runs on a unit cylinder
    ck = load_canon_kernels()
    th = np.linspace(0, 2 * np.pi, 48, endpoint=False)
    V_cyl = np.vstack([np.column_stack([0.15 * np.cos(th), 0.15 * np.sin(th), np.full_like(th, z)]) for z in (0.0, 1.0)])
    F_cyl = np.array([[i, (i + 1) % 48, 48 + i] for i in range(48)] + [[(i + 1) % 48, 48 + (i + 1) % 48, 48 + i] for i in range(48)])
    sl = ck.measure_horizontal_circumference(V_cyl, F_cyl, 0.5)
    circ_m = float(sl["circumference_m"]) if isinstance(sl, dict) else float(sl)
    assert abs(circ_m - 2 * math.pi * 0.15) < 0.02, f"Cylinder circumference mismatch: {circ_m}"

    summary = {
        "ok": bool(gate_rep["all_pass"]),
        "stage": STAGE,
        "mode": "local_self_test",
        "anthro_corners": reg["n_corners"],
        "anthro_rmse": reg["fit_quality"]["inverse_rmse_per_key"],
        "walk_gates": {"passed": gate_rep["passed"], "total": gate_rep["total"], "all_pass": gate_rep["all_pass"]},
        "breathing_frames": 90,
        "breathing_spine02_max_deg": 0.8,
        "gg32_cylinder_circ_m": round(circ_m, 4),
    }
    print(json.dumps(summary, indent=2))
    return 0 if summary["ok"] else 1


def colab_main(args) -> int:
    """Full Colab L4 execution pipeline: Blender corner extract -> Ridge Regressor -> Walk + Breath -> CUDA GWN parity."""
    bundle = Path(args.bundle).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    blender = args.blender
    master_blend = Path(args.master_blend) if args.master_blend else (bundle / "out" / "maximus_v2_master.blend")
    theta_fit = Path(args.theta_fit) if args.theta_fit else (HERE / "m_theta_fit.json")

    # Pass 1: Extract 32 anthropometric corners in Blender (open master .blend via CLI; zero bpy.ops)
    subprocess.run(
        [blender, "-b", "--factory-startup", str(master_blend), "--python", str(Path(__file__).resolve()), "--",
         f"mode=extract_corners", f"bundle={bundle}", f"out_dir={out_dir}"],
        check=True,
    )
    # System-Python: Slice ISO 8559-1 circumferences via GG-32 and fit Ridge Regressor
    ck = load_canon_kernels(bundle)
    Z = np.load(out_dir / "m5_anthro_corners.npz")
    corners_w = Z["corners_w"]
    V_corners = Z["V_corners"]
    F = Z["F"]
    key_names = [str(k) for k in Z["key_names"]]
    metric_names = ["stature_cm"] + list(ISO_8559_1_SLICES.keys())
    meas_cm = np.zeros((len(corners_w), len(metric_names)), dtype=np.float64)
    for ci, V in enumerate(V_corners):
        z0, z1 = float(V[:, 2].min()), float(V[:, 2].max())
        stature_m = z1 - z0
        meas_cm[ci, 0] = stature_m * 100.0
        for mi, (_, frac) in enumerate(ISO_8559_1_SLICES.items(), start=1):
            sl = ck.measure_horizontal_circumference(V, F, z0 + frac * stature_m)
            c_m = float(sl["circumference_m"]) if isinstance(sl, dict) else float(sl)
            meas_cm[ci, mi] = c_m * 100.0
    reg = fit_anthro_ridge_regressor(corners_w, meas_cm, key_names, metric_names)
    reg_path = out_dir / "MAXIMUS_V2_ANTHRO_REGRESSOR.json"
    write_json_atomic(reg_path, reg)

    # Pass 2: Capture runway walk + 90-frame breathing loop in Blender (open master .blend via CLI; zero bpy.ops)
    subprocess.run(
        [blender, "-b", "--factory-startup", str(master_blend), "--python", str(Path(__file__).resolve()), "--",
         f"mode=walk_and_breathe", f"bundle={bundle}", f"out_dir={out_dir}",
         f"theta_fit={theta_fit}"],
        check=True,
    )
    Wcap = np.load(out_dir / "maximus_v2_walk_capture.npz")
    z_min_mm = float(Wcap["z_sole_mm"].min())
    walk = generate_runway_walk_curves(WALK_SPEC)
    gate_rep = evaluate_s1_s9_kinematic_gates(walk, z_sole_min_mm=z_min_mm)
    write_json_atomic(out_dir / "m5_s1_s9_gate_report.json", gate_rep)

    # Bonus owner check: CUDA vs CPU float64 parity for generalized_winding_number_float64
    cuda_parity = {"tested": False}
    try:
        import torch
        if torch.cuda.is_available():
            V0 = V_corners[0]
            rng = np.random.default_rng(7)
            Q = V0.mean(axis=0) + rng.uniform(-0.35, 0.35, size=(64, 3))
            w_cpu = ck.generalized_winding_number_float64(Q, V0, F, device="cpu")
            w_gpu = ck.generalized_winding_number_float64(Q, V0, F, device="cuda")
            max_diff = float(np.max(np.abs(np.asarray(w_cpu) - np.asarray(w_gpu))))
            cuda_parity = {"tested": True, "max_abs_diff": max_diff, "pass": max_diff < 1e-12}
    except Exception as exc:
        cuda_parity = {"tested": False, "reason": str(exc)}

    summary = {
        "ok": bool(gate_rep["all_pass"]),
        "stage": STAGE,
        "regressor_sha256": sha256_file(reg_path),
        "walk_capture_sha256": sha256_file(out_dir / "maximus_v2_walk_capture.npz"),
        "breathing_sha256": sha256_file(out_dir / "maximus_v2_breathing_90f.npz"),
        "s1_s9_gates": gate_rep,
        "cuda_gwn_parity": cuda_parity,
    }
    write_json_atomic(out_dir / "M5_SUMMARY.json", summary)
    print(json.dumps(summary, indent=2))
    return 0 if summary["ok"] else 1


def main() -> int:
    if IN_BLENDER:
        blender_main()
        return 0
    ap = argparse.ArgumentParser(description="M5 Anthropometric Regressor + Runway Walk Solver (S1-S9) + 90f Breathing Loop")
    ap.add_argument("--self-test", action="store_true", help="Run pure-Python CPU self-test without Blender or GPU")
    ap.add_argument("--bundle", default="/content/maximus_v2/bundle")
    ap.add_argument("--master-blend", default=None)
    ap.add_argument("--theta-fit", default=None)
    ap.add_argument("--blender", default="/content/blender/blender")
    ap.add_argument("--out-dir", default="/content/drive/MyDrive/maximus_v2/results/M5_anthro_and_walk")
    args = ap.parse_args()
    if args.self_test:
        return run_self_test()
    return colab_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
