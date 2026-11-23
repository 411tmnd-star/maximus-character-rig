"""motion_emote_engine.py  (Maximus runtime_v3, module 4 of 8)  Parametric minimum-jerk motion: SWAG_01-04 pose strikes, EMOTE_01 groove,
and the micro-action used by the storefront fitting page. No captured or learned motion data (see LICENSE_BOUNDARY.md).

Minimum jerk (Flash & Hogan 1985): x(tau) = x0 + (x1 - x0)(10 tau^3 - 15 tau^4 + 6 tau^5), zero velocity and acceleration at both ends.
Joint targets are anatomical deltas (pitch, roll, yaw) in degrees per canonical joint, the same convention as the canonical pose library
(pose_retarget.py): pitch + = bend forward (arms: + = swing forward), roll + = right side up, yaw - = toward the model's right.
Clavicles are never driven (static-pose invariant). Gates measured on the posed mesh (Blender side): Z_sole >= 0 mm every frame,
foot slide <= 2 mm, hand corridor >= 15 mm (hand vertices to torso surface) for EMOTE_01.
"""
from __future__ import annotations

import math

import numpy as np


def min_jerk(tau):
    t = np.clip(np.asarray(tau, float), 0.0, 1.0); return t ** 3 * (10 - 15 * t + 6 * t * t)


def keyframe_track(keys, fps, duration):
    """keys: [(time_s, {joint: (p, r, y)})]; minimum-jerk between consecutive keys; holds at the ends. Returns frames x {joint: (n,3)}."""
    n = int(round(duration * fps)); t = np.arange(n) / fps; joints = sorted({j for _, d in keys for j in d}); out = {j: np.zeros((n, 3)) for j in joints}
    for j in joints:
        pts = [(tk, np.array(d.get(j, (0.0, 0.0, 0.0)), float)) for tk, d in keys]
        for i in range(n):
            if t[i] <= pts[0][0]:
                out[j][i] = pts[0][1]; continue
            if t[i] >= pts[-1][0]:
                out[j][i] = pts[-1][1]; continue
            k = max(q for q in range(len(pts) - 1) if pts[q][0] <= t[i]); (t0, a), (t1, b) = pts[k], pts[k + 1]
            out[j][i] = a + (b - a) * min_jerk((t[i] - t0) / (t1 - t0))
    return out


SWAG = {   # 3-second editorial strikes: neutral -> strike (1.2 s) -> hold -> settle 10 % back, ending in the hold
    "SWAG_01": [(0.0, {}), (1.2, {"pelvis": (0, 3.0, -6.0), "spine_02": (2.0, -2.0, 6.0), "head": (-2.0, 0, -8.0), "upper_arm_R": (6, -10, 0), "forearm_R": (40, 0, 0)}),
                (3.0, {"pelvis": (0, 2.7, -5.4), "spine_02": (1.8, -1.8, 5.4), "head": (-1.8, 0, -7.2), "upper_arm_R": (5.4, -9, 0), "forearm_R": (36, 0, 0)})],
    "SWAG_02": [(0.0, {}), (1.0, {"spine_02": (5.0, 0, 0), "upper_arm_L": (14, 12, 0), "forearm_L": (50, 0, 0), "upper_arm_R": (14, -12, 0), "forearm_R": (50, 0, 0)}),
                (3.0, {"spine_02": (4.5, 0, 0), "upper_arm_L": (12.6, 10.8, 0), "forearm_L": (45, 0, 0), "upper_arm_R": (12.6, -10.8, 0), "forearm_R": (45, 0, 0)})],
    "SWAG_03": [(0.0, {}), (1.4, {"pelvis": (0, -4.0, 8.0), "spine_03": (0, 2.0, -5.0), "neck": (0, 0, 10.0), "thigh_R": (-6, 3, 0), "shin_R": (10, 0, 0)}),
                (3.0, {"pelvis": (0, -3.6, 7.2), "spine_03": (0, 1.8, -4.5), "neck": (0, 0, 9.0), "thigh_R": (-5.4, 2.7, 0), "shin_R": (9, 0, 0)})],
    "SWAG_04": [(0.0, {}), (1.1, {"spine_01": (-2, 0, 0), "head": (3, 0, 0), "upper_arm_L": (-8, 6, 0), "hand_L": (0, 0, 10)}),
                (3.0, {"spine_01": (-1.8, 0, 0), "head": (2.7, 0, 0), "upper_arm_L": (-7.2, 5.4, 0), "hand_L": (0, 0, 9)})],
}


def swag(name, fps=24.0):
    return keyframe_track(SWAG[name], fps, 3.0)


def emote_groove(fps=24.0, duration=10.0, bpm=114.0):
    """EMOTE_01 '9ja Fit-Check Groove': 1.9 Hz (114 BPM) knee-led bounce with shoulder and hip counter-sway; feet stay planted (the
    bounce is knee flexion + pelvis drop, never a foot translation). Smooth onset/offset over one beat (minimum-jerk envelope)."""
    n = int(round(duration * fps)); t = np.arange(n) / fps; f = bpm / 60.0; beat = 1.0 / f
    env = np.minimum(min_jerk(t / beat), min_jerk((duration - t) / beat))
    bounce = env * (0.5 - 0.5 * np.cos(2 * math.pi * f * t))                 # 0..1 per beat
    sway = env * np.sin(math.pi * f * t)                                       # half rate: side to side every two beats
    z = np.zeros(n)
    tracks = {"pelvis": np.c_[z, 2.5 * sway, 3.0 * sway], "thigh_L": np.c_[-9.0 * bounce, z, z], "thigh_R": np.c_[-9.0 * bounce, z, z],
              "shin_L": np.c_[16.0 * bounce, z, z], "shin_R": np.c_[16.0 * bounce, z, z], "foot_L": np.c_[-7.0 * bounce, z, z], "foot_R": np.c_[-7.0 * bounce, z, z],
              "spine_02": np.c_[2.0 * bounce, -2.0 * sway, -4.0 * sway], "head": np.c_[-1.5 * bounce, z, 2.0 * sway],
              "upper_arm_L": np.c_[8.0 * bounce, 4.0 * env, z], "upper_arm_R": np.c_[8.0 * bounce, -4.0 * env, z],
              "forearm_L": np.c_[20.0 + 10.0 * bounce, z, z] * env[:, None], "forearm_R": np.c_[20.0 + 10.0 * bounce, z, z] * env[:, None]}
    return tracks, {"bpm": bpm, "hz": round(f, 4), "frames": n, "fps": fps}


def micro_action_collar_settle(fps=9.6, frames=24):
    """24-frame (2.5 s at 9.6 fps) seamless loop for the fitting page: shoulders roll back, chin lifts, a half-turn to camera, release.
    Loops because the last key equals the first (frame 24 == frame 0)."""
    dur = frames / fps
    keys = [(0.0, {}), (0.30 * dur, {"spine_03": (-1.6, 0, 0), "head": (-2.2, 0, 1.5), "upper_arm_L": (-3.0, 1.5, 0), "upper_arm_R": (-3.0, -1.5, 0)}),
            (0.60 * dur, {"spine_03": (-0.8, 0, 2.0), "spine_02": (0, 0, 2.0), "head": (-1.0, 0, 4.0), "upper_arm_L": (-1.5, 0.8, 0), "upper_arm_R": (-1.5, -0.8, 0)}),
            (dur, {})]
    return keyframe_track(keys, fps, dur)


def jerk_ratio(x, fps):
    """peak / median of |d^3 x / dt^3| (per channel), the S9-style smoothness measure."""
    j = np.abs(np.diff(np.asarray(x, float), n=3, axis=0)) * fps ** 3; med = np.median(j, axis=0)
    return np.where(med > 1e-12, j.max(axis=0) / np.where(med > 1e-12, med, 1.0), 0.0)
