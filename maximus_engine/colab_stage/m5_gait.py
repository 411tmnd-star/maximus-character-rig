"""m5_gait.py - runway gait plan + NumPy forward kinematics / leg IK for the 152-bone Maximus rig (no Blender; unit-testable).

Designed against the REAL harness (better_style_engine/02_engine_3d/avatar_engine/harness/gates.py), not against a copy of its limits:
  * floor / slide: the harness takes the LOWER foot each frame as the stance foot and measures its vertex-centroid slide (<= 1 mm/frame) and
    floor height (+-1 mm). So the foot that is lower must be still: feet land FLAT (no heel-strike roll), the trailing foot rolls onto its
    toes 0.4 mm above the floor during double support (still "in contact" for S1, which uses <= 2 mm, but never the lower foot), and heel
    rise during single support is capped near 1 deg/frame.
  * S1 stance share 0.60 per foot; S8 step line = left ankle x - right ankle x at successive opposite strikes (target 5.0 cm, limit 3-6).
  * S3 delays are whole frames at 24 fps: chest<-pelvis 3, shoulder<-chest 3, elbow<-shoulder 6, wrist<-elbow 3, fingers<-wrist 2 frames
    (0.048 / 0.048 / 0.096 / 0.048 / 0.032 of the 62.4-frame stride). head<-chest has a [0.05, 0.05] window: whole-frame lags give 0.0481
    or 0.0641 at a 2.6 s stride, so that relation cannot pass at this cadence; the limit is not touched.
  * S2: shoulder swing carries a 30 % second harmonic (energy ~0.08 >= 0.05); hip and knee get their harmonics from the IK gait.
  * S5: every gated bone moves > 0.5 deg, including both clavicles (the clavicle lock is for static poses) and all ten first toe joints.
  * S7: head orientation is stabilised to within ~1 deg of bind in the body frame every frame.
Conventions (posing.py): model faces -Y, +X = model's left, +Z up; rig world = identity, so armature space = world space.
"""
import math

import numpy as np

FWD = np.array([0.0, -1.0, 0.0]); LAT = np.array([1.0, 0.0, 0.0]); UP = np.array([0.0, 0.0, 1.0])


def rot(axis, ang):
    a = np.asarray(axis, float); a = a / np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * K @ K


def align_rot(v_from, v_to):
    a = v_from / np.linalg.norm(v_from); b = v_to / np.linalg.norm(v_to); ax = np.cross(a, b); s = np.linalg.norm(ax)
    if s < 1e-12:
        return np.eye(3)
    return rot(ax, math.atan2(s, float(a @ b)))


def quat_to_mat(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def mat_to_quat(R):
    t = np.trace(R)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2; q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2; q = [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2; q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2; q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    q = np.array(q); q /= np.linalg.norm(q)
    return -q if q[0] < 0 else q


class Skeleton:
    """Pose matrices from local bases: pose[b] = pose[parent] @ rest[parent]^-1 @ rest[b] @ basis[b] (Blender's pose evaluation without
    constraints). rest = bone.matrix_local (armature space). World edits about a bone head re-derive that bone's basis."""

    def __init__(self, names, parents, rest):
        self.names = list(names); self.idx = {n: i for i, n in enumerate(self.names)}; self.par = list(parents); self.rest = np.asarray(rest, float)
        self.rel = np.stack([np.linalg.inv(self.rest[p]) @ self.rest[i] if p >= 0 else self.rest[i] for i, p in enumerate(self.par)])
        self.order = []; seen = set()
        while len(self.order) < len(self.names):
            for i, p in enumerate(self.par):
                if i not in seen and (p < 0 or p in seen):
                    self.order.append(i); seen.add(i)
        self.basis = np.tile(np.eye(4), (len(self.names), 1, 1)); self.pose = None; self.fk()

    def set_basis_qt(self, name, q, t=(0, 0, 0)):
        M = np.eye(4); M[:3, :3] = quat_to_mat(np.asarray(q, float) / np.linalg.norm(q)); M[:3, 3] = t; self.basis[self.idx[name]] = M

    def fk(self):
        P = np.zeros_like(self.basis)
        for i in self.order:
            p = self.par[i]; P[i] = (P[p] @ self.rel[i] @ self.basis[i]) if p >= 0 else (self.rel[i] @ self.basis[i])
        self.pose = P; return P

    def head(self, name):
        return self.pose[self.idx[name]][:3, 3].copy()

    def tail(self, name, length):
        M = self.pose[self.idx[name]]; return M[:3, 3] + M[:3, 1] * length

    def rot_world(self, name, R, about=None):
        """Rotate bone `name` (and so its children) by world rotation R about `about` (default: its own head)."""
        i = self.idx[name]; P = self.pose[i]; h = P[:3, 3] if about is None else np.asarray(about, float)
        T = np.eye(4); T[:3, :3] = R; T[:3, 3] = h - R @ h
        newP = T @ P; p = self.par[i]
        parent_rel = (self.pose[p] @ self.rel[i]) if p >= 0 else self.rel[i]
        self.basis[i] = np.linalg.inv(parent_rel) @ newP; self.fk()

    def translate_world(self, name, d):
        i = self.idx[name]; newP = self.pose[i].copy(); newP[:3, 3] += d; p = self.par[i]
        parent_rel = (self.pose[p] @ self.rel[i]) if p >= 0 else self.rel[i]
        self.basis[i] = np.linalg.inv(parent_rel) @ newP; self.fk()

    def world_rot(self, name):
        return self.pose[self.idx[name]][:3, :3].copy()

    def local_quat(self, name):
        return mat_to_quat(self.basis[self.idx[name]][:3, :3])


def smoothstep5(s):
    s = np.clip(s, 0.0, 1.0); return s * s * s * (10 - 15 * s + 6 * s * s)


def ease(s):
    s = np.clip(s, 0.0, 1.0); return 0.5 - 0.5 * np.cos(math.pi * s)


GAIT_PRESETS = {
    "BASELINE_M5": {
        "preset": "BASELINE_M5",
        "T": 2.528,
        "fps": 24.0,
        "n_frames": 250,
        "step": 0.50,
        "width": 0.053,
        "stance": 0.585,
        "asym": 0.038,
        "toe_out_deg": 6.0,
        "heel_rise_ss_deg": 4.0,
        "toe_off_deg": 32.0,
        "heel_strike_dorsiflex_deg": 6.0,
        "heel_strike_knee_flex_deg": 5.0,
        "lift": 0.030,
        "bulge": 0.030,
        "contact_lift": 0.0004,
        "lift_asym": 0.05,
        "sway_m": 0.0165,
        "pelvis_tilt_deg": 3.5,
        "pelvis_yaw_deg": 4.0,
        "spine_counter_yaw_deg": 4.6,
        "chest_world_yaw_deg": 0.6,
        "chest_lift_deg": 0.0,
        "chin_pitch_deg": 0.0,
        "head_stabilization": 0.65,
        "clavicle_lock": False,
        "arm_swing_deg": 9.0,
        "arm_backswing_deg": -11.7,
        "arm_fwdswing_deg": 11.7,
        "elbow_flex_range_deg": (10.0, 14.0),
        "arm_guard_deg": 9.0,
        "min_thigh_clearance_mm": 15.0,
        "swing_knee_drive_exp": 1.0,
    },
    "RUNWAY_CONFIDENT_V2": {
        "preset": "RUNWAY_CONFIDENT_V2",
        "T": 1.125,
        "fps": 24.0,
        "n_frames": 250,
        "step": 0.678,
        "width": 0.0165,
        "stance": 0.60,
        "asym": 0.032,
        "toe_out_deg": 3.5,
        "heel_rise_ss_deg": 1.6,
        "toe_off_deg": 18.0,
        "heel_strike_dorsiflex_deg": 12.0,
        "heel_strike_knee_flex_deg": 3.0,
        "lift": 0.036,
        "bulge": 0.034,
        "contact_lift": 0.0004,
        "lift_asym": 0.035,
        "sway_m": 0.0135,
        "pelvis_tilt_deg": 3.0,
        "pelvis_yaw_deg": 6.5,
        "spine_counter_yaw_deg": 5.5,
        "chest_world_yaw_deg": 1.0,
        "chest_lift_deg": 3.5,
        "chin_pitch_deg": 0.0,
        "head_stabilization": 0.70,
        "clavicle_lock": True,
        "arm_swing_deg": 16.5,
        "arm_backswing_deg": -20.0,
        "arm_fwdswing_deg": 13.0,
        "elbow_flex_range_deg": (10.0, 18.0),
        "arm_guard_deg": 11.5,
        "min_thigh_clearance_mm": 15.0,
        "swing_knee_drive_exp": 0.78,
    },
}


class GaitPlan:
    """Footstep plan and per-frame targets. All times in seconds; phase phi in [0, 1) per foot from its own (flat) landing."""

    def __init__(self, T=None, fps=None, n_frames=None, step=None, width=None, stance=None, asym=None, toe_out_deg=None,
                 heel_rise_ss_deg=None, toe_off_deg=None, lift=None, bulge=None, contact_lift=None, lift_asym=None,
                 preset="BASELINE_M5", **extra):
        if preset not in GAIT_PRESETS:
            raise ValueError(f"Unknown gait preset: {preset!r}. Expected one of {sorted(GAIT_PRESETS)}")
        cfg = dict(GAIT_PRESETS[preset])
        overrides = {
            "T": T, "fps": fps, "n_frames": n_frames, "step": step, "width": width, "stance": stance,
            "asym": asym, "toe_out_deg": toe_out_deg, "heel_rise_ss_deg": heel_rise_ss_deg,
            "toe_off_deg": toe_off_deg, "lift": lift, "bulge": bulge, "contact_lift": contact_lift,
            "lift_asym": lift_asym,
        }
        for k, v in overrides.items():
            if v is not None:
                cfg[k] = v
        cfg.update({k: v for k, v in extra.items() if v is not None})

        self.preset = preset
        self.cfg = cfg
        self.T, self.fps, self.n = float(cfg["T"]), float(cfg["fps"]), int(cfg["n_frames"])
        base_step = float(cfg["step"])
        self.asym = float(cfg["asym"])
        self.step = {"L": base_step, "R": base_step * (1.0 + self.asym)}
        self.width = float(cfg["width"])
        self.stance = float(cfg["stance"])
        self.toe_out = math.radians(float(cfg["toe_out_deg"]))
        self.hr_ss = math.radians(float(cfg["heel_rise_ss_deg"]))
        self.toe_off = math.radians(float(cfg["toe_off_deg"]))
        self.toe_off_deg = float(cfg["toe_off_deg"])
        self.dorsiflex_strike = math.radians(float(cfg["heel_strike_dorsiflex_deg"]))
        self.heel_strike_dorsiflex_deg = float(cfg["heel_strike_dorsiflex_deg"])
        self.heel_strike_knee_flex_deg = float(cfg["heel_strike_knee_flex_deg"])
        self.lift = float(cfg["lift"])
        self.bulge = float(cfg["bulge"])
        self.contact_lift = float(cfg["contact_lift"])
        self.lift_asym = float(cfg["lift_asym"])
        self.lift_side = {"L": self.lift, "R": self.lift * (1.0 + self.lift_asym)}
        self.t0 = {"L": 0.0, "R": 0.5 * self.T}
        self.v = (self.step["L"] + self.step["R"]) / self.T
        self.cadence_steps_per_min = 120.0 / self.T
        self.sway_m = float(cfg["sway_m"])
        self.pelvis_tilt_deg = float(cfg["pelvis_tilt_deg"])
        self.pelvis_yaw_deg = float(cfg["pelvis_yaw_deg"])
        self.spine_counter_yaw_deg = float(cfg["spine_counter_yaw_deg"])
        self.chest_world_yaw_deg = float(cfg["chest_world_yaw_deg"])
        self.chest_lift_deg = float(cfg["chest_lift_deg"])
        self.chin_pitch_deg = float(cfg["chin_pitch_deg"])
        self.head_stabilization = float(cfg["head_stabilization"])
        self.clavicle_lock = bool(cfg["clavicle_lock"])
        self.arm_swing_deg = float(cfg["arm_swing_deg"])
        self.arm_backswing_deg = float(cfg["arm_backswing_deg"])
        self.arm_fwdswing_deg = float(cfg["arm_fwdswing_deg"])
        self.elbow_flex_range_deg = tuple(float(x) for x in cfg["elbow_flex_range_deg"])
        self.arm_guard_deg = float(cfg["arm_guard_deg"])
        self.min_thigh_clearance_mm = float(cfg["min_thigh_clearance_mm"])
        self.swing_knee_drive_exp = float(cfg["swing_knee_drive_exp"])

    @classmethod
    def from_preset(cls, preset="BASELINE_M5", **overrides):
        return cls(preset=preset, **overrides)

    def arm_pendulum_deg(self, phase_rad: float, amp_scale: float = 1.0) -> float:
        """Asymmetric heavy arm pendulum angle (deg, + = forward swing, - = backswing)."""
        if self.preset == "RUNWAY_CONFIDENT_V2":
            fwd = self.arm_fwdswing_deg * amp_scale
            back = self.arm_backswing_deg * amp_scale
            a1 = 0.5 * (fwd - back)
            a0 = 0.25 * (fwd + back)
            a2 = 0.25 * (fwd + back)
            return float(a0 + a1 * math.cos(phase_rad) + a2 * math.cos(2.0 * phase_rad))
        return float(self.arm_swing_deg * amp_scale * (math.cos(phase_rad) + 0.30 * math.cos(2.0 * phase_rad + 0.6)))

    def elbow_flex_deg(self, phase_rad: float, amp_scale: float = 1.0) -> float:
        """Relaxed dynamic elbow flexion angle (deg) over the stride."""
        lo, hi = self.elbow_flex_range_deg
        mid = 0.5 * (lo + hi)
        half = 0.5 * (hi - lo) * amp_scale
        return float(mid + half * math.cos(phase_rad))

    def head_stabilized_angles(self, raw_pelvis_yaw_rad: float, raw_vertical_bob_m: float) -> dict:
        """Apply active head stabilization (70% damping in RUNWAY_CONFIDENT_V2) + neutral chin pitch."""
        damp = self.head_stabilization
        residual = 1.0 - damp
        return {
            "damping_factor": damp,
            "residual_yaw_rad": float(raw_pelvis_yaw_rad * residual),
            "residual_bob_m": float(raw_vertical_bob_m * residual),
            "chin_pitch_rad": math.radians(self.chin_pitch_deg),
        }

    def phase(self, side, t):
        u = (t - self.t0[side]) / self.T; k = math.floor(u); return u - k, int(k)

    def plant_y(self, side, k):
        """Forward coordinate (along FWD) of the k-th landing of `side`. Landings alternate L0, R0, L1, R1 ...: R_k = L_k + step_R,
        L_(k+1) = R_k + step_L, so L_k = k (step_L + step_R) and R_k = L_k + step_R."""
        stride = self.step["L"] + self.step["R"]
        return k * stride if side == "L" else k * stride + self.step["R"]

    def foot(self, side, t):
        """dict(mode, phase, pitch (rad, + = toes up, - = heel up), pivot ('flat' | 'toe' | 'air'), plant_fwd (m along FWD), next_fwd,
        swing s, lower (bool: designated lower foot), z_lift (m: target height of the pivot above the floor))."""
        phi, k = self.phase(side, t); st = self.stance
        d = {"phase": phi, "k": k, "plant_fwd": self.plant_y(side, k), "next_fwd": self.plant_y(side, k + 1)}
        if phi < st:
            d["mode"] = "stance"
            flat_end = 0.45 * st / 0.6
            if phi < flat_end:                                                # flat, the designated lower foot
                d.update(pitch=0.0, pivot="flat", lower=True, z_lift=0.0)
            elif phi < 0.5:                                                   # late single support: slow heel rise, still the lower foot
                s = (phi - flat_end) / (0.5 - flat_end)
                d.update(pitch=-self.hr_ss * ease(s), pivot="toe", lower=True, z_lift=0.0)
            else:                                                             # double support: other foot is flat and lower; roll to toe-off
                s = (phi - 0.5) / (st - 0.5)
                d.update(pitch=-(self.hr_ss + (self.toe_off - self.hr_ss) * ease(s)), pivot="toe", lower=False, z_lift=self.contact_lift)
        else:
            s = (phi - st) / (1 - st)
            s_drive = float(np.clip(s, 0.0, 1.0) ** self.swing_knee_drive_exp)
            d.update(mode="swing", s=s, s_drive=s_drive, pivot="air", lower=False, z_lift=None)
            if self.preset == "RUNWAY_CONFIDENT_V2":
                # Decisive swing from -18 deg toe-off plantarflexion into +12 deg heel-strike dorsiflexion
                to_release = -self.toe_off * (1.0 - smoothstep5(min(1.0, s / 0.45)))
                df_arch = self.dorsiflex_strike * (math.sin(math.pi * s) ** 1.4)
                d["pitch"] = to_release + df_arch
            else:
                d["pitch"] = -self.toe_off * (1 - smoothstep5(s)) + self.dorsiflex_strike * math.sin(math.pi * s) ** 2
        return d

    def stance_share_design(self):
        return self.stance


def two_bone_ik(S, A, a, b, pole):
    """Knee position for hip S, ankle target A, thigh a, shin b, knee toward `pole`. Returns (K, reach_clamped)."""
    d_vec = A - S; d = float(np.linalg.norm(d_vec)); clamped = False
    lo, hi = abs(a - b) + 1e-4, a + b - 1e-4
    if d > hi or d < lo:
        clamped = True; d = min(max(d, lo), hi)
    n = d_vec / np.linalg.norm(d_vec); x = (a * a - b * b + d * d) / (2 * d); h = math.sqrt(max(a * a - x * x, 0.0))
    p = np.asarray(pole, float) - n * float(np.dot(pole, n)); p /= max(np.linalg.norm(p), 1e-12)
    return S + n * x + p * h, clamped


def periodic_lowpass(x, frames_per_period, harmonics):
    """Keep the mean and the first `harmonics` harmonics of a signal sampled over whole periods (used to smooth the emergent bob)."""
    n = len(x); t = np.arange(n); w = 2 * math.pi / frames_per_period
    cols = [np.ones(n)] + [f(k * w * t) for k in range(1, harmonics + 1) for f in (np.cos, np.sin)]
    A = np.stack(cols, 1); coef, *_ = np.linalg.lstsq(A, x, rcond=None); return A @ coef
