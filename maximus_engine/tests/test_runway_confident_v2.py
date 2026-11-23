"""Unit tests for Track A: Maximus Confident Runway Walk (RUNWAY_CONFIDENT_V2) and Editorial Camera."""
from __future__ import annotations

import json
import math
import sys
import unittest
from pathlib import Path

import numpy as np

ENGINE_ROOT = Path(__file__).resolve().parents[1]
COLAB_STAGE = ENGINE_ROOT / "colab_stage"
HARNESS_DIR = ENGINE_ROOT.parent / "better_style_engine" / "02_engine_3d" / "avatar_engine" / "harness"

if str(COLAB_STAGE) not in sys.path:
    sys.path.insert(0, str(COLAB_STAGE))

import m5_gait as G  # noqa: E402
import m8_motion_reels as M8  # noqa: E402


class TestRunwayConfidentV2(unittest.TestCase):
    def test_baseline_m5_preserved(self):
        plan_default = G.GaitPlan()
        plan_base = G.GaitPlan.from_preset("BASELINE_M5")
        self.assertEqual(plan_default.preset, "BASELINE_M5")
        self.assertAlmostEqual(plan_default.T, 2.528, places=6)
        self.assertAlmostEqual(plan_default.width, 0.053, places=6)
        self.assertAlmostEqual(plan_default.v, plan_base.v, places=9)
        self.assertAlmostEqual(plan_default.v, 0.4031, places=3)

    def test_runway_confident_v2_cadence_velocity_and_step_width(self):
        plan = G.GaitPlan.from_preset("RUNWAY_CONFIDENT_V2")
        self.assertEqual(plan.preset, "RUNWAY_CONFIDENT_V2")
        # Stride period T in [1.08, 1.14] s (~106-111 BPM)
        self.assertGreaterEqual(plan.T, 1.08)
        self.assertLessEqual(plan.T, 1.14)
        self.assertGreaterEqual(plan.cadence_steps_per_min, 105.0)
        self.assertLessEqual(plan.cadence_steps_per_min, 112.0)
        # Forward velocity in [1.18, 1.28] m/s (~12.5 m over 240-250 frames @ 24fps)
        self.assertGreaterEqual(plan.v, 1.18)
        self.assertLessEqual(plan.v, 1.28)
        dist_250 = plan.v * (250.0 / 24.0)
        self.assertGreaterEqual(dist_250, 12.0)
        self.assertLessEqual(dist_250, 13.4)
        # Step width in [1.0, 2.2] cm ([0.010, 0.022] m)
        width_cm = plan.width * 100.0
        self.assertGreaterEqual(width_cm, 1.0)
        self.assertLessEqual(width_cm, 2.2)

    def test_stance_swing_heel_strike_and_toe_off(self):
        plan = G.GaitPlan.from_preset("RUNWAY_CONFIDENT_V2")
        # 60% stance / 40% swing
        self.assertAlmostEqual(plan.stance, 0.60, places=6)
        # 2-4 deg near-straight heel-strike extension, +12 deg dorsiflexion, -18 deg plantarflexion
        self.assertGreaterEqual(plan.heel_strike_knee_flex_deg, 2.0)
        self.assertLessEqual(plan.heel_strike_knee_flex_deg, 4.0)
        self.assertAlmostEqual(plan.heel_strike_dorsiflex_deg, 12.0, places=4)
        self.assertAlmostEqual(plan.toe_off_deg, 18.0, places=4)

        # Sample across one full stride cycle
        ts = np.linspace(0.0, plan.T, 240, endpoint=False)
        feet = [plan.foot("L", float(t)) for t in ts]
        stance_frac = sum(1 for f in feet if f["mode"] == "stance") / len(feet)
        self.assertAlmostEqual(stance_frac, 0.60, delta=0.01)
        pitches_deg = [math.degrees(f["pitch"]) for f in feet]
        self.assertAlmostEqual(min(pitches_deg), -18.0, delta=0.2)
        self.assertAlmostEqual(max(pitches_deg), 12.0, delta=0.5)

    def test_pelvic_torque_posture_head_damping_and_arm_pendulum(self):
        plan = G.GaitPlan.from_preset("RUNWAY_CONFIDENT_V2")
        self.assertAlmostEqual(plan.pelvis_yaw_deg, 6.5, places=4)
        self.assertAlmostEqual(plan.pelvis_tilt_deg, 3.0, places=4)
        self.assertAlmostEqual(plan.spine_counter_yaw_deg, 5.5, places=4)
        self.assertTrue(plan.clavicle_lock)
        self.assertAlmostEqual(plan.chest_lift_deg, 3.5, places=4)
        self.assertAlmostEqual(plan.chin_pitch_deg, 0.0, places=4)
        self.assertAlmostEqual(plan.head_stabilization, 0.70, places=4)

        # Verify 70% head stabilization damping calculation
        hs = plan.head_stabilized_angles(math.radians(6.5), 0.024)
        self.assertAlmostEqual(hs["damping_factor"], 0.70, places=6)
        self.assertAlmostEqual(math.degrees(hs["residual_yaw_rad"]), 6.5 * 0.30, places=5)
        self.assertAlmostEqual(hs["residual_bob_m"], 0.024 * 0.30, places=6)
        self.assertAlmostEqual(hs["chin_pitch_rad"], 0.0, places=9)

        # Verify asymmetric arm pendulum (-20 deg backswing / +13 deg forward swing) & 10-18 deg elbow flexion
        phs = np.linspace(0.0, 2.0 * math.pi, 361)
        arms = [plan.arm_pendulum_deg(float(p)) for p in phs]
        self.assertAlmostEqual(max(arms), 13.0, places=4)
        self.assertAlmostEqual(min(arms), -20.0, places=4)
        elbows = [plan.elbow_flex_deg(float(p)) for p in phs]
        self.assertAlmostEqual(min(elbows), 10.0, places=3)
        self.assertAlmostEqual(max(elbows), 18.0, places=3)
        self.assertGreaterEqual(plan.min_thigh_clearance_mm, 15.0)

    def test_authoritative_editorial_camera_z092_and_headroom(self):
        # Synthetic standing human bounding points from Z=0.00m (sole) to Z=1.7313m (crown)
        rng = np.random.default_rng(42)
        X = rng.uniform([-0.22, -0.14, 0.05], [0.22, 0.14, 1.68], size=(400, 3))
        X[0] = [0.0, 0.0, 0.0]       # exact sole contact Z = 0.00 m
        X[1] = [0.0, 0.0, 1.7313]    # exact crown Z = 1.7313 m
        anchor = np.array([0.0, 0.0], dtype=np.float64)

        for u in np.linspace(0.0, 1.0, 11):
            az = -14.0 + 30.0 * float(u)
            lens = 85.0 + 15.0 * float(u)
            cam_loc, cam_tgt, dist_m, tm, bf = M8.solve_walk_editorial_camera(
                X, anchor, view_deg=az, lens_mm=lens, cam_z=0.92, top_margin=0.070, sole_frac=0.905
            )
            self.assertAlmostEqual(float(cam_loc[2]), 0.92, places=6)
            self.assertGreaterEqual(tm * 100.0, 6.0)
            self.assertLessEqual(tm * 100.0, 8.0)
            self.assertAlmostEqual(tm * 100.0, 7.0, places=2)
            self.assertGreater(dist_m, 3.0)

    def test_harness_gate_limits_registered(self):
        lim_path = HARNESS_DIR / "gate_limits.json"
        self.assertTrue(lim_path.exists())
        lim = json.loads(lim_path.read_text(encoding="utf-8"))
        self.assertIn("presets", lim)
        self.assertIn("RUNWAY_CONFIDENT_V2", lim["presets"])
        rc = lim["presets"]["RUNWAY_CONFIDENT_V2"]
        self.assertTrue(rc["clavicle_lock"])
        self.assertEqual(rc["max_G5"], 0)
        self.assertEqual(rc["p01_gates"]["S8_step_line_cm"]["limit"], [1.0, 2.2])
        self.assertEqual(rc["targets_not_gates"]["stride_period_s"], [1.08, 1.14])


if __name__ == "__main__":
    unittest.main()
