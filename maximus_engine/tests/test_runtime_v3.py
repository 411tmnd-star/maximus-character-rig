"""Unit tests for runtime_v3 modules 2-7 (no Blender).   python -I tests/test_runtime_v3.py"""
import importlib.util, json, math, sys, tempfile, time, unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "runtime_v3")); sys.path.insert(0, str(ROOT / "pipeline"))
import canon_geometry_kernels as K          # noqa: E402
_gt_spec = importlib.util.spec_from_file_location("runtime_v3_garment_transport", ROOT / "runtime_v3" / "garment_transport.py")
GT = importlib.util.module_from_spec(_gt_spec); _gt_spec.loader.exec_module(GT)  # noqa: E402
import fabric_dynamics as FD                # noqa: E402
import motion_emote_engine as ME            # noqa: E402
import camera_director as CD                # noqa: E402
import lighting_director as LD              # noqa: E402
import presentation_engine as PE            # noqa: E402


def sphere(n=16, r=0.1):
    th = np.linspace(0.15, math.pi - 0.15, n); ph = np.linspace(0, 2 * math.pi, 2 * n, endpoint=False)
    V = [[0, 0, r], [0, 0, -r]] + [[r * math.sin(t) * math.cos(p), r * math.sin(t) * math.sin(p), r * math.cos(t)] for t in th for p in ph]
    V = np.array(V); m = 2 * n; F = []
    for i in range(n - 1):
        for j in range(m):
            a = 2 + i * m + j; b = 2 + i * m + (j + 1) % m; c = a + m; d = b + m; F += [[a, b, d], [a, d, c]]
    for j in range(m):
        F += [[0, 2 + (j + 1) % m, 2 + j], [1, 2 + (n - 1) * m + j, 2 + (n - 1) * m + (j + 1) % m]]
    F = np.array(F); c = V[F].mean(1); nrm = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    F[np.einsum("ij,ij->i", nrm, c) < 0] = F[np.einsum("ij,ij->i", nrm, c) < 0][:, ::-1]
    return V, F


class RuntimeV3(unittest.TestCase):
    def test_transport_roundtrip_and_rigid_invariance(self):
        V, F = sphere(); G = V[2:] * 1.06                                     # a shell 6 mm outside a 0.1 m sphere
        b = GT.bind(K, G, V, F)
        self.assertLess(np.abs(GT.transport(K, b, V, F) - G).max(), 1e-12)
        R = np.linalg.qr(np.random.default_rng(0).normal(size=(3, 3)))[0]; R *= np.sign(np.linalg.det(R)); t = np.array([0.3, -0.2, 0.9])
        G2 = GT.transport(K, b, V @ R.T + t, F)
        self.assertLess(np.abs(G2 - (G @ R.T + t)).max(), 1e-12)
        c = GT.clearance(K, G, V, F, device="cpu"); self.assertEqual(c["inside"], 0); self.assertGreater(c["min_signed_mm"], 3.0)

    def test_transport_clearance_mode_keeps_normal_gap(self):
        V, F = sphere(); G = V[2:] * 1.05; b = GT.bind(K, G, V, F)
        Vs = V * 0.8                                                           # body shrinks 20 %
        gap_paper = np.linalg.norm(GT.transport(K, b, Vs, F, "paper"), axis=1) - 0.08
        gap_clear = np.linalg.norm(GT.transport(K, b, Vs, F, "clearance"), axis=1) - 0.08
        self.assertLess(abs(np.median(gap_clear) - np.median(b["normal_m"])), 3e-4)
        self.assertLess(np.median(gap_paper), np.median(gap_clear))

    def test_xpbd_settles_and_trims_are_rigid(self):
        V, F = sphere(10); sim = FD.XPBDInertia(V, F)
        for _ in range(60):
            x, st = sim.step(V)
        self.assertLess(st["delta_inertia_max_mm"], 0.05)
        trims = V[:5] + np.array([0, 0, 0.002]); b = FD.bind_trims(trims, V, F)
        R = np.linalg.qr(np.random.default_rng(1).normal(size=(3, 3)))[0]; R *= np.sign(np.linalg.det(R))
        self.assertLess(np.abs(FD.transport_trims(b, V @ R.T, F) - trims @ R.T).max(), 1e-12)

    def test_min_jerk_and_loops(self):
        self.assertEqual(float(ME.min_jerk(0.0)), 0.0); self.assertEqual(float(ME.min_jerk(1.0)), 1.0)
        tr = ME.micro_action_collar_settle()
        for j, x in tr.items():
            self.assertEqual(x.shape[0], 24); self.assertLess(np.abs(x[0]).max(), 1e-12)
        for name in ME.SWAG:
            self.assertEqual(next(iter(ME.swag(name).values())).shape[0], 72)
        tracks, info = ME.emote_groove(); self.assertAlmostEqual(info["hz"], 1.9, places=6)
        self.assertNotIn("clavicle_L", tracks); self.assertNotIn("clavicle_R", tracks)

    def test_camera_solve_hits_top_margin(self):
        rng = np.random.default_rng(2); X = rng.normal(size=(5000, 3)) * [0.15, 0.1, 0.3] + [0, 0, 1.2]; X[:, 2] = np.clip(X[:, 2], 0, 1.75)
        s = CD.solve(X); self.assertAlmostEqual(s["predicted_top_margin"], CD.TOP_MARGIN, places=6)
        a = np.zeros((1000, 800)); a[71:, 300:500] = 1.0; self.assertTrue(CD.check(a)["in_band_6_8_percent"])

    def test_lighting_puts_dark_and_light_garments_in_band(self):
        for hexc in ("#4A1420", "#F0EEE8", "#1C1C1C"):
            r = LD.solve(hexc)
            self.assertTrue(105 <= r["predicted_codes"]["garment_torso"] <= 155, (hexc, r["predicted_codes"]))
            self.assertLessEqual(r["predicted_codes"]["rim_highlight"], 248); self.assertLessEqual(r["predicted_codes"]["mannequin_torso"], 245)
        for hexc in ("#4A1420", "#F0EEE8"):                    # separable merch: >= 22 codes from the chosen finish
            r = LD.solve(hexc); self.assertGreaterEqual(abs(r["predicted_codes"]["garment_torso"] - r["predicted_codes"]["mannequin_torso"]), 22, (hexc, r))
        self.assertEqual(LD.solve("#F0EEE8")["finish"], "M1")
        self.assertGreater(LD.solve("#1C1C1C")["score"], 1.0)   # near-black: key/fill cannot separate it from any finish; reported, not hidden

    def test_lighting_multilayer_reports_dynamic_range(self):
        # white singlet under a dark jacket on M1: 6.9 stops of albedo > 5.9 usable stops -> infeasible, said so, highlight held near 245
        r = LD.solve_multilayer(["#F2F2EE", "#141518"], "#18181B")
        self.assertFalse(r["dynamic_range"]["feasible"])
        self.assertTrue(r["warnings"])
        self.assertLessEqual(r["predicted_codes"]["layers_torso_unclamped"][0], 250.0)
        # two mid-tone layers fit the window: no clip, no crush
        r2 = LD.solve_multilayer(["#8A8F99", "#4A1420"], "#F2EFE9")
        self.assertTrue(r2["dynamic_range"]["feasible"])
        self.assertTrue(r2["no_highlight_clip"] and r2["no_shadow_crush"], r2["predicted_codes"])
        # legacy single-garment solve is unchanged
        self.assertEqual(LD.solve("#F0EEE8")["finish"], "M1")

    def test_presentation_families_are_disjoint(self):
        H, W = 40, 30; rgba = np.random.default_rng(3).random((H, W, 4)); rgba[..., 3] = 1
        ids = np.zeros((H, W), int); ids[10:30, 5:25] = 1; n = np.dstack([np.zeros((H, W)), np.zeros((H, W)), np.ones((H, W))])
        with tempfile.TemporaryDirectory() as d:
            man = PE.pack("t", d, rgba, ids, n, np.full((H, W), 3.4), np.full((H, W), 0.7), np.ones((H, W)), {0: "background", 1: "polo"})
            self.assertTrue(PE.validate(man)["separated"]); self.assertEqual(len(list(Path(d).glob("t_*.webp"))), 4)


if __name__ == "__main__":
    t0 = time.time(); res = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(RuntimeV3))
    print(f"{res.testsRun - len(res.failures) - len(res.errors)} of {res.testsRun} tests passed in {time.time() - t0:.2f}s")
    sys.exit(0 if res.wasSuccessful() else 1)
