"""Unit tests for runtime_v3/pose_space_engine.py (independent of Blender; NumPy + SciPy).   python -I tests/test_pose_space_engine.py"""
import json, math, sys, time, unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "runtime_v3")); sys.path.insert(0, str(ROOT / "pipeline"))
import pose_space_engine as PSE          # noqa: E402
import canon_geometry_kernels as K       # noqa: E402


def qaxis(axis, deg):
    a = np.asarray(axis, float); a /= np.linalg.norm(a); t = math.radians(deg) / 2
    return np.r_[math.cos(t), math.sin(t) * a]


def grid(n=12, h=0.004):
    xs, ys = np.meshgrid(np.arange(n) * h, np.arange(n) * h); V = np.c_[xs.ravel(), ys.ravel(), np.zeros(n * n)]
    F = []
    for j in range(n - 1):
        for i in range(n - 1):
            a = j * n + i; F += [[a, a + 1, a + n + 1], [a, a + n + 1, a + n]]
    return V, np.array(F)


class PoseSpaceEngine(unittest.TestCase):
    def test_geodesic_float32_noise_and_double_cover(self):
        rng = np.random.default_rng(0)
        for _ in range(50):
            q = rng.normal(size=4); q /= np.linalg.norm(q); q32 = q.astype(np.float32).astype(float) * (1 + 3e-7)
            self.assertLess(float(PSE.quat_geodesic(q, q32)), 2e-6)          # acos form gave ~1e-3 rad here
            self.assertLess(float(PSE.quat_geodesic(q, -q)), 1e-12)
        self.assertAlmostEqual(float(PSE.quat_geodesic(qaxis([0, 0, 1], 0), qaxis([0, 0, 1], 90))), math.pi / 2, places=12)

    def test_rbf_is_cardinal_and_bind_anchored(self):
        rng = np.random.default_rng(1); S, J = 9, 3
        X = np.stack([np.stack([qaxis(rng.normal(size=3), rng.uniform(5, 80)) for _ in range(J)]) for _ in range(S)])
        X[0] = np.tile([1.0, 0, 0, 0], (J, 1))                               # sample 0 = bind
        T = np.eye(S)[:, 1:]                                                 # one key per non-bind sample, bind target 0
        rbf = PSE.QuatRBF(X, T)
        W = rbf(X)
        self.assertLess(np.abs(W - T).max(), 1e-6)
        self.assertEqual(float(np.abs(rbf(X[0])).max()), 0.0)

    def test_duplicate_samples_merge_and_bind_wins_conflicts(self):
        q = np.tile([1.0, 0, 0, 0], (2, 1)); p = np.stack([qaxis([1, 0, 0], 30), qaxis([0, 1, 0], 10)])
        X = np.stack([q, p, p.copy()]); T = np.array([[0.0], [1.0], [1.0]])
        rbf = PSE.QuatRBF(X, T); self.assertEqual(len(rbf.X), 2); self.assertEqual(rbf.conflicts, [])
        rbf2 = PSE.QuatRBF(np.stack([q, q.copy(), p]), np.array([[0.0], [1.0], [0.0]]))   # a pose identical to bind asks for weight 1
        self.assertEqual(rbf2.conflicts, [[0, 1]]); self.assertEqual(float(rbf2(q)[0]), 0.0)

    def test_unskin_is_exact_inverse(self):
        rng = np.random.default_rng(2); n = 500
        R = np.stack([np.linalg.qr(rng.normal(size=(3, 3)))[0] for _ in range(n)]); M = 0.6 * R + 0.4 * np.eye(3)[None]
        c = rng.normal(size=(n, 3)) * 1e-3
        d, singular = PSE.unskin(M, c)
        self.assertLess(np.abs(np.einsum("vij,vj->vi", M, d) - c).max(), 1e-12); self.assertEqual(singular, 0)

    def test_graph_distance_multi_source_is_min_of_single(self):
        V, F = grid(14)
        src = [0, 13, 14 * 14 - 1]
        multi = PSE.graph_distance(V, F, src)
        single = np.min([PSE.graph_distance(V, F, [s]) for s in src], axis=0)
        self.assertLess(np.abs(multi - single).max(), 1e-15)
        self.assertEqual(float(multi[src].max()), 0.0)                         # every seed reads 0 (MC-50 heat method did not)

    def test_repair_opens_collapsed_triangle_and_stays_local(self):
        V, F = grid(16, 0.004); X = V.copy(); n = 16
        c = 8 * n + 8; X[c] = X[c + 1] * 0.92 + X[c] * 0.08                      # squash one vertex toward its neighbour
        M = np.tile(np.eye(3), (len(V), 1, 1))
        a0 = PSE.tri_areas(V, F); r0 = PSE.area_ratio(V, X, F, a0)
        self.assertGreater(int((r0 < 0.25).sum()), 0)
        d, st = PSE.arap_corrective(K, V, X, F, M, radius=0.012, arap_iters=0, a0=a0)
        r1 = PSE.area_ratio(V, X + d, F, a0)
        self.assertEqual(int((r1 < 0.25).sum()), 0); self.assertEqual(st["G5_after"], 0)
        far = PSE.graph_distance(V, F, np.unique(F[r0 < 0.30].ravel())) >= 0.012
        self.assertEqual(float(np.abs(d[far]).max()), 0.0)                       # nothing outside the region moves

    def test_model_json_roundtrip(self):
        rng = np.random.default_rng(3); S = 5
        X = np.stack([np.stack([qaxis(rng.normal(size=3), rng.uniform(5, 60))]) for _ in range(S)])
        fam = {"axilla_L": {"joints": ["upperarm01.L"], "keys": [f"k{i}" for i in range(S - 1)], "rbf": PSE.QuatRBF(X, np.eye(S)[:, 1:])}}
        m = PSE.PSDModel(fam); m2 = PSE.PSDModel.from_json(m.to_json())
        for s in range(S):
            lq = {"upperarm01.L": X[s, 0]}
            a, b = m.key_weights(lq), m2.key_weights(lq)
            self.assertLess(max(abs(a[k] - b[k]) for k in a), 1e-12)
        self.assertEqual(json.loads(m.to_json())["schema"], "maximus.v2.psd/1")


if __name__ == "__main__":
    t0 = time.time(); res = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(PoseSpaceEngine))
    print(f"{res.testsRun - len(res.failures) - len(res.errors)} of {res.testsRun} tests passed in {time.time() - t0:.2f}s")
    sys.exit(0 if res.wasSuccessful() else 1)
