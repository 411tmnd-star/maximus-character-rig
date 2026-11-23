"""Fast unit tests for pipeline/canon_geometry_kernels.py (NumPy + SciPy only, < 3 s). Run:  python -I tests/test_canon_geometry_kernels.py
(or with pytest). Each kernel is checked against a case with a known analytic answer, never against itself."""
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))
import canon_geometry_kernels as K  # noqa: E402


def grid(n, h=0.05, z=None):
    """n x n flat grid in XY, counter-clockwise triangles facing +Z."""
    xs = (np.arange(n) - (n - 1) / 2) * h; X, Y = np.meshgrid(xs, xs); V = np.stack([X.ravel(), Y.ravel(), np.zeros(n * n) if z is None else z.ravel()], 1)
    F = []
    for i in range(n - 1):
        for j in range(n - 1):
            a, b, c, d = i * n + j, i * n + j + 1, (i + 1) * n + j, (i + 1) * n + j + 1; F += [[a, b, d], [a, d, c]]
    return V, np.array(F)


def octahedron():
    V = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], float)
    F = np.array([[0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4], [2, 0, 5], [1, 2, 5], [3, 1, 5], [0, 3, 5]]); return V, F   # outward normals


def cylinder(r=0.15, h=1.0, n=48, rings=11):
    th = 2 * np.pi * np.arange(n) / n; zs = np.linspace(0, h, rings)
    V = np.array([[r * math.cos(t), r * math.sin(t), z] for z in zs for t in th]); F = []
    for k in range(rings - 1):
        for i in range(n):
            a, b = k * n + i, k * n + (i + 1) % n; c, d = a + n, b + n; F += [[a, b, d], [a, d, c]]
    return V, np.array(F)


def test_cotangent_laplacian_and_mass():
    V, F = octahedron(); L, M = K.cotangent_laplacian_and_mass(V, F); Ld = L.toarray()
    assert np.allclose(Ld, Ld.T) and np.allclose(Ld.sum(1), 0, atol=1e-12)
    assert np.linalg.eigvalsh(Ld).min() > -1e-10                                  # positive semi-definite
    assert abs(M.diagonal().sum() - 4 * math.sqrt(3)) < 1e-12                     # octahedron area = 8 * (sqrt(3)/4) * 2 = 4 sqrt(3)
    Vg, Fg = grid(9); Lg, Mg = K.cotangent_laplacian_and_mass(Vg, Fg)             # a linear function is harmonic: interior rows of L x vanish
    x = Vg[:, 0]; interior = np.array([i for i in range(81) if 0 < i // 9 < 8 and 0 < i % 9 < 8])
    assert np.abs((Lg @ x)[interior]).max() < 1e-12 and abs(Mg.diagonal().sum() - 0.4 ** 2) < 1e-12


def test_heat_geodesic_distance_on_plane():
    V, F = grid(21); src = 10 * 21 + 10; d = K.heat_geodesic_distance(V, F, [src]); true = np.linalg.norm(V - V[src], axis=1)
    far = true > 0.2; assert d[src] == 0 and (np.abs(d[far] - true[far]) / true[far]).mean() < 0.08
    assert np.corrcoef(d, true)[0, 1] > 0.995


def test_geodesic_smooth_mask():
    V, F = grid(21); src = 10 * 21 + 10; m = K.geodesic_smooth_mask(V, F, [src], r_zero=0.1, r_ramp=0.2); d = K.heat_geodesic_distance(V, F, [src])
    assert m.min() >= 0 and m.max() <= 1 and m[src] == 0 and (m[d < 0.09] == 0).all() and (m[d > 0.33] > 0.99).all()
    o = np.argsort(d); assert (np.diff(m[o]) > -1e-9).all()                       # monotone in distance


def test_biharmonic_fairing_flattens_noise():
    n = 15; V, F = grid(n); rng = np.random.default_rng(3); ring = np.minimum.reduce([np.arange(n * n) // n, np.arange(n * n) % n, n - 1 - np.arange(n * n) // n, n - 1 - np.arange(n * n) % n])
    interior = np.flatnonzero(ring >= 2); pinned = np.flatnonzero(ring < 2); Vn = V.copy(); Vn[interior, 2] = rng.normal(0, 0.01, len(interior))
    out = K.biharmonic_surface_fairing(Vn, F, interior, pinned)
    assert np.abs(out[interior, 2]).max() < 1e-9 and np.array_equal(out[pinned], Vn[pinned])    # boundary is flat, so the biharmonic solution is flat


def test_arap_recovers_rigid_motion():
    n = 13; V, F = grid(n); th = 0.7; R = np.array([[math.cos(th), 0, math.sin(th)], [0, 1, 0], [-math.sin(th), 0, math.cos(th)]]); target = V @ R.T + [0.2, -0.1, 0.5]
    ring = np.minimum.reduce([np.arange(n * n) // n, np.arange(n * n) % n, n - 1 - np.arange(n * n) // n, n - 1 - np.arange(n * n) % n])
    act = np.flatnonzero(ring >= 2); pinned = np.flatnonzero(ring < 2); rng = np.random.default_rng(5); start = target.copy(); start[act] += rng.normal(0, 0.03, (len(act), 3))
    out, en = K.arap_pose_relaxation(V, start, F, act, pinned, n_iters=8)
    e0 = np.linalg.norm(start[act] - target[act], axis=1).max(); e1 = np.linalg.norm(out[act] - target[act], axis=1).max()
    assert e1 < 0.05 * e0 and en[-1] < 0.01 * en[0] and np.array_equal(out[pinned], start[pinned])


def test_triangle_frames_bind_and_transform():
    V, F = octahedron(); fr = K.compute_triangle_frames(V, F); R = fr["R"]
    assert np.allclose(np.einsum("nij,nkj->nik", R, R), np.eye(3), atol=1e-12) and np.allclose(np.linalg.det(R), 1.0)
    n0 = np.cross(V[F[0, 1]] - V[F[0, 0]], V[F[0, 2]] - V[F[0, 0]]); n0 /= np.linalg.norm(n0)
    assert np.allclose(R[0][:, 0], n0) and np.allclose(R[0][:, 1], (V[F[0, 1]] - V[F[0, 0]]) / np.linalg.norm(V[F[0, 1]] - V[F[0, 0]]))   # paper order: n, edge, n x edge
    B = np.linalg.norm(V[F[0, 1]] - V[F[0, 0]]); Hh = np.linalg.norm(np.cross(V[F[0, 1]] - V[F[0, 0]], V[F[0, 2]] - V[F[0, 0]])) / B; assert abs(fr["k"][0] - 0.5 * (B + Hh)) < 1e-12
    rng = np.random.default_rng(1); fid = rng.integers(0, len(F), 40); bary = rng.dirichlet([1, 1, 1], 40); pts = np.einsum("na,nai->ni", bary, V[F[fid]]) + 0.05 * fr["R"][fid][:, :, 0]
    rots = np.stack([np.eye(3)] * 40); sc = np.full((40, 3), 0.01)
    local = K.bind_elements(pts, fid, fr, rots, sc, V=V, barycentric=bary)
    assert np.allclose(local["position"][:, 1:], 0, atol=1e-12)                    # on the surface in-plane offset is zero, offset along the normal remains
    th = 0.9; Rz = np.array([[math.cos(th), -math.sin(th), 0], [math.sin(th), math.cos(th), 0], [0, 0, 1]]); s = 1.7; V2 = s * (V @ Rz.T) + [0.3, 0.2, -0.4]
    fr2 = K.compute_triangle_frames(V2, F); back = K.transform_bound_elements(local, fid, fr2, V_new=V2, barycentric=bary)
    assert np.allclose(back["position"], s * (pts @ Rz.T) + [0.3, 0.2, -0.4], atol=1e-10)      # follows the similarity transform exactly
    assert np.allclose(back["scale"], sc * s) and np.allclose(back["rotation"], np.einsum("nij,njk->nik", fr2["R"][fid], local["rotation"]))
    same = K.transform_bound_elements(K.bind_elements(pts, fid, fr), fid, fr); assert np.allclose(same["position"], pts, atol=1e-12)   # centroid origin round trip


def test_winding_number():
    V, F = octahedron(); q = np.array([[0, 0, 0], [0.2, 0.1, -0.1], [2, 0, 0], [0, -3, 1], [0.34, 0.33, 0.32], [0.4, 0.4, 0.4]])
    w = K.generalized_winding_number_float64(q, V, F, device="cpu")
    assert np.allclose(w[:2], 1.0, atol=1e-9) and np.allclose(w[2:4], 0.0, atol=1e-9) and w[4] > 0.99 and w[5] < 0.01   # |x|+|y|+|z| < 1 is inside
    assert np.allclose(K.generalized_winding_number_float64(q, V, F[:, ::-1], device="cpu")[:2], -1.0, atol=1e-9)


def test_signed_distance_and_two_phase_losses():
    V, F = octahedron(); P = np.array([[0.1, 0.1, 0.1], [0.3, 0.2, 0.2], [1.5, 0.0, 0.0]]); s, cp = K.signed_distance_to_mesh(P, V, F, k=8, device="cpu")
    face_d = (1 - P[:, :].sum(1)) / math.sqrt(3)                                   # distance to the +++ face plane x + y + z = 1
    assert s[0] < 0 and s[1] < 0 and s[2] > 0 and abs(s[0] + face_d[0]) < 1e-9 and abs(s[1] + face_d[1]) < 1e-9
    ve = np.array([[0, 1], [1, 2]]); Lt = np.array([0.5, 0.5]); Q = np.array([[0, 0, 0], [0.1, 0, 0], [0.1, 0.4, 0.0]])
    r = K.two_phase_registration_losses(Q, V, F, ve, Lt, iteration=0, n_iters=10); assert r["phase"] == "VE" and r["loss"] > 0
    fd = np.zeros_like(Q); h = 1e-6
    for i in range(3):
        for c in range(3):
            Qp = Q.copy(); Qp[i, c] += h; Qm = Q.copy(); Qm[i, c] -= h
            fd[i, c] = (K.two_phase_registration_losses(Qp, V, F, ve, Lt, 0, 10)["loss"] - K.two_phase_registration_losses(Qm, V, F, ve, Lt, 0, 10)["loss"]) / (2 * h)
    assert np.allclose(r["grad"], fd, atol=1e-6)
    pts = np.array([[0.15, 0.1, 0.05], [0.3, 0.3, 0.2], [1.5, 0.2, 0.1]]); b = K.two_phase_registration_losses(pts, V, F, None, None, iteration=9, n_iters=10, eps_body=0.05, k=8, device="cpu")
    assert b["phase"] == "body" and b["points_inside"] == 2 and b["loss"] > 0
    fd = np.zeros_like(pts)
    for i in range(3):
        for c in range(3):
            Pp = pts.copy(); Pp[i, c] += h; Pm = pts.copy(); Pm[i, c] -= h
            fd[i, c] = (K.two_phase_registration_losses(Pp, V, F, None, None, 9, 10, 0.05, 8, "cpu")["loss"] - K.two_phase_registration_losses(Pm, V, F, None, None, 9, 10, 0.05, 8, "cpu")["loss"]) / (2 * h)
    assert np.allclose(b["grad"], fd, atol=1e-5)


def test_horizontal_circumference():
    r, n = 0.15, 48; V, F = cylinder(r=r, n=n); exact = 2 * n * r * math.sin(math.pi / n)
    assert abs(K.measure_horizontal_circumference(V, F, 0.43) - exact) < 1e-9
    assert abs(K.measure_horizontal_circumference(V, F, 0.43, x_bounds=(0.0, 1.0)) - exact / 2) < 2 * r * math.sin(math.pi / n) + 1e-9   # right half, to within one facet
    assert K.measure_horizontal_circumference(V, F, 2.0) == 0.0
    assert abs(K.measure_horizontal_circumference(V, F, 0.5) - exact) < 1e-9                                   # plane exactly through a vertex ring


def run():
    t0 = time.time(); tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t1 = time.time(); t(); print(f"ok  {t.__name__:46s} {time.time() - t1:5.2f}s")
    print(f"{len(tests)} tests passed in {time.time() - t0:.2f}s"); return time.time() - t0


if __name__ == "__main__":
    sys.exit(0 if run() < 30 else 1)
