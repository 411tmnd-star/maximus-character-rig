"""Clean-room clearance and registration energies (NumPy; SciPy used only as an optional speed-up).

Written from the published formulas only (Rong, Grigorev et al., "Gaussian Garments", arXiv:2409.08189, eq. 4 and
supplementary eq. 16). No code from the unlicensed eth-ait repository was copied or consulted.

  L_body = sum_i max(eps_body - (v_i - f_i) . n_i, 0)^3        eps_body = 3 mm   (one-sided cubic barrier)
  L_VE   = sum_e max(L_e - l_e, 0)^2                            template length L_e vs current length l_e
           over "virtual edges" that join opposite sides of a surface (found by inward ray casts). The paper uses
           L_VE for the first half of each optimisation and L_body for the second half.

Also: exact point-triangle distance, ray/segment-triangle intersection (Moller-Trumbore), and the generalised
winding number (solid-angle sum, float64) for inside/outside tests.
"""
import numpy as np

from .geometry import as_f64, face_normals, tri_corners

EPS_BODY = 0.003  # metres

try:  # optional accelerator (not available inside Blender's Python)
    from scipy.spatial import cKDTree as _KD
except Exception:  # pragma: no cover
    _KD = None


def _dot(a, b):
    return np.einsum("ij,ij->i", a, b)


def closest_point_on_triangles(P, A, B, C):
    """Row-wise closest point on triangle (A,B,C) to P (Ericson, Real-Time Collision Detection 5.1.5). All (k,3)."""
    ab, ac, ap = B - A, C - A, P - A
    d1, d2 = _dot(ab, ap), _dot(ac, ap)
    bp = P - B; d3, d4 = _dot(ab, bp), _dot(ac, bp)
    cp = P - C; d5, d6 = _dot(ab, cp), _dot(ac, cp)
    vc = d1 * d4 - d3 * d2; vb = d5 * d2 - d1 * d6; va = d3 * d6 - d5 * d4
    with np.errstate(divide="ignore", invalid="ignore"):
        v_ab = d1 / (d1 - d3)
        w_ac = d2 / (d2 - d6)
        w_bc = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        den = 1.0 / (va + vb + vc)
    v_in, w_in = vb * den, vc * den
    conds = [(d1 <= 0) & (d2 <= 0), (d3 >= 0) & (d4 <= d3), (vc <= 0) & (d1 >= 0) & (d3 <= 0),
             (d6 >= 0) & (d5 <= d6), (vb <= 0) & (d2 >= 0) & (d6 <= 0), (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)]
    choices = [A, B, A + v_ab[:, None] * ab, C, A + w_ac[:, None] * ac, B + w_bc[:, None] * (C - B)]
    inside = A + v_in[:, None] * ab + w_in[:, None] * ac
    out = inside.copy()
    taken = np.zeros(len(P), bool)
    for c, val in zip(conds, choices):
        m = c & ~taken
        out[m] = val[m]; taken |= m
    return out


def _knn_faces(P, centroids, k):
    k = min(k, len(centroids))
    if _KD is not None:
        _, idx = _KD(centroids).query(P, k=k)
        return idx.reshape(len(P), k)
    out = np.empty((len(P), k), np.int64)
    for s in range(0, len(P), 256):
        d = ((P[s:s + 256, None, :] - centroids[None]) ** 2).sum(-1)
        out[s:s + 256] = np.argpartition(d, k - 1, axis=1)[:, :k]
    return out


def point_mesh_distance(P, V, F, k=16):
    """Closest point on the mesh for each P (searching the k faces with nearest centroids).
    Returns dict: distance, signed (along the closest face's normal), face, closest."""
    P = as_f64(P); V = as_f64(V); F = np.asarray(F, np.int64)
    A, B, C = tri_corners(V, F); cen = (A + B + C) / 3.0; N = face_normals(V, F)
    cand = _knn_faces(P, cen, k)
    best_d = np.full(len(P), np.inf); best_f = np.zeros(len(P), np.int64); best_q = np.zeros_like(P)
    for j in range(cand.shape[1]):
        f = cand[:, j]
        q = closest_point_on_triangles(P, A[f], B[f], C[f])
        d = np.linalg.norm(P - q, axis=1)
        m = d < best_d
        best_d[m], best_f[m], best_q[m] = d[m], f[m], q[m]
    signed = _dot(P - best_q, N[best_f])
    return {"distance": best_d, "signed": signed, "face": best_f, "closest": best_q}


def l_body(P, V, F, eps=EPS_BODY, k=16):
    """Body-penetration energy and its gradient w.r.t. P. Returns (energy, grad, info)."""
    r = point_mesh_distance(P, V, F, k=k)
    N = face_normals(V, F)[r["face"]]
    viol = np.maximum(eps - r["signed"], 0.0)
    energy = float((viol ** 3).sum())
    grad = (-3.0 * viol ** 2)[:, None] * N
    info = {"eps_m": eps, "n_within_eps": int((viol > 0).sum()), "n_penetrating": int((r["signed"] < 0).sum()),
            "min_signed_m": float(r["signed"].min()) if len(P) else None, "signed": r["signed"]}
    return energy, grad, info


def l_ve(V, VE, L_template):
    """Virtual-edge energy (compression only) and gradient w.r.t. V. VE: (k,2) vertex pairs."""
    V = as_f64(V); d = V[VE[:, 0]] - V[VE[:, 1]]; l = np.linalg.norm(d, axis=1)
    short = np.maximum(L_template - l, 0.0)
    energy = float((short ** 2).sum())
    g_edge = (-2.0 * short / np.maximum(l, 1e-12))[:, None] * d
    G = np.zeros_like(V)
    np.add.at(G, VE[:, 0], g_edge); np.add.at(G, VE[:, 1], -g_edge)
    return energy, G


def ray_triangle(orig, dirs, V, F, t_min=1e-6, chunk=256):
    """First hit (t, face) of each ray against all triangles; t = inf where nothing is hit (Moller-Trumbore)."""
    orig = as_f64(orig); dirs = as_f64(dirs)
    A, B, C = tri_corners(V, F); e1, e2 = B - A, C - A
    T = np.full(len(orig), np.inf); FI = np.full(len(orig), -1, np.int64)
    for s in range(0, len(orig), chunk):
        o = orig[s:s + chunk, None, :]; d = dirs[s:s + chunk, None, :]
        p = np.cross(d, e2[None]); det = (e1[None] * p).sum(-1)
        ok = np.abs(det) > 1e-14
        inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
        tv = o - A[None]; u = (tv * p).sum(-1) * inv
        q = np.cross(tv, e1[None]); v = (d * q).sum(-1) * inv; t = (e2[None] * q).sum(-1) * inv
        hit = ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > t_min)
        t = np.where(hit, t, np.inf)
        j = np.argmin(t, axis=1); tt = t[np.arange(len(j)), j]
        T[s:s + chunk] = tt; FI[s:s + chunk] = np.where(np.isfinite(tt), j, -1)
    return T, FI


def build_virtual_edges(V, F, faces=None, max_len=None):
    """Join opposite sides of a surface: cast an inward ray (-normal) from each face centroid; connect the face's
    vertex nearest the centroid with the hit face's vertex nearest the hit point. Returns unique (k,2) pairs."""
    V = as_f64(V); F = np.asarray(F, np.int64)
    faces = np.arange(len(F)) if faces is None else np.asarray(faces)
    sub = F[faces]
    A, B, C = tri_corners(V, sub); cen = (A + B + C) / 3.0; N = face_normals(V, sub)
    t, hit = ray_triangle(cen + N * -1e-6, -N, V, sub)
    ok = np.isfinite(t) & (hit >= 0)
    if max_len is not None:
        ok &= t <= max_len
    src = sub[ok]; dst = sub[hit[ok]]; hp = cen[ok] - N[ok] * t[ok, None]
    a = src[np.arange(len(src)), np.argmin(((V[src] - cen[ok, None]) ** 2).sum(-1), axis=1)]
    b = dst[np.arange(len(dst)), np.argmin(((V[dst] - hp[:, None]) ** 2).sum(-1), axis=1)]
    E = np.stack([a, b], 1); E = E[E[:, 0] != E[:, 1]]; E.sort(axis=1)
    return np.unique(E, axis=0) if len(E) else E.reshape(0, 2)


def winding_number(P, V, F, chunk=64):
    """Generalised winding number of points P w.r.t. the triangle soup (V,F): ~1 inside, ~0 outside (float64)."""
    P = as_f64(P); A, B, C = tri_corners(V, F)
    w = np.zeros(len(P))
    for s in range(0, len(P), chunk):
        p = P[s:s + chunk, None, :]
        a, b, c = A[None] - p, B[None] - p, C[None] - p
        la, lb, lc = (np.linalg.norm(x, axis=-1) for x in (a, b, c))
        det = (a * np.cross(b, c)).sum(-1)
        den = la * lb * lc + (a * b).sum(-1) * lc + (b * c).sum(-1) * la + (c * a).sum(-1) * lb
        w[s:s + chunk] = (2.0 * np.arctan2(det, den)).sum(1) / (4.0 * np.pi)
    return w


def segment_triangle_hits(S0, S1, V, F, candidate_radius=None):
    """Count crossings between segments (S0->S1) and triangles. Returns (n_hits, array of (segment, face))."""
    S0 = as_f64(S0); S1 = as_f64(S1); A, B, C = tri_corners(V, F)
    cen = (A + B + C) / 3.0
    rad = np.max(np.stack([np.linalg.norm(X - cen, axis=1) for X in (A, B, C)]), axis=0)
    mid = (S0 + S1) / 2; half = np.linalg.norm(S1 - S0, axis=1) / 2
    R = float(rad.max() + half.max()) if candidate_radius is None else candidate_radius
    if _KD is not None:
        lists = _KD(cen).query_ball_point(mid, R)
        si = np.repeat(np.arange(len(mid)), [len(l) for l in lists]); fi = np.concatenate([np.asarray(l, np.int64) for l in lists]) if len(si) else np.zeros(0, np.int64)
    else:
        si_l, fi_l = [], []
        for s in range(0, len(mid), 256):
            d = np.linalg.norm(mid[s:s + 256, None] - cen[None], axis=-1)
            a, b = np.nonzero(d <= R); si_l.append(a + s); fi_l.append(b)
        si, fi = np.concatenate(si_l), np.concatenate(fi_l)
    if len(si) == 0:
        return 0, np.zeros((0, 2), np.int64)
    o = S0[si]; d = S1[si] - S0[si]; e1 = B[fi] - A[fi]; e2 = C[fi] - A[fi]
    p = np.cross(d, e2); det = _dot(e1, p); ok = np.abs(det) > 1e-14
    inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
    tv = o - A[fi]; u = _dot(tv, p) * inv; q = np.cross(tv, e1); v = _dot(d, q) * inv; t = _dot(e2, q) * inv
    hit = ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t >= 0) & (t <= 1)
    pairs = np.stack([si[hit], fi[hit]], 1)
    return int(len(pairs)), pairs
