"""Clean-room geometry kernels for the Maximus / 411 corpus (NumPy + SciPy, optional PyTorch CUDA for the winding number).

Plain Python: no Blender module, safe inside Blender's bundled Python as long as SciPy exists (Colab: yes; Blender: no, use system Python).
Units: metres. V is (n, 3) float vertices, F is (m, 3) int triangles, counter-clockwise seen from outside.

PROVENANCE. Written from the published formulas only, by this author, without copying any library code:
  MC-49  cotangent Laplacian + lumped mass            Pinkall & Polthier 1993; Meyer et al. 2003 (barycentric lumped mass)
  MC-50  heat-method geodesics + geodesic smooth mask  Crane, Weischedel & Wardetzky 2013 (heat method), smoothstep ramp is ours
  MC-51  biharmonic surface fairing                    Botsch & Kobbelt 2004 (Delta^2 X = 0 with fixed outer rings)
  MC-52  ARAP pose relaxation                          Sorkine & Alexa 2007 (local SVD / global cotangent solve)
  GG-30  triangle frames and bound elements            Gaussian Garments (arXiv 2409.08189): R_j in SO(3), k_j = (B + H) / 2, tau_j centroid
  GG-31  generalised winding number + L_VE -> L_body   Jacobson et al. 2013 / Van Oosterom & Strakhov 1983; GG schedule: L_VE then L_body
  GG-32  horizontal circumference                      plane / triangle slicing, perimeter of the in-bounds contour segments
NOTE ON SOURCES: the capability numbers come from the project's extraction plan for the Gaussian-Garments / libigl / PyTorch3D / Open3D
archive. GG-30 was checked against Supplementary B of arXiv 2409.08189 (corpus copy): the local frame of a Gaussian is R_j in SO(3) with the
face normal, one triangle edge and their cross product as basis, k_j = (B + H) / 2, and eqs. 10-12 (r = R_j r_local, mu = k_j R_j mu_local + tau,
s = k_j s_local), where tau is the Gaussian's own surface point (barycentric coordinates on the face), which is why bind_elements takes an optional
barycentric origin and defaults to the face centroid. The paper does not say which edge, so this file uses the first edge (v1 - v0), with B its
length and H the height of v2 over it. The other methods are the standard published forms; nothing was copied from the archive PDFs 05-08
(chat-style SEO write-ups, not primary sources). Every function has an independent unit test (Maximus_build/tests/test_canon_geometry_kernels.py).

Conventions:
  L (cotangent Laplacian)  positive semi-definite stiffness matrix, (L x)_i = 1/2 sum_j (cot a_ij + cot b_ij) (x_i - x_j); L 1 = 0
  M (mass)                 diagonal sparse, M_ii = one third of the area of the triangles around vertex i (sum = surface area)
"""
import math

import numpy as np

try:
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    from scipy.spatial import cKDTree as _KD
except Exception as _e:  # pragma: no cover
    sp = spla = _KD = None
    _IMPORT_ERROR = _e

CAPABILITIES = {
    "MC-49": "cotangent_laplacian_and_mass", "MC-50": "heat_geodesic_distance, geodesic_smooth_mask", "MC-51": "biharmonic_surface_fairing",
    "MC-52": "arap_pose_relaxation", "GG-30": "compute_triangle_frames, bind_elements, transform_bound_elements",
    "GG-31": "generalized_winding_number_float64, signed_distance_to_mesh, two_phase_registration_losses",
    "GG-32": "measure_horizontal_circumference",
}


def _need_scipy():
    if sp is None:
        raise ImportError(f"canon_geometry_kernels needs SciPy: {_IMPORT_ERROR}")


def _f64(a):
    return np.ascontiguousarray(a, dtype=np.float64)


def _i64(a):
    return np.ascontiguousarray(a, dtype=np.int64)


# ----------------------------------------------------------------------------------------------------------------------
# MC-49  cotangent Laplacian and mass
# ----------------------------------------------------------------------------------------------------------------------
def _cotangents(V, F):
    """Per-face cotangents of the angles at vertices (A, B, C) = (F0, F1, F2) and per-face areas. Degenerate faces get cot = 0."""
    A, B, C = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(B - A, C - A), axis=1); ok = area > 1e-14; den = np.where(ok, 2.0 * area, 1.0)
    dot = lambda a, b: np.einsum("ij,ij->i", a, b)   # noqa: E731
    cA = np.where(ok, dot(B - A, C - A) / den, 0.0); cB = np.where(ok, dot(A - B, C - B) / den, 0.0); cC = np.where(ok, dot(A - C, B - C) / den, 0.0)
    return cA, cB, cC, area


def cotangent_weights(V, F):
    """Symmetric sparse W with W_ij = 1/2 (cot alpha + cot beta) for every mesh edge (negative weights are kept)."""
    _need_scipy(); V = _f64(V); F = _i64(F); n = len(V); cA, cB, cC, _ = _cotangents(V, F)
    i = np.concatenate([F[:, 1], F[:, 2], F[:, 0]]); j = np.concatenate([F[:, 2], F[:, 0], F[:, 1]])   # edge opposite A, B, C
    w = 0.5 * np.concatenate([cA, cB, cC]); W = sp.coo_matrix((w, (i, j)), shape=(n, n)).tocsr()
    return (W + W.T).tocsr()


def cotangent_laplacian_and_mass(V, F):
    """MC-49. Returns (L, M): L = diag(W 1) - W (PSD, L 1 = 0), M = lumped barycentric mass (diagonal csr). Both (n, n) float64."""
    _need_scipy(); V = _f64(V); F = _i64(F); n = len(V)
    W = cotangent_weights(V, F); L = (sp.diags(np.asarray(W.sum(axis=1)).ravel()) - W).tocsr()
    _, _, _, area = _cotangents(V, F); m = np.zeros(n)
    for k in range(3):
        np.add.at(m, F[:, k], area / 3.0)
    return L, sp.diags(m).tocsr()


# ----------------------------------------------------------------------------------------------------------------------
# MC-50  heat-method geodesics and geodesic smooth mask
# ----------------------------------------------------------------------------------------------------------------------
def _mean_edge_length(V, F):
    e = np.concatenate([V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 1]], V[F[:, 0]] - V[F[:, 2]]])
    return float(np.linalg.norm(e, axis=1).mean())


def heat_geodesic_distance(V, F, source_indices, t=None):
    """MC-50. Geodesic distance (m) from the vertex set `source_indices` by the heat method: (M + t L) u = delta, X = -grad u / |grad u|,
    L phi = -div X, phi shifted to 0 at the sources. t defaults to (mean edge length)^2. Accurate to a few percent on well-shaped meshes."""
    _need_scipy(); V = _f64(V); F = _i64(F); n = len(V); src = np.atleast_1d(_i64(source_indices))
    L, M = cotangent_laplacian_and_mass(V, F); t = _mean_edge_length(V, F) ** 2 if t is None else float(t)
    u0 = np.zeros(n); u0[src] = 1.0; u = spla.splu((M + t * L).tocsc()).solve(u0)
    A, B, C = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]; N = np.cross(B - A, C - A); twice_area = np.linalg.norm(N, axis=1); ok = twice_area > 1e-14
    Nu = N / np.where(ok, twice_area, 1.0)[:, None]
    grad = (u[F[:, 0], None] * np.cross(Nu, C - B) + u[F[:, 1], None] * np.cross(Nu, A - C) + u[F[:, 2], None] * np.cross(Nu, B - A)) / np.where(ok, twice_area, 1.0)[:, None]
    gl = np.linalg.norm(grad, axis=1); X = np.where((gl > 1e-30)[:, None], -grad / np.where(gl > 1e-30, gl, 1.0)[:, None], 0.0)
    cA, cB, cC, _ = _cotangents(V, F); dot = lambda a, b: np.einsum("ij,ij->i", a, b)   # noqa: E731
    div = np.zeros(n)
    np.add.at(div, F[:, 0], 0.5 * (cC * dot(B - A, X) + cB * dot(C - A, X)))
    np.add.at(div, F[:, 1], 0.5 * (cA * dot(C - B, X) + cC * dot(A - B, X)))
    np.add.at(div, F[:, 2], 0.5 * (cB * dot(A - C, X) + cA * dot(B - C, X)))
    reg = 1e-8 * float(L.diagonal().mean()) / max(float(M.diagonal().mean()), 1e-300)
    phi = spla.splu((L + reg * M).tocsc()).solve(-div)            # L is PSD (= -Laplace-Beltrami), so L phi = -div X
    phi = phi - phi[src].min()
    return np.maximum(phi, 0.0)


def geodesic_smooth_mask(V, F, crease_seed_indices, r_zero, r_ramp):
    """MC-50. Smoothing mask in [0, 1]: 0 within r_zero (m) of the crease seeds, smoothstep up to 1 over the next r_ramp (m), 1 beyond."""
    d = heat_geodesic_distance(V, F, crease_seed_indices)
    x = np.clip((d - float(r_zero)) / max(float(r_ramp), 1e-12), 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


# ----------------------------------------------------------------------------------------------------------------------
# MC-51  biharmonic fairing
# ----------------------------------------------------------------------------------------------------------------------
def biharmonic_surface_fairing(V, F, interior_indices, pinned_indices=None):
    """MC-51. Solve Delta^2 X = 0 (Q = L M^-1 L) for the vertices in `interior_indices`; every other vertex keeps its position (the
    pinned set must lie outside the interior; give two rings of pins for C1 continuity). Returns the faired copy of V."""
    _need_scipy(); V = _f64(V); F = _i64(F); n = len(V); inter = np.unique(_i64(interior_indices))
    if inter.size == 0:
        return V.copy()
    if pinned_indices is not None and np.intersect1d(inter, _i64(pinned_indices)).size:
        raise ValueError("pinned_indices overlap interior_indices")
    L, M = cotangent_laplacian_and_mass(V, F); Q = (L @ sp.diags(1.0 / np.maximum(M.diagonal(), 1e-300)) @ L).tocsr()
    fixed = np.setdiff1d(np.arange(n), inter); Qff = Q[inter][:, inter].tocsc(); Qfc = Q[inter][:, fixed]
    out = V.copy(); out[inter] = spla.splu(Qff).solve(-(Qfc @ V[fixed])); return out


# ----------------------------------------------------------------------------------------------------------------------
# MC-52  ARAP pose relaxation
# ----------------------------------------------------------------------------------------------------------------------
def _best_rotations(S):
    """Rotation R_i = V diag(1, 1, det(V U^T)) U^T for S_i = U Sigma V^T, batched (n, 3, 3). Maximises tr(R^T S^T) -> R maps P onto Q."""
    U, _, Vt = np.linalg.svd(S); R = np.einsum("nji,nkj->nik", Vt, U)       # V U^T
    d = np.linalg.det(R); fix = d < 0
    if fix.any():
        Vt2 = Vt.copy(); Vt2[fix, 2, :] *= -1.0; R = np.einsum("nji,nkj->nik", Vt2, U)
    return R


def arap_pose_relaxation(V_rest, V_posed, F, active_indices, pinned_indices=None, n_iters=5):
    """MC-52. As-rigid-as-possible relaxation of the `active_indices` of V_posed against the rest shape V_rest: alternate the per-vertex
    best rotation (SVD) with a global cotangent solve. Vertices outside `active_indices` (and any in pinned_indices) stay at V_posed.
    Returns (V_relaxed, energies) where energies[k] is the ARAP energy before iteration k (non-increasing up to the Dirichlet solve)."""
    _need_scipy(); P = _f64(V_rest); Q = _f64(V_posed).copy(); F = _i64(F); n = len(P)
    act = np.unique(_i64(active_indices))
    if pinned_indices is not None:
        act = np.setdiff1d(act, _i64(pinned_indices))
    if act.size == 0:
        return Q, []
    W = cotangent_weights(P, F).tocoo(); i, j, w = W.row, W.col, W.data
    L = (sp.diags(np.asarray(W.sum(axis=1)).ravel()) - W).tocsr(); fixed = np.setdiff1d(np.arange(n), act)
    Lff = spla.splu(L[act][:, act].tocsc()); Lfc = L[act][:, fixed]; energies = []
    pe = P[i] - P[j]
    for _ in range(int(n_iters)):
        qe = Q[i] - Q[j]; S = np.zeros((n, 3, 3)); np.add.at(S, i, w[:, None, None] * pe[:, :, None] * qe[:, None, :])
        R = _best_rotations(S)
        energies.append(float(np.sum(w * np.sum((qe - np.einsum("eab,eb->ea", R[i], pe)) ** 2, axis=1))))
        rhs = np.zeros((n, 3)); np.add.at(rhs, i, 0.5 * w[:, None] * np.einsum("eab,eb->ea", R[i] + R[j], pe))
        Q[act] = Lff.solve(rhs[act] - Lfc @ Q[fixed])
    return Q, energies


# ----------------------------------------------------------------------------------------------------------------------
# GG-30  triangle frames and bound elements
# ----------------------------------------------------------------------------------------------------------------------
def compute_triangle_frames(V, F):
    """GG-30. Per-triangle rigid frame: R_j = [n | e1 | n x e1] in SO(3) (the paper's order: face normal, one edge, their cross product), scale k_j = (B + H) / 2
    with B = |v1 - v0| the base and H the height of v2 over that base, and tau_j = centroid. Returns {"R": (m,3,3), "k": (m,), "tau": (m,3), "faces": (m,3) int};
    degenerate faces get R = I, k = 0."""
    V = _f64(V); F = _i64(F); A, B_, C = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    e = B_ - A; base = np.linalg.norm(e, axis=1); n_raw = np.cross(e, C - A); nl = np.linalg.norm(n_raw, axis=1); ok = (base > 1e-14) & (nl > 1e-14)
    e1 = e / np.where(ok, base, 1.0)[:, None]; n = n_raw / np.where(ok, nl, 1.0)[:, None]; b = np.cross(n, e1)
    R = np.stack([n, e1, b], axis=2); R[~ok] = np.eye(3); H = np.abs(np.einsum("ij,ij->i", C - A, b)); k = np.where(ok, 0.5 * (base + H), 0.0)
    return {"R": R, "k": k, "tau": (A + B_ + C) / 3.0, "faces": F}


def _origin(V, frames, f, barycentric):
    if barycentric is None:
        return frames["tau"][f]
    b = _f64(barycentric); T = V[frames["faces"][f]]                       # (n, 3, 3): the three corners of each element's face
    return np.einsum("na,nai->ni", b, T)


def bind_elements(positions, face_ids, frames, rotations=None, scales=None, V=None, barycentric=None):
    """GG-30. Express world-space elements (e.g. Gaussians) in the frame of the triangle they ride on:
    mu_local = R_j^T (mu - tau) / k_j, rotation_local = R_j^T rotation, scale_local = scale / k_j. rotations: (n,3,3) or None.
    tau is the face centroid, or, with `barycentric` ((n,3) weights) and the vertices V of the SAME mesh the frames were computed on, the
    element's own surface point as in the paper."""
    p = _f64(positions); f = _i64(face_ids); R, k = frames["R"][f], frames["k"][f]; tau = _origin(_f64(V) if V is not None else None, frames, f, barycentric)
    out = {"position": np.einsum("nba,nb->na", R, p - tau) / k[:, None]}
    if rotations is not None:
        out["rotation"] = np.einsum("nba,nbc->nac", R, _f64(rotations))
    if scales is not None:
        s = _f64(scales); out["scale"] = s / (k if s.ndim == 1 else k[:, None])
    return out


def transform_bound_elements(local, face_ids, frames_new, V_new=None, barycentric=None):
    """GG-30 (paper eqs. 10-12). Inverse of bind_elements on a deformed mesh: mu = k_j R_j mu_local + tau, rotation = R_j rotation_local,
    scale = k_j scale_local, with tau the centroid or (barycentric + V_new) the element's surface point on the deformed face."""
    f = _i64(face_ids); R, k = frames_new["R"][f], frames_new["k"][f]; tau = _origin(_f64(V_new) if V_new is not None else None, frames_new, f, barycentric)
    out = {"position": k[:, None] * np.einsum("nab,nb->na", R, local["position"]) + tau}
    if "rotation" in local:
        out["rotation"] = np.einsum("nab,nbc->nac", R, local["rotation"])
    if "scale" in local:
        s = local["scale"]; out["scale"] = s * (k if s.ndim == 1 else k[:, None])
    return out


# ----------------------------------------------------------------------------------------------------------------------
# GG-31  winding number, signed distance, two-phase registration losses
# ----------------------------------------------------------------------------------------------------------------------
def _winding_numpy(Q, A, B, C, max_elems=3_000_000):
    m = len(A); step = max(1, max_elems // max(m, 1)); w = np.empty(len(Q))
    for s in range(0, len(Q), step):
        q = Q[s:s + step, None, :]; a = A[None] - q; b = B[None] - q; c = C[None] - q
        la, lb, lc = (np.linalg.norm(x, axis=2) for x in (a, b, c))
        det = np.einsum("qmi,qmi->qm", a, np.cross(b, c))
        den = la * lb * lc + np.einsum("qmi,qmi->qm", a, b) * lc + np.einsum("qmi,qmi->qm", b, c) * la + np.einsum("qmi,qmi->qm", c, a) * lb
        w[s:s + step] = (2.0 * np.arctan2(det, den)).sum(axis=1) / (4.0 * math.pi)
    return w


def _winding_torch(Q, A, B, C, device, max_elems=36_000_000):
    import torch
    f64 = torch.float64; Qt = torch.as_tensor(Q, dtype=f64, device=device)
    At, Bt, Ct = (torch.as_tensor(x, dtype=f64, device=device) for x in (A, B, C)); m = len(A); step = max(1, max_elems // max(m, 1)); out = []
    for s in range(0, len(Q), step):
        q = Qt[s:s + step, None, :]; a = At[None] - q; b = Bt[None] - q; c = Ct[None] - q
        la, lb, lc = a.norm(dim=2), b.norm(dim=2), c.norm(dim=2); det = (a * torch.cross(b, c, dim=2)).sum(2)
        den = la * lb * lc + (a * b).sum(2) * lc + (b * c).sum(2) * la + (c * a).sum(2) * lb
        out.append((2.0 * torch.atan2(det, den)).sum(1) / (4.0 * math.pi))
    return torch.cat(out).cpu().numpy()


def generalized_winding_number_float64(query_points, V_surface, F_surface, device="auto"):
    """GG-31. Generalised winding number of each query point w.r.t. the closed, outward-oriented triangle mesh (V, F), float64
    (solid-angle sum, Van Oosterom-Strakhov): ~1 inside, ~0 outside, fractional for open or non-manifold surfaces.
    device: "cpu" (NumPy), "cuda" (PyTorch float64) or "auto" (CUDA when PyTorch sees a GPU and the problem is large)."""
    Q = _f64(query_points).reshape(-1, 3); V = _f64(V_surface); F = _i64(F_surface); A, B, C = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    use = device
    if device == "auto":
        use = "cpu"
        if len(Q) * len(F) > 2_000_000:
            try:
                import torch
                use = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:  # noqa: BLE001
                use = "cpu"
    return _winding_torch(Q, A, B, C, "cuda") if use == "cuda" else _winding_numpy(Q, A, B, C)


def _closest_point_on_triangles(P, A, B, C):
    """Row-wise closest point on triangle (A, B, C) to P (all (k, 3)); region test on the barycentric/Voronoi regions of the triangle."""
    ab, ac = B - A, C - A; ap, bp, cp = P - A, P - B, P - C; dot = lambda a, b: np.einsum("ij,ij->i", a, b)   # noqa: E731
    d1, d2, d3, d4, d5, d6 = dot(ab, ap), dot(ac, ap), dot(ab, bp), dot(ac, bp), dot(ab, cp), dot(ac, cp)
    vc, vb, va = d1 * d4 - d3 * d2, d5 * d2 - d1 * d6, d3 * d6 - d5 * d4
    den = np.where(np.abs(va + vb + vc) > 1e-300, va + vb + vc, 1.0); res = A + ab * (vb / den)[:, None] + ac * (vc / den)[:, None]
    sel = lambda m, pt: np.where(m[:, None], pt, res)   # noqa: E731
    safe = lambda x: np.where(np.abs(x) > 1e-300, x, 1.0)   # noqa: E731
    res = sel((va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0), B + (C - B) * ((d4 - d3) / safe((d4 - d3) + (d5 - d6)))[:, None])
    res = sel((vb <= 0) & (d2 >= 0) & (d6 <= 0), A + ac * (d2 / safe(d2 - d6))[:, None])
    res = sel((d6 >= 0) & (d5 <= d6), C)
    res = sel((vc <= 0) & (d1 >= 0) & (d3 <= 0), A + ab * (d1 / safe(d1 - d3))[:, None])
    res = sel((d3 >= 0) & (d4 <= d3), B)
    return sel((d1 <= 0) & (d2 <= 0), A)


def signed_distance_to_mesh(P, V, F, k=16, device="auto"):
    """GG-31. Signed distance (positive outside) of points P to the closed mesh (V, F): unsigned distance from the closest point among the
    k faces with the nearest centroids, sign from the winding number (> 0.5 = inside). Returns (signed, closest_points)."""
    _need_scipy(); P = _f64(P).reshape(-1, 3); V = _f64(V); F = _i64(F); A, B, C = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    k = min(int(k), len(F)); cen = (A + B + C) / 3.0; _, idx = _KD(cen).query(P, k=k); idx = idx.reshape(len(P), k)
    best = np.full(len(P), np.inf); cp = np.zeros_like(P)
    for c in range(k):
        f = idx[:, c]; q = _closest_point_on_triangles(P, A[f], B[f], C[f]); d = np.linalg.norm(P - q, axis=1); better = d < best
        best = np.where(better, d, best); cp = np.where(better[:, None], q, cp)
    inside = generalized_winding_number_float64(P, V, F, device) > 0.5
    return np.where(inside, -best, best), cp


def two_phase_registration_losses(P, V_body, F_body, virtual_edges, L_template, iteration, n_iters, eps_body=0.003, k=16, device="auto"):
    """GG-31. One evaluation of the two-phase registration schedule: the first half of the iterations minimises the virtual-edge compression
    L_VE = sum_e max(L_e - |P_i - P_j|, 0)^2 (virtual_edges: (E, 2) indices into P, L_template: (E,) template lengths), the second half the
    one-sided cubic body barrier L_body = sum_i max(eps_body - s_i, 0)^3 (s_i = signed distance of P_i to the body mesh, positive outside).
    Returns {"phase": "VE" | "body", "loss": float, "grad": (n,3) gradient w.r.t. P, ...}."""
    P = _f64(P); n = len(P); grad = np.zeros((n, 3))
    if int(iteration) < int(n_iters) // 2:
        ve = _i64(virtual_edges); Le = _f64(L_template); d = P[ve[:, 0]] - P[ve[:, 1]]; l = np.linalg.norm(d, axis=1)
        viol = np.maximum(Le - l, 0.0); g = (-2.0 * viol / np.maximum(l, 1e-12))[:, None] * d
        np.add.at(grad, ve[:, 0], g); np.add.at(grad, ve[:, 1], -g)
        return {"phase": "VE", "loss": float((viol ** 2).sum()), "grad": grad, "active_edges": int((viol > 0).sum())}
    s, cp = signed_distance_to_mesh(P, V_body, F_body, k=k, device=device); pen = np.maximum(float(eps_body) - s, 0.0)
    dirv = P - cp; dl = np.linalg.norm(dirv, axis=1); sign = np.where(s < 0, -1.0, 1.0)
    grad_s = sign[:, None] * dirv / np.maximum(dl, 1e-12)[:, None]
    grad = -3.0 * (pen ** 2)[:, None] * grad_s
    return {"phase": "body", "loss": float((pen ** 3).sum()), "grad": grad, "points_in_barrier": int((pen > 0).sum()),
            "points_inside": int((s < 0).sum()), "min_signed_distance_m": float(s.min()) if len(s) else None}


# ----------------------------------------------------------------------------------------------------------------------
# GG-32  horizontal circumference
# ----------------------------------------------------------------------------------------------------------------------
def horizontal_slice_segments(V, F, z_height, exclude_vertex_mask=None):
    """All segments where the plane z = z_height cuts the triangles: returns (S0, S1) (k, 3) end points. A vertex exactly on the plane counts
    as above it (consistent rule, so a contour through vertices is neither dropped nor doubled). If `exclude_vertex_mask` (bool (n,)) is
    given, triangles touching any excluded vertex (e.g. arms/hands/fingers during torso girth slicing) are dropped before slicing."""
    V = _f64(V); F = _i64(F)
    if exclude_vertex_mask is not None:
        ex = np.asarray(exclude_vertex_mask, dtype=bool)
        F = F[~ex[F].any(axis=1)]
    h = float(z_height); above = V[:, 2] >= h; fa = above[F]; cut = fa.any(axis=1) & ~fa.all(axis=1)
    S0, S1 = [], []
    for tri in F[cut]:
        a = above[tri]; lone = int(np.flatnonzero(a)[0]) if a.sum() == 1 else int(np.flatnonzero(~a)[0])   # the vertex alone on its side
        o1, o2 = [x for x in range(3) if x != lone]
        pts = []
        for o in (o1, o2):
            p, q = V[tri[lone]], V[tri[o]]; t = (h - p[2]) / (q[2] - p[2]); pts.append(p + t * (q - p))
        S0.append(pts[0]); S1.append(pts[1])
    return (np.array(S0).reshape(-1, 3), np.array(S1).reshape(-1, 3))


def measure_horizontal_circumference(V, F, z_height, x_bounds=None, exclude_vertex_mask=None):
    """GG-32. Length (m) of the cross-section contour at z = z_height. x_bounds = (x_min, x_max) keeps only the segments whose midpoint lies in
    that x interval (e.g. the torso, excluding the arms); None keeps everything. exclude_vertex_mask (optional bool (n,)) masks out arm/hand
    vertices before slicing. The horizontal plane makes every segment lie in XY."""
    S0, S1 = horizontal_slice_segments(V, F, z_height, exclude_vertex_mask=exclude_vertex_mask)
    if len(S0) == 0:
        return 0.0
    if x_bounds is not None:
        mx = 0.5 * (S0[:, 0] + S1[:, 0]); keep = (mx >= float(x_bounds[0])) & (mx <= float(x_bounds[1])); S0, S1 = S0[keep], S1[keep]
    return float(np.linalg.norm(S1[:, :2] - S0[:, :2], axis=1).sum())
