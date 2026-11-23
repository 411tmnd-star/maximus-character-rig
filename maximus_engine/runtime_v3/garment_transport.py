"""garment_transport.py  (Maximus runtime_v3, module 2 of 8)  Zero-refit garment binding and transport on the body's triangle frames (GG-30).

Bind once against a reference body (same 13,380-vertex topology as Maximus v2: the v1 canonical body and every garment master fitted to it):
    for garment vertex v_k: j(k) = body triangle with the closest surface point, beta_k = barycentric coordinates of that point,
    tau_j = that surface point, mu_k = R_j^T (v_k - tau_j) / k_j          (R_j = [n | e1 | n x e1] in SO(3), k_j = (B + H) / 2)
Transport to any posed body (no refit, no simulation):
    v_k(t) = tau_j(t) + k_j(t) R_j(t) mu_k        (paper mode, eqs. 10-12 of arXiv 2409.08189 Supp. B: the offset scales with k_j)
    clearance mode: the normal component of mu_k is kept in metres (unscaled), so a 3 mm gap stays 3 mm when the triangle shrinks.
Clearance gate: no garment vertex inside the body (generalised winding number > 0.5, float64, CUDA when available) and the minimum signed
distance (positive outside) is reported against the 3 mm target; vertices under 3 mm are counted.
"""
from __future__ import annotations

import numpy as np

try:
    from scipy.spatial import cKDTree as _KD
except Exception:  # pragma: no cover
    _KD = None


def frames(K, V, F):
    return K.compute_triangle_frames(np.asarray(V, float), np.asarray(F, np.int64))


def _closest(K, P, V, F, k=12):
    """closest surface point of each P on (V, F) among the k triangles with the nearest centroids: (face id, barycentric, point, distance)."""
    A, B, C = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]; cen = (A + B + C) / 3.0
    k = min(k, len(F)); _, idx = _KD(cen).query(P, k=k); idx = idx.reshape(len(P), k)
    best = np.full(len(P), np.inf); fid = np.zeros(len(P), np.int64); cp = np.zeros_like(P)
    for c in range(k):
        f = idx[:, c]; q = K._closest_point_on_triangles(P, A[f], B[f], C[f]); d = np.linalg.norm(P - q, axis=1); m = d < best
        best = np.where(m, d, best); fid = np.where(m, f, fid); cp = np.where(m[:, None], q, cp)
    a, b, c_ = A[fid], B[fid], C[fid]; v0, v1, v2 = b - a, c_ - a, cp - a
    d00, d01, d11 = (v0 * v0).sum(1), (v0 * v1).sum(1), (v1 * v1).sum(1); d20, d21 = (v2 * v0).sum(1), (v2 * v1).sum(1)
    den = np.where(np.abs(d00 * d11 - d01 * d01) > 1e-30, d00 * d11 - d01 * d01, 1.0)
    bv = (d11 * d20 - d01 * d21) / den; bw = (d00 * d21 - d01 * d20) / den
    return fid, np.c_[1 - bv - bw, bv, bw], cp, best


def bind(K, garment_V, body_V, body_F, k=12):
    G = np.asarray(garment_V, float); V = np.asarray(body_V, float); F = np.asarray(body_F, np.int64)
    fr = frames(K, V, F); fid, bary, cp, dist = _closest(K, G, V, F, k)
    R, kk = fr["R"][fid], fr["k"][fid]
    mu = np.einsum("nba,nb->na", R, G - cp) / kk[:, None]
    normal_m = np.einsum("nb,nb->n", R[:, :, 0], G - cp)                         # signed offset along the face normal, metres
    return {"face": fid, "bary": bary, "mu": mu, "normal_m": normal_m, "k_bind": kk, "dist_bind_m": dist, "n": len(G)}


def transport(K, binding, body_V, body_F, mode="paper"):
    V = np.asarray(body_V, float); F = np.asarray(body_F, np.int64); fr = frames(K, V, F); f = binding["face"]
    tri = V[F[f]]; tau = np.einsum("na,nai->ni", binding["bary"], tri); R, kk = fr["R"][f], fr["k"][f]
    if mode == "paper":
        return tau + kk[:, None] * np.einsum("nab,nb->na", R, binding["mu"])
    mu_t = binding["mu"].copy(); mu_t[:, 0] = 0.0                                 # tangential part scales with k, normal part stays in metres
    return tau + kk[:, None] * np.einsum("nab,nb->na", R, mu_t) + R[:, :, 0] * binding["normal_m"][:, None]


def clearance(K, garment_V, body_V, body_F, target_m=0.003, device="auto", k=16):
    s, _ = K.signed_distance_to_mesh(np.asarray(garment_V, float), np.asarray(body_V, float), np.asarray(body_F, np.int64), k=k, device=device)
    inside = int((s < 0).sum())
    return {"points": int(len(s)), "inside": inside, "min_signed_mm": round(float(s.min()) * 1e3, 3), "p01_mm": round(float(np.percentile(s, 1)) * 1e3, 3),
            "median_mm": round(float(np.median(s)) * 1e3, 3), "under_target": int((s < target_m).sum()), "target_mm": target_m * 1e3,
            "gate_no_penetration": inside == 0, "gate_min_ge_target": bool(s.min() >= target_m)}
