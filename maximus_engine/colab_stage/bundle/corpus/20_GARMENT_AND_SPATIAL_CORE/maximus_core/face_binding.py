"""Clean-room face binding: attach points (decals, chatons, Gaussians) to triangles so they follow a deforming mesh.

From the published formulation (Gaussian Garments, arXiv:2409.08189, supp. eq. 10-12):
    mu' = k_j R_j mu + tau_j        r' = R_j r        s' = k_j s
R_j: the face frame (edge direction, in-plane perpendicular, normal). k_j = (B + H) / 2 (base and height of the
triangle). tau_j: the surface point at the bound barycentric coordinates.
"""
import numpy as np

from .geometry import as_f64, tri_corners


def face_frames(V, F):
    """Per-face rotation R (m,3,3) with columns (e1, e2, n) and scale k = (base + height) / 2."""
    A, B, C = tri_corners(V, F)
    e1 = B - A; base = np.linalg.norm(e1, axis=1); e1 = e1 / np.maximum(base, 1e-15)[:, None]
    n = np.cross(B - A, C - A); twice_area = np.linalg.norm(n, axis=1); n = n / np.maximum(twice_area, 1e-15)[:, None]
    e2 = np.cross(n, e1)
    R = np.stack([e1, e2, n], axis=2)
    height = twice_area / np.maximum(base, 1e-15)
    return R, 0.5 * (base + height)


def barycentric(P, A, B, C):
    v0, v1, v2 = B - A, C - A, P - A
    d00 = (v0 * v0).sum(1); d01 = (v0 * v1).sum(1); d11 = (v1 * v1).sum(1); d20 = (v2 * v0).sum(1); d21 = (v2 * v1).sum(1)
    den = d00 * d11 - d01 * d01
    v = (d11 * d20 - d01 * d21) / den; w = (d00 * d21 - d01 * d20) / den
    return np.stack([1 - v - w, v, w], 1)


def bind(P, V, F, face_idx, bary=None):
    """Return a binding dict for points P on faces face_idx. If bary is None it is computed from P's projection."""
    P = as_f64(P); face_idx = np.asarray(face_idx, np.int64)
    A, B, C = tri_corners(V, np.asarray(F)[face_idx])
    if bary is None:
        bary = barycentric(P, A, B, C)
    tau = bary[:, :1] * A + bary[:, 1:2] * B + bary[:, 2:] * C
    R, k = face_frames(V, np.asarray(F)[face_idx])
    mu = np.einsum("nji,nj->ni", R, P - tau) / k[:, None]  # R^T (P - tau) / k
    return {"face": face_idx, "bary": np.asarray(bary, float), "mu": mu}


def unbind(binding, V, F):
    """Positions of bound points on a (deformed) mesh with the same topology: mu' = k R mu + tau."""
    f = binding["face"]; b = binding["bary"]
    A, B, C = tri_corners(V, np.asarray(F)[f])
    tau = b[:, :1] * A + b[:, 1:2] * B + b[:, 2:] * C
    R, k = face_frames(V, np.asarray(F)[f])
    return np.einsum("nij,nj->ni", R, binding["mu"]) * k[:, None] + tau
