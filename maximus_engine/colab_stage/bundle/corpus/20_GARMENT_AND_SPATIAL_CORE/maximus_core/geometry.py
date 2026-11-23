"""Triangle-mesh basics in plain NumPy (float64). Runs in system Python and in Blender's bundled Python.

V: (n, 3) float vertex positions.  F: (m, 3) int triangle indices.  Units: metres.
"""
import numpy as np


def as_f64(V):
    return np.ascontiguousarray(V, dtype=np.float64)


def tri_corners(V, F):
    V = as_f64(V); F = np.asarray(F, dtype=np.int64)
    return V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]


def tri_areas(V, F):
    A, B, C = tri_corners(V, F)
    return 0.5 * np.linalg.norm(np.cross(B - A, C - A), axis=1)


def face_normals(V, F, normalize=True):
    A, B, C = tri_corners(V, F)
    N = np.cross(B - A, C - A)
    if normalize:
        L = np.linalg.norm(N, axis=1, keepdims=True)
        N = np.divide(N, L, out=np.zeros_like(N), where=L > 0)
    return N


def vertex_normals(V, F):
    """Area-weighted vertex normals."""
    N = face_normals(V, F, normalize=False)
    VN = np.zeros((len(V), 3))
    for k in range(3):
        np.add.at(VN, np.asarray(F)[:, k], N)
    L = np.linalg.norm(VN, axis=1, keepdims=True)
    return np.divide(VN, L, out=np.zeros_like(VN), where=L > 0)


def unique_edges(F):
    F = np.asarray(F, dtype=np.int64)
    E = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    E.sort(axis=1)
    return np.unique(E, axis=0)


def edge_lengths(V, E):
    V = as_f64(V)
    return np.linalg.norm(V[E[:, 0]] - V[E[:, 1]], axis=1)


def umbrella(V, E, n=None):
    """Uniform Laplacian: mean of neighbours minus the vertex (zero for isolated vertices)."""
    V = as_f64(V); n = len(V) if n is None else n
    S = np.zeros((n, 3)); deg = np.zeros(n)
    np.add.at(S, E[:, 0], V[E[:, 1]]); np.add.at(S, E[:, 1], V[E[:, 0]])
    np.add.at(deg, E[:, 0], 1); np.add.at(deg, E[:, 1], 1)
    out = np.zeros((n, 3)); m = deg > 0
    out[m] = S[m] / deg[m, None] - V[m]
    return out


def collapsed_triangles(V_rest, V_posed, F, ratio=0.25, rest_min_area=1e-10):
    """Triangles whose posed area fell below `ratio` x their rest area (rest-degenerate triangles are excluded)."""
    a0 = tri_areas(V_rest, F); a1 = tri_areas(V_posed, F)
    valid = a0 > rest_min_area
    r = np.full(len(F), np.inf); r[valid] = a1[valid] / a0[valid]
    idx = np.nonzero(r < ratio)[0]
    return {"count": int(len(idx)), "indices": idx.tolist(), "threshold_ratio": ratio,
            "min_ratio": float(r[valid].min()) if valid.any() else None, "rest_degenerate_excluded": int((~valid).sum())}


def flipped_triangles(V_ref, V_new, F, mask=None):
    """Triangles whose normal turned by more than 90 degrees between V_ref and V_new."""
    n0 = face_normals(V_ref, F); n1 = face_normals(V_new, F)
    d = np.einsum("ij,ij->i", n0, n1)
    bad = d < 0
    if mask is not None:
        bad &= mask
    return np.nonzero(bad)[0]
