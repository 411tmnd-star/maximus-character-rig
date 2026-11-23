"""pose_space_engine.py  (Maximus runtime_v3, module 1 of 8)  Pose-space corrective deformation for the pure-LBS Maximus v2 rig.

Two parts, both plain NumPy/SciPy (no Blender, no torch):

1. Online corrective solver (what M4 used to bake the sample correctives, and what the runtime uses for an out-of-sample pose):
       X_p        posed vertices (armature LBS + the rig's Corrective Smooth, as evaluated)
       M_i(p)     per-vertex linear-blend 3x3 (sum_b w_ib R_b), the "skinning matrix" of vertex i
   a) seeds = vertices of triangles whose area ratio (posed / rest) is below `seed_ratio`
   b) active region = geodesic disc (heat method, MC-50) of radius `radius` around the seeds; everything else is pinned at X_p
   c) ARAP relaxation (MC-52) of the active region against the REST triangles: restores local rigidity, so collapsed triangles open
   d) residual repair: while a triangle is below `target_ratio`, push its three vertices along the in-plane area gradient (all inside the
      active region), Laplacian-smoothed so the correction stays smooth; bounded number of steps
   e) un-skin: the posed-space displacement c_i is mapped back to rest space with the exact inverse of the vertex's own blend matrix,
      d_i = M_i(p)^-1 c_i (not the transpose: LBS blend matrices are not orthogonal)
   The result d (rest space) is a corrective shape-key delta for pose p.

2. Pose-space interpolation (Lewis et al. 2000 PSD, with a quaternion metric):
       psi_s(theta) = exp(-d_H(theta, theta_s)^2 / (2 sigma^2)),   d_H^2 = sum over the family's driving joints of (2 acos |<q_j, q_j,s>|)^2
   Cardinal interpolation: lambda = (Phi + ridge I)^-1 T so that the weight of corrective k is 1 at its own sample pose and 0 at every other
   sample (including bind, which anchors every weight at 0 so the rest shape never moves). Weights are clipped to [0, 1].
   A family = a body region (e.g. axilla_L) with its driving joints; each of its keys is one sample pose's corrective restricted to the region.

Units: metres, quaternions (w, x, y, z) as pose-bone local rotations (matrix_basis), canonical hemisphere.
"""
from __future__ import annotations

import json
import math

import numpy as np

try:
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
except Exception:  # pragma: no cover
    sp = spla = None


# ----------------------------------------------------------------------------------------------------------------------------------
# quaternion helpers
# ----------------------------------------------------------------------------------------------------------------------------------
SAME_POSE_RAD = 1e-5        # two driving rotations closer than this are the same sample (Blender matrices are float32: ~1e-7 noise)


def quat_canon(q):
    """Unit-normalised (float64) quaternion in the w >= 0 hemisphere."""
    q = np.asarray(q, float); q = q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-300)
    return np.where((q[..., :1] < 0), -q, q)


def quat_geodesic(qa, qb):
    """Rotation angle (radians) between quaternions, double-cover safe and accurate near 0: 2 atan2(|vec(conj(a) b)|, |w(conj(a) b)|).
    (2 acos|<a, b>| loses all precision near 0: float32-sourced unit quaternions gave ~1e-3 rad for a pose against itself.)"""
    a = quat_canon(qa); b = quat_canon(qb)
    aw, av, bw, bv = a[..., :1], a[..., 1:], b[..., :1], b[..., 1:]
    w = np.abs(aw[..., 0] * bw[..., 0] + np.sum(av * bv, axis=-1))
    vec = aw * bv - bw * av - np.cross(av, bv)
    return 2.0 * np.arctan2(np.linalg.norm(vec, axis=-1), w)


def feature_distance2(Fa, Fb):
    """Squared quaternion-space distance between pose features (..., J, 4): sum over joints of squared geodesic angles."""
    return np.sum(quat_geodesic(Fa, Fb) ** 2, axis=-1)


# ----------------------------------------------------------------------------------------------------------------------------------
# RBF interpolator
# ----------------------------------------------------------------------------------------------------------------------------------
class QuatRBF:
    """Gaussian RBF in quaternion space with cardinal targets.  samples: (S, J, 4); targets: (S, K)."""

    def __init__(self, samples, targets, sigma=None, ridge=1e-9, dedupe_tol=SAME_POSE_RAD):
        X = quat_canon(np.asarray(samples, float)); Tg = np.asarray(targets, float)
        if Tg.ndim == 1:
            Tg = Tg[:, None]
        # merge duplicate samples (identical driving rotations); their targets must agree, otherwise the data is contradictory
        keep, groups = [], []
        for s in range(len(X)):
            hit = next((g for g, k in enumerate(keep) if feature_distance2(X[s], X[k]) <= dedupe_tol ** 2), None)
            if hit is None:
                keep.append(s); groups.append([s])
            else:
                groups[hit].append(s)
        T2 = np.zeros((len(keep), Tg.shape[1])); self.conflicts = []
        for g, members in enumerate(groups):
            vals = Tg[members]
            if np.ptp(vals, axis=0).max() > 1e-9:
                self.conflicts.append(members)
            T2[g] = vals[0]                               # first member decides: sample 0 is bind, so bind always keeps weight 0
        self.X = X[keep]; self.T = T2; self.groups = groups
        D2 = feature_distance2(self.X[:, None], self.X[None, :])
        if sigma is None:
            nn = np.sqrt(np.where(np.eye(len(self.X), dtype=bool), np.inf, D2).min(axis=1)) if len(self.X) > 1 else np.array([1.0])
            sigma = float(max(np.median(nn[np.isfinite(nn)]) * 0.75, math.radians(4.0)))
        self.sigma = float(sigma); self.ridge = float(ridge)
        Phi = np.exp(-D2 / (2.0 * self.sigma ** 2))
        self.lam = np.linalg.solve(Phi + self.ridge * np.eye(len(Phi)), self.T)
        self.cond = float(np.linalg.cond(Phi + self.ridge * np.eye(len(Phi))))

    def __call__(self, feat):
        F = quat_canon(np.asarray(feat, float))
        single = F.ndim == 2
        F = F[None] if single else F
        psi = np.exp(-feature_distance2(F[:, None], self.X[None]) / (2.0 * self.sigma ** 2))     # (N, S)
        w = np.clip(psi @ self.lam, 0.0, 1.0)
        # the conditioning ridge makes cardinality approximate (~1e-9 at bind in tests): snap so bind is EXACTLY untouched
        w[w < self.SNAP] = 0.0; w[w > 1.0 - self.SNAP] = 1.0
        return w[0] if single else w

    SNAP = 1e-6

    def to_dict(self):
        return {"samples": self.X.round(12).tolist(), "lambda": self.lam.tolist(), "sigma": self.sigma, "ridge": self.ridge, "cond": self.cond}

    @classmethod
    def from_dict(cls, d):
        obj = cls.__new__(cls); obj.X = np.asarray(d["samples"], float); obj.lam = np.asarray(d["lambda"], float)
        obj.sigma = float(d["sigma"]); obj.ridge = float(d.get("ridge", 0.0)); obj.cond = d.get("cond"); obj.conflicts = []; return obj


class PSDModel:
    """families: {family: {"joints": [bone names], "keys": [shape-key names], "rbf": QuatRBF}}"""

    def __init__(self, families):
        self.families = families

    def key_weights(self, local_quats):
        """local_quats: {bone name: (w, x, y, z)} for at least every driving joint. Returns {shape key: weight in [0, 1]}."""
        out = {}
        for fam in self.families.values():
            F = np.array([local_quats[j] for j in fam["joints"]], float)
            w = fam["rbf"](F)
            out.update({k: float(v) for k, v in zip(fam["keys"], np.atleast_1d(w))})
        return out

    def to_json(self):
        return json.dumps({"schema": "maximus.v2.psd/1", "metric": "sum_j (2 acos|<q_j, q_j,s>|)^2, gaussian, cardinal",
                           "families": {n: {"joints": f["joints"], "keys": f["keys"], "rbf": f["rbf"].to_dict()} for n, f in self.families.items()}})

    @classmethod
    def from_json(cls, s):
        d = json.loads(s)
        return cls({n: {"joints": f["joints"], "keys": f["keys"], "rbf": QuatRBF.from_dict(f["rbf"])} for n, f in d["families"].items()})


# ----------------------------------------------------------------------------------------------------------------------------------
# geometry helpers
# ----------------------------------------------------------------------------------------------------------------------------------
def tri_areas(V, F):
    return 0.5 * np.linalg.norm(np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]), axis=1)


def area_ratio(V_rest, V, F, a0=None):
    a0 = tri_areas(V_rest, F) if a0 is None else a0
    return tri_areas(V, F) / np.maximum(a0, 1e-30)


def unskin(M, c):
    """Exact per-vertex inverse of the LBS blend matrix: solves M_i d_i = c_i for every vertex (M: (n,3,3), c: (n,3))."""
    M = np.asarray(M, float); c = np.asarray(c, float)
    det = np.linalg.det(M); ok = np.abs(det) > 1e-10
    d = np.zeros_like(c)
    if ok.any():
        d[ok] = np.linalg.solve(M[ok], c[ok][..., None])[..., 0]
    return d, int((~ok).sum())


def graph_distance(V, F, sources, limit=np.inf):
    """Exact multi-source shortest-path distance along mesh edges (Dijkstra, min over sources), metres; inf beyond `limit`.
    Used for region masks instead of MC-50 heat_geodesic_distance, which mis-offsets every seed cluster but the nearest one when the
    sources are far apart (measured: 0.22 m at seed vertices that must be 0, theta_fit hip-crease seeds vs. finger seeds)."""
    from scipy.sparse.csgraph import dijkstra
    E = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]); E = np.unique(np.sort(E, axis=1), axis=0)
    w = np.linalg.norm(V[E[:, 0]] - V[E[:, 1]], axis=1); n = len(V)
    G = sp.coo_matrix((np.concatenate([w, w]), (np.concatenate([E[:, 0], E[:, 1]]), np.concatenate([E[:, 1], E[:, 0]]))), shape=(n, n)).tocsr()
    return dijkstra(G, directed=False, indices=np.unique(np.asarray(sources, np.int64)), min_only=True, limit=limit)


def _uniform_laplacian(F, n):
    E = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]); E = np.unique(np.sort(E, axis=1), axis=0)
    A = sp.coo_matrix((np.ones(len(E)), (E[:, 0], E[:, 1])), shape=(n, n)); A = (A + A.T).tocsr()
    deg = np.asarray(A.sum(axis=1)).ravel()
    return A, deg


def _area_gradients(X, F, tris):
    """d area / d vertex for the listed triangles (in-plane, perpendicular to the opposite edge)."""
    A, B, C = X[F[tris, 0]], X[F[tris, 1]], X[F[tris, 2]]
    n = np.cross(B - A, C - A); ln = np.linalg.norm(n, axis=1, keepdims=True); n = n / np.maximum(ln, 1e-18)
    return 0.5 * np.cross(n, C - B), 0.5 * np.cross(n, A - C), 0.5 * np.cross(n, B - A)


def arap_corrective(K, V_rest, X_posed, F, M, seed_ratio=0.30, target_ratio=0.28, radius=0.025, arap_iters=12, repair_steps=400,
                    region_limit=None, a0=None, geo_cache=None):
    """Rest-space corrective delta for one posed mesh. K = canon_geometry_kernels module. Returns (delta (n,3), stats dict).
    region_limit: optional boolean (n,) mask; vertices outside it are never moved (e.g. to keep a correction off the egg head)."""
    V_rest = np.asarray(V_rest, float); X = np.asarray(X_posed, float); F = np.asarray(F, np.int64); n = len(V_rest)
    a0 = tri_areas(V_rest, F) if a0 is None else a0
    r = area_ratio(V_rest, X, F, a0)
    stats = {"tris_below_target_before": int((r < target_ratio).sum()), "G5_before": int((r < 0.25).sum()), "min_ratio_before": float(r.min())}
    bad = np.nonzero(r < seed_ratio)[0]
    if len(bad) == 0:
        stats.update(G5_after=stats["G5_before"], tris_below_target_after=stats["tris_below_target_before"], min_ratio_after=stats["min_ratio_before"],
                     active_vertices=0, max_rest_delta_mm=0.0, max_posed_move_mm=0.0, repair_steps=0)
        return np.zeros((n, 3)), stats
    seeds = np.unique(F[bad].ravel())
    d = graph_distance(V_rest, F, seeds, limit=radius * 1.5) if geo_cache is None else geo_cache(seeds)
    active = d < radius
    if region_limit is not None:
        active &= region_limit
    act_idx = np.nonzero(active)[0]
    Xr, energies = K.arap_pose_relaxation(V_rest, X, F, act_idx, n_iters=arap_iters)
    # residual repair restricted to the active region
    A_adj, deg = _uniform_laplacian(F, n)
    steps = 0
    for steps in range(1, repair_steps + 1):
        r = area_ratio(V_rest, Xr, F, a0); viol = np.nonzero(r < target_ratio)[0]
        if len(viol) == 0:
            steps -= 1; break
        g0, g1, g2 = _area_gradients(Xr, F, viol)
        deficit = (target_ratio * 1.02 - r[viol]) * a0[viol]
        denom = (g0 * g0).sum(1) + (g1 * g1).sum(1) + (g2 * g2).sum(1) + 1e-30
        s = (deficit / denom)[:, None]
        push = np.zeros((n, 3))
        np.add.at(push, F[viol, 0], s * g0); np.add.at(push, F[viol, 1], s * g1); np.add.at(push, F[viol, 2], s * g2)
        push[~active] = 0.0
        for _ in range(2):                                   # two Jacobi smoothing passes inside the region keep the push field smooth
            avg = (A_adj @ push) / np.maximum(deg, 1)[:, None]
            push = np.where(active[:, None], 0.5 * push + 0.5 * avg, 0.0)
        Xr = Xr + 0.8 * push
    r = area_ratio(V_rest, Xr, F, a0)
    c = Xr - X; c[~active] = 0.0
    delta, singular = unskin(M, c)
    stats.update(G5_after=int((r < 0.25).sum()), tris_below_target_after=int((r < target_ratio).sum()), min_ratio_after=float(r.min()),
                 active_vertices=int(active.sum()), max_rest_delta_mm=round(float(np.linalg.norm(delta, axis=1).max()) * 1e3, 3),
                 max_posed_move_mm=round(float(np.linalg.norm(c, axis=1).max()) * 1e3, 3), repair_steps=int(steps),
                 arap_energy_first_last=[round(energies[0], 9), round(energies[-1], 9)] if energies else None, singular_blend_matrices=singular)
    return delta, stats


def build_psd_model(sample_feats_by_family, family_keys, family_joints, key_sample_index, n_samples, sigma=None):
    """sample_feats_by_family: {family: (S, J, 4)}; family_keys: {family: [key]}; key_sample_index: {key: [sample indices where it is 1]}."""
    fams = {}
    for fam, keys in family_keys.items():
        Tg = np.zeros((n_samples, len(keys)))
        for k_i, k in enumerate(keys):
            Tg[key_sample_index[k], k_i] = 1.0
        fams[fam] = {"joints": family_joints[fam], "keys": keys, "rbf": QuatRBF(sample_feats_by_family[fam], Tg, sigma=sigma)}
    return PSDModel(fams)
