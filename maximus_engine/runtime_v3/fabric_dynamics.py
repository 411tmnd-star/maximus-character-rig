"""fabric_dynamics.py  (Maximus runtime_v3, module 3 of 8)  Secondary inertia on top of garment transport, and rigid trims.

delta_inertia: the transported garment x*(t) (garment_transport) is where the cloth would be with zero inertia. A small XPBD solve adds the
lag/overshoot of real fabric:
    predict  x <- x + dt v + dt^2 g_eff            (g_eff = gravity, optional; default 0 because x* already hangs as built)
    project  edge-length constraints (rest lengths from the bound garment), anisotropic compliance: alpha_warp for edges near the body's
             vertical axis, alpha_weft for horizontal ones (XPBD: dlambda = (-C - alpha~ lambda) / (w_i + w_j + alpha~), alpha~ = alpha / dt^2)
             attachment constraint |x - x*| -> 0 with compliance alpha_attach (soft skinning to the transported shape)
             body collision: points closer than `skin_m` to the body are pushed out along the closest-point normal
    velocity v <- (x - x_prev) / dt * (1 - damping)
    delta_inertia = x - x*  (reported per frame: max / p99 mm)
Rigid trims (buttons, rhinestones, labels): each trim vertex is bound to its nearest garment vertex's local frame (nearest-vertex rigid
attachment weight transfer), so a button follows the placket rigidly and never drifts into or off the cloth.
"""
from __future__ import annotations

import numpy as np


def edges(F):
    E = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]); return np.unique(np.sort(E, axis=1), axis=0)


class XPBDInertia:
    def __init__(self, X_rest, F, dt=1 / 24, substeps=4, iters=6, alpha_warp=2e-7, alpha_weft=8e-7, alpha_attach=4e-6, damping=0.08,
                 skin_m=0.002, gravity=(0.0, 0.0, 0.0)):
        self.F = np.asarray(F, np.int64); self.E = edges(self.F); X = np.asarray(X_rest, float)
        d = X[self.E[:, 1]] - X[self.E[:, 0]]; self.L0 = np.linalg.norm(d, axis=1)
        vert = np.abs(d[:, 2]) / np.maximum(self.L0, 1e-12)                       # 1 = vertical (warp), 0 = horizontal (weft)
        self.alpha = alpha_weft + (alpha_warp - alpha_weft) * vert
        self.dt, self.sub, self.iters, self.a_att, self.damp, self.skin = dt, substeps, iters, alpha_attach, damping, skin_m
        self.g = np.asarray(gravity, float); self.x = X.copy(); self.v = np.zeros_like(X)

    def step(self, x_target, body=None):
        """Advance one frame toward the transported target x_target. body: optional (closest_point_fn) -> (cp, normal) for collision."""
        h = self.dt / self.sub; xt = np.asarray(x_target, float)
        for _ in range(self.sub):
            xp = self.x.copy(); self.x = self.x + h * self.v + h * h * self.g
            lam = np.zeros(len(self.E)); at = self.a_att / (h * h)
            for _ in range(self.iters):
                i, j = self.E[:, 0], self.E[:, 1]; d = self.x[j] - self.x[i]; L = np.linalg.norm(d, axis=1); n = d / np.maximum(L, 1e-12)[:, None]
                C = L - self.L0; al = self.alpha / (h * h); dl = (-C - al * lam) / (2.0 + al); lam += dl
                corr = n * dl[:, None]; np.add.at(self.x, i, -corr); np.add.at(self.x, j, corr)
                self.x += (xt - self.x) * (1.0 / (1.0 + at))                        # attachment (unit mass, XPBD compliance)
            if body is not None:
                cp, nrm, signed = body(self.x)
                bad = signed < self.skin
                self.x[bad] = cp[bad] + nrm[bad] * self.skin
            self.v = (self.x - xp) / h * (1.0 - self.damp)
        dlt = np.linalg.norm(self.x - xt, axis=1)
        return self.x.copy(), {"delta_inertia_max_mm": round(float(dlt.max()) * 1e3, 3), "delta_inertia_p99_mm": round(float(np.percentile(dlt, 99)) * 1e3, 3)}


def bind_trims(trim_V, cloth_V, cloth_F):
    """Nearest-vertex rigid attachment: each trim vertex stores its offset in the local frame of its nearest cloth vertex
    (frame = vertex normal + first incident edge). Returns a binding for transport_trims."""
    from scipy.spatial import cKDTree
    T = np.asarray(trim_V, float); X = np.asarray(cloth_V, float); F = np.asarray(cloth_F, np.int64)
    _, nn = cKDTree(X).query(T); fr = vertex_frames(X, F)
    off = np.einsum("nba,nb->na", fr[nn], T - X[nn])
    return {"anchor": nn, "offset": off}


def vertex_frames(X, F):
    fn = np.cross(X[F[:, 1]] - X[F[:, 0]], X[F[:, 2]] - X[F[:, 0]]); vn = np.zeros_like(X)
    for c in range(3):
        np.add.at(vn, F[:, c], fn)
    vn /= np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-12)
    first = np.full(len(X), -1); first[F[:, 0]] = F[:, 1]; first[F[:, 1]] = F[:, 2]; first[F[:, 2]] = F[:, 0]
    e = X[np.where(first >= 0, first, np.arange(len(X)))] - X; e -= vn * np.einsum("ij,ij->i", e, vn)[:, None]
    e /= np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-12); b = np.cross(vn, e)
    return np.stack([vn, e, b], axis=2)


def transport_trims(binding, cloth_V, cloth_F):
    X = np.asarray(cloth_V, float); fr = vertex_frames(X, np.asarray(cloth_F, np.int64))
    a = binding["anchor"]; return X[a] + np.einsum("nab,nb->na", fr[a], binding["offset"])
