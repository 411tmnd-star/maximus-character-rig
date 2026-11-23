"""Faceless sculptural egg head, fitted IN PLACE (vertex count, order, UVs, weights and shape-key deltas untouched).

Positions only change. Each vertex has a blend weight w in [0, 1] (its total skin weight to the head and face bones):
w = 1 vertices end exactly on the egg surface; w = 0 vertices (neck and below) do not move; the band in between is
optimised so the head joins the neck smoothly.

Optimisation (gradient descent, NumPy only), following the two-phase schedule of the Gaussian Garments paper:
  E = fit (pull to egg, weight w^2) + anchor (stay at rest, weight (1-w)^2) + fairing (uniform umbrella, weight w)
      + first half:  L_VE   on virtual edges across the head (template = egg span), so fairing cannot implode it
      + second half: L_body vs the egg surface shrunk by eps (one-sided cubic), so no vertex stays inside the egg
Ears, nose, lips and eyelids are removed by fairing + projection: their vertices spread over the egg surface.
Radial projection alone leaves those features folded (a pocket covers the same egg area two or three times), so the
full-weight patch is then re-embedded injectively by unfold(): a Tutte / harmonic map in a stereographic chart of the
egg, followed by a flip-guarded tangential relaxation. No triangle of the patch faces inward afterwards.
"""
import numpy as np

from .clearance import build_virtual_edges, EPS_BODY
from .geometry import as_f64, face_normals, tri_areas, umbrella, unique_edges


def estimate_egg(V, w, tilt_deg=12.0, taper=0.08):
    """Egg parameters from landmarks of the rest head (metres). JSON-safe dict.

    centre y : midway between the back of the skull and the forehead (nose and brows excluded)
    centre z : midway between crown and chin;  c = half that height along the tilted axis
    a        : skull half-width above the ears;  b = half the skull depth;  tilt: crown back, chin forward
    """
    V = as_f64(V); H = V[w > 0.99]
    z_top = float(H[:, 2].max()); z_chin = float(H[:, 2].min()); z_mid = 0.5 * (z_top + z_chin)
    y_back = float(H[:, 1].max())
    fore = H[(H[:, 2] > z_mid + 0.035) & (H[:, 2] < z_mid + 0.075)]
    y_fore = float(fore[:, 1].min())
    upper = H[H[:, 2] > z_mid + 0.030]
    a = float(np.percentile(np.abs(upper[:, 0]), 99.5))
    t = np.radians(tilt_deg)
    u = np.array([0.0, np.sin(t), np.cos(t)])             # long axis: crown leans back (+Y)
    v = np.array([0.0, -np.cos(t), np.sin(t)])            # forward (model faces -Y)
    centre = np.array([0.0, 0.5 * (y_back + y_fore), z_mid])
    c = 0.5 * (z_top - z_chin) / np.cos(t)
    b = 0.5 * (y_back - y_fore)
    return {"centre": centre.tolist(), "u": u.tolist(), "v": v.tolist(), "a": a, "b": b, "c": float(c), "taper": taper,
            "tilt_deg": tilt_deg, "landmarks": {"z_top": z_top, "z_chin": z_chin, "y_back": y_back, "y_forehead": y_fore}}


def _local(egg, X):
    c = np.asarray(egg["centre"]); u = np.asarray(egg["u"]); v = np.asarray(egg["v"])
    R = X - c
    return R[:, 0], R @ v, R @ u, c, u, v   # lateral, forward, along-axis


def _norm(egg, l, f, s):
    g = np.clip(1.0 + egg["taper"] * s / egg["c"], 0.2, 2.0)
    return (l / (egg["a"] * g)) ** 2 + (f / (egg["b"] * g)) ** 2 + (s / egg["c"]) ** 2


def project(egg, X, shrink=0.0):
    """Radial projection from the egg centre onto the egg surface (optionally an inner shell `shrink` metres in)."""
    X = as_f64(X); l, f, s, c, u, v = _local(egg, X)
    r = np.linalg.norm(X - c, axis=1)
    hi = 3.0 * max(egg["a"], egg["b"], egg["c"]) / np.maximum(r, 1e-12)   # beyond the surface for every direction
    lo = np.zeros(len(X))
    for _ in range(64):                                   # bisection on the radial scale t
        t = 0.5 * (lo + hi); inside = _norm(egg, t * l, t * f, t * s) < 1.0
        lo = np.where(inside, t, lo); hi = np.where(inside, hi, t)
    t = 0.5 * (lo + hi)
    P = c + (t[:, None] * (X - c))
    if shrink:
        d = X - c; n = np.linalg.norm(d, axis=1, keepdims=True); P = P - shrink * d / np.maximum(n, 1e-12)
    return P


def surface_normals(egg, X, h=1e-6):
    """Outward unit normals of the egg implicit surface at X (gradient of the egg norm, central differences).
    Used as custom shading normals for egg vertices, so folded-away flap triangles cannot bend the shading."""
    X = as_f64(X); l, f, s, c, u, v = _local(egg, X)
    dl = (_norm(egg, l + h, f, s) - _norm(egg, l - h, f, s)) / (2 * h)
    df = (_norm(egg, l, f + h, s) - _norm(egg, l, f - h, s)) / (2 * h)
    ds = (_norm(egg, l, f, s + h) - _norm(egg, l, f, s - h)) / (2 * h)
    G = dl[:, None] * np.array([1.0, 0.0, 0.0]) + df[:, None] * v + ds[:, None] * u
    return G / np.maximum(np.linalg.norm(G, axis=1, keepdims=True), 1e-15)


def signed_distance(egg, X):
    """Approximate signed distance to the egg surface along the radial direction (+ outside)."""
    P = project(egg, X); c = np.asarray(egg["centre"])
    return np.linalg.norm(X - c, axis=1) - np.linalg.norm(P - c, axis=1)


def sink_folds(X, F, w, egg, depth=0.0015, rounds=3):
    """Folded-away flaps (ear backs, lip and eyelid interiors) end up coincident with the outer egg skin and z-fight
    in ray tracing. Vertices whose every incident head triangle faces inward are moved `depth` below the egg surface
    (along the egg normal), so the outer skin always wins. Positions only; topology untouched."""
    X = X.copy(); F = np.asarray(F, np.int64); head = np.all(w[F] > 0.99, axis=1); sunk = np.zeros(len(X), bool)
    for _ in range(rounds):
        n = face_normals(X, F); rad = X[F].mean(1) - np.asarray(egg["centre"])
        inward = head & (np.einsum("ij,ij->i", n, rad) < 0)
        tot = np.zeros(len(X)); bad = np.zeros(len(X))
        np.add.at(tot, F[head].ravel(), 1); np.add.at(bad, F[inward].ravel(), 1)
        sel = (tot > 0) & (bad == tot) & ~sunk & (w > 0.99)
        if not sel.any():
            break
        X[sel] = project(egg, X[sel]) - depth * surface_normals(egg, X[sel]); sunk |= sel
    return X, sunk


def _chart_frame(pole):
    """Unit pole p and an orthonormal pair (e1, e2) spanning the plane perpendicular to it."""
    p = np.asarray(pole, float); p = p / np.linalg.norm(p)
    h = np.array([1.0, 0.0, 0.0]) if abs(p[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(p, h); e1 /= np.linalg.norm(e1)
    return p, e1, np.cross(p, e1)


def to_chart(egg, X, pole):
    """Stereographic chart of the egg as seen from its centre. d = unit direction of (X - centre); the chart is the
    stereographic projection of d from `pole` (a direction that must lie outside the charted region):
        (u, v) = (d.e1, d.e2) / (1 - d.p)
    The egg is star-shaped about its centre, so direction <-> surface point is one-to-one and the chart is a
    homeomorphism of the egg surface minus the pole direction onto the plane."""
    D = as_f64(X) - np.asarray(egg["centre"]); D = D / np.maximum(np.linalg.norm(D, axis=1, keepdims=True), 1e-15)
    p, e1, e2 = _chart_frame(pole); den = np.maximum(1.0 - D @ p, 1e-12)
    return np.stack([(D @ e1) / den, (D @ e2) / den], axis=1)


def from_chart(egg, UV, pole):
    """Inverse of to_chart: d = (2u e1 + 2v e2 + (r^2 - 1) p) / (r^2 + 1), then radial projection onto the egg."""
    UV = as_f64(UV); p, e1, e2 = _chart_frame(pole); r2 = (UV ** 2).sum(1)
    D = (2 * UV[:, :1] * e1 + 2 * UV[:, 1:] * e2 + (r2 - 1)[:, None] * p) / (r2 + 1)[:, None]
    return project(egg, np.asarray(egg["centre"]) + max(egg["a"], egg["b"], egg["c"]) * D)


def _inward(X, F, tris, centre):
    """Mask over F: triangles of `tris` whose geometric normal does not point away from the egg centre."""
    n = face_normals(X, F, normalize=False); rad = X[F].mean(1) - centre
    return tris & (np.einsum("ij,ij->i", n, rad) <= 0)


def unfold(X, F, w, egg, relax_iters=30, relax_step=0.5, island_depth=0.004):
    """Fold-free re-embedding of the full-weight head patch on the egg (positions only; topology, vertex order, UVs
    and weights are not touched). Replaces the old 'fold the flaps away and sink them' displacement.

    Radial projection maps every pocket (mouth, nostril, eye socket) and fin (ear) onto the same piece of egg two or
    three times over, so the mesh covers the egg non-injectively: the extra layers are the inward-facing 'flap'
    triangles. Here the patch is given an injective parametrisation instead:

      1. fixed set  = full-weight vertices that touch a non-full vertex (the neck ring; they keep their position, so
                      the blend band below does not notice anything)
         free set   = every other full-weight vertex connected to that ring
      2. chart      = stereographic projection of centre directions from the middle of the neck hole (to_chart)
      3. Tutte / discrete-harmonic embedding in the chart: each free vertex is the mean of its neighbours,
                      L_ff u_f = -L_fb u_b   (uniform weights, dense solve, one right-hand side per chart axis).
                      A convex-combination map of a disc-topology patch is an embedding when the boundary polygon is
                      convex (Tutte 1963, Floater 1997) and, for a non-convex boundary, when every reflex boundary
                      vertex lies in the convex hull of its neighbours (Gortler, Gotsman, Thurston 2006). The result
                      is checked, not assumed: 'inward_after_harmonic' must be 0.
      4. lift       = inverse chart + radial projection, so every vertex is exactly on the egg
      5. relaxation = tangential umbrella steps re-projected onto the egg. A step is rolled back on every vertex of
                      any triangle it would turn inward, so the embedding of step 3 is an invariant of the loop.

    Full-weight components with no path to the neck ring (closed inner shells, if the mesh has any) cannot be
    embedded; they are moved `island_depth` under the skin and reported.
    Returns (X_new, report, island_vertex_mask)."""
    X = as_f64(X).copy(); F = np.asarray(F, np.int64); n = len(X); w = np.asarray(w, float)
    c = np.asarray(egg["centre"], float); full = w > 0.99
    E = unique_edges(F); e_full = full[E[:, 0]] & full[E[:, 1]]; e_mixed = full[E[:, 0]] ^ full[E[:, 1]]
    fixed = np.zeros(n, bool); fixed[E[e_mixed].ravel()] = True; fixed &= full
    Eh = E[e_full]
    lab = np.arange(n)                                    # connected components of the full-weight graph
    for _ in range(n):
        new = lab.copy()
        np.minimum.at(new, Eh[:, 0], lab[Eh[:, 1]]); np.minimum.at(new, Eh[:, 1], lab[Eh[:, 0]])
        if np.array_equal(new, lab):
            break
        lab = new
    anchored = np.zeros(n, bool); anchored[lab[fixed]] = True
    island = full & ~fixed & ~anchored[lab]
    free = full & ~fixed & ~island
    head = np.all(full[F], axis=1); patch = head & ~np.any(island[F], axis=1)
    rep = {"method": "tutte_stereographic+guarded_tangential_relaxation", "fixed_ring_vertices": int(fixed.sum()),
           "free_vertices": int(free.sum()), "island_vertices": int(island.sum()),
           "inward_before": int(_inward(X, F, patch, c).sum())}
    if island.any():
        X[island] = project(egg, X[island], shrink=island_depth)
    if not fixed.any() or not free.any():
        rep.update({"inward_after_harmonic": rep["inward_before"], "inward_after": rep["inward_before"], "relax_iters_run": 0})
        return X, rep, island
    D = X[fixed] - c; D /= np.maximum(np.linalg.norm(D, axis=1, keepdims=True), 1e-15)
    pole = D.mean(0)
    pole = pole / np.linalg.norm(pole) if np.linalg.norm(pole) > 1e-6 else -np.asarray(egg["u"], float)
    UV = to_chart(egg, X, pole)
    idx = np.nonzero(free)[0]; loc = np.full(n, -1, np.int64); loc[idx] = np.arange(len(idx)); nf = len(idx)
    A = np.zeros((nf, nf)); B = np.zeros((nf, 2))
    for p_, q_ in ((Eh[:, 0], Eh[:, 1]), (Eh[:, 1], Eh[:, 0])):
        m = free[p_] & free[q_]
        np.add.at(A, (loc[p_[m]], loc[q_[m]]), -1.0); np.add.at(A, (loc[p_[m]], loc[p_[m]]), 1.0)
        m = free[p_] & fixed[q_]
        np.add.at(A, (loc[p_[m]], loc[p_[m]]), 1.0); np.add.at(B, loc[p_[m]], UV[q_[m]])
    X[idx] = from_chart(egg, np.linalg.solve(A, B), pole)
    del A
    rep["chart_pole"] = pole.tolist(); rep["chart_boundary_radius_min_max"] = [float(np.linalg.norm(UV[fixed], axis=1).min()),
                                                                               float(np.linalg.norm(UV[fixed], axis=1).max())]
    rep["inward_after_harmonic"] = int(_inward(X, F, patch, c).sum())
    done = 0
    if rep["inward_after_harmonic"] == 0:
        for _ in range(relax_iters):
            Y = X.copy(); Y[idx] = project(egg, X[idx] + relax_step * umbrella(X, Eh, n)[idx])
            ok = False
            for _ in range(8):                            # roll back the vertices of any triangle the step turned inward
                bad = _inward(Y, F, patch, c)
                if not bad.any():
                    ok = True; break
                v = np.unique(F[bad]); v = v[free[v]]; Y[v] = X[v]
            if not ok:
                break
            X = Y; done += 1
    rep["relax_iters_run"] = done; rep["inward_after"] = int(_inward(X, F, patch, c).sum())
    a = tri_areas(X, F)[patch]
    rep["patch_tri_area_mm2_min_median_max"] = [float(a.min() * 1e6), float(np.median(a) * 1e6), float(a.max() * 1e6)]
    return X, rep, island


def normal_blend(F, w, rings=4):
    """Per-vertex factor in [0, 1] for mixing the analytic egg normal into the mesh normal: 0 on non-full vertices and
    on the full-weight ring that touches them, rising with a smoothstep over `rings` edge hops to 1 inside the head.
    Shading normals then pass from the neck's mesh normals to the egg's analytic normals without a step."""
    F = np.asarray(F, np.int64); w = np.asarray(w, float); n = len(w); E = unique_edges(F)
    seen = ~(w > 0.99); d = np.zeros(n)
    for k in range(1, rings + 1):
        nb = np.zeros(n, bool); nb[E[seen[E[:, 0]], 1]] = True; nb[E[seen[E[:, 1]], 0]] = True
        new = nb & ~seen; d[new] = k; seen |= new
    d[~seen] = rings
    t = np.clip((d - 1.0) / max(rings - 1.0, 1.0), 0.0, 1.0)
    return np.where(w > 0.99, t * t * (3 - 2 * t), 0.0)


def fit(V, F, w, egg, iters=600, step=0.25, w_fit=1.0, w_anchor=1.0, w_fair=0.6, w_ve=1.0, w_body=2e3, eps=EPS_BODY,
        max_move=0.002, fold_depth=0.0015, unfold_patch=True, relax_iters=30):
    """Return (X, report). V: rest positions (n,3); F: triangles; w: per-vertex blend weight."""
    V = as_f64(V); F = np.asarray(F, np.int64); w = np.clip(np.asarray(w, float), 0, 1)
    X = V.copy(); act = w > 1e-4
    E = unique_edges(F); E = E[act[E[:, 0]] | act[E[:, 1]]]
    head_faces = np.nonzero(np.all(w[F] > 0.99, axis=1))[0]
    VE = build_virtual_edges(V, F, faces=head_faces, max_len=0.30)
    VE = VE[np.all(w[VE] > 0.99, axis=1)] if len(VE) else VE
    P0 = project(egg, V)
    L_t = np.linalg.norm(P0[VE[:, 0]] - P0[VE[:, 1]], axis=1) if len(VE) else np.zeros(0)
    hist = []
    for it in range(iters):
        P = project(egg, X)
        G = w_fit * (w ** 2)[:, None] * (X - P) + w_anchor * ((1 - w) ** 2)[:, None] * (X - V)
        G -= w_fair * w[:, None] * umbrella(X, E, len(X))
        phase = "L_VE" if it < iters // 2 else "L_body"
        if phase == "L_VE" and len(VE):
            d = X[VE[:, 0]] - X[VE[:, 1]]; l = np.linalg.norm(d, axis=1); sh = np.maximum(L_t - l, 0)
            g = (-2 * sh / np.maximum(l, 1e-12))[:, None] * d
            Gve = np.zeros_like(X); np.add.at(Gve, VE[:, 0], g); np.add.at(Gve, VE[:, 1], -g)
            G += w_ve * Gve; e_extra = float((sh ** 2).sum())
        else:
            sd = signed_distance(egg, X) + eps          # distance to the inner shell (egg shrunk by eps)
            viol = np.maximum(eps - sd, 0) * (w > 0.5)
            c = np.asarray(egg["centre"]); rd = X - c; rd /= np.maximum(np.linalg.norm(rd, axis=1, keepdims=True), 1e-12)
            G += w_body * (-3 * viol ** 2)[:, None] * rd; e_extra = float((viol ** 3).sum())
        D = step * G; n = np.linalg.norm(D, axis=1, keepdims=True)
        D *= np.minimum(1.0, max_move / np.maximum(n, 1e-15))        # bounded step (stability)
        X[act] -= D[act]
        if it % 50 == 0 or it == iters - 1:
            Pn = project(egg, X)
            hist.append({"iter": it, "phase": phase, "fit_rms_mm": float(np.sqrt(((w > 0.99) * ((X - Pn) ** 2).sum(1)).sum() / max((w > 0.99).sum(), 1)) * 1000),
                         "extra_energy": e_extra})
    full = w > 0.99
    X[full] = project(egg, X[full])                       # w = 1 vertices exactly on the egg
    sunk = np.zeros(len(X), bool); urep = None; island = np.zeros(len(X), bool)
    if unfold_patch:                                      # injective re-embedding: no flap is left to hide
        X, urep, island = unfold(X, F, w, egg, relax_iters=relax_iters)
    if urep is None or urep["inward_after"] > 0:          # legacy fallback only where a fold survived
        X, sunk = sink_folds(X, F, w, egg, depth=fold_depth)
    X[~act] = V[~act]                                     # untouched region bit-exact
    n_out = face_normals(X, F); cen = X[F].mean(1); rad = cen - np.asarray(egg["centre"])
    head_tris = np.all(w[F] > 0.99, axis=1) & ~np.any(island[F], axis=1)
    inward = head_tris & (np.einsum("ij,ij->i", n_out, rad) < 0)
    a1 = tri_areas(X, F)
    rep = {"egg": egg, "iters": iters, "virtual_edges": int(len(VE)), "moved_vertices": int(act.sum()), "unfold": urep,
           "full_weight_vertices": int(full.sum()), "head_tris": int(head_tris.sum()),
           "head_tris_facing_inward": int(inward.sum()), "head_tris_area_lt_1e-9": int((head_tris & (a1 < 1e-9)).sum()),
           "max_move_mm": float(np.linalg.norm(X - V, axis=1).max() * 1000), "fold_vertices_sunk": int(sunk.sum()), "fold_depth_mm": fold_depth * 1000,
           "unmoved_vertices_bit_exact": bool(np.array_equal(X[~act], V[~act])), "history": hist}
    return X, rep
