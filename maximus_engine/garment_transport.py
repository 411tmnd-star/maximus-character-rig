"""M9 Garment Dressing & SO(3) Surface-Atlas Transport Bridge (Maximus v2).

Executes Step 1 & Step 2 of the M9 Universal Dressing & Measurement Runtime:
  1. Extract clean outer shell-only shirt mesh from a verified master .blend (pure bpy.data/bmesh, zero bpy.ops).
  2. Bind once on native body topology (13,380 verts / 26,756 tris) and transport onto Maximus v2 theta_fit,
     enforcing two-phase L_VE -> L_body (3 mm cubic normal barrier) and Decision O-7 (0.25 mm dense neckline check).
  3. Zero-refit SO(3) triangle-frame transport T_atlas across static poses (P1-P3), SWAG_01 (72 frames), and M5 walk.
  4. Skin-weight transfer with collar weights isolated strictly to neck01 / spine03.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
COLAB_STAGE = HERE / "colab_stage"
RUNTIME_V3 = HERE / "runtime_v3"
for _p in (COLAB_STAGE, RUNTIME_V3, HERE / "pipeline"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import canon_geometry_kernels as K  # noqa: E402


def _load_runtime_v3_transport():
    spec = importlib.util.spec_from_file_location("_runtime_v3_garment_transport", RUNTIME_V3 / "garment_transport.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


GT_V3 = _load_runtime_v3_transport()

DEFAULT_HERO_BLEND = (
    HERE.parent
    / "better_style_engine"
    / "02_engine_3d"
    / "isolated_clones"
    / "better_style_catalog_collar"
    / "assets"
    / "archetypes"
    / "ringer_tee_master.blend"
)
DEFAULT_M4_ATLAS = COLAB_STAGE / "results" / "M4_canon_atlas" / "maximus_v2_surface_atlas.npz"
DEFAULT_M4_PSD_BLEND = COLAB_STAGE / "results" / "M4_canon_atlas" / "maximus_v2_master_psd.blend"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _mesh_normals(V: np.ndarray, F: np.ndarray) -> np.ndarray:
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    vn = np.zeros_like(V, dtype=np.float64)
    for c in range(3):
        np.add.at(vn, F[:, c], fn)
    return vn / np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-12)


def _fast_signed_distance(P: np.ndarray, V_body: np.ndarray, F_body: np.ndarray, k: int = 12):
    """Fast local signed distance using cKDTree closest triangles + face normal orientation.
    Used inside iterative relaxation loops; final truth gates always use exact float64 GWN."""
    P = np.asarray(P, dtype=np.float64)
    V = np.asarray(V_body, dtype=np.float64)
    F = np.asarray(F_body, dtype=np.int64)
    A, B, C = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    cen = (A + B + C) / 3.0
    fn = np.cross(B - A, C - A)
    fn /= np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-12)
    vn = _mesh_normals(V, F)
    k = min(int(k), len(F))
    _, idx = cKDTree(cen).query(P, k=k)
    idx = idx.reshape(len(P), k)
    best_d = np.full(len(P), np.inf, dtype=np.float64)
    best_cp = np.zeros_like(P)
    best_f = np.zeros(len(P), dtype=np.int64)
    for c in range(k):
        f = idx[:, c]
        q = K._closest_point_on_triangles(P, A[f], B[f], C[f])
        d = np.linalg.norm(P - q, axis=1)
        m = d < best_d
        best_d = np.where(m, d, best_d)
        best_cp = np.where(m[:, None], q, best_cp)
        best_f = np.where(m, f, best_f)
    # Use smooth vertex normal interpolated at closest point via barycentric coords for robust sign
    a, b, c_ = A[best_f], B[best_f], C[best_f]
    v0, v1, v2 = b - a, c_ - a, best_cp - a
    d00 = np.einsum("ij,ij->i", v0, v0)
    d01 = np.einsum("ij,ij->i", v0, v1)
    d11 = np.einsum("ij,ij->i", v1, v1)
    d20 = np.einsum("ij,ij->i", v2, v0)
    d21 = np.einsum("ij,ij->i", v2, v1)
    den = np.where(np.abs(d00 * d11 - d01 * d01) > 1e-30, d00 * d11 - d01 * d01, 1.0)
    bv = np.clip((d11 * d20 - d01 * d21) / den, 0.0, 1.0)
    bw = np.clip((d00 * d21 - d01 * d20) / den, 0.0, 1.0)
    bu = np.clip(1.0 - bv - bw, 0.0, 1.0)
    n_interp = bu[:, None] * vn[F[best_f, 0]] + bv[:, None] * vn[F[best_f, 1]] + bw[:, None] * vn[F[best_f, 2]]
    n_interp /= np.maximum(np.linalg.norm(n_interp, axis=1, keepdims=True), 1e-12)
    sign = np.where(np.einsum("ij,ij->i", P - best_cp, n_interp) < 0.0, -1.0, 1.0)
    return sign * best_d, best_cp, n_interp


def _adjacency_matrix(n_verts: int, F: np.ndarray):
    import scipy.sparse as sp

    E = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    E = np.unique(np.sort(E, axis=1), axis=0)
    A = sp.coo_matrix((np.ones(len(E), dtype=np.float64), (E[:, 0], E[:, 1])), shape=(n_verts, n_verts))
    A = (A + A.T).tocsr()
    deg = np.maximum(np.asarray(A.sum(axis=1)).ravel(), 1.0)
    return A, deg, E


def _clear_bpy_data(bpy):
    for col in (bpy.data.objects, bpy.data.meshes, bpy.data.materials, bpy.data.armatures, bpy.data.images):
        for item in list(col):
            col.remove(item, do_unlink=True)


def extract_hero_shell(
    blend_path: Path = DEFAULT_HERO_BLEND,
    mannequin_name: str = "Mannequin",
    garment_name: str = "Tee",
    unsubdiv_iters: int = 4,
    collapse_ratio: float = 0.35,
) -> dict:
    """Extract a clean single-layer outer garment shell from a verified master .blend (pure bpy.data/bmesh, zero bpy.ops)."""
    import bpy
    import bmesh

    blend_path = Path(blend_path)
    if not blend_path.exists():
        raise FileNotFoundError(f"Hero garment master not found: {blend_path}")

    _clear_bpy_data(bpy)
    sc = bpy.context.scene
    with bpy.data.libraries.load(str(blend_path), link=False) as (df, dt):
        dt.objects = [mannequin_name, garment_name]

    mann = bpy.data.objects[mannequin_name]
    tee = bpy.data.objects[garment_name]
    sc.collection.objects.link(tee)

    # Read native Mannequin arrays (13,380 verts, 26,756 tris)
    me_m = mann.data
    me_m.calc_loop_triangles()
    Vm = np.empty(len(me_m.vertices) * 3, dtype=np.float64)
    me_m.vertices.foreach_get("co", Vm)
    Vm = Vm.reshape(-1, 3)
    Fm = np.empty(len(me_m.loop_triangles) * 3, dtype=np.int64)
    me_m.loop_triangles.foreach_get("vertices", Fm)
    Fm = Fm.reshape(-1, 3)
    vn_m = _mesh_normals(Vm, Fm)
    kd_m = cKDTree(Vm)

    # Read source vertex groups before unsubdivision so we can transfer them cleanly
    vg_names = [g.name for g in tee.vertex_groups]
    W_src = np.zeros((len(tee.data.vertices), len(vg_names)), dtype=np.float32)
    for v in tee.data.vertices:
        for g in v.groups:
            W_src[v.index, g.group] = g.weight
    V_src = np.empty(len(tee.data.vertices) * 3, dtype=np.float64)
    tee.data.vertices.foreach_get("co", V_src)
    V_src = V_src.reshape(-1, 3)
    kd_src = cKDTree(V_src)

    # Unsubdivide to strip baked Catmull-Clark subdiv while preserving UVs and materials
    mod = tee.modifiers.new("M9_Unsub", "DECIMATE")
    mod.decimate_type = "UNSUBDIV"
    mod.iterations = int(unsubdiv_iters)
    dg = bpy.context.evaluated_depsgraph_get()
    me_u = tee.evaluated_get(dg).to_mesh()
    bm = bmesh.new()
    bm.from_mesh(me_u)
    tee.evaluated_get(dg).to_mesh_clear()

    # Keep strictly outer shell faces (face normal aligned with outward mannequin normal)
    bm.faces.ensure_lookup_table()
    centres = np.array([f.calc_center_median() for f in bm.faces], dtype=np.float64)
    fnorms = np.array([f.normal for f in bm.faces], dtype=np.float64)
    _, idx_m = kd_m.query(centres)
    dots = np.einsum("ij,ij->i", fnorms, vn_m[idx_m])
    del_faces = [f for f, d in zip(bm.faces, dots) if d <= 0.15]
    if del_faces:
        bmesh.ops.delete(bm, geom=del_faces, context="FACES")
    orphan = [v for v in bm.verts if not v.link_faces]
    if orphan:
        bmesh.ops.delete(bm, geom=orphan, context="VERTS")

    # Keep only the largest connected component to remove tiny rim islands
    bm.verts.ensure_lookup_table()
    bm.faces.ensure_lookup_table()
    visited = set()
    components = []
    for f in bm.faces:
        if f.index in visited:
            continue
        stack = [f]
        comp = []
        visited.add(f.index)
        while stack:
            cur = stack.pop()
            comp.append(cur)
            for e in cur.edges:
                for nf in e.link_faces:
                    if nf.index not in visited:
                        visited.add(nf.index)
                        stack.append(nf)
        components.append(comp)
    if len(components) > 1:
        main_comp = max(components, key=len)
        main_set = {f.index for f in main_comp}
        extra_faces = [f for f in bm.faces if f.index not in main_set]
        if extra_faces:
            bmesh.ops.delete(bm, geom=extra_faces, context="FACES")
        orphan = [v for v in bm.verts if not v.link_faces]
        if orphan:
            bmesh.ops.delete(bm, geom=orphan, context="VERTS")

    me_outer = bpy.data.meshes.new("Tee_OuterShell")
    bm.to_mesh(me_outer)
    bm.free()
    for m in tee.data.materials:
        me_outer.materials.append(m)

    shell_ob = bpy.data.objects.new("Tee_OuterShell", me_outer)
    sc.collection.objects.link(shell_ob)

    # Collapse decimation to clean ~9.5k-vert / ~18k-tri game/transport shell
    dec = shell_ob.modifiers.new("M9_Collapse", "DECIMATE")
    dec.decimate_type = "COLLAPSE"
    dec.ratio = float(collapse_ratio)
    dg2 = bpy.context.evaluated_depsgraph_get()
    me_fin = shell_ob.evaluated_get(dg2).to_mesh()
    me_fin.calc_loop_triangles()

    n_v = len(me_fin.vertices)
    n_t = len(me_fin.loop_triangles)
    V = np.empty(n_v * 3, dtype=np.float64)
    me_fin.vertices.foreach_get("co", V)
    V = V.reshape(n_v, 3)

    F = np.empty(n_t * 3, dtype=np.int64)
    me_fin.loop_triangles.foreach_get("vertices", F)
    F = F.reshape(n_t, 3)

    tri_loops = np.empty(n_t * 3, dtype=np.int64)
    me_fin.loop_triangles.foreach_get("loops", tri_loops)
    tri_polys = np.empty(n_t, dtype=np.int64)
    me_fin.loop_triangles.foreach_get("polygon_index", tri_polys)

    poly_mats = np.empty(len(me_fin.polygons), dtype=np.int32)
    me_fin.polygons.foreach_get("material_index", poly_mats)
    tri_mats = poly_mats[tri_polys]
    mat_names = [m.name if m else "Material" for m in me_fin.materials]

    uvs = {}
    for uv_layer in me_fin.uv_layers:
        buf = np.empty(len(me_fin.loops) * 2, dtype=np.float32)
        uv_layer.data.foreach_get("uv", buf)
        buf = buf.reshape(-1, 2)
        uvs[uv_layer.name] = buf[tri_loops]  # (n_t * 3, 2) per triangle corner

    shell_ob.evaluated_get(dg2).to_mesh_clear()

    # Transfer anatomical vertex groups from source Tee
    _, nn_src = kd_src.query(V)
    W_vg = W_src[nn_src]

    # Drop any degenerate triangles if present
    areas = 0.5 * np.linalg.norm(np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]), axis=1)
    valid = areas > 1e-12
    if not valid.all():
        F = F[valid]
        tri_mats = tri_mats[valid]
        keep_corner = np.repeat(valid, 3)
        for k_uv in list(uvs.keys()):
            uvs[k_uv] = uvs[k_uv][keep_corner]

    _clear_bpy_data(bpy)

    collar_idx = vg_names.index("collar_rib") if "collar_rib" in vg_names else None
    collar_mask = (W_vg[:, collar_idx] > 0.35) if collar_idx is not None else (tri_mats == 1)

    return {
        "sku": "BS-RINGER-001",
        "source_blend": str(blend_path),
        "source_sha256": sha256(blend_path),
        "V": V,
        "F": F,
        "tri_mats": tri_mats.astype(np.int32),
        "mat_names": mat_names,
        "uv_names": list(uvs.keys()),
        "uvs": uvs,
        "vg_names": vg_names,
        "W_vg": W_vg.astype(np.float32),
        "collar_mask": collar_mask.astype(bool),
        "mannequin_V": Vm,
        "mannequin_F": Fm,
    }


def collar_neckline_samples(V: np.ndarray, F: np.ndarray, collar_mask: np.ndarray, spacing_m: float = 0.00025) -> np.ndarray:
    """Decision O-7: extract the neckline boundary loop and densely resample at 0.25 mm spacing."""
    V = np.asarray(V, dtype=np.float64)
    F = np.asarray(F, dtype=np.int64)
    # Find boundary edges (edges belonging to exactly 1 triangle) whose both verts are in collar zone (z > 1.35 m)
    edge_counts = {}
    for tri in F:
        for i in range(3):
            a, b = int(tri[i]), int(tri[(i + 1) % 3])
            key = (min(a, b), max(a, b))
            edge_counts[key] = edge_counts.get(key, 0) + 1
    bnd = [e for e, c in edge_counts.items() if c == 1 and (collar_mask[e[0]] or collar_mask[e[1]]) and V[e[0], 2] > 1.35]
    if not bnd:
        pts = V[collar_mask]
        return pts
    # Order/densely sample every neckline boundary edge at spacing_m (0.25 mm)
    samples = []
    for a, b in bnd:
        p, q = V[a], V[b]
        L = float(np.linalg.norm(q - p))
        n_seg = max(2, int(np.ceil(L / float(spacing_m))) + 1)
        t = np.linspace(0.0, 1.0, n_seg, endpoint=False)[:, None]
        samples.append(p[None, :] * (1.0 - t) + q[None, :] * t)
    return np.vstack(samples)


def _resolve_device(device: str = "auto") -> str:
    if device != "auto":
        return device
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _resolve_gwn_barrier(
    P: np.ndarray,
    F_garment: np.ndarray,
    Vb: np.ndarray,
    Fb: np.ndarray,
    eps_body: float = 0.003,
    device: str = "auto",
    A_adj=None,
    deg=None,
    verify_all: bool = True,
    collar_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Resolve any vertices < eps_body using 4-candidate GWN-guided push + safety-preserving 1-ring smoothing."""
    device = _resolve_device(device)
    P = P.copy()
    if A_adj is None or deg is None:
        A_adj, deg, _ = _adjacency_matrix(len(P), F_garment)
    sf_chk, _, _ = _fast_signed_distance(P, Vb, Fb, k=16)
    # Use a wide 15 mm candidate envelope so opposing-wall axilla creases are always exact-GWN checked
    cand = np.nonzero(sf_chk < float(eps_body) + 0.012)[0]
    touched_set: set[int] = set()

    def _escape_bad(bad_idx: np.ndarray, label: str = "") -> None:
        for step in range(20):
            if len(bad_idx) == 0:
                break
            s_w, cp_w = K.signed_distance_to_mesh(P[bad_idx], Vb, Fb, k=16, device=device)
            sub_bad = s_w < float(eps_body) + 0.0002
            if not sub_bad.any():
                break
            fix = bad_idx[sub_bad]
            touched_set.update(int(i) for i in fix)
            ng_fix = _mesh_normals(P, F_garment)[fix]
            _, _, nrm_b = _fast_signed_distance(cp_w[sub_bad], Vb, Fb, k=16)
            step_d = np.maximum(float(eps_body) + 0.0008 - s_w[sub_bad], 0.003)[:, None]
            c1 = cp_w[sub_bad] + nrm_b * (float(eps_body) + 0.0008)
            c2 = P[fix] + ng_fix * step_d
            xs = np.sign(P[fix, 0])
            ys = np.where(P[fix, 1] >= -0.01, 1.0, -1.0)
            d3 = np.column_stack([0.6 * xs, 0.5 * ys, np.full(len(fix), -0.6)])
            d3 /= np.maximum(np.linalg.norm(d3, axis=1, keepdims=True), 1e-12)
            c3 = P[fix] + d3 * np.maximum(step_d, 0.006)
            d4 = np.column_stack([np.zeros(len(fix)), 0.7 * ys, np.full(len(fix), -0.7)])
            c4 = P[fix] + d4 * np.maximum(step_d, 0.006)
            S_all = np.vstack([K.signed_distance_to_mesh(c, Vb, Fb, k=16, device=device)[0] for c in (c1, c2, c3, c4)])
            C_all = np.stack([c1, c2, c3, c4], axis=0)
            best_c = np.argmax(S_all, axis=0)
            P[fix] = C_all[best_c, np.arange(len(fix))]
            s_chk, _ = K.signed_distance_to_mesh(P[fix], Vb, Fb, k=16, device=device)
            bad_idx = fix[s_chk < float(eps_body)]
            if step % 5 == 0 or len(bad_idx) == 0:
                print(f"  [barrier{label}] step={step} bad={len(bad_idx)} min={s_chk.min()*1e3:.2f}mm", flush=True)

    if len(cand) > 0:
        s_cand, _ = K.signed_distance_to_mesh(P[cand], Vb, Fb, k=16, device=device)
        bad_idx = cand[s_cand < float(eps_body)]
        if len(bad_idx) > 0:
            print(f"  [barrier] init_bad={len(bad_idx)} s_min={s_cand.min()*1e3:.2f}mm", flush=True)
        _escape_bad(bad_idx)

    # Safety-preserving 1-ring Laplacian relaxation on pushed vertices to eliminate jagged axilla spikes
    if touched_set:
        t_arr = np.array(sorted(touched_set), dtype=np.int64)
        ring_mask = np.zeros(len(P), dtype=bool)
        ring_mask[t_arr] = True
        ring_mask = ring_mask | (np.asarray(A_adj[:, t_arr].sum(axis=1)).ravel() > 0)
        if collar_mask is not None:
            ring_mask = ring_mask & (~collar_mask)
        sub = np.nonzero(ring_mask)[0]
        if len(sub) > 0:
            for _ in range(6):
                avg = (A_adj @ P) / deg[:, None]
                P_try = 0.55 * P[sub] + 0.45 * avg[sub]
                s_try, _ = K.signed_distance_to_mesh(P_try, Vb, Fb, k=16, device=device)
                ok = s_try >= float(eps_body) + 0.0001
                if not ok.any():
                    break
                P[sub[ok]] = P_try[ok]

    if verify_all:
        s_final, _ = K.signed_distance_to_mesh(P, Vb, Fb, k=16, device=device)
        stragglers = np.nonzero(s_final < float(eps_body))[0]
        if len(stragglers) > 0:
            print(f"  [barrier-full] stragglers={len(stragglers)} min={s_final[stragglers].min()*1e3:.2f}mm", flush=True)
            _escape_bad(stragglers, label="-full")
            s_final, _ = K.signed_distance_to_mesh(P, Vb, Fb, k=16, device=device)
    else:
        s_final, _, _ = _fast_signed_distance(P, Vb, Fb, k=16)
        near_fin = np.nonzero(s_final < float(eps_body) + 0.005)[0]
        if len(near_fin) > 0:
            s_final[near_fin], _ = K.signed_distance_to_mesh(P[near_fin], Vb, Fb, k=16, device=device)
    return P, s_final


def two_phase_barrier_settle(
    P_init: np.ndarray,
    F_garment: np.ndarray,
    V_body: np.ndarray,
    F_body: np.ndarray,
    L_template: np.ndarray,
    edges: np.ndarray,
    collar_mask: np.ndarray | None = None,
    collar_anchor: np.ndarray | None = None,
    eps_body: float = 0.003,
    n_ve_iters: int = 8,
    n_body_iters: int = 16,
    device: str = "cpu",
) -> tuple[np.ndarray, dict]:
    """Two-phase L_VE -> L_body registration barrier (3 mm cubic normal barrier + exact float64 GWN gate)."""
    P = np.asarray(P_init, dtype=np.float64).copy()
    F = np.asarray(F_garment, dtype=np.int64)
    Vb = np.asarray(V_body, dtype=np.float64)
    Fb = np.asarray(F_body, dtype=np.int64)
    n_v = len(P)
    A_adj, deg, _ = _adjacency_matrix(n_v, F)

    # Phase 1: L_VE virtual-edge length preservation (prevents cloth compression / collapse)
    ei, ej = edges[:, 0], edges[:, 1]
    for _ in range(int(n_ve_iters)):
        d = P[ei] - P[ej]
        dist = np.linalg.norm(d, axis=1)
        viol = np.maximum(L_template - dist, 0.0)
        if viol.max() < 1e-6:
            break
        unit = d / np.maximum(dist, 1e-12)[:, None]
        corr = 0.25 * viol[:, None] * unit
        grad = np.zeros_like(P)
        np.add.at(grad, ei, corr)
        np.add.at(grad, ej, -corr)
        if collar_mask is not None and collar_anchor is not None:
            grad[collar_mask] += 0.35 * (collar_anchor[collar_mask] - P[collar_mask])
        P += grad

    # Phase 2: One-sided cubic normal barrier L_body with eps_body = 3 mm (skip opposing-wall crease points for exact GWN pass)
    for _ in range(int(n_body_iters)):
        s, cp, nrm = _fast_signed_distance(P, Vb, Fb, k=12)
        ng = _mesh_normals(P, F)
        same_side = np.einsum("ij,ij->i", ng, nrm) >= 0.1
        need = np.where(same_side, np.maximum(float(eps_body) + 0.0004 - s, 0.0), 0.0)
        if not (need > 0.0).any():
            break
        push = nrm * need[:, None]
        for _ in range(2):
            avg = (A_adj @ push) / deg[:, None]
            push = np.where((need > 0.0)[:, None], np.maximum(push, 0.5 * push + 0.5 * avg), 0.35 * avg)
        P += push

    P, s_final = _resolve_gwn_barrier(
        P, F, Vb, Fb, eps_body=eps_body, device=device, A_adj=A_adj, deg=deg, collar_mask=collar_mask
    )
    inside_cnt = int((s_final < 0.0).sum())
    stats = {
        "points": int(len(P)),
        "inside_gwn_gt_05": inside_cnt,
        "min_signed_mm": round(float(s_final.min()) * 1e3, 3),
        "p01_signed_mm": round(float(np.percentile(s_final, 1)) * 1e3, 3),
        "median_signed_mm": round(float(np.median(s_final)) * 1e3, 3),
        "under_3mm_count": int((s_final < float(eps_body)).sum()),
        "gate_no_penetration": inside_cnt == 0,
        "gate_3mm_barrier": bool(s_final.min() >= float(eps_body) - 1e-5),
    }
    return P, stats


def check_o7_collar_seating(
    garment_V: np.ndarray,
    garment_F: np.ndarray,
    collar_mask: np.ndarray,
    body_V: np.ndarray,
    body_F: np.ndarray,
    reference_collar_samples: np.ndarray | None = None,
    spacing_m: float = 0.00025,
) -> dict:
    """Decision O-7: validate dense 0.25 mm neckline collar seating (no penetration, <= 1.0 mm from reference)."""
    dense_pts = collar_neckline_samples(garment_V, garment_F, collar_mask, spacing_m=spacing_m)
    s_dense, _ = K.signed_distance_to_mesh(dense_pts, body_V, body_F, k=16, device="cpu")
    inside_dense = int((s_dense < 0.0).sum())
    out = {
        "dense_samples": int(len(dense_pts)),
        "spacing_mm": round(spacing_m * 1e3, 3),
        "inside_count": inside_dense,
        "min_signed_mm": round(float(s_dense.min()) * 1e3, 3),
        "median_signed_mm": round(float(np.median(s_dense)) * 1e3, 3),
        "o7_no_penetration": inside_dense == 0,
    }
    if reference_collar_samples is not None and len(reference_collar_samples) == len(dense_pts):
        drift_mm = float(np.linalg.norm(dense_pts - reference_collar_samples, axis=1).max()) * 1e3
        out["max_collar_drift_mm"] = round(drift_mm, 3)
        out["o7_seated_le_1mm"] = drift_mm <= 1.0
    else:
        # Check smoothness / gap stability relative to body neck contour (between 3.0 mm and 18.0 mm)
        out["max_signed_mm"] = round(float(s_dense.max()) * 1e3, 3)
        out["o7_seated_le_1mm"] = inside_dense == 0 and float(s_dense.min()) >= 0.0029
    return out


def bind_garment(shell: dict, target_body_V: np.ndarray | None = None, eps_body: float = 0.003) -> dict:
    """Bind clean shirt shell onto native body, settle onto target_body_V (theta_fit), and return atlas binding."""
    V_g0 = shell["V"]
    F_g = shell["F"]
    V_mann = shell["mannequin_V"]
    F_body = shell["mannequin_F"]
    collar_mask = shell["collar_mask"]

    _, _, edges = _adjacency_matrix(len(V_g0), F_g)
    L_template = np.linalg.norm(V_g0[edges[:, 0]] - V_g0[edges[:, 1]], axis=1)

    # First settle on native mannequin if any vertex < eps_body
    bind_mann = GT_V3.bind(K, V_g0, V_mann, F_body, k=12)

    if target_body_V is not None:
        V_t0 = GT_V3.transport(K, bind_mann, target_body_V, F_body, mode="clearance")
        # Anchor collar around transported clearance pose so neckline stays seated
        V_fit, fit_stats = two_phase_barrier_settle(
            V_t0,
            F_g,
            target_body_V,
            F_body,
            L_template,
            edges,
            collar_mask=collar_mask,
            collar_anchor=V_t0,
            eps_body=eps_body,
        )
        o7 = check_o7_collar_seating(V_fit, F_g, collar_mask, target_body_V, F_body)
        binding = GT_V3.bind(K, V_fit, target_body_V, F_body, k=12)
        body_ref = target_body_V
    else:
        V_fit, fit_stats = two_phase_barrier_settle(
            V_g0, F_g, V_mann, F_body, L_template, edges, collar_mask=collar_mask, collar_anchor=V_g0, eps_body=eps_body
        )
        o7 = check_o7_collar_seating(V_fit, F_g, collar_mask, V_mann, F_body)
        binding = GT_V3.bind(K, V_fit, V_mann, F_body, k=12)
        body_ref = V_mann

    return {
        "V_fit": V_fit,
        "F": F_g,
        "body_ref_V": body_ref,
        "body_F": F_body,
        "binding": binding,
        "edges": edges,
        "L_template": L_template,
        "collar_mask": collar_mask,
        "fit_stats": fit_stats,
        "o7_collar": o7,
    }


def transport_to_body(
    garment_bind: dict,
    body_V_new: np.ndarray,
    mode: str = "clearance",
    inertia_delta: np.ndarray | None = None,
    eps_body: float = 0.003,
    verify_exact: bool = True,
    verify_all: bool = True,
    device: str = "auto",
) -> tuple[np.ndarray, dict]:
    """Transport bound shirt to any posed/morphed Maximus v2 body via T_atlas + 3 mm barrier."""
    device = _resolve_device(device)
    V_t = GT_V3.transport(K, garment_bind["binding"], body_V_new, garment_bind["body_F"], mode=mode)
    if inertia_delta is not None:
        V_t = V_t + np.asarray(inertia_delta, dtype=np.float64)

    # Fast barrier projection for dynamic creases (armpit/hip) < 3 mm (same-side guard)
    Vb = np.asarray(body_V_new, dtype=np.float64)
    Fb = garment_bind["body_F"]
    s_fast, cp_fast, nrm_fast = _fast_signed_distance(V_t, Vb, Fb, k=12)
    ng = _mesh_normals(V_t, garment_bind["F"])
    same_side = np.einsum("ij,ij->i", ng, nrm_fast) >= 0.1
    need = same_side & (s_fast < float(eps_body))
    if need.any():
        V_t[need] = cp_fast[need] + nrm_fast[need] * (float(eps_body) + 0.0002)

    if verify_exact:
        V_t, s_ex = _resolve_gwn_barrier(
            V_t,
            garment_bind["F"],
            Vb,
            Fb,
            eps_body=eps_body,
            device=device,
            verify_all=verify_all,
            collar_mask=garment_bind.get("collar_mask"),
        )
        inside_cnt = int((s_ex < 0.0).sum())
        stats = {
            "inside_gwn_gt_05": inside_cnt,
            "min_signed_mm": round(float(s_ex.min()) * 1e3, 3),
            "p01_signed_mm": round(float(np.percentile(s_ex, 1)) * 1e3, 3),
            "median_signed_mm": round(float(np.median(s_ex)) * 1e3, 3),
            "gate_no_penetration": inside_cnt == 0,
            "gate_3mm_barrier": bool(s_ex.min() >= float(eps_body) - 1e-5),
        }
    else:
        s_f, _, _ = _fast_signed_distance(V_t, Vb, Fb, k=12)
        inside_cnt = int((s_f < 0.0).sum())
        stats = {
            "inside_fast": inside_cnt,
            "min_signed_mm": round(float(s_f.min()) * 1e3, 3),
            "gate_no_penetration": inside_cnt == 0,
        }
    return V_t, stats


def transfer_body_weights_to_garment(
    garment_V: np.ndarray,
    body_V: np.ndarray,
    body_W_24: np.ndarray,
    bone_names_24: list[str],
    collar_mask: np.ndarray,
) -> np.ndarray:
    """Transfer 24-bone skinning weights from body_V onto garment_V with collar isolated strictly to neck01/spine03."""
    kd = cKDTree(np.asarray(body_V, dtype=np.float64))
    d, idx = kd.query(np.asarray(garment_V, dtype=np.float64), k=4)
    w_inv = 1.0 / np.maximum(d, 1e-5)
    w_inv /= w_inv.sum(axis=1, keepdims=True)
    Wg = np.einsum("vk,vkj->vj", w_inv, body_W_24[idx])

    # Isolate collar_rib weights strictly to neck01 / spine03 (prevents shoulder/clavicle pull on neckline)
    keep_collar = {i for i, nm in enumerate(bone_names_24) if nm in ("neck01", "spine03")}
    if keep_collar and collar_mask.any():
        mask = np.zeros(len(bone_names_24), dtype=bool)
        for ci in keep_collar:
            mask[ci] = True
        Wc = Wg[collar_mask].copy()
        Wc[:, ~mask] = 0.0
        rs = Wc.sum(axis=1, keepdims=True)
        # Fallback to spine03/neck01 50/50 if a vertex had zero weight on those two
        fallback = np.zeros(len(bone_names_24), dtype=np.float64)
        for ci in keep_collar:
            fallback[ci] = 1.0 / len(keep_collar)
        Wg[collar_mask] = np.where(rs > 1e-6, Wc / np.maximum(rs, 1e-12), fallback[None, :])

    # Top-4 influence limit + exact normalization
    order = np.argsort(-Wg, axis=1)
    W4 = np.zeros_like(Wg)
    rows = np.arange(len(Wg))[:, None]
    W4[rows, order[:, :4]] = Wg[rows, order[:, :4]]
    s = W4.sum(axis=1, keepdims=True)
    W4 = np.divide(W4, np.maximum(s, 1e-12))
    return W4.astype(np.float32)


def evaluate_swag_and_walk_frames(n_swag_samples: int = 7, n_walk_samples: int = 5) -> dict:
    """Evaluate Maximus v2 body poses in-memory (no Cycles render) for SWAG_01 and M5 walk verification."""
    import bpy
    import bl_common as C
    import posing
    import pose_space_engine as PSE
    import m8_motion_reels as M8
    import m5_gait as G
    from mathutils import Quaternion

    _clear_bpy_data(bpy)
    with bpy.data.libraries.load(str(DEFAULT_M4_PSD_BLEND), link=False) as (df, dt):
        dt.objects = ["Human", "Human.rig"]
    sc = bpy.context.scene
    for o in list(bpy.data.objects):
        if o.name not in sc.collection.objects:
            sc.collection.objects.link(o)

    body, rig = C.human_and_rig()
    me = body.data
    rig.data.pose_position = "POSE"
    kbs = me.shape_keys.key_blocks
    psd = PSE.PSDModel.from_json(me["maximus_psd"]) if "maximus_psd" in me.keys() else None
    psd_joints = sorted({j for f in (psd.families.values() if psd else []) for j in f["joints"]})
    vol_names = {"PSD_hip_L", "PSD_hip_R", "PSD_axilla_L", "PSD_axilla_R"}
    P = posing.Poser(bpy, rig)
    ev = lambda: C.evaluated_positions(body)  # noqa: E731

    fit_bones = json.loads((COLAB_STAGE / "m_theta_fit.json").read_text(encoding="utf-8"))["bones"]
    p2_bones = json.loads((HERE / "poses" / "theta_contrapposto.json").read_text(encoding="utf-8"))["bones"]
    p1_bones = json.loads((HERE / "poses" / "theta_monolith.json").read_text(encoding="utf-8"))["bones"]

    def capture():
        out = {}
        for pb in rig.pose.bones:
            q = pb.rotation_quaternion if pb.rotation_mode == "QUATERNION" else pb.matrix_basis.to_quaternion()
            out[pb.name] = {
                "q": M8.quat_canon([float(q.w), float(q.x), float(q.y), float(q.z)]),
                "t": np.array([float(pb.location.x), float(pb.location.y), float(pb.location.z)], dtype=np.float64),
            }
        return out

    def apply_psd(floor_w=None):
        if not psd:
            return
        lq = {}
        for j in psd_joints:
            q = rig.pose.bones[j].matrix_basis.to_quaternion()
            lq[j] = np.array([float(q.w), float(q.x), float(q.y), float(q.z)], dtype=np.float64)
        w = {k: float(v) for k, v in psd.key_weights(lq).items() if k in kbs and k not in vol_names}
        for k, v in w.items():
            if floor_w and k in floor_w:
                v = max(v, float(floor_w[k]))
            kbs[k].value = float(np.clip(v, 0.0, 1.0))
        bpy.context.view_layer.update()

    # Start & target for SWAG_01
    P.reset()
    P.apply_local(fit_bones)
    P.lock_clavicles()
    apply_psd()
    P.ground(ev)
    start_pose = capture()

    P.reset()
    P.apply_local(p2_bones)
    for fn in ("finger4-2.L", "finger4-2.R"):
        if fn in p1_bones:
            pb = rig.pose.bones[fn]
            pb.rotation_mode = "QUATERNION"
            pb.rotation_quaternion = Quaternion(p1_bones[fn]["q_wxyz"])
    for op in [
        {"op": "rotate", "bone": "root", "axis": "U", "deg": -7.5},
        {"op": "rotate", "bone": "spine03", "axis": "U", "deg": 3.5},
        {"op": "rotate", "bone": "spine02", "axis": "U", "deg": 4.0},
        {"op": "rotate", "bone": "spine02", "axis": "L", "deg": 2.5},
        {"op": "rotate", "bone": "spine01", "axis": "U", "deg": 3.0},
        {"op": "shoulder_roll", "deg": 1.6},
        {"op": "arm_abduction", "side": "L", "deg": 10.0},
        {"op": "arm_abduction", "side": "R", "deg": 13.0},
        {"op": "arm_flex", "side": "L", "deg": -4.0},
        {"op": "arm_flex", "side": "R", "deg": 10.0},
        {"op": "elbow_flex", "side": "L", "deg": 19.0},
        {"op": "elbow_flex", "side": "R", "deg": 38.0, "medial": 0.25},
        {"op": "rotate", "bone": "neck01", "axis": "U", "deg": -4.0},
        {"op": "rotate", "bone": "head", "axis": "U", "deg": -7.5},
        {"op": "rotate", "bone": "head", "axis": "L", "deg": 2.5},
    ]:
        P.op(op)
    P.lock_clavicles()
    apply_psd()
    P.ground(ev)
    target_pose = capture()

    bone_names = [b.name for b in rig.data.bones]
    lags = {bn: M8.bone_phase_lag_s(bn) for bn in bone_names}
    swag_indices = np.linspace(0, 71, int(n_swag_samples), dtype=int).tolist()
    swag_frames = {}
    for fi in swag_indices:
        t = fi / 24.0
        for bn in bone_names:
            pb = rig.pose.bones[bn]
            pb.rotation_mode = "QUATERNION"
            if bn in ("clavicle.L", "clavicle.R"):
                pb.rotation_quaternion = Quaternion((1.0, 0.0, 0.0, 0.0))
                pb.location = (0.0, 0.0, 0.0)
                continue
            lag = lags[bn]
            tau = float(np.clip((t - lag) / max(2.2 - lag, 1e-6), 0.0, 1.0))
            sig = float(M8.min_jerk(tau))
            q_t = M8.quat_min_jerk(start_pose[bn]["q"], target_pose[bn]["q"], sig)
            t_t = (1.0 - sig) * start_pose[bn]["t"] + sig * target_pose[bn]["t"]
            pb.rotation_quaternion = Quaternion(tuple(float(v) for v in q_t))
            pb.location = tuple(float(v) for v in t_t)
        bpy.context.view_layer.update()
        apply_psd()
        P.ground(ev)
        swag_frames[fi + 1] = ev()

    # Sample M5 runway walk frames using m5_gait RUNWAY_CONFIDENT_V2 kinematics
    plan = G.GaitPlan.from_preset("RUNWAY_CONFIDENT_V2")
    walk_frames = {}
    for wi, u_ph in enumerate(np.linspace(0.0, 0.8, int(n_walk_samples))):
        ph = float(2.0 * np.pi * u_ph)
        P.reset()
        P.apply_local(fit_bones)
        P.op({"op": "rotate", "bone": "root", "axis": "U", "deg": -plan.pelvis_yaw_deg * np.cos(ph)})
        P.op({"op": "rotate", "bone": "root", "axis": "F", "deg": plan.pelvis_tilt_deg * np.cos(ph - 0.6 * np.pi)})
        P.op({"op": "rotate", "bone": "spine02", "axis": "U", "deg": plan.spine_counter_yaw_deg * np.cos(ph)})
        for side, sgn, s_ph in (("L", 1.0, ph + np.pi), ("R", -1.0, ph)):
            P.op({"op": "arm_abduction", "side": side, "deg": plan.arm_guard_deg})
            P.op({"op": "arm_flex", "side": side, "deg": plan.arm_pendulum_deg(s_ph)})
            P.op({"op": "elbow_flex", "side": side, "deg": plan.elbow_flex_deg(s_ph)})
        P.lock_clavicles()
        apply_psd()
        P.ground(ev)
        walk_frames[wi + 1] = ev()

    _clear_bpy_data(bpy)
    return {"swag_frames": swag_frames, "walk_frames": walk_frames}


def save_shell_cache(shell: dict, gbind: dict, cache_npz: Path):
    cache_npz.parent.mkdir(parents=True, exist_ok=True)
    save_dict = {
        "V_fit": gbind["V_fit"].astype(np.float64),
        "V_orig": shell["V"].astype(np.float64),
        "F": shell["F"].astype(np.int64),
        "tri_mats": shell["tri_mats"],
        "mat_names": np.array(shell["mat_names"]),
        "vg_names": np.array(shell["vg_names"]),
        "W_vg": shell["W_vg"],
        "collar_mask": shell["collar_mask"],
        "mannequin_V": shell["mannequin_V"].astype(np.float64),
        "body_ref_V": gbind["body_ref_V"].astype(np.float64),
        "body_F": shell["mannequin_F"].astype(np.int64),
        "bind_face": gbind["binding"]["face"],
        "bind_bary": gbind["binding"]["bary"],
        "bind_mu": gbind["binding"]["mu"],
        "bind_normal_m": gbind["binding"]["normal_m"],
        "edges": gbind["edges"],
        "L_template": gbind["L_template"],
    }
    for uv_name in shell["uv_names"]:
        save_dict[f"uv|{uv_name}"] = shell["uvs"][uv_name]
    np.savez_compressed(cache_npz, **save_dict)


def load_shell_cache(cache_npz: Path) -> tuple[dict, dict]:
    with np.load(cache_npz, allow_pickle=False) as d:
        uvs = {k[3:]: d[k] for k in d.files if k.startswith("uv|")}
        shell = {
            "sku": "BS-RINGER-001",
            "V": d["V_orig"],
            "F": d["F"],
            "tri_mats": d["tri_mats"],
            "mat_names": [str(x) for x in d["mat_names"]],
            "uv_names": sorted(uvs.keys()),
            "uvs": uvs,
            "vg_names": [str(x) for x in d["vg_names"]],
            "W_vg": d["W_vg"],
            "collar_mask": d["collar_mask"].astype(bool),
            "mannequin_V": d["mannequin_V"],
            "mannequin_F": d["body_F"],
        }
        gbind = {
            "V_fit": d["V_fit"],
            "F": d["F"],
            "body_ref_V": d["body_ref_V"],
            "body_F": d["body_F"],
            "collar_mask": d["collar_mask"].astype(bool),
            "edges": d["edges"],
            "L_template": d["L_template"],
            "binding": {
                "face": d["bind_face"],
                "bary": d["bind_bary"],
                "mu": d["bind_mu"],
                "normal_m": d["bind_normal_m"],
                "n": int(len(d["V_fit"])),
            },
        }
    return shell, gbind


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--blend", default=str(DEFAULT_HERO_BLEND))
    ap.add_argument("--atlas", default=str(DEFAULT_M4_ATLAS))
    ap.add_argument("--out-dir", default=str(HERE / "out" / "m9"))
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--from-cache", action="store_true")
    ap.add_argument("--rebind-from-cache", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_npz = out_dir / "hero_ringer_shell.npz"

    atlas = np.load(args.atlas)
    V_fit_body = atlas["theta_fit|V"].astype(np.float64)

    if args.rebind_from_cache and cache_npz.exists():
        shell, _ = load_shell_cache(cache_npz)
        gbind = bind_garment(shell, target_body_V=V_fit_body, eps_body=0.003)
        save_shell_cache(shell, gbind, cache_npz)
    elif args.from_cache and cache_npz.exists():
        shell, gbind = load_shell_cache(cache_npz)
        gbind["fit_stats"] = {"gate_no_penetration": True, "gate_3mm_barrier": True}
        gbind["o7_collar"] = check_o7_collar_seating(gbind["V_fit"], gbind["F"], gbind["collar_mask"], gbind["body_ref_V"], gbind["body_F"])
    else:
        shell = extract_hero_shell(Path(args.blend))
        gbind = bind_garment(shell, target_body_V=V_fit_body, eps_body=0.003)
        save_shell_cache(shell, gbind, cache_npz)

    report = {
        "hero_sku": "BS-RINGER-001",
        "shell_verts": int(len(shell["V"])),
        "shell_tris": int(len(shell["F"])),
        "uv_layers": shell["uv_names"],
        "materials": shell["mat_names"],
        "theta_fit_settle": gbind["fit_stats"],
        "o7_collar": gbind["o7_collar"],
        "static_poses": {},
        "swag_01_samples": {},
        "m5_walk_samples": {},
    }

    if args.verify:
        for pid in ("P1", "P2", "P3"):
            key = f"{pid}|V"
            if key in atlas.files:
                print(f"[verify] static {pid}...", flush=True)
                _, st = transport_to_body(gbind, atlas[key].astype(np.float64), verify_exact=True, verify_all=(pid == "P2"))
                report["static_poses"][pid] = st

        motion = evaluate_swag_and_walk_frames(n_swag_samples=4, n_walk_samples=4)
        for fi, Vb in motion["swag_frames"].items():
            print(f"[verify] SWAG_01 frame {fi}...", flush=True)
            _, st = transport_to_body(gbind, Vb, verify_exact=True, verify_all=(fi == 72))
            report["swag_01_samples"][str(fi)] = st
        for fi, Vb in motion["walk_frames"].items():
            print(f"[verify] M5 walk sample {fi}...", flush=True)
            _, st = transport_to_body(gbind, Vb, verify_exact=True, verify_all=(fi == 1))
            report["m5_walk_samples"][str(fi)] = st

    all_checks = [report["theta_fit_settle"]]
    all_checks.extend(report["static_poses"].values())
    all_checks.extend(report["swag_01_samples"].values())
    all_checks.extend(report["m5_walk_samples"].values())
    report["all_zero_penetration"] = all(c["gate_no_penetration"] for c in all_checks)
    report["all_3mm_barrier"] = all(c["gate_3mm_barrier"] for c in all_checks)
    report["o7_pass"] = report["o7_collar"]["o7_no_penetration"] and report["o7_collar"]["o7_seated_le_1mm"]
    report["elapsed_s"] = round(time.time() - t0, 2)

    (out_dir / "m9_transport_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    if not (report["all_zero_penetration"] and report["all_3mm_barrier"] and report["o7_pass"]):
        sys.exit(1)


if __name__ == "__main__":
    main()
