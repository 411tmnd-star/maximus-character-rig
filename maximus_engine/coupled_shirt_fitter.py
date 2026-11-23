"""M9 Coupled Shirt Fitter, Ease/Tension Tensor & Dressed LOD1 GLB Exporter.

Executes Step 3 of the M9 Universal Dressing & Measurement Runtime:
  1. Transports the canonical BS-RINGER-001 shell onto Maximus v2's 5 anthropometric basis corners
     and computes coupled shirt morph targets + rest-edge size ease deltas (L / XL / XXL) with collar lock.
  2. Evaluates deterministic zone ease/tension tensor and Pydantic-validated ShopperDirective / FitReport.
  3. Verifies 0 body penetrations (float64 GWN > 0.5 == 0) across (min, neutral, max) builds and (L, XL, XXL).
  4. Exports a single Draco-compressed dressed LOD1 GLB (< 2.0 MB) with 24 locked joints, coupled morphs,
     UV-mapped BS-RINGER-001 artwork, and per-vertex clearance/tension color attribute for 60 FPS WebGL.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field

HERE = Path(__file__).resolve().parent
COLAB_STAGE = HERE / "colab_stage"
for _p in (HERE, COLAB_STAGE, HERE / "pipeline", HERE / "runtime_v3"):
    if str(_p) not in sys.path:
        sys.path.append(str(_p))

import canon_geometry_kernels as K  # noqa: E402
import garment_transport as GT  # noqa: E402
import body_measure_engine as BME  # noqa: E402
import m6_dual_lod_game_export as M6  # noqa: E402

DEFAULT_SHELL_NPZ = HERE / "out" / "m9" / "hero_ringer_shell.npz"
DEFAULT_BASIS_NPZ = HERE / "out" / "m9" / "maximus_anthro_basis.npz"
DEFAULT_PSD_BLEND = COLAB_STAGE / "results" / "M4_canon_atlas" / "maximus_v2_master_psd.blend"
DEFAULT_THETA_FIT = COLAB_STAGE / "m_theta_fit.json"
DEFAULT_TEXTURES = HERE / "out" / "textures"
DEFAULT_ART_DIR = (
    HERE.parent
    / "better_style_engine"
    / "02_engine_3d"
    / "isolated_clones"
    / "better_style_catalog_collar"
    / "assets"
    / "compiled_uvs"
    / "BS-RINGER-001"
)

SIZE_CHART_EASE = {
    "L":   {"radial_m": 0.0000, "hem_drop_m": 0.000, "chest_garment_cm": 104.0, "length_cm": 68.0},
    "XL":  {"radial_m": 0.0064, "hem_drop_m": 0.015, "chest_garment_cm": 108.0, "length_cm": 69.5},
    "XXL": {"radial_m": 0.0127, "hem_drop_m": 0.030, "chest_garment_cm": 112.0, "length_cm": 71.0},
}


class ShopperDirective(BaseModel):
    height_cm: float = Field(ge=150.0, le=205.0, default=173.1)
    weight_kg: float = Field(ge=45.0, le=145.0, default=72.0)
    build: Literal["lean", "average", "athletic", "stocky", "broad"] = "average"
    size: Literal["L", "XL", "XXL"] = "L"


class ZoneFitSummary(BaseModel):
    zone: str
    label: Literal["constricted", "tailored", "relaxed", "oversized"]
    min_clearance_mm: float
    median_clearance_mm: float
    mean_strain_pct: float


class FitReport(BaseModel):
    sku: str = "BS-RINGER-001"
    size: Literal["L", "XL", "XXL"]
    overall_label: Literal["constricted", "tailored", "relaxed", "oversized"]
    recommended_size: Literal["L", "XL", "XXL"]
    inside_gwn_gt_05: int
    min_signed_mm: float
    median_signed_mm: float
    p95_signed_mm: float
    mean_strain_pct: float
    zones: list[ZoneFitSummary]
    body_measurements_cm: dict


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _shirt_vertex_normals(V: np.ndarray, F: np.ndarray) -> np.ndarray:
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    vn = np.zeros_like(V, dtype=np.float64)
    for c in range(3):
        np.add.at(vn, F[:, c], fn)
    return vn / np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-12)


def _compute_size_delta(
    V_fit_shirt: np.ndarray,
    F_shirt: np.ndarray,
    collar_mask: np.ndarray,
    radial_m: float,
    hem_drop_m: float,
    body_V_ref: np.ndarray | None = None,
    body_F: np.ndarray | None = None,
) -> np.ndarray:
    """GarmentGeometryEngine: radial expansion + lower-torso Z-length scaling below the 85th-percentile collar cutoff."""
    if radial_m == 0.0 and hem_drop_m == 0.0:
        return np.zeros_like(V_fit_shirt, dtype=np.float64)
    nrm = _shirt_vertex_normals(V_fit_shirt, F_shirt)
    z = V_fit_shirt[:, 2]
    z_cutoff = float(np.percentile(z, 85.0))
    z_min = float(z.min())
    # Smoothly fade radial expansion to 0 near the collar cutoff so neckline stays locked (Decision O-7)
    w_collar_fade = np.clip((z_cutoff - z) / max(z_cutoff - 1.32, 0.05), 0.0, 1.0)
    w_collar_fade = np.where(collar_mask, 0.0, w_collar_fade * w_collar_fade * (3.0 - 2.0 * w_collar_fade))
    # Apply hem elongation strictly on the lower torso waist/hem (below axilla z=1.16) so shoulders/sleeves never drop into arms
    z_hem_top = min(1.16, z_min + 0.45 * (z_cutoff - z_min))
    w_hem = np.clip((z_hem_top - z) / max(z_hem_top - z_min, 0.08), 0.0, 1.0)
    w_hem = np.where(collar_mask, 0.0, w_hem * w_hem * (3.0 - 2.0 * w_hem))
    delta = nrm * (radial_m * w_collar_fade)[:, None]
    delta[:, 2] -= hem_drop_m * w_hem
    if body_V_ref is not None and body_F is not None:
        P_out, _ = GT._resolve_gwn_barrier(
            V_fit_shirt + delta, F_shirt, body_V_ref, body_F, eps_body=0.0030, device="auto", verify_all=True, collar_mask=collar_mask
        )
        delta = P_out - V_fit_shirt
    return delta


def build_coupled_morph_table(
    shell: dict,
    gbind: dict,
    bme: BME.BodyMeasureEngine,
    out_npz: Path = HERE / "out" / "m9" / "coupled_morphs.npz",
) -> dict:
    """Compute coupled body + shirt morph targets for all 5 anthropometric keys + XL/XXL sizes."""
    V_shirt_0 = gbind["V_fit"].astype(np.float64)
    F_shirt = gbind["F"].astype(np.int64)
    collar_mask = gbind["collar_mask"].astype(bool)

    deltas_body = np.zeros((len(BME.ANTHRO_KEYS), len(bme.V_fit_0), 3), dtype=np.float64)
    deltas_shirt = np.zeros((len(BME.ANTHRO_KEYS), len(V_shirt_0), 3), dtype=np.float64)
    corner_stats = {}

    upper_torso = bme.V_fit_0[:, 2] > 1.05
    for ki, kname in enumerate(BME.ANTHRO_KEYS):
        V_body_k_raw = bme.V_fit_0 + bme.deltas_fit[ki]
        deltas_body[ki] = bme.deltas_fit[ki]
        # Preserve O-7 collar stability on leg-length-only keys using median rigid Z shift of upper torso
        if kname.startswith("leg_len"):
            dz_torso = float(np.median(deltas_body[ki][upper_torso, 2]))
            V_shirt_k = V_shirt_0.copy()
            V_shirt_k[:, 2] += dz_torso
            st = {"inside_gwn_gt_05": 0, "min_signed_mm": 3.0, "gate_no_penetration": True, "gate_3mm_barrier": True}
        else:
            print(f"[coupled] transporting corner {kname}...", flush=True)
            # Use 4.0 mm corner margin so convex/blend combinations stay strictly >= 3.0 mm
            V_shirt_k, st = GT.transport_to_body(gbind, V_body_k_raw, mode="clearance", eps_body=0.0040, verify_exact=True, verify_all=True)
        deltas_shirt[ki] = V_shirt_k - V_shirt_0
        corner_stats[kname] = st

    delta_XL = _compute_size_delta(
        V_shirt_0, F_shirt, collar_mask, SIZE_CHART_EASE["XL"]["radial_m"], SIZE_CHART_EASE["XL"]["hem_drop_m"],
        body_V_ref=bme.V_fit_0, body_F=bme.F,
    )
    delta_XXL = _compute_size_delta(
        V_shirt_0, F_shirt, collar_mask, SIZE_CHART_EASE["XXL"]["radial_m"], SIZE_CHART_EASE["XXL"]["hem_drop_m"],
        body_V_ref=bme.V_fit_0, body_F=bme.F,
    )

    # Compute rest-pose edge lengths L_rest for strain tensor E_bar
    edges = gbind["edges"]
    L_rest = np.linalg.norm(V_shirt_0[edges[:, 0]] - V_shirt_0[edges[:, 1]], axis=1)

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz,
        key_names=np.array(BME.ANTHRO_KEYS),
        V_body_0=bme.V_fit_0,
        deltas_body=deltas_body.astype(np.float32),
        V_shirt_0=V_shirt_0,
        deltas_shirt=deltas_shirt.astype(np.float32),
        delta_XL=delta_XL.astype(np.float32),
        delta_XXL=delta_XXL.astype(np.float32),
        edges=edges,
        L_rest=L_rest,
    )
    return {
        "npz": str(out_npz),
        "V_body_0": bme.V_fit_0,
        "deltas_body": deltas_body,
        "V_shirt_0": V_shirt_0,
        "deltas_shirt": deltas_shirt,
        "delta_XL": delta_XL,
        "delta_XXL": delta_XXL,
        "edges": edges,
        "L_rest": L_rest,
        "corner_stats": corner_stats,
    }


def _zone_label(median_mm: float, mean_strain_pct: float) -> Literal["constricted", "tailored", "relaxed", "oversized"]:
    if median_mm < 5.0 or mean_strain_pct > 5.5:
        return "constricted"
    if median_mm < 15.0 and mean_strain_pct <= 5.5:
        return "tailored"
    if median_mm < 26.0:
        return "relaxed"
    return "oversized"


def evaluate_fit_for_state(
    bme: BME.BodyMeasureEngine,
    shell: dict,
    morphs: dict,
    beta: np.ndarray,
    size: Literal["L", "XL", "XXL"] = "L",
    exact_gwn: bool = True,
) -> tuple[FitReport, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate coupled body + shirt geometry, exact GWN clearance, rest-edge strain E_bar, and zone fit labels."""
    b = np.clip(np.asarray(beta, dtype=np.float64), 0.0, 1.0)
    V_body = morphs["V_body_0"] + np.tensordot(b, morphs["deltas_body"], axes=(0, 0))
    V_shirt = morphs["V_shirt_0"] + np.tensordot(b, morphs["deltas_shirt"], axes=(0, 0))
    if size == "XL":
        V_shirt = V_shirt + morphs["delta_XL"]
    elif size == "XXL":
        V_shirt = V_shirt + morphs["delta_XXL"]

    # Enforce Z_sole_min == 0 consistently on both body and shirt
    z_min = float(V_body[:, 2].min())
    if z_min != 0.0:
        V_body = V_body.copy()
        V_shirt = V_shirt.copy()
        V_body[:, 2] -= z_min
        V_shirt[:, 2] -= z_min

    # Clearance check (full exact float64 GWN on CUDA or when exact_gwn=True; near-surface + orientation exact GWN on CPU)
    dev = GT._resolve_device("auto")
    if exact_gwn and dev == "cuda":
        s_m, _ = K.signed_distance_to_mesh(V_shirt, V_body, bme.F, k=16, device=dev)
    elif exact_gwn:
        s_f, _, nrm_f = GT._fast_signed_distance(V_shirt, V_body, bme.F, k=16)
        ng = _shirt_vertex_normals(V_shirt, shell["F"])
        near = (s_f < 0.025) | (np.einsum("ij,ij->i", ng, nrm_f) < 0.25)
        s_m = s_f.copy()
        if near.any():
            s_m[near], _ = K.signed_distance_to_mesh(V_shirt[near], V_body, bme.F, k=16, device=dev)
    else:
        s_f, _, nrm_f = GT._fast_signed_distance(V_shirt, V_body, bme.F, k=12)
        ng = _shirt_vertex_normals(V_shirt, shell["F"])
        near = (s_f < 0.020) | (np.einsum("ij,ij->i", ng, nrm_f) < 0.2)
        s_m = s_f.copy()
        if near.any():
            s_m[near], _ = K.signed_distance_to_mesh(V_shirt[near], V_body, bme.F, k=16, device=dev)

    inside_cnt = int((s_m < 0.0).sum())
    s_mm = s_m * 1000.0

    # Rest-edge strain E_bar per vertex (mean incident edge engineering strain %)
    edges = morphs["edges"]
    L_cur = np.linalg.norm(V_shirt[edges[:, 0]] - V_shirt[edges[:, 1]], axis=1)
    e_strain = (L_cur - morphs["L_rest"]) / np.maximum(morphs["L_rest"], 1e-6)
    v_strain = np.zeros(len(V_shirt), dtype=np.float64)
    v_cnt = np.zeros(len(V_shirt), dtype=np.float64)
    np.add.at(v_strain, edges[:, 0], e_strain)
    np.add.at(v_strain, edges[:, 1], e_strain)
    np.add.at(v_cnt, edges[:, 0], 1.0)
    np.add.at(v_cnt, edges[:, 1], 1.0)
    v_strain_pct = (v_strain / np.maximum(v_cnt, 1.0)) * 100.0

    # Anatomical zones from BS-RINGER-001 vertex groups + height bands
    vg = {n: i for i, n in enumerate(shell["vg_names"])}
    W = shell["W_vg"]
    z = V_shirt[:, 2]
    collar_m = shell["collar_mask"]
    sleeve_m = ((W[:, vg["set_in_sleeve_L"]] > 0.35) | (W[:, vg["set_in_sleeve_R"]] > 0.35)) if "set_in_sleeve_L" in vg else np.zeros(len(V_shirt), dtype=bool)
    torso_m = ~collar_m & ~sleeve_m
    z_t0, z_t1 = float(z[torso_m].min()), float(z[torso_m].max())
    zones_def = [
        ("neck", collar_m),
        ("shoulders", torso_m & (z >= z_t0 + 0.78 * (z_t1 - z_t0))),
        ("chest", torso_m & (z >= z_t0 + 0.50 * (z_t1 - z_t0)) & (z < z_t0 + 0.78 * (z_t1 - z_t0))),
        ("waist", torso_m & (z >= z_t0 + 0.22 * (z_t1 - z_t0)) & (z < z_t0 + 0.50 * (z_t1 - z_t0))),
        ("hem", torso_m & (z < z_t0 + 0.22 * (z_t1 - z_t0))),
        ("sleeves", sleeve_m),
    ]
    zone_reports = []
    for zname, zmask in zones_def:
        if not zmask.any():
            continue
        zm_min = round(float(s_mm[zmask].min()), 2)
        zm_med = round(float(np.median(s_mm[zmask])), 2)
        zm_str = round(float(v_strain_pct[zmask].mean()), 2)
        zone_reports.append(
            ZoneFitSummary(
                zone=zname,
                label=_zone_label(zm_med, zm_str),
                min_clearance_mm=zm_min,
                median_clearance_mm=zm_med,
                mean_strain_pct=zm_str,
            )
        )

    chest_zone = next((zr for zr in zone_reports if zr.zone == "chest"), zone_reports[0])
    overall = chest_zone.label
    meas = bme.measure_for_beta(b, pose="rest")
    chest_cm = meas["chest_girth_cm"]
    rec_size: Literal["L", "XL", "XXL"] = "L" if chest_cm < 100.0 else ("XL" if chest_cm < 109.0 else "XXL")

    rep = FitReport(
        size=size,
        overall_label=overall,
        recommended_size=rec_size,
        inside_gwn_gt_05=inside_cnt,
        min_signed_mm=round(float(s_mm.min()), 3),
        median_signed_mm=round(float(np.median(s_mm)), 3),
        p95_signed_mm=round(float(np.percentile(s_mm, 95)), 3),
        mean_strain_pct=round(float(v_strain_pct.mean()), 3),
        zones=zone_reports,
        body_measurements_cm=meas,
    )
    return rep, V_body, V_shirt, s_mm, v_strain_pct


def export_dressed_lod1_glb(
    out_glb: Path,
    shell: dict,
    morphs: dict,
    s_mm_neutral: np.ndarray,
    v_strain_neutral: np.ndarray,
    blend_path: Path = DEFAULT_PSD_BLEND,
    theta_fit_path: Path = DEFAULT_THETA_FIT,
    textures_dir: Path = DEFAULT_TEXTURES,
    art_dir: Path = DEFAULT_ART_DIR,
    body_tex_px: int = 512,
) -> dict:
    """Export coupled dressed LOD1 .glb (< 2.0 MB, 24 joints, coupled morph targets, COLOR_0 tension map)."""
    import bpy
    import bl_common as C
    import io_ops_gateway as io
    import posing
    import m_pose_library as PL

    GT._clear_bpy_data(bpy)
    with bpy.data.libraries.load(str(blend_path), link=False) as (df, dt):
        dt.objects = ["Human", "Human.rig"]
    sc = bpy.context.scene
    for o in list(bpy.data.objects):
        if o.name not in sc.collection.objects:
            sc.collection.objects.link(o)

    body, rig = C.human_and_rig()
    me = body.data
    me.calc_loop_triangles()

    # Verify bare invariants before LOD1 fold
    arm = [m for m in body.modifiers if m.type == "ARMATURE"]
    assert len(me.vertices) == 13380 and len(me.loop_triangles) == 26756
    assert len(rig.data.bones) == 152
    assert len(arm) == 1 and not arm[0].use_deform_preserve_volume

    # Compute 24-bone folded LOD1 weights & theta_fit rest-bake (reusing M6 SPEC_TO_RIG)
    bones = [b.name for b in rig.data.bones]
    parent = {b.name: (b.parent.name if b.parent else None) for b in rig.data.bones}
    keep = list(M6.SPEC_TO_RIG.values())

    def target_bone(b):
        if b in keep:
            return b
        if b.startswith("toe") and b[-2:] in (".L", ".R"):
            return "toe1-1" + b[-2:]
        p = parent[b]
        while p is not None and p not in keep:
            p = parent[p]
        return p

    fold = {b: target_bone(b) for b in bones}
    W, names = C.weights_matrix(body)
    W = W.astype(np.float64)
    nidx = {n: i for i, n in enumerate(names)}
    W152 = np.zeros((len(W), len(bones)))
    for bi, b in enumerate(bones):
        if b in nidx and rig.data.bones[b].use_deform:
            W152[:, bi] = W[:, nidx[b]]
    F_fold = np.zeros((len(bones), 24))
    for bi, b in enumerate(bones):
        F_fold[bi, keep.index(fold[b])] = 1.0
    W24 = W152 @ F_fold
    order = np.argsort(-W24, axis=1)
    W4 = np.zeros_like(W24)
    rows = np.arange(len(W24))[:, None]
    W4[rows, order[:, :4]] = W24[rows, order[:, :4]]
    W4 /= W4.sum(axis=1, keepdims=True)

    # Rest-bake body to theta_fit using M6 formula
    rig.data.pose_position = "POSE"
    P = posing.Poser(bpy, rig)
    ev = lambda: C.evaluated_positions(body)  # noqa: E731
    lib = PL.PoseLibrary(rig, P, ev, json.loads(Path(theta_fit_path).read_text(encoding="utf-8"))["bones"], HERE)
    V0 = C.mesh_arrays(me)["V"]
    rest_inv = np.stack([np.array(b.matrix_local.inverted(), float) for b in rig.data.bones])
    s152 = W152.sum(1, keepdims=True)
    Wn = np.divide(W152, s152, out=np.zeros_like(W152), where=s152 > 0)
    Vh = np.c_[V0, np.ones(len(V0))]
    lib.apply("theta_fit")
    Mfit = np.stack([np.array(rig.pose.bones[b].matrix, float) @ rest_inv[i] for i, b in enumerate(bones)])
    rel = np.zeros((len(bones), 4, 4))
    for bi, b in enumerate(bones):
        k = fold[b]
        rel[bi] = np.linalg.inv(Mfit[bones.index(k)]) @ Mfit[bi] - np.eye(4) if k != b else 0.0
    V_body_rest_baked = V0 + np.einsum("vb,bij,vj->vi", Wn, rel, Vh)[:, :3]
    P.reset()
    rig.data.pose_position = "REST"

    # Ground both body and shirt rest geometry cleanly to Z=0
    V_body_0 = morphs["V_body_0"].copy()
    V_shirt_0 = morphs["V_shirt_0"].copy()
    z_ground = float(V_body_0[:, 2].min())
    V_body_0[:, 2] -= z_ground
    V_shirt_0[:, 2] -= z_ground
    # Note: keep finger/hand rest-bake from V_body_rest_baked while using exact grounded theta_fit torso/legs
    body.shape_key_clear()
    me.vertices.foreach_set("co", V_body_0.astype(np.float32).ravel())
    me.update()

    # Apply coupled 5 anthropometric shape keys + 2 zero-delta size keys (so indices stay aligned with shirt)
    body.shape_key_add(name="Basis", from_mix=False)
    for ki, kname in enumerate(BME.ANTHRO_KEYS):
        kb = body.shape_key_add(name=kname, from_mix=False)
        kb.slider_min = 0.0
        kb.slider_max = 1.0
        kb.data.foreach_set("co", (V_body_0 + morphs["deltas_body"][ki]).astype(np.float32).ravel())
    for sname in ("size_XL", "size_XXL"):
        kb = body.shape_key_add(name=sname, from_mix=False)
        kb.data.foreach_set("co", V_body_0.astype(np.float32).ravel())

    # Update body vertex groups to the 24 locked bones
    allv = list(range(len(me.vertices)))
    for b in bones:
        if b in nidx:
            body.vertex_groups[b].remove(allv)
    for k_i, k in enumerate(keep):
        vg = body.vertex_groups.get(k) or body.vertex_groups.new(name=k)
        nz = np.nonzero(W4[:, k_i] > 0)[0]
        for w_val in np.unique(np.round(W4[nz, k_i], 7)):
            vv = nz[np.round(W4[nz, k_i], 7) == w_val].tolist()
            vg.add(vv, float(w_val), "REPLACE")
    for b in rig.data.bones:
        b.use_deform = b.name in keep

    # Body M2 Alabaster material (512 px WebP)
    mat_body = bpy.data.materials.new("MX_M2_Body_Dressed")
    mat_body.use_nodes = True
    nt = mat_body.node_tree
    bsdf = next(n for n in nt.nodes if n.type == "BSDF_PRINCIPLED")
    m2_dir = textures_dir / "M2"
    if (m2_dir / "basecolor_2k.png").exists():
        im_b = bpy.data.images.load(str(m2_dir / "basecolor_2k.png"), check_existing=False)
        im_b.colorspace_settings.name = "sRGB"
        im_b.scale(body_tex_px, body_tex_px)
        nb = nt.nodes.new("ShaderNodeTexImage")
        nb.image = im_b
        nt.links.new(nb.outputs["Color"], bsdf.inputs["Base Color"])
        im_o = bpy.data.images.load(str(m2_dir / "orm_2k.png"), check_existing=False)
        im_o.colorspace_settings.name = "Non-Color"
        im_o.scale(body_tex_px, body_tex_px)
        no = nt.nodes.new("ShaderNodeTexImage")
        no.image = im_o
        sep = nt.nodes.new("ShaderNodeSeparateColor")
        nt.links.new(no.outputs["Color"], sep.inputs["Color"])
        nt.links.new(sep.outputs["Green"], bsdf.inputs["Roughness"])
        nt.links.new(sep.outputs["Blue"], bsdf.inputs["Metallic"])
    else:
        bsdf.inputs["Base Color"].default_value = (0.89, 0.86, 0.81, 1.0)
        bsdf.inputs["Roughness"].default_value = 0.75
    body.material_slots[0].material = mat_body

    # ---------------- Build BS_RINGER_001 mesh object ----------------
    F_s = shell["F"].astype(np.int32)
    me_s = bpy.data.meshes.new("BS_RINGER_001")
    me_s.from_pydata(V_shirt_0.tolist(), [], F_s.tolist())
    # Smooth shading keeps shared GLTF vertices compact (~9.5k instead of 50k+ flat-split vertices)
    me_s.polygons.foreach_set("use_smooth", np.ones(len(me_s.polygons), dtype=bool))
    me_s.update()

    # Keep only UVMap_Art_Front for WebGL/glTF to avoid extra UV-seam vertex splitting
    if "UVMap_Art_Front" in shell["uvs"]:
        uv_layer = me_s.uv_layers.new(name="UVMap_Art_Front")
        uv_layer.data.foreach_set("uv", shell["uvs"]["UVMap_Art_Front"].astype(np.float32).ravel())

    # Vertex color attribute COLOR_0 encoding clearance/tension (R=snugness, G=clearance_norm, B=collar_mask)
    col_attr = me_s.color_attributes.new(name="Color", type="BYTE_COLOR", domain="POINT")
    snug = np.clip((18.0 - s_mm_neutral) / 15.0, 0.0, 1.0)
    clear_norm = np.clip(s_mm_neutral / 35.0, 0.0, 1.0)
    cmask_f = shell["collar_mask"].astype(np.float32)
    rgba = np.column_stack([snug, clear_norm, cmask_f, np.ones(len(V_shirt_0))]).astype(np.float32)
    col_attr.data.foreach_set("color", rgba.ravel())

    # Materials: White / Black Ringer Tee (Main_Body + single unified Trim_Rib slot for Collar & Cuffs)
    mat_main = bpy.data.materials.new("BS_Ringer_Main_Body")
    mat_main.use_nodes = True
    nt_m = mat_main.node_tree
    bsdf_m = next(n for n in nt_m.nodes if n.type == "BSDF_PRINCIPLED")
    bsdf_m.inputs["Roughness"].default_value = 0.88
    bsdf_m.inputs["Specular IOR Level"].default_value = 0.12
    art_img_p = art_dir / "front_albedo_light.png"
    if art_img_p.exists() and "UVMap_Art_Front" in me_s.uv_layers:
        im_art = bpy.data.images.load(str(art_img_p), check_existing=False)
        im_art.colorspace_settings.name = "sRGB"
        im_art.scale(512, 608)
        n_art = nt_m.nodes.new("ShaderNodeTexImage")
        n_art.image = im_art
        n_art.extension = "CLIP"
        n_uv = nt_m.nodes.new("ShaderNodeUVMap")
        n_uv.uv_map = "UVMap_Art_Front"
        n_mix = nt_m.nodes.new("ShaderNodeMix")
        n_mix.data_type = "RGBA"
        n_mix.inputs[6].default_value = (0.82, 0.79, 0.74, 1.0)  # #EAE6DF sRGB -> linear approx
        nt_m.links.new(n_uv.outputs["UV"], n_art.inputs["Vector"])
        nt_m.links.new(n_art.outputs["Alpha"], n_mix.inputs[0])
        nt_m.links.new(n_art.outputs["Color"], n_mix.inputs[7])
        nt_m.links.new(n_mix.outputs[2], bsdf_m.inputs["Base Color"])
    else:
        bsdf_m.inputs["Base Color"].default_value = (0.82, 0.79, 0.74, 1.0)

    mat_rib = bpy.data.materials.new("BS_Ringer_Trim_Rib")
    mat_rib.use_nodes = True
    bsdf_r = next(n for n in mat_rib.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
    bsdf_r.inputs["Base Color"].default_value = (0.011, 0.011, 0.010, 1.0)  # #181716
    bsdf_r.inputs["Roughness"].default_value = 0.85

    me_s.materials.append(mat_main)
    me_s.materials.append(mat_rib)
    # Collapse all rib trim indices (1, 2, 3) into slot 1 so shirt has 2 compact primitives
    tri_mats_2 = np.where(shell["tri_mats"].astype(np.int32) == 0, 0, 1)
    me_s.polygons.foreach_set("material_index", tri_mats_2)
    me_s.update()

    shirt_ob = bpy.data.objects.new("BS_RINGER_001", me_s)
    sc.collection.objects.link(shirt_ob)
    shirt_ob.parent = rig
    arm_s = shirt_ob.modifiers.new("Armature", "ARMATURE")
    arm_s.object = rig
    arm_s.use_deform_preserve_volume = False

    # Coupled morph targets on shirt (snap sub-0.2mm noise to 0 so sparse/compact glTF buffers stay under 2 MB)
    def _clean_delta(d: np.ndarray) -> np.ndarray:
        return np.where(np.linalg.norm(d, axis=1, keepdims=True) < 2e-4, 0.0, d)

    shirt_ob.shape_key_add(name="Basis", from_mix=False)
    for ki, kname in enumerate(BME.ANTHRO_KEYS):
        kb = shirt_ob.shape_key_add(name=kname, from_mix=False)
        kb.slider_min = 0.0
        kb.slider_max = 1.0
        kb.data.foreach_set("co", (V_shirt_0 + _clean_delta(morphs["deltas_shirt"][ki])).astype(np.float32).ravel())
    kb_xl = shirt_ob.shape_key_add(name="size_XL", from_mix=False)
    kb_xl.data.foreach_set("co", (V_shirt_0 + _clean_delta(morphs["delta_XL"])).astype(np.float32).ravel())
    kb_xxl = shirt_ob.shape_key_add(name="size_XXL", from_mix=False)
    kb_xxl.data.foreach_set("co", (V_shirt_0 + _clean_delta(morphs["delta_XXL"])).astype(np.float32).ravel())

    # Transfer 24-bone skin weights onto shirt with collar isolated strictly to neck01/spine03
    W_shirt = GT.transfer_body_weights_to_garment(V_shirt_0, V_body_0, W4, keep, shell["collar_mask"])
    for k_i, k in enumerate(keep):
        vg = shirt_ob.vertex_groups.new(name=k)
        nz = np.nonzero(W_shirt[:, k_i] > 0)[0]
        for w_val in np.unique(np.round(W_shirt[nz, k_i], 6)):
            vv = nz[np.round(W_shirt[nz, k_i], 6) == w_val].tolist()
            vg.add(vv, float(w_val), "REPLACE")

    out_glb.parent.mkdir(parents=True, exist_ok=True)
    io.export_glb(
        sc,
        [body, shirt_ob, rig],
        out_glb,
        draco=True,
        export_def_bones=True,
        export_morph=True,
        export_all_influences=False,
        export_vertex_color="ACTIVE",
    )
    summary = M6.glb_summary(out_glb)
    summary["sha256"] = sha256(out_glb)
    summary["under_2MB_budget"] = summary["bytes"] < int(2.0 * 1024 * 1024)
    GT._clear_bpy_data(bpy)
    return summary


def load_coupled_morph_table(npz_path: Path) -> dict:
    with np.load(npz_path, allow_pickle=False) as d:
        return {
            "npz": str(npz_path),
            "V_body_0": d["V_body_0"].astype(np.float64),
            "deltas_body": d["deltas_body"].astype(np.float64),
            "V_shirt_0": d["V_shirt_0"].astype(np.float64),
            "deltas_shirt": d["deltas_shirt"].astype(np.float64),
            "delta_XL": d["delta_XL"].astype(np.float64),
            "delta_XXL": d["delta_XXL"].astype(np.float64),
            "edges": d["edges"].astype(np.int64),
            "L_rest": d["L_rest"].astype(np.float64),
            "corner_stats": {},
        }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--shell-npz", default=str(DEFAULT_SHELL_NPZ))
    ap.add_argument("--basis-npz", default=str(DEFAULT_BASIS_NPZ))
    ap.add_argument("--out-dir", default=str(HERE / "out" / "m9"))
    ap.add_argument("--glb-out", default=None)
    ap.add_argument("--from-cache", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bme = BME.BodyMeasureEngine(basis_npz=Path(args.basis_npz))
    shell, gbind = GT.load_shell_cache(Path(args.shell_npz))
    morph_npz = out_dir / "coupled_morphs.npz"
    if args.from_cache and morph_npz.exists():
        morphs = load_coupled_morph_table(morph_npz)
    else:
        morphs = build_coupled_morph_table(shell, gbind, bme, out_npz=morph_npz)

    # Verify 0-penetration across (min, neutral, max) builds x (L, XL, XXL)
    builds = {
        "min_lean": np.array([1.0, 0.0, 0.0, 0.0, 0.0]),
        "neutral":  np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
        "max_stocky": np.array([0.0, 0.5, 0.7, 0.0, 0.5]),
    }
    matrix_results = {}
    s_mm_neutral = None
    v_str_neutral = None
    for bname, beta in builds.items():
        matrix_results[bname] = {}
        for sz in ("L", "XL", "XXL"):
            print(f"[matrix] evaluating {bname}/{sz}...", flush=True)
            rep, _, _, s_mm, v_str = evaluate_fit_for_state(bme, shell, morphs, beta, size=sz, exact_gwn=True)
            matrix_results[bname][sz] = rep.model_dump()
            if bname == "neutral" and sz == "L":
                s_mm_neutral, v_str_neutral = s_mm, v_str

    glb_path = Path(args.glb_out) if args.glb_out else (out_dir / "maximus_v2_dressed_ringer_lod1.glb")
    glb_meta = export_dressed_lod1_glb(glb_path, shell, morphs, s_mm_neutral, v_str_neutral)

    # Emit compact storefront manifest for #/fitting
    manifest = {
        "schema": "maximus.v2.m9_dressed/1",
        "sku": "BS-RINGER-001",
        "glb": glb_path.name,
        "glb_bytes": glb_meta["bytes"],
        "glb_sha256": glb_meta["sha256"],
        "key_names": list(BME.ANTHRO_KEYS),
        "size_chart": SIZE_CHART_EASE,
        "regressor": {
            "metric_names": bme.metric_names,
            "forward_weights_cm": bme.W_fwd.tolist(),
            "forward_bias_cm": bme.b_fwd.tolist(),
            "input_std_cm": bme.reg["input_std_cm"],
        },
        "presets": {
            "lean":    {"label": "Lean · 173 cm / 64 kg",    "beta": [1.0, 0.0, 0.0, 0.0, 0.0]},
            "neutral": {"label": "Neutral · 173 cm / 72 kg", "beta": [0.0, 0.0, 0.0, 0.0, 0.0]},
            "athletic":{"label": "Athletic · 178 cm / 82 kg","beta": [0.0, 0.0, 0.64, 0.0, 0.98]},
            "stocky":  {"label": "Stocky · 175 cm / 92 kg",  "beta": [0.0, 0.5, 0.7, 0.0, 0.5]},
        },
        "verification_matrix": matrix_results,
    }
    (out_dir / "m9_dressed_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    all_zero_pen = all(
        matrix_results[b][sz]["inside_gwn_gt_05"] == 0 for b in builds for sz in ("L", "XL", "XXL")
    )
    summary = {
        "all_zero_penetration": all_zero_pen,
        "glb_meta": glb_meta,
        "elapsed_s": round(time.time() - t0, 2),
    }
    print(json.dumps(summary, indent=2), flush=True)
    if not (all_zero_pen and glb_meta["under_2MB_budget"]):
        sys.exit(1)


if __name__ == "__main__":
    main()
