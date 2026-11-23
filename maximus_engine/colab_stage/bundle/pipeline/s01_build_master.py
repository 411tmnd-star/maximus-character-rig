"""Stage 01 (Blender): derive maximus_v2_master.blend from the read-only canonical rig. Pure bpy.data / NumPy.

1. verify the source hash (1be2d4ca...); the source file is never saved (the copy is written by the gateway)
2. faceless egg head IN PLACE (maximus_core.egg_head): positions only; the same displacement is added to every
   shape-key block, so each morph's delta (key - basis) is preserved. The full-weight head patch is embedded on the
   egg without folds (egg_head.unfold: harmonic map + guarded tangential relaxation)
3. drop the missing mannequin_wire_2048.png and the old matte material; build procedural M1/M2/M3 + joint seams
4. 'rest_pos' and 'seam_region' point attributes (texture space that follows the skinned surface)
5. Linear Blend Skinning (no dual quaternion) + Corrective Smooth (repeat 5, factor 0.50) limited to the shoulder-cap
   vertex group '_cs_mask_shoulders', which ramps smoothly to zero before the armpit / inner-arm fold
6. invariant check (vertex count, topology, UVs, 152-bone weights, shape-key deltas) before saving
"""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # noqa: E401,E702
import json

import numpy as np

import bl_common as C
import io_ops_gateway as io
import materials

from corpus_bridge import contracts
from maximus_core import egg_head, geometry

FACE_PREFIX = ("special", "oris", "levator", "eye", "orbicularis", "oculi", "risorius", "temporalis")
CS_BONES = ("clavicle", "shoulder01")   # shoulder cap: least-collapse localised mask measured in tools/diag_cs_masks.py
# Corrective Smooth mask shape (gate G5). The old mask was clip(1.5 * s, 0, 1) cut at 0.01, s = clavicle + shoulder01
# weight: a hard step with a skirt of 1-5 % weights that reached the armpit / inner-arm fold. Those triangles are
# already compressed by LBS, and the smoothing pulled three more per side under 0.25 x rest area in P1. New mask:
#   m = smoothstep((s - CS_LO) / (CS_HI - CS_LO)) * smoothstep((d - CS_R0) / (CS_R1 - CS_R0))
# d = geodesic (edge-path) distance in metres from the fold set {upperarm02 weight > CS_FOLD_MIN}. Inside CS_R0 the
# mask is exactly 0, so every vertex of a fold triangle is pure LBS (the measured CS-off count there is the baseline);
# between CS_R0 and CS_R1 it rises with zero slope at both ends, so there is no step for border triangles to pile
# up against.
CS_FOLD_BONES = ("upperarm02",)
CS_FOLD_MIN = 0.05
CS_LO, CS_HI = 0.10, 0.60
CS_R0, CS_R1 = 0.020, 0.060
NORMAL_BLEND_RINGS = 4


def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def graph_geodesic(V, E, seed, iters=64):
    """Shortest edge-path distance (metres) from the seed vertices: Bellman-Ford sweeps in NumPy. inf = unreachable
    or further than `iters` edges, which is far beyond the ramp."""
    d = np.where(seed, 0.0, np.inf); L = np.linalg.norm(V[E[:, 0]] - V[E[:, 1]], axis=1)
    for _ in range(iters):
        prev = d.copy()
        np.minimum.at(d, E[:, 0], prev[E[:, 1]] + L); np.minimum.at(d, E[:, 1], prev[E[:, 0]] + L)
        if np.array_equal(d, prev):
            break
    return d


def head_weight(W, names):
    cols = [i for i, n in enumerate(names) if n in ("head", "jaw") or n.startswith(FACE_PREFIX)]
    return np.clip(W[:, cols].sum(1), 0.0, 1.0).astype(np.float64)


def set_positions(me, X, delta):
    me.vertices.foreach_set("co", X.astype(np.float32).ravel())
    if me.shape_keys:
        for kb in me.shape_keys.key_blocks:
            a = np.empty(len(kb.data) * 3, np.float32); kb.data.foreach_get("co", a)
            a = (a.reshape(-1, 3).astype(np.float64) + delta).astype(np.float32)
            kb.data.foreach_set("co", a.ravel())
    me.update()


def add_point_attr(me, name, typ, values):
    if name in me.attributes:
        me.attributes.remove(me.attributes[name])
    at = me.attributes.new(name, typ, "POINT")
    at.data.foreach_set("vector" if typ == "FLOAT_VECTOR" else "value", np.asarray(values, np.float32).ravel())


def main():
    src = C.bpy.data.filepath
    if contracts.sha256_file(src) != C.V1_SHA256:
        raise RuntimeError("source hash mismatch")
    body, rig = C.human_and_rig(); me = body.data
    fp_before = C.fingerprint(body)
    keys_before = C.shape_key_arrays(me)
    arr = C.mesh_arrays(me); V = arr["V"]; T = arr["T"]
    W, names = C.weights_matrix(body)

    # --- 2. egg head
    w = head_weight(W, names)
    egg = egg_head.estimate_egg(V, w)
    X, rep = egg_head.fit(V, T, w, egg)
    head_tris = np.all(w[T] > 0.99, axis=1)
    a_rest = geometry.tri_areas(V, T); a_new = geometry.tri_areas(X, T)
    n = geometry.face_normals(X, T); rad = X[T].mean(1) - np.asarray(egg["centre"])
    inward = head_tris & (np.einsum("ij,ij->i", n, rad) < 0)
    rep["inward_area_fraction_of_head"] = float(a_new[inward].sum() / max(a_new[head_tris].sum(), 1e-12))
    rep["head_surface_area_m2_before_after"] = [float(a_rest[head_tris].sum()), float(a_new[head_tris].sum())]
    set_positions(me, X, X - V)

    # --- 3. materials (drop the wire texture dependency)
    for img in list(C.bpy.data.images):
        if img.name.startswith("mannequin_wire"):
            C.bpy.data.images.remove(img)
    old = [s.material for s in body.material_slots if s.material]
    bones = {b.name: b for b in rig.data.bones}
    planes = []
    for s in ("L", "R"):
        wrist = np.array(bones[f"wrist.{s}"].head_local); elbow = np.array(bones[f"lowerarm01.{s}"].head_local)
        shoulder = np.array(bones[f"upperarm01.{s}"].head_local)
        fore = wrist - elbow; upper = elbow - shoulder
        planes.append({"point": tuple(wrist), "normal": tuple(fore / np.linalg.norm(fore)), "radius": 0.045})
        planes.append({"point": tuple(elbow), "normal": tuple(upper / np.linalg.norm(upper)), "radius": 0.060})
    seams = materials.seam_group(planes)
    mats = {k: materials.finish_material(k, spec, seams) for k, spec in C.FINISHES.items()}
    body.material_slots[0].material = mats["M2"]
    for m in old:
        if m.users == 0:
            C.bpy.data.materials.remove(m)
    for p in me.polygons:
        p.use_smooth = True
    # shading normals: analytic egg normals inside the head, Blender's own smooth normals on the neck and below, and a
    # smoothstep mix over NORMAL_BLEND_RINGS edge rings in between, so the normal field has no step at the egg / neck
    # ring (custom split normals; geometry, UVs and weights untouched). The head patch is embedded without folds
    # (egg_head.unfold), so no back-facing triangle is left for Cycles to shade with a flipped normal.
    me.update()
    VN = np.empty(len(me.vertices) * 3, np.float32); me.vertex_normals.foreach_get("vector", VN); VN = VN.reshape(-1, 3).astype(np.float64)
    egg_v = w > 0.99
    nb = egg_head.normal_blend(T, w, rings=NORMAL_BLEND_RINGS)[egg_v][:, None]
    mix = (1.0 - nb) * VN[egg_v] + nb * egg_head.surface_normals(egg, X[egg_v])
    VN[egg_v] = mix / np.maximum(np.linalg.norm(mix, axis=1, keepdims=True), 1e-15)
    me.normals_split_custom_set_from_vertices([tuple(n) for n in VN])
    rep["custom_normals_on_egg_vertices"] = int(egg_v.sum()); rep["normal_blend_rings"] = NORMAL_BLEND_RINGS
    rep["normal_blend_full_egg_vertices"] = int((nb[:, 0] >= 1.0).sum())

    # --- 4. attributes
    add_point_attr(me, "rest_pos", "FLOAT_VECTOR", X)
    seam_cols = [i for i, n_ in enumerate(names) if n_.startswith(("upperarm02", "lowerarm01", "lowerarm02", "wrist"))]
    add_point_attr(me, "seam_region", "FLOAT", np.clip(W[:, seam_cols].sum(1), 0, 1))

    # --- 5. skinning: LBS + localised Corrective Smooth
    arm_mod = [m for m in body.modifiers if m.type == "ARMATURE"][0]
    arm_mod.use_deform_preserve_volume = False          # LBS, no dual quaternion
    cs_cols = [i for i, n_ in enumerate(names) if n_.split(".")[0] in CS_BONES]
    fold_cols = [i for i, n_ in enumerate(names) if n_.split(".")[0] in CS_FOLD_BONES]
    fold = W[:, fold_cols].sum(1) > CS_FOLD_MIN
    d_fold = graph_geodesic(X, geometry.unique_edges(T), fold)
    cs_w = smoothstep((W[:, cs_cols].sum(1).astype(np.float64) - CS_LO) / (CS_HI - CS_LO)) * smoothstep((d_fold - CS_R0) / (CS_R1 - CS_R0))
    vg = body.vertex_groups.get("_cs_mask_shoulders") or body.vertex_groups.new(name="_cs_mask_shoulders")
    vg.remove(list(range(len(me.vertices))))
    for i in np.nonzero(cs_w > 0.01)[0]:
        vg.add([int(i)], float(cs_w[i]), "REPLACE")
    cs = body.modifiers.get("CorrectiveSmooth") or body.modifiers.new("CorrectiveSmooth", "CORRECTIVE_SMOOTH")
    cs.factor = 0.50; cs.iterations = 5; cs.smooth_type = "SIMPLE"; cs.rest_source = "ORCO"; cs.vertex_group = vg.name
    cs.use_only_smooth = False; cs.use_pin_boundary = False

    # --- 6. invariants
    fp_after = C.fingerprint(body)
    keys_after = C.shape_key_arrays(me)
    kd = {k: float(np.abs((keys_after[k] - keys_after["Basis"]) - (keys_before[k] - keys_before["Basis"])).max())
          for k in keys_before if k != "Basis"}
    inv = {"n_verts": fp_after["n_verts"] == fp_before["n_verts"] == 13380,
           "topology": fp_after["topology_sha"] == fp_before["topology_sha"], "uv": fp_after["uv_sha"] == fp_before["uv_sha"],
           "bone_weights": fp_after["bone_weights_sha"] == fp_before["bone_weights_sha"] and fp_after["n_bones"] == 152,
           "shape_keys": fp_after["shape_keys"] == fp_before["shape_keys"] and max(kd.values()) < 1e-6,
           "lbs_only": not arm_mod.use_deform_preserve_volume}
    if not all(inv.values()):
        raise RuntimeError(f"invariant failed: {inv} key_delta_drift={kd}")

    body["maximus"] = json.dumps({"version": "v2", "source_sha256": C.V1_SHA256, "egg": egg})
    C.OUT.mkdir(parents=True, exist_ok=True)
    io.save_copy(C.MASTER)
    np.savez_compressed(C.CACHE / "v2_rest.npz", V=X, T=T, w_head=w, cs_mask=cs_w)
    contracts.atomic_write_json(C.OUT / "reports" / "s01_master_report.json",
                                {"egg_fit": rep, "invariants": inv, "shape_key_delta_drift_m": kd, "fingerprint_before": fp_before,
                                 "fingerprint_after": fp_after, "materials": {k: m.name for k, m in mats.items()}, "seam_planes": planes,
                                 "corrective_smooth": {"factor": cs.factor, "iterations": cs.iterations, "rest_source": cs.rest_source,
                                                       "vertex_group": vg.name, "n_vertices": int((cs_w > 0.01).sum()),
                                                       "mask": {"weight_ramp": [CS_LO, CS_HI], "fold_bones": list(CS_FOLD_BONES),
                                                                "fold_weight_min": CS_FOLD_MIN, "fold_ramp_m": [CS_R0, CS_R1],
                                                                "fold_seed_vertices": int(fold.sum()),
                                                                "vertices_zeroed_by_fold_ramp": int(((W[:, cs_cols].sum(1) > CS_LO) & (d_fold <= CS_R0)).sum())}},
                                 "gateway_calls": io.CALLS})
    return ({"master": str(C.MASTER), "master_sha256": contracts.sha256_file(C.MASTER), "report": str(C.OUT / "reports" / "s01_master_report.json")},
            {"invariants": inv, "egg_max_move_mm": rep["max_move_mm"], "head_tris_inward": rep["head_tris_facing_inward"],
             "egg_unfold": rep.get("unfold"), "cs_mask_vertices": int((cs_w > 0.01).sum()),
             "inward_area_fraction": rep["inward_area_fraction_of_head"], "virtual_edges": rep["virtual_edges"]})


C.run("s01_build_master", main)
