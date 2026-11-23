"""Blender-side helpers shared by the Maximus pipeline stages. Pure bpy.data / NumPy: no bpy.ops here.

Every stage is launched by corpus_bridge.blender_runner (EC-04) as
    blender.exe --background <file.blend> --factory-startup --python <stage.py> -- key=value ...
and ends by printing one EC-02 envelope.
"""
import json, os, sys, traceback
from pathlib import Path

import bpy
import numpy as np

ROOT = Path(__file__).resolve().parents[1]           # Maximus_build
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "pipeline"))
import corpus_link  # noqa: E402

corpus_link.link(strict=os.environ.get("MAXIMUS_STRICT", "1") == "1")
from corpus_bridge import contracts  # noqa: E402

OUT = ROOT / "out"
CACHE = OUT / "cache"
V1_BLEND = Path(os.path.normpath(os.path.expanduser('~/Desktop/4041_MechMix/02_engines/better_style_engine/02_engine_3d/isolated_clones/better_style_catalog_collar/out_fbl_008/inputs/03_full_rigged.blend')))
V1_SHA256 = "1be2d4ca305dca0a197215f6adc8181a2fe9853b03f5d316e77169e3aee82148"
MASTER = OUT / "maximus_v2_master.blend"
THETA_FIT = Path(os.path.normpath(os.path.expanduser('~/Desktop/4041_MechMix/02_engines/better_style_engine/02_engine_3d/avatar_engine/poses/theta_fit.json')))
CANONICAL_POSES = Path(os.path.normpath(os.path.expanduser('~/Desktop/artifacts/02_documents_and_research/DAMN/canonical_poses.json')))

# Spec finishes (canonical_poses_spec PDF / MAXIMUS_LOOK). Hex is sRGB; converted to linear before use.
# M1 specular response: base albedo of #18181B is 0.0091 linear. A Principled dielectric at the default Specular IOR
# Level 0.5 (F0 = 0.04) and roughness 0.68 reflects about 4-5 % of ANY light over a broad lobe, five times the diffuse
# term, so the default lifted the whole body to mid-grey. M1 therefore lowers the rough base lobe (specular 0.22 ->
# F0 = 0.0176) and adds a thin, sharper clear coat (weight 0.10, roughness 0.20): the coat only returns light along
# mirror directions, which gives crisp rim / edge highlights without raising the body tone. Roughness stays 0.68.
FINISHES = {
    "M1": {"name": "Obsidian Noir", "hex": "#18181B", "roughness": 0.68, "sss": 0.00, "backdrop": ["#f4f1ea", "#cbc4b8"],
           "specular": 0.22, "coat": 0.10, "coat_roughness": 0.20},
    "M2": {"name": "Alabaster Chalk", "hex": "#F2EFE9", "roughness": 0.75, "sss": 0.05, "backdrop": ["#34322f", "#151413"]},
    "M3": {"name": "Travertine Warm", "hex": "#D8CEBE", "roughness": 0.80, "sss": 0.02, "backdrop": ["#e6e0d6", "#b5ab9c"]},
}


def args():
    a = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    return dict(x.split("=", 1) for x in a if "=" in x)


def srgb_to_linear(c):
    c = np.asarray(c, float)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def hex_linear(h):
    h = h.lstrip("#")
    return tuple(srgb_to_linear([int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)]).tolist())


def human_and_rig():
    body = bpy.data.objects["Human"]; rig = bpy.data.objects["Human.rig"]
    return body, rig


def mesh_arrays(me):
    """Basis positions, triangles (+ source polygon), loop UVs, polygon loop starts. All via foreach_get."""
    n = len(me.vertices)
    V = np.empty(n * 3, np.float64); me.vertices.foreach_get("co", V); V = V.reshape(n, 3)
    me.calc_loop_triangles()
    nt = len(me.loop_triangles)
    T = np.empty(nt * 3, np.int64); me.loop_triangles.foreach_get("vertices", T); T = T.reshape(nt, 3)
    TP = np.empty(nt, np.int64); me.loop_triangles.foreach_get("polygon_index", TP)
    uv = np.empty(len(me.loops) * 2, np.float64); me.uv_layers[0].data.foreach_get("uv", uv)
    lv = np.empty(len(me.loops), np.int64); me.loops.foreach_get("vertex_index", lv)
    return {"V": V, "T": T, "TP": TP, "uv": uv.reshape(-1, 2), "loop_vert": lv}


def weights_matrix(obj):
    """Dense (n_verts, n_groups) float32 weight matrix and group names."""
    me = obj.data; names = [g.name for g in obj.vertex_groups]
    W = np.zeros((len(me.vertices), len(names)), np.float32)
    for v in me.vertices:
        for g in v.groups:
            W[v.index, g.group] = g.weight
    return W, names


def shape_key_arrays(me):
    out = {}
    if me.shape_keys:
        for kb in me.shape_keys.key_blocks:
            a = np.empty(len(kb.data) * 3, np.float64); kb.data.foreach_get("co", a); out[kb.name] = a.reshape(-1, 3)
    return out


def bone_table(rig):
    return {b.name: {"head": list(b.head_local), "tail": list(b.tail_local), "parent": b.parent.name if b.parent else None}
            for b in rig.data.bones}


def evaluated_positions(obj):
    """World-space vertex positions of the evaluated (armature + modifiers) mesh, same vertex order."""
    dg = bpy.context.evaluated_depsgraph_get(); ev = obj.evaluated_get(dg)
    me = ev.to_mesh()
    try:
        n = len(me.vertices); V = np.empty(n * 3, np.float64); me.vertices.foreach_get("co", V)
        M = np.array(obj.matrix_world, float)
        V = V.reshape(n, 3) @ M[:3, :3].T + M[:3, 3]
    finally:
        ev.to_mesh_clear()
    return V


def fingerprint(obj):
    """Hashes of the invariants that must survive the rebuild (owner rule 1)."""
    import hashlib
    me = obj.data
    def h(a): return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
    arr = mesh_arrays(me); W, names = weights_matrix(obj)
    pv = np.empty(len(me.polygons) * 0 + len(me.loops), np.int64); me.loops.foreach_get("vertex_index", pv)
    bone_names = set(b.name for b in obj.parent.data.bones) if obj.parent else set()
    bone_cols = [i for i, n in enumerate(names) if n in bone_names]
    keys = shape_key_arrays(me)
    basis = keys.get("Basis")
    deltas = {k: h(np.round(v - basis, 9)) for k, v in keys.items() if k != "Basis"} if basis is not None else {}
    return {"n_verts": len(me.vertices), "n_polys": len(me.polygons), "n_loops": len(me.loops),
            "topology_sha": h(pv), "uv_sha": h(np.round(arr["uv"], 9)),
            "bone_weights_sha": h(np.round(W[:, bone_cols], 6)), "bone_groups": [names[i] for i in bone_cols],
            "n_bones": len(obj.parent.data.bones) if obj.parent else 0,
            "shape_keys": [k for k in keys], "shape_key_delta_sha": deltas}


def run(stage, fn):
    """Run fn() -> (outputs, metrics); always emit an EC-02 envelope."""
    try:
        outputs, metrics = fn()
        contracts.emit(contracts.envelope(stage, "ok", inputs={"blend": bpy.data.filepath, "args": args()},
                                          outputs=outputs, metrics=metrics))
    except Exception as e:  # noqa: BLE001
        contracts.emit(contracts.envelope(stage, "fail", inputs={"blend": bpy.data.filepath, "args": args()},
                                          errors=[repr(e), traceback.format_exc()]))
        sys.exit(3)
