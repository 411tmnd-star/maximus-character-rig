"""Stage 04 (Blender): bake M1/M2/M3 texture sets from the procedural shaders and export the Draco core GLB.

Procedural Cycles nodes do not survive glTF export, so each finish is baked at rest pose onto the MPFB UVs:
  out/textures/<M>/basecolor_2k.png (sRGB), roughness_2k.png, normal_2k.png (tangent, Non-Color), orm_2k.png
  out/textures/ao_1k.png (shared; finish-independent geometry occlusion)
Core GLB: out/maximus_v2_core.glb = Human (13,380 verts, 5 morphs) + 152-bone skin + M2 baked material (1K WebP),
Draco-compressed, budget < 2.5 MB. Only the gateway calls operators (bake, export). The master is not re-saved.
"""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # noqa: E401,E702
import json, struct

import numpy as np

import bl_common as C
import io_ops_gateway as io

from corpus_bridge import contracts
from maximus_delight import pack_orm

TEX = C.OUT / "textures"
GLB = C.OUT / "maximus_v2_core.glb"
BUDGET = 2.5 * 1024 * 1024


def new_image(name, size, colorspace):
    img = C.bpy.data.images.get(name)
    if img:
        C.bpy.data.images.remove(img)
    img = C.bpy.data.images.new(name, size, size, alpha=False, float_buffer=False)
    img.colorspace_settings.name = colorspace
    return img


def save_png(img, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    img.filepath_raw = str(path); img.file_format = "PNG"; img.save()
    return str(path)


def bake_into(scene, body, mat, img, bake_type, **kw):
    nt = mat.node_tree; node = nt.nodes.new("ShaderNodeTexImage"); node.image = img
    for n in nt.nodes:
        n.select = False
    node.select = True; nt.nodes.active = node
    try:
        io.bake(scene, body, bake_type, **kw)
    finally:
        nt.nodes.remove(node)


def pixels(img):
    a = np.empty(img.size[0] * img.size[1] * 4, np.float32); img.pixels.foreach_get(a)
    return a.reshape(img.size[1], img.size[0], 4)


def glb_summary(path):
    """Parse the GLB JSON chunk (standard library): what the file really contains."""
    b = open(path, "rb").read(); magic, ver, total = struct.unpack_from("<4sII", b, 0)
    clen, ctype = struct.unpack_from("<I4s", b, 12); j = json.loads(b[20:20 + clen])
    prims = [p for m in j.get("meshes", []) for p in m["primitives"]]
    pos_counts = [j["accessors"][p["attributes"]["POSITION"]]["count"] for p in prims]
    return {"bytes": len(b), "gltf_version": j["asset"].get("version"), "extensionsUsed": j.get("extensionsUsed", []),
            "meshes": len(j.get("meshes", [])), "primitives": len(prims), "vertices_gltf": int(sum(pos_counts)),
            "morph_targets": max((len(p.get("targets", [])) for p in prims), default=0),
            "morph_target_names": (j["meshes"][0].get("extras", {}) or {}).get("targetNames", []),
            "skins": len(j.get("skins", [])), "joints": len(j["skins"][0]["joints"]) if j.get("skins") else 0,
            "materials": [m.get("name") for m in j.get("materials", [])], "images": [i.get("mimeType") for i in j.get("images", [])]}


def main():
    if contracts.sha256_file(C.bpy.data.filepath) != contracts.sha256_file(C.MASTER):
        raise RuntimeError("stage 04 must run on out/maximus_v2_master.blend")
    body, rig = C.human_and_rig(); sc = C.bpy.context.scene
    rig.data.pose_position = "REST"
    sc.render.engine = "CYCLES"; sc.cycles.device = "CPU"; sc.render.bake.margin = 8
    size = int(C.args().get("size", 2048)); out = {}
    # shared AO (geometry only)
    m2 = C.bpy.data.materials["MX_M2_AlabasterChalk"]; body.material_slots[0].material = m2
    ao = new_image("MX_ao_1k", 1024, "Non-Color"); sc.cycles.samples = int(C.args().get("ao_samples", 32))
    bake_into(sc, body, m2, ao, "AO"); out["ao"] = save_png(ao, TEX / "ao_1k.png")
    ao_px = pixels(ao)[..., 0]
    sc.cycles.samples = 1
    for key in ("M1", "M2", "M3"):
        mat = [m for m in C.bpy.data.materials if m.name.startswith(f"MX_{key}_")][0]
        body.material_slots[0].material = mat
        bc = new_image(f"MX_{key}_basecolor", size, "sRGB"); bake_into(sc, body, mat, bc, "DIFFUSE", pass_filter={"COLOR"})
        rg = new_image(f"MX_{key}_roughness", size, "Non-Color"); bake_into(sc, body, mat, rg, "ROUGHNESS")
        nm = new_image(f"MX_{key}_normal", size, "Non-Color"); bake_into(sc, body, mat, nm, "NORMAL", normal_space="TANGENT")
        d = TEX / key
        out[key] = {"basecolor": save_png(bc, d / "basecolor_2k.png"), "roughness": save_png(rg, d / "roughness_2k.png"),
                    "normal": save_png(nm, d / "normal_2k.png")}
        r = pixels(rg)[..., 0]
        ao_up = np.kron(ao_px, np.ones((size // 1024, size // 1024), np.float32))
        orm = pack_orm(ao_up, r); orm_img = new_image(f"MX_{key}_orm", size, "Non-Color")
        orm_img.pixels.foreach_set(np.dstack([orm, np.ones(orm.shape[:2])]).astype(np.float32).ravel())
        out[key]["orm"] = save_png(orm_img, d / "orm_2k.png")
        out[key]["roughness_mean"] = float(r.mean())
    # core GLB material: M2 baked, 1K
    core = C.bpy.data.materials.new("MX_core_M2_baked"); core.use_nodes = True; nt = core.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    def tex(name, cs, src):
        im = C.bpy.data.images[src].copy(); im.name = name; im.scale(1024, 1024); im.colorspace_settings.name = cs
        n = nt.nodes.new("ShaderNodeTexImage"); n.image = im; return n
    nb = tex("M2_basecolor_1k", "sRGB", "MX_M2_basecolor"); nt.links.new(nb.outputs["Color"], bsdf.inputs["Base Color"])
    no = tex("M2_orm_1k", "Non-Color", "MX_M2_orm"); sep = nt.nodes.new("ShaderNodeSeparateColor")
    nt.links.new(no.outputs["Color"], sep.inputs["Color"]); nt.links.new(sep.outputs["Green"], bsdf.inputs["Roughness"]); nt.links.new(sep.outputs["Blue"], bsdf.inputs["Metallic"])
    nn = tex("M2_normal_1k", "Non-Color", "MX_M2_normal"); nmap = nt.nodes.new("ShaderNodeNormalMap")
    nt.links.new(nn.outputs["Color"], nmap.inputs["Color"]); nt.links.new(nmap.outputs["Normal"], bsdf.inputs["Normal"])
    body.material_slots[0].material = core
    io.export_glb(sc, [body, rig], GLB, draco=True)
    summ = glb_summary(GLB); summ["sha256"] = contracts.sha256_file(GLB); summ["under_budget"] = summ["bytes"] < BUDGET
    contracts.atomic_write_json(C.OUT / "reports" / "s04_bake_export.json", {"textures": out, "glb": summ, "gateway_calls": io.CALLS})
    if not summ["under_budget"]:
        raise RuntimeError(f"GLB over budget: {summ['bytes']} bytes")
    return {"glb": str(GLB), "textures": out}, {"glb": summ}


C.run("s04_bake_export", main)
