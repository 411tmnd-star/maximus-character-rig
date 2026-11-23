"""Dev tool (Blender): render the head region with candidate vertex positions, without saving anything.
  blender --background 03_full_rigged.blend --factory-startup --python tools/preview_head.py -- positions=<.npy> out=<png prefix>
"""
import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pipeline"))  # noqa
import math

import numpy as np

import bl_common as C
import io_ops_gateway as io

bpy = C.bpy


def look_at(cam, target):
    from mathutils import Vector
    d = Vector(target) - cam.location
    cam.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()


def main():
    a = C.args(); body, rig = C.human_and_rig(); me = body.data
    if "positions" in a:
        X = np.load(a["positions"]).astype(np.float64).ravel()
        me.vertices.foreach_set("co", X)
        if me.shape_keys:
            me.shape_keys.key_blocks["Basis"].data.foreach_set("co", X)
        me.update()
    mat = bpy.data.materials.new("preview_grey"); mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF"); bsdf.inputs["Base Color"].default_value = (0.6, 0.58, 0.55, 1); bsdf.inputs["Roughness"].default_value = 0.7
    body.material_slots[0].material = mat
    for p in me.polygons: p.use_smooth = True
    sc = bpy.context.scene; sc.render.engine = "CYCLES"; sc.cycles.device = "CPU"; sc.cycles.samples = int(a.get("samples", 24))
    sc.cycles.use_denoising = True; sc.render.resolution_x = int(a.get("rx", 420)); sc.render.resolution_y = int(a.get("ry", 520))
    sc.world = sc.world or bpy.data.worlds.new("W"); sc.world.color = (0.05, 0.05, 0.05)
    key = bpy.data.lights.new("key", "AREA"); key.energy = 120; key.size = 1.0
    ko = bpy.data.objects.new("key", key); sc.collection.objects.link(ko); ko.location = (0.8, -1.2, 2.2); look_at(ko, (0, -0.05, 1.6))
    fill = bpy.data.lights.new("fill", "AREA"); fill.energy = 40; fill.size = 2.0
    fo = bpy.data.objects.new("fill", fill); sc.collection.objects.link(fo); fo.location = (-1.2, -0.8, 1.7); look_at(fo, (0, -0.05, 1.6))
    cam = bpy.data.cameras.new("cam"); cam.lens = 105; co = bpy.data.objects.new("cam", cam); sc.collection.objects.link(co); sc.camera = co
    tgt = (0, -0.05, float(a.get("tz", 1.60))); dist = float(a.get("dist", 1.6))
    for name, ang in (("front", 0), ("q34", 35), ("side", 90)):
        r = math.radians(ang); co.location = (dist * math.sin(r), -dist * math.cos(r), tgt[2] + 0.02); look_at(co, tgt)
        io.render_still(sc, a["out"] + f"_{name}.png")
    return {"renders": a["out"]}, {}


C.run("preview_head", main)
