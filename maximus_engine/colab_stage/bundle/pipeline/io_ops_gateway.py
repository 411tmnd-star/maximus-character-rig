"""The ONLY module in Maximus_build allowed to call bpy.ops.

Owner rule (2026-11-16): zero bpy.ops for mesh, rig, vertex-group, shape-key, material or pose work; those use
bpy.data / bmesh / NumPy. The operators below have no data-API equivalent in Blender 5.2, so they are isolated here,
each with an explicit context so it does not depend on UI state (active object, selection, mode):

  save_copy()    bpy.ops.wm.save_as_mainfile(copy=True)  write a .blend (the open source file is never overwritten)
  render_still() bpy.ops.render.render(write_still=True)   final Cycles still
  bake()         bpy.ops.object.bake(...)                   texture bake (needs one active object + active image node)
  export_glb()   bpy.ops.export_scene.gltf(...)             Draco GLB export of explicitly listed objects

Files are opened by passing the .blend on the Blender command line (no operator needed).
grep check: tests/test_no_bpy_ops.py fails if bpy.ops appears in any other pipeline file.
"""
import os

import bpy

CALLS = []  # audit trail of every operator call in this process


def _log(name, **kw):
    CALLS.append({"op": name, **{k: str(v) for k, v in kw.items()}})


def save_copy(path):
    path = os.path.abspath(str(path))
    _log("wm.save_as_mainfile", path=path)
    return bpy.ops.wm.save_as_mainfile(filepath=str(path), copy=True, compress=True)


def render_still(scene, path):
    path = os.path.abspath(str(path))
    scene.render.filepath = path
    _log("render.render", scene=scene.name, path=path)
    with bpy.context.temp_override(scene=scene):
        return bpy.ops.render.render(write_still=True, scene=scene.name)


def resave_render(scene, path):
    """Write the last render again through the scene's current view settings (exposure / look), without rendering.
    Data API only (Image.save_render on the Render Result); logged like the operator calls."""
    path = os.path.abspath(str(path))
    _log("data.save_render", scene=scene.name, path=path, exposure=round(scene.view_settings.exposure, 4))
    rr = next(i for i in bpy.data.images if i.type == "RENDER_RESULT")
    rr.save_render(filepath=path, scene=scene)


def bake(scene, obj, bake_type, **kw):
    """obj must have an active Image Texture node in every material slot (set by the caller via bpy.data)."""
    vl = scene.view_layers[0]
    for o in scene.objects:
        o.select_set(o == obj, view_layer=vl)
    vl.objects.active = obj
    _log("object.bake", obj=obj.name, type=bake_type, **kw)
    with bpy.context.temp_override(scene=scene, view_layer=vl, active_object=obj, object=obj, selected_objects=[obj],
                                   selected_editable_objects=[obj]):
        return bpy.ops.object.bake(type=bake_type, use_clear=True, margin=8, **kw)


def export_glb(scene, objects, path, draco=True, draco_level=7, **kw):
    path = os.path.abspath(str(path))
    vl = scene.view_layers[0]
    for o in scene.objects:
        o.select_set(o in objects, view_layer=vl)
    vl.objects.active = objects[0]
    _log("export_scene.gltf", path=path, objects=[o.name for o in objects], draco=draco)
    opts = dict(filepath=str(path), export_format="GLB", use_selection=True, export_apply=False, export_skins=True,
                export_morph=True, export_morph_normal=False, export_animations=False, export_yup=True,
                export_draco_mesh_compression_enable=draco, export_draco_mesh_compression_level=draco_level,
                export_draco_position_quantization=14, export_draco_normal_quantization=10,
                export_draco_texcoord_quantization=12, export_image_format="WEBP")
    opts.update(kw)
    with bpy.context.temp_override(scene=scene, view_layer=vl, active_object=objects[0], selected_objects=list(objects)):
        return bpy.ops.export_scene.gltf(**opts)
