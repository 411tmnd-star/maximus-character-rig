"""Shared still-render helpers for the M4 / M5 / runtime_v3 Blender passes (bpy.data only; the render itself goes through io_ops_gateway).

Copied from pipeline/s05_previews.py (look_at, light rig, 105 mm camera, frame_camera solve, image_stats) because s05 runs its own main()
when imported. Same constants, same maths: the crown sits TOP_MARGIN below the top edge and world height FADE_Z sits on the 82 % fade line.
"""
import math

import numpy as np

LENS_MM, SENSOR_MM = 105.0, 36.0
TAN_HALF = (SENSOR_MM / 2) / LENS_MM
TOP_MARGIN = 0.07
FADE_FRAC = 0.82
FADE_Z = 0.852
TARGET_XY = (0.0, -0.02)
LIGHT_TARGET = (0.0, -0.02, 1.22)
BASE_W = {"MX_key": 450.0, "MX_fill": 140.0 * 0.5, "MX_rim": 260.0}     # s05 applies fill_scale 0.5 by default


def look_at(obj, target):
    from mathutils import Vector
    obj.rotation_euler = (Vector(target) - obj.location).to_track_quat("-Z", "Y").to_euler()


def light(bpy, sc, name, kind, energy, size, loc, target):
    ld = bpy.data.lights.new(name, kind); ld.energy = energy
    if kind == "AREA":
        ld.size = size
    ob = bpy.data.objects.new(name, ld); sc.collection.objects.link(ob); ob.location = loc; look_at(ob, target); return ob


def studio(bpy, sc, samples=128, res=(1638, 2048), exposure=-1.0, world=(0.03, 0.03, 0.03), weights=None):
    """s05's three-light rig + 105 mm camera. weights: optional {light: watts} override (lighting_director output)."""
    sc.cycles.samples = int(samples); sc.cycles.use_denoising = True
    sc.render.resolution_x, sc.render.resolution_y = int(res[0]), int(res[1]); sc.render.resolution_percentage = 100
    sc.render.film_transparent = True; sc.render.image_settings.file_format = "PNG"; sc.render.image_settings.color_mode = "RGBA"
    try:
        sc.view_settings.view_transform = "AgX"
    except TypeError:
        sc.view_settings.view_transform = "Standard"
    sc.view_settings.exposure = float(exposure)
    sc.world = sc.world or bpy.data.worlds.new("MX_world"); sc.world.color = world
    w = dict(BASE_W); w.update(weights or {})
    lights = {"MX_key": light(bpy, sc, "MX_key", "AREA", w["MX_key"], 1.6, (1.7, -2.3, 2.7), LIGHT_TARGET),
              "MX_fill": light(bpy, sc, "MX_fill", "AREA", w["MX_fill"], 3.0, (-2.4, -1.6, 1.6), LIGHT_TARGET),
              "MX_rim": light(bpy, sc, "MX_rim", "AREA", w["MX_rim"], 1.2, (-0.8, 2.2, 2.4), (0, 0, 1.45))}
    cd = bpy.data.cameras.new("MX_cam"); cd.lens = LENS_MM; cd.sensor_fit = "VERTICAL"; cd.sensor_height = SENSOR_MM
    cam = bpy.data.objects.new("MX_cam", cd); sc.collection.objects.link(cam); sc.camera = cam
    return cam, lights


def frame_camera(cam, X, view_deg=0.0, top_margin=TOP_MARGIN, fade_z=FADE_Z, fade_frac=FADE_FRAC):
    """s05_previews.frame_camera, unchanged: solve camera distance D and height zc from the posed mesh X (see s05 for the derivation)."""
    r = math.radians(view_deg); c_hat = np.array([math.sin(r), -math.cos(r)])
    a = (X[:, :2] - np.asarray(TARGET_XY)) @ c_hat; z = X[:, 2]
    alpha = 2 * TAN_HALF * (0.5 - top_margin); beta = 2 * TAN_HALF * (fade_frac - 0.5)
    D, zc = 3.4, 1.22
    for _ in range(12):
        k = int(np.argmax((z - zc) / (D - a)))
        D_new = (z[k] - fade_z + alpha * a[k]) / (alpha + beta); zc_new = fade_z + beta * D_new
        moved = abs(D_new - D) + abs(zc_new - zc); D, zc = D_new, zc_new
        if moved < 1e-9:
            break
    cam.location = (TARGET_XY[0] + D * c_hat[0], TARGET_XY[1] + D * c_hat[1], zc)
    look_at(cam, (TARGET_XY[0], TARGET_XY[1], zc))
    top = 0.5 - float(((z - zc) / (D - a)).max()) / (2 * TAN_HALF)
    return {"lens_mm": LENS_MM, "distance_m": round(D, 5), "height_m": round(zc, 5), "view_deg": view_deg, "crown_z_m": round(float(z.max()), 5),
            "predicted_top_margin": round(top, 5), "fade_z_m": fade_z, "fade_frac": fade_frac}


def image_stats(bpy, path):
    """Display-referred stats of a saved RGBA render (sRGB codes, Rec.709 luma, opaque pixels): torso median, top margin, clipping."""
    img = bpy.data.images.load(str(path), check_existing=False)
    try:
        try:
            img.colorspace_settings.name = "Non-Color"
        except TypeError:
            pass
        w, h = img.size; px = np.empty(w * h * 4, np.float32); img.pixels.foreach_get(px); px = px.reshape(h, w, 4)
    finally:
        bpy.data.images.remove(img)
    luma = 255.0 * (0.2126 * px[..., 0] + 0.7152 * px[..., 1] + 0.0722 * px[..., 2]); opaque = px[..., 3] > 0.99
    box = (slice(int(h * 0.40), int(h * 0.60)), slice(int(w * 0.40), int(w * 0.60)))
    torso = luma[box][opaque[box]]; body = luma[opaque]; rows = np.nonzero((px[..., 3] > 0.5).any(axis=1))[0]
    return {"torso_median": round(float(np.median(torso)), 2) if torso.size else None,
            "body_p999": round(float(np.percentile(body, 99.9)), 2) if body.size else None,
            "clipped_fraction_ge_250": round(float((body >= 250).mean()), 6) if body.size else None,
            "top_margin": round(float((h - 1 - rows.max()) / h), 5) if rows.size else None, "opaque_pixels": int(opaque.sum())}
