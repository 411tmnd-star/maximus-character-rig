"""Stage 05 (Blender): one low-sample 2K preview per pose x finish (P1-P3 x M1-M3), CPU Cycles, native RGBA.

Camera: 105 mm portrait lens, level, crown-to-mid-thigh bust framing (the storefront fades the bottom with CSS,
82% -> 98%). The camera distance and height are SOLVED per pose from the posed, evaluated mesh (frame_camera):
the projected crown sits TOP_MARGIN (7 %) below the top edge and world height FADE_Z sits on the 82 % fade line.
P1/P2 front view, P3 three-quarter (35 deg) as in MAXIMUS_LOOK. Poses are read from poses/*.json (local quaternions on
the native rig) and applied through pose_bone data, not operators.

Light: one rig, scaled per finish (RIG). M1 Obsidian Noir gets a lower key and fill and a stronger rim, and its view
exposure is trimmed after the render until the median luminance of the lit torso matches RIG["M1"]["target_median"]
(the tone of the MAXIMUS_LOOK black mannequin). The trim re-saves the finished render through the new view exposure;
nothing is rendered twice. M2 and M3 keep the rig and exposure they were calibrated with.
Renders are written by the gateway only. Heavy 4K final renders: Maximus_build/colab_stage/run_batch_4k.py.
"""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # noqa: E401,E702
import json, math
from pathlib import Path

import numpy as np

import bl_common as C
import io_ops_gateway as io
import posing

from corpus_bridge import contracts

POSE_FILES = {"P1": "theta_monolith.json", "P2": "theta_contrapposto.json", "P3": "theta_cropped_editorial.json"}
VIEW_DEG = {"P1": 0.0, "P2": 0.0, "P3": 35.0}

LENS_MM, SENSOR_MM = 105.0, 36.0
TAN_HALF = (SENSOR_MM / 2) / LENS_MM      # tan of half the vertical field of view (sensor fit VERTICAL)
TOP_MARGIN = 0.07                         # crown headroom as a fraction of frame height (accepted band 6-8 %)
FADE_FRAC = 0.82                          # storefront CSS: the bust fade starts at 82 % of the frame height
FADE_Z = 0.852                            # world height (m) that sits on the 82 % line. Unchanged from the previous
#                                           framing (target z 1.22, span 1.15 m: 1.22 + 0.575 - 0.82 * 1.15), so hems
#                                           keep landing at 77-80 % and the crop still ends at mid-thigh.
TARGET_XY = (0.0, -0.02)
LIGHT_TARGET = (0.0, -0.02, 1.22)
BASE_W = {"MX_key": 450.0, "MX_fill": 140.0, "MX_rim": 260.0}   # area-light watts before the per-finish scale
BASE_EXPOSURE = -1.0                      # stops; calibrated for M2 / M3 (M2 body ~ sRGB 201, M3 ~ 188, no clipping)
TORSO_BOX = (0.40, 0.60, 0.40, 0.60)      # x0, x1, y0, y1 as frame fractions: sternum to navel, inside the torso

# Per-finish scale on the rig and starting exposure. Why M1 differs:
#   A white diffuse patch facing an area light of P watts at distance d renders at about P / (4 pi d^2). The key
#   (450 W at 3.2 m) puts ~3.5 on the chest, exposure -1 halves it, so M2 (albedo 0.87) lands near scene value 1.3
#   and AgX shows it at sRGB ~200: correct for chalk. M1's albedo is 0.0091. AgX is a low-contrast curve around
#   0.18: it shows scene value 0.009 at sRGB ~60, not at the 24-27 of the #18181B swatch, and the old material's
#   rough specular lobe added another 4-5 % reflectance on top (see bl_common.FINISHES). Result: mid-grey.
#   So for M1 the key is lowered, the fill is lowered further (deeper core shadow = more sculptural), the rim is
#   raised (crisp edge light on the clear coat), and the exposure is then trimmed to the measured target.
RIG = {
    "M1": {"MX_key": 0.75, "MX_fill": 0.35, "MX_rim": 1.80, "exposure": -0.75, "target_median": 36.0, "tol": 3.0},
    "M2": {"MX_key": 1.00, "MX_fill": 1.00, "MX_rim": 1.00, "exposure": BASE_EXPOSURE},
    "M3": {"MX_key": 1.00, "MX_fill": 1.00, "MX_rim": 1.00, "exposure": BASE_EXPOSURE},
}
CODES_PER_STOP = 22.0                     # first-guess slope of AgX in its toe (sRGB codes per stop near code 35)


def look_at(obj, target):
    from mathutils import Vector
    obj.rotation_euler = (Vector(target) - obj.location).to_track_quat("-Z", "Y").to_euler()


def light(sc, name, kind, energy, size, loc, target):
    ld = C.bpy.data.lights.new(name, kind); ld.energy = energy
    if kind == "AREA":
        ld.size = size
    ob = C.bpy.data.objects.new(name, ld); sc.collection.objects.link(ob); ob.location = loc; look_at(ob, target); return ob


def setup(sc, a):
    sc.render.engine = "CYCLES"; sc.cycles.samples = int(a.get("samples", 16)); sc.cycles.use_denoising = True
    sc.cycles.device = "CPU"
    if a.get("device", "CPU").upper() == "GPU":           # Colab: OptiX, else CUDA (preferences data API, no operator)
        prefs = C.bpy.context.preferences.addons["cycles"].preferences
        for kind in ("OPTIX", "CUDA"):
            try:
                prefs.compute_device_type = kind; prefs.get_devices()
                gpus = [d for d in prefs.devices if d.type == kind]
                if gpus:
                    for d in prefs.devices:
                        d.use = d.type == kind
                    sc.cycles.device = "GPU"; break
            except TypeError:
                continue
    sc.render.resolution_x = int(a.get("rx", 1638)); sc.render.resolution_y = int(a.get("ry", 2048)); sc.render.resolution_percentage = 100
    sc.render.film_transparent = True; sc.render.image_settings.file_format = "PNG"; sc.render.image_settings.color_mode = "RGBA"
    try:                                                   # enum is built at runtime; set it and fall back if absent
        sc.view_settings.view_transform = "AgX"
    except TypeError:
        sc.view_settings.view_transform = "Standard"
    sc.world = sc.world or C.bpy.data.worlds.new("MX_world"); sc.world.color = (0.03, 0.03, 0.03)
    ls = float(a.get("light_scale", 1.0)); base = dict(BASE_W)
    base["MX_key"] *= ls; base["MX_fill"] *= ls * float(a.get("fill_scale", 0.5)); base["MX_rim"] *= ls
    lights = {"MX_key": light(sc, "MX_key", "AREA", base["MX_key"], 1.6, (1.7, -2.3, 2.7), LIGHT_TARGET),   # raking key: shows the stone grain
              "MX_fill": light(sc, "MX_fill", "AREA", base["MX_fill"], 3.0, (-2.4, -1.6, 1.6), LIGHT_TARGET),
              "MX_rim": light(sc, "MX_rim", "AREA", base["MX_rim"], 1.2, (-0.8, 2.2, 2.4), (0, 0, 1.45))}
    cd = C.bpy.data.cameras.new("MX_cam"); cd.lens = LENS_MM; cd.sensor_fit = "VERTICAL"; cd.sensor_height = SENSOR_MM
    cam = C.bpy.data.objects.new("MX_cam", cd); sc.collection.objects.link(cam); sc.camera = cam
    return cam, lights, base


def frame_camera(cam, X, view_deg, top_margin, fade_z, fade_frac=FADE_FRAC):
    """Place the level camera so that (1) the highest projected point of the posed mesh is `top_margin` below the top
    edge and (2) world height `fade_z`, at the depth of the body axis, is on the `fade_frac` line.

    Pinhole, level camera at height zc and distance D from the body axis; a = how far a vertex sits toward the
    camera from that axis. Its height in the frame, as a fraction from the top:
        y = 0.5 - (z - zc) / (2 tan(fov/2) (D - a))
    (1) with the crown vertex k:  z_k - zc = alpha (D - a_k),  alpha = 2 tan(fov/2) (0.5 - top_margin)
    (2)                           zc - fade_z = beta D,        beta  = 2 tan(fov/2) (fade_frac - 0.5)
    =>  D = (z_k - fade_z + alpha a_k) / (alpha + beta),  zc = fade_z + beta D.
    The crown vertex depends on (D, zc) through perspective, so the two lines are iterated until it stops changing.
    The old code used a fixed span and target height on the body-axis plane, ignoring both perspective (the head is
    nearer the lens than that plane) and the posed crown height, which is why the measured headroom was ~3 %."""
    r = math.radians(view_deg); c_hat = np.array([math.sin(r), -math.cos(r)])        # body axis -> camera, in XY
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
    return {"lens_mm": LENS_MM, "distance_m": round(D, 5), "height_m": round(zc, 5), "view_deg": view_deg,
            "span_at_axis_m": round(2 * D * TAN_HALF, 5), "crown_z_m": round(float(z.max()), 5),
            "predicted_top_margin": round(top, 5), "fade_z_m": fade_z, "fade_frac": fade_frac}


def image_stats(path):
    """Display-referred statistics of a saved RGBA render (sRGB code values 0-255, Rec.709 luma). Opaque pixels only."""
    img = C.bpy.data.images.load(str(path), check_existing=False)
    try:
        try:
            img.colorspace_settings.name = "Non-Color"     # read the stored code values, no linearisation
        except TypeError:
            pass
        w, h = img.size; px = np.empty(w * h * 4, np.float32); img.pixels.foreach_get(px); px = px.reshape(h, w, 4)
    finally:
        C.bpy.data.images.remove(img)
    luma = 255.0 * (0.2126 * px[..., 0] + 0.7152 * px[..., 1] + 0.0722 * px[..., 2]); opaque = px[..., 3] > 0.99
    x0, x1, y0, y1 = TORSO_BOX                             # image rows run bottom-up; the box is symmetric in y
    box = (slice(int(h * (1 - y1)), int(h * (1 - y0))), slice(int(w * x0), int(w * x1)))
    torso = luma[box][opaque[box]]; body = luma[opaque]
    rows = np.nonzero((px[..., 3] > 0.5).any(axis=1))[0]
    return {"torso_median": round(float(np.median(torso)), 2) if torso.size else None, "torso_pixels": int(torso.size),
            "body_p50": round(float(np.median(body)), 2) if body.size else None,
            "body_p999": round(float(np.percentile(body, 99.9)), 2) if body.size else None,
            "clipped_fraction_ge_250": round(float((body >= 250).mean()), 6) if body.size else None,
            "top_margin": round(float((h - 1 - rows.max()) / h), 5) if rows.size else None}


def trim_exposure(sc, path, target, tol, e0, st, max_steps=3):
    """Secant search on the view exposure so the lit-torso median reaches `target` (sRGB codes). Each step re-saves the
    finished render through the new exposure. Returns (exposure of the file on disk, its stats, history)."""
    hist = [{"exposure": round(e0, 4), "torso_median": st["torso_median"]}]
    e, m, e_prev, m_prev = e0, st["torso_median"], None, None
    for _ in range(max_steps):
        if m is None or abs(m - target) <= tol:
            break
        k = CODES_PER_STOP if e_prev is None or abs(m - m_prev) < 0.5 else (m - m_prev) / (e - e_prev)
        k = min(max(k, 6.0), 60.0)
        e_new = min(max(e + (target - m) / k, e0 - 3.0), e0 + 3.0)
        sc.view_settings.exposure = e_new; io.resave_render(sc, path)
        st = image_stats(path); e_prev, m_prev, e, m = e, m, e_new, st["torso_median"]
        hist.append({"exposure": round(e, 4), "torso_median": m})
    return e, st, hist


def main():
    a = C.args(); sc = C.bpy.context.scene; body, rig = C.human_and_rig(); rig.data.pose_position = "POSE"
    cam, lights, base_w = setup(sc, a)
    P = posing.Poser(C.bpy, rig); out = {}; cams = {}
    top_margin = float(a.get("top_margin", TOP_MARGIN)); fade_z = float(a.get("fade_z", FADE_Z))
    rigs = {k: dict(v) for k, v in RIG.items()}
    rigs["M1"]["exposure"] = float(a.get("m1_exposure", rigs["M1"]["exposure"])); rigs["M1"]["target_median"] = float(a.get("m1_target", rigs["M1"]["target_median"]))
    if "exposure" in a:                                    # global override for the fixed-exposure finishes
        rigs["M2"]["exposure"] = rigs["M3"]["exposure"] = float(a["exposure"])
    auto = a.get("auto_exposure", "1") == "1"
    poses = a.get("poses", "P1,P2,P3").split(","); finishes = a.get("finishes", "M1,M2,M3").split(",")
    for pid in poses:
        pj = json.loads((C.ROOT / "poses" / POSE_FILES[pid]).read_text())
        P.reset(); P.apply_local(pj["bones"])
        cams[pid] = frame_camera(cam, C.evaluated_positions(body), VIEW_DEG[pid], top_margin, fade_z)
        for key in finishes:
            mat = [m for m in C.bpy.data.materials if m.name.startswith(f"MX_{key}_")][0]
            body.material_slots[0].material = mat
            rg = rigs[key]
            for name, ob in lights.items():
                ob.data.energy = base_w[name] * rg[name]
            sc.view_settings.exposure = rg["exposure"]
            out_dir = Path(a["out_dir"]) if "out_dir" in a else C.OUT / "previews"
            path = out_dir / f"maximus_v2_{pid}_{key}_{a.get('tag', '2k')}.png"
            io.render_still(sc, path)
            rec = {"path": str(path), "exposure": rg["exposure"], "light_watts": {n: round(base_w[n] * rg[n], 2) for n in lights}}
            try:
                st = image_stats(path); rec["stats"] = st
                if auto and "target_median" in rg:
                    e, st, hist = trim_exposure(sc, path, rg["target_median"], rg["tol"], rg["exposure"], st)
                    rec.update({"exposure": round(e, 4), "stats": st,
                                "auto_exposure": {"status": "ok", "target_median": rg["target_median"], "tol": rg["tol"], "history": hist,
                                                  "converged": st["torso_median"] is not None and abs(st["torso_median"] - rg["target_median"]) <= rg["tol"]}})
            except Exception as ex:  # noqa: BLE001  (statistics are a check, not a reason to lose the render)
                rec["auto_exposure"] = {"status": "skipped", "error": repr(ex)}
            rec["sha256"] = contracts.sha256_file(path)
            out[f"{pid}_{key}"] = rec
    report = C.OUT / "reports" / ("s05_previews.json" if a.get("tag", "2k") == "2k" else f"s05_previews_{a['tag']}.json")
    contracts.atomic_write_json(report,
                                {"renders": out, "samples": sc.cycles.samples, "resolution": [sc.render.resolution_x, sc.render.resolution_y],
                                 "view_transform": sc.view_settings.view_transform, "camera": cams, "rig": rigs, "base_light_watts": base_w,
                                 "framing": {"top_margin": top_margin, "fade_z_m": fade_z, "fade_frac": FADE_FRAC},
                                 "device": sc.cycles.device, "gateway_calls": io.CALLS})
    return {"renders": out}, {"n": len(out), "camera": cams,
                              "torso_median": {k: (v.get("stats") or {}).get("torso_median") for k, v in out.items()},
                              "top_margin": {k: (v.get("stats") or {}).get("top_margin") for k, v in out.items()}}


C.run("s05_previews", main)
