"""M3 - 2.5D storefront relighting passes + M4 Bronze / M5 Kintsugi hero stills (Colab L4, Cycles GPU).

    !python /content/drive/MyDrive/maximus_v2/m3_aov_surface_bake.py

Blender stage (data API + io_ops_gateway.render_still only), same framing as the 2K/4K previews (s05 frame_camera: 105 mm, 7 % headroom,
82 % fade line, P3 at 35 deg) and the same calibrated light rig (s05_previews.py is exec'd with its C.run() tail stripped):
  * Cycles passes Normal, Z (Depth) and Mist enabled on the view layer, plus a 'rough' AOV: every finish material gets a ShaderNodeOutputAOV
    ('rough') fed from whatever drives its Principled Roughness (the procedural roughness, not a constant), because Cycles has no built-in
    roughness pass. 3 poses x 3 finishes (M1-M3) are rendered to multilayer 32-bit EXR at 1638x2048.
  * M4 Bronze and M5 Kintsugi hero stills (3 poses x 2) at 1638x2048, 256 samples, OIDN, PNG RGBA.
System-Python stage (pip-installs OpenEXR if missing; uses PIL), per pose x finish:
  <pose>_<finish>_surface.webp       lossless RGBA   RGB = camera-space normal (n*0.5+0.5; x right, y up, z toward the viewer), A = coverage
  <pose>_<finish>_surface_aux.webp   lossless RGB    R = roughness (0-1), G = depth code (255 = nearest subject surface, 1 = farthest, 0 = background;
                                                     metres in the JSON), B = Cycles mist pass as rendered
  <pose>_<finish>_surface.json       decode constants: depth near/far (m), camera rotation / intrinsics, mist settings, EXR sha256, and the
                                     measured correlation between mist and depth (so the mist direction is recorded, not assumed)
Why two files instead of one webp: depth must not live in the alpha channel of an image that a browser will decode (premultiplied alpha
quantises RGB where alpha is small). Normals keep a true alpha (coverage); the other three data channels sit in an opaque companion.
M4 / M5 ARE NOT IN THE PROJECT SPEC (materials.py only builds M1-M3; no document names their colours). They are authored here from scratch,
procedurally on the same 'rest_pos' texture space as M1-M3, and are a FIRST DRAFT for review, not an approved look:
  M4 Bronze   bronze #8C5A2E metal (rough 0.30) with a noise-masked verdigris patina #4E9482 (dielectric, rough 0.70)
  M5 Kintsugi ceramic #E9E3D6 (rough 0.42) cut by warped Voronoi-edge veins filled with gold #D4AF37 (metal, rough 0.25), veins slightly raised
Metals reflect the world, so the hero world is a dim grey studio (--hero-world, default 0.30) instead of the near-black used for chalk.
Outputs -> <out>/M3_aov_surface_bake/ (webp + json + hero PNGs + M3_surface_preview.png + M3_aov_surface_bake.json). EXRs stay on the local disk
unless --keep-exr.
"""
import argparse, hashlib, importlib.util, json, math, os, shutil, subprocess, sys, time
from pathlib import Path

import numpy as np

try:
    import bpy  # noqa: F401
    IN_BLENDER = True
except ImportError:
    IN_BLENDER = False

STAGE = "m3_aov_surface_bake"
HERE = Path(__file__).resolve().parent
POSES = ("P1", "P2", "P3"); FINISHES = ("M1", "M2", "M3")
HEX = {"bronze": "#8C5A2E", "verdigris": "#4E9482", "ceramic": "#E9E3D6", "gold": "#D4AF37"}


def kv_args():
    a = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    return dict(x.split("=", 1) for x in a if "=" in x)


# ----------------------------------------------------------------------------------------------------------------------
# Blender stage
# ----------------------------------------------------------------------------------------------------------------------
def load_s05(C):
    p = C.ROOT / "pipeline" / "s05_previews.py"; src = p.read_text(encoding="utf-8"); tail = 'C.run("s05_previews", main)'
    if tail not in src:
        raise RuntimeError("s05_previews.py changed shape: cannot strip its C.run() tail")
    ns = {"__name__": "s05_helpers", "__file__": str(p)}
    exec(compile(src.replace(tail, "pass"), str(p), "exec"), ns)
    return ns


def blender_main():
    A = kv_args(); B = Path(A["bundle"]); sys.path.insert(0, str(B / "pipeline"))
    import bl_common as C
    import io_ops_gateway as io
    import posing
    from corpus_bridge import contracts
    bpy = C.bpy
    s05 = load_s05(C)

    def say(msg):
        print("PROGRESS " + msg, flush=True)

    def node(nt, typ, x, y, **props):
        nd = nt.nodes.new(typ); nd.location = (x, y)
        for k, v in props.items():
            setattr(nd, k, v)
        return nd

    def lin(hexstr):
        return (*C.hex_linear(hexstr), 1.0)

    def fresh_material(name):
        m = bpy.data.materials.get(name) or bpy.data.materials.new(name)
        m.use_nodes = True; m.use_fake_user = True; nt = m.node_tree; nt.nodes.clear()
        out = node(nt, "ShaderNodeOutputMaterial", 1400, 0); bsdf = node(nt, "ShaderNodeBsdfPrincipled", 1000, 0)
        nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
        rest = node(nt, "ShaderNodeAttribute", -1200, 0, attribute_name="rest_pos", attribute_type="GEOMETRY")
        return m, nt, bsdf, rest

    def mrange(nt, x, y, fmin, fmax, tmin, tmax):
        r = node(nt, "ShaderNodeMapRange", x, y, clamp=True)
        for k, v in (("From Min", fmin), ("From Max", fmax), ("To Min", tmin), ("To Max", tmax)):
            r.inputs[k].default_value = v
        return r

    def build_bronze():
        m, nt, bsdf, rest = fresh_material("MX_M4_Bronze")
        big = node(nt, "ShaderNodeTexNoise", -900, 300); big.inputs["Scale"].default_value = 22.0; big.inputs["Detail"].default_value = 7.0
        nt.links.new(rest.outputs["Vector"], big.inputs["Vector"])
        mask = mrange(nt, -600, 300, 0.42, 0.60, 0.0, 1.0); nt.links.new(big.outputs["Fac"], mask.inputs["Value"])
        col = node(nt, "ShaderNodeMix", 200, 300, data_type="RGBA")
        col.inputs["A"].default_value = lin(HEX["bronze"]); col.inputs["B"].default_value = lin(HEX["verdigris"])
        nt.links.new(mask.outputs["Result"], col.inputs["Factor"]); nt.links.new(col.outputs["Result"], bsdf.inputs["Base Color"])
        met = mrange(nt, 200, 0, 0.0, 1.0, 1.0, 0.12); nt.links.new(mask.outputs["Result"], met.inputs["Value"])
        nt.links.new(met.outputs["Result"], bsdf.inputs["Metallic"])
        fine = node(nt, "ShaderNodeTexNoise", -900, -100); fine.inputs["Scale"].default_value = 120.0; fine.inputs["Detail"].default_value = 4.0
        nt.links.new(rest.outputs["Vector"], fine.inputs["Vector"])
        rgh = node(nt, "ShaderNodeMath", 200, -150, operation="MULTIPLY"); rgh.inputs[1].default_value = 0.12
        nt.links.new(fine.outputs["Fac"], rgh.inputs[0])
        base_r = mrange(nt, 0, -150, 0.0, 1.0, 0.30, 0.70); nt.links.new(mask.outputs["Result"], base_r.inputs["Value"])
        radd = node(nt, "ShaderNodeMath", 500, -150, operation="ADD", use_clamp=True)
        nt.links.new(base_r.outputs["Result"], radd.inputs[0]); nt.links.new(rgh.outputs[0], radd.inputs[1]); nt.links.new(radd.outputs[0], bsdf.inputs["Roughness"])
        grit = node(nt, "ShaderNodeTexNoise", -900, -450); grit.inputs["Scale"].default_value = 900.0; grit.inputs["Detail"].default_value = 2.0
        nt.links.new(rest.outputs["Vector"], grit.inputs["Vector"])
        bump = node(nt, "ShaderNodeBump", 700, -400); bump.inputs["Strength"].default_value = 0.25; bump.inputs["Distance"].default_value = 0.0004
        nt.links.new(grit.outputs["Fac"], bump.inputs["Height"]); nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
        return m

    def build_kintsugi():
        m, nt, bsdf, rest = fresh_material("MX_M5_Kintsugi")
        warp = node(nt, "ShaderNodeTexNoise", -1000, 300); warp.inputs["Scale"].default_value = 6.0; warp.inputs["Detail"].default_value = 2.0
        nt.links.new(rest.outputs["Vector"], warp.inputs["Vector"])
        wsc = node(nt, "ShaderNodeVectorMath", -750, 300, operation="MULTIPLY"); wsc.inputs[1].default_value = (0.03, 0.03, 0.03)
        nt.links.new(warp.outputs["Color"], wsc.inputs[0])
        wadd = node(nt, "ShaderNodeVectorMath", -500, 300, operation="ADD")
        nt.links.new(rest.outputs["Vector"], wadd.inputs[0]); nt.links.new(wsc.outputs["Vector"], wadd.inputs[1])
        vor = node(nt, "ShaderNodeTexVoronoi", -250, 300, feature="DISTANCE_TO_EDGE"); vor.inputs["Scale"].default_value = 7.0
        nt.links.new(wadd.outputs["Vector"], vor.inputs["Vector"])
        vein = mrange(nt, 0, 300, 0.0, 0.03, 1.0, 0.0); nt.links.new(vor.outputs["Distance"], vein.inputs["Value"])
        col = node(nt, "ShaderNodeMix", 300, 300, data_type="RGBA")
        col.inputs["A"].default_value = lin(HEX["ceramic"]); col.inputs["B"].default_value = lin(HEX["gold"])
        nt.links.new(vein.outputs["Result"], col.inputs["Factor"]); nt.links.new(col.outputs["Result"], bsdf.inputs["Base Color"])
        nt.links.new(vein.outputs["Result"], bsdf.inputs["Metallic"])
        rgh = mrange(nt, 300, 0, 0.0, 1.0, 0.42, 0.25); nt.links.new(vein.outputs["Result"], rgh.inputs["Value"])
        nt.links.new(rgh.outputs["Result"], bsdf.inputs["Roughness"])
        bump = node(nt, "ShaderNodeBump", 700, -300); bump.inputs["Strength"].default_value = 0.5; bump.inputs["Distance"].default_value = 0.0005
        nt.links.new(vein.outputs["Result"], bump.inputs["Height"]); nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
        return m

    def emission_material(name, build):
        """Material that shows one data channel as raw emission (no lighting involved): build(nt) returns the output socket."""
        m = bpy.data.materials.get(name) or bpy.data.materials.new(name); m.use_nodes = True; m.use_fake_user = True
        nt = m.node_tree; nt.nodes.clear()
        out = node(nt, "ShaderNodeOutputMaterial", 700, 0); em = node(nt, "ShaderNodeEmission", 400, 0)
        em.inputs["Strength"].default_value = 1.0; nt.links.new(em.outputs["Emission"], out.inputs["Surface"])
        nt.links.new(build(nt), em.inputs["Color"]); return m

    def build_normal(nt):                                    # camera-space shading normal, stored as n * 0.5 + 0.5
        geo = node(nt, "ShaderNodeNewGeometry", -600, 0)
        vt = node(nt, "ShaderNodeVectorTransform", -350, 0, vector_type="NORMAL", convert_from="WORLD", convert_to="CAMERA")
        nt.links.new(geo.outputs["Normal"], vt.inputs["Vector"])
        # measured on Blender 5.2.3 / Cycles: the shader camera space has +Z pointing AWAY from the viewer (front-facing normals read
        # z = -0.99), x right, y up. Flip z so the stored normal has z toward the viewer, as documented.
        fz = node(nt, "ShaderNodeVectorMath", -225, 0, operation="MULTIPLY"); fz.inputs[1].default_value = (1.0, 1.0, -1.0)
        nt.links.new(vt.outputs["Vector"], fz.inputs[0])
        ma = node(nt, "ShaderNodeVectorMath", -100, 0, operation="MULTIPLY_ADD")
        ma.inputs[1].default_value = (0.5, 0.5, 0.5); ma.inputs[2].default_value = (0.5, 0.5, 0.5)
        nt.links.new(fz.outputs["Vector"], ma.inputs[0]); return ma.outputs["Vector"]

    def build_depth(nt):                                     # camera-space distance along the view axis, metres
        geo = node(nt, "ShaderNodeNewGeometry", -600, 0)
        vt = node(nt, "ShaderNodeVectorTransform", -350, 0, vector_type="POINT", convert_from="WORLD", convert_to="CAMERA")
        nt.links.new(geo.outputs["Position"], vt.inputs["Vector"])
        sep = node(nt, "ShaderNodeSeparateXYZ", -100, 0); nt.links.new(vt.outputs["Vector"], sep.inputs["Vector"])
        return sep.outputs["Z"]                              # Cycles shader camera space: +Z is forward, so Z is the positive distance

    def rough_material(src):
        """Clone of a finish material whose surface is emission = whatever drives its Principled Roughness (the procedural value)."""
        m = src.copy(); m.name = src.name + "_roughpass"; nt = m.node_tree
        bsdf = next(n for n in nt.nodes if n.type == "BSDF_PRINCIPLED"); out = next(n for n in nt.nodes if n.type == "OUTPUT_MATERIAL")
        sock = bsdf.inputs["Roughness"]; em = node(nt, "ShaderNodeEmission", 1000, 250); em.inputs["Strength"].default_value = 1.0
        if sock.is_linked:
            nt.links.new(sock.links[0].from_socket, em.inputs["Color"])
        else:
            c = float(sock.default_value); em.inputs["Color"].default_value = (c, c, c, 1.0)
        nt.links.new(em.outputs["Emission"], out.inputs["Surface"]); return m

    def set_world(sc, v):
        try:
            sc.world.color = (v, v, v)
            if sc.world.use_nodes:
                bg = next((n for n in sc.world.node_tree.nodes if n.type == "BACKGROUND"), None)
                if bg:
                    bg.inputs["Color"].default_value = (v, v, v, 1.0)
        except Exception:  # noqa: BLE001  (world API drift: keep the s05 world, report it)
            say("world colour could not be set; hero stills use the s05 world")

    def main():
        out = Path(A["out_dir"]); exr_dir = out / "exr"; hero_dir = out / "hero"; exr_dir.mkdir(parents=True, exist_ok=True); hero_dir.mkdir(exist_ok=True)
        device = A.get("device", "GPU"); rx, ry = int(A.get("rx", 1638)), int(A.get("ry", 2048))
        aov_samples, hero_samples = int(A.get("aov_samples", 48)), int(A.get("hero_samples", 256))
        sc = bpy.context.scene; body, rig = C.human_and_rig(); rig.data.pose_position = "POSE"
        cam, lights, base_w = s05["setup"](sc, {"samples": aov_samples, "rx": rx, "ry": ry, "device": device})
        try:
            sc.cycles.denoiser = "OPENIMAGEDENOISE"
        except (TypeError, AttributeError):
            pass
        sc.cycles.use_denoising = True
        try:
            sc.cycles.denoising_use_gpu = True
        except AttributeError:
            pass
        mats = {k: [m for m in bpy.data.materials if m.name.startswith(f"MX_{k}_") and not m.name.endswith("_roughpass")][0] for k in FINISHES}
        mat_normal = emission_material("MX_PASS_normal", build_normal); mat_depth = emission_material("MX_PASS_depth", build_depth)
        mat_rough = {k: rough_material(mats[k]) for k in FINISHES}
        m4, m5 = build_bronze(), build_kintsugi()
        P = posing.Poser(bpy, rig); ev = lambda: C.evaluated_positions(body)   # noqa: E731
        rg = s05["RIG"]["M2"]
        for n, ob in lights.items():
            ob.data.energy = base_w[n] * rg[n]
        sc.view_settings.exposure = float(A.get("hero_exposure", s05["BASE_EXPOSURE"]))
        top_margin, fade_z = s05["TOP_MARGIN"], s05["FADE_Z"]
        report = {"surface": {}, "hero": {}, "device": sc.cycles.device, "resolution": [rx, ry], "samples": {"aov": aov_samples, "hero": hero_samples}}
        poses_json = {pid: json.loads((C.ROOT / "poses" / s05["POSE_FILES"][pid]).read_text())["bones"] for pid in POSES}
        cams = {}
        vt0, exp0 = sc.view_settings.view_transform, sc.view_settings.exposure
        for pid in POSES:                                         # ---- data passes as emission renders (float EXR, no denoise, no tone curve)
            P.reset(); P.apply_local(poses_json[pid])
            cams[pid] = s05["frame_camera"](cam, C.evaluated_positions(body), s05["VIEW_DEG"][pid], top_margin, fade_z)
            bpy.context.view_layer.update(); D = cams[pid]["distance_m"]
            ims = sc.render.image_settings
            if hasattr(ims, "media_type"):
                ims.media_type = "IMAGE"
            ims.file_format = "OPEN_EXR"; ims.color_mode = "RGBA"; ims.color_depth = "32"; ims.exr_codec = "ZIP"
            sc.cycles.use_denoising = False; sc.cycles.samples = aov_samples; sc.render.resolution_x, sc.render.resolution_y = rx, ry
            try:
                sc.view_settings.view_transform = "Standard"
            except TypeError:
                pass
            sc.view_settings.exposure = 0.0
            plan = [("normal", mat_normal), ("depth", mat_depth)] + [(f"rough_{k}", mat_rough[k]) for k in FINISHES]
            files = {}
            for tag, mat in plan:
                body.material_slots[0].material = mat
                path = exr_dir / f"{pid}_{tag}.exr"; io.render_still(sc, path); files[tag] = {"path": str(path), "sha256": contracts.sha256_file(path)}
            rot = np.array(cam.matrix_world.to_3x3(), float)
            for key in FINISHES:
                meta = {"pose": pid, "finish": key, "exr": {"normal": files["normal"], "depth": files["depth"], "rough": files[f"rough_{key}"]},
                        "resolution": [rx, ry], "cam_rot_world": rot.tolist(), "cam_location_m": [round(v, 5) for v in cam.location],
                        "lens_mm": s05["LENS_MM"], "sensor_mm": s05["SENSOR_MM"], "sensor_fit": "VERTICAL", "camera": cams[pid],
                        "mist": {"start_m": round(D - 0.7, 4), "depth_m": 1.4, "falloff": "LINEAR", "definition": "1 - clamp((z - start) / depth), computed from the depth render"},
                        "method": "emission-shader data renders (Blender 5.2 did not write Normal/Z/Mist/AOV passes into the EXR)"}
                contracts.atomic_write_json(exr_dir / f"{pid}_{key}_meta.json", meta); report["surface"][f"{pid}_{key}"] = meta
            say(f"surface data passes {pid} done (5 renders)")
        ims = sc.render.image_settings                              # ---- hero stills: PNG, grey studio world, the s05 tone curve and exposure
        if hasattr(ims, "media_type"):
            ims.media_type = "IMAGE"
        ims.file_format = "PNG"; ims.color_mode = "RGBA"; ims.color_depth = "8"
        sc.view_settings.view_transform = vt0; sc.view_settings.exposure = float(A.get("hero_exposure", exp0)); sc.cycles.use_denoising = True
        set_world(sc, float(A.get("hero_world", 0.30))); sc.cycles.samples = hero_samples
        for pid in POSES:
            P.reset(); P.apply_local(poses_json[pid])
            s05["frame_camera"](cam, C.evaluated_positions(body), s05["VIEW_DEG"][pid], top_margin, fade_z)
            for tag, mat in (("M4_bronze", m4), ("M5_kintsugi", m5)):
                body.material_slots[0].material = mat
                path = hero_dir / f"{pid}_{tag}_hero.png"; io.render_still(sc, path)
                report["hero"][f"{pid}_{tag}"] = {"path": str(path), "sha256": contracts.sha256_file(path), "material": mat.name}
                say(f"hero {pid} {tag} done")
        report["authored_materials"] = {"M4": {"name": m4.name, "hex": [HEX["bronze"], HEX["verdigris"]]}, "M5": {"name": m5.name, "hex": [HEX["ceramic"], HEX["gold"]]},
                                        "note": "not in the project spec; first draft for review"}
        report["gateway_calls"] = len(io.CALLS)
        contracts.atomic_write_json(out / "m3_blender_report.json", report)
        return {"report": str(out / "m3_blender_report.json")}, {"surface_exr": len(report["surface"]), "hero": len(report["hero"])}

    C.run(STAGE, main)


# ----------------------------------------------------------------------------------------------------------------------
# system-Python stage: EXR -> webp
# ----------------------------------------------------------------------------------------------------------------------
def read_exr(path):
    """{channel name: float32 (h, w)} for every channel in an EXR (legacy OpenEXR/Imath API, installed on demand)."""
    try:
        import OpenEXR, Imath
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "OpenEXR", "Imath"], check=True)
        import OpenEXR, Imath
    f = OpenEXR.InputFile(str(path)); hdr = f.header(); dw = hdr["dataWindow"]; w, h = dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1
    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    return {name: np.frombuffer(f.channel(name, pt), np.float32).reshape(h, w) for name in hdr["channels"]}


def group_passes(ch):
    """{pass name (lower case, view-layer prefix dropped): {channel suffix: array}}"""
    g = {}
    for name, arr in ch.items():
        prefix, _, suffix = name.rpartition(".")
        g.setdefault(prefix.split(".")[-1].lower(), {})[suffix.upper()] = arr
    return g


def read_rgba(path):
    """Straight (un-premultiplied) float RGB and alpha of a single-layer EXR written with a transparent film."""
    ch = {name.rpartition(".")[2].upper(): arr for name, arr in read_exr(path).items()}
    need = [k for k in "RGBA" if k not in ch]
    if need:
        raise RuntimeError(f"{Path(path).name}: channels {need} missing; found {sorted(ch)}")
    a = ch["A"].astype(np.float64); safe = np.where(a > 1e-4, a, 1.0)
    return np.stack([ch[k].astype(np.float64) / safe for k in "RGB"], -1), np.clip(a, 0, 1)


def pack_surface(meta, out_dir, Image):
    nrm, alpha = read_rgba(meta["exr"]["normal"]["path"]); dep, _ = read_rgba(meta["exr"]["depth"]["path"]); rgh, _ = read_rgba(meta["exr"]["rough"]["path"])
    cover = alpha > 0.5
    n_c = nrm * 2.0 - 1.0; ln = np.linalg.norm(n_c, axis=-1, keepdims=True)           # already camera space (x right, y up, z toward viewer)
    n_c = np.where(ln > 1e-8, n_c / np.maximum(ln, 1e-8), np.array([0.0, 0.0, 1.0])); n_c[~cover] = (0.0, 0.0, 1.0)
    rgba = np.dstack([np.clip(n_c * 0.5 + 0.5, 0, 1) * 255.0, alpha * 255.0]).round().astype(np.uint8)
    z = dep[..., 0]; valid = cover & np.isfinite(z) & (z > 0)
    if int(valid.sum()) < 0.5 * int(cover.sum()) or not cover.any():
        raise RuntimeError(f"{meta['pose']}_{meta['finish']}: only {int(valid.sum())} of {int(cover.sum())} covered pixels have a positive depth; "
                           f"depth range {float(np.nanmin(z[cover])) if cover.any() else None}..{float(np.nanmax(z[cover])) if cover.any() else None}")
    near, far = (float(np.percentile(z[valid], 0.1)), float(np.percentile(z[valid], 99.9))) if valid.any() else (0.0, 1.0)
    depth_code = np.where(valid, 1 + 254 * (1 - np.clip((z - near) / max(far - near, 1e-6), 0, 1)), 0)
    rough = np.clip(rgh[..., 0], 0, 1); m = meta["mist"]; mist = np.where(valid, 1.0 - np.clip((z - m["start_m"]) / m["depth_m"], 0, 1), 0.0)
    aux = np.dstack([rough * 255.0, depth_code, mist * 255.0]).round().astype(np.uint8)
    stem = f"{meta['pose']}_{meta['finish']}_surface"
    Image.fromarray(rgba).save(out_dir / f"{stem}.webp", "WEBP", lossless=True, quality=100, method=6)
    Image.fromarray(aux).save(out_dir / f"{stem}_aux.webp", "WEBP", lossless=True, quality=100, method=6)
    side = {"pose": meta["pose"], "finish": meta["finish"], "resolution": meta["resolution"], "method": meta["method"],
            "surface_webp": {"RGB": "camera-space normal, n = rgb/255*2-1 (x right, y up, z toward viewer)", "A": "coverage"},
            "aux_webp": {"R": "roughness = r/255", "G": "depth code: 0 background, else metres = far - (g-1)/254 * (far - near)", "B": "mist = b/255 (1 near .. 0 far)"},
            "depth_near_m": round(near, 5), "depth_far_m": round(far, 5), "depth_definition": "camera-space distance along the view axis",
            "mist": m, "normal_note": "geometric smooth normal; the micro bump of the finish is not in this normal map",
            "camera": {"rot_world_3x3": meta["cam_rot_world"], "location_m": meta["cam_location_m"], "lens_mm": meta["lens_mm"],
                       "sensor_mm": meta["sensor_mm"], "sensor_fit": meta["sensor_fit"], "framing": meta["camera"]},
            "source_exr_sha256": {k: v["sha256"] for k, v in meta["exr"].items()}, "coverage_fraction": round(float(cover.mean()), 5),
            "roughness_range": [round(float(rough[valid].min()), 4), round(float(rough[valid].max()), 4)] if valid.any() else None}
    (out_dir / f"{stem}.json").write_text(json.dumps(side, indent=1))
    return side


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def verify_bundle(B):
    man = json.loads((B / "MANIFEST.json").read_text())
    bad = [r for r, h in man["files"].items() if not (B / r).exists() or sha256(B / r) != h]
    if bad:
        sys.exit(f"bundle hash mismatch: {bad}")
    if not man.get("gates_all_pass", False):
        sys.exit(f"bundle was packed with failing gates {man.get('gates_failed')}")
    return man


def find_blender(B, explicit):
    if explicit:
        return Path(explicit)
    exe = next(Path("/content/blender").glob("blender-*/blender"), None)
    if exe:
        return exe
    spec = importlib.util.spec_from_file_location("run_batch_4k", B.parent / "run_batch_4k.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod.blender()


def run_blender(exe, B, kv, log):
    env = dict(os.environ, MAXIMUS_CORPUS=str(B / "corpus"), MAXIMUS_STRICT="1")
    cmd = [str(exe), "--background", str(B / "out" / "maximus_v2_master.blend"), "--factory-startup", "--python-exit-code", "3",
           "--python", str(Path(__file__).resolve()), "--"] + [f"{k}={v}" for k, v in kv.items()]
    print(" ".join(cmd), flush=True)
    p = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True); buf = []
    for line in p.stdout:
        buf.append(line)
        if line.startswith("PROGRESS "):
            print(line.rstrip(), flush=True)
    p.wait(); log.parent.mkdir(parents=True, exist_ok=True); log.write_text("".join(buf))
    env_line = [l for l in buf if l.startswith("EC02 ")]
    return json.loads(env_line[-1][5:]) if env_line else {"status": "fail", "errors": ["no EC-02 envelope", "".join(buf[-40:])]}


def preview_montage(work, sides, out_path, Image, th=(328, 410)):
    cols = ("normal", "rough M1", "rough M3", "depth M2")
    sheet = Image.new("RGB", (th[0] * len(cols), th[1] * len(POSES)), (60, 60, 64))
    for i, pid in enumerate(POSES):
        n = Image.open(work / "surface" / f"{pid}_M2_surface.webp").convert("RGBA"); a = Image.open(work / "surface" / f"{pid}_M2_surface_aux.webp")
        r1 = Image.open(work / "surface" / f"{pid}_M1_surface_aux.webp"); r3 = Image.open(work / "surface" / f"{pid}_M3_surface_aux.webp")
        tiles = [n.resize(th), r1.split()[0].resize(th).convert("RGB"), r3.split()[0].resize(th).convert("RGB"), a.split()[1].resize(th).convert("RGB")]
        for j, t in enumerate(tiles):
            if t.mode == "RGBA":
                bg = Image.new("RGB", th, (60, 60, 64)); bg.paste(t, (0, 0), t); t = bg
            sheet.paste(t, (j * th[0], i * th[1]))
    sheet.save(out_path); return out_path


def launch():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--bundle", default="/content/maximus_v2/bundle"); ap.add_argument("--out", default="/content/drive/MyDrive/maximus_v2/results")
    ap.add_argument("--blender"); ap.add_argument("--device", default="GPU")
    ap.add_argument("--aov-samples", default="48"); ap.add_argument("--hero-samples", default="256"); ap.add_argument("--hero-world", default="0.30")
    ap.add_argument("--keep-exr", action="store_true")
    a = ap.parse_args(); B = Path(a.bundle).resolve(); t0 = time.time()
    from PIL import Image
    man = verify_bundle(B); exe = find_blender(B, a.blender)
    work = B / "out" / "M3_aov_and_patinas"; work.mkdir(parents=True, exist_ok=True)
    env = run_blender(exe, B, {"bundle": B, "out_dir": work, "device": a.device, "aov_samples": a.aov_samples, "hero_samples": a.hero_samples,
                               "hero_world": a.hero_world}, work / "m3_blender.log")
    if env["status"] != "ok":
        print(json.dumps(env.get("errors"), indent=1)[:4000]); sys.exit(1)
    rep = json.loads((work / "m3_blender_report.json").read_text()); (work / "surface").mkdir(exist_ok=True); sides = {}
    for k, meta in rep["surface"].items():
        sides[k] = pack_surface(meta, work / "surface", Image); print(f"packed {k}: rough {sides[k]['roughness_range']} depth {sides[k]['depth_near_m']}-{sides[k]['depth_far_m']} m", flush=True)
    preview_montage(work, sides, work / "M3_surface_preview.png", Image)
    summary = {"stage": STAGE, "master_sha256": man.get("master_sha256"), "device": rep["device"], "samples": rep["samples"], "resolution": rep["resolution"],
               "surface": sides, "hero": rep["hero"], "authored_materials": rep["authored_materials"], "elapsed_s": round(time.time() - t0, 1)}
    (work / "M3_aov_surface_bake.json").write_text(json.dumps(summary, indent=1))
    keep = (sorted((work / "surface").glob("*")) + sorted((work / "hero").glob("*.png"))
            + [work / "M3_surface_preview.png", work / "M3_aov_surface_bake.json", work / "m3_blender.log"])
    if a.keep_exr:
        keep += sorted((work / "exr").glob("*"))
    dst = Path(a.out) / "M3_aov_surface_bake"; dst.mkdir(parents=True, exist_ok=True)
    for f in keep:
        rel = f.relative_to(work); (dst / rel).parent.mkdir(parents=True, exist_ok=True); shutil.copy2(f, dst / rel)
    if hasattr(os, "sync"):
        os.sync()
    bad = [str(f.relative_to(work)) for f in keep if not (dst / f.relative_to(work)).exists() or sha256(dst / f.relative_to(work)) != sha256(f)]
    print(json.dumps({"to": str(dst), "files": len(keep), "surface_pairs": len(sides), "hero": len(rep["hero"]), "readback_mismatch": bad,
                      "elapsed_s": summary["elapsed_s"]}, indent=1))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    blender_main() if IN_BLENDER else launch()
