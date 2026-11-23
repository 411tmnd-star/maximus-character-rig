"""M7 (v2, 2026-11-19) - Cycles AgX Macro Zipper Render Harness + 2.5D AOV Relight Maps + 4-Panel Lemaire Macro Pack + Instanced GLB.

    Colab:  !python /content/drive/MyDrive/maximus_v2/m7_zipper_macro.py
    Local:  python -I colab_stage/m7_zipper_macro.py --root <maximus_engine> --out <dir> --device cpu-smoke

Outputs (results/M7_zipper_macro/):
  Output A - Macro Hover-Zoom Scrub Sequence (48 frames, WebP/AVIF + 2.5D AOV maps):
             Critically damped spring camera (zeta = 1.0) gliding from jacket hero framing (85 mm, d = 1.8 m) into extreme macro close-up
             of interlocking teeth & slider throat (135 mm, d = 0.14 m) while scrubbing open_factor from 0.00 -> 0.18.
             Locked two-file 2.5D AOV standard via presentation_engine.py:
               m7_macro_surface.webp      (camera-space Normal * 0.5 + 0.5)
               m7_macro_surface_aux.webp  (R = Z-Depth, G = Roughness, B = AO)
               m7_macro_idmask.webp, m7_macro_luma.webp, m7_macro_presentation.json
  Output B - Lemaire-Grade 4-Panel Macro Detail Pack (1024x1024 WebP):
               01_slider_throat.webp
               02_interlocking_teeth.webp
               03_pull_tab_chamfer.webp
               04_top_stop_tape_stitch.webp
  Output C - Instanced Interactive GLB:
               zipper_hero_instanced.glb (EXT_mesh_gpu_instancing, < 400 KB, Zip_Scrub_0_to_1 animation)
  Plus:      M7_macro_contact_sheet.png, M7_zipper_macro.json, MANIFEST_SHA256.txt
"""
import argparse, hashlib, json, math, os, shutil, subprocess, sys, time
from pathlib import Path
from typing import Any, Dict

import numpy as np

IN_BLENDER = "--" in sys.argv
HERE = Path(__file__).resolve().parent
STAGE = "m7_zipper_macro"


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def critically_damped_spring(u, omega=6.5):
    """Exact analytical response of a critically damped 2nd-order spring (zeta = 1.0) normalized to [0, 1] on u in [0, 1]."""
    u = np.clip(u, 0.0, 1.0)
    raw = 1.0 - (1.0 + omega * u) * np.exp(-omega * u)
    end = 1.0 - (1.0 + omega) * math.exp(-omega)
    return raw / max(end, 1e-12)


# ------------------------------------------------------------------------------------------------------------------------------------
# Blender Pass (bpy.data only; renders strictly through io_ops_gateway.render_still -> ZERO bpy.ops here!)
# ------------------------------------------------------------------------------------------------------------------------------------
def blender_main():
    A = dict(x.split("=", 1) for x in sys.argv[sys.argv.index("--") + 1:] if "=" in x)
    root = Path(A["root"]).resolve()
    sys.path.insert(0, str(root / "pipeline")); sys.path.insert(0, str(HERE))
    for p in (HERE / "runtime_v3", HERE.parent / "runtime_v3"):
        if (p / "lighting_director.py").exists():
            sys.path.insert(0, str(p))
    import bl_common as C
    import io_ops_gateway as io
    import m_render as R
    import zipper_engine as ZE
    out = Path(A["out_dir"]); out.mkdir(parents=True, exist_ok=True)

    def main():
        bpy = C.bpy; sc = bpy.context.scene
        body, rig = C.human_and_rig(); rig.data.pose_position = "REST"
        body.material_slots[0].material = bpy.data.materials["MX_M1_ObsidianNoir"]

        # Configure Cycles GPU (OptiX / CUDA) with AgX color management
        sc.render.engine = "CYCLES"
        dev = "CPU"
        if A.get("device", "gpu") == "gpu":
            prefs = bpy.context.preferences.addons["cycles"].preferences
            for dt in ("OPTIX", "CUDA"):
                try:
                    prefs.compute_device_type = dt; prefs.get_devices()
                    if any(d.type == dt for d in prefs.devices):
                        for d in prefs.devices:
                            d.use = d.type == dt
                        dev = dt; break
                except Exception:  # noqa: BLE001
                    pass
            if dev == "CPU" and A.get("require_gpu", "1") == "1":
                raise RuntimeError("no OPTIX/CUDA device: refusing to run M7 macro renders on CPU")
        sc.cycles.device = "GPU" if dev != "CPU" else "CPU"
        sc.cycles.use_denoising = True
        sc.render.film_transparent = False
        sc.render.image_settings.file_format = "PNG"
        sc.render.image_settings.color_mode = "RGBA"
        try:
            sc.view_settings.view_transform = "AgX"
            sc.view_settings.look = "AgX - Medium High Contrast"
        except Exception:  # noqa: BLE001
            pass

        # Build dark matte heavyweight wool/cotton jacket front panels flanking the 420 mm center-front zipper seam
        atlas_path = Path(A["atlas"]) if A.get("atlas") and Path(A["atlas"]).exists() else None
        C0, N0 = ZE.default_jacket_centerline(0.420, n=72, atlas_npz=atlas_path)

        def build_jacket_panels(open_factor):
            s_grid, Q, T, B, N, total_len = ZE._resample_curve(C0, N0, n_samples=72, flip=False)
            s_bot = 0.010; s_top = total_len - 0.008
            s_slider = s_bot + (1.0 - open_factor) * (s_top - s_bot)
            dy, _, hlift = ZE._split_profile(s_grid, s_slider, s_top, 1.0, s_bot=s_bot)
            # Left and right heavyweight wool panels starting from the outer edge of the zipper tape (13.5 mm -> 125 mm)
            Vs, Fs = [], []
            off = 0
            xs_span = np.linspace(0.0135, 0.125, 8)
            thick = 0.0024
            top_pts = [[x, 0.45 * thick] for x in xs_span]
            bot_pts = [[x, -0.55 * thick] for x in xs_span[::-1]]
            base_prof = np.array(top_pts + bot_pts, float)
            for side in (+1, -1):
                prof = base_prof.copy()
                if side < 0:
                    prof[:, 0] = -prof[:, 0]; prof = prof[::-1].copy()
                K = len(prof); M = len(s_grid)
                Vp = np.zeros((M * K, 3), float)
                for i in range(M):
                    base = Q[i] + side * dy[i] * B[i] + hlift[i] * N[i]
                    for k in range(K):
                        x_off = prof[k, 0]
                        chest_drop = 0.12 * (abs(x_off) - 0.0135) ** 2
                        Vp[i * K + k] = base + x_off * B[i] + (prof[k, 1] - chest_drop) * N[i]
                Fp = []
                for i in range(M - 1):
                    r0, r1 = i * K, (i + 1) * K
                    for k in range(K):
                        kn = (k + 1) % K
                        Fp.append([r0 + k + off, r1 + k + off, r1 + kn + off])
                        Fp.append([r0 + k + off, r1 + kn + off, r0 + kn + off])
                Vs.append(Vp); Fs.append(np.array(Fp, np.int64)); off += len(Vp)
            return np.vstack(Vs), np.vstack(Fs)

        # Heavyweight dark wool/cotton material (#141518, Roughness 0.88, Sheen 0.28)
        mat_wool = bpy.data.materials.new("MX_Jacket_HeavyWool")
        if not getattr(mat_wool, "use_nodes", True):
            mat_wool.use_nodes = True
        bsdf_w = next(n for n in mat_wool.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
        bsdf_w.inputs["Base Color"].default_value = (*ZE._hex_to_linear("#16171B"), 1.0)
        bsdf_w.inputs["Roughness"].default_value = 0.88
        if "Sheen Weight" in bsdf_w.inputs:
            bsdf_w.inputs["Sheen Weight"].default_value = 0.28

        me_wool = bpy.data.meshes.new("MX_JacketPanels")
        Vw0, Fw0 = build_jacket_panels(0.0)
        me_wool.from_pydata(Vw0.tolist(), [], Fw0.tolist())
        me_wool.materials.append(mat_wool)
        ob_wool = bpy.data.objects.new("MX_JacketPanels", me_wool)
        sc.collection.objects.link(ob_wool)

        # Studio lighting: key + soft fill + grazing Fresnel metallic rim light (lighting_director principles)
        rim_boost = float(A.get("rim_boost", "1.0"))
        exposure = float(A.get("exposure", "-0.65"))
        cam, lights = R.studio(bpy, sc, samples=int(A.get("samples", "128")), res=(1024, 1024), exposure=exposure,
                               world=(0.012, 0.012, 0.014),
                               weights={"MX_key": 420.0, "MX_fill": 85.0, "MX_rim": 360.0 * rim_boost})
        # Critical for macro close-ups at d = 95-140 mm: lower camera clip_start from Blender's 100 mm default to 5 mm!
        cam.data.clip_start = 0.005
        cam.data.clip_end = 25.0
        # Add dedicated close-range grazing macro specular rim light for beveled Excella teeth + off-axis soft fill (zero mirror blowout)
        grazing = R.light(bpy, sc, "MX_macro_grazing_rim", "AREA", 26.0 * rim_boost, 0.38, (-0.42, -0.20, 1.34), (0.0, -0.14, 1.22))
        key_fill_sub = R.light(bpy, sc, "MX_macro_fill", "AREA", 9.0, 0.65, (0.55, -0.45, 1.38), (0.0, -0.14, 1.22))

        zip_ob = None

        def update_zipper(open_f, tooth_idx=0, head_idx=0, size_mult=1.0):
            nonlocal zip_ob
            if zip_ob is not None:
                old_me = zip_ob.data
                bpy.data.objects.remove(zip_ob, do_unlink=True)
                bpy.data.meshes.remove(old_me, do_unlink=True)
            Vw, _ = build_jacket_panels(open_f)
            me_wool.vertices.foreach_set("co", Vw.astype(np.float32).ravel()); me_wool.update()
            cfg = ZE.ZipperConfig(size_multiplier=size_mult, tooth_index=tooth_idx, head_index=head_idx, open_factor=open_f,
                                  tooth_material="brushed_gunmetal", head_material="brushed_gunmetal", textile_uv_attribute="textile UV")
            zip_ob, zdata = ZE.build_zipper_blender(bpy, C0, cfg, curve_normals=N0, name="MX_Zipper_Hero")
            return zdata

        # ----------------------------------------------------------------------------------------------------------------------------
        # Output A: 48-frame Macro Hover-Zoom Scrub Sequence (85 mm, d=1.80 m -> 135 mm, d=0.14 m; open_factor 0.00 -> 0.18)
        # ----------------------------------------------------------------------------------------------------------------------------
        n_frames = int(A.get("scrub_frames", "48"))
        scrub_res = int(A.get("scrub_px", "768"))
        sc.render.resolution_x = sc.render.resolution_y = scrub_res
        sc.cycles.samples = int(A.get("scrub_samples", "64"))
        scrub_dir = out / "scrub_frames"; scrub_dir.mkdir(exist_ok=True)

        # Target anchor: tracks from mid-jacket at hero distance to the slider throat & Y-split entry at macro distance
        hero_target = np.array([0.0, -0.14, 1.16])
        t0 = time.time()
        scrub_meta = []
        last_zdata = None
        for fi in range(n_frames):
            u = fi / max(n_frames - 1, 1)
            s_damp = float(critically_damped_spring(u, omega=6.5))
            open_f = round(0.18 * s_damp, 5)
            zdata = update_zipper(open_f)
            last_zdata = zdata
            sl_pos = np.array(zdata["slider_state"]["slider_pos_m"], float)
            # Macro target sits 3 mm above slider center right in the Y-bifurcation throat
            macro_target = sl_pos + np.array([0.0, 0.0, 0.002])
            tgt = (1.0 - s_damp) * hero_target + s_damp * macro_target
            dist = (1.0 - s_damp) * 1.80 + s_damp * 0.14
            lens = (1.0 - s_damp) * 85.0 + s_damp * 135.0
            # Slight 6-deg oblique azimuth at macro so beveled tooth facets catch the grazing Fresnel rim light
            az = math.radians(6.0 * s_damp)
            el = math.radians(4.0 * s_damp)
            cam.data.lens = lens
            cam.location = (
                float(tgt[0] + dist * math.sin(az)),
                float(tgt[1] - dist * math.cos(az) * math.cos(el)),
                float(tgt[2] + dist * math.sin(el)),
            )
            R.look_at(cam, tuple(tgt))
            R.look_at(grazing, tuple(sl_pos))
            R.look_at(key_fill_sub, tuple(sl_pos))
            f_png = scrub_dir / f"frame_{fi:03d}.png"
            io.render_still(sc, f_png)
            scrub_meta.append({"frame": fi, "u": round(u, 4), "spring": round(s_damp, 5), "open_factor": open_f,
                               "lens_mm": round(lens, 2), "distance_m": round(dist, 4), "slider_z_m": round(float(sl_pos[2]), 5)})
            if fi % 8 == 0 or fi == n_frames - 1:
                print(f"PROGRESS M7 scrub frame {fi+1}/{n_frames} open={open_f:.3f} d={dist:.3f}m lens={lens:.1f}mm on {dev} {time.time()-t0:.0f}s", flush=True)

        # ----------------------------------------------------------------------------------------------------------------------------
        # Output B: Lemaire-Grade 4-Panel Macro Detail Pack (1024x1024)
        # ----------------------------------------------------------------------------------------------------------------------------
        panel_px = int(A.get("panel_px", "1024"))
        sc.render.resolution_x = sc.render.resolution_y = panel_px
        sc.cycles.samples = int(A.get("panel_samples", "128"))

        zdata_p = update_zipper(0.16, tooth_idx=0, head_idx=0, size_mult=1.0)
        sl_p = np.array(zdata_p["slider_state"]["slider_pos_m"], float)
        v_stops = zdata_p["parts"]["stops"]["V"]
        m_top_l = (v_stops[:, 2] > sl_p[2]) & (v_stops[:, 0] > 0.0)
        top_pos = v_stops[m_top_l].mean(axis=0) if np.any(m_top_l) else (sl_p + np.array([0.022, 0.0, 0.055]))

        panels_spec = [
            ("01_slider_throat",        135.0, 0.125, sl_p + np.array([0.0, 0.0, 0.003]),  8.0,   5.0),
            ("02_interlocking_teeth",   150.0, 0.095, sl_p + np.array([0.0, 0.0, -0.020]), 12.0,  2.0),
            ("03_pull_tab_chamfer",     135.0, 0.115, sl_p + np.array([0.0, -0.003, -0.009]), 22.0, 12.0),
            ("04_top_stop_tape_stitch", 135.0, 0.095, top_pos + np.array([0.001, 0.0, -0.005]), -10.0, 3.0),
        ]
        panel_stats = {}
        for pname, plens, pdist, ptgt, paz_deg, pel_deg in panels_spec:
            cam.data.lens = plens
            az, el = math.radians(paz_deg), math.radians(pel_deg)
            cam.location = (
                float(ptgt[0] + pdist * math.sin(az)),
                float(ptgt[1] - pdist * math.cos(az) * math.cos(el)),
                float(ptgt[2] + pdist * math.sin(el)),
            )
            R.look_at(cam, tuple(ptgt))
            R.look_at(grazing, tuple(ptgt))
            R.look_at(key_fill_sub, tuple(ptgt))
            png_p = out / f"{pname}.png"
            io.render_still(sc, png_p)
            panel_stats[pname] = R.image_stats(bpy, png_p)
            print(f"PROGRESS M7 panel {pname} rendered ({panel_px}x{panel_px}) on {dev}", flush=True)

        # ----------------------------------------------------------------------------------------------------------------------------
        # Bake Macro 2.5D AOV Emission Passes (Camera-Space Normal + Z-Depth + Material ID) on the 01_slider_throat framing
        # ----------------------------------------------------------------------------------------------------------------------------
        ptgt = sl_p + np.array([0.0, 0.0, 0.003])
        pdist = 0.125; az, el = math.radians(8.0), math.radians(5.0)
        cam.data.lens = 135.0
        cam.location = (float(ptgt[0] + pdist * math.sin(az)), float(ptgt[1] - pdist * math.cos(az) * math.cos(el)), float(ptgt[2] + pdist * math.sin(el)))
        R.look_at(cam, tuple(ptgt))

        orig_vt = sc.view_settings.view_transform
        orig_look = sc.view_settings.look
        orig_exp = sc.view_settings.exposure
        orig_samples = sc.cycles.samples
        orig_denoise = sc.cycles.use_denoising
        try:
            sc.view_settings.view_transform = "Raw"
        except Exception:  # noqa: BLE001
            sc.view_settings.view_transform = "Standard"
        sc.view_settings.look = "None"
        sc.view_settings.exposure = 0.0
        sc.cycles.samples = 16
        sc.cycles.use_denoising = False

        # Save original material assignments
        zip_mats_orig = [m for m in zip_ob.data.materials]
        wool_mat_orig = ob_wool.data.materials[0]
        body_mat_orig = body.material_slots[0].material

        # 1) Camera-space normal pass: n_cam * 0.5 + 0.5 (with -Z camera forward flipped to +Z outward)
        mat_nrm = bpy.data.materials.new("MX_AOV_CamNormal")
        nt = mat_nrm.node_tree; N = nt.nodes; L = nt.links; N.clear()
        out_n = N.new("ShaderNodeOutputMaterial")
        em_n = N.new("ShaderNodeEmission")
        geo_n = N.new("ShaderNodeNewGeometry")
        vt_n = N.new("ShaderNodeVectorTransform"); vt_n.vector_type = "NORMAL"; vt_n.convert_from = "WORLD"; vt_n.convert_to = "CAMERA"
        L.new(geo_n.outputs["Normal"], vt_n.inputs["Vector"])
        sep_n = N.new("ShaderNodeSeparateXYZ"); L.new(vt_n.outputs["Vector"], sep_n.inputs["Vector"])
        comb_n = N.new("ShaderNodeCombineXYZ")
        for idx, ch in enumerate(("X", "Y", "Z")):
            ma = N.new("ShaderNodeMath"); ma.operation = "MULTIPLY_ADD"
            ma.inputs[1].default_value = 0.5; ma.inputs[2].default_value = 0.5
            L.new(sep_n.outputs[ch], ma.inputs[0]); L.new(ma.outputs[0], comb_n.inputs[idx])
        L.new(comb_n.outputs["Vector"], em_n.inputs["Color"]); L.new(em_n.outputs["Emission"], out_n.inputs["Surface"])

        for i in range(len(zip_ob.data.materials)):
            zip_ob.data.materials[i] = mat_nrm
        ob_wool.data.materials[0] = mat_nrm
        body.material_slots[0].material = mat_nrm
        io.render_still(sc, out / "_aov_normal.png")

        # 2) Depth & Roughness & ID pass (R = normalized depth around [0.08, 0.22] m, G = physical roughness, B = part ID / 4.0)
        def make_aux_mat(name, rough_val, id_val):
            m = bpy.data.materials.new(name); nt = m.node_tree; N = nt.nodes; L = nt.links; N.clear()
            out_m = N.new("ShaderNodeOutputMaterial"); em = N.new("ShaderNodeEmission")
            cd = N.new("ShaderNodeCameraData")
            # Normalize view Z depth in [0.08, 0.22] m -> [0, 1]
            mr = N.new("ShaderNodeMapRange")
            mr.inputs["From Min"].default_value = 0.08; mr.inputs["From Max"].default_value = 0.22
            mr.inputs["To Min"].default_value = 0.0; mr.inputs["To Max"].default_value = 1.0
            L.new(cd.outputs["View Z Depth"], mr.inputs["Value"])
            comb = N.new("ShaderNodeCombineXYZ")
            L.new(mr.outputs["Result"], comb.inputs["X"])
            comb.inputs["Y"].default_value = float(rough_val)
            comb.inputs["Z"].default_value = float(id_val) / 4.0
            L.new(comb.outputs["Vector"], em.inputs["Color"]); L.new(em.outputs["Emission"], out_m.inputs["Surface"])
            return m

        # slots on zip_ob: 0=tape (id=1, rough=0.84), 1=teeth (id=2, rough=0.20), 2=head/stops (id=3, rough=0.20); wool/body (id=4, rough=0.88)
        zip_ob.data.materials[0] = make_aux_mat("MX_AOV_Tape", 0.84, 1)
        zip_ob.data.materials[1] = make_aux_mat("MX_AOV_Teeth", 0.20, 2)
        zip_ob.data.materials[2] = make_aux_mat("MX_AOV_Head", 0.20, 3)
        m_bg = make_aux_mat("MX_AOV_Wool", 0.88, 4)
        ob_wool.data.materials[0] = m_bg; body.material_slots[0].material = m_bg
        io.render_still(sc, out / "_aov_aux.png")

        # Restore materials & view settings
        for i, m in enumerate(zip_mats_orig):
            zip_ob.data.materials[i] = m
        ob_wool.data.materials[0] = wool_mat_orig
        body.material_slots[0].material = body_mat_orig
        sc.view_settings.view_transform = orig_vt
        sc.view_settings.look = orig_look
        sc.view_settings.exposure = orig_exp
        sc.cycles.samples = orig_samples
        sc.cycles.use_denoising = orig_denoise

        rep = {
            "device": dev,
            "scrub_frames": n_frames,
            "scrub_resolution": [scrub_res, scrub_res],
            "panel_resolution": [panel_px, panel_px],
            "panels": panel_stats,
            "clearance_gate": last_zdata["clearance_gate"],
            "slider_state_final": last_zdata["slider_state"],
            "zipper_stats": last_zdata["stats"],
            "scrub_trajectory": scrub_meta,
            "gateway_calls": len(io.CALLS),
        }
        (out / "_m7_blender_report.json").write_text(json.dumps(rep, indent=1))
        return {"report": str(out / "_m7_blender_report.json")}, {"device": dev, "frames": n_frames, "panels": 4}

    C.run(STAGE, main)


# ------------------------------------------------------------------------------------------------------------------------------------
# System-Python post-processing: pack 2.5D AOVs (presentation_engine.py), encode WebP/AVIF scrub loop, export instanced GLB
# ------------------------------------------------------------------------------------------------------------------------------------
def pack_outputs(work: Path, root: Path) -> Dict[str, Any]:
    from PIL import Image, ImageDraw
    sys.path.insert(0, str(HERE))
    for p in (HERE / "runtime_v3", root / "runtime_v3"):
        if (p / "presentation_engine.py").exists():
            sys.path.insert(0, str(p))
    import presentation_engine as PE
    import zipper_engine as ZE

    res: Dict[str, Any] = {}

    # 1. Convert 4 Lemaire macro panels to lossless/high-craft 1024x1024 WebP
    panel_names = ["01_slider_throat", "02_interlocking_teeth", "03_pull_tab_chamfer", "04_top_stop_tape_stitch"]
    panel_webps = {}
    for pn in panel_names:
        im = Image.open(work / f"{pn}.png").convert("RGB")
        wp = work / f"{pn}.webp"
        im.save(wp, quality=94, method=6)
        panel_webps[f"{pn}.webp"] = wp.stat().st_size
    res["panels_webp_bytes"] = panel_webps

    # 2. Pack locked two-file 2.5D AOV standard (`m7_macro_surface.webp` & `m7_macro_surface_aux.webp`) via presentation_engine.py
    beauty = np.asarray(Image.open(work / "01_slider_throat.png").convert("RGBA"), dtype=np.float64) / 255.0
    nrm_img = np.asarray(Image.open(work / "_aov_normal.png").convert("RGB"), dtype=np.float64) / 255.0
    aux_img = np.asarray(Image.open(work / "_aov_aux.png").convert("RGB"), dtype=np.float64) / 255.0
    nrm_cam = nrm_img * 2.0 - 1.0
    depth_m = 0.08 + aux_img[..., 0] * (0.22 - 0.08)
    rough = aux_img[..., 1]
    ids = np.clip(np.round(aux_img[..., 2] * 4.0), 0, 4).astype(int)
    # Approximate micro-cavity AO from normal Z component
    ao = np.clip(0.35 + 0.65 * np.clip(nrm_cam[..., 2], 0.0, 1.0), 0.0, 1.0)
    id_names = {0: "background", 1: "twill_tape", 2: "excella_teeth", 3: "slider_puller", 4: "jacket_wool"}
    man = PE.pack("m7_macro", work, beauty, ids, nrm_cam, depth_m, rough, ao, id_names, depth_range=(0.08, 0.22))
    val = PE.validate(man)
    assert val["separated"], f"presentation_engine separation check failed: {val}"
    res["aov_presentation"] = {"manifest": "m7_macro_presentation.json", "validation": val}

    # 3. Encode 48-frame Hover-Zoom Scrub Sequence to animated WebP + AVIF
    sfiles = sorted((work / "scrub_frames").glob("frame_*.png"))
    simgs = [Image.open(f).convert("RGB") for f in sfiles]
    simgs[0].save(work / "m7_hover_zoom_hero.webp", quality=92, method=6)
    simgs[-1].save(work / "m7_hover_zoom_macro.webp", quality=94, method=6)
    simgs[0].save(work / "m7_hover_zoom_loop.webp", save_all=True, append_images=simgs[1:], duration=int(1000 / 24), loop=0, quality=84, method=4)
    res["hover_zoom_webp_bytes"] = (work / "m7_hover_zoom_loop.webp").stat().st_size

    for enc in ("libaom-av1", "libsvtav1"):
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-framerate", "24", "-i", str(work / "scrub_frames" / "frame_%03d.png"),
               "-c:v", enc, "-crf", "30", "-b:v", "0", "-pix_fmt", "yuv420p"]
        cmd += (["-cpu-used", "6", "-row-mt", "1"] if enc == "libaom-av1" else ["-preset", "8"]) + [str(work / "m7_hover_zoom_loop.avif")]
        p = subprocess.run(cmd, capture_output=True, text=True)
        if p.returncode == 0 and (work / "m7_hover_zoom_loop.avif").exists():
            res["avif_encoder"] = enc; res["hover_zoom_avif_bytes"] = (work / "m7_hover_zoom_loop.avif").stat().st_size; break

    # 4. Export Output C: Instanced Interactive GLB (`zipper_hero_instanced.glb`, EXT_mesh_gpu_instancing, < 400 KB)
    atlas_p = next((p for p in (work.parent / "M4_canon_atlas" / "maximus_v2_surface_atlas.npz",
                                Path("/content/drive/MyDrive/maximus_v2/results/M4_canon_atlas/maximus_v2_surface_atlas.npz")) if p.exists()), None)
    C0, N0 = ZE.default_jacket_centerline(0.420, n=72, atlas_npz=atlas_p)
    glb_info = ZE.export_instanced_zipper_glb(work / "zipper_hero_instanced.glb", C0, ZE.ZipperConfig(open_factor=0.18), curve_normals=N0)
    res["instanced_glb"] = glb_info

    # 5. Build visual contact sheet (`M7_macro_contact_sheet.png`) for vision inspection
    sheet = Image.new("RGB", (2048, 1536), (18, 19, 22))
    dr = ImageDraw.Draw(sheet)
    for i, pn in enumerate(panel_names):
        im = Image.open(work / f"{pn}.png").convert("RGB").resize((500, 500))
        sheet.paste(im, (i * 512 + 6, 36))
        dr.text((i * 512 + 12, 10), pn, fill=(235, 235, 240))
    # Row 2: 4 scrub keyframes (frames 0, 15, 31, 47)
    idxs = [0, len(simgs) // 3, (2 * len(simgs)) // 3, len(simgs) - 1]
    for i, fi in enumerate(idxs):
        im = simgs[fi].resize((500, 500))
        sheet.paste(im, (i * 512 + 6, 580))
        dr.text((i * 512 + 12, 554), f"scrub frame {fi:02d}/{len(simgs)-1}", fill=(210, 220, 230))
    # Row 3: 2.5D AOV maps (_surface, _surface_aux, _idmask, _luma)
    for i, suf in enumerate(("_surface", "_surface_aux", "_idmask", "_luma")):
        im = Image.open(work / f"m7_macro{suf}.webp").convert("RGB").resize((420, 420))
        sheet.paste(im, (i * 512 + 46, 1105))
        dr.text((i * 512 + 48, 1082), f"m7_macro{suf}.webp", fill=(200, 215, 200))
    sheet.save(work / "M7_macro_contact_sheet.png")
    return res


def launch():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default="/content/maximus_v2/bundle")
    ap.add_argument("--master-psd", default=None)
    ap.add_argument("--atlas", default="/content/drive/MyDrive/maximus_v2/results/M4_canon_atlas/maximus_v2_surface_atlas.npz")
    ap.add_argument("--out", default="/content/drive/MyDrive/maximus_v2/results")
    ap.add_argument("--blender", default=None)
    ap.add_argument("--device", default="gpu", choices=["gpu", "cpu-smoke"])
    ap.add_argument("--scrub-frames", default="48")
    ap.add_argument("--scrub-px", default="768")
    ap.add_argument("--scrub-samples", default="64")
    ap.add_argument("--panel-px", default="1024")
    ap.add_argument("--panel-samples", default="128")
    ap.add_argument("--rim-boost", default="1.0")
    ap.add_argument("--exposure", default="-0.65")
    a = ap.parse_args(); t0 = time.time()
    root = Path(a.root).resolve()
    master = Path(a.master_psd) if a.master_psd else (
        root / "out" / "maximus_v2_master_psd.blend" if (root / "out" / "maximus_v2_master_psd.blend").exists()
        else root / "out" / "maximus_v2_master.blend"
    )
    exe = Path(a.blender) if a.blender else next(Path("/content/blender").glob("blender-*/blender"))
    work = Path("/content/m7_work") if a.out.startswith("/content/drive") else Path(a.out) / "_m7_work"
    work.mkdir(parents=True, exist_ok=True)

    smoke = a.device == "cpu-smoke"
    kv = {
        "root": root,
        "out_dir": work,
        "atlas": a.atlas,
        "device": "cpu" if smoke else "gpu",
        "require_gpu": "0" if smoke else "1",
        "scrub_frames": "6" if smoke else a.scrub_frames,
        "scrub_px": "256" if smoke else a.scrub_px,
        "scrub_samples": "4" if smoke else a.scrub_samples,
        "panel_px": "256" if smoke else a.panel_px,
        "panel_samples": "4" if smoke else a.panel_samples,
        "rim_boost": a.rim_boost,
        "exposure": a.exposure,
    }
    env = dict(os.environ, MAXIMUS_STRICT="1")
    if (root / "corpus").is_dir():
        env["MAXIMUS_CORPUS"] = str(root / "corpus")
    cmd = [str(exe), "--background", str(master), "--factory-startup", "--python-exit-code", "3", "--python", str(Path(__file__).resolve()), "--"] + [f"{k}={v}" for k, v in kv.items()]
    p = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True); buf = []
    for line in p.stdout:
        buf.append(line)
        if line.startswith("PROGRESS"):
            print("   " + line.rstrip(), flush=True)
    p.wait(); (work / "m7_blender.log").write_text("".join(buf), encoding="utf-8")
    envl = [l for l in buf if l.startswith("EC02 ")]
    rep_ec = json.loads(envl[-1][5:]) if envl else {"status": "fail", "errors": ["".join(buf[-80:])]}
    if rep_ec.get("status") != "ok":
        print(json.dumps(rep_ec.get("errors"), indent=1)[:6000]); sys.exit(1)

    brep = json.loads((work / "_m7_blender_report.json").read_text())
    post = pack_outputs(work, root)
    full = {
        "stage": STAGE,
        "blender_pass": brep,
        "outputs": post,
        "elapsed_s": round(time.time() - t0, 1),
        "verdict": {
            "clearance_gate_pass": brep["clearance_gate"]["pass"],
            "all_parts_positive_volume": brep["zipper_stats"]["all_parts_positive_volume"],
            "zero_degenerate_triangles": brep["zipper_stats"]["degenerate_triangles"] == 0,
            "aov_layers_separated": post["aov_presentation"]["validation"]["separated"],
            "instanced_glb_under_400KB": post["instanced_glb"]["under_400KB"],
            "panels_4_of_4": len(post["panels_webp_bytes"]) == 4,
        },
    }
    (work / "M7_zipper_macro.json").write_text(json.dumps(full, indent=1))

    keep = [
        "M7_zipper_macro.json", "M7_macro_contact_sheet.png",
        "01_slider_throat.webp", "02_interlocking_teeth.webp", "03_pull_tab_chamfer.webp", "04_top_stop_tape_stitch.webp",
        "01_slider_throat.png", "02_interlocking_teeth.png", "03_pull_tab_chamfer.png", "04_top_stop_tape_stitch.png",
        "m7_hover_zoom_loop.webp", "m7_hover_zoom_loop.avif", "m7_hover_zoom_hero.webp", "m7_hover_zoom_macro.webp",
        "m7_macro_surface.webp", "m7_macro_surface_aux.webp", "m7_macro_idmask.webp", "m7_macro_luma.webp", "m7_macro_presentation.json",
        "zipper_hero_instanced.glb", "m7_blender.log",
    ]
    keep = [f for f in keep if (work / f).exists()]
    dst = Path(a.out) / "M7_zipper_macro"; (dst / "scrub_frames").mkdir(parents=True, exist_ok=True)
    for f in keep:
        shutil.copy2(work / f, dst / f)
    for sf in sorted((work / "scrub_frames").glob("frame_*.png")):
        shutil.copy2(sf, dst / "scrub_frames" / sf.name)
    (dst / "MANIFEST_SHA256.txt").write_text("".join(f"{sha256(work / f)}  {f}\n" for f in keep), newline="\n")
    if hasattr(os, "sync"):
        os.sync()
    bad = [f for f in keep if sha256(dst / f) != sha256(work / f)]
    print(json.dumps({"to": str(dst), "files": len(keep), "readback_mismatch": bad, "verdict": full["verdict"],
                      "instanced_glb_KB": post["instanced_glb"]["KB"], "elapsed_s": full["elapsed_s"]}, indent=1))
    sys.exit(0 if all(full["verdict"].values()) and not bad else 1)


if __name__ == "__main__":
    blender_main() if IN_BLENDER else launch()
