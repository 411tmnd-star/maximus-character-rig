"""M9 Dressed Reels & Studio Stills (Colab L4 Cycles OPTIX + OIDN + AgX).

Renders the dressed Maximus v2 + BS-RINGER-001 hero shirt across:
  - SWAG_01 3s editorial pose strike (72 frames @ 24 fps, 1080x1350)
  - M5 Runway Walk (250 frames @ 24 fps, 1080x1350)
  - M1 / M2 / M3 RGBA storefront stills (1200x1500, front / three_quarter / back)

All motion uses zero-refit SO(3) triangle-frame transport T_atlas + 3 mm clearance barrier
with CUDA float64 GWN == 0 verification, pure bpy.data / NumPy APIs (zero bpy.ops outside io_ops_gateway).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

IN_BLENDER = "--" in sys.argv
HERE = Path(__file__).resolve().parent
ENGINE_ROOT = HERE.parent
STAGE = "m9_dressed_reels"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def blender_main():
    A = dict(x.split("=", 1) for x in sys.argv[sys.argv.index("--") + 1:] if "=" in x)
    root = Path(A["root"]).resolve()
    for p in (root, root / "pipeline", HERE, root / "runtime_v3"):
        if str(p) not in sys.path:
            sys.path.append(str(p))

    import bl_common as C
    import io_ops_gateway as io
    import m8_motion_reels as M8
    import m_render as R
    import posing
    import pose_space_engine as PSE
    import garment_transport as GT
    from maximus_core import geometry
    from mathutils import Quaternion

    bpy = C.bpy
    sc = bpy.context.scene
    mode = A.get("mode", "pose")
    out_dir = Path(A["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    res_w = int(A.get("w", "1080"))
    res_h = int(A.get("h", "1350"))
    samples = int(A.get("samples", "48"))

    body, rig = C.human_and_rig()
    me = body.data
    me.calc_loop_triangles()
    arm = [m for m in body.modifiers if m.type == "ARMATURE"]
    cs = [m for m in body.modifiers if m.type == "CORRECTIVE_SMOOTH"]
    assert len(me.vertices) == 13380 and len(me.loop_triangles) == 26756
    assert len(rig.data.bones) == 152
    assert len(arm) == 1 and not arm[0].use_deform_preserve_volume
    assert len(cs) == 1 and abs(cs[0].factor - 0.5) < 1e-9 and cs[0].iterations == 5

    # Load cached dressed shell binding
    shell, gbind = GT.load_shell_cache(Path(A["shell_npz"]))
    art_dir = Path(A["art_dir"]) if "art_dir" in A else None

    # Create Cycles-ready shirt mesh object (updated per frame via T_atlas, pure bpy.data)
    me_s = bpy.data.meshes.new("BS_RINGER_001_M9")
    me_s.from_pydata(gbind["V_fit"].tolist(), [], shell["F"].astype(np.int32).tolist())
    for uv_name in ("UVMap_Art_Front", "UVMap_Fabric"):
        if uv_name in shell["uvs"]:
            uvl = me_s.uv_layers.new(name=uv_name)
            uvl.data.foreach_set("uv", shell["uvs"][uv_name].astype(np.float32).ravel())

    mat_main = bpy.data.materials.new("BS_Ringer_Main_Cycles")
    mat_main.use_nodes = True
    nt_m = mat_main.node_tree
    bsdf_m = next(n for n in nt_m.nodes if n.type == "BSDF_PRINCIPLED")
    bsdf_m.inputs["Roughness"].default_value = 0.88
    bsdf_m.inputs["Specular IOR Level"].default_value = 0.12
    art_p = (art_dir / "front_albedo_light.png") if art_dir else None
    if art_p and art_p.exists() and "UVMap_Art_Front" in me_s.uv_layers:
        im_art = bpy.data.images.load(str(art_p), check_existing=False)
        im_art.colorspace_settings.name = "sRGB"
        n_art = nt_m.nodes.new("ShaderNodeTexImage")
        n_art.image = im_art
        n_art.extension = "CLIP"
        n_uv = nt_m.nodes.new("ShaderNodeUVMap")
        n_uv.uv_map = "UVMap_Art_Front"
        n_mix = nt_m.nodes.new("ShaderNodeMix")
        n_mix.data_type = "RGBA"
        n_mix.inputs[6].default_value = (0.82, 0.79, 0.74, 1.0)
        nt_m.links.new(n_uv.outputs["UV"], n_art.inputs["Vector"])
        nt_m.links.new(n_art.outputs["Alpha"], n_mix.inputs[0])
        nt_m.links.new(n_art.outputs["Color"], n_mix.inputs[7])
        nt_m.links.new(n_mix.outputs[2], bsdf_m.inputs["Base Color"])
    else:
        bsdf_m.inputs["Base Color"].default_value = (0.82, 0.79, 0.74, 1.0)

    mat_rib = bpy.data.materials.new("BS_Ringer_Rib_Cycles")
    mat_rib.use_nodes = True
    bsdf_r = next(n for n in mat_rib.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
    bsdf_r.inputs["Base Color"].default_value = (0.011, 0.011, 0.010, 1.0)
    bsdf_r.inputs["Roughness"].default_value = 0.85
    me_s.materials.append(mat_main)
    for _ in range(max(len(shell["mat_names"]) - 1, 3)):
        me_s.materials.append(mat_rib)
    me_s.polygons.foreach_set("material_index", shell["tri_mats"].astype(np.int32))
    me_s.update()

    shirt_ob = bpy.data.objects.new("BS_RINGER_001_M9", me_s)
    sc.collection.objects.link(shirt_ob)
    # Outward solidify gives 1.5 mm physical hem thickness at render time without moving inner boundary < 3 mm
    sol = shirt_ob.modifiers.new("Solidify", "SOLIDIFY")
    sol.thickness = 0.0015
    sol.offset = 1.0

    def configure_cycles(transparent: bool):
        sc.render.engine = "CYCLES"
        dev = "CPU"
        if A.get("device", "gpu") == "gpu":
            prefs = bpy.context.preferences.addons["cycles"].preferences
            for dt in ("OPTIX", "CUDA"):
                try:
                    prefs.compute_device_type = dt
                    prefs.get_devices()
                    if any(d.type == dt for d in prefs.devices):
                        for d in prefs.devices:
                            d.use = d.type == dt
                        dev = dt
                        break
                except Exception:
                    pass
        sc.cycles.device = "GPU" if dev != "CPU" else "CPU"
        sc.cycles.samples = samples
        sc.cycles.use_adaptive_sampling = True
        sc.cycles.adaptive_threshold = 0.03
        sc.cycles.use_denoising = True
        sc.render.resolution_x = res_w
        sc.render.resolution_y = res_h
        sc.render.resolution_percentage = 100
        sc.render.film_transparent = transparent
        sc.render.image_settings.file_format = "PNG"
        sc.render.image_settings.color_mode = "RGBA"
        try:
            sc.view_settings.view_transform = "AgX"
        except TypeError:
            sc.view_settings.view_transform = "Standard"
        sc.view_settings.exposure = float(A.get("exposure", "-0.85"))
        return dev

    def run_dressed_pose():
        rig.data.pose_position = "POSE"
        body.material_slots[0].material = bpy.data.materials.get("MX_M1_ObsidianNoir") or body.material_slots[0].material
        arr = C.mesh_arrays(me)
        V_basis, T_mesh = arr["V"], arr["T"]
        P = posing.Poser(bpy, rig)
        ev = lambda: C.evaluated_positions(body)  # noqa: E731
        fit_bones = json.loads(Path(A["theta_fit"]).read_text(encoding="utf-8"))["bones"]
        p2_bones = json.loads((root / "poses" / "theta_contrapposto.json").read_text(encoding="utf-8"))["bones"]

        P.reset()
        P.apply_local(fit_bones)
        P.lock_clavicles()
        P.ground(ev)
        start_pose = {
            pb.name: {
                "q": M8.quat_canon([float(pb.rotation_quaternion.w), float(pb.rotation_quaternion.x), float(pb.rotation_quaternion.y), float(pb.rotation_quaternion.z)]),
                "t": np.array([float(pb.location.x), float(pb.location.y), float(pb.location.z)], dtype=np.float64),
            }
            for pb in rig.pose.bones
        }
        P.reset()
        P.apply_local(p2_bones)
        for op in [
            {"op": "rotate", "bone": "root", "axis": "U", "deg": -7.5},
            {"op": "rotate", "bone": "spine02", "axis": "U", "deg": 4.0},
            {"op": "arm_abduction", "side": "L", "deg": 10.0},
            {"op": "arm_abduction", "side": "R", "deg": 13.0},
            {"op": "elbow_flex", "side": "L", "deg": 19.0},
            {"op": "elbow_flex", "side": "R", "deg": 38.0, "medial": 0.25},
        ]:
            P.op(op)
        P.lock_clavicles()
        P.ground(ev)
        target_pose = {
            pb.name: {
                "q": M8.quat_canon([float(pb.rotation_quaternion.w), float(pb.rotation_quaternion.x), float(pb.rotation_quaternion.y), float(pb.rotation_quaternion.z)]),
                "t": np.array([float(pb.location.x), float(pb.location.y), float(pb.location.z)], dtype=np.float64),
            }
            for pb in rig.pose.bones
        }

        dev = configure_cycles(transparent=False)
        cd = bpy.data.cameras.new("MX_M9_Cam")
        cd.lens = 85.0
        cd.sensor_fit = "VERTICAL"
        cd.sensor_height = M8.SENSOR_MM
        cam = bpy.data.objects.new("MX_M9_Cam", cd)
        sc.collection.objects.link(cam)
        sc.camera = cam
        R.light(bpy, sc, "MX_M9_Key", "AREA", 370.0, 1.5, (2.2, -2.4, 2.5), (0.0, 0.0, 1.15))
        R.light(bpy, sc, "MX_M9_Fill", "AREA", 70.0, 2.8, (-2.5, -1.7, 1.6), (0.0, 0.0, 1.10))
        R.light(bpy, sc, "MX_M9_RimL", "AREA", 180.0, 0.6, (1.85, 1.85, 1.3), (0.0, 0.0, 1.05))
        R.light(bpy, sc, "MX_M9_RimR", "AREA", 180.0, 0.6, (-1.85, 1.85, 1.3), (0.0, 0.0, 1.05))

        n_frames = int(A.get("frames", "72"))
        frames_dir = out_dir / "dressed_pose_frames"
        frames_dir.mkdir(exist_ok=True)
        bone_names = [b.name for b in rig.data.bones]
        lags = {bn: M8.bone_phase_lag_s(bn) for bn in bone_names}
        pen_max = 0
        g5_max = 0
        for fi in range(n_frames):
            t = fi / 24.0
            for bn in bone_names:
                pb = rig.pose.bones[bn]
                pb.rotation_mode = "QUATERNION"
                if bn in ("clavicle.L", "clavicle.R"):
                    pb.rotation_quaternion = Quaternion((1.0, 0.0, 0.0, 0.0))
                    pb.location = (0.0, 0.0, 0.0)
                    continue
                lag = lags[bn]
                tau = float(np.clip((t - lag) / max(2.2 - lag, 1e-6), 0.0, 1.0))
                sig = float(M8.min_jerk(tau))
                q_t = M8.quat_min_jerk(start_pose[bn]["q"], target_pose[bn]["q"], sig)
                t_t = (1.0 - sig) * start_pose[bn]["t"] + sig * target_pose[bn]["t"]
                pb.rotation_quaternion = Quaternion(tuple(float(v) for v in q_t))
                pb.location = tuple(float(v) for v in t_t)
            bpy.context.view_layer.update()
            P.ground(ev)
            Xb = ev()
            g5_max = max(g5_max, int(geometry.collapsed_triangles(V_basis, Xb, T_mesh, ratio=0.25)["count"]))
            V_shirt_t, st = GT.transport_to_body(gbind, Xb, mode="clearance", eps_body=0.003, verify_exact=(dev != "CPU"))
            pen_max = max(pen_max, int(st.get("inside_gwn_gt_05", st.get("inside_fast", 0))))
            me_s.vertices.foreach_set("co", V_shirt_t.astype(np.float32).ravel())
            me_s.update()

            u = fi / max(n_frames - 1, 1)
            s_cam = float(M8.critically_damped_ease(u, omega=5.8))
            az = 35.0 * (1.0 - s_cam) + 10.0 * s_cam
            lens = 85.0 * (1.0 - s_cam) + 105.0 * s_cam
            dist = 4.15 * (1.0 - s_cam) + 3.35 * s_cam
            cam_z = 1.04 * (1.0 - s_cam) + 0.90 * s_cam
            cloc, ctgt, _, _ = M8.solve_pitched_camera_headroom(Xb, np.array([0.0, -0.02]), az, lens, cam_z, dist, top_margin=0.070)
            cam.data.lens = lens
            cam.location = tuple(float(v) for v in cloc)
            R.look_at(cam, tuple(float(v) for v in ctgt))
            io.render_still(sc, frames_dir / f"frame_{fi + 1:04d}.png")

        return {"dir": str(frames_dir)}, {"frames": n_frames, "max_G5": g5_max, "max_penetrations": pen_max, "device": dev}

    def run_stills():
        dev = configure_cycles(transparent=True)
        cam, _ = R.studio(bpy, sc, samples=samples, res=(1200, 1500), exposure=-1.0)
        rig.data.pose_position = "POSE"
        P = posing.Poser(bpy, rig)
        ev = lambda: C.evaluated_positions(body)  # noqa: E731
        P.reset()
        P.apply_local(json.loads(Path(A["theta_fit"]).read_text(encoding="utf-8"))["bones"])
        P.lock_clavicles()
        P.ground(ev)
        Xb = ev()
        V_shirt_t, st = GT.transport_to_body(gbind, Xb, mode="clearance", eps_body=0.003, verify_exact=True)
        me_s.vertices.foreach_set("co", V_shirt_t.astype(np.float32).ravel())
        me_s.update()
        renders = []
        for finish in ("M1", "M2", "M3"):
            mat_name = {"M1": "MX_M1_ObsidianNoir", "M2": "MX_M2_AlabasterChalk", "M3": "MX_M3_TravertineWarm"}.get(finish)
            if mat_name in bpy.data.materials:
                body.material_slots[0].material = bpy.data.materials[mat_name]
            for vname, ang in (("front", 0.0), ("three_quarter", 35.0), ("back", 180.0)):
                R.frame_camera(cam, Xb, view_deg=ang)
                out_png = out_dir / finish / f"BS-RINGER-001_white-black_mannequin_{finish}_{vname}.png"
                out_png.parent.mkdir(parents=True, exist_ok=True)
                io.render_still(sc, out_png)
                renders.append(str(out_png))
        return {"renders": renders}, {"stills": len(renders), "penetration_stats": st, "device": dev}

    def run_dressed_walk():
        dev = configure_cycles(transparent=False)
        rig.data.pose_position = "POSE"
        body.material_slots[0].material = bpy.data.materials.get("MX_M1_ObsidianNoir") or body.material_slots[0].material
        arr = C.mesh_arrays(me)
        V_basis, T_mesh = arr["V"], arr["T"]
        cd = bpy.data.cameras.new("MX_M9_WalkCam")
        cd.lens = 85.0
        cd.sensor_fit = "VERTICAL"
        cd.sensor_height = M8.SENSOR_MM
        cam = bpy.data.objects.new("MX_M9_WalkCam", cd)
        sc.collection.objects.link(cam)
        sc.camera = cam
        R.light(bpy, sc, "MX_M9_WKey", "AREA", 380.0, 1.6, (2.0, -2.2, 2.6), (0, 0, 1.15))
        R.light(bpy, sc, "MX_M9_WFill", "AREA", 62.0, 3.0, (-2.3, -1.6, 1.6), (0, 0, 1.05))
        R.light(bpy, sc, "MX_M9_WRimL", "AREA", 220.0, 0.55, (1.85, 1.85, 1.35), (0, 0, 1.05))
        R.light(bpy, sc, "MX_M9_WRimR", "AREA", 220.0, 0.55, (-1.85, 1.85, 1.35), (0, 0, 1.05))
        n_frames = int(A.get("frames", "250"))
        frames_dir = out_dir / "dressed_walk_frames"
        frames_dir.mkdir(exist_ok=True)
        pen_max = 0
        g5_max = 0
        for fi in range(n_frames):
            sc.frame_set(fi + 1)
            bpy.context.view_layer.update()
            Xb = C.evaluated_positions(body)
            g5_max = max(g5_max, int(geometry.collapsed_triangles(V_basis, Xb, T_mesh, ratio=0.25)["count"]))
            V_shirt_t, st = GT.transport_to_body(gbind, Xb, mode="clearance", eps_body=0.003, verify_exact=(dev != "CPU"))
            pen_max = max(pen_max, int(st.get("inside_gwn_gt_05", st.get("inside_fast", 0))))
            me_s.vertices.foreach_set("co", V_shirt_t.astype(np.float32).ravel())
            me_s.update()
            p_root = np.array(rig.matrix_world @ rig.pose.bones["root"].head, dtype=np.float64)
            p_chest = np.array(rig.matrix_world @ rig.pose.bones["spine02"].head, dtype=np.float64)
            anchor = 0.5 * (p_root[:2] + p_chest[:2])
            u = fi / max(n_frames - 1, 1)
            s_b = 0.35 * u + 0.65 * (0.5 - 0.5 * np.cos(np.pi * u))
            az = -14.0 + 30.0 * s_b
            lens = 85.0 + 15.0 * s_b
            cloc, ctgt, _, _, _ = M8.solve_walk_editorial_camera(Xb, anchor, az, lens_mm=lens, cam_z=0.92)
            cam.data.lens = float(lens)
            cam.location = tuple(float(v) for v in cloc)
            R.look_at(cam, tuple(float(v) for v in ctgt))
            io.render_still(sc, frames_dir / f"frame_{fi + 1:04d}.png")
        return {"dir": str(frames_dir)}, {"frames": n_frames, "max_G5": g5_max, "max_penetrations": pen_max, "device": dev}

    C.run(f"{STAGE}_{mode}", {"pose": run_dressed_pose, "walk": run_dressed_walk, "stills": run_stills}[mode])


def launch():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default="/content/maximus_v2/bundle")
    ap.add_argument("--master-psd", default="/content/drive/MyDrive/maximus_v2/results/M4_canon_atlas/maximus_v2_master_psd.blend")
    ap.add_argument("--walk-scene", default="/content/drive/MyDrive/maximus_v2/results/M5_kinematics/m5_walk_scene.blend")
    ap.add_argument("--shell-npz", default="/content/drive/MyDrive/maximus_v2/results/M9_dressed/hero_ringer_shell.npz")
    ap.add_argument("--art-dir", default="/content/drive/MyDrive/maximus_v2/results/M9_dressed/art")
    ap.add_argument("--out", default="/content/drive/MyDrive/maximus_v2/results/M9_dressed")
    ap.add_argument("--mode", default="pose", choices=["pose", "walk", "stills"])
    ap.add_argument("--blender", default=None)
    ap.add_argument("--frames", default="72")
    ap.add_argument("--samples", default="48")
    args = ap.parse_args()

    exe = Path(args.blender) if args.blender else next(Path("/content/blender").glob("blender-*/blender"))
    fit = next(p for p in (HERE / "m_theta_fit.json", Path("/content/drive/MyDrive/maximus_v2/m_theta_fit.json")) if p.exists())
    work = Path("/content/m9_work")
    work.mkdir(parents=True, exist_ok=True)
    blend_in = args.walk_scene if args.mode == "walk" else args.master_psd
    cmd = [
        str(exe), "--background", str(blend_in), "--factory-startup", "--python-exit-code", "3",
        "--python", str(Path(__file__).resolve()), "--",
        f"root={args.root}", f"out_dir={work}", f"mode={args.mode}", f"theta_fit={fit}",
        f"shell_npz={args.shell_npz}", f"art_dir={args.art_dir}",
        f"frames={args.frames}", f"samples={args.samples}",
    ]
    p = subprocess.run(cmd, env=dict(os.environ, MAXIMUS_STRICT="1"), capture_output=True, text=True)
    (work / f"m9_{args.mode}.log").write_text(p.stdout + p.stderr, encoding="utf-8")
    if p.returncode != 0:
        print(p.stdout[-3000:], p.stderr[-3000:])
        sys.exit(p.returncode)

    dst = Path(args.out)
    dst.mkdir(parents=True, exist_ok=True)
    if args.mode in ("pose", "walk"):
        mp4_name = "maximus_v2_dressed_pose_strike.mp4" if args.mode == "pose" else "maximus_v2_dressed_walk_reel.mp4"
        f_sub = "dressed_pose_frames" if args.mode == "pose" else "dressed_walk_frames"
        mp4 = work / mp4_name
        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error", "-framerate", "24",
            "-i", str(work / f_sub / "frame_%04d.png"),
            "-c:v", "libx264", "-preset", "slow", "-pix_fmt", "yuv420p", "-crf", "19", "-movflags", "+faststart", str(mp4),
        ], check=True)
        shutil.copy2(mp4, dst / mp4.name)
        with open(dst / "MANIFEST_SHA256.txt", "a", encoding="utf-8", newline="\n") as mf:
            mf.write(f"{sha256(mp4)}  {mp4.name}\n")
        assert sha256(dst / mp4.name) == sha256(mp4), "Drive SHA-256 readback mismatch"
        print(json.dumps({"mp4": str(dst / mp4.name), "sha256": sha256(dst / mp4.name)}, indent=2))


if __name__ == "__main__":
    blender_main() if IN_BLENDER else launch()
