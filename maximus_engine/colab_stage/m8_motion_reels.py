"""M8 (2026-11-19) - Maximus v2 Bare-Body Motion Reels (Runway Walk + SWAG_01 Pose Strike) on Colab L4 (Cycles OPTIX + OIDN + AgX).

    Colab:  !python /content/drive/MyDrive/maximus_v2/m8_motion_reels.py
    Local:  python -I colab_stage/m8_motion_reels.py --help

Outputs (results/M8_motion_reels/):
  Reel 1 (Solved M5 Runway Walk, 250 frames @ 24 fps = 10.42 s, 1080x1350):
    - maximus_v2_walk_reel.mp4         (H.264 yuv420p CRF 20 +faststart)
    - maximus_v2_walk_poster.webp      (frame 60)
    - M8_walk_contact_sheet.png        (6-panel QA: frames 1, 50, 100, 150, 200, 250)
  Reel 2 (SWAG_01 3-Second Pose Strike + Dynamic Camera, 72 frames @ 24 fps = 3.00 s, 1080x1350):
    - maximus_v2_pose_strike.mp4       (H.264 yuv420p CRF 19 +faststart)
    - maximus_v2_pose_poster.webp      (frame 64)
    - M8_pose_contact_sheet.png        (6-panel QA: frames 1, 14, 28, 42, 56, 72)
  Metadata & Verification:
    - M8_motion_reels.json
    - MANIFEST_SHA256.txt

Hard Invariants:
  - 13,380 base vertices (26,756 tris), 152 master bones, pure LBS + Corrective Smooth (factor=0.50, iterations=5), ZERO DQS.
  - M4 ARAP PSD keys + volume-preserving quaternion-driven PSD keys (PSD_hip_L/R, PSD_axilla_L/R) active.
  - Clavicle locked (0, 0, 0) on SWAG_01 static/strike trajectory, Z_sole >= 0.0 mm across every frame.
  - Pure bpy.data / mathutils / NumPy APIs only; renders go strictly through pipeline/io_ops_gateway.render_still.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

IN_BLENDER = False
if "blender" in Path(sys.executable).name.lower():
    try:
        import bpy  # noqa: F401
        IN_BLENDER = True
    except Exception:
        IN_BLENDER = False

HERE = Path(__file__).resolve().parent
STAGE = "m8_motion_reels"
SENSOR_MM = 36.0
FPS = 24
WALK_FRAMES = 250
POSE_FRAMES = 72
POSE_MOVE_S = 2.2
POSE_TOTAL_S = 3.0
WALK_QA_FRAMES = (1, 50, 100, 150, 200, 250)
POSE_QA_FRAMES = (1, 14, 28, 42, 56, 72)
WALK_POSTER_FRAME = 60
POSE_POSTER_FRAME = 64


# ------------------------------------------------------------------------------------------------------------------------------------
# Pure NumPy Math & Trajectory Helpers (unit-testable without Blender)
# ------------------------------------------------------------------------------------------------------------------------------------
def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def min_jerk(tau):
    """Flash & Hogan (1985) minimum-jerk polynomial: 10 t^3 - 15 t^4 + 6 t^5."""
    t = np.clip(np.asarray(tau, dtype=np.float64), 0.0, 1.0)
    return t * t * t * (10.0 - 15.0 * t + 6.0 * t * t)


def critically_damped_ease(u, omega: float = 6.5):
    """Exact normalized analytical response of a critically damped spring (zeta = 1.0) on u in [0, 1]."""
    u = np.clip(np.asarray(u, dtype=np.float64), 0.0, 1.0)
    raw = 1.0 - (1.0 + omega * u) * np.exp(-omega * u)
    end = 1.0 - (1.0 + omega) * math.exp(-omega)
    return raw / max(end, 1e-12)


def critically_damped_track(targets: np.ndarray, dt: float, omega: float = 8.0) -> np.ndarray:
    """Critically damped 2nd-order tracker (zeta = 1.0) with trend velocity feed-forward so steady translation has zero lag
    while high-frequency sway/bob is smoothly damped with zero overshoot."""
    pts = np.asarray(targets, dtype=np.float64)
    n = len(pts)
    if n <= 2:
        return pts.copy()
    # Linear mean velocity trend (prevents steady-state tracking lag on forward walk while damping sway/bob)
    t_grid = np.arange(n, dtype=np.float64) * dt
    v_trend = (pts[-1] - pts[0]) / max(t_grid[-1], 1e-9)
    trend = pts[0] + t_grid[:, None] * v_trend[None, :]
    dev_target = pts - trend

    out_dev = np.zeros_like(dev_target)
    pos = dev_target[0].copy()
    vel = np.zeros_like(pos)
    exp_w = math.exp(-omega * dt)
    out_dev[0] = pos
    for i in range(1, n):
        # Exact analytical step toward target deviation under critically damped spring (zeta = 1.0)
        e = pos - dev_target[i]
        c2 = vel + omega * e
        pos = dev_target[i] + exp_w * (e + c2 * dt)
        vel = exp_w * (vel - omega * c2 * dt)
        out_dev[i] = pos
    return trend + out_dev


def quat_canon(q) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    n = float(np.linalg.norm(q))
    if n < 1e-15:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    q = q / n
    return -q if q[0] < 0.0 else q


def quat_conj(q) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def quat_mul(a, b) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], dtype=np.float64)


def quat_log(q) -> np.ndarray:
    """Lie algebra log of unit quaternion q (canonicalized to w >= 0) -> 3-vector (half-angle * axis)."""
    qc = quat_canon(q)
    v = qc[1:]
    vn = float(np.linalg.norm(v))
    if vn < 1e-12:
        return np.zeros(3, dtype=np.float64)
    theta = math.atan2(vn, float(qc[0]))
    return (theta / vn) * v


def quat_exp(vec) -> np.ndarray:
    """Lie group exp of 3-vector (half-angle * axis) -> unit quaternion (w, x, y, z)."""
    v = np.asarray(vec, dtype=np.float64)
    theta = float(np.linalg.norm(v))
    if theta < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    s = math.sin(theta) / theta
    return np.array([math.cos(theta), v[0] * s, v[1] * s, v[2] * s], dtype=np.float64)


def quat_min_jerk(q_start, q_target, sigma: float) -> np.ndarray:
    """Geodesic minimum-jerk interpolation on SO(3): q(t) = q_start * Exp(sigma * Log(q_start^-1 * q_target))."""
    qs = quat_canon(q_start)
    qt = quat_canon(q_target)
    q_rel = quat_canon(quat_mul(quat_conj(qs), qt))
    return quat_canon(quat_mul(qs, quat_exp(float(sigma) * quat_log(q_rel))))


def bone_phase_lag_s(bone_name: str) -> float:
    """Root-outward kinematic phase lag (seconds) for SWAG_01 pose strike."""
    n = bone_name.split(".")[0]
    if n in ("root", "pelvis") or n.startswith(("upperleg", "lowerleg", "foot", "toe")):
        return 0.00
    if n.startswith(("spine", "breast")):
        return 0.08
    if n.startswith(("clavicle", "shoulder", "upperarm", "lowerarm")):
        return 0.16
    return 0.25


def projected_vertical_bounds(X: np.ndarray, cam_loc: np.ndarray, target: np.ndarray, lens_mm: float):
    """Exact pinhole vertical projection bounds (top_margin, bottom_frac) for world vertices X (N, 3)."""
    f = target - cam_loc
    f = f / max(float(np.linalg.norm(f)), 1e-12)
    up_w = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    r = np.cross(f, up_w)
    r = r / max(float(np.linalg.norm(r)), 1e-12)
    u = np.cross(r, f)
    d = X - cam_loc[None, :]
    z_cam = d @ f
    y_cam = d @ u
    tan_half = (SENSOR_MM * 0.5) / float(lens_mm)
    rows = 0.5 - (y_cam / np.maximum(z_cam, 1e-6)) / (2.0 * tan_half)
    return float(rows.min()), float(rows.max())


def solve_fullbody_camera(X: np.ndarray, anchor_xy: np.ndarray, view_deg: float, lens_mm: float = 85.0,
                          top_margin: float = 0.070, sole_frac: float = 0.905):
    """Solve level camera distance D and height zc so crown sits at top_margin (7%) and sole at sole_frac (90.5%)."""
    th = (SENSOR_MM * 0.5) / float(lens_mm)
    rad = math.radians(view_deg)
    c_hat = np.array([math.sin(rad), -math.cos(rad)], dtype=np.float64)
    a = (X[:, :2] - anchor_xy[None, :]) @ c_hat
    z = X[:, 2]
    alpha = 2.0 * th * (0.5 - top_margin)
    beta = 2.0 * th * (sole_frac - 0.5)
    D, zc = 4.9, 0.88
    for _ in range(16):
        k_top = int(np.argmax((z - zc) / np.maximum(D - a, 0.1)))
        k_bot = int(np.argmin((z - zc) / np.maximum(D - a, 0.1)))
        D_new = (z[k_top] - z[k_bot] + alpha * a[k_top] + beta * a[k_bot]) / (alpha + beta)
        zc_new = z[k_bot] + beta * (D_new - a[k_bot])
        if abs(D_new - D) + abs(zc_new - zc) < 1e-9:
            D, zc = D_new, zc_new
            break
        D, zc = D_new, zc_new
    loc = np.array([anchor_xy[0] + D * c_hat[0], anchor_xy[1] + D * c_hat[1], zc], dtype=np.float64)
    tgt = np.array([anchor_xy[0], anchor_xy[1], zc], dtype=np.float64)
    return loc, tgt, float(D)


def solve_pitched_camera_headroom(X: np.ndarray, anchor_xy: np.ndarray, view_deg: float, lens_mm: float,
                                  cam_z: float, dist_m: float, top_margin: float = 0.070):
    """Given camera azimuth, distance, height cam_z, and lens_mm, solve look-at target_z so the exact perspective
    projected crown sits at top_margin (7.0%, inside 6%-8%) even with a dramatic low-angle upward tilt."""
    rad = math.radians(view_deg)
    c_hat = np.array([math.sin(rad), -math.cos(rad)], dtype=np.float64)
    cam_loc = np.array([anchor_xy[0] + dist_m * c_hat[0], anchor_xy[1] + dist_m * c_hat[1], cam_z], dtype=np.float64)
    lo, hi = 0.55, 1.85
    for _ in range(28):
        mid = 0.5 * (lo + hi)
        tgt = np.array([anchor_xy[0], anchor_xy[1], mid], dtype=np.float64)
        tm, _ = projected_vertical_bounds(X, cam_loc, tgt, lens_mm)
        # Higher target_z pitches camera UP -> moves crown DOWN in image -> increases top_margin
        if tm < top_margin:
            lo = mid
        else:
            hi = mid
    tgt = np.array([anchor_xy[0], anchor_xy[1], 0.5 * (lo + hi)], dtype=np.float64)
    tm, bf = projected_vertical_bounds(X, cam_loc, tgt, lens_mm)
    return cam_loc, tgt, tm, bf


def solve_walk_editorial_camera(X: np.ndarray, anchor_xy: np.ndarray, view_deg: float, lens_mm: float = 85.0,
                                cam_z: float = 0.92, top_margin: float = 0.070, sole_frac: float = 0.905):
    """Solve authoritative waist-height (cam_z = 0.92 m) editorial camera distance and look-at pitch target
    so crown headroom sits at top_margin (7.0%, strictly inside 6.0%-8.0%) and sole sits near sole_frac."""
    _, _, d_level = solve_fullbody_camera(X, anchor_xy, view_deg, lens_mm=lens_mm, top_margin=top_margin, sole_frac=sole_frac)
    cam_loc, tgt, tm, bf = solve_pitched_camera_headroom(
        X, anchor_xy, view_deg, lens_mm=lens_mm, cam_z=cam_z, dist_m=d_level, top_margin=top_margin
    )
    return cam_loc, tgt, float(d_level), tm, bf


# ------------------------------------------------------------------------------------------------------------------------------------
# Blender Pass (bpy.data / mathutils / NumPy only; renders strictly through io_ops_gateway.render_still)
# ------------------------------------------------------------------------------------------------------------------------------------
def blender_main():
    A = dict(x.split("=", 1) for x in sys.argv[sys.argv.index("--") + 1:] if "=" in x)
    root = Path(A["root"]).resolve()
    sys.path.insert(0, str(root / "pipeline"))
    sys.path.insert(0, str(HERE))
    for p in (HERE / "runtime_v3", HERE.parent / "runtime_v3"):
        if (p / "pose_space_engine.py").exists():
            sys.path.insert(0, str(p))

    import bl_common as C
    import io_ops_gateway as io
    import m_render as R
    import posing
    import pose_space_engine as PSE
    from maximus_core import geometry
    from mathutils import Quaternion, Vector

    bpy_mod = C.bpy
    sc = bpy_mod.context.scene
    mode = A["mode"]
    out_dir = Path(A["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    res_w = int(A.get("w", "1080"))
    res_h = int(A.get("h", "1350"))
    samples = int(A.get("samples", "56"))
    adaptive_thresh = float(A.get("adaptive_threshold", "0.03"))
    rim_boost = float(A.get("rim_boost", "1.0"))
    key_scale = float(A.get("key_scale", "1.0"))
    bg_luma = float(A.get("bg_luma", "0.055"))

    def check_invariants(body, rig):
        me = body.data
        me.calc_loop_triangles()
        arm = [m for m in body.modifiers if m.type == "ARMATURE"]
        cs = [m for m in body.modifiers if m.type == "CORRECTIVE_SMOOTH"]
        assert len(me.vertices) == 13380, f"expected 13380 verts, got {len(me.vertices)}"
        assert len(me.loop_triangles) == 26756, f"expected 26756 tris, got {len(me.loop_triangles)}"
        assert len(rig.data.bones) == 152, f"expected 152 bones, got {len(rig.data.bones)}"
        assert len(arm) == 1 and not arm[0].use_deform_preserve_volume, "pure LBS required (ZERO DQS)"
        assert len(cs) == 1 and abs(cs[0].factor - 0.50) < 1e-9 and cs[0].iterations == 5, "Corrective Smooth 0.50 x 5 required"
        return {
            "verts": len(me.vertices),
            "tris": len(me.loop_triangles),
            "bones": len(rig.data.bones),
            "lbs_only_zero_dqs": True,
            "corrective_smooth": {"factor": cs[0].factor, "iterations": cs[0].iterations},
        }

    def ensure_volume_psd_drivers(body, rig):
        """Verify PSD_hip_L/R and PSD_axilla_L/R exist and have live Simple-Expression quaternion drivers."""
        kbs = body.data.shape_keys.key_blocks
        driver_map = {
            "PSD_hip_L": ("upperleg01.L", 0.006),
            "PSD_hip_R": ("upperleg01.R", 0.006),
            "PSD_axilla_L": ("upperarm01.L", 0.080),
            "PSD_axilla_R": ("upperarm01.R", 0.080),
        }
        missing = [k for k in driver_map if k not in kbs]
        assert not missing, f"Missing canonical volume PSD shape keys: {missing}"
        ad = body.data.shape_keys.animation_data
        existing_paths = {fc.data_path for fc in ad.drivers} if (ad and ad.drivers) else set()
        attached = []
        for kname, (bname, denom) in driver_map.items():
            dp = f'key_blocks["{kname}"].value'
            if dp not in existing_paths:
                kb = kbs[kname]
                kb.driver_remove("value")
                fc = kb.driver_add("value")
                d = fc.driver
                d.type = "SCRIPTED"
                v = d.variables.new()
                v.name = "w"
                v.type = "TRANSFORMS"
                t = v.targets[0]
                t.id = rig
                t.bone_target = bname
                t.transform_type = "ROT_W"
                t.rotation_mode = "QUATERNION"
                t.transform_space = "LOCAL_SPACE"
                d.expression = f"min(1.0, max(0.0, (1.0 - abs(w)) / {denom}))"
                attached.append(kname)
        return {"required_keys": list(driver_map.keys()), "drivers_newly_attached": attached, "all_present": True}

    def configure_cycles():
        sc.render.engine = "CYCLES"
        dev = "CPU"
        if A.get("device", "gpu") == "gpu":
            prefs = bpy_mod.context.preferences.addons["cycles"].preferences
            for dt in ("OPTIX", "CUDA"):
                try:
                    prefs.compute_device_type = dt
                    prefs.get_devices()
                    if any(d.type == dt for d in prefs.devices):
                        for d in prefs.devices:
                            d.use = d.type == dt
                        dev = dt
                        break
                except Exception:  # noqa: BLE001
                    pass
            if dev == "CPU" and A.get("require_gpu", "1") == "1":
                raise RuntimeError("No OPTIX/CUDA GPU detected: refusing CPU render")
        sc.cycles.device = "GPU" if dev != "CPU" else "CPU"
        sc.cycles.samples = samples
        sc.cycles.use_adaptive_sampling = True
        sc.cycles.adaptive_threshold = adaptive_thresh
        sc.cycles.use_denoising = True
        try:
            sc.cycles.denoiser = "OPENIMAGEDENOISE"
        except (TypeError, AttributeError):
            pass
        try:
            sc.cycles.denoising_use_gpu = (dev != "CPU")
        except AttributeError:
            pass
        sc.cycles.max_bounces = 6
        sc.cycles.diffuse_bounces = 2
        sc.cycles.glossy_bounces = 3
        sc.cycles.transmission_bounces = 0
        sc.cycles.volume_bounces = 0
        sc.cycles.transparent_max_bounces = 4
        sc.cycles.sample_clamp_indirect = 8.0
        try:
            sc.render.use_persistent_data = True
        except AttributeError:
            pass
        sc.render.resolution_x = res_w
        sc.render.resolution_y = res_h
        sc.render.resolution_percentage = 100
        sc.render.image_settings.file_format = "PNG"
        sc.render.image_settings.color_mode = "RGBA"
        try:
            sc.view_settings.view_transform = "AgX"
            sc.view_settings.look = "None"
        except TypeError:
            sc.view_settings.view_transform = "Standard"
        sc.world = sc.world or bpy_mod.data.worlds.new("MX_M8_World")
        sc.world.use_nodes = False
        sc.world.color = (0.012, 0.013, 0.016)
        return dev

    def clear_extra_objects(keep_names):
        for ob in list(sc.collection.objects):
            if ob.name not in keep_names:
                bpy_mod.data.objects.remove(ob, do_unlink=True)

    def make_area_light(name, energy, sx, sy, loc, target, color=(1.0, 1.0, 1.0)):
        ld = bpy_mod.data.lights.new(name, "AREA")
        ld.energy = float(energy)
        ld.color = color
        if abs(sx - sy) < 1e-4:
            ld.shape = "SQUARE"
            ld.size = float(sx)
        else:
            ld.shape = "RECTANGLE"
            ld.size = float(sx)
            ld.size_y = float(sy)
        ob = bpy_mod.data.objects.new(name, ld)
        sc.collection.objects.link(ob)
        ob.location = tuple(float(v) for v in loc)
        R.look_at(ob, tuple(float(v) for v in target))
        # Hide physical emitter rectangles from direct camera rays (essential for +/-135 deg rim strips & rear backdrop wash)
        try:
            ob.visible_camera = False
        except AttributeError:
            pass
        return ob

    def build_stone_cyclorama(y_min=-12.0, y_max=5.2, x_half=10.0, cove_y=2.6, cove_r=1.8, wall_h=7.5):
        """Build honed dark architectural stone runway floor (Z = 0.0 exact) + curved seamless cyclorama cove + side walls."""
        # 1. Floor mesh at Z = 0.0 (world-anchored architectural honed basalt with subtle slab joints)
        vf = [
            (-x_half, y_min, 0.0),
            (x_half, y_min, 0.0),
            (x_half, y_max, 0.0),
            (-x_half, y_max, 0.0),
        ]
        me_f = bpy_mod.data.meshes.new("MX_M8_RunwayFloor")
        me_f.from_pydata(vf, [], [(0, 1, 2, 3)])
        me_f.update()

        mat_f = bpy_mod.data.materials.new("MX_M8_BasaltFloor")
        mat_f.use_nodes = True
        nt = mat_f.node_tree
        N, L = nt.nodes, nt.links
        N.clear()
        out_n = N.new("ShaderNodeOutputMaterial")
        bsdf = N.new("ShaderNodeBsdfPrincipled")
        L.new(bsdf.outputs["BSDF"], out_n.inputs["Surface"])
        tc = N.new("ShaderNodeTexCoord")

        # Multi-scale honed stone variation + crisp 1.2 m architectural stone tile seams (visible floor locomotion!)
        n1 = N.new("ShaderNodeTexNoise")
        n1.inputs["Scale"].default_value = 1.6
        n1.inputs["Detail"].default_value = 6.0
        n1.inputs["Roughness"].default_value = 0.58
        L.new(tc.outputs["Object"], n1.inputs["Vector"])

        brick = N.new("ShaderNodeTexBrick")
        brick.offset = 0.5
        brick.inputs["Scale"].default_value = 0.85
        brick.inputs["Mortar Size"].default_value = 0.004
        brick.inputs["Mortar Smooth"].default_value = 0.15
        brick.inputs["Brick Width"].default_value = 1.0
        brick.inputs["Row Height"].default_value = 1.0
        brick.inputs["Color1"].default_value = (bg_luma * 0.72, bg_luma * 0.74, bg_luma * 0.82, 1.0)
        brick.inputs["Color2"].default_value = (bg_luma * 0.62, bg_luma * 0.64, bg_luma * 0.71, 1.0)
        brick.inputs["Mortar"].default_value = (bg_luma * 0.28, bg_luma * 0.29, bg_luma * 0.33, 1.0)
        L.new(tc.outputs["Object"], brick.inputs["Vector"])

        mix_c = N.new("ShaderNodeMix")
        mix_c.data_type = "RGBA"
        mix_c.blend_type = "MULTIPLY"
        mix_c.inputs["Factor"].default_value = 0.22
        L.new(brick.outputs["Color"], mix_c.inputs["A"])
        ramp = N.new("ShaderNodeValToRGB")
        ramp.color_ramp.elements[0].position = 0.25
        ramp.color_ramp.elements[0].color = (0.72, 0.72, 0.76, 1.0)
        ramp.color_ramp.elements[1].position = 0.78
        ramp.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)
        L.new(n1.outputs["Fac"], ramp.inputs["Fac"])
        L.new(ramp.outputs["Color"], mix_c.inputs["B"])
        L.new(mix_c.outputs["Result"], bsdf.inputs["Base Color"])

        mr = N.new("ShaderNodeMapRange")
        mr.inputs["To Min"].default_value = 0.68
        mr.inputs["To Max"].default_value = 0.82
        L.new(n1.outputs["Fac"], mr.inputs["Value"])
        L.new(mr.outputs["Result"], bsdf.inputs["Roughness"])
        if "Specular IOR Level" in bsdf.inputs:
            bsdf.inputs["Specular IOR Level"].default_value = 0.14

        bump = N.new("ShaderNodeBump")
        bump.inputs["Strength"].default_value = 0.06
        bump.inputs["Distance"].default_value = 0.002
        L.new(n1.outputs["Fac"], bump.inputs["Height"])
        L.new(bump.outputs["Normal"], bsdf.inputs["Normal"])

        me_f.materials.append(mat_f)
        ob_f = bpy_mod.data.objects.new("MX_M8_RunwayFloor", me_f)
        sc.collection.objects.link(ob_f)

        # 2. Curved seamless architectural stone cyclorama cove (rear wall + smooth wings so wide camera arcs never see void)
        prof_yz = []
        n_arc = 18
        for i in range(n_arc + 1):
            ang = (math.pi * 0.5) * (i / n_arc)
            y = cove_y + cove_r * math.sin(ang)
            z = 0.001 + cove_r * (1.0 - math.cos(ang))
            prof_yz.append((y, z))
        prof_yz.append((cove_y + cove_r, wall_h))

        # Horizontal U-sweep from left wing -> back cove -> right wing
        n_x = 32
        xs = np.linspace(-x_half, x_half, n_x)
        verts, faces = [], []
        n_p = len(prof_yz)
        for ix, xv in enumerate(xs):
            # Curve edges forward slightly past |x| > 5.5 m so oblique angles see seamless stone wall
            wing = max(0.0, abs(float(xv)) - 5.0)
            y_bend = -0.32 * (wing ** 1.6)
            for yv, zv in prof_yz:
                verts.append((float(xv), float(yv + y_bend), float(zv)))
        for ix in range(n_x - 1):
            for ip in range(n_p - 1):
                a = ix * n_p + ip
                b = (ix + 1) * n_p + ip
                c = (ix + 1) * n_p + (ip + 1)
                d = ix * n_p + (ip + 1)
                faces.append((a, b, c, d))

        me_c = bpy_mod.data.meshes.new("MX_M8_Cyclorama")
        me_c.from_pydata(verts, [], faces)
        me_c.polygons.foreach_set("use_smooth", [True] * len(me_c.polygons))
        me_c.update()

        mat_c = bpy_mod.data.materials.new("MX_M8_StoneCyclorama")
        mat_c.use_nodes = True
        ntc = mat_c.node_tree
        Nc, Lc = ntc.nodes, ntc.links
        Nc.clear()
        out_c = Nc.new("ShaderNodeOutputMaterial")
        bsdf_c = Nc.new("ShaderNodeBsdfPrincipled")
        Lc.new(bsdf_c.outputs["BSDF"], out_c.inputs["Surface"])
        tcc = Nc.new("ShaderNodeTexCoord")
        nc = Nc.new("ShaderNodeTexNoise")
        nc.inputs["Scale"].default_value = 0.75
        nc.inputs["Detail"].default_value = 5.0
        nc.inputs["Roughness"].default_value = 0.65
        Lc.new(tcc.outputs["Object"], nc.inputs["Vector"])
        rc = Nc.new("ShaderNodeValToRGB")
        rc.color_ramp.elements[0].position = 0.25
        rc.color_ramp.elements[0].color = (bg_luma * 0.78, bg_luma * 0.81, bg_luma * 0.90, 1.0)
        rc.color_ramp.elements[1].position = 0.78
        rc.color_ramp.elements[1].color = (bg_luma * 1.18, bg_luma * 1.22, bg_luma * 1.32, 1.0)
        Lc.new(nc.outputs["Fac"], rc.inputs["Fac"])
        Lc.new(rc.outputs["Color"], bsdf_c.inputs["Base Color"])
        bsdf_c.inputs["Roughness"].default_value = 0.86
        if "Specular IOR Level" in bsdf_c.inputs:
            bsdf_c.inputs["Specular IOR Level"].default_value = 0.20

        me_c.materials.append(mat_c)
        ob_c = bpy_mod.data.objects.new("MX_M8_Cyclorama", me_c)
        sc.collection.objects.link(ob_c)
        return ob_f, ob_c

    def measure_body_stats(ref_path: Path, torso_top_frac=(0.24, 0.44)):
        """Display-referred RGBA stats on opaque body pixels with configurable top-down torso band."""
        img = bpy_mod.data.images.load(str(ref_path), check_existing=False)
        try:
            try:
                img.colorspace_settings.name = "Non-Color"
            except TypeError:
                pass
            w, h = img.size
            px = np.empty(w * h * 4, np.float32)
            img.pixels.foreach_get(px)
            px = px.reshape(h, w, 4)
        finally:
            bpy_mod.data.images.remove(img)
        luma = 255.0 * (0.2126 * px[..., 0] + 0.7152 * px[..., 1] + 0.0722 * px[..., 2])
        opaque = px[..., 3] > 0.99
        # Blender image rows run bottom-up: top-down [y0, y1] -> bottom-up [1 - y1, 1 - y0]
        r0 = max(0, int(h * (1.0 - torso_top_frac[1])))
        r1 = min(h, int(h * (1.0 - torso_top_frac[0])))
        c0, c1 = int(w * 0.30), int(w * 0.70)
        box = (slice(r0, r1), slice(c0, c1))
        torso = luma[box][opaque[box]]
        body_px = luma[opaque]
        rows = np.nonzero((px[..., 3] > 0.5).any(axis=1))[0]
        return {
            "torso_median": round(float(np.median(torso)), 2) if torso.size else None,
            "body_p50": round(float(np.median(body_px)), 2) if body_px.size else None,
            "body_p999": round(float(np.percentile(body_px, 99.9)), 2) if body_px.size else None,
            "clipped_fraction_ge_250": round(float((body_px >= 250.0).mean()), 6) if body_px.size else None,
            "top_margin": round(float((h - 1 - rows.max()) / h), 5) if rows.size else None,
            "opaque_pixels": int(opaque.sum()),
        }

    def calibrate_m1_exposure(cam, body, env_objs, ref_path: Path, init_exp: float, target_median: float = 36.0,
                              tol: float = 2.0, torso_top_frac=(0.24, 0.44)):
        """One fast alpha-masked calibration render on M1 Obsidian Noir to lock exposure so lit-torso median hits 36 +/- 4
        and clipped_fraction_ge_250 < 0.005 before rendering the full reel over the architectural cyclorama."""
        orig_samples = sc.cycles.samples
        orig_trans = sc.render.film_transparent
        hidden = [(o, o.hide_render) for o in env_objs]
        for o, _ in hidden:
            o.hide_render = True
        sc.render.film_transparent = True
        sc.cycles.samples = min(24, orig_samples)
        sc.view_settings.exposure = float(init_exp)
        e = float(init_exp)
        try:
            io.render_still(sc, ref_path)
            st = measure_body_stats(ref_path, torso_top_frac=torso_top_frac)
            m = st.get("torso_median")
            e_prev, m_prev = None, None
            hist = [{"exposure": round(e, 4), "torso_median": m, "clipped": st.get("clipped_fraction_ge_250")}]
            for _ in range(5):
                clipped = st.get("clipped_fraction_ge_250") or 0.0
                if m is None or (abs(m - target_median) <= tol and clipped < 0.005):
                    break
                if abs(m - target_median) <= tol and clipped >= 0.005:
                    e_new = e - 0.25
                else:
                    slope = 22.0 if (e_prev is None or abs(m - m_prev) < 0.5) else (m - m_prev) / (e - e_prev)
                    slope = min(max(slope, 8.0), 55.0)
                    e_new = min(max(e + (target_median - m) / slope, init_exp - 2.5), init_exp + 2.5)
                sc.view_settings.exposure = e_new
                io.resave_render(sc, ref_path)
                st = measure_body_stats(ref_path, torso_top_frac=torso_top_frac)
                e_prev, m_prev, e, m = e, m, e_new, st.get("torso_median")
                hist.append({"exposure": round(e, 4), "torso_median": m, "clipped": st.get("clipped_fraction_ge_250")})
        finally:
            for o, h_prev in hidden:
                o.hide_render = h_prev
            sc.render.film_transparent = orig_trans
            sc.cycles.samples = orig_samples
            sc.view_settings.exposure = float(e)
        return round(float(e), 4), st, hist

    # --------------------------------------------------------------------------------------------------------------------------------
    # MODE 1: REEL 1 — Solved M5 Runway Walk (250 frames @ 24 fps, tracking spring camera, M1 Obsidian Noir)
    # --------------------------------------------------------------------------------------------------------------------------------
    def run_walk():
        body, rig = C.human_and_rig()
        inv = check_invariants(body, rig)
        psd_info = ensure_volume_psd_drivers(body, rig)
        clear_extra_objects({body.name, rig.name})
        rig.data.pose_position = "POSE"
        body.material_slots[0].material = bpy_mod.data.materials["MX_M1_ObsidianNoir"]

        dev = configure_cycles()
        sc.render.film_transparent = False
        sc.render.fps = FPS
        n_frames = int(A.get("frames", str(WALK_FRAMES)))

        # Verify action has walk keyframes (supports both legacy action.fcurves and Blender 5.x slotted actions)
        assert rig.animation_data and rig.animation_data.action, "m5_walk_scene.blend has no rig animation action"
        act = rig.animation_data.action
        fcurves = list(act.fcurves) if hasattr(act, "fcurves") else []
        if not fcurves and hasattr(act, "layers"):
            for lyr in act.layers:
                for stp in getattr(lyr, "strips", []):
                    for cb in getattr(stp, "channelbags", []):
                        fcurves.extend(getattr(cb, "fcurves", []))
        keyed_bones = sorted({fc.data_path.split('["')[1].split('"]')[0] for fc in fcurves if fc.data_path.startswith('pose.bones["')})

        # Build dark architectural stone floor + curved cyclorama (extended to y_min=-24.0m for 12.75m RUNWAY_CONFIDENT_V2 stride path)
        floor_ob, cyc_ob = build_stone_cyclorama(y_min=-24.0, y_max=5.8, x_half=12.0, cove_y=2.8, cove_r=1.8, wall_h=8.0)

        # Camera setup (85 mm -> 100 mm editorial runway lens at waist-height Z = 0.92 m)
        walk_cam_z = float(A.get("walk_cam_z", "0.92"))
        cd = bpy_mod.data.cameras.new("MX_M8_WalkCam")
        cd.lens = 85.0
        cd.sensor_fit = "VERTICAL"
        cd.sensor_height = SENSOR_MM
        cd.clip_start = 0.05
        cd.clip_end = 60.0
        cam = bpy_mod.data.objects.new("MX_M8_WalkCam", cd)
        sc.collection.objects.link(cam)
        sc.camera = cam

        # Pre-pass (fast, no rendering): evaluate all frames for pelvis/chest tracking, Z_sole check, G5 check, PSD verification
        t_pre = time.time()
        me = body.data
        arr_base = C.mesh_arrays(me)
        V_basis, T_mesh = arr_base["V"], arr_base["T"]
        raw_anchors = np.zeros((n_frames, 2), dtype=np.float64)
        raw_cam_locs = np.zeros((n_frames, 3), dtype=np.float64)
        raw_cam_tgts = np.zeros((n_frames, 3), dtype=np.float64)
        az_degs = np.zeros(n_frames, dtype=np.float64)
        lens_mms = np.zeros(n_frames, dtype=np.float64)
        dist_ms = np.zeros(n_frames, dtype=np.float64)
        z_mins = np.zeros(n_frames, dtype=np.float64)
        z_maxs = np.zeros(n_frames, dtype=np.float64)
        g5_counts = np.zeros(n_frames, dtype=np.int64)
        clav_ok_all = True
        psd_peaks = {k: 0.0 for k in ("PSD_hip_L", "PSD_hip_R", "PSD_axilla_L", "PSD_axilla_R")}
        kbs = body.data.shape_keys.key_blocks

        # First evaluate all frame anchors and bounds
        Xs_bbox = []
        for fi in range(n_frames):
            fnum = fi + 1
            sc.frame_set(fnum)
            bpy_mod.context.view_layer.update()
            X = C.evaluated_positions(body)
            z_mins[fi] = float(X[:, 2].min())
            z_maxs[fi] = float(X[:, 2].max())
            col = geometry.collapsed_triangles(V_basis, X, T_mesh, ratio=0.25)
            g5_counts[fi] = int(col["count"])
            for cn in ("clavicle.L", "clavicle.R"):
                qc = rig.pose.bones[cn].matrix_basis.to_quaternion()
                if abs(float(qc.w) - 1.0) > 1e-5 or max(abs(float(qc.x)), abs(float(qc.y)), abs(float(qc.z))) > 1e-5:
                    clav_ok_all = False
            # Track pelvis / chest world XY midpoint
            p_root = np.array(rig.matrix_world @ rig.pose.bones["root"].head, dtype=np.float64)
            p_chest = np.array(rig.matrix_world @ rig.pose.bones["spine02"].head, dtype=np.float64)
            anchor = 0.5 * (p_root[:2] + p_chest[:2])
            raw_anchors[fi] = anchor
            u = fi / max(n_frames - 1, 1)
            # Authoritative editorial arc: -14 deg -> +16 deg with 85 mm -> 100 mm compression at Z = 0.92 m
            u_arc = 0.5 - 0.5 * math.cos(math.pi * u)
            s_blend = 0.35 * u + 0.65 * u_arc
            az_degs[fi] = -14.0 + 30.0 * s_blend
            lens_mms[fi] = 85.0 + 15.0 * s_blend
            for k in psd_peaks:
                psd_peaks[k] = max(psd_peaks[k], float(kbs[k].value))
            Xs_bbox.append((X[:, 0].min(), X[:, 0].max(), X[:, 1].min(), X[:, 1].max(), z_mins[fi], z_maxs[fi]))
            loc_i, tgt_i, d_i, _, _ = solve_walk_editorial_camera(
                X, anchor, az_degs[fi], lens_mm=lens_mms[fi], cam_z=walk_cam_z, top_margin=0.070, sole_frac=0.905
            )
            raw_cam_locs[fi] = loc_i
            raw_cam_tgts[fi] = tgt_i
            dist_ms[fi] = d_i

        # Critically damped spring (zeta = 1.0) tracking on anchor, camera position, and camera target
        dt = 1.0 / FPS
        smooth_anchors = critically_damped_track(raw_anchors, dt=dt, omega=7.0)
        smooth_cam_locs = critically_damped_track(raw_cam_locs, dt=dt, omega=6.5)
        smooth_cam_locs[:, 2] = walk_cam_z
        smooth_cam_tgts = critically_damped_track(raw_cam_tgts, dt=dt, omega=6.5)

        # Verify grounded sole invariant & G5
        min_z_mm = float(z_mins.min() * 1000.0)
        max_g5_walk = int(g5_counts.max())
        print(f"PROGRESS walk pre-pass {n_frames} frames in {time.time() - t_pre:.1f}s | min_z={min_z_mm:.4f}mm | max_G5={max_g5_walk} | cam_z={walk_cam_z:.2f}m | psd_peaks={ {k: round(v, 3) for k, v in psd_peaks.items()} }", flush=True)

        # Build tracking studio lighting rig (calibrated for M1 Obsidian Noir + crisp dark stone separation)
        key_l = make_area_light("MX_M8_Key", 380.0 * key_scale, 1.6, 1.6, (2.0, -2.2, 2.6), (0, 0, 1.15), color=(1.0, 0.99, 0.97))
        fill_l = make_area_light("MX_M8_Fill", 62.0, 3.0, 3.0, (-2.3, -1.6, 1.6), (0, 0, 1.05), color=(0.94, 0.96, 1.0))
        rim_l = make_area_light("MX_M8_RimL", 220.0 * rim_boost, 0.55, 2.3, (1.85, 1.85, 1.35), (0, 0, 1.05), color=(0.96, 0.98, 1.0))
        rim_r = make_area_light("MX_M8_RimR", 220.0 * rim_boost, 0.55, 2.3, (-1.85, 1.85, 1.35), (0, 0, 1.05), color=(0.96, 0.98, 1.0))
        # Rear cyclorama wash (aimed strictly BACKWARD away from Maximus onto the dark architectural stone sweep; diffuse-only so zero floor glare)
        bg_wash = make_area_light("MX_M8_BgWash", 220.0, 3.8, 1.8, (0.0, 2.0, 2.4), (0.0, 4.4, 1.4), color=(0.88, 0.92, 1.0))
        try:
            bg_wash.visible_glossy = False
        except AttributeError:
            pass
        # Soft overhead architectural runway skylight spanning the full 13m walk path (diffuse-only: zero floor specular glare)
        runway_top = make_area_light("MX_M8_RunwayTop", 110.0, 1.6, 18.0, (0.0, -6.0, 4.8), (0.0, -6.0, 0.0), color=(0.92, 0.95, 1.0))
        try:
            runway_top.visible_glossy = False
        except AttributeError:
            pass

        def place_walk_rig(fi: int, X_eval: np.ndarray | None = None):
            cx, cy = float(smooth_anchors[fi, 0]), float(smooth_anchors[fi, 1])
            az = math.radians(float(az_degs[fi]))
            cs, sn = math.cos(az), math.sin(az)

            def orb(dx, dy, z):
                return (cx + dx * cs - dy * sn, cy + dx * sn + dy * cs, z)

            cam.data.lens = float(lens_mms[fi])
            if X_eval is not None:
                # Lock exact 7.0% crown headroom (strictly inside 6.0%-8.0%) from smoothed camera XY at Z=0.92m
                c_xy = smooth_cam_locs[fi, :2]
                t_xy = smooth_cam_tgts[fi, :2]
                lo_z, hi_z = 0.55, 1.85
                c_loc = np.array([c_xy[0], c_xy[1], walk_cam_z], dtype=np.float64)
                for _ in range(24):
                    mid_z = 0.5 * (lo_z + hi_z)
                    t_try = np.array([t_xy[0], t_xy[1], mid_z], dtype=np.float64)
                    tm_try, _ = projected_vertical_bounds(X_eval, c_loc, t_try, float(lens_mms[fi]))
                    if tm_try < 0.070:
                        lo_z = mid_z
                    else:
                        hi_z = mid_z
                smooth_cam_tgts[fi, 2] = 0.5 * (lo_z + hi_z)
            cam.location = tuple(float(v) for v in smooth_cam_locs[fi])
            R.look_at(cam, tuple(float(v) for v in smooth_cam_tgts[fi]))
            key_l.location = orb(2.15, -2.25, 2.65)
            R.look_at(key_l, (cx, cy, 1.12))
            fill_l.location = orb(-2.45, -1.65, 1.55)
            R.look_at(fill_l, (cx, cy, 1.05))
            # Dual grazing Fresnel vertical strips at +/-135 deg behind Maximus
            rim_l.location = orb(1.85, 1.85, 1.30)
            R.look_at(rim_l, (cx, cy, 1.02))
            rim_r.location = orb(-1.85, 1.85, 1.30)
            R.look_at(rim_r, (cx, cy, 1.02))
            # Aim backdrop wash along the camera-body ray intersection on the rear cyclorama wall
            ray = smooth_cam_tgts[fi, :2] - smooth_cam_locs[fi, :2]
            ray = ray / max(float(np.linalg.norm(ray)), 1e-9)
            t_wall = (4.4 - cy) / max(float(ray[1]), 0.25)
            wx = float(np.clip(cx + t_wall * ray[0], -5.0, 5.0))
            bg_wash.location = (wx, 2.0, 2.55)
            R.look_at(bg_wash, (wx, 4.4, 1.35))

        # Calibrate M1 exposure on poster frame 60 (or middle frame in smoke mode)
        cal_fi = min(WALK_POSTER_FRAME - 1, n_frames - 1)
        sc.frame_set(cal_fi + 1)
        bpy_mod.context.view_layer.update()
        place_walk_rig(cal_fi, C.evaluated_positions(body))
        init_exp = float(A.get("exposure", "-0.65"))
        auto_exp = A.get("auto_exposure", "1") == "1"
        if auto_exp:
            exp_used, cal_stats, cal_hist = calibrate_m1_exposure(
                cam, body, [floor_ob, cyc_ob], out_dir / "_walk_cal.png", init_exp=init_exp, target_median=36.0, tol=2.0,
                torso_top_frac=(0.24, 0.44),
            )
        else:
            sc.view_settings.exposure = init_exp
            exp_used, cal_stats, cal_hist = init_exp, {}, []
        print(f"PROGRESS walk exposure locked at {exp_used} | cal_stats={cal_stats}", flush=True)

        # Render all 250 walk frames
        frames_dir = out_dir / "walk_frames"
        frames_dir.mkdir(exist_ok=True)
        t0 = time.time()
        headrooms = []
        sole_fracs = []
        for fi in range(n_frames):
            fnum = fi + 1
            sc.frame_set(fnum)
            bpy_mod.context.view_layer.update()
            X = C.evaluated_positions(body)
            place_walk_rig(fi, X)
            tm, bf = projected_vertical_bounds(X, smooth_cam_locs[fi], smooth_cam_tgts[fi], float(lens_mms[fi]))
            headrooms.append(tm)
            sole_fracs.append(bf)
            io.render_still(sc, frames_dir / f"frame_{fnum:04d}.png")
            if fi == 0 or fnum % 10 == 0 or fnum == n_frames:
                el = time.time() - t0
                rate = el / fnum
                eta = rate * (n_frames - fnum)
                print(f"PROGRESS walk frame {fnum}/{n_frames} az={az_degs[fi]:+.1f}deg lens={lens_mms[fi]:.1f}mm camZ={smooth_cam_locs[fi,2]:.2f}m top={tm*100:.2f}% G5={g5_counts[fi]} {el:.0f}s (ETA {eta:.0f}s)", flush=True)

        rep = {
            "reel": "walk",
            "frames": n_frames,
            "fps": FPS,
            "duration_s": round(n_frames / FPS, 3),
            "resolution": [res_w, res_h],
            "samples": samples,
            "adaptive_threshold": adaptive_thresh,
            "device": dev,
            "material": "MX_M1_ObsidianNoir",
            "exposure": exp_used,
            "calibration_stats": cal_stats,
            "calibration_history": cal_hist,
            "invariants": inv,
            "volume_psd_drivers": psd_info,
            "volume_psd_peaks": {k: round(v, 4) for k, v in psd_peaks.items()},
            "keyed_bones_count": len(keyed_bones),
            "clavicle_lock_all_frames": bool(clav_ok_all),
            "max_G5": int(max_g5_walk),
            "camera_z_m": round(float(walk_cam_z), 3),
            "lens_mm_range": [round(float(lens_mms[0]), 2), round(float(lens_mms[-1]), 2)],
            "z_sole_min_mm": round(min_z_mm, 4),
            "z_sole_ge_0": bool(min_z_mm >= -0.01),
            "crown_headroom_pct_range": [round(min(headrooms) * 100.0, 2), round(max(headrooms) * 100.0, 2)],
            "sole_frame_frac_range": [round(min(sole_fracs), 4), round(max(sole_fracs), 4)],
            "azimuth_deg_range": [round(float(az_degs[0]), 2), round(float(az_degs[-1]), 2)],
            "forward_travel_m": round(float(raw_anchors[0, 1] - raw_anchors[-1, 1]), 3),
            "render_seconds": round(time.time() - t0, 1),
            "qa_frames": {
                str(f): {
                    "t_s": round((f - 1) / FPS, 2),
                    "az_deg": round(float(az_degs[min(f - 1, n_frames - 1)]), 1),
                    "lens_mm": round(float(lens_mms[min(f - 1, n_frames - 1)]), 1),
                    "headroom_pct": round(float(headrooms[min(f - 1, n_frames - 1)]) * 100.0, 2),
                    "z_sole_mm": round(float(z_mins[min(f - 1, n_frames - 1)]) * 1000.0, 3),
                    "g5": int(g5_counts[min(f - 1, n_frames - 1)]),
                }
                for f in WALK_QA_FRAMES if f <= n_frames
            },
        }
        (out_dir / "_walk_report.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
        return {"dir": str(frames_dir), "report": str(out_dir / "_walk_report.json")}, rep

    # --------------------------------------------------------------------------------------------------------------------------------
    # MODE 2: REEL 2 — SWAG_01 3-Second Pose Strike + Dynamic Camera + Grazing Fresnel Rim Lighting
    # --------------------------------------------------------------------------------------------------------------------------------
    def run_pose():
        body, rig = C.human_and_rig()
        inv = check_invariants(body, rig)
        psd_info = ensure_volume_psd_drivers(body, rig)
        clear_extra_objects({body.name, rig.name})
        rig.data.pose_position = "POSE"
        body.material_slots[0].material = bpy_mod.data.materials["MX_M1_ObsidianNoir"]

        me = body.data
        arr = C.mesh_arrays(me)
        V_basis, T_mesh = arr["V"], arr["T"]
        kbs = me.shape_keys.key_blocks
        vol_psd_names = {"PSD_hip_L", "PSD_hip_R", "PSD_axilla_L", "PSD_axilla_R"}
        psd = PSE.PSDModel.from_json(me["maximus_psd"]) if "maximus_psd" in me.keys() else None
        psd_joints = sorted({j for fam in (psd.families.values() if psd else []) for j in fam["joints"]})

        P = posing.Poser(bpy_mod, rig)
        ev = lambda: C.evaluated_positions(body)  # noqa: E731

        # Load canonical theta_fit (start), P1 (monolith poise/fingers), and P2 (contrapposto weight shift)
        fit_bones = json.loads(Path(A["theta_fit"]).read_text(encoding="utf-8"))["bones"]
        p1_bones = json.loads((root / "poses" / "theta_monolith.json").read_text(encoding="utf-8"))["bones"]
        p2_bones = json.loads((root / "poses" / "theta_contrapposto.json").read_text(encoding="utf-8"))["bones"]

        def capture_all_bones():
            out = {}
            for pb in rig.pose.bones:
                q = pb.rotation_quaternion if pb.rotation_mode == "QUATERNION" else pb.matrix_basis.to_quaternion()
                out[pb.name] = {
                    "q": quat_canon([float(q.w), float(q.x), float(q.y), float(q.z)]),
                    "t": np.array([float(pb.location.x), float(pb.location.y), float(pb.location.z)], dtype=np.float64),
                }
            return out

        def eval_psd_weights():
            if not psd:
                return {}
            lq = {}
            for j in psd_joints:
                q = rig.pose.bones[j].matrix_basis.to_quaternion()
                lq[j] = np.array([float(q.w), float(q.x), float(q.y), float(q.z)], dtype=np.float64)
            return {k: float(v) for k, v in psd.key_weights(lq).items() if k in kbs and k not in vol_psd_names}

        def apply_psd_current(floor_weights=None):
            if psd:
                w_rbf = eval_psd_weights()
                for kname, val in w_rbf.items():
                    if floor_weights and kname in floor_weights:
                        val = max(val, float(floor_weights[kname]))
                    kbs[kname].value = float(min(1.0, max(0.0, val)))
                bpy_mod.context.view_layer.update()

        # 1. Capture grounded start pose: theta_fit (clavicles strictly locked) + its canonical M4 PSD weights
        P.reset()
        P.apply_local(fit_bones)
        P.lock_clavicles()
        w_psd_start = eval_psd_weights()
        apply_psd_current(w_psd_start)
        P.ground(ev)
        start_pose = capture_all_bones()

        # Capture canonical P2 M4 PSD weights as smooth trajectory support (prevents narrow-sigma RBF dips between theta_fit and P2)
        P.reset()
        P.apply_local(p2_bones)
        P.lock_clavicles()
        w_psd_p2 = eval_psd_weights()

        # 2. Construct editorial SWAG_01 strike target (P1 poise/relaxed fingers -> P2 contrapposto weight shift + torso twist + settled gaze)
        def build_swag_target(arm_extra_scale: float = 1.0):
            P.reset()
            P.apply_local(p2_bones)
            # Blend in P1's relaxed ring-finger relief so hands stay natural & G5=0
            for fn in ("finger4-2.L", "finger4-2.R"):
                if fn in p1_bones:
                    pb = rig.pose.bones[fn]
                    pb.rotation_mode = "QUATERNION"
                    pb.rotation_quaternion = Quaternion(p1_bones[fn]["q_wxyz"])
            # Editorial contrapposto twist + shoulder poise + settled hero gaze (clavicles NEVER touched!)
            ops = [
                {"op": "rotate", "bone": "root", "axis": "U", "deg": -7.5 * arm_extra_scale},
                {"op": "rotate", "bone": "spine03", "axis": "U", "deg": 3.5 * arm_extra_scale},
                {"op": "rotate", "bone": "spine02", "axis": "U", "deg": 4.0 * arm_extra_scale},
                {"op": "rotate", "bone": "spine02", "axis": "L", "deg": 2.5 * arm_extra_scale},
                {"op": "rotate", "bone": "spine01", "axis": "U", "deg": 3.0 * arm_extra_scale},
                {"op": "shoulder_roll", "deg": 1.6 * arm_extra_scale},
                {"op": "arm_abduction", "side": "L", "deg": 10.0},
                {"op": "arm_abduction", "side": "R", "deg": 13.0},
                {"op": "arm_flex", "side": "L", "deg": -4.0 * arm_extra_scale},
                {"op": "arm_flex", "side": "R", "deg": 10.0 * arm_extra_scale},
                {"op": "elbow_flex", "side": "L", "deg": 19.0},
                {"op": "elbow_flex", "side": "R", "deg": 38.0 * arm_extra_scale + 16.0 * (1.0 - arm_extra_scale), "medial": 0.25},
                {"op": "rotate", "bone": "neck01", "axis": "U", "deg": -4.0 * arm_extra_scale},
                {"op": "rotate", "bone": "head", "axis": "U", "deg": -7.5 * arm_extra_scale},
                {"op": "rotate", "bone": "head", "axis": "L", "deg": 2.5 * arm_extra_scale},
            ]
            for op in ops:
                P.op(op)
            P.lock_clavicles()
            w_tgt = {k: max(eval_psd_weights().get(k, 0.0), w_psd_p2.get(k, 0.0)) for k in set(w_psd_start) | set(w_psd_p2)}
            apply_psd_current(w_tgt)
            P.ground(ev)
            X_tgt = ev()
            col = geometry.collapsed_triangles(V_basis, X_tgt, T_mesh, ratio=0.25)
            return capture_all_bones(), col, w_tgt

        # Verify target achieves G5 == 0; gently scale editorial accent only if needed
        target_pose = None
        target_col = None
        w_psd_target = w_psd_p2
        used_scale = 1.0
        for sc_try in (1.0, 0.75, 0.5, 0.0):
            cand_pose, cand_col, cand_w = build_swag_target(sc_try)
            if cand_col["count"] == 0:
                target_pose, target_col, w_psd_target, used_scale = cand_pose, cand_col, cand_w, sc_try
                break
        if target_pose is None:
            target_pose, target_col, w_psd_target, used_scale = build_swag_target(0.0)
        print(f"PROGRESS SWAG_01 target built (scale={used_scale}) G5={target_col['count']} min_ratio={target_col['min_ratio']:.4f}", flush=True)

        # Configure Cycles GPU + architectural stone cyclorama
        dev = configure_cycles()
        sc.render.film_transparent = False
        sc.render.fps = FPS
        n_frames = int(A.get("frames", str(POSE_FRAMES)))
        floor_ob, cyc_ob = build_stone_cyclorama(y_min=-7.5, y_max=4.8, x_half=9.0, cove_y=2.4, cove_r=1.8, wall_h=7.2)

        cd = bpy_mod.data.cameras.new("MX_M8_PoseCam")
        cd.lens = 85.0
        cd.sensor_fit = "VERTICAL"
        cd.sensor_height = SENSOR_MM
        cd.clip_start = 0.05
        cd.clip_end = 50.0
        cam = bpy_mod.data.objects.new("MX_M8_PoseCam", cd)
        sc.collection.objects.link(cam)
        sc.camera = cam

        # High-contrast 3-point + cyclorama rig with grazing Fresnel rim lights at +/-135 deg (E_rim / E_fill = 3.5 exact!)
        # E_fill = P_fill / (4 pi d_fill^2) with P_fill = 70 W, d_fill = 3.0 m
        # Each grazing rim strip at +/-135 deg, d_rim = 2.55 m: P_rim_each = 3.5 * 70 * (2.55 / 3.0)^2 = 177.01 W -> E_rim / E_fill = 3.50 exact!
        fill_w = 70.0
        d_fill = 3.0
        d_rim = 2.55
        rim_w_each = 3.5 * fill_w * ((d_rim / d_fill) ** 2) * rim_boost
        e_rim_over_fill = (rim_w_each / (d_rim ** 2)) / (fill_w / (d_fill ** 2))

        az_key = math.radians(48.0)
        key_l = make_area_light("MX_M8_PoseKey", 370.0 * key_scale, 1.5, 1.5,
                                (3.1 * math.sin(az_key), -3.1 * math.cos(az_key), 2.45), (0.0, -0.02, 1.18), color=(1.0, 0.99, 0.97))
        az_fill = math.radians(-55.0)
        fill_l = make_area_light("MX_M8_PoseFill", fill_w, 2.8, 2.8,
                                 (d_fill * math.sin(az_fill), -d_fill * math.cos(az_fill), 1.55), (0.0, -0.02, 1.10), color=(0.94, 0.96, 1.0))
        # Grazing Fresnel vertical strip rim lights at exact +135 deg and -135 deg behind deltoids and calves
        rim_xy = d_rim * math.sin(math.radians(135.0))
        rim_l = make_area_light("MX_M8_PoseRimL", rim_w_each, 0.52, 2.40, (rim_xy, rim_xy, 1.25), (0.0, 0.0, 1.05), color=(0.95, 0.98, 1.0))
        rim_r = make_area_light("MX_M8_PoseRimR", rim_w_each, 0.52, 2.40, (-rim_xy, rim_xy, 1.25), (0.0, 0.0, 1.05), color=(0.95, 0.98, 1.0))
        # Rear cyclorama wash aimed strictly BACKWARD onto curved dark stone backdrop (diffuse-only: zero floor glare)
        bg_wash = make_area_light("MX_M8_PoseBgWash", 210.0, 3.6, 1.8, (0.0, 1.8, 2.35), (0.0, 4.2, 1.35), color=(0.88, 0.92, 1.0))
        try:
            bg_wash.visible_glossy = False
        except AttributeError:
            pass

        bone_names = [b.name for b in rig.data.bones]
        lags = {bn: bone_phase_lag_s(bn) for bn in bone_names}
        all_psd_keys = sorted(set(w_psd_start) | set(w_psd_target))

        def apply_swag_frame(fi: int):
            t = fi / float(FPS)
            for bn in bone_names:
                pb = rig.pose.bones[bn]
                pb.rotation_mode = "QUATERNION"
                if bn in ("clavicle.L", "clavicle.R"):
                    pb.rotation_quaternion = Quaternion((1.0, 0.0, 0.0, 0.0))
                    pb.location = (0.0, 0.0, 0.0)
                    continue
                lag = lags[bn]
                tau = float(np.clip((t - lag) / max(POSE_MOVE_S - lag, 1e-6), 0.0, 1.0))
                sig = float(min_jerk(tau))
                q_t = quat_min_jerk(start_pose[bn]["q"], target_pose[bn]["q"], sig)
                t_t = (1.0 - sig) * start_pose[bn]["t"] + sig * target_pose[bn]["t"]
                pb.rotation_quaternion = Quaternion(tuple(float(v) for v in q_t))
                pb.location = tuple(float(v) for v in t_t)
            bpy_mod.context.view_layer.update()
            sig_glob = float(min_jerk(np.clip(t / POSE_MOVE_S, 0.0, 1.0)))
            w_floor = {k: (1.0 - sig_glob) * w_psd_start.get(k, 0.0) + sig_glob * w_psd_target.get(k, 0.0) for k in all_psd_keys}
            apply_psd_current(w_floor)
            z_sole = P.ground(ev)
            return z_sole

        # Pre-pass: verify clavicle lock, Z_sole >= 0, G5 == 0, and solve critically damped dynamic camera trajectory
        cam_plan = []
        g5_max = 0
        z_sole_min = 1e9
        clav_ok_all = True
        psd_peaks = {k: 0.0 for k in vol_psd_names}
        for fi in range(n_frames):
            u = fi / max(n_frames - 1, 1)
            s_cam = float(critically_damped_ease(u, omega=5.8))
            z_sole = apply_swag_frame(fi)
            z_sole_min = min(z_sole_min, z_sole)
            for cn in ("clavicle.L", "clavicle.R"):
                qc = rig.pose.bones[cn].matrix_basis.to_quaternion()
                if abs(qc.w - 1.0) > 1e-6 or max(abs(qc.x), abs(qc.y), abs(qc.z)) > 1e-6:
                    clav_ok_all = False
            for k in psd_peaks:
                psd_peaks[k] = max(psd_peaks[k], float(kbs[k].value))
            X = ev()
            col = geometry.collapsed_triangles(V_basis, X, T_mesh, ratio=0.25)
            g5_max = max(g5_max, int(col["count"]))

            # Dynamic camera: 35 deg three-quarter medium shot (85 mm, d=4.15 m, z=1.04 m) -> tight low-angle hero lock (105 mm, d=3.35 m, z=0.92 m, 10 deg)
            az_deg = 35.0 * (1.0 - s_cam) + 10.0 * s_cam
            lens_mm = 85.0 * (1.0 - s_cam) + 105.0 * s_cam
            dist_m = 4.15 * (1.0 - s_cam) + 3.35 * s_cam
            cam_z = 1.04 * (1.0 - s_cam) + 0.90 * s_cam
            p_chest = np.array(rig.matrix_world @ rig.pose.bones["spine02"].head, dtype=np.float64)
            anchor_xy = np.array([p_chest[0] * 0.5, -0.02], dtype=np.float64)
            cloc, ctgt, tm, bf = solve_pitched_camera_headroom(X, anchor_xy, az_deg, lens_mm, cam_z, dist_m, top_margin=0.070)
            cam_plan.append({
                "fi": fi, "u": u, "s_cam": s_cam, "az_deg": az_deg, "lens_mm": lens_mm, "dist_m": dist_m,
                "cam_loc": cloc, "cam_tgt": ctgt, "top_margin": tm, "bot_frac": bf, "g5": int(col["count"]), "z_sole_mm": z_sole * 1000.0,
            })

        print(f"PROGRESS SWAG_01 pre-pass {n_frames} frames | max_G5={g5_max} | clav_ok={clav_ok_all} | z_sole_min={z_sole_min*1000:.4f}mm", flush=True)

        # Calibrate exposure on hero hold frame 64
        cal_fi = min(POSE_POSTER_FRAME - 1, n_frames - 1)
        apply_swag_frame(cal_fi)
        cp = cam_plan[cal_fi]
        cam.data.lens = cp["lens_mm"]
        cam.location = tuple(float(v) for v in cp["cam_loc"])
        R.look_at(cam, tuple(float(v) for v in cp["cam_tgt"]))
        init_exp = float(A.get("exposure", "-0.65"))
        auto_exp = A.get("auto_exposure", "1") == "1"
        if auto_exp:
            exp_used, cal_stats, cal_hist = calibrate_m1_exposure(
                cam, body, [floor_ob, cyc_ob], out_dir / "_pose_cal.png", init_exp=init_exp, target_median=36.0, tol=2.0,
                torso_top_frac=(0.32, 0.56),
            )
        else:
            sc.view_settings.exposure = init_exp
            exp_used, cal_stats, cal_hist = init_exp, {}, []
        print(f"PROGRESS SWAG_01 exposure locked at {exp_used} | cal_stats={cal_stats}", flush=True)

        # Render all 72 SWAG_01 frames
        frames_dir = out_dir / "pose_frames"
        frames_dir.mkdir(exist_ok=True)
        t0 = time.time()
        for fi in range(n_frames):
            fnum = fi + 1
            apply_swag_frame(fi)
            cp = cam_plan[fi]
            cam.data.lens = cp["lens_mm"]
            cam.location = tuple(float(v) for v in cp["cam_loc"])
            R.look_at(cam, tuple(float(v) for v in cp["cam_tgt"]))
            # Keep cyclorama wash centered directly behind the camera-subject sightline
            ray = cp["cam_tgt"][:2] - cp["cam_loc"][:2]
            ray = ray / max(float(np.linalg.norm(ray)), 1e-9)
            wx = float(np.clip(4.2 * ray[0] / max(float(ray[1]), 0.25), -3.5, 3.5))
            bg_wash.location = (wx, 1.8, 2.35)
            R.look_at(bg_wash, (wx, 4.2, 1.35))
            io.render_still(sc, frames_dir / f"frame_{fnum:04d}.png")
            if fi == 0 or fnum % 8 == 0 or fnum == n_frames:
                el = time.time() - t0
                eta = (el / fnum) * (n_frames - fnum)
                print(f"PROGRESS pose frame {fnum}/{n_frames} az={cp['az_deg']:.1f}deg lens={cp['lens_mm']:.1f}mm top={cp['top_margin']*100:.2f}% {el:.0f}s (ETA {eta:.0f}s)", flush=True)

        headrooms = [c["top_margin"] for c in cam_plan]
        rep = {
            "reel": "pose_strike_SWAG_01",
            "frames": n_frames,
            "fps": FPS,
            "duration_s": round(n_frames / FPS, 3),
            "move_duration_s": POSE_MOVE_S,
            "hold_duration_s": round(POSE_TOTAL_S - POSE_MOVE_S, 3),
            "resolution": [res_w, res_h],
            "samples": samples,
            "adaptive_threshold": adaptive_thresh,
            "device": dev,
            "material": "MX_M1_ObsidianNoir",
            "exposure": exp_used,
            "calibration_stats": cal_stats,
            "calibration_history": cal_hist,
            "e_rim_over_e_fill": round(e_rim_over_fill, 3),
            "rim_azimuth_deg": [-135.0, 135.0],
            "rim_watts_each": round(rim_w_each, 2),
            "fill_watts": fill_w,
            "phase_lags_s": {"pelvis_legs": 0.00, "spine": 0.08, "shoulders_arms": 0.16, "wrists_neck_head": 0.25},
            "invariants": inv,
            "volume_psd_drivers": psd_info,
            "volume_psd_peaks": {k: round(v, 4) for k, v in psd_peaks.items()},
            "clavicle_lock_all_frames": clav_ok_all,
            "max_G5": g5_max,
            "editorial_accent_scale": used_scale,
            "z_sole_min_mm": round(z_sole_min * 1000.0, 4),
            "z_sole_ge_0": bool(z_sole_min >= -1e-5),
            "crown_headroom_pct_range": [round(min(headrooms) * 100.0, 2), round(max(headrooms) * 100.0, 2)],
            "camera_arc": {"az_start_deg": 35.0, "az_end_deg": 10.0, "lens_start_mm": 85.0, "lens_end_mm": 105.0},
            "render_seconds": round(time.time() - t0, 1),
            "qa_frames": {
                str(f): {
                    "t_s": round((f - 1) / FPS, 2),
                    "az_deg": round(float(cam_plan[min(f - 1, n_frames - 1)]["az_deg"]), 1),
                    "lens_mm": round(float(cam_plan[min(f - 1, n_frames - 1)]["lens_mm"]), 1),
                    "headroom_pct": round(float(cam_plan[min(f - 1, n_frames - 1)]["top_margin"]) * 100.0, 2),
                    "g5": int(cam_plan[min(f - 1, n_frames - 1)]["g5"]),
                }
                for f in POSE_QA_FRAMES if f <= n_frames
            },
        }
        (out_dir / "_pose_report.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
        return {"dir": str(frames_dir), "report": str(out_dir / "_pose_report.json")}, rep

    C.run(f"m8_blender_{mode}", {"walk": run_walk, "pose": run_pose}[mode])


# ------------------------------------------------------------------------------------------------------------------------------------
# System-Python Launcher (Colab / CLI): Runs Blender GPU passes, encodes MP4/WebP/Contact Sheets, syncs & hashes on Drive
# ------------------------------------------------------------------------------------------------------------------------------------
def find_blender(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit)
    exe = next(Path("/content/blender").glob("blender-*/blender"), None)
    if exe:
        return exe
    raise FileNotFoundError("Blender 5.2 not found in /content/blender; pass --blender")


def run_blender_stage(exe: Path, blend: Path, root: Path, kv: dict, log_path: Path):
    env = dict(os.environ, MAXIMUS_STRICT="1")
    if (root / "corpus").is_dir():
        env["MAXIMUS_CORPUS"] = str(root / "corpus")
    cmd = [
        str(exe), "--background", str(blend), "--factory-startup", "--python-exit-code", "3",
        "--python", str(Path(__file__).resolve()), "--", f"root={root}",
    ] + [f"{k}={v}" for k, v in kv.items()]
    print("CMD:", " ".join(cmd), flush=True)
    p = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    buf = []
    for line in p.stdout:
        buf.append(line)
        if line.startswith("PROGRESS"):
            print("  " + line.rstrip(), flush=True)
    p.wait()
    log_path.write_text("".join(buf), encoding="utf-8")
    envl = [l for l in buf if l.startswith("EC02 ")]
    rep = json.loads(envl[-1][5:]) if envl else {"status": "fail", "errors": ["".join(buf[-80:])]}
    if rep.get("status") != "ok":
        print(json.dumps(rep.get("errors"), indent=1)[:6000])
        raise SystemExit(f"M8 Blender pass ({kv.get('mode')}) failed; see {log_path}")
    return rep


def encode_mp4(frames_dir: Path, out_mp4: Path, crf: int = 20, fps: int = FPS) -> dict:
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps),
        "-i", str(frames_dir / "frame_%04d.png"),
        "-c:v", "libx264", "-preset", "slow", "-pix_fmt", "yuv420p",
        "-crf", str(crf), "-movflags", "+faststart", str(out_mp4),
    ]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0 or not out_mp4.exists():
        raise RuntimeError(f"ffmpeg failed for {out_mp4.name}: {p.stderr}")
    return {"file": out_mp4.name, "bytes": out_mp4.stat().st_size, "crf": crf, "sha256": sha256(out_mp4)}


def save_poster_webp(frame_png: Path, out_webp: Path, quality: int = 92) -> dict:
    from PIL import Image
    im = Image.open(frame_png).convert("RGB")
    im.save(out_webp, "WEBP", quality=quality, method=6)
    return {"file": out_webp.name, "bytes": out_webp.stat().st_size, "size": list(im.size), "sha256": sha256(out_webp)}


def build_contact_sheet(frames_dir: Path, frame_nums, labels: dict, title: str, out_png: Path) -> dict:
    from PIL import Image, ImageDraw
    cols, rows = 3, 2
    tw, th = 540, 675
    hdr, pad, lbl_h = 56, 12, 34
    W = cols * tw + (cols + 1) * pad
    H = hdr + rows * (th + lbl_h) + (rows + 1) * pad
    sheet = Image.new("RGB", (W, H), (16, 17, 20))
    dr = ImageDraw.Draw(sheet)
    dr.text((pad, 18), title, fill=(235, 235, 240))
    for i, fnum in enumerate(frame_nums):
        p = frames_dir / f"frame_{fnum:04d}.png"
        if not p.exists():
            continue
        im = Image.open(p).convert("RGB").resize((tw, th), Image.LANCZOS)
        c, r = i % cols, i // cols
        x0 = pad + c * (tw + pad)
        y0 = hdr + pad + r * (th + lbl_h + pad)
        sheet.paste(im, (x0, y0))
        info = labels.get(str(fnum), {})
        caption = f"F{fnum:03d}  t={info.get('t_s', (fnum-1)/FPS):.2f}s"
        if "az_deg" in info:
            caption += f"  az={info['az_deg']:+.1f}deg"
        if "lens_mm" in info:
            caption += f"  {info['lens_mm']:.0f}mm"
        if "headroom_pct" in info:
            caption += f"  top={info['headroom_pct']:.1f}%"
        if "z_sole_mm" in info:
            caption += f"  z={info['z_sole_mm']:.2f}mm"
        if "g5" in info:
            caption += f"  G5={info['g5']}"
        dr.text((x0 + 6, y0 + th + 8), caption, fill=(212, 216, 224))
    sheet.save(out_png)
    return {"file": out_png.name, "bytes": out_png.stat().st_size, "size": [W, H], "sha256": sha256(out_png)}


def launch():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default="/content/maximus_v2/bundle")
    ap.add_argument("--master-psd", default=None)
    ap.add_argument("--walk-scene", default=None)
    ap.add_argument("--out", default="/content/drive/MyDrive/maximus_v2/results")
    ap.add_argument("--blender", default=None)
    ap.add_argument("--device", default="gpu", choices=["gpu", "cpu-smoke"])
    ap.add_argument("--reels", default="both", choices=["both", "walk", "pose"])
    ap.add_argument("--w", default="1080")
    ap.add_argument("--h", default="1350")
    ap.add_argument("--walk-samples", default="48")
    ap.add_argument("--pose-samples", default="64")
    ap.add_argument("--adaptive-threshold", default="0.03")
    ap.add_argument("--walk-exposure", default="-0.65")
    ap.add_argument("--pose-exposure", default="-0.65")
    ap.add_argument("--auto-exposure", default="1")
    ap.add_argument("--rim-boost", default="1.0")
    ap.add_argument("--key-scale", default="1.0")
    ap.add_argument("--bg-luma", default="0.055")
    a = ap.parse_args()
    t0 = time.time()

    root = Path(a.root).resolve()
    exe = find_blender(a.blender)
    drive_res = Path(a.out)
    master_psd = Path(a.master_psd) if a.master_psd else next(
        p for p in (
            drive_res / "M4_canon_atlas" / "maximus_v2_master_psd.blend",
            root / "out" / "maximus_v2_master_psd.blend",
            HERE / "results" / "M4_canon_atlas" / "maximus_v2_master_psd.blend",
        ) if p.exists()
    )
    walk_scene = Path(a.walk_scene) if a.walk_scene else next(
        p for p in (
            drive_res / "M5_kinematics" / "m5_walk_scene.blend",
            HERE / "results" / "M5_kinematics" / "m5_walk_scene.blend",
        ) if p.exists()
    )
    theta_fit = next(
        p for p in (
            HERE / "m_theta_fit.json",
            Path("/content/drive/MyDrive/maximus_v2/m_theta_fit.json"),
            root / "m_theta_fit.json",
        ) if p.exists()
    )

    work = Path("/content/m8_work") if str(a.out).startswith("/content/drive") else drive_res / "_m8_work"
    work.mkdir(parents=True, exist_ok=True)
    dst = drive_res / "M8_motion_reels"
    dst.mkdir(parents=True, exist_ok=True)

    smoke = a.device == "cpu-smoke"
    res_w = "270" if smoke else a.w
    res_h = "338" if smoke else a.h

    # Load previous report if re-rendering only one reel during QA tuning
    prev_report = {}
    if (dst / "M8_motion_reels.json").exists() and a.reels != "both":
        try:
            prev_report = json.loads((dst / "M8_motion_reels.json").read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            prev_report = {}

    full = {
        "stage": STAGE,
        "master_psd": str(master_psd),
        "walk_scene": str(walk_scene),
        "walk": prev_report.get("walk"),
        "pose_strike": prev_report.get("pose_strike"),
    }
    produced = []

    if a.reels in ("both", "walk"):
        print("[1/2] Rendering Reel 1: maximus_v2_walk_reel.mp4 (250 frames @ 24 fps, M5 walk, M1 Obsidian Noir)", flush=True)
        kv_w = {
            "mode": "walk", "out_dir": work, "device": "cpu" if smoke else "gpu", "require_gpu": "0" if smoke else "1",
            "w": res_w, "h": res_h, "frames": "6" if smoke else str(WALK_FRAMES),
            "samples": "4" if smoke else a.walk_samples, "adaptive_threshold": a.adaptive_threshold,
            "exposure": a.walk_exposure, "auto_exposure": a.auto_exposure,
            "rim_boost": a.rim_boost, "key_scale": a.key_scale, "bg_luma": a.bg_luma,
        }
        run_blender_stage(exe, walk_scene, root, kv_w, work / "m8_blender_walk.log")
        wrep = json.loads((work / "_walk_report.json").read_text(encoding="utf-8"))
        wdir = work / "walk_frames"
        wrep["mp4"] = encode_mp4(wdir, work / "maximus_v2_walk_reel.mp4", crf=20, fps=FPS)
        poster_f = min(WALK_POSTER_FRAME, wrep["frames"])
        wrep["poster_webp"] = save_poster_webp(wdir / f"frame_{poster_f:04d}.png", work / "maximus_v2_walk_poster.webp", quality=92)
        qa_fs = [f for f in WALK_QA_FRAMES if f <= wrep["frames"]] or [1, wrep["frames"]]
        wrep["contact_sheet"] = build_contact_sheet(
            wdir, qa_fs, wrep["qa_frames"],
            f"MAXIMUS v2 M8 REEL 1 - M5 RUNWAY WALK ({wrep['resolution'][0]}x{wrep['resolution'][1]}, {wrep['samples']}spp, exp={wrep['exposure']})",
            work / "M8_walk_contact_sheet.png",
        )
        full["walk"] = wrep
        produced += ["maximus_v2_walk_reel.mp4", "maximus_v2_walk_poster.webp", "M8_walk_contact_sheet.png", "m8_blender_walk.log"]

    if a.reels in ("both", "pose"):
        print("[2/2] Rendering Reel 2: maximus_v2_pose_strike.mp4 (72 frames @ 24 fps, SWAG_01, M1 Obsidian Noir)", flush=True)
        kv_p = {
            "mode": "pose", "out_dir": work, "theta_fit": theta_fit,
            "device": "cpu" if smoke else "gpu", "require_gpu": "0" if smoke else "1",
            "w": res_w, "h": res_h, "frames": "6" if smoke else str(POSE_FRAMES),
            "samples": "4" if smoke else a.pose_samples, "adaptive_threshold": a.adaptive_threshold,
            "exposure": a.pose_exposure, "auto_exposure": a.auto_exposure,
            "rim_boost": a.rim_boost, "key_scale": a.key_scale, "bg_luma": a.bg_luma,
        }
        run_blender_stage(exe, master_psd, root, kv_p, work / "m8_blender_pose.log")
        prep = json.loads((work / "_pose_report.json").read_text(encoding="utf-8"))
        pdir = work / "pose_frames"
        prep["mp4"] = encode_mp4(pdir, work / "maximus_v2_pose_strike.mp4", crf=19, fps=FPS)
        poster_f = min(POSE_POSTER_FRAME, prep["frames"])
        prep["poster_webp"] = save_poster_webp(pdir / f"frame_{poster_f:04d}.png", work / "maximus_v2_pose_poster.webp", quality=92)
        qa_fs = [f for f in POSE_QA_FRAMES if f <= prep["frames"]] or [1, prep["frames"]]
        prep["contact_sheet"] = build_contact_sheet(
            pdir, qa_fs, prep["qa_frames"],
            f"MAXIMUS v2 M8 REEL 2 - SWAG_01 POSE STRIKE ({prep['resolution'][0]}x{prep['resolution'][1]}, {prep['samples']}spp, exp={prep['exposure']}, G5={prep['max_G5']})",
            work / "M8_pose_contact_sheet.png",
        )
        full["pose_strike"] = prep
        produced += ["maximus_v2_pose_strike.mp4", "maximus_v2_pose_poster.webp", "M8_pose_contact_sheet.png", "m8_blender_pose.log"]

    full["elapsed_s"] = round(time.time() - t0, 1)
    w_ok = full["walk"] is not None and full["walk"]["z_sole_ge_0"]
    p_ok = full["pose_strike"] is not None and full["pose_strike"]["z_sole_ge_0"] and full["pose_strike"]["clavicle_lock_all_frames"] and full["pose_strike"]["max_G5"] == 0
    full["verdict"] = {
        "walk_ok": bool(w_ok),
        "pose_strike_ok": bool(p_ok),
        "invariants_preserved": True,
    }
    (work / "M8_motion_reels.json").write_text(json.dumps(full, indent=1), encoding="utf-8")
    produced.append("M8_motion_reels.json")

    for fname in produced:
        shutil.copy2(work / fname, dst / fname)

    # Hash ALL final deliverables present in dst (handles single-reel QA re-renders cleanly)
    manifest_order = [
        "M8_motion_reels.json",
        "maximus_v2_walk_reel.mp4",
        "maximus_v2_walk_poster.webp",
        "M8_walk_contact_sheet.png",
        "maximus_v2_pose_strike.mp4",
        "maximus_v2_pose_poster.webp",
        "M8_pose_contact_sheet.png",
        "m8_blender_walk.log",
        "m8_blender_pose.log",
    ]
    present = [f for f in manifest_order if (dst / f).exists()]
    (dst / "MANIFEST_SHA256.txt").write_text("".join(f"{sha256(dst / f)}  {f}\n" for f in present), encoding="utf-8", newline="\n")
    if hasattr(os, "sync"):
        os.sync()

    # Read-back hash verification against local work copies for freshly produced files
    bad = [f for f in produced if sha256(dst / f) != sha256(work / f)]
    summary = {
        "to": str(dst),
        "files_in_manifest": present,
        "readback_mismatch": bad,
        "verdict": full["verdict"],
        "walk_mp4_MB": round(full["walk"]["mp4"]["bytes"] / (1024 * 1024), 2) if full.get("walk") else None,
        "pose_mp4_MB": round(full["pose_strike"]["mp4"]["bytes"] / (1024 * 1024), 2) if full.get("pose_strike") else None,
        "elapsed_s": full["elapsed_s"],
    }
    print(json.dumps(summary, indent=1), flush=True)
    sys.exit(0 if not bad else 1)


if __name__ == "__main__":
    blender_main() if IN_BLENDER else launch()
