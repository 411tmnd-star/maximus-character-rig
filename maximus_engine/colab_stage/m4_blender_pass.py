"""M4 Blender passes (run by m4_arap_psd_and_atlas.py; never by hand). bpy.data / pose-bone matrices only; the only operators used are the
io_ops_gateway ones (save_copy, bake, render_still).

    blender --background <master.blend> --factory-startup --python-exit-code 3 --python m4_blender_pass.py -- mode=<mode> key=value ...

modes
  extract     16 poses (m_pose_library.ALL_POSES): evaluated world vertices (armature + Corrective Smooth), per-vertex linear-blend 3x3
              matrices, local quaternions of all 152 bones, pose facts  -> m4_pose_inputs.npz
  evaluate    same 16 poses with the PSD keys from psd_npz added to the mesh and each key's weight set from the pose-space weights in
              psd_weights_json (pose -> key -> weight); writes evaluated vertices -> out npz. Used for the depsgraph polish loop.
  apply_bake  add the PSD keys (value 0, slider 0..1), store the pose-space interpolator on the mesh as a custom property, verify G5 on the
              depsgraph for all 16 poses, save maximus_v2_master_psd.blend (io.save_copy), bake 2K outer / inner AO (io.bake, Cycles GPU)
  render      one still of a pose with a given material setup (bronze verdigris) -> png (io.render_still)
"""
import json, sys, time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def kv():
    a = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    return dict(x.split("=", 1) for x in a if "=" in x)


A = kv()
ROOT = Path(A["root"]).resolve()                      # bundle root (Colab) or project root (local): holds pipeline/, poses/, corpus_link.py
sys.path.insert(0, str(ROOT / "pipeline")); sys.path.insert(0, str(HERE))
import bl_common as C                                 # noqa: E402  (links the corpus, strict)
import io_ops_gateway as io                           # noqa: E402
import posing                                         # noqa: E402
import m_pose_library as PL                           # noqa: E402
from corpus_bridge import contracts                   # noqa: E402
from maximus_core import geometry                     # noqa: E402

STAGE = "m4_blender_" + A.get("mode", "extract")
OUT = Path(A["out_dir"]); OUT.mkdir(parents=True, exist_ok=True)


def setup():
    body, rig = C.human_and_rig(); rig.data.pose_position = "POSE"
    arm = [m for m in body.modifiers if m.type == "ARMATURE"]
    cs = [m for m in body.modifiers if m.type == "CORRECTIVE_SMOOTH"]
    assert len(arm) == 1 and not arm[0].use_deform_preserve_volume, "pure LBS required (DQS / preserve volume is forbidden)"
    assert len(cs) == 1 and abs(cs[0].factor - 0.5) < 1e-9 and cs[0].iterations == 5, "Corrective Smooth must stay factor 0.50, repeat 5"
    assert len(body.data.vertices) == 13380 and len(rig.data.bones) == 152
    ev = lambda: C.evaluated_positions(body)        # noqa: E731
    P = posing.Poser(C.bpy, rig)
    fit = json.loads(Path(A["theta_fit"]).read_text(encoding="utf-8"))["bones"]
    lib = PL.PoseLibrary(rig, P, ev, fit, ROOT)
    return body, rig, ev, P, lib


def lbs_setup(body, rig):
    W, names = C.weights_matrix(body)
    bones = [b.name for b in rig.data.bones]
    Wb = np.zeros((len(body.data.vertices), len(bones)))
    for bi, bn in enumerate(bones):
        if bn in names and rig.data.bones[bn].use_deform:
            Wb[:, bi] = W[:, names.index(bn)]
    s = Wb.sum(1, keepdims=True); Wb = np.divide(Wb, s, out=np.zeros_like(Wb), where=s > 0)   # armature modifier normalises by total weight
    rest_inv = np.stack([np.array(b.matrix_local.inverted(), float) for b in rig.data.bones])

    def skin3():
        pbs = rig.pose.bones
        Mb = np.stack([np.array(pbs[b.name].matrix, float) @ rest_inv[i] for i, b in enumerate(rig.data.bones)])
        return np.einsum("vb,bij->vij", Wb, Mb[:, :3, :3])
    return bones, skin3


def mode_extract():
    body, rig, ev, P, lib = setup(); me = body.data
    arr = C.mesh_arrays(me); V_basis, T, uv = arr["V"], arr["T"], arr["uv"]
    lt = np.empty(len(me.loop_triangles) * 3, np.int64); me.loop_triangles.foreach_get("loops", lt); tri_uv = uv[lt.reshape(-1, 3)]
    bones, skin3 = lbs_setup(body, rig)
    P.reset(); V_bind = ev()
    out = {"V_basis": V_basis, "V_bind_eval": V_bind, "T": T, "tri_uv": tri_uv, "bone_names": np.array(bones)}
    W, names = C.weights_matrix(body); out["W"] = W.astype(np.float32); out["group_names"] = np.array(names)
    facts = {}
    for pid in PL.ALL_POSES:
        t0 = time.time(); f = lib.apply(pid)
        out[f"V|{pid}"] = ev(); out[f"M|{pid}"] = skin3().astype(np.float32); out[f"Q|{pid}"] = lib.local_quats(bones)
        f["seconds"] = round(time.time() - t0, 2); facts[pid] = f
        col = geometry.collapsed_triangles(V_basis, out[f"V|{pid}"], T, ratio=0.25)
        print(f"PROGRESS extract {pid}: G5={col['count']} min_ratio={col['min_ratio']:.4f} z_sole={f['z_sole_m']*1000:.4f}mm clav_ok={f['clavicle_lock_ok']}", flush=True)
    P.reset()
    out["facts_json"] = np.array(json.dumps(facts))
    np.savez_compressed(OUT / "m4_pose_inputs.npz", **out)
    return {"npz": str(OUT / "m4_pose_inputs.npz")}, {"poses": len(PL.ALL_POSES), "bind_vs_basis_max_m": float(np.abs(V_bind - V_basis).max())}


def add_keys(body, deltas, V_basis):
    me = body.data
    for k in me.shape_keys.key_blocks:
        k.value = 0.0
    for name, d in deltas.items():
        kb = me.shape_keys.key_blocks.get(name) or body.shape_key_add(name=name, from_mix=False)
        kb.slider_min = 0.0; kb.slider_max = 1.0; kb.value = 0.0; kb.relative_key = me.shape_keys.key_blocks["Basis"]
        kb.data.foreach_set("co", (V_basis + d).astype(np.float32).ravel())
    me.update()


def set_weights(body, w):
    kbs = body.data.shape_keys.key_blocks
    for name, val in w.items():
        kbs[name].value = float(val)


def mode_evaluate(save_blend=False):
    body, rig, ev, P, lib = setup(); me = body.data
    V_basis = C.mesh_arrays(me)["V"]; T = C.mesh_arrays(me)["T"]
    Z = np.load(A["psd_npz"]); keys = [k for k in Z.files if k.startswith("PSD_")]
    deltas = {k: Z[k] for k in keys}; weights = json.loads(Path(A["psd_weights_json"]).read_text())
    add_keys(body, deltas, V_basis)
    out = {}; g5 = {}
    for pid in PL.ALL_POSES:
        set_weights(body, weights[pid]); lib.apply(pid)
        X = ev(); out[f"V|{pid}"] = X
        col = geometry.collapsed_triangles(V_basis, X, T, ratio=0.25); g5[pid] = {"G5": col["count"], "min_ratio": col["min_ratio"], "indices": col["indices"][:40],
                                                                                  "z_sole_mm": round(float(X[:, 2].min()) * 1000, 4)}
        print(f"PROGRESS evaluate {pid}: G5={col['count']} min_ratio={col['min_ratio']:.4f} w={ {k: round(v, 3) for k, v in weights[pid].items()} }", flush=True)
    set_weights(body, {k: 0.0 for k in keys}); P.reset(); rig.data.pose_position = "REST"
    np.savez_compressed(Path(A["out_npz"]), **out)
    (Path(A["out_npz"]).with_suffix(".json")).write_text(json.dumps(g5, indent=1))
    res = {"evaluated": str(A["out_npz"]), "g5": g5}
    if save_blend:
        rig.data.pose_position = "POSE"; P.reset()
        me["maximus_psd"] = json.dumps(json.loads(Path(A["interp_json"]).read_text()))   # pose-space interpolator travels with the mesh
        io.save_copy(OUT / "maximus_v2_master_psd.blend"); res["master_psd"] = str(OUT / "maximus_v2_master_psd.blend")
    return res, {"max_G5": max(v["G5"] for v in g5.values())}


def gpu_setup(sc):
    sc.render.engine = "CYCLES"
    prefs = C.bpy.context.preferences.addons["cycles"].preferences; used = None
    for dt in ("OPTIX", "CUDA"):
        try:
            prefs.compute_device_type = dt; prefs.get_devices()
            devs = [d for d in prefs.devices if d.type == dt]
            if devs:
                for d in prefs.devices:
                    d.use = d.type == dt
                used = dt; break
        except Exception:  # noqa: BLE001
            pass
    sc.cycles.device = "GPU" if used else "CPU"
    if A.get("require_gpu", "1") == "1" and not used:
        raise RuntimeError("no OPTIX/CUDA device: refusing to bake/render on CPU")
    return used or "CPU"


def mode_bake():
    """2K outer AO (normals as modelled) and inner AO (normals flipped: occlusion seen from inside the shell = thickness proxy)."""
    body, rig, ev, P, lib = setup(); me = body.data; sc = C.bpy.context.scene
    P.reset(); rig.data.pose_position = "REST"
    dev = gpu_setup(sc); sc.cycles.samples = int(A.get("ao_samples", "64")); sc.render.bake.margin = 8
    mat = C.bpy.data.materials["MX_M2_AlabasterChalk"]; body.material_slots[0].material = mat
    res = {}
    sc.world = sc.world or C.bpy.data.worlds.new("MX_bake_world")
    # outer: occlusion within 0.25 m (body-scale AO); inner: normals flipped with a 35 mm ray length, so a closed shell reads as a thickness map
    # (with the default world distance every inward ray hits the far wall and the map is black: measured mean 0.00044 in the smoke run)
    for name, flip, dist in (("ao_outer_2k", False, 0.25), ("ao_inner_2k", True, 0.035)):
        sc.world.light_settings.distance = dist
        if flip:
            me.flip_normals(); me.update()
        img = C.bpy.data.images.new("MX_" + name, 2048, 2048, alpha=False, float_buffer=False); img.colorspace_settings.name = "Non-Color"
        nt = mat.node_tree; node = nt.nodes.new("ShaderNodeTexImage"); node.image = img
        for n in nt.nodes:
            n.select = False
        node.select = True; nt.nodes.active = node
        t0 = time.time()
        try:
            io.bake(sc, body, "AO")
        finally:
            nt.nodes.remove(node)
            if flip:
                me.flip_normals(); me.update()
        img.filepath_raw = str(OUT / f"{name}.png"); img.file_format = "PNG"; img.save()
        px = np.empty(2048 * 2048 * 4, np.float32); img.pixels.foreach_get(px); px = px.reshape(-1, 4)[:, 0]
        res[name] = {"png": str(OUT / f"{name}.png"), "mean": round(float(px.mean()), 5), "p01": round(float(np.percentile(px, 1)), 5),
                     "p99": round(float(np.percentile(px, 99)), 5), "ao_distance_m": dist, "normals_flipped": flip, "seconds": round(time.time() - t0, 1)}
        print(f"PROGRESS baked {name} mean={res[name]['mean']} on {dev}", flush=True)
    res["device"] = dev; res["samples"] = sc.cycles.samples
    return res, {"device": dev}


def bronze_material(cavity_png, ao_png):
    """M4 Bronze with cavity-driven verdigris: base bronze (metallic 1, warm albedo) mixed toward a matte blue-green patina where the baked
    curvature-cavity map is concave and the AO map is occluded. Image textures + math nodes only, so the look is reproducible."""
    bpy = C.bpy; mat = bpy.data.materials.new("MX_M4_Bronze_Verdigris"); mat.use_nodes = True
    nt = mat.node_tree; N = nt.nodes; L = nt.links
    bsdf = next(n for n in N if n.type == "BSDF_PRINCIPLED")
    tc = N.new("ShaderNodeTexCoord")
    cav = N.new("ShaderNodeTexImage"); cav.image = bpy.data.images.load(str(cavity_png)); cav.image.colorspace_settings.name = "Non-Color"
    ao = N.new("ShaderNodeTexImage"); ao.image = bpy.data.images.load(str(ao_png)); ao.image.colorspace_settings.name = "Non-Color"
    for t in (cav, ao):
        L.new(tc.outputs["UV"], t.inputs["Vector"])
    # patina mask: cavity term clamp((0.49 - cav) * 12) (concave creases only: cav < 0.5 is concave) OR deep-occlusion term
    # clamp((0.40 - AO) * 5) (only strongly occluded areas; open torso AO is ~0.6 and must stay bronze), then the noise only breaks up the
    # mask edges: mask = clamp(m * (0.55 + 0.9 * noise)). First draft (AO gain 2.2 from 1.0) put patina on most of the torso (smoke render).
    inv = N.new("ShaderNodeMath"); inv.operation = "SUBTRACT"; inv.inputs[0].default_value = 0.49; L.new(cav.outputs["Color"], inv.inputs[1])
    gain = N.new("ShaderNodeMath"); gain.operation = "MULTIPLY"; gain.inputs[1].default_value = 12.0; gain.use_clamp = True; L.new(inv.outputs[0], gain.inputs[0])
    aoi = N.new("ShaderNodeMath"); aoi.operation = "SUBTRACT"; aoi.inputs[0].default_value = 0.40; L.new(ao.outputs["Color"], aoi.inputs[1])
    aog = N.new("ShaderNodeMath"); aog.operation = "MULTIPLY"; aog.inputs[1].default_value = 5.0; aog.use_clamp = True; L.new(aoi.outputs[0], aog.inputs[0])
    mx = N.new("ShaderNodeMath"); mx.operation = "MAXIMUM"; L.new(gain.outputs[0], mx.inputs[0]); L.new(aog.outputs[0], mx.inputs[1])
    noise = N.new("ShaderNodeTexNoise"); noise.inputs["Scale"].default_value = 90.0; L.new(tc.outputs["Object"], noise.inputs["Vector"])
    nb = N.new("ShaderNodeMath"); nb.operation = "MULTIPLY_ADD"; nb.inputs[1].default_value = 0.9; nb.inputs[2].default_value = 0.55
    L.new(noise.outputs["Fac"], nb.inputs[0])
    mask = N.new("ShaderNodeMath"); mask.operation = "MULTIPLY"; mask.use_clamp = True; L.new(mx.outputs[0], mask.inputs[0]); L.new(nb.outputs[0], mask.inputs[1])
    bronze = C.hex_linear("#8C5A2B"); patina = C.hex_linear("#4E8C7A")
    col = N.new("ShaderNodeMix"); col.data_type = "RGBA"; col.inputs[6].default_value = (*bronze, 1.0); col.inputs[7].default_value = (*patina, 1.0)
    L.new(mask.outputs[0], col.inputs[0]); L.new(col.outputs[2], bsdf.inputs["Base Color"])
    met = N.new("ShaderNodeMapRange"); met.inputs[3].default_value = 1.0; met.inputs[4].default_value = 0.0; L.new(mask.outputs[0], met.inputs[0]); L.new(met.outputs[0], bsdf.inputs["Metallic"])
    rough = N.new("ShaderNodeMapRange"); rough.inputs[3].default_value = 0.34; rough.inputs[4].default_value = 0.86; L.new(mask.outputs[0], rough.inputs[0]); L.new(rough.outputs[0], bsdf.inputs["Roughness"])
    return mat


def mode_render():
    """One still: pose `pose` (PSD keys applied at that pose's weights), M4 Bronze with cavity-driven verdigris, s05 rig and framing, GPU."""
    import m_render as R
    body, rig, ev, P, lib = setup(); sc = C.bpy.context.scene
    if A.get("psd_npz"):
        Z = np.load(A["psd_npz"]); add_keys(body, {k: Z[k] for k in Z.files if k.startswith("PSD_")}, C.mesh_arrays(body.data)["V"])
    pid = A.get("pose", "P1")
    if A.get("psd_weights_json"):
        set_weights(body, json.loads(Path(A["psd_weights_json"]).read_text())[pid])
    lib.apply(pid)
    dev = gpu_setup(sc)
    cam, lights = R.studio(C.bpy, sc, samples=int(A.get("samples", "256")), res=(int(A.get("w", "1638")), int(A.get("h", "2048"))),
                           exposure=float(A.get("exposure", "-0.6")))
    camera = R.frame_camera(cam, ev(), view_deg=float(A.get("view_deg", "0")))
    mat = bronze_material(Path(A["cavity_png"]), Path(A["ao_png"])); body.material_slots[0].material = mat
    out_png = Path(A["out_png"]); t0 = time.time(); io.render_still(sc, out_png)
    st = R.image_stats(C.bpy, out_png)
    return {"png": str(out_png), "camera": camera, "stats": st, "device": dev, "seconds": round(time.time() - t0, 1)}, {"device": dev}


MODES = {"extract": mode_extract, "evaluate": lambda: mode_evaluate(False), "apply_save": lambda: mode_evaluate(True), "bake": mode_bake, "render": mode_render}
C.run(STAGE, MODES[A.get("mode", "extract")])
