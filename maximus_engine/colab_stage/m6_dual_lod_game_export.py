"""M6 - dual-LOD export of Maximus v2 (core LOD0 + game LOD1), Colab or local.

    Colab:  !python /content/drive/MyDrive/maximus_v2/m6_dual_lod_game_export.py
    Local:  python -I colab_stage/m6_dual_lod_game_export.py --root <maximus_engine> --master-psd <...psd.blend> --textures out/textures --out <dir>

LOD0 maximus_v2_core_lod0.glb  152-bone skin, 5 anthropometric keys + every PSD corrective key from M4, Draco, budget < 1.8 MB.
     The PSD interpolator (runtime_v3/pose_space_engine.py JSON) rides along as mesh extras "maximus_psd" so a runtime can drive the morph weights.
LOD1 maximus_v2_game_lod1.glb  24-bone skin, Draco, max 4 influences, weights renormalised to sum 1, budget < 1.0 MB, no morph targets.

Locked 24-bone map, resolved against the real bone names of maximus_v2_master.blend (MakeHuman 'default' rig):
  spec name   rig bone     absorbs (weights folded in; every other bone folds into its nearest kept ancestor)
  pelvis      root         pelvis.L/R, spine05, spine04  (the rig has no 'pelvis' bone; root is the pelvis.  NOTE: in this rig spine05 is the
                                                          LOWEST spine bone, so the locked list keeps the upper torso and folds the lumbar
                                                          spine04/05 into the pelvis; the LOD1 fidelity check below measures what that costs)
  spine01-03  spine01-03   breast.L/R -> spine02
  neck01      neck01       neck02, neck03
  head        head         49 facial / cranial bones
  clavicle    clavicle.S   shoulder01.S
  upperarm01  upperarm01.S upperarm02.S
  lowerarm01  lowerarm01.S
  lowerarm02  lowerarm02.S (forearm twist)
  wrist       wrist.S      metacarpals + fingers (19 bones)
  upperleg01  upperleg01.S upperleg02.S
  lowerleg01  lowerleg01.S lowerleg02.S
  foot        foot.S
  toes        toe1-1.S     all toe chains toe1-2, toe2-1..toe5-3 (the rig has no single 'toes' bone; the big-toe root stands for it)
No bone is created or deleted (that would need edit-mode operators, which the owner rule forbids): non-kept bones are marked use_deform=False in
memory and the exporter's deform-bones-only option re-parents each kept bone to its nearest kept ancestor. Nothing is saved to any .blend.
"""
import argparse, hashlib, json, os, shutil, struct, subprocess, sys, time
from pathlib import Path

import numpy as np

# The Blender pass is the only caller that passes "--" (blender ... --python this.py -- key=value). Do not detect Blender by
# "import bpy": a machine with the bpy wheel installed (this PC) would run the Blender branch from system Python.
IN_BLENDER = "--" in sys.argv

HERE = Path(__file__).resolve().parent
STAGE = "m6_dual_lod_game_export"
SPEC_TO_RIG = {"pelvis": "root", "spine01": "spine01", "spine02": "spine02", "spine03": "spine03", "neck01": "neck01", "head": "head",
               **{f"{b}.{s}": f"{b}.{s}" for s in "LR" for b in ("clavicle", "upperarm01", "lowerarm01", "lowerarm02", "wrist", "upperleg01", "lowerleg01", "foot")},
               "toes.L": "toe1-1.L", "toes.R": "toe1-1.R"}
LOD0_BUDGET = int(1.8 * 1024 * 1024); LOD1_BUDGET = int(1.0 * 1024 * 1024)
FIDELITY_POSES = ("theta_fit", "P1", "P2", "P3", "POSE_03", "POSE_08", "arm_elev_120")


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ------------------------------------------------------------------------------------------------------------------------------------
# GLB reading (standard library + NumPy): what the file really contains
# ------------------------------------------------------------------------------------------------------------------------------------
def read_glb(path):
    b = Path(path).read_bytes(); assert b[:4] == b"glTF"
    jlen = struct.unpack_from("<I", b, 12)[0]; J = json.loads(b[20:20 + jlen]); off = 20 + jlen; BIN = b""
    if off < len(b):
        blen = struct.unpack_from("<I", b, off)[0]; BIN = b[off + 8: off + 8 + blen]
    return J, BIN, len(b)


def accessor(J, BIN, idx):
    a = J["accessors"][idx]; bv = J["bufferViews"][a["bufferView"]]
    dt = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}[a["componentType"]]
    nc = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}[a["type"]]; isz = np.dtype(dt).itemsize
    stride = bv.get("byteStride", nc * isz); start = bv.get("byteOffset", 0) + a.get("byteOffset", 0)
    raw = np.frombuffer(BIN, np.uint8, count=stride * (a["count"] - 1) + nc * isz, offset=start)
    rows = np.lib.stride_tricks.as_strided(raw, shape=(a["count"], nc * isz), strides=(stride, 1)).copy()
    arr = rows.view(dt).reshape(a["count"], nc).astype(np.float64)
    if a.get("normalized"):
        arr /= float(np.iinfo(dt).max)
    return arr


def glb_summary(path):
    J, BIN, n = read_glb(path)
    prims = [p for m in J.get("meshes", []) for p in m["primitives"]]; skin = (J.get("skins") or [{}])[0]
    targets = max((len(p.get("targets", [])) for p in prims), default=0)
    sparse = sum(1 for p in prims for t in p.get("targets", []) for k, ai in t.items() if "sparse" in J["accessors"][ai])
    return {"bytes": n, "MB": round(n / 1024 / 1024, 4), "extensionsUsed": J.get("extensionsUsed", []), "primitives": len(prims),
            "vertices_gltf": int(sum(J["accessors"][p["attributes"]["POSITION"]]["count"] for p in prims)),
            "triangles_gltf": int(sum(J["accessors"][p["indices"]]["count"] for p in prims if "indices" in p) // 3),
            "morph_targets": targets, "morph_targets_sparse_accessors": sparse,
            "morph_target_names": (J["meshes"][0].get("extras", {}) or {}).get("targetNames", []),
            "joints": len(skin.get("joints", [])), "joint_names": [J["nodes"][i].get("name") for i in skin.get("joints", [])],
            "has_JOINTS_1": any("JOINTS_1" in p["attributes"] for p in prims), "images": [i.get("mimeType") for i in J.get("images", [])],
            "mesh_extras_keys": sorted((J["meshes"][0].get("extras") or {}).keys())}


def verify_lod1_twin(path):
    """Uncompressed twin of LOD1: weights per vertex <= 4, sum to 1, joint indices in range, 24 joints."""
    J, BIN, _ = read_glb(path); out = {}
    for p in [p for m in J["meshes"] for p in m["primitives"]]:
        Jn = accessor(J, BIN, p["attributes"]["JOINTS_0"]).astype(int); Wt = accessor(J, BIN, p["attributes"]["WEIGHTS_0"])
        s = Wt.sum(1)
        out = {"vertices": int(len(Wt)), "max_influences_used": int((Wt > 0).sum(1).max()), "has_JOINTS_1": "JOINTS_1" in p["attributes"],
               "weight_sum_min": round(float(s.min()), 6), "weight_sum_max": round(float(s.max()), 6), "max_joint_index": int(Jn.max()),
               "weights_component_type": J["accessors"][p["attributes"]["WEIGHTS_0"]]["componentType"]}
    out["joints"] = len(J["skins"][0]["joints"])
    out["ok"] = (out["joints"] == 24 and out["max_influences_used"] <= 4 and not out["has_JOINTS_1"] and out["max_joint_index"] < 24
                 and abs(out["weight_sum_min"] - 1) <= 2e-3 and abs(out["weight_sum_max"] - 1) <= 2e-3)
    return out


# ------------------------------------------------------------------------------------------------------------------------------------
# Blender side
# ------------------------------------------------------------------------------------------------------------------------------------
def blender_main():
    A = dict(x.split("=", 1) for x in sys.argv[sys.argv.index("--") + 1:] if "=" in x)
    root = Path(A["root"]); sys.path.insert(0, str(root / "pipeline")); sys.path.insert(0, str(HERE))
    import bl_common as C
    import io_ops_gateway as io
    import posing
    import m_pose_library as PL
    out = Path(A["out_dir"]); out.mkdir(parents=True, exist_ok=True)

    def main():
        body, rig = C.human_and_rig(); me = body.data; sc = C.bpy.context.scene
        me.calc_loop_triangles()
        arm = [m for m in body.modifiers if m.type == "ARMATURE"]; cs = [m for m in body.modifiers if m.type == "CORRECTIVE_SMOOTH"]
        assert len(me.vertices) == 13380 and len(me.loop_triangles) == 26756, f"base topology drift: verts={len(me.vertices)} tris={len(me.loop_triangles)}"
        assert len(rig.data.bones) == 152, f"rig bone count drift: {len(rig.data.bones)}"
        assert len(arm) == 1 and not arm[0].use_deform_preserve_volume, "pure LBS required (ZERO DQS)"
        assert len(cs) == 1 and abs(cs[0].factor - 0.5) < 1e-9 and cs[0].iterations == 5, "Corrective Smooth must be factor=0.50, repeat=5"
        keys = [k.name for k in me.shape_keys.key_blocks]
        anthro = [k for k in keys if k in ("gt_lean", "gt_stocky_nat", "gt_stocky_ext", "leg_len_m5cm", "leg_len_p5cm")]
        psd = [k for k in keys if k.startswith("PSD_")]
        assert len(anthro) == 5, f"anthropometric keys: {anthro}"
        for k in me.shape_keys.key_blocks:
            k.value = 0.0
        rig.data.pose_position = "REST"
        # --- material: M2 baked set (s04), 1K for LOD0, `lod1_tex` px for LOD1
        tex = Path(A["textures"])

        def core_material(name, px):
            mat = C.bpy.data.materials.new(name)
            if not getattr(mat, "use_nodes", True):
                mat.use_nodes = True
            nt = mat.node_tree
            bsdf = next(n for n in nt.nodes if n.type == "BSDF_PRINCIPLED")

            def img(file, cs_):
                im = C.bpy.data.images.load(str(tex / "M2" / file), check_existing=False); im.colorspace_settings.name = cs_
                im.scale(px, px); n = nt.nodes.new("ShaderNodeTexImage"); n.image = im; return n
            nb = img("basecolor_2k.png", "sRGB"); nt.links.new(nb.outputs["Color"], bsdf.inputs["Base Color"])
            no = img("orm_2k.png", "Non-Color"); sep = nt.nodes.new("ShaderNodeSeparateColor"); nt.links.new(no.outputs["Color"], sep.inputs["Color"])
            nt.links.new(sep.outputs["Green"], bsdf.inputs["Roughness"]); nt.links.new(sep.outputs["Blue"], bsdf.inputs["Metallic"])
            nn = img("normal_2k.png", "Non-Color"); nm = nt.nodes.new("ShaderNodeNormalMap"); nt.links.new(nn.outputs["Color"], nm.inputs["Color"])
            nt.links.new(nm.outputs["Normal"], bsdf.inputs["Normal"])
            return mat

        res = {"anthro_keys": anthro, "psd_keys": len(psd), "bone_map_resolution": {}, "gateway_calls": None}
        if "maximus_psd" in me.keys():
            res["psd_interpolator_bytes"] = len(me["maximus_psd"])
        # ---------------- LOD0
        body.material_slots[0].material = core_material("MX_core_M2_baked_lod0", 1024)
        lod0 = out / "maximus_v2_core_lod0.glb"
        io.export_glb(sc, [body, rig], lod0, draco=True, export_extras=True, export_try_sparse_sk=True)
        res["lod0"] = glb_summary(lod0); res["lod0"]["sha256"] = sha256(lod0); res["lod0"]["under_budget"] = res["lod0"]["bytes"] < LOD0_BUDGET
        print(f"PROGRESS LOD0 {res['lod0']['MB']} MB, joints {res['lod0']['joints']}, morphs {res['lod0']['morph_targets']} (sparse accessors {res['lod0']['morph_targets_sparse_accessors']})", flush=True)

        # ---------------- LOD1: resolve names, fold, top-4, normalise (in memory)
        bones = [b.name for b in rig.data.bones]; parent = {b.name: (b.parent.name if b.parent else None) for b in rig.data.bones}
        missing = [r for r in SPEC_TO_RIG.values() if r not in bones]
        assert not missing, f"locked map names not in the rig: {missing}"
        keep = list(SPEC_TO_RIG.values()); assert len(set(keep)) == 24
        res["bone_map_resolution"] = {s: r for s, r in SPEC_TO_RIG.items() if s != r}

        def target(b):
            if b in keep:
                return b
            if b.startswith("toe") and b[-2:] in (".L", ".R"):
                return "toe1-1" + b[-2:]
            p = parent[b]
            while p is not None and p not in keep:
                p = parent[p]
            return p
        fold = {b: target(b) for b in bones}
        assert all(v is not None for v in fold.values()), "a bone has no kept ancestor"
        res["fold_map"] = {k: [b for b in bones if fold[b] == k and b != k] for k in keep}
        W, names = C.weights_matrix(body); W = W.astype(np.float64); nidx = {n: i for i, n in enumerate(names)}
        W152 = np.zeros((len(W), len(bones)))
        for bi, b in enumerate(bones):
            if b in nidx and rig.data.bones[b].use_deform:
                W152[:, bi] = W[:, nidx[b]]
        F = np.zeros((len(bones), 24))
        for bi, b in enumerate(bones):
            F[bi, keep.index(fold[b])] = 1.0
        W24 = W152 @ F
        order = np.argsort(-W24, axis=1); W4 = np.zeros_like(W24); rows = np.arange(len(W24))[:, None]
        W4[rows, order[:, :4]] = W24[rows, order[:, :4]]
        s = W4.sum(1, keepdims=True); assert (s > 0).all(), "vertex with no weight after folding"
        W4 /= s
        res["lod1_weights"] = {"vertices_over_4_before_limit": int(((W24 > 1e-6).sum(1) > 4).sum()), "max_dropped_weight": round(float((W24 - W24 * (W4 > 0)).max()), 5),
                               "max_influences": int((W4 > 0).sum(1).max()), "sum_min": float(W4.sum(1).min()), "sum_max": float(W4.sum(1).max())}

        # Reference-pose articulation of the folded bones is baked into LOD1's rest geometry: v' = v + sum_b w_b (M_k(fit)^-1 M_b(fit) - I) v
        # for every folded bone b with kept target k = fold(b). Without it the fingers (3,075 finger/metacarpal-dominant vertices) snap back to
        # the splayed bind hand: measured 67 mm max / 58 mm p99 LOD1-vs-LOD0 error at theta_fit. Vertex count, order and topology unchanged.
        rig.data.pose_position = "POSE"; P = posing.Poser(C.bpy, rig); ev = lambda: C.evaluated_positions(body)  # noqa: E731
        lib = PL.PoseLibrary(rig, P, ev, json.loads(Path(A["theta_fit"]).read_text())["bones"], root)
        V0 = C.mesh_arrays(me)["V"]; rest_inv = np.stack([np.array(b.matrix_local.inverted(), float) for b in rig.data.bones])
        s152 = W152.sum(1, keepdims=True); Wn = np.divide(W152, s152, out=np.zeros_like(W152), where=s152 > 0)
        kidx = [bones.index(k) for k in keep]; Vh = np.c_[V0, np.ones(len(V0))]

        def skin_mats():
            return np.stack([np.array(rig.pose.bones[b].matrix, float) @ rest_inv[i] for i, b in enumerate(bones)])
        lib.apply("theta_fit"); Mfit = skin_mats()
        rel = np.zeros((len(bones), 4, 4))
        for bi, b in enumerate(bones):
            k = fold[b]; rel[bi] = np.linalg.inv(Mfit[bones.index(k)]) @ Mfit[bi] - np.eye(4) if k != b else 0.0
        V1 = V0 + np.einsum("vb,bij,vj->vi", Wn, rel, Vh)[:, :3]; V1h = np.c_[V1, np.ones(len(V1))]
        res["lod1_rest_bake"] = {"reference_pose": "theta_fit", "vertices_moved_over_0p1mm": int((np.linalg.norm(V1 - V0, axis=1) > 1e-4).sum()),
                                 "max_move_mm": round(float(np.linalg.norm(V1 - V0, axis=1).max()) * 1e3, 3)}
        fid = {}; clav_ok = {}
        for pid in FIDELITY_POSES:
            facts = lib.apply(pid); clav_ok[pid] = bool(facts["clavicle_lock_ok"]); Mb = skin_mats()
            assert facts["clavicle_lock_ok"], f"clavicle lock violated on static pose {pid}: {facts['clavicles']}"
            X0 = np.einsum("vb,bij,vj->vi", Wn, Mb, Vh)[:, :3]
            X1 = np.einsum("vk,kij,vj->vi", W4, Mb[kidx], V1h)[:, :3]; X1n = np.einsum("vk,kij,vj->vi", W4, Mb[kidx], Vh)[:, :3]
            e = np.linalg.norm(X1 - X0, axis=1) * 1e3; en = np.linalg.norm(X1n - X0, axis=1) * 1e3
            fid[pid] = {"max_mm": round(float(e.max()), 3), "p99_mm": round(float(np.percentile(e, 99)), 3), "mean_mm": round(float(e.mean()), 4),
                        "without_rest_bake_max_mm": round(float(en.max()), 3), "without_rest_bake_p99_mm": round(float(np.percentile(en, 99)), 3)}
            print(f"PROGRESS LOD1 fidelity {pid}: max {fid[pid]['max_mm']} mm p99 {fid[pid]['p99_mm']} mm (no rest bake: {fid[pid]['without_rest_bake_max_mm']} mm)", flush=True)
        P.reset(); rig.data.pose_position = "REST"; res["lod1_fidelity_vs_lod0"] = fid
        res["invariants"] = {"base_verts": len(me.vertices), "base_tris": len(me.loop_triangles), "bones": len(rig.data.bones),
                             "lbs_only_zero_dqs": True, "corrective_smooth": {"factor": float(cs[0].factor), "iterations": int(cs[0].iterations)},
                             "clavicle_lock_per_pose": clav_ok, "clavicle_lock_all_poses": all(clav_ok.values())}
        body.shape_key_clear()                               # LOD1 carries no morphs; clearing lets the baked rest positions be the exported mesh
        me.vertices.foreach_set("co", V1.astype(np.float32).ravel()); me.update()

        # write the folded weights into the vertex groups (memory only) and mark non-kept bones non-deforming
        allv = list(range(len(me.vertices)))
        for b in bones:
            if b in nidx:
                body.vertex_groups[b].remove(allv)
        for k_i, k in enumerate(keep):
            vg = body.vertex_groups.get(k) or body.vertex_groups.new(name=k)
            nz = np.nonzero(W4[:, k_i] > 0)[0]
            for w_val in np.unique(np.round(W4[nz, k_i], 7)):
                vv = nz[np.round(W4[nz, k_i], 7) == w_val].tolist(); vg.add(vv, float(w_val), "REPLACE")
        for b in rig.data.bones:
            b.use_deform = b.name in keep
        sizes = []
        for px in (int(A.get("lod1_tex", "1024")), 512, 256):
            body.material_slots[0].material = core_material(f"MX_core_M2_baked_lod1_{px}", px)
            lod1 = out / "maximus_v2_game_lod1.glb"
            io.export_glb(sc, [body, rig], lod1, draco=True, export_def_bones=True, export_morph=False, export_all_influences=False)
            sizes.append((px, lod1.stat().st_size))
            if lod1.stat().st_size < LOD1_BUDGET:
                break
        twin = out / "maximus_v2_game_lod1_uncompressed_twin.glb"
        io.export_glb(sc, [body, rig], twin, draco=False, export_def_bones=True, export_morph=False, export_all_influences=False)
        res["lod1"] = glb_summary(lod1); res["lod1"]["sha256"] = sha256(lod1); res["lod1"]["texture_px"] = px; res["lod1"]["size_by_texture_px"] = sizes
        res["lod1"]["under_budget"] = res["lod1"]["bytes"] < LOD1_BUDGET; res["lod1_twin_check"] = verify_lod1_twin(twin)
        res["lod1"]["joint_names_match_locked_map"] = sorted(res["lod1"]["joint_names"]) == sorted(keep)
        res["gateway_calls"] = io.CALLS
        (out / "M6_dual_lod.json").write_text(json.dumps(res, indent=1, default=str))
        return {"report": str(out / "M6_dual_lod.json")}, {"lod0_MB": res["lod0"]["MB"], "lod1_MB": res["lod1"]["MB"], "lod1_ok": res["lod1_twin_check"]["ok"]}

    C.run(STAGE, main)


# ------------------------------------------------------------------------------------------------------------------------------------
def launch():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default="/content/maximus_v2/bundle"); ap.add_argument("--master-psd", default=None)
    ap.add_argument("--textures", default="/content/drive/MyDrive/maximus_v2/textures"); ap.add_argument("--out", default="/content/drive/MyDrive/maximus_v2/results")
    ap.add_argument("--blender"); ap.add_argument("--lod1-tex", default="1024")
    a = ap.parse_args(); t0 = time.time(); root = Path(a.root).resolve()
    master = Path(a.master_psd) if a.master_psd else root / "out" / "maximus_v2_master_psd.blend"
    exe = Path(a.blender) if a.blender else next(Path("/content/blender").glob("blender-*/blender"))
    fit = next(p for p in (HERE / "m_theta_fit.json", Path("/content/drive/MyDrive/maximus_v2/m_theta_fit.json")) if p.exists())
    tex_dir = next((p for p in (Path(a.textures), root / "out" / "textures", HERE.parent / "out" / "textures") if (p / "M2" / "basecolor_2k.png").exists()), Path(a.textures))
    work = Path("/content/m6_work") if a.out.startswith("/content/drive") else Path(a.out) / "_m6_work"; work.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, MAXIMUS_STRICT="1")
    if (root / "corpus").is_dir():
        env["MAXIMUS_CORPUS"] = str(root / "corpus")
    cmd = [str(exe), "--background", str(master), "--factory-startup", "--python-exit-code", "3", "--python", str(Path(__file__).resolve()), "--",
           f"root={root}", f"out_dir={work}", f"textures={tex_dir}", f"theta_fit={fit}", f"lod1_tex={a.lod1_tex}"]
    p = subprocess.run(cmd, env=env, capture_output=True, text=True); (work / "m6_blender.log").write_text(p.stdout + p.stderr, encoding="utf-8")
    for line in p.stdout.splitlines():
        if line.startswith("PROGRESS"):
            print("   " + line)
    envl = [l for l in p.stdout.splitlines() if l.startswith("EC02 ")]
    rep = json.loads(envl[-1][5:]) if envl else {"status": "fail", "errors": [p.stdout[-3000:], p.stderr[-3000:]]}
    if rep.get("status") != "ok":
        print(json.dumps(rep.get("errors"), indent=1)[:6000]); sys.exit(1)
    R = json.loads((work / "M6_dual_lod.json").read_text()); R["master_psd_sha256"] = sha256(master); R["elapsed_s"] = round(time.time() - t0, 1)
    inv = R.get("invariants", {}); tf_fid = R["lod1_fidelity_vs_lod0"]["theta_fit"]
    R["verdict"] = {"core_invariants_ok": (inv.get("base_verts") == 13380 and inv.get("base_tris") == 26756 and inv.get("bones") == 152
                                           and inv.get("lbs_only_zero_dqs") is True
                                           and inv.get("corrective_smooth") == {"factor": 0.5, "iterations": 5}
                                           and inv.get("clavicle_lock_all_poses") is True),
                    "lod0_under_1p8MB": R["lod0"]["under_budget"], "lod0_152_joints": R["lod0"]["joints"] == 152,
                    "lod0_morphs_5_plus_psd": R["lod0"]["morph_targets"] == 5 + R["psd_keys"], "lod1_under_1MB": R["lod1"]["under_budget"],
                    "lod1_24_locked_joints": R["lod1"]["joint_names_match_locked_map"], "lod1_weights_ok": R["lod1_twin_check"]["ok"],
                    "lod1_no_morphs": R["lod1"]["morph_targets"] == 0,
                    "lod1_rest_bake_effective": tf_fid["p99_mm"] < 1.0 and tf_fid["p99_mm"] < tf_fid["without_rest_bake_p99_mm"],
                    "lod1_topology_equals_lod0": (R["lod1"]["vertices_gltf"], R["lod1"]["triangles_gltf"]) == (R["lod0"]["vertices_gltf"], R["lod0"]["triangles_gltf"]),
                    "lod0_triangles_26756": R["lod0"]["triangles_gltf"] == 26756}
    R["fidelity_note"] = "LOD1-vs-LOD0 LBS deviation is measured and reported; rest-bake must bring theta_fit p99 < 1.0 mm"
    (work / "M6_dual_lod.json").write_text(json.dumps(R, indent=1))
    dst = Path(a.out) / "M6_game_canon"; dst.mkdir(parents=True, exist_ok=True)
    keep = ["maximus_v2_core_lod0.glb", "maximus_v2_game_lod1.glb", "maximus_v2_game_lod1_uncompressed_twin.glb", "M6_dual_lod.json", "m6_blender.log"]
    for f in keep:
        shutil.copy2(work / f, dst / f)
    (dst / "MANIFEST_SHA256.txt").write_text("".join(f"{sha256(work / f)}  {f}\n" for f in keep), newline="\n")   # LF so `sha256sum -c` works
    if hasattr(os, "sync"):
        os.sync()
    bad = [f for f in keep if sha256(dst / f) != sha256(work / f)]
    print(json.dumps({"to": str(dst), "readback_mismatch": bad, "lod0_MB": R["lod0"]["MB"], "lod1_MB": R["lod1"]["MB"], "verdict": R["verdict"],
                      "lod1_fidelity_vs_lod0": R["lod1_fidelity_vs_lod0"]}, indent=1))
    sys.exit(0 if all(R["verdict"].values()) and not bad else 1)


if __name__ == "__main__":
    blender_main() if IN_BLENDER else launch()
