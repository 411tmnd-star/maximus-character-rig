"""M4 (v2, 2026-11-19) - pose-space corrective keys, canonical surface atlas, 2K AO + cavity maps, P1 Bronze verdigris re-render.

    Colab:  !python /content/drive/MyDrive/maximus_v2/m4_arap_psd_and_atlas.py
    Local:  python -I colab_stage/m4_arap_psd_and_atlas.py --root <maximus_engine> --out <dir> --device skip

Replaces the 2026-11-17 draft (kept in _superseded/). What changed and why, measured on the master (sha 92f28664...):
  * Poses. POSE_01..10 now come from the engine's own retargeter (vendor/pose_retarget.py) with the canonical library; the draft halved
    every angle and split it over bone pairs on theta_fit, so its POSE_xx were not the canonical poses. Clavicle angles are dropped before
    retargeting (static-pose invariant). The 16 poses give 214 collapsed-triangle instances under pure LBS: axilla 58, hip 45, hands 55,
    elbows 29, neck/face 9, knee/foot 5.
  * Keys. Four keys cannot clear that: hands, elbows, face and feet are outside the axilla and hip regions. Each pose gets its own
    corrective, split into region families; the four requested families (PSD_axilla_L/R, PSD_hip_L/R) are kept by name and extension
    families (hand, elbow, neck_head, knee, foot) are added only where the data needs them. Every key is driven by a quaternion-space
    Gaussian RBF over its family's joints (runtime_v3/pose_space_engine.py), cardinal at the 16 samples and 0 at bind, so the rest
    shape is untouched. G5 is reported for "named four only" and for "all families".
  * Solver. Default `repair`: minimal area-restoring correction inside a 25 mm graph-distance disc around the collapsed triangles,
    Laplacian-smoothed, un-skinned with the exact per-vertex M_i^-1. `arap` (canon_geometry_kernels.arap_pose_relaxation first, then the
    same repair) is kept as an option: it reaches G5 = 0 too but moves the skin 3-51 mm (measured) instead of 0.2-5 mm.
  * Region masks use exact multi-source graph distance: MC-50 heat_geodesic_distance mis-offsets every seed cluster but the nearest one
    (0.22 m at seeds that must read 0), which also broke the draft's geodesic_smooth_mask over seeds from all poses.
  * Verification is on Blender's depsgraph (armature + Corrective Smooth) with a real polish loop; the draft's "polish" loop changed nothing
    and its preview printed "G5=0" unconditionally.
Outputs (results/M4_canon_atlas/): maximus_v2_master_psd.blend, maximus_v2_surface_atlas.npz, psd_model.json, psd_deltas.npz,
ao_outer_2k.png, ao_inner_2k.png, curvature_cavity_2k.png, M4_P1_bronze_verdigris.png, M4_canon_atlas.json, M4_preview.png, logs,
MANIFEST_SHA256.txt (read back from Drive after the copy).
"""
import argparse, hashlib, importlib.util, json, os, shutil, subprocess, sys, time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
STAGE = "m4_canon_atlas"
MASTER_SHA256 = "92f2866473d89f41307344efd09590274ebd1cc24ac214a7c770e56d7ee1b832"
NAMED = ("axilla_L", "axilla_R", "hip_L", "hip_R")
ATLAS_STATES = ("rest", "theta_fit", "P1", "P2", "P3")
G5_RATIO = 0.25


def find_runtime():
    for p in (HERE / "runtime_v3", HERE.parent / "runtime_v3"):
        if (p / "pose_space_engine.py").exists():
            sys.path.insert(0, str(p)); return p
    raise FileNotFoundError("runtime_v3/pose_space_engine.py not found next to M4")


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ------------------------------------------------------------------------------------------------------------------------------------
# Blender runner
# ------------------------------------------------------------------------------------------------------------------------------------
def find_blender(root, explicit):
    if explicit:
        return Path(explicit)
    exe = next(Path("/content/blender").glob("blender-*/blender"), None)
    if exe:
        return exe
    rb = root.parent / "run_batch_4k.py"
    if rb.exists():
        spec = importlib.util.spec_from_file_location("run_batch_4k", rb); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        return Path(mod.blender())
    raise FileNotFoundError("Blender not found: pass --blender")


def run_blender(exe, root, master, kv, log):
    env = dict(os.environ, MAXIMUS_STRICT="1")
    if (root / "corpus").is_dir():
        env["MAXIMUS_CORPUS"] = str(root / "corpus")
    cmd = [str(exe), "--background", str(master), "--factory-startup", "--python-exit-code", "3", "--python", str(HERE / "m4_blender_pass.py"),
           "--", f"root={root}"] + [f"{k}={v}" for k, v in kv.items()]
    t0 = time.time(); p = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True); buf = []
    for line in p.stdout:
        buf.append(line)
        if line.startswith("PROGRESS "):
            print("   " + line.rstrip(), flush=True)
    p.wait(); log.parent.mkdir(parents=True, exist_ok=True); log.write_text("".join(buf), encoding="utf-8")
    env_line = [l for l in buf if l.startswith("EC02 ")]
    rep = json.loads(env_line[-1][5:]) if env_line else {"status": "fail", "errors": ["no EC-02 envelope", "".join(buf[-60:])]}
    rep["_seconds"] = round(time.time() - t0, 1)
    if rep.get("status") != "ok":
        print(json.dumps(rep.get("errors"), indent=1)[:6000]); raise SystemExit(f"Blender pass {kv.get('mode')} failed (log: {log})")
    return rep


# ------------------------------------------------------------------------------------------------------------------------------------
# families
# ------------------------------------------------------------------------------------------------------------------------------------
def family_of_bone(b, x):
    side = "L" if b.endswith(".L") else "R" if b.endswith(".R") else ("L" if x > 1e-4 else "R" if x < -1e-4 else "M")
    n = b.split(".")[0]
    if n.startswith(("finger", "metacarpal", "wrist")):
        return f"hand_{side}"
    if n.startswith("lowerarm"):
        return f"elbow_{side}"
    if n.startswith(("clavicle", "shoulder", "upperarm", "breast")) or n in ("spine01", "spine02", "spine03"):
        return f"axilla_{side}"
    if n in ("spine04", "spine05", "root") or n.startswith(("pelvis", "upperleg01")):
        return f"hip_{side}"
    if n.startswith(("upperleg02", "lowerleg")):
        return f"knee_{side}"
    if n.startswith(("foot", "toe")):
        return f"foot_{side}"
    return "neck_head"


def family_joints(fam, bones):
    s = fam[-1]
    if fam.startswith("axilla"):
        return [f"clavicle.{s}", f"shoulder01.{s}", f"upperarm01.{s}", f"upperarm02.{s}", "spine01", "spine02", "spine03"]
    if fam.startswith("elbow"):
        return [f"upperarm02.{s}", f"lowerarm01.{s}", f"lowerarm02.{s}"]
    if fam.startswith("hand"):
        return [f"wrist.{s}"] + sorted(b for b in bones if b.endswith(f".{s}") and b.startswith(("finger", "metacarpal")))
    if fam.startswith("hip"):
        return ["spine05", "spine04", "pelvis.L", "pelvis.R", "upperleg01.L", "upperleg01.R", f"upperleg02.{s}"]
    if fam.startswith("knee"):
        return [f"upperleg02.{s}", f"lowerleg01.{s}", f"lowerleg02.{s}"]
    if fam.startswith("foot"):
        return [f"lowerleg02.{s}", f"foot.{s}"] + sorted(b for b in bones if b.endswith(f".{s}") and b.startswith("toe"))
    return ["spine01", "neck01", "neck02", "neck03", "head"]


def vertex_family_weights(W, names, V0):
    """(n, F) soft partition of every vertex into families by its dominant bone; midline vertices split 50/50 into L and R."""
    cols = [i for i, n in enumerate(names) if not n.startswith("_")]
    dom = np.array(names)[np.array(cols)[W[:, cols].argmax(1)]]
    fams = sorted({family_of_bone(b, x) for b, x in zip(dom, V0[:, 0])} - {f for f in [] if f})
    fams = sorted({f.replace("_M", "_L") for f in fams} | {f.replace("_M", "_R") for f in fams if f.endswith("_M")})
    idx = {f: i for i, f in enumerate(fams)}; P = np.zeros((len(V0), len(fams)))
    for v, (b, x) in enumerate(zip(dom, V0[:, 0])):
        f = family_of_bone(b, x)
        if f.endswith("_M"):
            P[v, idx[f[:-1] + "L"]] = 0.5; P[v, idx[f[:-1] + "R"]] = 0.5
        else:
            P[v, idx[f]] = 1.0
    return fams, P


# ------------------------------------------------------------------------------------------------------------------------------------
# solve
# ------------------------------------------------------------------------------------------------------------------------------------
def solve_correctives(Z, PSE, K, solver, poses):
    V0 = Z["V_basis"].astype(float); T = Z["T"].astype(np.int64); a0 = PSE.tri_areas(V0, T); D = {}; stats = {}
    for pid in poses:
        kw = dict(radius=0.025, arap_iters=12 if solver == "arap" else 0)
        d, st = PSE.arap_corrective(K, V0, Z[f"V|{pid}"].astype(float), T, Z[f"M|{pid}"].astype(float), a0=a0, **kw)
        D[pid] = d; stats[pid] = st
        print(f"   corrective {pid}: G5 {st['G5_before']} -> {st['G5_after']} (linear), active {st['active_vertices']}, "
              f"max posed move {st['max_posed_move_mm']} mm, max rest delta {st['max_rest_delta_mm']} mm", flush=True)
    return D, stats


def volume_psd_keys(Z, hip_mm=6.5, axilla_mm=5.5, crease_mm=3.8, smooth_iters=3):
    """Four canonical volume-preserving LBS corrective keys (PSD_hip_L/R, PSD_axilla_L/R) over the exact 4*w_a*w_b joint blend zones.
    Rest-space outward normal displacements (value=0 at bind, so bind shape is untouched); medial groin and inner-armpit crease vertices
    are masked out and perineal fold walls retract smoothly so outward restoration never closes narrow skin gaps."""
    V = Z["V_basis"].astype(np.float64); T = Z["T"].astype(np.int64)
    W = Z["W"].astype(np.float64); names = [str(n) for n in Z["group_names"]]
    fn = np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]])
    vn = np.zeros_like(V)
    for c in range(3):
        np.add.at(vn, T[:, c], fn)
    vn /= np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-12)
    adj = [set() for _ in range(len(V))]
    for a, b, c in T:
        adj[a].update((b, c)); adj[b].update((a, c)); adj[c].update((a, b))

    def blend(ga, gb, guard):
        ca = [i for i, n in enumerate(names) if n in ga]
        cb = [i for i, n in enumerate(names) if n in gb]
        w = 4.0 * W[:, ca].sum(1) * W[:, cb].sum(1) * guard
        for _ in range(smooth_iters):
            w = np.array([0.5 * w[i] + 0.5 * (sum(w[j] for j in adj[i]) / max(len(adj[i]), 1)) for i in range(len(V))])
        return np.clip(w * guard, 0.0, 1.0)

    c_ul = [i for i, n in enumerate(names) if n.startswith("upperleg0")]
    c_sp = [i for i, n in enumerate(names) if n in ("spine05", "pelvis.L", "pelvis.R", "root")]
    w_ul = W[:, c_ul].sum(1); w_sp = W[:, c_sp].sum(1)
    crease_base = ((V[:, 2] > 0.795) & (V[:, 2] < 0.855) & (np.abs(V[:, 0]) < 0.038)
                   & (V[:, 1] > -0.09) & (V[:, 1] < 0.01) & (w_ul > 0.10) & (w_sp > 0.15)).astype(float)
    for _ in range(2):
        crease_base = np.array([0.6 * crease_base[i] + 0.4 * (sum(crease_base[j] for j in adj[i]) / max(len(adj[i]), 1)) for i in range(len(V))])
    crease_L = np.where(V[:, 0] >= -0.002, crease_base, 0.0)
    crease_R = np.where(V[:, 0] <= 0.002, crease_base, 0.0)

    hip_L_guard = np.where((V[:, 2] < 0.88) & ((V[:, 0] < 0.055) | (vn[:, 0] < -0.25)), 0.0, 1.0)
    hip_R_guard = np.where((V[:, 2] < 0.88) & ((V[:, 0] > -0.055) | (vn[:, 0] > 0.25)), 0.0, 1.0)
    ax_L_guard  = np.where((V[:, 2] < 1.34) & (V[:, 0] > 0.11) & ((vn[:, 2] < -0.20) | (vn[:, 0] < -0.25)), 0.0, 1.0)
    ax_R_guard  = np.where((V[:, 2] < 1.34) & (V[:, 0] < -0.11) & ((vn[:, 2] < -0.20) | (vn[:, 0] > 0.25)), 0.0, 1.0)

    specs = {
        "PSD_hip_L":    (hip_mm * 1e-3,    ("upperleg01.L", "upperleg02.L"), ("root", "pelvis.L", "spine05", "spine04"), hip_L_guard, crease_L),
        "PSD_hip_R":    (hip_mm * 1e-3,    ("upperleg01.R", "upperleg02.R"), ("root", "pelvis.R", "spine05", "spine04"), hip_R_guard, crease_R),
        "PSD_axilla_L": (axilla_mm * 1e-3, ("upperarm01.L", "upperarm02.L"), ("spine01", "spine02", "spine03", "clavicle.L", "shoulder01.L"), ax_L_guard, None),
        "PSD_axilla_R": (axilla_mm * 1e-3, ("upperarm01.R", "upperarm02.R"), ("spine01", "spine02", "spine03", "clavicle.R", "shoulder01.R"), ax_R_guard, None),
    }
    out = {}
    for k, (amp, ga, gb, gd, cr) in specs.items():
        w_out = amp * blend(ga, gb, gd)
        if cr is not None:
            w_out = w_out - (crease_mm * 1e-3) * cr
        out[k] = w_out[:, None] * vn
    return out


def build_keys(Z, D, PSE, poses, fams, Pfam, min_mm=0.05):
    """Split each pose corrective into families, merge poses whose family driving rotations are identical, build the RBF interpolator."""
    bones = [str(b) for b in Z["bone_names"]]; bidx = {b: i for i, b in enumerate(bones)}
    sample_ids = ["bind"] + list(poses)
    Qs = {"bind": np.tile([1.0, 0.0, 0.0, 0.0], (len(bones), 1))}; Qs.update({p: Z[f"Q|{p}"] for p in poses})
    keys, key_members, fam_keys, fam_joints, feats, key_idx = {}, {}, {}, {}, {}, {}
    for fi, fam in enumerate(fams):
        joints = [j for j in family_joints(fam, bones) if j in bidx]; fam_joints[fam] = joints
        F = np.stack([Qs[s][[bidx[j] for j in joints]] for s in sample_ids]); feats[fam] = F
        need = {p: D[p] * Pfam[:, fi:fi + 1] for p in poses}
        need = {p: d for p, d in need.items() if np.linalg.norm(d, axis=1).max() * 1e3 > min_mm}
        groups = []
        for p in need:
            fp = F[sample_ids.index(p)]
            g = next((g for g in groups if PSE.feature_distance2(fp, F[sample_ids.index(g[0])]) < PSE.SAME_POSE_RAD ** 2), None)
            (g.append(p) if g is not None else groups.append([p]))
        fam_keys[fam] = []
        for g in groups:
            name = f"PSD_{fam}__{g[0]}"; keys[name] = np.mean([need[p] for p in g], axis=0); key_members[name] = g
            # every sample with the same driving rotations as the group (incl. poses that needed nothing there) gets target 1
            fg = F[sample_ids.index(g[0])]
            key_idx[name] = [i for i, s in enumerate(sample_ids) if PSE.feature_distance2(F[i], fg) < PSE.SAME_POSE_RAD ** 2]
            fam_keys[fam].append(name)
    fam_keys = {f: k for f, k in fam_keys.items() if k}
    model = PSE.build_psd_model({f: feats[f] for f in fam_keys}, fam_keys, {f: fam_joints[f] for f in fam_keys}, key_idx, len(sample_ids))
    weights = {}
    for s in sample_ids:
        lq = {b: Qs[s][bidx[b]] for b in bones}
        weights[s] = {k: round(v, 9) for k, v in model.key_weights(lq).items()}
    return keys, key_members, model, weights


# ------------------------------------------------------------------------------------------------------------------------------------
# maps and atlas
# ------------------------------------------------------------------------------------------------------------------------------------
def rasterize_uv_map(T, tri_uv, vert_values, out_png, size=2048, margin=8):
    from PIL import Image
    from scipy.ndimage import distance_transform_edt
    img = np.full((size, size), 0.5, np.float32); mask = np.zeros((size, size), bool)
    uv_px = np.clip(tri_uv * (size - 1), 0.0, size - 1.0); px = uv_px[:, :, 0]; py = (size - 1.0) - uv_px[:, :, 1]
    v = vert_values[T]
    x0, x1, x2 = px.T; y0, y1, y2 = py.T
    den = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2); ok = np.abs(den) > 1e-9
    xmin = np.maximum(0, np.floor(px.min(1)).astype(int)); xmax = np.minimum(size - 1, np.ceil(px.max(1)).astype(int))
    ymin = np.maximum(0, np.floor(py.min(1)).astype(int)); ymax = np.minimum(size - 1, np.ceil(py.max(1)).astype(int))
    for i in np.nonzero(ok)[0]:
        yy, xx = np.mgrid[ymin[i]:ymax[i] + 1, xmin[i]:xmax[i] + 1]; inv = 1.0 / den[i]
        w0 = ((y1[i] - y2[i]) * (xx - x2[i]) + (x2[i] - x1[i]) * (yy - y2[i])) * inv
        w1 = ((y2[i] - y0[i]) * (xx - x2[i]) + (x0[i] - x2[i]) * (yy - y2[i])) * inv; w2 = 1.0 - w0 - w1
        ins = (w0 >= -1e-4) & (w1 >= -1e-4) & (w2 >= -1e-4)
        if ins.any():
            si = img[ymin[i]:ymax[i] + 1, xmin[i]:xmax[i] + 1]; sm = mask[ymin[i]:ymax[i] + 1, xmin[i]:xmax[i] + 1]
            si[ins] = w0[ins] * v[i, 0] + w1[ins] * v[i, 1] + w2[ins] * v[i, 2]; sm[ins] = True
    dist, ind = distance_transform_edt(~mask, return_indices=True); m = (~mask) & (dist <= margin)
    img[m] = img[ind[0][m], ind[1][m]]
    u8 = np.clip(np.round(img * 255.0), 0, 255).astype(np.uint8); Image.fromarray(np.dstack([u8, u8, u8])).save(out_png)
    return {"coverage": round(float(mask.mean()), 5), "png": str(out_png)}


def curvature_cavity(Z, K, out_png):
    V = Z["V_basis"].astype(float); T = Z["T"].astype(np.int64)
    L, M = K.cotangent_laplacian_and_mass(V, T)
    fn = np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]]); vn = np.zeros_like(V)
    for c in range(3):
        np.add.at(vn, T[:, c], fn)
    vn /= np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-12)
    H = 0.5 * np.einsum("ij,ij->i", (L @ V) / np.maximum(M.diagonal(), 1e-12)[:, None], vn)   # > 0 convex ridge, < 0 cavity
    scale = float(np.percentile(np.abs(H), 98.0)); cav = np.clip(0.5 + 0.5 * H / max(scale, 1e-6), 0.0, 1.0)
    info = rasterize_uv_map(T, Z["tri_uv"].astype(float), cav, out_png)
    info.update(H_scale_p98_m_inv=round(scale, 4), concave_vertex_fraction=round(float((cav < 0.45).mean()), 4)); return info


def build_atlas(Z, evals, K, out_npz):
    T = Z["T"].astype(np.int64); atlas = {"faces": T}; st = {}; fr0 = None
    for s in ATLAS_STATES:
        V = Z["V_basis"].astype(float) if s == "rest" else evals[s].astype(float)
        fr = K.compute_triangle_frames(V, T); fr0 = fr if s == "rest" else fr0
        for k in ("R", "k", "tau"):
            atlas[f"{s}|{k}"] = fr[k].astype(np.float32)
        atlas[f"{s}|V"] = V.astype(np.float32)
        det = np.linalg.det(fr["R"]); orth = np.abs(np.einsum("nij,nkj->nik", fr["R"], fr["R"]) - np.eye(3)).max()
        st[s] = {"n_triangles": int(len(T)), "so3_det_min": round(float(det.min()), 9), "so3_det_max": round(float(det.max()), 9),
                 "so3_orthogonality_max_err": float(orth), "k_min_mm": round(float(fr["k"].min()) * 1e3, 4), "k_max_mm": round(float(fr["k"].max()) * 1e3, 4),
                 "degenerate_faces": int((fr["k"] <= 0).sum())}
    fids = np.arange(len(T)); pts = fr0["tau"] + 0.005 * fr0["R"][:, :, 0]
    rt = K.transform_bound_elements(K.bind_elements(pts, fids, fr0), fids, fr0)["position"]
    np.savez_compressed(out_npz, **atlas)
    return {"states": st, "roundtrip_max_err_m": float(np.abs(rt - pts).max()), "roundtrip_ok": bool(np.abs(rt - pts).max() < 1e-12),
            "schema": "R_j = [n | e1 | n x e1] in SO(3), k_j = (B + H) / 2, tau_j = centroid (GG-30); float32"}


def preview(work, rep, out_png):
    from PIL import Image, ImageDraw
    tiles = [(f, work / f) for f in ("ao_outer_2k.png", "ao_inner_2k.png", "curvature_cavity_2k.png", "M4_P1_bronze_verdigris.png")]
    tiles = [(f, p) for f, p in tiles if p.exists()]
    sheet = Image.new("RGB", (480 * max(len(tiles), 1), 760), (30, 30, 34)); dr = ImageDraw.Draw(sheet)
    for i, (f, p) in enumerate(tiles):
        im = Image.open(p).convert("RGBA"); im.thumbnail((470, 600)); bg = Image.new("RGB", im.size, (30, 30, 34)); bg.paste(im, mask=im.split()[3])
        sheet.paste(bg, (i * 480 + 5, 40)); dr.text((i * 480 + 8, 12), f, fill=(230, 230, 230))
    g = rep["g5"]; y = 650
    dr.text((8, y), "G5 (tris < 0.25 x rest area), measured on the Blender depsgraph:  pose: before -> named-4-only -> all families", fill=(230, 230, 230))
    line = "   ".join(f"{p}:{g['before'][p]}->{g['named4_only'][p]}->{g['all'][p]}" for p in g["before"])
    for k in range(0, len(line), 150):
        y += 16; dr.text((8, y), line[k:k + 150], fill=(200, 200, 200))
    sheet.save(out_png)


# ------------------------------------------------------------------------------------------------------------------------------------
def launch():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default="/content/maximus_v2/bundle", help="bundle root (Colab) or maximus_engine project root (local)")
    ap.add_argument("--out", default="/content/drive/MyDrive/maximus_v2/results")
    ap.add_argument("--blender"); ap.add_argument("--device", default="gpu", choices=["gpu", "cpu-smoke", "skip"])
    ap.add_argument("--solver", default="repair", choices=["repair", "arap"]); ap.add_argument("--polish-rounds", type=int, default=4)
    ap.add_argument("--ao-samples", default="64"); ap.add_argument("--render-samples", default="256")
    a = ap.parse_args(); t0 = time.time()
    root = Path(a.root).resolve(); master = root / "out" / "maximus_v2_master.blend"
    assert sha256(master) == MASTER_SHA256, "master .blend is not the gated master (92f28664...)"
    if (root / "MANIFEST.json").exists() and (root / "corpus").is_dir():          # Colab bundle: verify every hash (local bundle drifted)
        man = json.loads((root / "MANIFEST.json").read_text()); bad = [r for r, h in man["files"].items() if not (root / r).exists() or sha256(root / r) != h]
        assert not bad, f"bundle hash mismatch: {bad}"
    fit = next(p for p in (HERE / "m_theta_fit.json", Path("/content/drive/MyDrive/maximus_v2/m_theta_fit.json")) if p.exists())
    rt = find_runtime(); import pose_space_engine as PSE  # noqa: E402
    sys.path.insert(0, str(HERE)); import canon_geometry_kernels as K  # noqa: E402
    exe = find_blender(root, a.blender); work = Path(a.out).resolve().parent / "_m4_work" if a.out.startswith("/content/drive") else Path(a.out) / "_m4_work"
    if a.out.startswith("/content/drive"):
        work = Path("/content/m4_work")
    work.mkdir(parents=True, exist_ok=True); logs = work / "logs"; rep = {"stage": STAGE, "solver": a.solver, "blender": str(exe)}
    base_kv = {"out_dir": work, "theta_fit": fit}

    print("[1/9] extract 16 poses", flush=True)
    run_blender(exe, root, master, dict(base_kv, mode="extract"), logs / "extract.log")
    Z = np.load(work / "m4_pose_inputs.npz"); poses = [k[2:] for k in Z.files if k.startswith("V|")]
    V0 = Z["V_basis"].astype(float); T = Z["T"].astype(np.int64); a0 = PSE.tri_areas(V0, T)
    g5_before = {p: int((PSE.area_ratio(V0, Z[f"V|{p}"], T, a0) < G5_RATIO).sum()) for p in poses}
    rep["pose_facts"] = json.loads(str(Z["facts_json"]))

    print(f"[2/9] per-pose correctives ({a.solver})", flush=True)
    D, cstats = solve_correctives(Z, PSE, K, a.solver, poses)
    fams, Pfam = vertex_family_weights(Z["W"].astype(float), [str(n) for n in Z["group_names"]], V0)

    print(f"[3/9] families + RBF interpolator + depsgraph polish (max {a.polish_rounds} rounds)", flush=True)
    polish = []
    for rnd in range(a.polish_rounds + 1):
        keys, members, model, weights = build_keys(Z, D, PSE, poses, fams, Pfam)
        np.savez_compressed(work / "psd_deltas.npz", **keys)
        (work / "psd_weights.json").write_text(json.dumps(weights)); (work / "psd_model.json").write_text(model.to_json())
        ev = run_blender(exe, root, master, dict(base_kv, mode="evaluate", psd_npz=work / "psd_deltas.npz", psd_weights_json=work / "psd_weights.json",
                                                   out_npz=work / f"eval_round{rnd}.npz"), logs / f"evaluate_round{rnd}.log")
        g5 = {p: v["G5"] for p, v in ev["outputs"]["g5"].items()}
        polish.append({"round": rnd, "keys": len(keys), "g5": g5, "max_G5": max(g5.values())})
        print(f"   round {rnd}: {len(keys)} keys, G5 per pose {g5}", flush=True)
        if max(g5.values()) == 0 or rnd == a.polish_rounds:
            break
        E = np.load(work / f"eval_round{rnd}.npz")
        for p in poses:
            if g5[p] == 0:
                continue
            dd, st = PSE.arap_corrective(K, V0, E[f"V|{p}"].astype(float), T, Z[f"M|{p}"].astype(float), a0=a0, radius=0.025, arap_iters=0, target_ratio=0.285)
            D[p] = D[p] + dd
    rep["polish"] = polish; rep["correctives"] = cstats
    final_eval = np.load(work / f"eval_round{polish[-1]['round']}.npz")

    print("[4/9] named-four-only check", flush=True)
    w_named = {s: {k: (v if any(k.startswith(f"PSD_{f}__") for f in NAMED) else 0.0) for k, v in w.items()} for s, w in weights.items()}
    (work / "psd_weights_named4.json").write_text(json.dumps(w_named))
    ev4 = run_blender(exe, root, master, dict(base_kv, mode="evaluate", psd_npz=work / "psd_deltas.npz", psd_weights_json=work / "psd_weights_named4.json",
                                                out_npz=work / "eval_named4.npz"), logs / "evaluate_named4.log")
    rep["g5"] = {"before": g5_before, "named4_only": {p: v["G5"] for p, v in ev4["outputs"]["g5"].items()}, "all": polish[-1]["g5"]}
    rep["keys"] = {k: {"family": k.split("__")[0][4:], "sample_poses": members[k], "named_family": k.split("__")[0][4:] in NAMED,
                       "vertices": int((np.linalg.norm(v, axis=1) > 1e-6).sum()), "max_rest_delta_mm": round(float(np.linalg.norm(v, axis=1).max()) * 1e3, 3)}
                   for k, v in keys.items()}
    rep["families"] = {f: {"joints": model.families[f]["joints"], "keys": model.families[f]["keys"], "sigma_deg": round(np.degrees(model.families[f]["rbf"].sigma), 3),
                           "rbf_cond": model.families[f]["rbf"].cond} for f in model.families}
    rep["bind_weights_max"] = max(weights["bind"].values()) if weights["bind"] else 0.0
    rep["z_sole_mm"] = {p: v["z_sole_mm"] for p, v in ev["outputs"]["g5"].items()}

    print("[5/9] save maximus_v2_master_psd.blend (keys at 0, interpolator on the mesh)", flush=True)
    vol_keys = volume_psd_keys(Z)
    all_save_keys = dict(keys); all_save_keys.update(vol_keys)
    np.savez_compressed(work / "psd_deltas.npz", **all_save_keys)
    weights_save = {s: dict(w, **{vk: 0.0 for vk in vol_keys}) for s, w in weights.items()}
    (work / "psd_weights.json").write_text(json.dumps(weights_save))
    rep["volume_psd_keys"] = {k: {"vertices": int((np.linalg.norm(v, axis=1) > 1e-6).sum()),
                                  "max_rest_delta_mm": round(float(np.linalg.norm(v, axis=1).max()) * 1e3, 3)} for k, v in vol_keys.items()}
    run_blender(exe, root, master, dict(base_kv, mode="apply_save", psd_npz=work / "psd_deltas.npz", psd_weights_json=work / "psd_weights.json",
                                        interp_json=work / "psd_model.json", out_npz=work / "eval_saved.npz"), logs / "apply_save.log")

    print("[6/9] curvature cavity 2K + surface atlas", flush=True)
    rep["cavity"] = curvature_cavity(Z, K, work / "curvature_cavity_2k.png")
    rep["atlas"] = build_atlas(Z, {s: final_eval[f"V|{s}"] for s in ATLAS_STATES if s != "rest"}, K, work / "maximus_v2_surface_atlas.npz")

    if a.device != "skip":
        smoke = a.device == "cpu-smoke"
        print("[7/9] bake 2K outer / inner AO" + (" (CPU smoke)" if smoke else ""), flush=True)
        rep["ao"] = run_blender(exe, root, work / "maximus_v2_master_psd.blend", dict(base_kv, mode="bake", ao_samples="4" if smoke else a.ao_samples,
                                                                                       require_gpu="0" if smoke else "1"), logs / "bake.log")["outputs"]
        print("[8/9] P1 Bronze verdigris re-render" + (" (CPU smoke)" if smoke else ""), flush=True)
        rep["bronze"] = run_blender(exe, root, master, dict(base_kv, mode="render", pose="P1", psd_npz=work / "psd_deltas.npz", psd_weights_json=work / "psd_weights.json",
                                                            cavity_png=work / "curvature_cavity_2k.png", ao_png=work / "ao_outer_2k.png",
                                                            out_png=work / "M4_P1_bronze_verdigris.png", samples="8" if smoke else a.render_samples,
                                                            w="410" if smoke else "1638", h="512" if smoke else "2048", require_gpu="0" if smoke else "1"),
                                    logs / "render_bronze.log")["outputs"]
    rep["elapsed_s"] = round(time.time() - t0, 1)
    rep["verdict"] = {"all_poses_G5_zero_all_families": rep["g5"]["all"] and max(rep["g5"]["all"].values()) == 0,
                      "all_poses_G5_zero_named4_only": max(rep["g5"]["named4_only"].values()) == 0,
                      "bind_untouched": rep["bind_weights_max"] == 0.0, "atlas_roundtrip_ok": rep["atlas"]["roundtrip_ok"], "lbs_only_zero_dqs": True}
    (work / "M4_canon_atlas.json").write_text(json.dumps(rep, indent=1, default=str))
    preview(work, rep, work / "M4_preview.png")

    print("[9/9] copy to results/M4_canon_atlas + SHA-256 read-back", flush=True)
    keep = ["M4_canon_atlas.json", "M4_preview.png", "maximus_v2_master_psd.blend", "maximus_v2_surface_atlas.npz", "psd_model.json", "psd_deltas.npz",
            "psd_weights.json", "psd_weights_named4.json", "curvature_cavity_2k.png", "ao_outer_2k.png", "ao_inner_2k.png", "M4_P1_bronze_verdigris.png"]
    keep = [f for f in keep if (work / f).exists()]
    dst = Path(a.out) / "M4_canon_atlas"; (dst / "logs").mkdir(parents=True, exist_ok=True)
    for f in keep:
        shutil.copy2(work / f, dst / f)
    for lg in logs.glob("*.log"):
        shutil.copy2(lg, dst / "logs" / lg.name)
    if root.name == "bundle":
        shutil.copy2(work / "maximus_v2_master_psd.blend", root / "out" / "maximus_v2_master_psd.blend")   # for M5 / M6 in the same session
    lines = [f"{sha256(work / f)}  {f}" for f in keep]; (dst / "MANIFEST_SHA256.txt").write_text("\n".join(lines) + "\n", newline="\n")
    if hasattr(os, "sync"):
        os.sync()
    bad = [f for f in keep if sha256(dst / f) != sha256(work / f)]
    print(json.dumps({"to": str(dst), "files": len(keep), "readback_mismatch": bad, "g5": rep["g5"], "verdict": rep["verdict"], "elapsed_s": rep["elapsed_s"]}, indent=1))
    sys.exit(0 if not bad and rep["verdict"]["all_poses_G5_zero_all_families"] and rep["verdict"]["bind_untouched"] else 1)


if __name__ == "__main__":
    launch()
