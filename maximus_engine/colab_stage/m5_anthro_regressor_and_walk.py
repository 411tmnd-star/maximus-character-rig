"""M5 (v2, 2026-11-19) - CUDA winding-number parity, ISO 8559-1 anthropometric regressor, runway walk against the REAL S1-S9 harness,
CUDA self-penetration count, 90-frame Living Statue loop.

    Colab:  !python /content/drive/MyDrive/maximus_v2/m5_anthro_regressor_and_walk.py
    Local:  python -I colab_stage/m5_anthro_regressor_and_walk.py --root <maximus_engine> --master-psd <psd.blend> --harness <avatar_engine/harness>
                   --prototypes <02_engine_3d/prototypes> --out <dir> --device skip --allow-no-cuda

Replaces the 2026-11-19 Batch-1 draft (kept in _superseded/), which never ran the real harness: its "S1" and "S8" read the spec constants
back, its S3 window was widened to 0.01-0.15, S6 was missing, its walk ignored theta_fit (it expected lists where the file holds q_wxyz
dicts), walked in place, locked the clavicles (S5 gates them) and animated one toe of five, and nothing was rendered.

Order (fail-fast first):
  1. CUDA vs CPU float64 parity of canon_geometry_kernels.generalized_winding_number_float64 on the Maximus body: stop if max |dw| > 1e-9
     or any inside/outside classification differs.
  2. 32 corners of the 5 anthropometric keys at bind; girths are TAPE girths (convex hull of the contour loop that encloses the body axis,
     as a tape bridges concavities) found by search in ISO 8559-1 style bands (neck min, chest max, waist min, hip max, left thigh max);
     ridge regressor both ways; R^2 and max tape residual (mm) reported.
  3. Walk solve (m5_gait + m5_blender_pass mode=walk) on maximus_v2_master_psd.blend with the PSD keys driven per frame.
  4. avatar_engine/harness/capture_walk_blender.py and gates.py, UNCHANGED, staged in their expected folder layout; --no-clearance (the
     mannequin walks alone, there is no garment). Every gate result is reported as the harness printed it.
  5. Self-penetration: every body vertex, pushed 0.5 mm inward, has winding number ~1 against the whole closed body; ~2 means it is inside
     another body part. Count (w > 1.5) per captured frame, CUDA float64.
  6. 90-frame 24 fps loop (+0.8 deg spine02), GPU Cycles RGBA -> AVIF (ffmpeg libaom, over the M2 backdrop), animated WebP (RGBA), PNG/WebP stills.
"""
import argparse, hashlib, json, math, os, shutil, subprocess, sys, time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
STAGE = "m5_kinematics"
METRICS = ["stature_cm", "neck_girth_cm", "chest_girth_cm", "waist_girth_cm", "hip_girth_cm", "thigh_girth_cm"]
ARM_HAND_PREFIXES = ("upperarm", "lowerarm", "wrist", "finger", "metacarpal")
ARM_WEIGHT_THRESHOLD = 0.25
PELVIS_GROUPS = ("root", "pelvis.L", "pelvis.R")
# (lo, hi, min|max, part, max x-span m, mask_arms): arm/hand/finger vertices (weight > 0.25) are masked out before slicing chest/waist/hip/thigh
BANDS = {
    "neck_girth_cm":  (0.830, 0.875, "min", "torso",    0.22, False),
    "chest_girth_cm": (0.660, 0.740, "max", "torso",    0.65, True),
    "waist_girth_cm": (0.570, 0.650, "min", "torso",    0.65, True),
    "hip_girth_cm":   (0.460, 0.540, "max", "torso",    0.75, True),
    "thigh_girth_cm": (0.400, 0.465, "max", "left_leg", 0.35, True),
}


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def kernels():
    sys.path.insert(0, str(HERE)); import canon_geometry_kernels as K  # noqa: E402
    return K


# ------------------------------------------------------------------------------------------------------------------------------------
# 1. CUDA parity
# ------------------------------------------------------------------------------------------------------------------------------------
def cuda_parity(K, V, F, n_each=1500, seed=7):
    try:
        import torch
        has = torch.cuda.is_available()
    except Exception:  # noqa: BLE001
        has = False
    if not has:
        return {"tested": False, "reason": "no CUDA device (torch.cuda.is_available() is False)"}
    rng = np.random.default_rng(seed)
    vn = np.zeros_like(V); fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    for c in range(3):
        np.add.at(vn, F[:, c], fn)
    vn /= np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-12)
    pick = rng.choice(len(V), n_each, replace=False)
    Q = np.vstack([V[pick] - 5e-4 * vn[pick], V[pick] + 5e-4 * vn[pick], V.min(0) + rng.random((n_each, 3)) * (V.max(0) - V.min(0))])
    t0 = time.time(); w_cpu = K.generalized_winding_number_float64(Q, V, F, device="cpu"); t_cpu = time.time() - t0
    t0 = time.time(); w_gpu = K.generalized_winding_number_float64(Q, V, F, device="cuda"); t_gpu = time.time() - t0
    diff = float(np.abs(w_cpu - w_gpu).max()); mism = int(((w_cpu > 0.5) != (w_gpu > 0.5)).sum())
    return {"tested": True, "device": torch.cuda.get_device_name(0), "queries": int(len(Q)), "triangles": int(len(F)), "max_abs_diff": diff,
            "inside_classification_mismatches": mism, "inside_count": int((w_cpu > 0.5).sum()), "cpu_s": round(t_cpu, 2), "gpu_s": round(t_gpu, 2),
            "pass": diff <= 1e-9 and mism == 0, "limits": {"max_abs_diff": 1e-9, "mismatches": 0}}


# ------------------------------------------------------------------------------------------------------------------------------------
# 2. anthropometry
# ------------------------------------------------------------------------------------------------------------------------------------
def arm_hand_vertex_mask(W, group_names, threshold=ARM_WEIGHT_THRESHOLD, prefixes=ARM_HAND_PREFIXES):
    """Boolean (N,) mask of arm/hand/finger vertices (sum of weights across matching vertex groups > threshold)."""
    if W is None or group_names is None:
        return None
    cols = [i for i, n in enumerate(group_names) if str(n).startswith(prefixes)]
    if not cols:
        return np.zeros(len(W), dtype=bool)
    return W[:, cols].sum(axis=1) > float(threshold)


def pelvis_vertex_mask(W, group_names, threshold=0.25):
    if W is None or group_names is None:
        return None
    cols = [i for i, n in enumerate(group_names) if str(n) in PELVIS_GROUPS]
    if not cols:
        return None
    m = W[:, cols].sum(axis=1) > float(threshold)
    return m if m.any() else None


def slice_loops(V, F, z, exclude_vertex_mask=None):
    """Contour loops of the plane z: points keyed by the mesh edge they lie on, so loop connectivity is exact. Returns [ (k,2) xy arrays ].
    If `exclude_vertex_mask` is provided, triangles containing any masked vertex (e.g. arms/hands/fingers) are excluded before slicing."""
    if exclude_vertex_mask is not None:
        F = F[~exclude_vertex_mask[F].any(axis=1)]
    above = V[:, 2] >= z; fa = above[F]; cut = fa.any(1) & ~fa.all(1)
    E = {}; adj = {}
    for tri in F[cut]:
        a = above[tri]; pts = []
        for i in range(3):
            u, v = tri[i], tri[(i + 1) % 3]
            if a[i] != a[(i + 1) % 3]:
                key = (min(u, v), max(u, v))
                if key not in E:
                    p, q = V[key[0]], V[key[1]]; t = (z - p[2]) / (q[2] - p[2]); E[key] = (p + t * (q - p))[:2]
                pts.append(key)
        if len(pts) == 2:
            adj.setdefault(pts[0], []).append(pts[1]); adj.setdefault(pts[1], []).append(pts[0])
    loops, seen = [], set()
    for start in adj:
        if start in seen:
            continue
        loop = [start]; seen.add(start); prev, cur = None, start
        while True:
            nxt = [x for x in adj[cur] if x != prev and x not in seen]
            if not nxt:
                break
            prev, cur = cur, nxt[0]; loop.append(cur); seen.add(cur)
        if len(loop) >= 3:
            loops.append(np.array([E[k] for k in loop]))
    return loops


def loop_stats(P):
    from scipy.spatial import ConvexHull
    per = float(np.linalg.norm(np.roll(P, -1, 0) - P, axis=1).sum()); x, y = P[:, 0], P[:, 1]
    area = 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))
    h = ConvexHull(P); hull = P[h.vertices]; tape = float(np.linalg.norm(np.roll(hull, -1, 0) - hull, axis=1).sum())
    return {"perimeter": per, "tape": tape, "area": area, "centroid": P.mean(0), "xspan": float(np.ptp(x))}


def contains(P, pt):
    x, y = P[:, 0], P[:, 1]; x2, y2 = np.roll(x, -1), np.roll(y, -1)
    cond = ((y > pt[1]) != (y2 > pt[1])) & (pt[0] < (x2 - x) * (pt[1] - y) / np.where(y2 != y, y2 - y, 1e-30) + x)
    return bool(cond.sum() % 2)


def measure(V, F, n_samples=15, arm_mask=None, pelvis_mask=None):
    z0, z1 = float(V[:, 2].min()), float(V[:, 2].max()); H = z1 - z0; out = {"stature_cm": H * 100}; detail = {}
    x_pelvis = float(V[pelvis_mask, 0].mean()) if (pelvis_mask is not None and pelvis_mask.any()) else 0.0
    for m, (lo, hi, mode, part, xmax0, mask_arms) in BANDS.items():
        best = None
        ex = arm_mask if mask_arms else None
        for relax in (1.0, 1.2, 1.44, 1.728, 2.074):           # widen the x-span limit only for a corner that needs it; recorded
            xmax = xmax0 * relax
            for frac in np.linspace(lo, hi, n_samples):
                loops = [(P, loop_stats(P)) for P in slice_loops(V, F, z0 + frac * H, exclude_vertex_mask=ex)]
                if part == "torso":
                    cand = [(P, s) for P, s in loops
                            if abs(float(s["centroid"][0]) - x_pelvis) < 0.08 and s["area"] > 1e-3 and s["xspan"] <= xmax]
                else:
                    cand = [(P, s) for P, s in loops
                            if (float(s["centroid"][0]) - x_pelvis) > 0.02 and s["area"] > 1e-3 and s["xspan"] <= xmax]
                if not cand:
                    continue
                P, s = max(cand, key=lambda c: c[1]["area"])
                if best is None or (s["tape"] > best[1]["tape"] if mode == "max" else s["tape"] < best[1]["tape"]):
                    best = (frac, s, len(loops), xmax)
            if best is not None:
                break
        if best is None:
            raise RuntimeError(f"{m}: no contour loop found in band {lo}-{hi} H even with a {xmax0 * 2.074:.2f} m x-span limit")
        frac, s, nl, xm = best
        out[m] = s["tape"] * 100
        detail[m] = {"height_frac": round(float(frac), 4), "tape_cm": round(s["tape"] * 100, 3), "mesh_perimeter_cm": round(s["perimeter"] * 100, 3),
                     "x_span_cm": round(s["xspan"] * 100, 2), "loops_in_slice": nl, "xspan_limit_m": round(xm, 3),
                     "arm_masked": bool(ex is not None), "x_pelvis_m": round(x_pelvis, 5)}
    return out, detail


def fit_regressor(W, X, keys, metrics, lam=1e-3):
    mu, sd = X.mean(0), np.where(X.std(0) > 1e-9, X.std(0), 1.0); Xs = (X - mu) / sd; Xa = np.c_[Xs, np.ones(len(X))]
    reg = lam * np.eye(Xa.shape[1]); reg[-1, -1] = 0
    inv = np.linalg.solve(Xa.T @ Xa + reg, Xa.T @ W); W_pred = np.clip(Xa @ inv, 0, 1)
    Wa = np.c_[W, np.ones(len(W))]; fwd, *_ = np.linalg.lstsq(Wa, X, rcond=None); X_pred = Wa @ fwd
    ss_res = ((X_pred - X) ** 2).sum(0); ss_tot = ((X - X.mean(0)) ** 2).sum(0)
    r2 = np.where(ss_tot > 1e-12, 1 - ss_res / np.where(ss_tot > 1e-12, ss_tot, 1), 1.0)
    rt = np.c_[W_pred, np.ones(len(W))] @ fwd                                  # measurements -> weights -> measurements
    return {"schema": "maximus.v2.anthro_regressor/2", "standard": "ISO 8559-1 (tape girths; landmarks approximated by band search on the mesh)",
            "corners": int(len(W)), "key_names": keys, "metric_names": metrics, "ridge_lambda": lam,
            "input_mean_cm": mu.tolist(), "input_std_cm": sd.tolist(), "inverse_weights": inv[:-1].tolist(), "inverse_bias": inv[-1].tolist(),
            "forward_weights_cm": fwd[:-1].tolist(), "forward_bias_cm": fwd[-1].tolist(),
            "fit": {"forward_R2": {m: round(float(v), 6) for m, v in zip(metrics, r2)},
                    "forward_max_tape_residual_mm": {m: round(float(v) * 10, 3) for m, v in zip(metrics, np.abs(X_pred - X).max(0))},
                    "forward_max_tape_residual_mm_overall": round(float(np.abs(X_pred - X)[:, 1:].max()) * 10, 3),
                    "inverse_rmse_per_key": {k: round(float(v), 5) for k, v in zip(keys, np.sqrt(((W_pred - W) ** 2).mean(0)))},
                    "roundtrip_max_residual_mm": round(float(np.abs(rt - X)[:, 1:].max()) * 10, 3)}}


# ------------------------------------------------------------------------------------------------------------------------------------
# Blender / harness runners
# ------------------------------------------------------------------------------------------------------------------------------------
def blender_exe(explicit):
    return Path(explicit) if explicit else next(Path("/content/blender").glob("blender-*/blender"))


def run_pass(exe, blend, root, kv, log, corpus=None):
    env = dict(os.environ, MAXIMUS_STRICT="1")
    if corpus:
        env["MAXIMUS_CORPUS"] = str(corpus)
    cmd = [str(exe), "--background", str(blend), "--factory-startup", "--python-exit-code", "3", "--python", str(HERE / "m5_blender_pass.py"), "--", f"root={root}"]
    cmd += [f"{k}={v}" for k, v in kv.items()]
    p = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True); buf = []
    for line in p.stdout:
        buf.append(line)
        if line.startswith("PROGRESS"):
            print("   " + line.rstrip(), flush=True)
    p.wait(); log.write_text("".join(buf), encoding="utf-8")
    env_l = [l for l in buf if l.startswith("EC02 ")]
    rep = json.loads(env_l[-1][5:]) if env_l else {"status": "fail", "errors": ["".join(buf[-60:])]}
    if rep.get("status") != "ok":
        print(json.dumps(rep.get("errors"), indent=1)[:6000]); raise SystemExit(f"m5 blender pass {kv.get('mode')} failed")
    return rep


def stage_harness(harness_dir, prototypes_dir, stage):
    """Copy the harness UNCHANGED into the folder layout gates.py expects: WS = harness/../../.. ; WS/solvers/prototypes/P3_pose_swing_twist."""
    ws = stage / "better_style_engine"; h = ws / "02_engine_3d" / "avatar_engine" / "harness"; h.mkdir(parents=True, exist_ok=True)
    for f in ("capture_walk_blender.py", "gates.py", "gate_limits.json"):
        shutil.copy2(Path(harness_dir) / f, h / f)
    for sub, f in (("P3_pose_swing_twist", "swing_twist.py"), ("P2_clearance_field", "clearance_field.py")):
        d = ws / "solvers" / "prototypes" / sub; d.mkdir(parents=True, exist_ok=True); shutil.copy2(Path(prototypes_dir) / sub / f, d / f)
    hashes = {f: sha256(h / f) for f in ("capture_walk_blender.py", "gates.py", "gate_limits.json")}
    return h, hashes


def self_penetration(K, cap, frames_step, device, W=None, group_names=None):
    Z = np.load(cap); B = Z["body"].astype(np.float64); F = Z["tri"].astype(np.int64); frames = Z["frames"]; out = []
    sel = list(range(0, len(B), frames_step)); t0 = time.time()
    all_hit_verts = {}

    def top_weights(vi):
        if W is None or group_names is None:
            return []
        row = W[vi]; order = np.argsort(-row)[:4]
        return [{str(group_names[c]): round(float(row[c]), 4)} for c in order if row[c] > 0.01]

    for k in sel:
        V = B[k]; fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]); vn = np.zeros_like(V)
        for c in range(3):
            np.add.at(vn, F[:, c], fn)
        vn /= np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-12)
        w = K.generalized_winding_number_float64(V - 5e-4 * vn, V, F, device=device)
        hits = np.nonzero(w > 1.5)[0]
        rec = {"frame": int(frames[k]), "inside_other_part": int(len(hits)), "w_max": round(float(w.max()), 4)}
        if len(hits):
            rec["vertex_indices"] = [int(i) for i in hits[:64]]
            for vi in hits:
                vii = int(vi)
                if vii not in all_hit_verts or float(w[vi]) > all_hit_verts[vii]["w_max"]:
                    all_hit_verts[vii] = {"vertex": vii, "w_max": round(float(w[vi]), 4), "sample_frame": int(frames[k]),
                                          "pos_m": [round(float(x), 4) for x in V[vi]], "weights": top_weights(vii)}
        out.append(rec)
    worst = max(out, key=lambda r: r["inside_other_part"])
    return {"device": device, "frames_checked": len(sel), "total_frames": int(len(B)), "frame_step": frames_step,
            "max_penetrating_vertices": worst["inside_other_part"], "worst_frame": worst["frame"],
            "frames_with_penetration": sum(1 for r in out if r["inside_other_part"] > 0),
            "unique_penetrating_vertices": [all_hit_verts[k] for k in sorted(all_hit_verts)],
            "rule": "body vertex moved 0.5 mm inward has generalised winding number ~1; > 1.5 means it lies inside another body part",
            "seconds": round(time.time() - t0, 1), "per_frame": out}


def encode_loop(frames_dir, out_dir, backdrop=("#34322f", "#151413")):
    from PIL import Image
    files = sorted(Path(frames_dir).glob("frame_*.png")); imgs = [Image.open(f).convert("RGBA") for f in files]; res = {}
    W, H = imgs[0].size; c0 = np.array([int(backdrop[0][i:i + 2], 16) for i in (1, 3, 5)], float); c1 = np.array([int(backdrop[1][i:i + 2], 16) for i in (1, 3, 5)], float)
    yy, xx = np.mgrid[0:H, 0:W]; r = np.clip(np.hypot((xx - W / 2) / (W * 0.7), (yy - H * 0.45) / (H * 0.7)), 0, 1)[..., None]
    bg = (c0 * (1 - r) + c1 * r).astype(np.uint8); flat = out_dir / "_flat"; flat.mkdir(exist_ok=True)
    for i, im in enumerate(imgs):
        comp = Image.alpha_composite(Image.fromarray(np.dstack([bg, np.full((H, W), 255, np.uint8)])), im); comp.convert("RGB").save(flat / f"f_{i:03d}.png")
    imgs[0].save(out_dir / "maximus_v2_breath_still.png"); imgs[0].save(out_dir / "maximus_v2_breath_still.webp", quality=90)
    imgs[0].save(out_dir / "maximus_v2_breath_loop.webp", save_all=True, append_images=imgs[1:], duration=int(1000 / 24), loop=0, quality=80, method=4)
    res["webp_loop_bytes"] = (out_dir / "maximus_v2_breath_loop.webp").stat().st_size
    for enc in ("libaom-av1", "libsvtav1"):
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-framerate", "24", "-i", str(flat / "f_%03d.png"), "-c:v", enc, "-crf", "34", "-b:v", "0", "-pix_fmt", "yuv420p"]
        cmd += (["-cpu-used", "6", "-row-mt", "1"] if enc == "libaom-av1" else ["-preset", "8"]) + [str(out_dir / "maximus_v2_breath_loop.avif")]
        p = subprocess.run(cmd, capture_output=True, text=True)
        if p.returncode == 0 and (out_dir / "maximus_v2_breath_loop.avif").exists():
            res["avif_encoder"] = enc; res["avif_bytes"] = (out_dir / "maximus_v2_breath_loop.avif").stat().st_size; break
        res[f"avif_{enc}_error"] = p.stderr[-300:]
    shutil.rmtree(flat, ignore_errors=True)
    return res


# ------------------------------------------------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default="/content/maximus_v2/bundle"); ap.add_argument("--master-psd", default=None)
    ap.add_argument("--harness", default="/content/drive/MyDrive/maximus_v2/harness"); ap.add_argument("--prototypes", default="/content/drive/MyDrive/maximus_v2/prototypes")
    ap.add_argument("--out", default="/content/drive/MyDrive/maximus_v2/results"); ap.add_argument("--blender")
    ap.add_argument("--device", default="gpu", choices=["gpu", "cpu-smoke", "skip"]); ap.add_argument("--allow-no-cuda", action="store_true")
    ap.add_argument("--pen-step", type=int, default=1); ap.add_argument("--breath-samples", default="96")
    ap.add_argument("--preset", default="BASELINE_M5", choices=["BASELINE_M5", "RUNWAY_CONFIDENT_V2"])
    a = ap.parse_args(); t0 = time.time(); root = Path(a.root).resolve(); K = kernels(); rep = {"stage": STAGE, "preset": a.preset}
    master = Path(a.master_psd) if a.master_psd else root / "out" / "maximus_v2_master_psd.blend"
    corpus = root / "corpus" if (root / "corpus").is_dir() else None
    fit = next(p for p in (HERE / "m_theta_fit.json", Path("/content/drive/MyDrive/maximus_v2/m_theta_fit.json")) if p.exists())
    work = Path("/content/m5_work") if a.out.startswith("/content/drive") else Path(a.out) / "_m5_work"; work.mkdir(parents=True, exist_ok=True)
    exe = blender_exe(a.blender); kv = {"out_dir": work, "theta_fit": fit}

    print("[1/6] corners (needed for the parity mesh too)", flush=True)
    run_pass(exe, master, root, dict(kv, mode="corners"), work / "corners.log", corpus)
    Zc = np.load(work / "m5_corners.npz"); F = Zc["T"].astype(np.int64)
    print("[1/6] CUDA vs CPU float64 winding-number parity", flush=True)
    rep["cuda_gwn_parity"] = cuda_parity(K, Zc["V"][0], F); print("   ", {k: v for k, v in rep["cuda_gwn_parity"].items() if k != "limits"}, flush=True)
    if rep["cuda_gwn_parity"].get("tested") and not rep["cuda_gwn_parity"]["pass"]:
        (work / "M5_kinematics.json").write_text(json.dumps(rep, indent=1)); raise SystemExit("FAIL-FAST: CUDA/CPU winding-number parity failed")
    if not rep["cuda_gwn_parity"].get("tested") and not a.allow_no_cuda:
        raise SystemExit("FAIL-FAST: no CUDA device for the parity test (pass --allow-no-cuda only for a local dry run)")

    print("[2/6] ISO 8559-1 girths on 32 corners + ridge regressor", flush=True)
    W_mesh = Zc["W"].astype(np.float64) if "W" in Zc.files else None
    gnames = [str(n) for n in Zc["group_names"]] if "group_names" in Zc.files else None
    arm_mask = arm_hand_vertex_mask(W_mesh, gnames)
    pel_mask = pelvis_vertex_mask(W_mesh, gnames)
    meas = []; details = []
    for ci in range(32):
        m, d = measure(Zc["V"][ci], F, arm_mask=arm_mask, pelvis_mask=pel_mask); meas.append([m[k] for k in METRICS]); details.append(d)
    X = np.array(meas); reg = fit_regressor(Zc["corners"], X, [str(k) for k in Zc["key_names"]], METRICS)
    reg["corner_measurements_cm"] = np.round(X, 3).tolist(); reg["basis_corner_landmarks"] = details[0]
    (work / "MAXIMUS_V2_ANTHRO_REGRESSOR.json").write_text(json.dumps(reg, indent=1))
    rep["anthro"] = {"R2": reg["fit"]["forward_R2"], "max_tape_residual_mm": reg["fit"]["forward_max_tape_residual_mm_overall"],
                     "arm_masked_vertices": int(arm_mask.sum()) if arm_mask is not None else 0,
                     "hip_girth_cm_range": [round(float(X[:, 4].min()), 3), round(float(X[:, 4].max()), 3)],
                     "basis_cm": {k: round(v, 2) for k, v in zip(METRICS, X[0])}, "landmarks_basis": details[0]}
    print("   ", rep["anthro"]["basis_cm"], "R2", rep["anthro"]["R2"], "max tape residual mm", rep["anthro"]["max_tape_residual_mm"], flush=True)

    print(f"[3/6] walk solve on the 13,380-vertex master (preset={a.preset}, PSD keys driven per frame)", flush=True)
    rep["walk_solve"] = run_pass(exe, master, root, dict(kv, mode="walk", preset=a.preset), work / "walk.log", corpus)["metrics"]

    print("[4/6] REAL harness: capture_walk_blender.py + gates.py (staged)", flush=True)
    hdir, hh = stage_harness(a.harness, a.prototypes, work / "harness_stage"); rep["harness_sha256"] = hh
    cap = work / "m5_walk_capture.npz"
    p = subprocess.run([str(exe), "-b", "--factory-startup", "--python-exit-code", "1", "-P", str(hdir / "capture_walk_blender.py"), "--",
                        str(work / "m5_walk_scene.blend"), str(cap)], capture_output=True, text=True)
    (work / "capture.log").write_text(p.stdout + p.stderr, encoding="utf-8")
    if p.returncode != 0 or not cap.exists():
        raise SystemExit(f"capture failed: {p.stderr[-2000:]}")
    gdir = work / "gates"
    gates_cmd = [sys.executable, "-B", str(hdir / "gates.py"), str(cap), str(gdir), "--no-clearance", "--preset", a.preset]
    p = subprocess.run(gates_cmd, capture_output=True, text=True)
    (work / "gates.log").write_text(p.stdout + p.stderr, encoding="utf-8")
    if p.returncode != 0:
        raise SystemExit(f"gates.py failed: {p.stderr[-3000:]}")
    G = json.loads((gdir / "gate_report.json").read_text()); rep["gates"] = {"summary": G["summary"], "counts": G["counts"],
                                                                             "targets_measured": G["facts"].get("targets_measured"), "stride": G["facts"].get("stride")}
    rep["gates"]["S1_to_S9"] = {k: v for k, v in G["summary"].items() if k.startswith("S")}
    s9_rec = next((r for r in G["results"] if r["gate"] == "S9.smooth"), None)
    if s9_rec:
        over = s9_rec["value"].get("over_limit", [])
        stance_joints = [j for j in over if j.startswith(("toe", "foot", "lowerleg", "upperleg"))]
        upper_over = [j for j in over if j not in stance_joints]
        rep["gates"]["S9_annotation"] = {
            "upper_body_all_pass": len(upper_over) == 0,
            "upper_body_over_limit": upper_over,
            "stationary_stance_phase_joints": stance_joints,
            "note": "Stance-phase foot/toe/leg joints stay motionless on the flat floor for >50% of the stride, driving median angular jerk to ~0 (zero-denominator ratio spike); do NOT deform toes to chase a zero-denominator jerk metric.",
        }
    print("   ", json.dumps(G["summary"]), flush=True)

    print("[5/6] self-penetration (winding number > 1.5) on the captured frames", flush=True)
    try:
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # noqa: BLE001
        dev = "cpu"
    if dev != "cuda" and not a.allow_no_cuda:
        raise SystemExit("FAIL-FAST: self-penetration check requires device='cuda' (pass --allow-no-cuda only for a local dry run)")
    step_used = a.pen_step if dev == "cuda" else max(a.pen_step, 50)
    rep["self_penetration"] = self_penetration(K, cap, step_used, dev, W=W_mesh, group_names=gnames)
    print("   ", {k: v for k, v in rep["self_penetration"].items() if k != "per_frame"}, flush=True)

    if a.device != "skip":
        smoke = a.device == "cpu-smoke"
        print("[6/6] 90-frame Living Statue loop" + (" (CPU smoke, 8 frames)" if smoke else ""), flush=True)
        rep["breath"] = run_pass(exe, master, root, dict(kv, mode="breath", frames="8" if smoke else "90", samples="4" if smoke else a.breath_samples,
                                                         w="240" if smoke else "960", h="300" if smoke else "1200", device="cpu" if smoke else "gpu",
                                                         require_gpu="0" if smoke else "1"), work / "breath.log", corpus)["metrics"]
        rep["breath"]["encode"] = encode_loop(work / "breath", work)
    rep["elapsed_s"] = round(time.time() - t0, 1)
    s = G["summary"]
    rep["verdict"] = {"cuda_parity_pass": rep["cuda_gwn_parity"].get("pass"), "S1_S9_all_pass": all(v == "PASS" for k, v in s.items() if k.startswith("S") and v != "NOT_APPLICABLE"),
                      "base_gates_pass": {k: v for k, v in s.items() if k.startswith("base")}, "z_sole_ge_0": rep["walk_solve"]["min_body_z_mm_over_clip"] >= 0.0,
                      "zero_self_penetration": rep["self_penetration"]["max_penetrating_vertices"] == 0}
    (work / "M5_kinematics.json").write_text(json.dumps(rep, indent=1, default=str))
    keep = ["M5_kinematics.json", "MAXIMUS_V2_ANTHRO_REGRESSOR.json", "m5_walk_capture.npz", "m5_walk_solve.json", "m5_walk_scene.blend", "gates/gate_report.json",
            "gates/section2_check.json", "maximus_v2_breath_loop.avif", "maximus_v2_breath_loop.webp", "maximus_v2_breath_still.png", "maximus_v2_breath_still.webp",
            "breath.json", "walk.log", "capture.log", "gates.log", "corners.log", "breath.log"]
    keep = [f for f in keep if (work / f).exists()]
    dst = Path(a.out) / "M5_kinematics"; (dst / "gates").mkdir(parents=True, exist_ok=True)
    for f in keep:
        shutil.copy2(work / f, dst / f)
    (dst / "MANIFEST_SHA256.txt").write_text("".join(f"{sha256(work / f)}  {f}\n" for f in keep), newline="\n")
    if hasattr(os, "sync"):
        os.sync()
    bad = [f for f in keep if sha256(dst / f) != sha256(work / f)]
    print(json.dumps({"to": str(dst), "readback_mismatch": bad, "verdict": rep["verdict"], "gates": s, "anthro": {"R2": rep["anthro"]["R2"], "max_mm": rep["anthro"]["max_tape_residual_mm"]},
                      "elapsed_s": rep["elapsed_s"]}, indent=1))


if __name__ == "__main__":
    main()
