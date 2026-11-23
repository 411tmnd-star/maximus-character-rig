"""M1 - multi-angle and 360-degree structural audit of maximus_v2 (Colab L4, Cycles GPU + OIDN).

One file, two roles, so it can be started from a plain Colab cell right after M0:

    !python /content/drive/MyDrive/maximus_v2/m1_turntable_audit.py

  * system Python (the cell): verifies the staged bundle's MANIFEST, finds the Blender that M0 installed
    (/content/blender, or downloads it through run_batch_4k.blender()), starts Blender on the master .blend with THIS file as
    the stage, then builds the contact sheet / montages with PIL and copies everything to --out, hashing each copy.
  * Blender (--background ... --python this_file -- key=value): loads the built maximus_v2_master.blend, poses P1 (poses/
    theta_monolith.json) and theta_fit (m_theta_fit.json, grounded exactly like s02), renders Front / Three-quarter (45 deg) /
    Side (90 deg) / Back (180 deg) stills at 1638x2048 and a 36-frame turntable of P1 at 1024x1280, 64 samples, OIDN, and runs
    the silhouette checks below on the rendered alpha. Renders go through io_ops_gateway only (no bpy.ops anywhere else).

Camera and light rig: the calibrated s05 rig is reused verbatim (s05_previews.py is exec'd with its trailing C.run() stripped,
so the 105 mm lens, the three area lights and the per-finish exposure cannot drift from the 4K batch). Camera and lights orbit the
body together, so every angle is lit the same way; the camera distance is one constant for all angles (no zoom wobble), full
body framing (crown to sole) because the audit needs the whole silhouette, not the storefront bust crop.

Checks (all measured on the alpha channel, threshold 0.5):
  front_mirror / back_mirror   IoU of the silhouette with its own left-right mirror, maximised over a horizontal shift (the shift is
                               the offset of the body axis from the frame centre, reported in px)
  front_vs_back                IoU of the front silhouette with the mirrored back silhouette (same body, opposite camera)
  profile_LR                   IoU of the 90 deg silhouette with the mirrored 270 deg one (270 deg is rendered at 16 samples,
                               check-only, saved under checks/)
  turntable_mirror_pairs       the same IoU for frame k against frame 36-k, k = 1..17 (P1 only)
  neck_back / neck_side        neck-seam continuity: over the image rows spanned by the neck-dominated vertices (+pad), the left
                               and right silhouette edge x(row); max |second difference| and max |step| in px. A seam crease or a
                               step at the head/neck blend band shows up as a kink the smooth neck does not have.
THRESHOLDS BELOW ARE PROVISIONAL (written by the script author, not owner thresholds). Every number is reported next to its
threshold and flagged provisional; the audit never edits a threshold to pass. --strict exits 1 if any check fails.
"""
import argparse, hashlib, importlib.util, json, math, os, shutil, subprocess, sys, time
from pathlib import Path

import numpy as np

try:
    import bpy  # noqa: F401
    IN_BLENDER = True
except ImportError:
    IN_BLENDER = False

STAGE = "m1_turntable_audit"
HERE = Path(__file__).resolve().parent
VIEWS = {"front": 0.0, "three_quarter": 45.0, "side": 90.0, "back": 180.0}
CHECK_VIEWS = {"side_R_check": 270.0}
THRESH = {"mirror_iou_min": 0.97, "neck_kink_px_max": 3.0, "neck_step_px_max": 4.0, "provisional": True,
          "note": "author-chosen starting values, not owner thresholds"}
V1_SHA256 = "1be2d4ca305dca0a197215f6adc8181a2fe9853b03f5d316e77169e3aee82148"   # rig theta_fit was authored on (bl_common.V1_SHA256)


# ----------------------------------------------------------------------------------------------------------------------
# pure NumPy checks (unit-testable anywhere)
# ----------------------------------------------------------------------------------------------------------------------
def best_mirror_iou(a, b, search=48):
    """IoU between mask a and the left-right mirror of mask b, maximised over a horizontal shift in [-search, search] px.
    a is b for a self-symmetry test. Returns (iou, shift_px); the mirror axis sits shift/2 px from the frame centre."""
    a = np.asarray(a, bool); b = np.asarray(b, bool)
    rows = np.flatnonzero(a.any(1) | b.any(1))
    if rows.size == 0:
        return 0.0, 0
    a = a[rows[0]:rows[-1] + 1]; bm = b[rows[0]:rows[-1] + 1, ::-1]
    area_a = int(np.count_nonzero(a)); area_b = int(np.count_nonzero(bm)); best = (0.0, 0)
    for s in range(-search, search + 1):
        inter = int(np.count_nonzero(a & np.roll(bm, s, axis=1)))
        iou = inter / max(area_a + area_b - inter, 1)
        if iou > best[0]:
            best = (iou, s)
    return float(best[0]), int(best[1])


def neck_kinks(mask, r0, r1, pad=6):
    """Edge-continuity statistics of a silhouette over image rows r0..r1 (+-pad). x_left(row) / x_right(row) are the first and last
    covered columns; kink = max |x[i-1] - 2 x[i] + x[i+1]|, step = max |x[i+1] - x[i]|. Pixel staircase noise on a smooth slanted
    edge is at most ~2 px of second difference, so a kink above ~3 px is geometry, not quantisation."""
    mask = np.asarray(mask, bool); h = mask.shape[0]
    lo, hi = max(0, int(math.floor(r0)) - pad), min(h - 1, int(math.ceil(r1)) + pad)
    xl, xr, gaps = [], [], 0
    for r in range(lo, hi + 1):
        cols = np.flatnonzero(mask[r])
        if cols.size == 0:
            gaps += 1; continue
        xl.append(int(cols[0])); xr.append(int(cols[-1]))
    xl, xr = np.array(xl), np.array(xr)
    if len(xl) < 3:
        return {"rows": [lo, hi], "n_rows": int(len(xl)), "gap_rows": gaps, "kink_px": None, "step_px": None}
    d2 = lambda x: float(np.abs(x[:-2] - 2 * x[1:-1] + x[2:]).max())   # noqa: E731
    d1 = lambda x: float(np.abs(np.diff(x)).max())                      # noqa: E731
    return {"rows": [lo, hi], "n_rows": int(len(xl)), "gap_rows": gaps,
            "max_d2_left_px": d2(xl), "max_d2_right_px": d2(xr), "kink_px": max(d2(xl), d2(xr)),
            "step_px": max(d1(xl), d1(xr)), "width_px_min": int((xr - xl).min()), "width_px_max": int((xr - xl).max())}


def judge(name, value, limit, higher_is_better):
    ok = value is not None and (value >= limit if higher_is_better else value <= limit)
    return {"check": name, "value": None if value is None else round(float(value), 5), "limit": limit,
            "pass": bool(ok), "provisional": True}


# ----------------------------------------------------------------------------------------------------------------------
# Blender stage
# ----------------------------------------------------------------------------------------------------------------------
def kv_args():
    a = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    return dict(x.split("=", 1) for x in a if "=" in x)


def load_s05(C):
    """Exec s05_previews.py without its trailing C.run(...) so its calibrated setup()/frame helpers are reused unchanged."""
    p = C.ROOT / "pipeline" / "s05_previews.py"; src = p.read_text(encoding="utf-8")
    tail = 'C.run("s05_previews", main)'
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
    s05 = load_s05(C)

    def read_rgba(path):                                  # top-down float RGBA, stored code values
        img = C.bpy.data.images.load(str(path), check_existing=False)
        try:
            try:
                img.colorspace_settings.name = "Non-Color"
            except TypeError:
                pass
            w, h = img.size; px = np.empty(w * h * 4, np.float32); img.pixels.foreach_get(px)
        finally:
            C.bpy.data.images.remove(img)
        return px.reshape(h, w, 4)[::-1]

    def say(msg):
        print("PROGRESS " + msg, flush=True)

    def main():
        out_dir = Path(A["out_dir"]); (out_dir / "stills").mkdir(parents=True, exist_ok=True)
        (out_dir / "checks").mkdir(exist_ok=True); (out_dir / "turntable").mkdir(exist_ok=True)
        finish = A.get("finish", "M2"); device = A.get("device", "GPU")
        still_samples = int(A.get("still_samples", 256)); tt_samples = int(A.get("turntable_samples", 64))
        n_frames = int(A.get("turntable_frames", 36)); tt_pose = A.get("turntable_pose", "P1")
        sc = C.bpy.context.scene; body, rig = C.human_and_rig(); rig.data.pose_position = "POSE"
        cam, lights, base_w = s05["setup"](sc, {"samples": still_samples, "rx": 1638, "ry": 2048, "device": device})
        try:
            sc.cycles.denoiser = "OPENIMAGEDENOISE"                  # OIDN explicitly; falls back to the engine default if renamed
        except (TypeError, AttributeError):
            pass
        sc.cycles.use_denoising = True
        try:
            sc.cycles.denoising_use_gpu = True
        except AttributeError:
            pass
        rg = s05["RIG"][finish]
        mat = [m for m in C.bpy.data.materials if m.name.startswith(f"MX_{finish}_")][0]
        body.material_slots[0].material = mat
        for n, ob in lights.items():
            ob.data.energy = base_w[n] * rg[n]
        sc.view_settings.exposure = rg["exposure"]
        base_loc = {n: tuple(ob.location) for n, ob in lights.items()}
        aim = {"MX_key": s05["LIGHT_TARGET"], "MX_fill": s05["LIGHT_TARGET"], "MX_rim": (0, 0, 1.45)}
        tx, ty = s05["TARGET_XY"]; TAN_HALF = s05["TAN_HALF"]

        W, gnames = C.weights_matrix(body); Wb = W.copy(); Wb[:, [i for i, n in enumerate(gnames) if n.startswith("_")]] = -1
        dom = np.array(gnames)[Wb.argmax(1)]; neck = np.array([d.startswith("neck") for d in dom])
        if not neck.any():
            raise RuntimeError("no neck-dominated vertices; bones with 'neck' in the name: " + str([n for n in gnames if "neck" in n]))
        fit = json.loads(Path(A["theta_fit"]).read_text())
        if fit["rig"]["sha256"] != V1_SHA256:
            raise RuntimeError("theta_fit.json was authored on a different rig")
        P = posing.Poser(C.bpy, rig); ev = lambda: C.evaluated_positions(body)   # noqa: E731

        def pose_to(pid):
            P.reset()
            if pid == "theta_fit":
                P.apply_local(fit["bones"]); P.ground(ev)                      # same recipe as s02's BASE baseline
            else:
                P.apply_local(json.loads((C.ROOT / "poses" / s05["POSE_FILES"][pid]).read_text())["bones"])
            return ev()

        def aim_rig(X, deg, margin=0.04):
            """Constant-distance full-body camera at azimuth `deg` (0 = front, 90 = model's left) and the light rig orbited with it."""
            r = math.radians(deg); c_hat = np.array([math.sin(r), -math.cos(r)]); cs, sn = math.cos(r), math.sin(r)
            z = X[:, 2]; zc = 0.5 * (z.max() + z.min()); half = 0.5 * (z.max() - z.min()) * (1 + 2 * margin)
            rmax = float(np.hypot(X[:, 0] - tx, X[:, 1] - ty).max()); D = half / TAN_HALF + rmax
            cam.location = (tx + D * c_hat[0], ty + D * c_hat[1], zc); s05["look_at"](cam, (tx, ty, zc))
            for n, ob in lights.items():
                x0, y0, z0 = base_loc[n]; dx, dy = x0 - tx, y0 - ty
                ob.location = (tx + dx * cs - dy * sn, ty + dx * sn + dy * cs, z0); s05["look_at"](ob, aim[n])
            C.bpy.context.view_layer.update()
            return {"distance_m": round(D, 4), "height_m": round(zc, 4), "azimuth_deg": deg}

        def neck_rows(Xn, W_px, H_px):
            M = np.array(cam.matrix_world.inverted(), float); pc = Xn @ M[:3, :3].T + M[:3, 3]
            rows = (1 - (pc[:, 1] / -pc[:, 2]) / TAN_HALF) * 0.5 * H_px
            return float(rows.min()), float(rows.max())

        report = {"finish": finish, "poses": {}, "turntable": None, "thresholds": THRESH, "device": sc.cycles.device,
                  "samples": {"stills": still_samples, "turntable": tt_samples}}
        masks_all, X_by_pose = {}, {}
        for pid in ("P1", "theta_fit"):
            X = pose_to(pid); X_by_pose[pid] = X; rec = {"stills": {}, "camera": {}}; masks = {}
            sc.render.resolution_x, sc.render.resolution_y = 1638, 2048
            for name, deg in {**VIEWS, **CHECK_VIEWS}.items():
                sc.cycles.samples = still_samples if name in VIEWS else 16
                rec["camera"][name] = aim_rig(X, deg)
                path = out_dir / ("stills" if name in VIEWS else "checks") / f"{pid}_{name}.png"
                io.render_still(sc, path); rgba = read_rgba(path); m = rgba[..., 3] > 0.5; masks[name] = m
                rows = np.flatnonzero(m.any(1)); cols = np.flatnonzero(m.any(0))
                rec["stills"][name] = {"path": str(path), "sha256": contracts.sha256_file(path), "silhouette_px": int(m.sum()),
                                       "bbox_rows": [int(rows[0]), int(rows[-1])] if rows.size else None,
                                       "bbox_cols": [int(cols[0]), int(cols[-1])] if cols.size else None,
                                       "neck_rows": [round(v, 1) for v in neck_rows(X[neck], 1638, 2048)]}
                say(f"{pid} {name} {deg:g} deg done")
            g = rec["checks"] = {}
            g["front_mirror"] = dict(zip(("iou", "shift_px"), best_mirror_iou(masks["front"], masks["front"])))
            g["back_mirror"] = dict(zip(("iou", "shift_px"), best_mirror_iou(masks["back"], masks["back"])))
            g["front_vs_back"] = dict(zip(("iou", "shift_px"), best_mirror_iou(masks["front"], masks["back"])))
            g["profile_LR"] = dict(zip(("iou", "shift_px"), best_mirror_iou(masks["side"], masks["side_R_check"])))
            for nm, view in (("neck_back", "back"), ("neck_side", "side")):
                r0, r1 = rec["stills"][view]["neck_rows"]; g[nm] = neck_kinks(masks[view], r0, r1)
            rec["verdicts"] = ([judge(k, g[k]["iou"], THRESH["mirror_iou_min"], True) for k in ("front_mirror", "back_mirror", "front_vs_back", "profile_LR")]
                               + [judge(f"{k}_kink_px", g[k]["kink_px"], THRESH["neck_kink_px_max"], False) for k in ("neck_back", "neck_side")]
                               + [judge(f"{k}_step_px", g[k]["step_px"], THRESH["neck_step_px_max"], False) for k in ("neck_back", "neck_side")])
            report["poses"][pid] = rec; masks_all[pid] = masks

        if A.get("skip_turntable", "0") != "1":                       # 36-frame 360 deg turntable of P1, same rig orbit
            X = pose_to(tt_pose); sc.render.resolution_x, sc.render.resolution_y = 1024, 1280; sc.cycles.samples = tt_samples
            frames, tmasks = [], []
            for k in range(n_frames):
                deg = 360.0 * k / n_frames; aim_rig(X, deg)
                path = out_dir / "turntable" / f"{tt_pose}_t{k:03d}_{int(round(deg)):03d}deg.png"
                io.render_still(sc, path); m = read_rgba(path)[..., 3] > 0.5; tmasks.append(m)
                frames.append({"frame": k, "azimuth_deg": deg, "path": str(path), "silhouette_px": int(m.sum())})
                if k % 6 == 5 or k == n_frames - 1:
                    say(f"turntable {k + 1}/{n_frames}")
            pairs = []
            for k in range(1, n_frames // 2):                         # frame k vs frame n-k: mirror images for a symmetric pose
                iou, s = best_mirror_iou(tmasks[k], tmasks[n_frames - k], search=24)
                pairs.append({"k": k, "azimuth_deg": [360.0 * k / n_frames, 360.0 * (n_frames - k) / n_frames], "iou": round(iou, 5), "shift_px": s})
            areas = np.array([f["silhouette_px"] for f in frames], float)
            report["turntable"] = {"pose": tt_pose, "frames": frames, "resolution": [1024, 1280], "mirror_pairs": pairs,
                                   "mirror_pair_iou_min": min(p["iou"] for p in pairs) if pairs else None,
                                   "silhouette_area_px": {"min": int(areas.min()), "max": int(areas.max())},
                                   "verdict": judge("turntable_mirror_pair_iou_min", min(p["iou"] for p in pairs) if pairs else None,
                                                    THRESH["mirror_iou_min"], True)}
        report["gateway_calls"] = len(io.CALLS)
        contracts.atomic_write_json(out_dir / "m1_blender_report.json", report)
        verdicts = [v for r in report["poses"].values() for v in r["verdicts"]] + ([report["turntable"]["verdict"]] if report["turntable"] else [])
        return {"report": str(out_dir / "m1_blender_report.json")}, {"checks": len(verdicts), "failed": [v["check"] for v in verdicts if not v["pass"]]}

    C.run(STAGE, main)


# ----------------------------------------------------------------------------------------------------------------------
# system-Python launcher
# ----------------------------------------------------------------------------------------------------------------------
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


def contact_sheet(frames, out_path, cols=6, thumb=(512, 640)):
    from PIL import Image, ImageDraw
    rows = math.ceil(len(frames) / cols); sheet = Image.new("RGB", (cols * thumb[0], rows * thumb[1]), (70, 70, 74))
    for i, f in enumerate(frames):
        im = Image.open(f["path"]).convert("RGBA").resize(thumb, Image.LANCZOS)
        cell = Image.new("RGB", thumb, (70, 70, 74)); cell.paste(im, (0, 0), im)
        ImageDraw.Draw(cell).text((8, 8), f"{f['azimuth_deg']:.0f} deg", fill=(235, 235, 235))
        sheet.paste(cell, ((i % cols) * thumb[0], (i // cols) * thumb[1]))
    sheet.save(out_path); return out_path


def four_view(pid, stills_dir, out_path, height=1024):
    from PIL import Image
    ims = []
    for name in VIEWS:
        im = Image.open(stills_dir / f"{pid}_{name}.png").convert("RGBA"); w = int(im.width * height / im.height)
        ims.append(im.resize((w, height), Image.LANCZOS))
    sheet = Image.new("RGB", (sum(i.width for i in ims), height), (70, 70, 74)); x = 0
    for im in ims:
        sheet.paste(im, (x, 0), im); x += im.width
    sheet.save(out_path); return out_path


def launch():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--bundle", default="/content/maximus_v2/bundle"); ap.add_argument("--out", default="/content/drive/MyDrive/maximus_v2/results")
    ap.add_argument("--blender"); ap.add_argument("--device", default="GPU"); ap.add_argument("--finish", default="M2")
    ap.add_argument("--still-samples", default="256"); ap.add_argument("--turntable-samples", default="64")
    ap.add_argument("--turntable-frames", default="36"); ap.add_argument("--skip-turntable", action="store_true")
    ap.add_argument("--keep-frames", action="store_true", help="also copy the individual turntable frames to --out")
    ap.add_argument("--strict", action="store_true", help="exit 1 when any provisional check fails")
    a = ap.parse_args(); B = Path(a.bundle).resolve(); t0 = time.time()
    fit = next((p for p in (HERE / "m_theta_fit.json", Path("/content/drive/MyDrive/maximus_v2/m_theta_fit.json")) if p.exists()), None)
    if fit is None:
        sys.exit("m_theta_fit.json not found next to this script")
    man = verify_bundle(B); exe = find_blender(B, a.blender)
    work = B / "out" / "M1_structural_audit"; work.mkdir(parents=True, exist_ok=True)
    env = run_blender(exe, B, {"bundle": B, "out_dir": work, "device": a.device, "finish": a.finish, "theta_fit": fit,
                               "still_samples": a.still_samples, "turntable_samples": a.turntable_samples,
                               "turntable_frames": a.turntable_frames, "skip_turntable": int(a.skip_turntable)}, work / "m1_blender.log")
    if env["status"] != "ok":
        print(json.dumps(env.get("errors"), indent=1)[:4000]); sys.exit(1)
    rep = json.loads((work / "m1_blender_report.json").read_text()); extras = []
    for pid in rep["poses"]:
        extras.append(four_view(pid, work / "stills", work / f"{pid}_four_view.png"))
    if rep["turntable"]:
        extras.append(contact_sheet(rep["turntable"]["frames"], work / f"{rep['turntable']['pose']}_turntable_contact_sheet.png"))
    verdicts = {pid: r["verdicts"] for pid, r in rep["poses"].items()}
    if rep["turntable"]:
        verdicts["turntable"] = [rep["turntable"]["verdict"]]
    failed = [f"{k}:{v['check']}" for k, vs in verdicts.items() for v in vs if not v["pass"]]
    summary = {"stage": STAGE, "master_sha256": man.get("master_sha256"), "finish": a.finish, "device": rep["device"],
               "thresholds": THRESH, "verdicts": verdicts, "failed_checks": failed, "all_checks_pass": not failed,
               "elapsed_s": round(time.time() - t0, 1), "blender_report": rep}
    (work / "M1_structural_audit.json").write_text(json.dumps(summary, indent=1))
    dst = Path(a.out) / "M1_volumetric_audit"; dst.mkdir(parents=True, exist_ok=True)
    keep = [work / "M1_structural_audit.json", work / "m1_blender.log"] + sorted((work / "stills").glob("*.png")) + sorted((work / "checks").glob("*.png")) + extras
    if a.keep_frames:
        keep += sorted((work / "turntable").glob("*.png"))
    bad = []
    for f in keep:
        rel = f.relative_to(work); (dst / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, dst / rel)
    if hasattr(os, "sync"):
        os.sync()
    for f in keep:                                                   # read back from Drive and compare hashes
        rel = f.relative_to(work)
        if not (dst / rel).exists() or sha256(dst / rel) != sha256(f):
            bad.append(str(rel))
    print(json.dumps({"copied": len(keep), "to": str(dst), "readback_mismatch": bad, "failed_checks": failed,
                      "elapsed_s": summary["elapsed_s"]}, indent=1))
    for pid, vs in verdicts.items():
        for v in vs:
            print(f"  {pid:10s} {v['check']:34s} {v['value']!s:>9} (limit {v['limit']}) {'PASS' if v['pass'] else 'FAIL'}  [provisional]")
    sys.exit(1 if bad or (failed and a.strict) else 0)


if __name__ == "__main__":
    blender_main() if IN_BLENDER else launch()
