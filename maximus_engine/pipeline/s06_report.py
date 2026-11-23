"""Stage 06 (system Python): storefront composites + MAXIMUS_REBUILD_REPORT.md generated from the stage reports.

Composites reproduce the storefront CSS: per-finish radial backdrop (ellipse 70% 60% at 50% 38%) and the bust mask
linear-gradient(#000 82%, rgba(0,0,0,.4) 91%, transparent 98%) applied to the native-RGBA render (no hem_trim()).
Every artefact is SHA-256 hashed into the report and out/reports/artifact_hashes.json.
"""
import datetime, json
from pathlib import Path

import numpy as np
from PIL import Image

from corpus_bridge import contracts

ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT / "out"; REP = OUT / "reports"
BACKDROP = {"M1": ("#f4f1ea", "#e2ddd3", "#cbc4b8", 0.55), "M2": ("#34322f", "#22201e", "#151413", 0.58), "M3": ("#e6e0d6", "#cfc7ba", "#b5ab9c", 0.55)}
FADE = (0.82, 0.91, 0.98)   # storefront store.css (82% start: hems sit at 77-80%)


def rgb(h):
    h = h.lstrip("#"); return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], np.float64)


def backdrop(w, h, key):
    c0, c1, c2, mid = BACKDROP[key]
    yy, xx = np.mgrid[0:h, 0:w]; dx = (xx / w - 0.5) / 0.70; dy = (yy / h - 0.38) / 0.60
    t = np.clip(np.sqrt(dx * dx + dy * dy) / np.sqrt(2) * 1.6, 0, 1)[..., None]
    a = np.where(t < mid, rgb(c0) + (rgb(c1) - rgb(c0)) * (t / mid), rgb(c1) + (rgb(c2) - rgb(c1)) * ((t - mid) / (1 - mid)))
    return a


def bust_mask(h):
    y = np.arange(h) / h; s, m, e = FADE
    return np.interp(y, [0, s, m, e, 1], [1, 1, 0.4, 0, 0])


def composite(src, key, dst):
    im = np.asarray(Image.open(src).convert("RGBA"), np.float64); h, w = im.shape[:2]
    a = im[..., 3:4] / 255 * bust_mask(h)[:, None, None]
    out = backdrop(w, h, key) * (1 - a) + im[..., :3] * a
    Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)).save(dst); return dst


def contact_sheet(paths, dst, cols=3, width=420):
    ims = [Image.open(p) for p in paths]; r = width / ims[0].width; th = int(ims[0].height * r)
    sheet = Image.new("RGB", (cols * width, ((len(ims) + cols - 1) // cols) * th), (20, 20, 20))
    for i, im in enumerate(ims):
        sheet.paste(im.resize((width, th)), ((i % cols) * width, (i // cols) * th))
    sheet.save(dst); return dst


def load(name):
    p = REP / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def main(**_):
    s01, s03, s04, s05 = load("s01_master_report.json"), load("s03_gates.json"), load("s04_bake_export.json"), load("s05_previews.json")
    comp = []
    if s05:
        cdir = OUT / "previews" / "composite"; cdir.mkdir(parents=True, exist_ok=True)
        for k, v in sorted(s05["renders"].items()):
            comp.append(composite(v["path"], k.split("_")[1], cdir / (Path(v["path"]).stem + "_storefront.png")))
        contact_sheet(comp, OUT / "previews" / "contact_sheet_storefront.png")
    arts = sorted([p for p in OUT.rglob("*") if p.is_file() and p.suffix in (".blend", ".glb", ".png", ".npz")] +
                  list((ROOT / "poses").glob("*.json")) + list((ROOT / "pipeline").glob("*.py")) + [ROOT / "corpus_lock.json"])
    hashes = {str(p.relative_to(ROOT)).replace("\\", "/"): contracts.sha256_file(p) for p in arts if p.exists() and "dev" not in p.parts}
    contracts.atomic_write_json(REP / "artifact_hashes.json", hashes)
    (ROOT / "MAXIMUS_REBUILD_REPORT.md").write_text(render_report(s01, s03, s04, s05, hashes), encoding="utf-8")
    ok = bool(s03) and s03["status"] == "ok"
    return contracts.envelope("s06_report", "ok", outputs={"report": str(ROOT / "MAXIMUS_REBUILD_REPORT.md"), "composites": [str(c) for c in comp]},
                              metrics={"gates_all_pass": ok, "artifacts_hashed": len(hashes)})


def render_report(s01, s03, s04, s05, hashes):
    import report_text
    return report_text.build(s01, s03, s04, s05, hashes, ROOT, datetime.date.today().isoformat())
