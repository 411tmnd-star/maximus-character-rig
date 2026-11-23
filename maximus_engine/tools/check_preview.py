"""Independent check of one preview render (system Python, PIL + NumPy; does not import the pipeline).

    python -I tools/check_preview.py out/previews/maximus_v2_P1_M1_fast.png M1

Measures on the saved RGBA PNG (sRGB code values, Rec.709 luma, opaque pixels only):
  top_margin        rows above the crown / frame height                       must be 0.06 .. 0.08
  torso_median      median luma in the box x 40-60 %, y 40-60 % (lit torso)   M1: 28 .. 44   M2: 170 .. 235   M3: 150 .. 225
  body_p999         99.9th percentile luma of the body                        M1: >= torso_median + 15 (rim highlights exist)
  clipped_fraction  share of body pixels at code >= 250                       <= 0.001
  head_residual     largest |luma - 9 px box blur| inside the eroded head     reported + head crop written; a crease or
                    silhouette (a crease / flap mark shows up here)           pinch reads as a thin line in the crop
Also reads out/reports/s01_master_report.json for the geometric cause of head marks:
  egg_fit.head_tris_facing_inward == 0 and egg_fit.fold_vertices_sunk == 0 and egg_fit.unfold.inward_after == 0
Writes <png>_head_crop.png (head region, luma stretched x4 around its median) for a visual look.
Exit code 0 only if every hard check passes. Prints one JSON object.
"""
import json, sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
TORSO = {"M1": (28, 44), "M2": (170, 235), "M3": (150, 225)}


def box_blur(a, k):
    """Separable k x k mean filter (k odd) via cumulative sums; edges use the valid part."""
    r = k // 2
    for ax in (0, 1):
        p = np.concatenate([np.zeros_like(a.take([0], ax)), np.cumsum(a, axis=ax)], axis=ax)
        n = a.shape[ax]; i = np.arange(n); lo = np.clip(i - r, 0, n); hi = np.clip(i + r + 1, 0, n)
        a = (p.take(hi, ax) - p.take(lo, ax)) / (hi - lo).reshape([-1 if d == ax else 1 for d in range(a.ndim)])
    return a


def main(png, finish):
    im = np.asarray(Image.open(png).convert("RGBA"), np.float64); h, w = im.shape[:2]
    luma = 0.2126 * im[..., 0] + 0.7152 * im[..., 1] + 0.0722 * im[..., 2]; alpha = im[..., 3] / 255.0; opaque = alpha > 0.99
    rows = np.nonzero((alpha > 0.5).any(axis=1))[0]
    top_margin = float(rows.min() / h)
    ty0, ty1, tx0, tx1 = int(0.40 * h), int(0.60 * h), int(0.40 * w), int(0.60 * w)
    torso = luma[ty0:ty1, tx0:tx1][opaque[ty0:ty1, tx0:tx1]]; body = luma[opaque]
    # head: from the crown down 19 % of the frame height (the egg is ~0.2245 m of a ~1.17 m frame: crown to chin),
    # eroded by ~12 px at 2K so the silhouette edge is not counted
    hy0, hy1 = int(rows.min()), int(rows.min() + 0.19 * h)
    cols = np.nonzero(opaque[hy0:hy1].any(axis=0))[0]; hx0, hx1 = int(cols.min()), int(cols.max()) + 1
    head = luma[hy0:hy1, hx0:hx1]; hm = opaque[hy0:hy1, hx0:hx1].astype(np.float64)
    k = max(3, (int(round(9 * h / 2048)) // 2) * 2 + 1); e = max(3, (int(round(25 * h / 2048)) // 2) * 2 + 1)
    inner = box_blur(hm, e) > 0.999
    resid = np.abs(head - box_blur(head * hm, k) / np.maximum(box_blur(hm, k), 1e-9)) * inner
    med = float(np.median(head[inner])) if inner.any() else 0.0
    crop = np.clip((head - med) * 4 + 128, 0, 255) * hm
    crop_path = Path(str(png)[:-4] + "_head_crop.png"); Image.fromarray(crop.astype(np.uint8)).save(crop_path)
    yx = np.unravel_index(int(resid.argmax()), resid.shape)
    s01 = json.loads((ROOT / "out" / "reports" / "s01_master_report.json").read_text(encoding="utf-8"))["egg_fit"]
    lo, hi = TORSO[finish]
    m = {"png": str(png), "finish": finish, "size": [w, h], "top_margin": round(top_margin, 4),
         "torso_median": round(float(np.median(torso)), 2), "body_p50": round(float(np.median(body)), 2),
         "body_p999": round(float(np.percentile(body, 99.9)), 2), "clipped_fraction": round(float((body >= 250).mean()), 6),
         "head_residual_max": round(float(resid.max()), 2), "head_residual_p999": round(float(np.percentile(resid[inner], 99.9)), 2) if inner.any() else None,
         "head_residual_at_xy_frac": [round((hx0 + int(yx[1])) / w, 4), round((hy0 + int(yx[0])) / h, 4)], "head_crop": str(crop_path),
         "egg_head_tris_facing_inward": s01.get("head_tris_facing_inward"), "egg_fold_vertices_sunk": s01.get("fold_vertices_sunk"),
         "egg_unfold": {k_: (s01.get("unfold") or {}).get(k_) for k_ in ("inward_before", "inward_after_harmonic", "inward_after", "island_vertices", "relax_iters_run")}}
    checks = {"top_margin_6_to_8_percent": 0.06 <= m["top_margin"] <= 0.08,
              f"torso_median_{lo}_to_{hi}": lo <= m["torso_median"] <= hi,
              "no_clipping": m["clipped_fraction"] <= 0.001,
              "egg_no_inward_triangles": m["egg_head_tris_facing_inward"] == 0 and m["egg_unfold"]["inward_after"] == 0,
              "egg_no_sunk_folds": m["egg_fold_vertices_sunk"] == 0}
    if finish == "M1":
        checks["rim_highlights_present"] = m["body_p999"] >= m["torso_median"] + 15
    m["checks"] = checks; m["pass"] = all(checks.values())
    print(json.dumps(m, indent=1)); return 0 if m["pass"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "M1"))
