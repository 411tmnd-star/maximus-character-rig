"""lighting_director.py  (Maximus runtime_v3, module 5 of 8)  Merch-complementary three-point light choice L*(G) for a garment G.

A small, explicit model (not a renderer): each area light of P watts at distance d gives irradiance ~ P / (4 pi d^2) on the torso
(the s05 calibration that matched M2's measured sRGB ~201); a surface of linear albedo rho shows scene value rho * E * 2^exposure, and AgX
around its 0.18 pivot is approximated by sRGB code ~ 128 + 37 * log2(value / 0.18) (clipped 0..255), the slope s05 measured.
Search: key watts x fill ratio x rim watts x exposure on a grid; score =
    garment torso code in its band [105, 155] (mid-tones, texture readable) - |garment - mannequin| separation under 22 codes
    - clipping (rim highlight > 248, mannequin torso > 245) ; tie-break: closest to the s05 rig.
Rim colour: the garment's complementary hue at low saturation (0.12), so the edge light separates the merch without tinting it.
Finish: lighting cannot separate a garment from a mannequin of the same albedo (both get the same light), so the director first picks the
finish (M1 Obsidian #18181B, M2 Alabaster #F2EFE9, M3 Travertine #D8CEBE) with the largest luminance contrast to the garment, then the lights.
Positions stay those of the s05 rig (key 1.7,-2.3,2.7 / fill -2.4,-1.6,1.6 / rim -0.8,2.2,2.4), so frames stay comparable.

solve_multilayer(layer_hexes, mannequin_hex) (Milestone 1): several garments seen at once (e.g. dark open jacket over a white singlet
on M1). In this model every surface on the torso gets the same irradiance, so the code spread between layers is fixed by their
albedos (37 codes per stop); light weights and exposure only slide the whole set along the curve. The solver places that set so that
no layer clips (> 245) or crushes (< 28), highlights weighted twice, and reports when the albedo range is wider than the usable
window, instead of pretending a light can fix it.
"""
from __future__ import annotations

import colorsys
import itertools
import math

import numpy as np

S05 = {"MX_key": 450.0, "MX_fill": 70.0, "MX_rim": 260.0, "exposure": -1.0}
DIST = {"MX_key": 3.21, "MX_fill": 2.94, "MX_rim": 3.12}
COS = {"MX_key": 0.80, "MX_fill": 0.55, "MX_rim": 0.0}         # mean cosine on the camera-facing torso (rim lights the silhouette only)


def srgb_to_linear(c):
    c = np.asarray(c, float); return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def hex_rgb(h):
    h = h.lstrip("#"); return np.array([int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)])


def luminance(rgb_lin):
    return float(np.dot(rgb_lin, [0.2126, 0.7152, 0.0722]))


def code(value):
    return float(np.clip(128 + 37 * math.log2(max(value, 1e-6) / 0.18), 0, 255))


def predict(rho, w, exposure):
    E = sum(w[k] * COS[k] / (4 * math.pi * DIST[k] ** 2) for k in ("MX_key", "MX_fill"))
    rim = w["MX_rim"] / (4 * math.pi * DIST["MX_rim"] ** 2)
    s = 2.0 ** exposure; return code(rho * E * s), code(max(rho, 0.6) * rim * 1.8 * s)


FINISHES = {"M1": "#18181B", "M2": "#F2EFE9", "M3": "#D8CEBE"}


def pick_finish(garment_hex):
    rg = luminance(srgb_to_linear(hex_rgb(garment_hex)))
    return max(FINISHES, key=lambda f: abs(math.log2(max(luminance(srgb_to_linear(hex_rgb(FINISHES[f]))), 1e-4) / max(rg, 1e-4))))


def solve(garment_hex, mannequin_hex=None):
    if mannequin_hex is None:                                  # try every finish, keep the lowest-score solve (contrast first, no clipping)
        best = min((dict(_solve(garment_hex, FINISHES[f]), finish=f) for f in FINISHES), key=lambda r: r["score"])
        best["finish_by_contrast_only"] = pick_finish(garment_hex); return best
    return dict(_solve(garment_hex, mannequin_hex), finish=None)


def _solve(garment_hex, mannequin_hex):
    rho_g = luminance(srgb_to_linear(hex_rgb(garment_hex))); rho_m = luminance(srgb_to_linear(hex_rgb(mannequin_hex)))
    best = None
    for key, fill_r, rim, ex in itertools.product(np.linspace(250, 900, 14), (0.15, 0.25, 0.35, 0.5), np.linspace(120, 520, 9), np.arange(-3.5, 2.01, 0.25)):
        w = {"MX_key": key, "MX_fill": key * fill_r, "MX_rim": rim}
        g, rim_code = predict(rho_g, w, ex); m, _ = predict(rho_m, w, ex)
        band = max(0.0, 105 - g) + max(0.0, g - 155); sep = max(0.0, 22 - abs(g - m)); clip = max(0.0, rim_code - 248) + max(0.0, m - 245)
        dist = abs(key - S05["MX_key"]) / 450 + abs(ex - S05["exposure"]) * 0.2
        score = band * 3 + sep * 2 + clip * 2 + dist
        if best is None or score < best[0]:
            best = (score, w, ex, g, m, rim_code)
    score, w, ex, g, m, rc = best
    h, l, s = colorsys.rgb_to_hls(*hex_rgb(garment_hex)); rim_rgb = colorsys.hls_to_rgb((h + 0.5) % 1.0, 0.85, 0.12)
    return {"garment_hex": garment_hex, "mannequin_hex": mannequin_hex, "weights_w": {k: round(v, 1) for k, v in w.items()}, "exposure": float(ex),
            "rim_rgb_linear": [round(float(x), 4) for x in srgb_to_linear(rim_rgb)], "predicted_codes": {"garment_torso": round(g, 1), "mannequin_torso": round(m, 1),
            "rim_highlight": round(rc, 1)}, "score": round(score, 3), "model": "irradiance P/(4 pi d^2) x cos, AgX ~ 128 + 37 log2(v/0.18)"}


USABLE_LO, USABLE_HI = 28.0, 245.0                              # shadow floor / highlight ceiling for texture to stay readable


def _predict_raw(rho, w, exposure):
    """predict() without the 0..255 clamp, so clipping and crushing are scored by how far past the limits a value really goes."""
    E = sum(w[k] * COS[k] / (4 * math.pi * DIST[k] ** 2) for k in ("MX_key", "MX_fill"))
    rim = w["MX_rim"] / (4 * math.pi * DIST["MX_rim"] ** 2)
    s = 2.0 ** exposure
    raw = lambda v: 128 + 37 * math.log2(max(v, 1e-6) / 0.18)   # noqa: E731
    return raw(rho * E * s), raw(max(rho, 0.6) * rim * 1.8 * s)


def solve_multilayer(layer_hexes, mannequin_hex):
    """Joint light + exposure for several visible garment layers over one mannequin finish (same s05 rig positions as solve()).

    layer_hexes: inner -> outer, e.g. ["#F2F2EE", "#141518"] (white singlet under a dark jacket); mannequin_hex e.g. "#18181B".
    Score = 4 * highlight clip (> 245, incl. rim > 248) + 2 * shadow crush (< 28) for every layer and the mannequin
    + 1 * separation shortfall (< 22 codes) between layers adjacent in luminance and between the outer layer and the mannequin
    + small tie-break toward the s05 rig. Returns per-layer codes, the albedo range in stops vs the usable window, and warnings.
    """
    hexes = list(layer_hexes)
    if not hexes:
        raise ValueError("layer_hexes must name at least one layer")
    rhos = [luminance(srgb_to_linear(hex_rgb(h))) for h in hexes]
    rho_m = luminance(srgb_to_linear(hex_rgb(mannequin_hex)))
    by_lum = sorted(range(len(hexes)), key=lambda i: rhos[i])
    best = None
    for key, fill_r, rim, ex in itertools.product(np.linspace(250, 900, 14), (0.15, 0.25, 0.35, 0.5), np.linspace(120, 520, 9), np.arange(-4.0, 2.01, 0.125)):
        w = {"MX_key": key, "MX_fill": key * fill_r, "MX_rim": rim}
        codes = [_predict_raw(r, w, ex) for r in rhos]
        g = [c[0] for c in codes]; rims = [c[1] for c in codes]
        m, _ = _predict_raw(rho_m, w, ex)
        clip = sum(max(0.0, c - USABLE_HI) for c in g + [m]) + sum(max(0.0, rc - 248) for rc in rims)
        crush = sum(max(0.0, USABLE_LO - c) for c in g + [m])
        sep = sum(max(0.0, 22 - abs(g[by_lum[i + 1]] - g[by_lum[i]])) for i in range(len(g) - 1)) + max(0.0, 22 - abs(g[-1] - m))
        dist = abs(key - S05["MX_key"]) / 450 + abs(ex - S05["exposure"]) * 0.2
        score = clip * 4 + crush * 2 + sep + dist * 0.25
        if best is None or score < best[0]:
            best = (score, w, ex, g, m, rims, clip, crush, sep)
    score, w, ex, g, m, rims, clip, crush, sep = best
    lums = rhos + [rho_m]
    stops_needed = math.log2(max(lums) / max(min(lums), 1e-6))
    stops_window = (USABLE_HI - USABLE_LO) / 37.0
    warnings = []
    if stops_needed > stops_window:
        warnings.append(f"albedo range {stops_needed:.2f} stops > usable {stops_window:.2f} stops: some layer must clip or crush in this "
                        "model; protect it with material (sheen/rim on the dark fabric) or a local fill, not with global exposure")
    if abs(g[-1] - m) < 22:
        warnings.append("outer layer and mannequin are within 22 codes: separation must come from rim light and material, not key/fill")
    return {
        "layer_hexes": hexes, "mannequin_hex": mannequin_hex,
        "weights_w": {k: round(v, 1) for k, v in w.items()}, "exposure": float(ex),
        "predicted_codes": {"layers_torso": [round(min(max(c, 0.0), 255.0), 1) for c in g], "mannequin_torso": round(min(max(m, 0.0), 255.0), 1),
                            "rim_highlights": [round(min(max(c, 0.0), 255.0), 1) for c in rims], "layers_torso_unclamped": [round(c, 1) for c in g]},
        "clip_codes": round(clip, 2), "crush_codes": round(crush, 2), "separation_shortfall_codes": round(sep, 2),
        "dynamic_range": {"albedo_stops_needed": round(stops_needed, 3), "usable_stops": round(stops_window, 3), "feasible": bool(stops_needed <= stops_window)},
        "no_highlight_clip": bool(clip == 0.0), "no_shadow_crush": bool(crush == 0.0),
        "warnings": warnings, "score": round(score, 3),
        "model": "irradiance P/(4 pi d^2) x cos, AgX ~ 128 + 37 log2(v/0.18), same irradiance on every torso layer",
    }
