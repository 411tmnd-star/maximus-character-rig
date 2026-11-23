"""camera_director.py  (Maximus runtime_v3, module 6 of 8)  Bust-framed 105 mm camera solve and its pixel check (no Blender).

Same maths as pipeline/s05_previews.frame_camera (and colab_stage/m_render.py): level pinhole camera, crown TOP_MARGIN below the top edge
(accepted band 6-8 %), world height FADE_Z on the storefront's 82 % CSS fade line. y = 0.5 - (z - zc) / (2 tan(fov/2) (D - a)).
"""
from __future__ import annotations

import math

import numpy as np

LENS_MM, SENSOR_MM = 105.0, 36.0
TAN_HALF = (SENSOR_MM / 2) / LENS_MM
TOP_MARGIN, TOP_BAND = 0.07, (0.06, 0.08)
FADE_FRAC, FADE_Z = 0.82, 0.852
TARGET_XY = (0.0, -0.02)


def solve(X, view_deg=0.0, top_margin=TOP_MARGIN, fade_z=FADE_Z, fade_frac=FADE_FRAC, lens_mm=LENS_MM):
    th = (SENSOR_MM / 2) / lens_mm; r = math.radians(view_deg); c_hat = np.array([math.sin(r), -math.cos(r)])
    X = np.asarray(X, float); a = (X[:, :2] - np.asarray(TARGET_XY)) @ c_hat; z = X[:, 2]
    alpha = 2 * th * (0.5 - top_margin); beta = 2 * th * (fade_frac - 0.5); D, zc = 3.4, 1.22
    for _ in range(20):
        k = int(np.argmax((z - zc) / (D - a)))
        D2 = (z[k] - fade_z + alpha * a[k]) / (alpha + beta); zc2 = fade_z + beta * D2
        if abs(D2 - D) + abs(zc2 - zc) < 1e-12:
            D, zc = D2, zc2; break
        D, zc = D2, zc2
    loc = (TARGET_XY[0] + D * c_hat[0], TARGET_XY[1] + D * c_hat[1], zc)
    top = 0.5 - float(((z - zc) / (D - a)).max()) / (2 * th)
    return {"location": loc, "look_at": (TARGET_XY[0], TARGET_XY[1], zc), "distance_m": D, "height_m": zc, "predicted_top_margin": top, "lens_mm": lens_mm}


def measure_top_margin(alpha):
    """Fraction of the frame height above the topmost opaque pixel (alpha image rows top-down)."""
    a = np.asarray(alpha); rows = np.nonzero((a > 0.5).any(axis=1))[0]
    return float(rows.min() / a.shape[0]) if rows.size else None


def check(alpha):
    tm = measure_top_margin(alpha)
    return {"top_margin": tm, "in_band_6_8_percent": tm is not None and TOP_BAND[0] <= tm <= TOP_BAND[1]}
