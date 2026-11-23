"""presentation_engine.py  (Maximus runtime_v3, module 7 of 8)  Image products of one render, with a hard separation between
identity layers and relighting layers.

Identity (Cryptomatte-style, never used for shading):  <stem>_idmask.webp  lossless palette ID per pixel (object/material id -> colour)
                                                        <stem>_luma.webp    Rec.709 luma of the beauty pass (8-bit, lossless)
2.5D relighting (never used for selection):           <stem>_surface.webp      camera-space normal packed n * 0.5 + 0.5 (RGB, lossless)
                                                        <stem>_surface_aux.webp  R = depth (near..far -> 255..0), G = roughness, B = ambient occlusion
A manifest records channel meanings, value ranges and the id palette, and the validator refuses a set where the two families share a file
or a channel (the separation the storefront relies on: the hover/selection code reads only identity layers, the relight shader only surface
layers). Blender 5.2.3 writes only Combined into EXR even with passes enabled (found in M3), so the data passes arrive as separate
emission renders and are packed here.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

PALETTE = [(0, 0, 0), (230, 25, 75), (60, 180, 75), (0, 130, 200), (245, 130, 48), (145, 30, 180), (70, 240, 240), (240, 50, 230), (210, 245, 60)]
FAMILIES = {"identity": ("_idmask", "_luma"), "relight": ("_surface", "_surface_aux")}


def _save(arr, path, lossless=True):
    from PIL import Image
    Image.fromarray(arr).save(path, lossless=lossless, quality=100, method=6)


def pack(stem, out_dir, beauty_rgba, ids, normal_cam, depth_m, roughness, ao, id_names, depth_range=None):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    rgb = np.clip(np.asarray(beauty_rgba, float)[..., :3], 0, 1); a = np.asarray(beauty_rgba, float)[..., 3]
    luma = (255 * (0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2])).round().astype(np.uint8)
    idc = np.zeros(ids.shape + (3,), np.uint8)
    for i in np.unique(ids):
        idc[ids == i] = PALETTE[int(i) % len(PALETTE)]
    n = np.clip(np.asarray(normal_cam, float) * 0.5 + 0.5, 0, 1)
    fg = a > 0.5; d = np.asarray(depth_m, float)
    lo, hi = depth_range or ((float(d[fg].min()), float(d[fg].max())) if fg.any() else (0.0, 1.0))
    dn = np.where(fg, 1 - np.clip((d - lo) / max(hi - lo, 1e-9), 0, 1), 0)
    aux = np.dstack([dn, np.clip(roughness, 0, 1), np.clip(ao, 0, 1)])
    files = {"_idmask": idc, "_luma": luma, "_surface": (n * 255).round().astype(np.uint8), "_surface_aux": (aux * 255).round().astype(np.uint8)}
    for suf, arr in files.items():
        _save(arr, out / f"{stem}{suf}.webp")
    man = {"schema": "maximus.presentation/1", "stem": stem, "families": {k: [stem + s + ".webp" for s in v] for k, v in FAMILIES.items()},
           "channels": {"_idmask": {"rgb": "palette id", "ids": {str(i): id_names.get(int(i), f"id{int(i)}") for i in np.unique(ids)},
                                    "palette": {str(i): PALETTE[int(i) % len(PALETTE)] for i in np.unique(ids)}},
                        "_luma": {"l": "Rec.709 luma of beauty, sRGB codes"},
                        "_surface": {"rgb": "camera-space normal * 0.5 + 0.5"},
                        "_surface_aux": {"r": f"depth, 255 = {lo:.4f} m .. 0 = {hi:.4f} m", "g": "roughness 0..1", "b": "ambient occlusion 0..1"}},
           "size": list(rgb.shape[:2])}
    (out / f"{stem}_presentation.json").write_text(json.dumps(man, indent=1))
    return man


def validate(manifest):
    ident = set(manifest["families"]["identity"]); rel = set(manifest["families"]["relight"])
    ok = ident.isdisjoint(rel) and len(ident) == 2 and len(rel) == 2
    return {"separated": ok, "identity": sorted(ident), "relight": sorted(rel)}
