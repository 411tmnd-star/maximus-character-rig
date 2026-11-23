"""Clean-room AO + normal delighting prep (NumPy only; system Python and Blender).

Idea taken from the published Gaussian Garments appearance model (arXiv:2409.08189 sec. 3.3): lighting effects are
predicted from ambient occlusion and normal maps, so the stored albedo stays shadow-free. This module does the
classical, non-learned version of that split, which is enough to prepare textures and scraped photos:

    I(p) ~= albedo(p) * AO(p) * S(n(p))        S = 2nd-order spherical-harmonic irradiance (9 coefficients)

fit_sh_lighting() solves the 9 SH coefficients by robust least squares (iteratively reweighted) assuming a
piecewise-smooth albedo; delight() divides the estimated shading out. No code from the unlicensed eth-ait repository
was copied or consulted.
"""
import numpy as np


def srgb_to_linear(c):
    c = np.asarray(c, np.float64)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(c):
    c = np.clip(np.asarray(c, np.float64), 0, None)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)


def hex_to_linear(h):
    h = h.lstrip("#")
    return srgb_to_linear([int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)])


def decode_normal_map(rgb):
    """Tangent/object-space normal map in [0,1] -> unit vectors in [-1,1]."""
    n = np.asarray(rgb, np.float64) * 2.0 - 1.0
    return n / np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-12)


def sh_basis(n):
    """Real SH basis up to l=2 (9 functions) for unit normals n (..., 3)."""
    x, y, z = n[..., 0], n[..., 1], n[..., 2]
    return np.stack([np.full_like(x, 0.282095), 0.488603 * y, 0.488603 * z, 0.488603 * x,
                     1.092548 * x * y, 1.092548 * y * z, 0.315392 * (3 * z * z - 1), 1.092548 * x * z,
                     0.546274 * (x * x - y * y)], axis=-1)


def fit_sh_lighting(lum, normals, ao=None, mask=None, iters=8):
    """Robust fit of luminance ~= AO * (B(n) . c). Returns c (9,) and per-pixel weights of the last iteration."""
    lum = np.asarray(lum, np.float64).ravel(); N = np.asarray(normals, np.float64).reshape(-1, 3)
    ao = np.ones_like(lum) if ao is None else np.asarray(ao, np.float64).ravel()
    m = np.ones_like(lum, bool) if mask is None else np.asarray(mask, bool).ravel()
    A = sh_basis(N[m]) * ao[m, None]; y = lum[m]; w = np.ones_like(y)
    for _ in range(iters):
        c = np.linalg.lstsq(A * w[:, None], y * w, rcond=None)[0]
        r = y - A @ c; s = 1.4826 * np.median(np.abs(r)) + 1e-9
        w = 1.0 / np.sqrt(1.0 + (r / (2.5 * s)) ** 2)      # Cauchy-style reweighting (albedo edges are outliers)
    return c, w


def shading(normals, c, ao=None):
    S = sh_basis(np.asarray(normals, np.float64)) @ c
    if ao is not None:
        S = S * np.asarray(ao, np.float64)
    return S


def delight(image_lin, normals, ao=None, mask=None, floor=0.05):
    """Remove AO + low-frequency lighting from a linear RGB image. Returns (albedo, shading, sh_coeffs).
    Albedo is normalised so its mean luminance equals the input's (only the lighting pattern is removed)."""
    img = np.asarray(image_lin, np.float64)
    lum = img @ np.array([0.2126, 0.7152, 0.0722])
    c, _ = fit_sh_lighting(lum, normals, ao, mask)
    S = shading(normals, c, ao)
    S = np.maximum(S, floor * max(float(np.median(S)), 1e-6))
    alb = img / S[..., None]
    m = np.ones(lum.shape, bool) if mask is None else np.asarray(mask, bool)
    alb *= lum[m].mean() / max(float((alb @ np.array([0.2126, 0.7152, 0.0722]))[m].mean()), 1e-9)
    return alb, S, c


def pack_orm(ao, roughness, metallic=None):
    """glTF ORM texture: R = occlusion, G = roughness, B = metallic."""
    ao = np.asarray(ao, np.float64); r = np.asarray(roughness, np.float64)
    m = np.zeros_like(r) if metallic is None else np.asarray(metallic, np.float64)
    return np.stack([ao, r, m], axis=-1)
