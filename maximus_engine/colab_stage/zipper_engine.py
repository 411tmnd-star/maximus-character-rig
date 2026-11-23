"""zipper_engine.py - Clean-room procedural macro zipper layer (Video 1 component topology + Video 2 8-parameter contract).

Plain NumPy geometry engine with optional Blender 5.2 `bpy.data` mesh / material / Geometry-Nodes wrapper (ZERO `bpy.ops` calls,
verified by `tests/test_no_bpy_ops.py`). Units: metres.

Video 2 8-parameter procedural interface (`ZipperConfig`):
    size_multiplier      #5 (1.0) or #8 (1.35) gauge scale
    tooth_index          0..4: Standard Excella, Chunky #8, Molded Vislon, Coil, Concealed
    head_index           0..3: Architectural Bar, Luxury D-Ring, Ergonomic Curve, Dual-Way
    open_factor          O in [0.0, 1.0] single source of truth driving slider position + Y-split flare
    flip                 Orientation inversion (reverses curve direction and lateral side)
    tooth_material       PBR material preset name (e.g. "brushed_gunmetal", "polished_nickel", "antique_brass", "matte_black", "gold_pvd")
    head_material        PBR material preset name for slider housing, puller head, and stops
    textile_uv_attribute Name of the continuous tape UV attribute ("textile UV")

Video 1 component topology (all closed outward-oriented meshes, positive signed volume):
    1. Left & right woven textile tapes with beaded inner cord welt and continuous arc-length `textile UV`
    2. Left & right half-pitch staggered interlocking tooth chains (5 distinct profiles)
    3. Diamond-wedge Y-channel slider body + crown bridge + puller head (4 distinct styles, incl. Dual-Way)
    4. Left & right U-clamped top stops + bottom separating box and insertion pin
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


# ------------------------------------------------------------------------------------------------------------------------------------
# 1. Video 2 8-parameter contract & preset registries
# ------------------------------------------------------------------------------------------------------------------------------------
TOOTH_TYPES: Dict[int, Dict[str, Any]] = {
    0: {"name": "Standard Excella", "pitch_m": 0.0030, "width_m": 0.0031, "height_m": 0.0024, "length_m": 0.00142,
        "bevel": 0.24, "profile": "excella"},
    1: {"name": "Chunky #8",        "pitch_m": 0.0036, "width_m": 0.0040, "height_m": 0.0031, "length_m": 0.00170,
        "bevel": 0.16, "profile": "chunky"},
    2: {"name": "Molded Vislon",    "pitch_m": 0.0034, "width_m": 0.0037, "height_m": 0.0028, "length_m": 0.00160,
        "bevel": 0.20, "profile": "vislon"},
    3: {"name": "Coil",             "pitch_m": 0.0022, "width_m": 0.0025, "height_m": 0.0020, "length_m": 0.00102,
        "bevel": 0.40, "profile": "coil"},
    4: {"name": "Concealed",        "pitch_m": 0.0022, "width_m": 0.0021, "height_m": 0.0016, "length_m": 0.00100,
        "bevel": 0.30, "profile": "concealed"},
}

HEAD_TYPES: Dict[int, Dict[str, Any]] = {
    0: {"name": "Architectural Bar", "pull_length_m": 0.024, "pull_width_m": 0.0085, "pull_thick_m": 0.0024, "dual_way": False},
    1: {"name": "Luxury D-Ring",     "pull_length_m": 0.022, "pull_width_m": 0.0170, "pull_thick_m": 0.0028, "dual_way": False},
    2: {"name": "Ergonomic Curve",   "pull_length_m": 0.027, "pull_width_m": 0.0105, "pull_thick_m": 0.0026, "dual_way": False},
    3: {"name": "Dual-Way",          "pull_length_m": 0.023, "pull_width_m": 0.0085, "pull_thick_m": 0.0024, "dual_way": True},
}

MATERIAL_PRESETS: Dict[str, Dict[str, Any]] = {
    "brushed_gunmetal": {"hex": "#626770", "metallic": 1.00, "roughness": 0.20, "anisotropic": 0.72, "coat": 0.08, "bump_scale": 650.0, "bump_strength": 0.035},
    "polished_nickel":  {"hex": "#D4D7DC", "metallic": 1.00, "roughness": 0.12, "anisotropic": 0.25, "coat": 0.10, "bump_scale": 800.0, "bump_strength": 0.01},
    "antique_brass":    {"hex": "#8E7245", "metallic": 0.94, "roughness": 0.36, "anisotropic": 0.40, "coat": 0.05, "bump_scale": 500.0, "bump_strength": 0.06},
    "matte_black":      {"hex": "#17181B", "metallic": 0.45, "roughness": 0.46, "anisotropic": 0.15, "coat": 0.04, "bump_scale": 450.0, "bump_strength": 0.03},
    "gold_pvd":         {"hex": "#D4AF5A", "metallic": 1.00, "roughness": 0.16, "anisotropic": 0.35, "coat": 0.10, "bump_scale": 750.0, "bump_strength": 0.02},
    "vislon_obsidian":  {"hex": "#1B1C20", "metallic": 0.00, "roughness": 0.38, "anisotropic": 0.00, "coat": 0.12, "bump_scale": 300.0, "bump_strength": 0.02},
    "twill_tape_noir":  {"hex": "#1A1B1E", "metallic": 0.00, "roughness": 0.84, "anisotropic": 0.00, "coat": 0.00, "sheen": 0.25},
}

PART_NAMES = ("tape", "teeth", "head", "stops")


@dataclass
class ZipperConfig:
    size_multiplier: float = 1.0        # #5 (1.0) or #8 (1.35) gauge scale
    tooth_index: int = 0                # 0..4: Standard Excella, Chunky #8, Molded Vislon, Coil, Concealed
    head_index: int = 0                 # 0..3: Architectural Bar, Luxury D-Ring, Ergonomic Curve, Dual-Way
    open_factor: float = 0.0            # O in [0.0, 1.0] single source of truth driving slider + Y-split
    flip: bool = False                  # Orientation inversion
    tooth_material: str = "brushed_gunmetal"
    head_material: str = "brushed_gunmetal"
    textile_uv_attribute: str = "textile UV"

    def __post_init__(self) -> None:
        self.size_multiplier = float(self.size_multiplier)
        self.tooth_index = int(self.tooth_index)
        self.head_index = int(self.head_index)
        self.open_factor = float(self.open_factor)
        self.flip = bool(self.flip)
        if self.size_multiplier <= 0.0:
            raise ValueError(f"size_multiplier must be > 0, got {self.size_multiplier}")
        if self.tooth_index not in TOOTH_TYPES:
            raise ValueError(f"tooth_index must be in 0..4, got {self.tooth_index}")
        if self.head_index not in HEAD_TYPES:
            raise ValueError(f"head_index must be in 0..3, got {self.head_index}")
        if not (0.0 <= self.open_factor <= 1.0):
            raise ValueError(f"open_factor must be in [0.0, 1.0], got {self.open_factor}")
        if not self.textile_uv_attribute or not isinstance(self.textile_uv_attribute, str):
            raise ValueError("textile_uv_attribute must be a non-empty string")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ------------------------------------------------------------------------------------------------------------------------------------
# 2. Math & curve parallel-transport framing
# ------------------------------------------------------------------------------------------------------------------------------------
def _smoothstep5(x: np.ndarray | float) -> np.ndarray | float:
    s = np.clip(x, 0.0, 1.0)
    return s * s * s * (10.0 - 15.0 * s + 6.0 * s * s)


def _dsmoothstep5(x: np.ndarray | float) -> np.ndarray | float:
    s = np.clip(x, 0.0, 1.0)
    return 30.0 * s * s * (1.0 - s) * (1.0 - s)


def _norm(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, eps)


def _resample_curve(pts: np.ndarray, normals: Optional[np.ndarray], n_samples: int, flip: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Arc-length resample a 3D guide curve and compute a twist-free Bishop frame (T, B, N).

    T = unit tangent along travel (bottom -> top)
    N = unit outward surface normal (orthogonalised against T)
    B = N x T (lateral binormal pointing toward left tape side)
    """
    P = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
    if len(P) < 2:
        raise ValueError("curve_pts must have at least 2 points")
    N_in = None if normals is None else np.asarray(normals, dtype=np.float64).reshape(-1, 3)
    if flip:
        P = P[::-1].copy()
        if N_in is not None:
            N_in = -N_in[::-1].copy()

    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total_len = float(cum[-1])
    if total_len < 1e-5:
        raise ValueError(f"curve length too small: {total_len}")

    s_grid = np.linspace(0.0, total_len, max(int(n_samples), 4))
    Q = np.column_stack([np.interp(s_grid, cum, P[:, c]) for c in range(3)])

    # Central-difference tangent
    T = np.zeros_like(Q)
    T[1:-1] = Q[2:] - Q[:-2]
    T[0] = Q[1] - Q[0]
    T[-1] = Q[-1] - Q[-2]
    T = _norm(T)

    # Initial outward normal
    if N_in is not None:
        N_raw = np.column_stack([np.interp(s_grid, cum, N_in[:, c]) for c in range(3)])
        N0 = N_raw[0] - np.dot(N_raw[0], T[0]) * T[0]
        if np.linalg.norm(N0) < 1e-8:
            N_in = None
    if N_in is None:
        # Default outward normal faces -Y (Maximus front) unless tangent aligns with -Y
        ref = np.array([0.0, -1.0, 0.0], dtype=np.float64)
        if flip:
            ref = -ref
        if abs(float(np.dot(ref, T[0]))) > 0.92:
            ref = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        N0 = ref - np.dot(ref, T[0]) * T[0]
    N = np.zeros_like(Q)
    N[0] = N0 / max(float(np.linalg.norm(N0)), 1e-12)

    # Parallel transport (Bishop frame) to eliminate torsion flips
    for i in range(1, len(Q)):
        if N_in is not None:
            ni = N_raw[i] - np.dot(N_raw[i], T[i]) * T[i]
            ln = float(np.linalg.norm(ni))
            N[i] = ni / ln if ln > 1e-8 else N[i - 1] - np.dot(N[i - 1], T[i]) * T[i]
            N[i] /= max(float(np.linalg.norm(N[i])), 1e-12)
        else:
            axis = np.cross(T[i - 1], T[i])
            s_ang = float(np.linalg.norm(axis))
            if s_ang < 1e-12:
                ni = N[i - 1]
            else:
                ax = axis / s_ang
                c_ang = float(np.clip(np.dot(T[i - 1], T[i]), -1.0, 1.0))
                ang = math.atan2(s_ang, c_ang)
                K = np.array([[0.0, -ax[2], ax[1]], [ax[2], 0.0, -ax[0]], [-ax[1], ax[0], 0.0]])
                R = np.eye(3) + math.sin(ang) * K + (1.0 - math.cos(ang)) * (K @ K)
                ni = R @ N[i - 1]
            ni = ni - np.dot(ni, T[i]) * T[i]
            N[i] = ni / max(float(np.linalg.norm(ni)), 1e-12)

    B = _norm(np.cross(T, N))
    # Re-orthogonalise N so [B, T, N] is strictly right-handed SO(3) with det = +1 (B x T = +N)
    N = _norm(np.cross(B, T))
    return s_grid, Q, T, B, N, total_len


def _interp_frame(s_val: float, s_grid: np.ndarray, Q: np.ndarray, T: np.ndarray, B: np.ndarray, N: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    q = np.array([np.interp(s_val, s_grid, Q[:, c]) for c in range(3)], dtype=np.float64)
    t = np.array([np.interp(s_val, s_grid, T[:, c]) for c in range(3)], dtype=np.float64)
    n = np.array([np.interp(s_val, s_grid, N[:, c]) for c in range(3)], dtype=np.float64)
    t /= max(float(np.linalg.norm(t)), 1e-12)
    n = n - np.dot(n, t) * t
    n /= max(float(np.linalg.norm(n)), 1e-12)
    b = np.cross(t, n)
    b /= max(float(np.linalg.norm(b)), 1e-12)
    n = np.cross(b, t)
    n /= max(float(np.linalg.norm(n)), 1e-12)
    return q, t, b, n


def _split_profile(
    s: np.ndarray,
    s_slider: float,
    s_top: float,
    scale: float,
    s_lower_slider: Optional[float] = None,
    s_bot: float = 0.0,
    params: Optional[Dict[str, float]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Slider-driven Y-bifurcation:
        C_{L/R}(s; s*) = C_0(s) +/- (w_0 + W_open * sigma((s - s*) / l_throat)) B(s) + h_lift(s) N(s)
    with C2 minimum-jerk sigma(u) = 10 u^3 - 15 u^4 + 6 u^5.
    Returns (dy, ddy_ds, h_lift).
    `params` (unscaled metres, multiplied by `scale`): l_throat, w_open, h_lift, slope_open. Default = legacy profile
    (l 24 mm, w 11.5 mm, h 1.2 mm, slope 0.14), whose tape reaches ~1.3-1.5x stretch in the throat when laid out by parameter;
    ISO_THROAT is the Milestone 1 isometric profile (used with build_zipper_numpy(isometric=True)).
    """
    if params is None:
        l_throat = max(0.024 * scale, 0.014)
        w_open = 0.0115 * scale
        h_max = 0.0012 * scale
        slope_open = 0.14
    else:
        l_throat = max(float(params["l_throat"]) * scale, 0.014)
        w_open = float(params["w_open"]) * scale
        h_max = float(params["h_lift"]) * scale
        slope_open = float(params.get("slope_open", 0.0))

    dy = np.zeros_like(s, dtype=np.float64)
    ddy = np.zeros_like(s, dtype=np.float64)
    hlift = np.zeros_like(s, dtype=np.float64)

    u_hi = (s - s_slider) / l_throat
    m_hi = s > s_slider
    if np.any(m_hi):
        ds = s[m_hi] - s_slider
        sig = _smoothstep5(u_hi[m_hi])
        dsig = _dsmoothstep5(u_hi[m_hi]) / l_throat
        dy[m_hi] = w_open * sig + slope_open * ds * sig
        ddy[m_hi] = (w_open + slope_open * ds) * dsig + slope_open * sig
        hlift[m_hi] = h_max * sig

    if s_lower_slider is not None and s_lower_slider > s_bot + 1e-4:
        u_lo = (s_lower_slider - s) / l_throat
        m_lo = s < s_lower_slider
        if np.any(m_lo):
            ds = s_lower_slider - s[m_lo]
            sig = _smoothstep5(u_lo[m_lo])
            dsig = _dsmoothstep5(u_lo[m_lo]) / l_throat
            dy[m_lo] += 0.65 * w_open * sig + 0.09 * ds * sig
            ddy[m_lo] -= (0.65 * w_open + 0.09 * ds) * dsig + 0.09 * sig
            hlift[m_lo] += 0.5 * h_max * sig
    return dy, ddy, hlift


# Milestone 1 isometric Y-throat (spec docs/MAXIMUS_LAYERED_MATH_SPEC.md §4): 75 mm throat, 26 mm half-opening (wedge half-angle
# atan(26/75) = 19.1 deg), 7 mm lift so a 1.4 mm singlet + 2.5 mm air gap clears when the lapels peel.
ISO_THROAT: Dict[str, float] = {"l_throat": 0.075, "w_open": 0.026, "h_lift": 0.007, "slope_open": 0.0}


def _interp_frames(u: np.ndarray, s_grid: np.ndarray, Q: np.ndarray, T: np.ndarray, B: np.ndarray, N: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Vectorised _interp_frame: positions and re-orthonormalised right-handed frames at parameters u."""
    u = np.asarray(u, dtype=np.float64)
    q = np.column_stack([np.interp(u, s_grid, Q[:, c]) for c in range(3)])
    t = _norm(np.column_stack([np.interp(u, s_grid, T[:, c]) for c in range(3)]))
    n = np.column_stack([np.interp(u, s_grid, N[:, c]) for c in range(3)])
    n = _norm(n - np.einsum("ij,ij->i", n, t)[:, None] * t)
    b = _norm(np.cross(t, n))
    n = _norm(np.cross(b, t))
    return q, t, b, n


def _iso_param_map(side: int, s_grid: np.ndarray, Q: np.ndarray, T: np.ndarray, B: np.ndarray, N: np.ndarray,
                   s_anchor: float, prof: Any, n_fine: int = 4000) -> Tuple[np.ndarray, np.ndarray]:
    """Tape material coordinate sigma(u) of one side, anchored at the slider (sigma(s*) = s*).

    The tape path is P(u) = C0(u) + side * dy(u) B(u) + h(u) N(u). Its true 3D arc length is integrated numerically, so
    laying tape material sigma at u(sigma) keeps it isometric (exact inverse of d sigma / du = |P'(u)|; the spec's
    sqrt(1 - dy'^2 - h'^2) is the first-order form of 1 / sqrt(1 + dy'^2 + h'^2) and ignores the lift-curvature term).
    Returns (u_fine, sigma_fine), both increasing; invert with np.interp(sigma, sigma_fine, u_fine).
    """
    u = np.linspace(float(s_grid[0]), float(s_grid[-1]), int(n_fine))
    q, _, b, n = _interp_frames(u, s_grid, Q, T, B, N)
    dy, _, h = prof(u)
    P = q + side * dy[:, None] * b + h[:, None] * n
    sig = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
    sig = sig - float(np.interp(s_anchor, u, sig)) + float(s_anchor)
    return u, sig


def tape_outer_offset_m(config: Optional[ZipperConfig] = None) -> float:
    """Lateral distance from the chain centreline to the outer (sewn) edge of one tape, closed state."""
    cfg = config if config is not None else ZipperConfig()
    scale = cfg.size_multiplier
    half_chain_w = TOOTH_TYPES[cfg.tooth_index]["width_m"] * scale * 0.78
    if TOOTH_TYPES[cfg.tooth_index]["profile"] == "concealed":
        return -0.35 * half_chain_w + 0.0125 * scale
    return 0.55 * half_chain_w + 0.0125 * scale


def default_jacket_centerline(length_m: float = 0.420, n: int = 64, atlas_npz: Optional[Any] = None, state: str = "rest") -> Tuple[np.ndarray, np.ndarray]:
    """Canonical 420 mm jacket center-front seam centerline C_0(s) and outward surface normals N(s).
    If `atlas_npz` (path or loaded maximus_v2_surface_atlas.npz) is provided, projects C_0(s) directly onto the torso triangle frames.
    """
    u = np.linspace(0.0, 1.0, max(int(n), 8))
    z = 0.94 + float(length_m) * u
    # Anatomical sternum/abdomen sagittal contour (-Y is front on Maximus)
    y = -0.132 - 0.014 * np.sin(math.pi * u) + 0.004 * (u ** 2)
    x = np.zeros_like(u)
    C0 = np.column_stack([x, y, z])
    dy_dz = (-0.014 * math.pi * np.cos(math.pi * u) + 0.008 * u) / max(float(length_m), 1e-6)
    N0 = _norm(np.column_stack([np.zeros_like(u), -np.ones_like(u), dy_dz]))

    if atlas_npz is not None:
        Z = np.load(atlas_npz) if not hasattr(atlas_npz, "files") else atlas_npz
        key_tau = f"{state}|tau" if f"{state}|tau" in Z.files else "rest|tau"
        tau = Z[key_tau].astype(np.float64)
        # Filter anterior chest/abdomen faces (|x| < 0.09, y < -0.05, z in [0.88, 1.45])
        front = (np.abs(tau[:, 0]) < 0.09) & (tau[:, 1] < -0.05) & (tau[:, 2] >= 0.88) & (tau[:, 2] <= 1.45)
        if np.any(front):
            tf = tau[front]
            # Smooth anterior jacket placket envelope (Gaussian window sigma_z = 35 mm over 10th-percentile anterior Y + 12 mm garment offset)
            y_env = np.zeros_like(z)
            for i, zi in enumerate(z):
                band = tf[np.abs(tf[:, 2] - zi) < 0.045]
                if len(band) == 0:
                    band = tf
                y_env[i] = float(np.percentile(band[:, 1], 8.0)) - 0.012
            # Smooth with a wide Gaussian kernel so C0(s) is C^2 smooth with zero zig-zag
            dz_grid = z[:, None] - z[None, :]
            W_kern = np.exp(-0.5 * (dz_grid / 0.045) ** 2)
            W_kern /= W_kern.sum(axis=1, keepdims=True)
            y_smooth = W_kern @ y_env
            dy_dz_s = np.gradient(y_smooth, z)
            N0 = _norm(np.column_stack([np.zeros_like(z), -np.ones_like(z), dy_dz_s]))
            C0 = np.column_stack([np.zeros_like(z), y_smooth, z])
    return C0, N0


# ------------------------------------------------------------------------------------------------------------------------------------
# 3. Closed outward-oriented manifold primitives (local space: X = lateral B, Y = travel T, Z = normal N)
# ------------------------------------------------------------------------------------------------------------------------------------
def _ensure_outward(V: np.ndarray, F: np.ndarray) -> np.ndarray:
    """Flip winding if signed enclosed volume is negative so every solid component is outward-oriented."""
    vol = float(np.einsum("ij,ij->i", V[F[:, 0]], np.cross(V[F[:, 1]], V[F[:, 2]])).sum() / 6.0)
    return F[:, ::-1].copy() if vol < 0.0 else F.copy()


def _extrude_polygon_beveled(poly_xy: np.ndarray, z_lo: float, z_hi: float, bevel: float = 0.15) -> Tuple[np.ndarray, np.ndarray]:
    """Closed beveled prism from a CCW 2D polygon `poly_xy` (K, 2), spanning [z_lo, z_hi] in Z."""
    P = np.asarray(poly_xy, dtype=np.float64)
    # Ensure CCW in XY
    x, y = P[:, 0], P[:, 1]
    if 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) < 0.0:
        P = P[::-1].copy()
    K = len(P)
    cen = P.mean(axis=0)
    h = z_hi - z_lo
    bz = min(bevel * h, 0.35 * h)
    inset = 1.0 - min(bevel * 0.45, 0.30)
    P_in = cen + (P - cen) * inset

    # 4 rings: bottom-cap-ring, lower-body-ring, upper-body-ring, top-cap-ring + 2 cap centers
    rings = [
        np.c_[P_in, np.full(K, z_lo)],
        np.c_[P,    np.full(K, z_lo + bz)],
        np.c_[P,    np.full(K, z_hi - bz)],
        np.c_[P_in, np.full(K, z_hi)],
    ]
    c_bot = np.array([[cen[0], cen[1], z_lo]])
    c_top = np.array([[cen[0], cen[1], z_hi]])
    V = np.vstack(rings + [c_bot, c_top])
    idx_bot = 4 * K
    idx_top = 4 * K + 1

    F: List[List[int]] = []
    # Side quads between consecutive rings
    for r in range(3):
        o0, o1 = r * K, (r + 1) * K
        for i in range(K):
            j = (i + 1) % K
            F.append([o0 + i, o0 + j, o1 + j])
            F.append([o0 + i, o1 + j, o1 + i])
    # Bottom cap (outward normal -Z)
    for i in range(K):
        j = (i + 1) % K
        F.append([idx_bot, j, i])
    # Top cap (outward normal +Z)
    o3 = 3 * K
    for i in range(K):
        j = (i + 1) % K
        F.append([idx_top, o3 + i, o3 + j])
    return V, _ensure_outward(V, np.array(F, dtype=np.int64))


def _box(x0: float, x1: float, y0: float, y1: float, z0: float, z1: float, bevel: float = 0.14) -> Tuple[np.ndarray, np.ndarray]:
    poly = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float64)
    return _extrude_polygon_beveled(poly, z0, z1, bevel=bevel)


def _torus_ring(rx: float, ry: float, tube_r: float, n_seg: int = 28, n_tube: int = 10, d_flat_y: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Closed manifold ring / D-ring in XY plane (for Luxury D-Ring puller)."""
    thetas = np.linspace(0.0, 2.0 * math.pi, n_seg, endpoint=False)
    spine = np.column_stack([rx * np.cos(thetas), ry * np.sin(thetas), np.zeros(n_seg)])
    if d_flat_y is not None:
        spine[:, 1] = np.minimum(spine[:, 1], d_flat_y)

    # Tangents and radial normals in XY
    T = np.roll(spine, -1, axis=0) - np.roll(spine, 1, axis=0)
    T = _norm(T)
    Z = np.tile([0.0, 0.0, 1.0], (n_seg, 1))
    R_out = _norm(np.cross(T, Z))

    phis = np.linspace(0.0, 2.0 * math.pi, n_tube, endpoint=False)
    V_list = []
    for i in range(n_seg):
        for p in phis:
            V_list.append(spine[i] + tube_r * (math.cos(p) * R_out[i] + math.sin(p) * Z[i]))
    V = np.array(V_list, dtype=np.float64)
    F = []
    for i in range(n_seg):
        ni = (i + 1) % n_seg
        for j in range(n_tube):
            nj = (j + 1) % n_tube
            a = i * n_tube + j
            b = ni * n_tube + j
            c = ni * n_tube + nj
            d = i * n_tube + nj
            F.append([a, b, c])
            F.append([a, c, d])
    return V, _ensure_outward(V, np.array(F, dtype=np.int64))


def _concat_meshes(parts: Sequence[Tuple[np.ndarray, np.ndarray]]) -> Tuple[np.ndarray, np.ndarray]:
    Vs, Fs = [], []
    off = 0
    for V, F in parts:
        if len(V) == 0 or len(F) == 0:
            continue
        Vs.append(V)
        Fs.append(F + off)
        off += len(V)
    if not Vs:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.int64)
    return np.vstack(Vs), np.vstack(Fs)


# ------------------------------------------------------------------------------------------------------------------------------------
# 4. Tooth prototypes (Video 1 topology x 5 Video 2 profiles)
# ------------------------------------------------------------------------------------------------------------------------------------
def _tooth_prototype(tooth_index: int, scale: float) -> Tuple[np.ndarray, np.ndarray]:
    """Local tooth mesh for the LEFT chain (+X = outer tape side, -X = interlock nose toward centerline, Y = along zipper, Z = normal)."""
    spec = TOOTH_TYPES[tooth_index]
    w = spec["width_m"] * scale      # total lateral length from tape clamp (+x) to interlock nose (-x)
    h = spec["height_m"] * scale     # thickness along normal Z
    l = spec["length_m"] * scale     # width along travel Y
    bev = spec["bevel"]
    prof = spec["profile"]

    if prof == "excella":
        # Diamond-bevel luxury Excella tooth: forked tape leg (+X) tapering into faceted interlocking head (-X)
        poly = np.array([
            [ 0.55 * w, -0.50 * l],
            [ 0.55 * w,  0.50 * l],
            [ 0.05 * w,  0.46 * l],
            [-0.42 * w,  0.56 * l],
            [-0.62 * w,  0.22 * l],
            [-0.62 * w, -0.22 * l],
            [-0.42 * w, -0.56 * l],
            [ 0.05 * w, -0.46 * l],
        ], dtype=np.float64)
        return _extrude_polygon_beveled(poly, -0.5 * h, 0.5 * h, bevel=bev)

    if prof == "chunky":
        # Architectural heavy #8 squared tooth with proud interlock crown
        poly = np.array([
            [ 0.58 * w, -0.52 * l],
            [ 0.58 * w,  0.52 * l],
            [-0.48 * w,  0.54 * l],
            [-0.64 * w,  0.32 * l],
            [-0.64 * w, -0.32 * l],
            [-0.48 * w, -0.54 * l],
        ], dtype=np.float64)
        return _extrude_polygon_beveled(poly, -0.52 * h, 0.52 * h, bevel=bev)

    if prof == "vislon":
        # Molded acetal Vislon tooth with trapezoidal wings + rounded interlocking nose
        poly = np.array([
            [ 0.56 * w, -0.50 * l],
            [ 0.56 * w,  0.50 * l],
            [ 0.10 * w,  0.42 * l],
            [-0.25 * w,  0.58 * l],
            [-0.60 * w,  0.34 * l],
            [-0.65 * w,  0.00 * l],
            [-0.60 * w, -0.34 * l],
            [-0.25 * w, -0.58 * l],
            [ 0.10 * w, -0.42 * l],
        ], dtype=np.float64)
        return _extrude_polygon_beveled(poly, -0.5 * h, 0.5 * h, bevel=bev)

    if prof == "coil":
        # Rounded nylon coil loop segment
        angs = np.linspace(0.0, 2.0 * math.pi, 10, endpoint=False)
        poly = np.column_stack([0.55 * w * np.cos(angs) - 0.05 * w, 0.48 * l * np.sin(angs)])
        return _extrude_polygon_beveled(poly, -0.48 * h, 0.48 * h, bevel=0.35)

    # prof == "concealed": compact teeth tucked under the reverse tape fold (-Z)
    poly = np.array([
        [ 0.45 * w, -0.46 * l],
        [ 0.45 * w,  0.46 * l],
        [-0.50 * w,  0.46 * l],
        [-0.58 * w,  0.00 * l],
        [-0.50 * w, -0.46 * l],
    ], dtype=np.float64)
    return _extrude_polygon_beveled(poly, -1.15 * h, -0.15 * h, bevel=bev)


# ------------------------------------------------------------------------------------------------------------------------------------
# 5. Slider housing & 4 puller heads (Video 1 + Video 2)
# ------------------------------------------------------------------------------------------------------------------------------------
def _slider_assembly_prototype(head_index: int, tooth_index: int, scale: float, invert_pull: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Local slider body (Y-channel diamond wedge + crown bridge) + puller head at origin."""
    tspec = TOOTH_TYPES[tooth_index]
    hspec = HEAD_TYPES[head_index]
    cw = tspec["width_m"] * scale * 2.25        # closed chain width envelope
    ch = tspec["height_m"] * scale * 1.45       # chain height envelope
    sl_len = 0.0115 * scale                     # slider body length along Y

    # Upper flange (diamond-tapered shield on +Z)
    flange_xy = np.array([
        [-0.56 * cw,  0.50 * sl_len],
        [-0.40 * cw, -0.50 * sl_len],
        [ 0.40 * cw, -0.50 * sl_len],
        [ 0.56 * cw,  0.50 * sl_len],
        [ 0.46 * cw,  0.58 * sl_len],
        [-0.46 * cw,  0.58 * sl_len],
    ], dtype=np.float64)
    upper_flange = _extrude_polygon_beveled(flange_xy, 0.38 * ch, 0.96 * ch, bevel=0.22)
    lower_flange = _extrude_polygon_beveled(flange_xy * np.array([0.96, 0.96]), -0.92 * ch, -0.38 * ch, bevel=0.20)
    # Central Y-splitter diamond post joining upper & lower flanges
    diamond_xy = np.array([
        [ 0.00,       0.48 * sl_len],
        [-0.17 * cw,  0.14 * sl_len],
        [ 0.00,      -0.18 * sl_len],
        [ 0.17 * cw,  0.14 * sl_len],
    ], dtype=np.float64)
    post = _extrude_polygon_beveled(diamond_xy, -0.55 * ch, 0.55 * ch, bevel=0.10)
    # Side guide rails
    rail_L = _box( 0.40 * cw,  0.55 * cw, -0.44 * sl_len, 0.44 * sl_len, -0.45 * ch, 0.45 * ch, bevel=0.12)
    rail_R = _box(-0.55 * cw, -0.40 * cw, -0.44 * sl_len, 0.44 * sl_len, -0.45 * ch, 0.45 * ch, bevel=0.12)
    # Crown bridge / bail arch on top of upper flange (+Z)
    bridge_base_1 = _box(-0.16 * cw, 0.16 * cw,  0.14 * sl_len,  0.34 * sl_len, 0.90 * ch, 1.55 * ch, bevel=0.15)
    bridge_base_2 = _box(-0.16 * cw, 0.16 * cw, -0.34 * sl_len, -0.14 * sl_len, 0.90 * ch, 1.55 * ch, bevel=0.15)
    bridge_top    = _box(-0.16 * cw, 0.16 * cw, -0.34 * sl_len,  0.34 * sl_len, 1.38 * ch, 1.72 * ch, bevel=0.22)

    # Puller head hanging downward (-Y by default, or +Y when dual-way bottom slider is inverted)
    pl = hspec["pull_length_m"] * scale
    pw = hspec["pull_width_m"] * scale
    pt = hspec["pull_thick_m"] * scale
    dir_y = 1.0 if invert_pull else -1.0
    pivot = np.array([0.0, 0.0, 1.32 * ch], dtype=np.float64)
    hang_tilt = math.radians(-10.0 * dir_y)     # slight natural clearance tilt off the tape

    parts: List[Tuple[np.ndarray, np.ndarray]] = [
        upper_flange, lower_flange, post, rail_L, rail_R, bridge_base_1, bridge_base_2, bridge_top
    ]

    if head_index in (0, 3):
        # Architectural Bar (also used on Dual-Way)
        bar_xy = np.array([
            [-0.42 * pw,  0.00],
            [ 0.42 * pw,  0.00],
            [ 0.50 * pw,  dir_y * 0.22 * pl],
            [ 0.46 * pw,  dir_y * pl],
            [-0.46 * pw,  dir_y * pl],
            [-0.50 * pw,  dir_y * 0.22 * pl],
        ], dtype=np.float64)
        Vp, Fp = _extrude_polygon_beveled(bar_xy, -0.5 * pt, 0.5 * pt, bevel=0.22)
    elif head_index == 1:
        # Luxury D-Ring
        rx = 0.48 * pw
        ry = 0.48 * pl
        Vp, Fp = _torus_ring(rx, ry, 0.48 * pt, n_seg=28, n_tube=10, d_flat_y=0.35 * ry)
        Vp[:, 1] = dir_y * (0.50 * pl - Vp[:, 1])
    else:
        # head_index == 2: Ergonomic Curve (waisted sculpted thumb puller)
        curve_xy = np.array([
            [-0.36 * pw,  0.00],
            [ 0.36 * pw,  0.00],
            [ 0.28 * pw,  dir_y * 0.25 * pl],
            [ 0.52 * pw,  dir_y * 0.62 * pl],
            [ 0.44 * pw,  dir_y * 0.94 * pl],
            [ 0.20 * pw,  dir_y * 1.00 * pl],
            [-0.20 * pw,  dir_y * 1.00 * pl],
            [-0.44 * pw,  dir_y * 0.94 * pl],
            [-0.52 * pw,  dir_y * 0.62 * pl],
            [-0.28 * pw,  dir_y * 0.25 * pl],
        ], dtype=np.float64)
        Vp, Fp = _extrude_polygon_beveled(curve_xy, -0.5 * pt, 0.5 * pt, bevel=0.26)
        # Subtle ergonomic thumb concavity along Z
        u_len = np.abs(Vp[:, 1]) / max(pl, 1e-6)
        Vp[:, 2] += 0.35 * pt * np.sin(math.pi * np.clip(u_len, 0.0, 1.0))

    # Tilt puller around X axis at pivot and position on bridge
    ca, sa = math.cos(hang_tilt), math.sin(hang_tilt)
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, ca, -sa], [0.0, sa, ca]])
    Vp = (Vp @ Rx.T) + pivot
    parts.append((Vp, Fp))
    return _concat_meshes(parts)


# ------------------------------------------------------------------------------------------------------------------------------------
# 6. Top stops, bottom separating box/pin, and woven textile tapes
# ------------------------------------------------------------------------------------------------------------------------------------
def _top_stop_prototype(scale: float, tooth_index: int) -> Tuple[np.ndarray, np.ndarray]:
    ts = TOOTH_TYPES[tooth_index]
    w = ts["width_m"] * scale * 1.05
    h = ts["height_m"] * scale * 1.18
    l = ts["length_m"] * scale * 1.55
    return _box(-0.45 * w, 0.58 * w, -0.5 * l, 0.5 * l, -0.5 * h, 0.5 * h, bevel=0.20)


def _bottom_stop_prototype(scale: float, tooth_index: int) -> Tuple[np.ndarray, np.ndarray]:
    ts = TOOTH_TYPES[tooth_index]
    cw = ts["width_m"] * scale * 2.15
    ch = ts["height_m"] * scale * 1.35
    bl = 0.0085 * scale
    box_body = _box(-0.52 * cw, 0.52 * cw, -bl, 0.15 * bl, -0.55 * ch, 0.55 * ch, bevel=0.18)
    pin_sleeve = _box(0.06 * cw, 0.48 * cw, 0.10 * bl, 0.65 * bl, -0.42 * ch, 0.42 * ch, bevel=0.16)
    return _concat_meshes([box_body, pin_sleeve])


def _build_tape_ribbon(
    side: int,
    s_grid: np.ndarray,
    Q: np.ndarray,
    T: np.ndarray,
    B: np.ndarray,
    N: np.ndarray,
    dy: np.ndarray,
    hlift: np.ndarray,
    half_chain_w: float,
    scale: float,
    concealed: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build one closed manifold textile tape ribbon (side = +1 left, -1 right) with beaded cord edge, top-stitch channel & continuous `textile UV`."""
    tape_w = 0.0125 * scale
    tape_t = 0.00065 * scale
    cord_r = 0.00095 * scale
    uv_repeat = 0.012 * scale  # 12 mm twill repeat per UV unit

    if concealed:
        x_in = -0.35 * half_chain_w
        prof_xz = np.array([
            [x_in,                 0.35 * tape_t],
            [x_in + 0.25 * tape_w, 0.50 * tape_t],
            [x_in + 0.65 * tape_w, 0.45 * tape_t],
            [x_in + 1.00 * tape_w, 0.42 * tape_t],
            [x_in + 1.00 * tape_w, -0.45 * tape_t],
            [x_in + 0.65 * tape_w, -0.48 * tape_t],
            [x_in + 0.25 * tape_w, -0.55 * tape_t],
            [x_in,                 -0.45 * tape_t],
        ], dtype=np.float64)
    else:
        x0 = 0.55 * half_chain_w
        # Includes recessed top-stitch channel at x0 + 0.26 * tape_w
        prof_xz = np.array([
            [x0 - 0.35 * cord_r,  0.00],
            [x0,                  cord_r],
            [x0 + 0.14 * tape_w,  0.52 * tape_t],
            [x0 + 0.26 * tape_w,  0.34 * tape_t],
            [x0 + 0.36 * tape_w,  0.52 * tape_t],
            [x0 + 0.65 * tape_w,  0.50 * tape_t],
            [x0 + 1.00 * tape_w,  0.46 * tape_t],
            [x0 + 1.00 * tape_w, -0.46 * tape_t],
            [x0 + 0.52 * tape_w, -0.50 * tape_t],
            [x0 + 0.14 * tape_w, -0.52 * tape_t],
            [x0,                 -cord_r],
        ], dtype=np.float64)

    if side < 0:
        prof_xz = prof_xz.copy()
        prof_xz[:, 0] = -prof_xz[:, 0]
        prof_xz = prof_xz[::-1].copy()

    K = len(prof_xz)
    M = len(s_grid)
    p_seg = np.linalg.norm(np.diff(np.vstack([prof_xz, prof_xz[:1]]), axis=0), axis=1)
    u_prof = np.concatenate([[0.0], np.cumsum(p_seg[:-1])]) / max(float(p_seg.sum()), 1e-12)

    V = np.zeros((M * K + 2, 3), dtype=np.float64)
    UV = np.zeros((M * K + 2, 2), dtype=np.float64)
    for i in range(M):
        base_pt = Q[i] + side * dy[i] * B[i] + hlift[i] * N[i]
        v_coord = float(s_grid[i]) / uv_repeat
        for k in range(K):
            V[i * K + k] = base_pt + prof_xz[k, 0] * B[i] + prof_xz[k, 1] * N[i]
            UV[i * K + k] = (u_prof[k], v_coord)

    c_start = M * K
    c_end = M * K + 1
    V[c_start] = V[:K].mean(axis=0)
    UV[c_start] = (0.5, float(s_grid[0]) / uv_repeat)
    V[c_end] = V[(M - 1) * K:M * K].mean(axis=0)
    UV[c_end] = (0.5, float(s_grid[-1]) / uv_repeat)

    F: List[List[int]] = []
    for i in range(M - 1):
        r0, r1 = i * K, (i + 1) * K
        for k in range(K):
            kn = (k + 1) % K
            F.append([r0 + k, r1 + k, r1 + kn])
            F.append([r0 + k, r1 + kn, r0 + kn])
    for k in range(K):
        kn = (k + 1) % K
        F.append([c_start, kn, k])
        o_end = (M - 1) * K
        F.append([c_end, o_end + k, o_end + kn])

    F_arr = _ensure_outward(V, np.array(F, dtype=np.int64))
    return V, F_arr, UV


# ------------------------------------------------------------------------------------------------------------------------------------
# 7. Full pure-NumPy zipper builder (rigid 4x4 SE(3) tooth transforms + closed-zone clearance gate)
# ------------------------------------------------------------------------------------------------------------------------------------
def _se3_matrix(origin: np.ndarray, B: np.ndarray, T: np.ndarray, N: np.ndarray, yaw_rad: float = 0.0) -> np.ndarray:
    if abs(yaw_rad) > 1e-12:
        c, s = math.cos(yaw_rad), math.sin(yaw_rad)
        Bp = c * B - s * T
        Tp = s * B + c * T
    else:
        Bp, Tp = B, T
    M = np.eye(4, dtype=np.float64)
    M[:3, :3] = np.column_stack([Bp, Tp, N])
    M[:3, 3] = origin
    return M


def _place_local(V_loc: np.ndarray, origin: np.ndarray, B: np.ndarray, T: np.ndarray, N: np.ndarray, yaw_rad: float = 0.0) -> np.ndarray:
    """Transform local (X=B, Y=T, Z=N) mesh into world space via rigid SE(3) transform (det(R) == +1)."""
    M = _se3_matrix(origin, B, T, N, yaw_rad=yaw_rad)
    return V_loc @ M[:3, :3].T + M[:3, 3]


def build_zipper_numpy(
    curve_pts: np.ndarray,
    config: Optional[ZipperConfig] = None,
    curve_normals: Optional[np.ndarray] = None,
    tape_samples: int = 80,
    isometric: bool = False,
) -> Dict[str, Any]:
    """Generate complete clean-room zipper geometry in pure NumPy.

    Returns dict with per-component meshes (`parts`: tape/teeth/head/stops), merged mesh (`V`, `F`, `uv`, `part_ids`),
    rigid 4x4 SE(3) tooth transforms, `clearance_gate`, `config`, `slider_state`, and manifold/volume sanity `stats`.

    isometric=True (Milestone 1): ISO_THROAT profile, and every tape row and tooth sits at its tape material coordinate
    (arc length along the tape, anchored at the slider), so the tape keeps its length through the Y-throat and the tooth pitch
    stays exact; the open part contracts along the seam instead of stretching. Also returns `tape_edges` (outer sewn edge of
    each tape per material row) for lapel peeling. `G_ZIP_CLEAR` (clearance + tape stretch) is reported in both modes.
    """
    cfg = config if config is not None else ZipperConfig()
    scale = cfg.size_multiplier
    tspec = TOOTH_TYPES[cfg.tooth_index]
    hspec = HEAD_TYPES[cfg.head_index]

    s_grid, Q, T, B, N, total_len = _resample_curve(curve_pts, curve_normals, n_samples=tape_samples, flip=cfg.flip)

    margin_bot = min(0.010 * scale, 0.08 * total_len)
    margin_top = min(0.008 * scale, 0.08 * total_len)
    s_bot = margin_bot
    s_top = max(total_len - margin_top, s_bot + 0.005)
    usable = s_top - s_bot

    # Safe open_factor clamp so slider throat never collides with bottom stop
    stop_clear_m = min(0.006 * scale, 0.15 * usable)
    max_safe_open = max(0.0, 1.0 - stop_clear_m / max(usable, 1e-6))
    eff_open = min(cfg.open_factor, max_safe_open)

    # Single source of truth: s* = s_bot + usable * (1 - O)
    s_slider = s_bot + (1.0 - eff_open) * usable
    s_lower_slider: Optional[float] = None
    if hspec["dual_way"]:
        s_lower_slider = s_bot + min(0.22 * eff_open, 0.35) * usable
        s_slider = max(s_slider, s_lower_slider + 0.016 * scale)
        s_slider = min(s_slider, s_top)

    prof_params = ISO_THROAT if isometric else None

    def prof(s_arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return _split_profile(np.atleast_1d(np.asarray(s_arr, dtype=np.float64)), s_slider, s_top, scale,
                              s_lower_slider=s_lower_slider, s_bot=s_bot, params=prof_params)

    if isometric:
        uL_f, sL_f = _iso_param_map(+1, s_grid, Q, T, B, N, s_slider, prof)
        uR_f, sR_f = _iso_param_map(-1, s_grid, Q, T, B, N, s_slider, prof)
        to_u = {+1: (lambda sig: np.interp(np.asarray(sig, dtype=np.float64), sL_f, uL_f)),
                -1: (lambda sig: np.interp(np.asarray(sig, dtype=np.float64), sR_f, uR_f))}
    else:
        to_u = {+1: (lambda sig: np.asarray(sig, dtype=np.float64)), -1: (lambda sig: np.asarray(sig, dtype=np.float64))}

    dy_grid, _, hlift_grid = prof(s_grid)
    half_chain_w = tspec["width_m"] * scale * 0.78
    concealed = tspec["profile"] == "concealed"

    # 1. Left & right textile tapes with `textile UV` (UV always follows the tape material coordinate s_grid)
    if isometric:
        side_rows = {}
        for side in (+1, -1):
            u_rows = to_u[side](s_grid)
            q_r, t_r, b_r, n_r = _interp_frames(u_rows, s_grid, Q, T, B, N)
            dy_r, _, h_r = prof(u_rows)
            side_rows[side] = (q_r, t_r, b_r, n_r, dy_r, h_r)
        Vt_L, Ft_L, UVt_L = _build_tape_ribbon(+1, s_grid, *side_rows[+1], half_chain_w, scale, concealed)
        Vt_R, Ft_R, UVt_R = _build_tape_ribbon(-1, s_grid, *side_rows[-1], half_chain_w, scale, concealed)
    else:
        Vt_L, Ft_L, UVt_L = _build_tape_ribbon(+1, s_grid, Q, T, B, N, dy_grid, hlift_grid, half_chain_w, scale, concealed)
        Vt_R, Ft_R, UVt_R = _build_tape_ribbon(-1, s_grid, Q, T, B, N, dy_grid, hlift_grid, half_chain_w, scale, concealed)
    V_tape, F_tape = _concat_meshes([(Vt_L, Ft_L), (Vt_R, Ft_R)])
    UV_tape = np.vstack([UVt_L, UVt_R])

    # 2. Staggered interlocking tooth chains (exact half-pitch p/2 stagger, rigid 4x4 SE(3) placement ONLY)
    pitch = tspec["pitch_m"] * scale
    tooth_len = tspec["length_m"] * scale
    Vt_proto_L, Ft_proto_L = _tooth_prototype(cfg.tooth_index, scale)
    # Opposing right tooth is rotated 180 deg around Z normal (rigid SO(3), det = +1, zero reflection)
    R180 = np.array([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    Vt_proto_R = Vt_proto_L @ R180.T
    Ft_proto_R = Ft_proto_L.copy()

    s_start_teeth = s_bot + 0.003 * scale
    s_end_teeth = max(s_top - 0.002 * scale, s_start_teeth + pitch)
    s_left = np.arange(s_start_teeth, s_end_teeth, pitch)
    s_right = np.arange(s_start_teeth + 0.5 * pitch, s_end_teeth, pitch)

    u_left = to_u[+1](s_left)
    u_right = to_u[-1](s_right)
    dy_L, ddy_L, hlift_L = prof(u_left)
    dy_R, ddy_R, hlift_R = prof(u_right)

    tooth_meshes: List[Tuple[np.ndarray, np.ndarray]] = []
    transforms_L: List[np.ndarray] = []
    transforms_R: List[np.ndarray] = []
    tooth_seat = 0.56 * half_chain_w
    for sv, dyv, ddyv, hlv in zip(u_left, dy_L, ddy_L, hlift_L):
        q, t, b, n = _interp_frame(float(sv), s_grid, Q, T, B, N)
        yaw = -math.atan(float(ddyv))
        origin = q + (tooth_seat + float(dyv)) * b + float(hlv) * n
        M_se3 = _se3_matrix(origin, b, t, n, yaw_rad=yaw)
        transforms_L.append(M_se3)
        tooth_meshes.append((Vt_proto_L @ M_se3[:3, :3].T + M_se3[:3, 3], Ft_proto_L))
    for sv, dyv, ddyv, hlv in zip(u_right, dy_R, ddy_R, hlift_R):
        q, t, b, n = _interp_frame(float(sv), s_grid, Q, T, B, N)
        yaw = math.atan(float(ddyv))
        origin = q - (tooth_seat + float(dyv)) * b + float(hlv) * n
        M_se3 = _se3_matrix(origin, b, t, n, yaw_rad=yaw)
        # Include the 180-deg opposing rotation in the instance transform so a single prototype mesh instances both L and R
        M_inst = M_se3.copy(); M_inst[:3, :3] = M_se3[:3, :3] @ R180
        transforms_R.append(M_inst)
        tooth_meshes.append((Vt_proto_L @ M_inst[:3, :3].T + M_inst[:3, 3], Ft_proto_R))
    V_teeth, F_teeth = _concat_meshes(tooth_meshes)

    # Closed-zone inter-tooth clearance gate (s < s_slider): half-pitch center spacing p/2 minus tooth thickness tooth_len
    closed_L = s_left[s_left < s_slider - 0.5 * pitch]
    closed_R = s_right[s_right < s_slider - 0.5 * pitch]
    if len(closed_L) and len(closed_R):
        min_center_ds = float(np.abs(closed_L[:, None] - closed_R[None, :]).min())
        min_clear_mm = (min_center_ds - tooth_len) * 1e3
    else:
        min_clear_mm = (0.5 * pitch - tooth_len) * 1e3
    all_M = np.stack(transforms_L + transforms_R) if (transforms_L or transforms_R) else np.eye(4)[None]
    dets = np.linalg.det(all_M[:, :3, :3])
    orth_err = float(np.abs(np.einsum("nij,nkj->nik", all_M[:, :3, :3], all_M[:, :3, :3]) - np.eye(3)).max())
    clearance_gate = {
        "pitch_mm": round(pitch * 1e3, 4),
        "half_pitch_stagger_mm": round(0.5 * pitch * 1e3, 4),
        "tooth_along_track_mm": round(tooth_len * 1e3, 4),
        "min_closed_intertooth_clearance_mm": round(min_clear_mm, 4),
        "limit_min_mm": 0.05,
        "rigid_se3_only": bool(np.allclose(dets, 1.0, atol=1e-9) and orth_err < 1e-9),
        "so3_det_min": round(float(dets.min()), 9),
        "so3_orth_max_err": orth_err,
        "pass": bool(min_clear_mm >= 0.05 and np.allclose(dets, 1.0, atol=1e-9)),
    }

    # 3. Slider body & puller head(s)
    head_meshes: List[Tuple[np.ndarray, np.ndarray]] = []
    Vh_main, Fh_main = _slider_assembly_prototype(cfg.head_index, cfg.tooth_index, scale, invert_pull=False)
    qs, ts_, bs, ns = _interp_frame(float(s_slider), s_grid, Q, T, B, N)
    head_meshes.append((_place_local(Vh_main, qs, bs, ts_, ns), Fh_main))
    if s_lower_slider is not None:
        Vh_lo, Fh_lo = _slider_assembly_prototype(cfg.head_index, cfg.tooth_index, scale, invert_pull=True)
        ql, tl, bl, nl = _interp_frame(float(s_lower_slider), s_grid, Q, T, B, N)
        head_meshes.append((_place_local(Vh_lo, ql, bl, tl, nl), Fh_lo))
    V_head, F_head = _concat_meshes(head_meshes)

    # 4. Top stops (follow Y-split at s_top) & bottom separating box/pin
    stop_meshes: List[Tuple[np.ndarray, np.ndarray]] = []
    Vts_L, Fts_L = _top_stop_prototype(scale, cfg.tooth_index)
    Vts_R = Vts_L.copy(); Vts_R[:, 0] = -Vts_R[:, 0]
    Fts_R = _ensure_outward(Vts_R, Fts_L[:, ::-1])
    u_top_L = float(to_u[+1](np.array([s_top]))[0])
    u_top_R = float(to_u[-1](np.array([s_top]))[0])
    dy_top, ddy_top, hlift_top = prof(np.array([u_top_L]))
    dy_topR, ddy_topR, hlift_topR = prof(np.array([u_top_R]))
    q_top, t_top, b_top, n_top = _interp_frame(u_top_L, s_grid, Q, T, B, N)
    q_topR, t_topR, b_topR, n_topR = _interp_frame(u_top_R, s_grid, Q, T, B, N)
    stop_meshes.append((_place_local(Vts_L, q_top + (tooth_seat + float(dy_top[0])) * b_top + float(hlift_top[0]) * n_top, b_top, t_top, n_top, yaw_rad=-math.atan(float(ddy_top[0]))), Fts_L))
    stop_meshes.append((_place_local(Vts_R, q_topR - (tooth_seat + float(dy_topR[0])) * b_topR + float(hlift_topR[0]) * n_topR, b_topR, t_topR, n_topR, yaw_rad=math.atan(float(ddy_topR[0]))), Fts_R))

    Vbs, Fbs = _bottom_stop_prototype(scale, cfg.tooth_index)
    q_bot, t_bot, b_bot, n_bot = _interp_frame(float(s_bot), s_grid, Q, T, B, N)
    stop_meshes.append((_place_local(Vbs, q_bot, b_bot, t_bot, n_bot), Fbs))
    V_stops, F_stops = _concat_meshes(stop_meshes)

    parts_data = {
        "tape":  {"V": V_tape,  "F": F_tape,  "uv": UV_tape,                         "material": "twill_tape_noir"},
        "teeth": {"V": V_teeth, "F": F_teeth, "uv": np.zeros((len(V_teeth), 2)),     "material": cfg.tooth_material},
        "head":  {"V": V_head,  "F": F_head,  "uv": np.zeros((len(V_head), 2)),      "material": cfg.head_material},
        "stops": {"V": V_stops, "F": F_stops, "uv": np.zeros((len(V_stops), 2)),     "material": cfg.head_material},
    }

    V_all, F_all = _concat_meshes([(parts_data[p]["V"], parts_data[p]["F"]) for p in PART_NAMES])
    UV_all = np.vstack([parts_data[p]["uv"] for p in PART_NAMES])
    part_ids = np.concatenate([np.full(len(parts_data[p]["F"]), i, dtype=np.int32) for i, p in enumerate(PART_NAMES)])

    # Tape stretch along every material line of both ribbons (row i -> i+1 at a fixed profile point), and the real tooth pitch
    n_prof = (len(Vt_L) - 2) // len(s_grid)
    ds_mat = np.diff(s_grid)
    stretch = []
    for Vr in (Vt_L, Vt_R):
        rows = Vr[: len(s_grid) * n_prof].reshape(len(s_grid), n_prof, 3)
        stretch.append(np.linalg.norm(np.diff(rows, axis=0), axis=2) / ds_mat[:, None])
    stretch_all = np.concatenate([x.ravel() for x in stretch])
    pitch_ratio = []
    for TM in (transforms_L, transforms_R):
        if len(TM) > 1:
            o = np.stack(TM)[:, :3, 3]
            pitch_ratio.append(np.linalg.norm(np.diff(o, axis=0), axis=1) / pitch)
    pr = np.concatenate(pitch_ratio) if pitch_ratio else np.ones(1)
    open_rows = s_grid[1:] > s_slider
    rowsL = Vt_L[: len(s_grid) * n_prof].reshape(len(s_grid), n_prof, 3)
    rowsR = Vt_R[: len(s_grid) * n_prof].reshape(len(s_grid), n_prof, 3)
    k_outer = int(np.argmax(np.abs(np.einsum("kj,j->k", rowsL[0] - Q[0], B[0]))))
    tape_edges = {
        "s_material": s_grid.copy(),
        "side_plus": rowsL[:, k_outer, :].copy(),
        "side_minus": rowsR[:, k_outer, :].copy(),
        "Q_closed": Q.copy(),
        "B_closed": B.copy(),
        "N_closed": N.copy(),
        "outer_offset_m": tape_outer_offset_m(cfg),
    }
    tape_stretch_max = float(stretch_all.max())
    g_zip_clear = {
        "isometric": bool(isometric),
        "min_closed_intertooth_clearance_mm": clearance_gate["min_closed_intertooth_clearance_mm"],
        "tape_stretch_max": round(tape_stretch_max, 5),
        "tape_stretch_max_open_zone": round(float(np.concatenate([x[open_rows].ravel() for x in stretch]).max()), 5) if open_rows.any() else 1.0,
        "tape_compression_min": round(float(stretch_all.min()), 5),
        "tooth_pitch_ratio_min": round(float(pr.min()), 5),
        "tooth_pitch_ratio_max": round(float(pr.max()), 5),
        "limit_tape_stretch": 1.01,
        "pass": bool(clearance_gate["pass"] and tape_stretch_max <= 1.01),
    }

    areas = 0.5 * np.linalg.norm(np.cross(V_all[F_all[:, 1]] - V_all[F_all[:, 0]], V_all[F_all[:, 2]] - V_all[F_all[:, 0]]), axis=1)
    vols = {
        p: round(float(np.einsum("ij,ij->i", parts_data[p]["V"][parts_data[p]["F"][:, 0]],
                                 np.cross(parts_data[p]["V"][parts_data[p]["F"][:, 1]], parts_data[p]["V"][parts_data[p]["F"][:, 2]])).sum() / 6.0) * 1e9, 4)
        for p in PART_NAMES
    }

    return {
        "config": cfg.to_dict(),
        "tooth_type": tspec["name"],
        "head_type": hspec["name"],
        "textile_uv_attribute": cfg.textile_uv_attribute,
        "parts": parts_data,
        "tooth_prototype": {"V": Vt_proto_L, "F": Ft_proto_L},
        "tooth_transforms_L": np.stack(transforms_L) if transforms_L else np.zeros((0, 4, 4)),
        "tooth_transforms_R": np.stack(transforms_R) if transforms_R else np.zeros((0, 4, 4)),
        "slider_prototype": {"V": Vh_main, "F": Fh_main},
        "clearance_gate": clearance_gate,
        "G_ZIP_CLEAR": g_zip_clear,
        "tape_edges": tape_edges,
        "V": V_all,
        "F": F_all,
        "uv": UV_all,
        "part_ids": part_ids,
        "slider_state": {
            "curve_length_m": round(total_len, 5),
            "s_bot_m": round(float(s_bot), 5),
            "s_top_m": round(float(s_top), 5),
            "s_slider_m": round(float(s_slider), 5),
            "s_lower_slider_m": None if s_lower_slider is None else round(float(s_lower_slider), 5),
            "effective_open_factor": round(float(eff_open), 5),
            "slider_pos_m": [round(float(x), 5) for x in qs],
            "top_split_half_gap_mm": round(float(dy_top[0]) * 1e3, 3),
            "teeth_count_left": int(len(s_left)),
            "teeth_count_right": int(len(s_right)),
        },
        "stats": {
            "vertices": int(len(V_all)),
            "triangles": int(len(F_all)),
            "degenerate_triangles": int((areas <= 1e-14).sum()),
            "signed_volume_mm3_by_part": vols,
            "all_parts_positive_volume": all(v > 0.0 for v in vols.values()),
        },
    }


# ------------------------------------------------------------------------------------------------------------------------------------
# 8. Rigid trim transport on deforming garments (GG-30 triangle frames via canon_geometry_kernels)
# ------------------------------------------------------------------------------------------------------------------------------------
def bind_zipper_to_mesh(K: Any, zipper_data: Dict[str, Any], V_host: np.ndarray, F_host: np.ndarray) -> Dict[str, Any]:
    """Bind rigid zipper components (teeth, head, stops) and tape vertices to nearest host mesh faces via GG-30 triangle frames.

    Rigid metal pieces stay rigid (no scaling/shearing of metal teeth or slider under cloth stretch); tape follows local frame scale.
    """
    from scipy.spatial import cKDTree
    frames = K.compute_triangle_frames(V_host, F_host)
    tree = cKDTree(frames["tau"])
    bindings = {}
    for pname in PART_NAMES:
        Vp = zipper_data["parts"][pname]["V"]
        _, fids = tree.query(Vp)
        fids = np.asarray(fids, dtype=np.int64)
        R = frames["R"][fids]
        tau = frames["tau"][fids]
        if pname == "tape":
            bindings[pname] = {"mode": "scaled", "face_ids": fids, "local": K.bind_elements(Vp, fids, frames)}
        else:
            # Rigid offset in metres (unscaled by k_j so metal gauge never stretches!)
            rel = np.einsum("nba,nb->na", R, Vp - tau)
            bindings[pname] = {"mode": "rigid", "face_ids": fids, "offset_local_m": rel}
    return bindings


def transport_zipper(K: Any, bindings: Dict[str, Any], V_host_new: np.ndarray, F_host: np.ndarray) -> Dict[str, np.ndarray]:
    """Transport bound zipper parts onto deformed garment mesh `(V_host_new, F_host)`."""
    frames = K.compute_triangle_frames(V_host_new, F_host)
    out = {}
    for pname, b in bindings.items():
        fids = b["face_ids"]
        if b["mode"] == "scaled":
            out[pname] = K.transform_bound_elements(b["local"], fids, frames)["position"]
        else:
            R = frames["R"][fids]
            tau = frames["tau"][fids]
            out[pname] = np.einsum("nab,nb->na", R, b["offset_local_m"]) + tau
    return out


# ------------------------------------------------------------------------------------------------------------------------------------
# 9. Blender 5.2 data-API builder + Geometry Nodes 8-parameter interface (ZERO bpy.ops)
# ------------------------------------------------------------------------------------------------------------------------------------
def bind_centerline_to_atlas(K: Any, curve_pts: np.ndarray, atlas_npz: Any, state: str = "rest") -> Dict[str, Any]:
    """Bind a 3D zipper centerline C_0(s) directly to `maximus_v2_surface_atlas.npz` triangle frames (R_j, k_j, tau_j)."""
    from scipy.spatial import cKDTree
    Z = np.load(atlas_npz) if not hasattr(atlas_npz, "files") else atlas_npz
    frames = {
        "R": Z[f"{state}|R"].astype(np.float64),
        "k": Z[f"{state}|k"].astype(np.float64),
        "tau": Z[f"{state}|tau"].astype(np.float64),
        "faces": Z["faces"].astype(np.int64),
    }
    tree = cKDTree(frames["tau"])
    _, fids = tree.query(np.asarray(curve_pts, dtype=np.float64))
    fids = np.asarray(fids, dtype=np.int64)
    local = K.bind_elements(curve_pts, fids, frames)
    return {"face_ids": fids, "local": local, "source_state": state}


def transport_centerline_from_atlas(K: Any, binding: Dict[str, Any], atlas_npz: Any, target_state: str = "theta_fit") -> Tuple[np.ndarray, np.ndarray]:
    """Transport a bound centerline C_0(s) and its surface normal N(s) to another surface atlas pose state."""
    Z = np.load(atlas_npz) if not hasattr(atlas_npz, "files") else atlas_npz
    frames_new = {
        "R": Z[f"{target_state}|R"].astype(np.float64),
        "k": Z[f"{target_state}|k"].astype(np.float64),
        "tau": Z[f"{target_state}|tau"].astype(np.float64),
        "faces": Z["faces"].astype(np.int64),
    }
    fids = binding["face_ids"]
    C_new = K.transform_bound_elements(binding["local"], fids, frames_new)["position"]
    N_new = _norm(frames_new["R"][fids, :, 0])
    return C_new, N_new


# ------------------------------------------------------------------------------------------------------------------------------------
# 8b. Milestone 1: zipper on the draped layer-2 seam + heat-geodesic lapel peeling over layer 1 (spec section 4)
# ------------------------------------------------------------------------------------------------------------------------------------
def extract_seam_centerline(
    V_jacket: np.ndarray,
    seam_L_idx: np.ndarray,
    seam_R_idx: np.ndarray,
    V_body: np.ndarray,
    F_body: np.ndarray,
    smooth_window: int = 2,
    sigma_c0_m: float = 0.020,
    sigma_n0_m: float = 0.060,
) -> Tuple[np.ndarray, np.ndarray]:
    """Zipper centreline C0(s) = (V2[dOmega_L] + V2[dOmega_R]) / 2 from the draped closed jacket, plus outward normals N0(s).

    seam_L_idx / seam_R_idx: matched vertex loops of the split front, both ordered bottom -> top (same length). N0 points from
    the closest body point to C0 (so it is the outward body normal there).
    A zipper is stiff in its own plane and flexible out of it, so it cannot follow a cloth seam's vertex-scale sideways zig-zag:
    on the Milestone 1 draped jacket seam (0.4 mm sideways per 10 mm row) that zig-zag strains the tape's outer edge (13.8 mm
    out) by ~12-14 %. N0 is smoothed along arc length (sigma_n0_m = 60 mm); C0 keeps its out-of-plane (along-N0) profile and
    loses only the in-plane wiggle (wide smoothing 3 x sigma_c0_m, then 0.5 x sigma_c0_m), ends fixed. Result there: C0 within
    1.6 mm of the seam polyline, tape stretch <= 1.008 at 80 and 160 tape rows.
    """
    from scipy.spatial import cKDTree  # type: ignore

    VL = np.asarray(V_jacket, dtype=np.float64)[np.asarray(seam_L_idx, dtype=np.int64)]
    VR = np.asarray(V_jacket, dtype=np.float64)[np.asarray(seam_R_idx, dtype=np.int64)]
    if VL.shape != VR.shape:
        raise ValueError(f"seam loops must be matched pairs, got {VL.shape} vs {VR.shape}")
    C0 = 0.5 * (VL + VR)
    Vb = np.asarray(V_body, dtype=np.float64)
    Fb = np.asarray(F_body, dtype=np.int64)
    A, Bv, Cv = Vb[Fb[:, 0]], Vb[Fb[:, 1]], Vb[Fb[:, 2]]
    cen = (A + Bv + Cv) / 3.0
    _, idx = cKDTree(cen).query(C0, k=min(16, len(cen)))
    best = np.full(len(C0), np.inf)
    cp = np.zeros_like(C0)
    import canon_geometry_kernels as _K  # type: ignore  (same folder)

    for c in range(idx.shape[1]):
        f = idx[:, c]
        q = _K._closest_point_on_triangles(C0, A[f], Bv[f], Cv[f])
        d = np.linalg.norm(C0 - q, axis=1)
        better = d < best
        best = np.where(better, d, best)
        cp = np.where(better[:, None], q, cp)
    N0 = _norm(C0 - cp)
    if smooth_window > 0 and len(N0) > 2 * smooth_window + 1:
        k = 2 * smooth_window + 1
        pad = np.pad(N0, ((smooth_window, smooth_window), (0, 0)), mode="edge")
        N0 = _norm(np.stack([pad[i:i + len(N0)] for i in range(k)]).mean(axis=0))

    s_arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(C0, axis=0), axis=1))])

    def _gauss(X: np.ndarray, sigma: float, keep_ends: bool) -> np.ndarray:
        if sigma <= 0.0 or len(X) < 3:
            return X
        W = np.exp(-0.5 * ((s_arc[:, None] - s_arc[None, :]) / sigma) ** 2)
        W /= W.sum(axis=1, keepdims=True)
        Y = W @ X
        if keep_ends:   # remove the end shrinkage of a truncated kernel: blend the linear end-to-end correction back in
            t = s_arc / max(float(s_arc[-1]), 1e-12)
            corr0, corr1 = X[0] - Y[0], X[-1] - Y[-1]
            Y = Y + (1.0 - t)[:, None] * corr0 + t[:, None] * corr1
        return Y

    # A zipper chain is stiff IN its plane (tape + interlocked teeth) and flexible OUT of it. So: keep the seam's normal
    # (body-following) profile, but remove its in-plane sideways wiggle. C_wide = wide smoothing; the deviation of the raw seam
    # from C_wide is kept only along N0 (out of plane), and a light final smoothing removes vertex-scale noise. Smoothing the
    # whole curve widely instead would pull C0 ~10 mm off the seam toward the chord (measured), i.e. into the layers below.
    N0 = _norm(_gauss(N0, float(sigma_n0_m), keep_ends=False))
    C_wide = _gauss(C0, max(float(sigma_c0_m), 1e-9) * 3.0, keep_ends=True)
    D = C0 - C_wide
    C0 = C_wide + np.einsum("ij,ij->i", D, N0)[:, None] * N0
    C0 = _gauss(C0, 0.5 * float(sigma_c0_m), keep_ends=True)
    return C0, N0


def _default_pbd_fn() -> Any:
    """newton_cloth_template.pbd_strain_limit_and_tangent_clearance (single owner, max_garment_eng/03_gpu_cloth_solver)."""
    import sys
    from pathlib import Path

    try:
        from newton_cloth_template import pbd_strain_limit_and_tangent_clearance  # type: ignore
        return pbd_strain_limit_and_tangent_clearance
    except ImportError:
        here = Path(__file__).resolve()
        for up in here.parents:
            cand = up / "max_garment_eng" / "03_gpu_cloth_solver"
            if (cand / "newton_cloth_template.py").is_file():
                sys.path.insert(0, str(cand))
                from newton_cloth_template import pbd_strain_limit_and_tangent_clearance  # type: ignore
                return pbd_strain_limit_and_tangent_clearance
        raise


def peel_jacket_lapels_over_layer1(
    V_jacket: np.ndarray,
    F_jacket: np.ndarray,
    seam_L_idx: np.ndarray,
    seam_R_idx: np.ndarray,
    zipper_data: Dict[str, Any],
    layer_collider: Any,
    sigma_peel_m: float = 0.07,
    edges: Optional[np.ndarray] = None,
    V_rest: Optional[np.ndarray] = None,
    geodesic_fn: Any = None,
    pbd_fn: Any = None,
    n_sweeps: int = 10,
) -> Dict[str, Any]:
    """Peel the split jacket lapels open onto the zipper tapes, over layer 1.

    1. Each seam vertex gets its tape material coordinate sigma (arc length of C0 at its pair) and moves onto the outer sewn edge of
       the tape on its side, as laid out by `zipper_data` (build_zipper_numpy(..., isometric=True) on extract_seam_centerline()).
    2. d_L, d_R = geodesic distance from the left / right seam loop (heat method; the slit stops heat crossing from one lapel to the
       other). Interior vertex v follows the displacement of the nearest seam vertex on its own side, weighted exp(-d^2 / 2 sigma^2).
    3. `n_sweeps` outer sweeps of tangential PBD (strain limit + LayerCollider push-out) with both seam loops pinned to the tape.

    geodesic_fn(V, F, sources) defaults to canon_geometry_kernels.heat_geodesic_distance (each seam loop is ONE connected seed
    cluster, the case where MC-50 is valid); pass alignment_adapters.heat_geodesic to use libigl when installed. pbd_fn defaults to
    newton_cloth_template.pbd_strain_limit_and_tangent_clearance. V_rest: rest positions for edge rest lengths (default V_jacket).
    """
    from scipy.spatial import cKDTree  # type: ignore

    V = np.asarray(V_jacket, dtype=np.float64)
    F = np.asarray(F_jacket, dtype=np.int64)
    sL = np.asarray(seam_L_idx, dtype=np.int64)
    sR = np.asarray(seam_R_idx, dtype=np.int64)
    Vr = V if V_rest is None else np.asarray(V_rest, dtype=np.float64)
    if edges is None:
        e = np.sort(np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1)
        edges = np.unique(e, axis=0)
    E = np.asarray(edges, dtype=np.int64)
    if geodesic_fn is None:
        import canon_geometry_kernels as _K  # type: ignore
        geodesic_fn = _K.heat_geodesic_distance
    if pbd_fn is None:
        pbd_fn = _default_pbd_fn()

    te = zipper_data["tape_edges"]
    s_mat = np.asarray(te["s_material"], dtype=np.float64)
    C0 = 0.5 * (V[sL] + V[sR])
    sig = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(C0, axis=0), axis=1))])
    sig *= float(s_mat[-1]) / max(float(sig[-1]), 1e-12)          # zipper resampled the same curve: same total length
    # which tape side (+B or -B) each loop is sewn to, measured in the closed frame
    Bc = np.column_stack([np.interp(sig, s_mat, te["B_closed"][:, c]) for c in range(3)])
    side_L = 1 if float(np.einsum("ij,ij->i", V[sL] - C0, Bc).mean()) >= 0.0 else -1
    edge_of = {+1: np.asarray(te["side_plus"]), -1: np.asarray(te["side_minus"])}

    def on_tape(side: int) -> np.ndarray:
        Pe = edge_of[side]
        return np.column_stack([np.interp(sig, s_mat, Pe[:, c]) for c in range(3)])

    tgt_L, tgt_R = on_tape(side_L), on_tape(-side_L)
    d_seam_L, d_seam_R = tgt_L - V[sL], tgt_R - V[sR]

    d_L = np.asarray(geodesic_fn(V, F, sL), dtype=np.float64)
    d_R = np.asarray(geodesic_fn(V, F, sR), dtype=np.float64)
    left = d_L <= d_R
    X = V.copy()
    for mask, seam, dseam, dist in ((left, sL, d_seam_L, d_L), (~left, sR, d_seam_R, d_R)):
        idx = np.flatnonzero(mask)
        if len(idx) == 0:
            continue
        _, nn = cKDTree(V[seam]).query(V[idx])
        w = np.exp(-(dist[idx] ** 2) / (2.0 * float(sigma_peel_m) ** 2))
        X[idx] = V[idx] + w[:, None] * dseam[nn]
    X[sL] = tgt_L
    X[sR] = tgt_R

    pinned = np.zeros(len(V), dtype=bool)
    pinned[sL] = True
    pinned[sR] = True
    l_bar = np.linalg.norm(Vr[E[:, 0]] - Vr[E[:, 1]], axis=1)
    body = layer_collider.body
    X, pbd_info = pbd_fn(X, l_bar, E, body.V, body.F, collider=layer_collider, pinned_mask=pinned,
                         n_sweeps=8 * int(n_sweeps), return_info=True)
    return {
        "V": X,
        "pinned": pinned,
        "seam_target_err_mm": round(float(max(np.abs(X[sL] - tgt_L).max(), np.abs(X[sR] - tgt_R).max()) * 1000.0), 6),
        "seam_max_move_mm": round(float(max(np.linalg.norm(d_seam_L, axis=1).max(), np.linalg.norm(d_seam_R, axis=1).max()) * 1000.0), 3),
        "geodesic_max_m": {"left": round(float(d_L.max()), 4), "right": round(float(d_R.max()), 4)},
        "left_lapel_verts": int(left.sum()),
        "pbd": pbd_info,
    }


def _rot_to_quat_xyzw(R: np.ndarray) -> np.ndarray:
    """Convert (N, 3, 3) rotation matrices in SO(3) to glTF unit quaternions (x, y, z, w)."""
    out = np.zeros((len(R), 4), dtype=np.float32)
    for i, m in enumerate(R):
        t = float(np.trace(m))
        if t > 0.0:
            s = math.sqrt(t + 1.0) * 2.0
            w, x, y, z = 0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s
        elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
            w, x, y, z = (m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s
        elif m[1, 1] > m[2, 2]:
            s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
            w, x, y, z = (m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s
        else:
            s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
            w, x, y, z = (m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s
        q = np.array([x, y, z, w], dtype=np.float64)
        q /= max(float(np.linalg.norm(q)), 1e-12)
        out[i] = q.astype(np.float32)
    return out


def export_instanced_zipper_glb(
    out_path: Any,
    curve_pts: Optional[np.ndarray] = None,
    config: Optional[ZipperConfig] = None,
    curve_normals: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Export `zipper_hero_instanced.glb` using `EXT_mesh_gpu_instancing` for all rigid teeth instances + `Zip_Scrub_0_to_1` animation (< 400 KB)."""
    import json as _json
    import struct as _struct
    from pathlib import Path as _Path

    cfg = config if config is not None else ZipperConfig(open_factor=0.0)
    if curve_pts is None:
        curve_pts, curve_normals = default_jacket_centerline(0.420)
    z = build_zipper_numpy(curve_pts, cfg, curve_normals=curve_normals)

    # Single tooth prototype mesh + N rigid instance matrices (L + R)
    Vt = z["tooth_prototype"]["V"].astype(np.float32)
    Ft = z["tooth_prototype"]["F"].astype(np.uint16)
    M_inst = np.concatenate([z["tooth_transforms_L"], z["tooth_transforms_R"]], axis=0)
    inst_T = M_inst[:, :3, 3].astype(np.float32)
    inst_R = _rot_to_quat_xyzw(M_inst[:, :3, :3])
    inst_S = np.ones((len(M_inst), 3), dtype=np.float32)

    # Slider prototype positioned at origin, animated along the centerline via `Zip_Scrub_0_to_1`
    Vh = z["slider_prototype"]["V"].astype(np.float32)
    Fh = z["slider_prototype"]["F"].astype(np.uint16)

    # Static tape + stops combined mesh
    V_ts, F_ts = _concat_meshes([(z["parts"]["tape"]["V"], z["parts"]["tape"]["F"]), (z["parts"]["stops"]["V"], z["parts"]["stops"]["F"])])
    V_ts = V_ts.astype(np.float32); F_ts = F_ts.astype(np.uint16)

    # Animation track `Zip_Scrub_0_to_1`: 25 keyframes (t = 0..1 s) translating the slider node from top (O=0) to bottom (O=1)
    s_grid, Q, T, B, N, _ = _resample_curve(curve_pts, curve_normals, n_samples=64, flip=cfg.flip)
    s_bot = z["slider_state"]["s_bot_m"]; s_top = z["slider_state"]["s_top_m"]
    n_kf = 25
    anim_t = np.linspace(0.0, 1.0, n_kf, dtype=np.float32)
    anim_pos = np.zeros((n_kf, 3), dtype=np.float32)
    anim_rot = np.zeros((n_kf, 4), dtype=np.float32)
    for k, o_val in enumerate(anim_t):
        sv = s_bot + (1.0 - float(o_val)) * (s_top - s_bot)
        qk, tk, bk, nk = _interp_frame(sv, s_grid, Q, T, B, N)
        anim_pos[k] = qk.astype(np.float32)
        anim_rot[k] = _rot_to_quat_xyzw(np.column_stack([bk, tk, nk])[None])[0]

    chunks: List[bytes] = []
    bvs: List[Dict[str, Any]] = []
    accs: List[Dict[str, Any]] = []

    def _add_acc(arr: np.ndarray, ctype: int, atype: str, target: Optional[int] = None) -> int:
        raw = arr.tobytes()
        pad = (4 - (len(raw) % 4)) % 4
        offset = sum(len(c) for c in chunks)
        chunks.append(raw + b"\x00" * pad)
        bv: Dict[str, Any] = {"buffer": 0, "byteOffset": offset, "byteLength": len(raw)}
        if target is not None:
            bv["target"] = target
        bvs.append(bv)
        acc: Dict[str, Any] = {
            "bufferView": len(bvs) - 1,
            "componentType": ctype,
            "count": int(len(arr)),
            "type": atype,
        }
        if atype in ("SCALAR", "VEC3") and ctype == 5126:
            acc["min"] = np.min(arr, axis=0).astype(float).tolist() if arr.ndim > 1 else [float(np.min(arr))]
            acc["max"] = np.max(arr, axis=0).astype(float).tolist() if arr.ndim > 1 else [float(np.max(arr))]
        accs.append(acc)
        return len(accs) - 1

    # Tooth mesh accessors
    a_vt = _add_acc(Vt, 5126, "VEC3", 34962)
    a_ft = _add_acc(Ft.ravel(), 5123, "SCALAR", 34963)
    # GPU instancing accessors
    a_it = _add_acc(inst_T, 5126, "VEC3")
    a_ir = _add_acc(inst_R, 5126, "VEC4")
    a_is = _add_acc(inst_S, 5126, "VEC3")
    # Slider mesh accessors
    a_vh = _add_acc(Vh, 5126, "VEC3", 34962)
    a_fh = _add_acc(Fh.ravel(), 5123, "SCALAR", 34963)
    # Tape + stops mesh accessors
    a_vts = _add_acc(V_ts, 5126, "VEC3", 34962)
    a_fts = _add_acc(F_ts.ravel(), 5123, "SCALAR", 34963)
    # Animation accessors
    a_at = _add_acc(anim_t, 5126, "SCALAR")
    a_ap = _add_acc(anim_pos, 5126, "VEC3")
    a_ar = _add_acc(anim_rot, 5126, "VEC4")

    bin_blob = b"".join(chunks)
    gltf = {
        "asset": {"version": "2.0", "generator": "Maximus v2 Clean-Room Zipper Engine (EXT_mesh_gpu_instancing)"},
        "extensionsUsed": ["EXT_mesh_gpu_instancing"],
        "scene": 0,
        "scenes": [{"name": "MaximusZipperHero", "nodes": [0, 1, 2]}],
        "nodes": [
            {
                "name": "Zipper_Teeth_Instanced",
                "mesh": 0,
                "extensions": {
                    "EXT_mesh_gpu_instancing": {
                        "attributes": {"TRANSLATION": a_it, "ROTATION": a_ir, "SCALE": a_is}
                    }
                },
            },
            {
                "name": "Zipper_Slider_Head",
                "mesh": 1,
                "translation": anim_pos[0].astype(float).tolist(),
                "rotation": anim_rot[0].astype(float).tolist(),
            },
            {"name": "Zipper_Tape_And_Stops", "mesh": 2},
        ],
        "materials": [
            {"name": f"MX_Zip_{cfg.tooth_material}", "pbrMetallicRoughness": {"baseColorFactor": [0.22, 0.24, 0.27, 1.0], "metallicFactor": 1.0, "roughnessFactor": 0.20}},
            {"name": "MX_Zip_Tape_Twill", "pbrMetallicRoughness": {"baseColorFactor": [0.03, 0.03, 0.035, 1.0], "metallicFactor": 0.0, "roughnessFactor": 0.84}},
        ],
        "meshes": [
            {"name": "Tooth_Prototype", "primitives": [{"attributes": {"POSITION": a_vt}, "indices": a_ft, "material": 0}]},
            {"name": "Slider_Assembly", "primitives": [{"attributes": {"POSITION": a_vh}, "indices": a_fh, "material": 0}]},
            {"name": "Tape_And_Stops",  "primitives": [{"attributes": {"POSITION": a_vts}, "indices": a_fts, "material": 1}]},
        ],
        "animations": [
            {
                "name": "Zip_Scrub_0_to_1",
                "samplers": [
                    {"input": a_at, "interpolation": "LINEAR", "output": a_ap},
                    {"input": a_at, "interpolation": "LINEAR", "output": a_ar},
                ],
                "channels": [
                    {"sampler": 0, "target": {"node": 1, "path": "translation"}},
                    {"sampler": 1, "target": {"node": 1, "path": "rotation"}},
                ],
            }
        ],
        "accessors": accs,
        "bufferViews": bvs,
        "buffers": [{"byteLength": len(bin_blob)}],
        "extras": {"zipper_config": cfg.to_dict(), "clearance_gate": z["clearance_gate"]},
    }

    jbytes = _json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    jpad = (4 - (len(jbytes) % 4)) % 4
    jchunk = jbytes + b" " * jpad
    total_size = 12 + 8 + len(jchunk) + 8 + len(bin_blob)
    header = _struct.pack("<4sII", b"glTF", 2, total_size)
    c0 = _struct.pack("<I4s", len(jchunk), b"JSON") + jchunk
    c1 = _struct.pack("<I4s", len(bin_blob), b"BIN\x00") + bin_blob

    p = _Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(header + c0 + c1)
    return {
        "path": str(p),
        "bytes": total_size,
        "KB": round(total_size / 1024.0, 2),
        "under_400KB": total_size < 400 * 1024,
        "instanced_teeth_count": int(len(M_inst)),
        "animation": "Zip_Scrub_0_to_1",
        "extensionsUsed": gltf["extensionsUsed"],
    }
def _hex_to_linear(hex_str: str) -> Tuple[float, float, float]:
    h = hex_str.lstrip("#")
    vals = [int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)]
    return tuple(c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in vals)


def build_pbr_material_blender(bpy: Any, preset_name: str, textile_uv_attribute: str = "textile UV", is_tape: bool = False) -> Any:
    """Create a Blender 5.2 Principled BSDF PBR material (metal preset or herringbone twill tape using `textile UV`)."""
    mat_name = f"MX_Zip_{preset_name}" if not is_tape else f"MX_Zip_Tape_{preset_name}"
    existing = bpy.data.materials.get(mat_name)
    if existing is not None:
        return existing
    spec = MATERIAL_PRESETS.get(preset_name, MATERIAL_PRESETS["brushed_gunmetal"])
    mat = bpy.data.materials.new(mat_name)
    if not getattr(mat, "use_nodes", True):
        mat.use_nodes = True
    nt = mat.node_tree; N = nt.nodes; L = nt.links
    bsdf = next(n for n in N if n.type == "BSDF_PRINCIPLED")
    rgb = _hex_to_linear(spec["hex"])
    bsdf.inputs["Base Color"].default_value = (*rgb, 1.0)
    bsdf.inputs["Metallic"].default_value = float(spec["metallic"])
    bsdf.inputs["Roughness"].default_value = float(spec["roughness"])
    if "Anisotropic" in bsdf.inputs and spec.get("anisotropic", 0.0) > 0.0:
        bsdf.inputs["Anisotropic"].default_value = float(spec["anisotropic"])
    if "Coat Weight" in bsdf.inputs and spec.get("coat", 0.0) > 0.0:
        bsdf.inputs["Coat Weight"].default_value = float(spec["coat"])
    if "Sheen Weight" in bsdf.inputs and spec.get("sheen", 0.0) > 0.0:
        bsdf.inputs["Sheen Weight"].default_value = float(spec["sheen"])

    if is_tape:
        # Procedural twill herringbone weave driven explicitly by the `textile UV` attribute
        attr = N.new("ShaderNodeAttribute"); attr.attribute_name = textile_uv_attribute
        sep = N.new("ShaderNodeSeparateXYZ"); L.new(attr.outputs["Vector"], sep.inputs["Vector"])
        # Diagonal twill ribs: sin(2*pi*(u*18 + v*24))
        m_u = N.new("ShaderNodeMath"); m_u.operation = "MULTIPLY"; m_u.inputs[1].default_value = 95.0; L.new(sep.outputs["X"], m_u.inputs[0])
        m_v = N.new("ShaderNodeMath"); m_v.operation = "MULTIPLY"; m_v.inputs[1].default_value = 130.0; L.new(sep.outputs["Y"], m_v.inputs[0])
        add = N.new("ShaderNodeMath"); add.operation = "ADD"; L.new(m_u.outputs[0], add.inputs[0]); L.new(m_v.outputs[0], add.inputs[1])
        sn = N.new("ShaderNodeMath"); sn.operation = "SINE"; L.new(add.outputs[0], sn.inputs[0])
        bump = N.new("ShaderNodeBump"); bump.inputs["Strength"].default_value = 0.16; bump.inputs["Distance"].default_value = 0.00025
        L.new(sn.outputs[0], bump.inputs["Height"]); L.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    elif spec.get("bump_strength", 0.0) > 0.0:
        tc = N.new("ShaderNodeTexCoord")
        noise = N.new("ShaderNodeTexNoise")
        noise.inputs["Scale"].default_value = float(spec.get("bump_scale", 600.0))
        noise.inputs["Detail"].default_value = 6.0
        L.new(tc.outputs["Object"], noise.inputs["Vector"])
        bump = N.new("ShaderNodeBump")
        bump.inputs["Strength"].default_value = float(spec["bump_strength"])
        bump.inputs["Distance"].default_value = 0.00015
        L.new(noise.outputs["Fac"], bump.inputs["Height"]); L.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    return mat


def build_zipper_gn_nodegroup(bpy: Any, config: ZipperConfig, name: str = "MX_ProceduralZipper_GN") -> Any:
    """Create a Blender 5.2 GeometryNodeTree exposing the exact 8 Video-2 parameter sockets (pass-through + metadata contract)."""
    ng = bpy.data.node_groups.get(name)
    if ng is None:
        ng = bpy.data.node_groups.new(name, "GeometryNodeTree")
        iface = ng.interface
        iface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
        iface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
        s_sz = iface.new_socket("Size Multiplier", in_out="INPUT", socket_type="NodeSocketFloat")
        s_sz.default_value = config.size_multiplier; s_sz.min_value = 0.3; s_sz.max_value = 3.0
        s_ti = iface.new_socket("Tooth Index", in_out="INPUT", socket_type="NodeSocketInt")
        s_ti.default_value = config.tooth_index; s_ti.min_value = 0; s_ti.max_value = 4
        s_hi = iface.new_socket("Head Index", in_out="INPUT", socket_type="NodeSocketInt")
        s_hi.default_value = config.head_index; s_hi.min_value = 0; s_hi.max_value = 3
        s_op = iface.new_socket("Open Factor", in_out="INPUT", socket_type="NodeSocketFloat")
        s_op.default_value = config.open_factor; s_op.min_value = 0.0; s_op.max_value = 1.0
        s_fl = iface.new_socket("Flip", in_out="INPUT", socket_type="NodeSocketBool")
        s_fl.default_value = config.flip
        iface.new_socket("Tooth Material", in_out="INPUT", socket_type="NodeSocketMaterial")
        iface.new_socket("Head Material", in_out="INPUT", socket_type="NodeSocketMaterial")
        s_uv = iface.new_socket("Textile UV Attribute", in_out="INPUT", socket_type="NodeSocketString")
        s_uv.default_value = config.textile_uv_attribute

        gin = ng.nodes.new("NodeGroupInput")
        gout = ng.nodes.new("NodeGroupOutput")
        ng.links.new(gin.outputs["Geometry"], gout.inputs["Geometry"])
    return ng


def build_zipper_blender(
    bpy: Any,
    curve_pts: np.ndarray,
    config: Optional[ZipperConfig] = None,
    curve_normals: Optional[np.ndarray] = None,
    name: str = "MX_Zipper",
    collection: Optional[Any] = None,
    create_gn_interface: bool = True,
) -> Tuple[Any, Dict[str, Any]]:
    """Build a real Blender 5.2 zipper mesh object with `textile UV`, PBR materials, and 8-socket GN contract (ZERO bpy.ops)."""
    cfg = config if config is not None else ZipperConfig()
    data = build_zipper_numpy(curve_pts, cfg, curve_normals=curve_normals)

    me = bpy.data.meshes.new(name)
    V = data["V"]; F = data["F"]; UV = data["uv"]; pids = data["part_ids"]
    me.from_pydata(V.tolist(), [], F.tolist())

    # Assign material slots: 0=tape, 1=teeth, 2=head/stops
    mat_tape = build_pbr_material_blender(bpy, "twill_tape_noir", textile_uv_attribute=cfg.textile_uv_attribute, is_tape=True)
    mat_tooth = build_pbr_material_blender(bpy, cfg.tooth_material, is_tape=False)
    mat_head = build_pbr_material_blender(bpy, cfg.head_material, is_tape=False)
    for m in (mat_tape, mat_tooth, mat_head):
        me.materials.append(m)
    slot_map = np.array([0, 1, 2, 2], dtype=np.int32)
    me.polygons.foreach_set("material_index", slot_map[pids].tolist())
    me.polygons.foreach_set("use_smooth", [False] * len(me.polygons))

    # Write continuous `textile UV` attribute onto loop UV layer
    uv_layer = me.uv_layers.new(name=cfg.textile_uv_attribute)
    loop_verts = np.empty(len(me.loops), dtype=np.int64)
    me.loops.foreach_get("vertex_index", loop_verts)
    uv_layer.data.foreach_set("uv", UV[loop_verts].astype(np.float32).ravel())
    me.update()

    ob = bpy.data.objects.new(name, me)
    col = collection if collection is not None else bpy.context.scene.collection
    col.objects.link(ob)

    # Store config & expose the 8 Video-2 parameter sockets via Geometry Nodes modifier
    ob["zipper_config"] = str(cfg.to_dict())
    if create_gn_interface:
        mod = ob.modifiers.new(name="ProceduralZipper", type="NODES")
        ng = build_zipper_gn_nodegroup(bpy, cfg)
        mod.node_group = ng
        # Populate modifier socket values by socket identifier
        sock_vals = {
            "Size Multiplier": cfg.size_multiplier,
            "Tooth Index": cfg.tooth_index,
            "Head Index": cfg.head_index,
            "Open Factor": cfg.open_factor,
            "Flip": cfg.flip,
            "Tooth Material": mat_tooth,
            "Head Material": mat_head,
            "Textile UV Attribute": cfg.textile_uv_attribute,
        }
        for item in ng.interface.items_tree:
            if getattr(item, "in_out", None) == "INPUT" and item.name in sock_vals:
                try:
                    mod[item.identifier] = sock_vals[item.name]
                except Exception:  # noqa: BLE001
                    pass

    return ob, data
