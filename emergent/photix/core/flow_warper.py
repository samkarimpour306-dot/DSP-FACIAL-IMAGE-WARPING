"""
flow_warper.py - FR-12, FR-13
Advanced landmark-guided dense optical-flow style warping.

The UI passes source/destination landmark positions rather than two video
frames, so this module builds a dense backward displacement field from sparse
facial controls.  The field is evaluated with an adaptive multi-scale RBF
kernel, softly constrained by stable anatomical anchors and faded outside the
face support region before cv2.remap samples the source image.
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

_MIN_DISP = 0.8        # px - below this a landmark is considered inactive
_LOCAL_SIGMA = 34.0    # px - detail around moved features
_GLOBAL_SIGMA = 92.0   # px - broad facial propagation
_ANCHOR_SIGMA = 48.0   # px - stabilises unmoved facial regions
_FACE_FEATHER = 34.0   # px - soft fade outside the facial hull
_MAX_OVERSHOOT = 1.35  # cap dense vectors relative to moved controls

_ANATOM_ANCHORS = [
    168, 6, 197, 4,        # nose bridge + tip
    159, 386,              # eye upper lids
    130, 359,              # eye outer corners
    133, 362,              # eye inner corners
    10, 67, 297, 54, 284,  # forehead
    152,                   # chin centre
    116, 345,              # cheekbones
]


def _perimeter(height: int, width: int, n: int = 18) -> np.ndarray:
    """Evenly spaced zero-displacement anchors around the image perimeter."""
    ts = np.linspace(0, 1, n, endpoint=True)
    pts = np.vstack([
        np.c_[ts * (width - 1), np.zeros(n)],
        np.c_[ts * (width - 1), np.full(n, height - 1.0)],
        np.c_[np.zeros(n), ts * (height - 1)],
        np.c_[np.full(n, width - 1.0), ts * (height - 1)],
    ])
    return np.unique(pts, axis=0).astype(np.float64)


def _grid_step(height: int, width: int) -> int:
    """Keep high quality on 512 px images without making large inputs slow."""
    longest = max(height, width)
    if longest <= 640:
        return 4
    if longest <= 1200:
        return 6
    return 8


def _face_support_mask(
    height: int,
    width: int,
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
) -> np.ndarray:
    """Soft mask that keeps expression flow on the face, not the background."""
    if len(src_pts) < 8:
        return np.ones((height, width), dtype=np.float32)

    pts = np.vstack([src_pts[:, :2], dst_pts[:, :2]]).astype(np.float32)
    pts[:, 0] = np.clip(pts[:, 0], 0, width - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, height - 1)

    hull = cv2.convexHull(pts.astype(np.int32))
    mask = np.zeros((height, width), dtype=np.float32)
    cv2.fillConvexPoly(mask, hull, 1.0)

    pad = max(6, int(round(_FACE_FEATHER * 0.7)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (pad * 2 + 1, pad * 2 + 1))
    mask = cv2.dilate(mask, kernel, iterations=1)
    mask = gaussian_filter(mask, sigma=_FACE_FEATHER).astype(np.float32)
    peak = float(mask.max())
    if peak > 1e-6:
        mask /= peak
    return np.clip(mask, 0.0, 1.0)


def _adaptive_sigmas(ctrl_pts: np.ndarray, active_count: int) -> np.ndarray:
    """Estimate a stable influence radius for every control point."""
    if len(ctrl_pts) <= 1:
        return np.full(len(ctrl_pts), _GLOBAL_SIGMA, dtype=np.float64)

    diff = ctrl_pts[:, None, :] - ctrl_pts[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    dist[dist < 1e-6] = np.inf
    nearest = np.min(dist, axis=1)

    sigmas = np.full(len(ctrl_pts), _ANCHOR_SIGMA, dtype=np.float64)
    active_radius = np.clip(nearest[:active_count] * 1.75, _LOCAL_SIGMA, _GLOBAL_SIGMA)
    sigmas[:active_count] = active_radius
    return sigmas


def _kernel_regression(
    query_x: np.ndarray,
    query_y: np.ndarray,
    ctrl_pts: np.ndarray,
    ctrl_vals: np.ndarray,
    sigmas: np.ndarray,
    confidence: np.ndarray,
) -> np.ndarray:
    """Evaluate anisotropic-quality Gaussian kernel regression at query points."""
    dx = query_x[:, None] - ctrl_pts[None, :, 0]
    dy = query_y[:, None] - ctrl_pts[None, :, 1]
    r2 = dx * dx + dy * dy
    s2 = 2.0 * (sigmas[None, :] ** 2)

    weights = np.exp(-r2 / s2) * confidence[None, :]
    weights_sum = weights.sum(axis=1, keepdims=True)
    return (weights @ ctrl_vals) / (weights_sum + 1e-12)


def _limit_flow_magnitude(dx: np.ndarray, dy: np.ndarray, max_mag: float) -> tuple[np.ndarray, np.ndarray]:
    """Prevent interpolation ringing from exceeding plausible control motion."""
    mag = np.sqrt(dx * dx + dy * dy)
    scale = np.minimum(1.0, max_mag / (mag + 1e-6)).astype(np.float32)
    return dx * scale, dy * scale


def apply_optical_flow_warp(
    src_img: np.ndarray,
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
    intensity: float = 1.0,
) -> np.ndarray:
    """Dense backward warp from sparse landmark motion.

    Backward-mapping semantics for cv2.remap:
        map(dst_pixel) = dst_pixel + backward_disp(dst_pixel)
        backward_disp = src - dst at active control points.
    """
    height, width = src_img.shape[:2]
    strength = float(np.clip(intensity, 0.0, 2.0))

    src = src_pts[:, :2].astype(np.float64)
    raw_dst = dst_pts[:, :2].astype(np.float64)
    disp = raw_dst - src
    active = np.linalg.norm(disp, axis=1) > _MIN_DISP

    if strength <= 1e-4 or not np.any(active):
        return src_img.copy()

    dst = src + disp * strength
    active_src = src[active]
    active_dst = dst[active]
    back_active = active_src - active_dst
    active_mags = np.linalg.norm(back_active, axis=1)

    perimeter = _perimeter(height, width)

    if len(src) >= 468:
        anchor_idx = [i for i in _ANATOM_ANCHORS if i < len(src) and not active[i]]
        anchor_pts = src[np.array(anchor_idx)].astype(np.float64) if anchor_idx else np.empty((0, 2))
    else:
        anchor_pts = np.empty((0, 2), dtype=np.float64)

    ctrl_pts = np.vstack([active_dst, anchor_pts, perimeter])
    ctrl_vals = np.vstack([
        back_active,
        np.zeros((len(anchor_pts), 2), dtype=np.float64),
        np.zeros((len(perimeter), 2), dtype=np.float64),
    ])

    active_count = len(active_dst)
    sigmas = _adaptive_sigmas(ctrl_pts, active_count)

    # Active controls must dominate their local feature.  Anchors are softer,
    # but perimeter controls are firm enough to stop background drift.
    confidence = np.concatenate([
        np.clip(active_mags / (np.median(active_mags) + 1e-6), 0.8, 2.2),
        np.full(len(anchor_pts), 0.65, dtype=np.float64),
        np.full(len(perimeter), 1.15, dtype=np.float64),
    ])

    step = _grid_step(height, width)
    small_h = max(2, int(np.ceil(height / step)))
    small_w = max(2, int(np.ceil(width / step)))
    grid_y, grid_x = np.mgrid[0:small_h, 0:small_w]
    query_x = np.minimum(grid_x.ravel() * step, width - 1).astype(np.float64)
    query_y = np.minimum(grid_y.ravel() * step, height - 1).astype(np.float64)

    local_field = _kernel_regression(
        query_x, query_y, ctrl_pts, ctrl_vals, sigmas, confidence
    )
    global_field = _kernel_regression(
        query_x,
        query_y,
        ctrl_pts,
        ctrl_vals,
        np.full(len(ctrl_pts), _GLOBAL_SIGMA, dtype=np.float64),
        confidence,
    )
    field = 0.68 * local_field + 0.32 * global_field

    bdx = cv2.resize(
        field[:, 0].reshape(small_h, small_w).astype(np.float32),
        (width, height),
        interpolation=cv2.INTER_CUBIC,
    )
    bdy = cv2.resize(
        field[:, 1].reshape(small_h, small_w).astype(np.float32),
        (width, height),
        interpolation=cv2.INTER_CUBIC,
    )

    # Bilateral flow smoothing keeps feature-local changes crisp while removing
    # coarse-grid ripples.  The final Gaussian pass removes sub-pixel ringing.
    bdx = cv2.bilateralFilter(bdx, d=0, sigmaColor=10.0, sigmaSpace=9.0)
    bdy = cv2.bilateralFilter(bdy, d=0, sigmaColor=10.0, sigmaSpace=9.0)
    bdx = gaussian_filter(bdx, sigma=2.2).astype(np.float32)
    bdy = gaussian_filter(bdy, sigma=2.2).astype(np.float32)

    support = _face_support_mask(height, width, src, dst)
    bdx *= support
    bdy *= support

    max_motion = max(1.0, float(active_mags.max()) * _MAX_OVERSHOOT)
    bdx, bdy = _limit_flow_magnitude(bdx, bdy, max_motion)

    gy, gx = np.mgrid[0:height, 0:width]
    map_x = np.clip(gx.astype(np.float32) + bdx, 0, width - 1)
    map_y = np.clip(gy.astype(np.float32) + bdy, 0, height - 1)

    result = cv2.remap(
        src_img,
        map_x,
        map_y,
        interpolation=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return np.clip(result, 0, 255).astype(np.uint8)
