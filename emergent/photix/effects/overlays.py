"""
effects/overlays.py — Photix face overlay engine
Landmark-accurate sunglasses and beard compositing using MediaPipe Face Mesh.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    from landmarks.detector import FaceLandmarks as _FaceLandmarks
except ImportError:
    _FaceLandmarks = None

_ASSETS_DIR = Path(__file__).parent / "assets"
_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class OverlayResult:
    """Output produced by FaceOverlayEngine.apply()."""
    final_image: np.ndarray
    intermediate_stages: list[np.ndarray] = field(default_factory=list)
    masks_used: dict[str, np.ndarray] = field(default_factory=dict)
    dsp_metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Core helper: alpha_blend
# ---------------------------------------------------------------------------

def alpha_blend(
    bg: np.ndarray,
    fg_rgba: np.ndarray,
    center_xy: tuple[int, int],
    alpha_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Paste fg_rgba (H, W, 4) onto bg (H, W, 3) centered at center_xy.

    Clips safely to image bounds — never crashes on partial off-frame pastes.
    """
    out = bg.copy()
    fh, fw = fg_rgba.shape[:2]
    bh, bw = bg.shape[:2]
    cx, cy = int(center_xy[0]), int(center_xy[1])

    dx1, dy1 = cx - fw // 2, cy - fh // 2
    dx2, dy2 = dx1 + fw, dy1 + fh

    sx1 = max(0, dx1);  sx2 = min(bw, dx2)
    sy1 = max(0, dy1);  sy2 = min(bh, dy2)
    if sx2 <= sx1 or sy2 <= sy1:
        return out

    fx1 = sx1 - dx1;  fx2 = fx1 + (sx2 - sx1)
    fy1 = sy1 - dy1;  fy2 = fy1 + (sy2 - sy1)

    patch = fg_rgba[fy1:fy2, fx1:fx2]
    alpha = patch[:, :, 3:4].astype(np.float32) / 255.0
    if alpha_mask is not None:
        mask_patch = alpha_mask[sy1:sy2, sx1:sx2].astype(np.float32)
        if mask_patch.ndim == 3:
            mask_patch = mask_patch[:, :, 0]
        if mask_patch.size and float(mask_patch.max()) > 1.0:
            mask_patch = mask_patch / 255.0
        alpha *= np.clip(mask_patch, 0.0, 1.0)[:, :, np.newaxis]
    fg_c  = patch[:, :, :3].astype(np.float32)
    bg_r  = out[sy1:sy2, sx1:sx2].astype(np.float32)
    out[sy1:sy2, sx1:sx2] = np.clip(
        fg_c * alpha + bg_r * (1.0 - alpha), 0, 255
    ).astype(np.uint8)
    return out


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _rotate_asset(img_rgba: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate BGRA image, expanding canvas so corners are not clipped."""
    h, w = img_rgba.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle_deg, 1.0)
    cos_a, sin_a = abs(M[0, 0]), abs(M[0, 1])
    new_w = int(h * sin_a + w * cos_a)
    new_h = int(h * cos_a + w * sin_a)
    M[0, 2] += (new_w - w) / 2.0
    M[1, 2] += (new_h - h) / 2.0
    return cv2.warpAffine(
        img_rgba, M, (new_w, new_h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )


def _pt(lm: np.ndarray, idx: int) -> tuple[float, float]:
    if idx < len(lm):
        return float(lm[idx, 0]), float(lm[idx, 1])
    return float(lm[:, 0].mean()), float(lm[:, 1].mean())


def _dist(lm: np.ndarray, i: int, j: int) -> float:
    return math.dist(_pt(lm, i), _pt(lm, j))


_LOWER_LIP_EDGE = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291]
_OUTER_LIP_RING = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
    291, 409, 270, 269, 267, 0, 37, 39, 40, 185,
]
_LEFT_JAW_TO_CHIN = [172, 136, 150, 149, 176, 148, 152]
_RIGHT_CHIN_TO_JAW = [377, 400, 378, 379, 365, 397]
_LEFT_FULL_BEARD_REGION = [
    93, 132, 58, 172, 136, 150, 149, 176, 148, 152,
    17, 84, 181, 91, 146, 61,
]
_RIGHT_FULL_BEARD_REGION = [
    291, 375, 321, 405, 314, 17, 152, 377, 400, 378,
    379, 365, 397, 288, 361, 323,
]
_BEARD_COLOR_TARGETS_BGR: dict[str, tuple[int, int, int]] = {
    "black": (36, 32, 28),
    "brown": (38, 58, 92),
    "blonde": (78, 142, 196),
}


def _points(lm: np.ndarray, indices: list[int]) -> list[tuple[float, float]]:
    return [_pt(lm, idx) for idx in indices]


def _fill_poly_mask(
    shape: tuple[int, int],
    points: list[tuple[float, float]],
) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    if len(points) >= 3:
        pts = np.round(np.asarray(points, dtype=np.float32)).astype(np.int32)
        cv2.fillPoly(mask, [pts], 255, cv2.LINE_AA)
    return mask.astype(np.float32) / 255.0


def _mask_bbox(mask: np.ndarray, threshold: float = 0.03) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask > threshold)
    if xs.size == 0 or ys.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _beard_asset_anchor_y_ratio(beard_rgba: np.ndarray) -> float:
    """Estimate where the asset's mustache band sits vertically."""
    if beard_rgba.ndim != 3 or beard_rgba.shape[2] < 4:
        return 0.5
    h, w = beard_rgba.shape[:2]
    if h <= 0 or w <= 0:
        return 0.5
    x1 = max(0, int(w * 0.25))
    x2 = min(w, int(w * 0.75))
    central_alpha = beard_rgba[:, x1:x2, 3].astype(np.float32).mean(axis=1)
    peak = float(central_alpha.max(initial=0.0))
    if peak <= 1.0:
        return 0.5
    threshold = max(12.0, peak * 0.16)
    rows = np.where(central_alpha > threshold)[0]
    if rows.size == 0:
        return 0.5
    anchor_y = float(rows[0]) + h * 0.055
    return float(np.clip(anchor_y / h, 0.28, 0.55))


def _resolve_beard_target_bgr(color: str | None) -> np.ndarray | None:
    if color is None:
        return None
    target = _BEARD_COLOR_TARGETS_BGR.get(color.lower())
    if target is None:
        return None
    return np.array(target, dtype=np.uint8)


def _upper_side_asset_weight(beard_rgba: np.ndarray) -> np.ndarray:
    """Weight the upper cheek/sideburn part of a wide beard asset."""
    h, w = beard_rgba.shape[:2]
    if h <= 0 or w <= 0:
        return np.zeros((0, 0), dtype=np.float32)
    yy, xx = np.indices((h, w), dtype=np.float32)
    xn = xx / max(float(w - 1), 1.0)
    yn = yy / max(float(h - 1), 1.0)
    alpha = beard_rgba[:, :, 3].astype(np.float32) / 255.0

    side = np.maximum(
        np.clip((0.46 - xn) / 0.46, 0.0, 1.0),
        np.clip((xn - 0.54) / 0.46, 0.0, 1.0),
    )
    upper = np.clip((yn - 0.05) / 0.16, 0.0, 1.0) * np.clip((0.62 - yn) / 0.30, 0.0, 1.0)
    bridge_x = np.clip((xn - 0.10) / 0.24, 0.0, 1.0) * np.clip((0.90 - xn) / 0.24, 0.0, 1.0)
    bridge_y = np.clip((yn - 0.34) / 0.10, 0.0, 1.0) * np.clip((0.55 - yn) / 0.13, 0.0, 1.0)
    hair = np.clip((alpha - 0.015) / 0.42, 0.0, 1.0)
    side_weight = side * upper * hair
    bridge_weight = bridge_x * bridge_y * hair * 0.72
    return np.clip(np.maximum(side_weight, bridge_weight), 0.0, 1.0)


def _boost_upper_side_beard_color(beard_rgba: np.ndarray, color: str | None) -> np.ndarray:
    """Make upper side beard fibers carry the selected color more clearly."""
    weight = _upper_side_asset_weight(beard_rgba)
    if weight.size == 0 or float(weight.max(initial=0.0)) <= 0.0:
        return beard_rgba

    target_bgr = _resolve_beard_target_bgr(color)
    if target_bgr is None:
        target_bgr = np.array((48, 74, 112), dtype=np.uint8)
    target_hsv = cv2.cvtColor(
        target_bgr.reshape(1, 1, 3),
        cv2.COLOR_BGR2HSV,
    )[0, 0].astype(np.float32)

    out = beard_rgba.copy()
    hsv = cv2.cvtColor(out[:, :, :3], cv2.COLOR_BGR2HSV).astype(np.float32)
    alpha = out[:, :, 3].astype(np.float32)

    side_mix = np.clip(weight * 0.96, 0.0, 0.92)
    hsv[:, :, 0] = hsv[:, :, 0] * (1.0 - side_mix) + target_hsv[0] * side_mix
    hsv[:, :, 1] = np.maximum(hsv[:, :, 1], target_hsv[1] * (0.78 + weight * 0.34))
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * (1.0 + weight * 0.11), 0.0, 255.0)
    out[:, :, :3] = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
    out[:, :, 3] = np.clip(alpha + (255.0 - alpha) * weight * 0.68, 0.0, 255.0).astype(np.uint8)
    return out


def _tint_beard_to_color(beard_rgba: np.ndarray, color: str | None) -> np.ndarray:
    """Apply the selected UI beard color while preserving hair texture."""
    target_bgr = _resolve_beard_target_bgr(color)
    if target_bgr is None:
        return beard_rgba

    out = beard_rgba.copy()
    alpha = out[:, :, 3].astype(np.float32) / 255.0
    hair = alpha > 0.03
    if not np.any(hair):
        return out

    hsv = cv2.cvtColor(out[:, :, :3], cv2.COLOR_BGR2HSV).astype(np.float32)
    target_hsv = cv2.cvtColor(
        target_bgr.reshape(1, 1, 3),
        cv2.COLOR_BGR2HSV,
    )[0, 0].astype(np.float32)

    values = hsv[:, :, 2][hair]
    lo = float(np.percentile(values, 8))
    hi = float(np.percentile(values, 98))
    texture = np.zeros_like(alpha, dtype=np.float32)
    if hi > lo + 1e-3:
        texture = np.clip((hsv[:, :, 2] - lo) / (hi - lo), 0.0, 1.0)

    desired_v = np.clip(target_hsv[2] * (0.54 + texture * 0.78), 0.0, 255.0)
    mix = np.where(hair, np.clip(0.34 + alpha * 1.35, 0.0, 1.0), 0.0)
    hue_mix = mix * 0.96
    sat_mix = mix * 0.94
    val_mix = mix * 0.84

    hsv[:, :, 0] = np.where(
        hair,
        hsv[:, :, 0] * (1.0 - hue_mix) + target_hsv[0] * hue_mix,
        hsv[:, :, 0],
    )
    hsv[:, :, 1] = np.where(
        hair,
        hsv[:, :, 1] * (1.0 - sat_mix) + target_hsv[1] * sat_mix,
        hsv[:, :, 1],
    )
    hsv[:, :, 2] = np.where(
        hair,
        hsv[:, :, 2] * (1.0 - val_mix) + desired_v * val_mix,
        hsv[:, :, 2],
    )

    out[:, :, :3] = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
    return _boost_upper_side_beard_color(out, color)


def _beard_region_mask(
    image_shape: tuple[int, int],
    lm: np.ndarray,
) -> np.ndarray:
    """Soft anatomical mask for beard and mustache placement."""
    h, w = image_shape[:2]
    p61 = _pt(lm, 61)
    p291 = _pt(lm, 291)
    p164 = _pt(lm, 164)
    p0 = _pt(lm, 0)
    mouth_w = max(_dist(lm, 61, 291), 1.0)
    roll_deg = math.degrees(math.atan2(p291[1] - p61[1], p291[0] - p61[0]))

    lower_poly = (
        [p61]
        + _points(lm, _LEFT_JAW_TO_CHIN)
        + _points(lm, _RIGHT_CHIN_TO_JAW)
        + [p291]
        + list(reversed(_points(lm, _LOWER_LIP_EDGE[1:-1])))
    )
    lower_mask = _fill_poly_mask((h, w), lower_poly)
    left_cheek_mask = _fill_poly_mask((h, w), _points(lm, _LEFT_FULL_BEARD_REGION))
    right_cheek_mask = _fill_poly_mask((h, w), _points(lm, _RIGHT_FULL_BEARD_REGION))

    mustache_mask = np.zeros((h, w), dtype=np.float32)
    mustache_cx = int(round((p61[0] + p291[0]) * 0.5))
    mustache_cy = int(round(p164[1] * 0.42 + p0[1] * 0.58))
    mustache_rx = max(2, int(round(mouth_w * 0.58)))
    mustache_ry = max(2, int(round(mouth_w * 0.15)))
    cv2.ellipse(
        mustache_mask,
        (mustache_cx, mustache_cy),
        (mustache_rx, mustache_ry),
        roll_deg,
        0,
        360,
        1.0,
        -1,
        cv2.LINE_AA,
    )

    connector_mask = np.zeros((h, w), dtype=np.float32)
    connector_ry = max(2, int(round(mouth_w * 0.24)))
    connector_rx = max(2, int(round(mouth_w * 0.32)))
    connector_y = int(round(mustache_cy + mouth_w * 0.09))
    connector_x_offset = mouth_w * 0.34
    for side in (-1, 1):
        connector_cx = int(round(mustache_cx + side * connector_x_offset))
        cv2.ellipse(
            connector_mask,
            (connector_cx, connector_y),
            (connector_rx, connector_ry),
            roll_deg + side * 18.0,
            0,
            360,
            1.0,
            -1,
            cv2.LINE_AA,
        )

    lip_mask = _fill_poly_mask((h, w), _points(lm, _OUTER_LIP_RING))
    guard_radius = max(1, int(round(mouth_w * 0.025)))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (guard_radius * 2 + 1, guard_radius * 2 + 1),
    )
    lip_guard = cv2.dilate(
        np.clip(lip_mask * 255.0, 0, 255).astype(np.uint8),
        kernel,
        iterations=1,
    ).astype(np.float32) / 255.0
    lip_guard = cv2.GaussianBlur(
        lip_guard,
        (0, 0),
        sigmaX=max(0.8, mouth_w * 0.015),
        sigmaY=max(0.8, mouth_w * 0.015),
    )

    mask = np.clip(
        lower_mask + left_cheek_mask + right_cheek_mask + mustache_mask + connector_mask,
        0.0,
        1.0,
    )
    edge_sigma = max(1.0, mouth_w * 0.018)
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=edge_sigma, sigmaY=edge_sigma)
    mask *= 1.0 - np.clip(lip_guard, 0.0, 1.0)
    return np.clip(mask, 0.0, 1.0)


# ---------------------------------------------------------------------------
# apply_sunglasses
# ---------------------------------------------------------------------------

def apply_sunglasses(
    image: np.ndarray,
    landmarks: np.ndarray,
    asset_path: str | None = None,
    style: str = "classic",
    scale_multiplier: float = 1.0,
    offset_x: int = 0,
    offset_y: int = 0,
    rotation_offset_deg: float = 0.0,
) -> np.ndarray:
    """Landmark-accurate sunglasses overlay.

    Placement (MediaPipe indices):
        33  = left eye outer corner
        263 = right eye outer corner

    Uses PNG asset when available; falls back to OpenCV procedural drawing.
    """
    lm = landmarks
    p33  = _pt(lm, 33)
    p263 = _pt(lm, 263)

    inter_eye = _dist(lm, 33, 263)
    roll_deg  = math.degrees(math.atan2(p263[1] - p33[1], p263[0] - p33[0]))
    roll_deg += rotation_offset_deg

    if abs(roll_deg) > 25:
        _log.warning(
            "apply_sunglasses: |roll|=%.1f° — extreme tilt, 2D overlay may look unnatural.",
            abs(roll_deg),
        )

    # Center: midpoint of outer eye corners, nudged DOWN 10% of inter-eye dist
    cx = (p33[0] + p263[0]) / 2.0 + offset_x
    cy = (p33[1] + p263[1]) / 2.0 + inter_eye * 0.10 + offset_y

    target_w = max(4, int(inter_eye * 1.6 * scale_multiplier))

    resolved = Path(asset_path) if asset_path else (_ASSETS_DIR / "sunglasses.png")
    if resolved.is_file():
        raw = cv2.imread(str(resolved), cv2.IMREAD_UNCHANGED)
        if raw is not None:
            if raw.ndim == 2:
                raw = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGRA)
            elif raw.shape[2] == 3:
                raw = cv2.cvtColor(raw, cv2.COLOR_BGR2BGRA)

            ah, aw = raw.shape[:2]
            target_h = max(1, int(target_w * ah / aw))
            interp   = cv2.INTER_AREA if target_w < aw else cv2.INTER_CUBIC
            resized  = cv2.resize(raw, (target_w, target_h), interpolation=interp)

            # If asset is nearly opaque, reduce to 85% so eye outlines show through
            if resized[:, :, 3].mean() > 230:
                resized = resized.copy()
                resized[:, :, 3] = np.clip(
                    resized[:, :, 3].astype(np.float32) * 0.85, 0, 255
                ).astype(np.uint8)

            rotated = _rotate_asset(resized, -roll_deg)
            return alpha_blend(image, rotated, (int(cx), int(cy)))

    # Procedural fallback
    return _procedural_sunglasses(image, lm, cx, cy, inter_eye, roll_deg, scale_multiplier, style)


def _procedural_sunglasses(
    image: np.ndarray,
    lm: np.ndarray,
    cx: float,
    cy: float,
    inter_eye: float,
    roll_deg: float,
    scale: float,
    style: str = "classic",
) -> np.ndarray:
    """Draw sunglasses using OpenCV primitives with spec-aligned placement."""
    h, w = image.shape[:2]
    is_clear    = (style == "glasses")
    angle_rad   = math.radians(roll_deg)

    # Per-eye centres derived from spec anchor (cx, cy is bridge midpoint)
    # Offset half of inter_eye along the roll direction to each eye
    dx, dy = math.cos(angle_rad), math.sin(angle_rad)
    l_cx = cx - dx * inter_eye * 0.5
    l_cy = cy - dy * inter_eye * 0.5
    r_cx = cx + dx * inter_eye * 0.5
    r_cy = cy + dy * inter_eye * 0.5

    lens_rx = inter_eye * _STYLE_LENS_X.get(style, 0.34) * scale
    lens_ry = lens_rx   * _STYLE_LENS_RATIO.get(style, 0.68)

    lens_color  = _STYLE_LENS_COLOR.get(style,  (22, 22, 26))
    frame_color = _STYLE_FRAME_COLOR.get(style, (10, 8, 8))
    frame_thick = max(3, int(inter_eye * 0.018 * scale))
    if style == "aviator":
        frame_thick = max(1, int(frame_thick * 0.55))

    overlay   = image.copy()
    lens_mask = np.zeros((h, w), dtype=np.float32)

    for ecx, ecy in ((l_cx, l_cy), (r_cx, r_cy)):
        _draw_lens(overlay, lens_mask, ecx, ecy, lens_rx, lens_ry,
                   roll_deg, style, lens_color, frame_color, frame_thick)

    _draw_bridge(overlay, l_cx, l_cy, r_cx, r_cy,
                 lens_rx, lens_ry, roll_deg, frame_color, frame_thick)

    if style == "aviator":
        _draw_aviator_brow_bar(overlay, l_cx, l_cy, r_cx, r_cy,
                               lens_rx, lens_ry, angle_rad, frame_color, frame_thick)

    pad_size = (max(3, int(lens_rx * 0.08)), max(2, int(lens_ry * 0.07)))
    for ecx, ecy, sign in ((l_cx, l_cy, 1), (r_cx, r_cy, -1)):
        px = int(ecx + sign * lens_rx * 0.72 * dx)
        py = int(ecy + sign * lens_rx * 0.72 * dy + lens_ry * 0.22)
        pad_col  = (238, 236, 226) if is_clear else (110, 108, 100)
        edge_col = (95, 92, 86)   if is_clear else (20, 20, 20)
        cv2.ellipse(overlay, (px, py), pad_size, roll_deg, 0, 360, pad_col,  -1, cv2.LINE_AA)
        cv2.ellipse(overlay, (px, py), pad_size, roll_deg, 0, 360, edge_col,  1, cv2.LINE_AA)

    temple_len = inter_eye * 0.52 * scale
    l_ts = (int(l_cx), int(l_cy))
    r_ts = (int(r_cx), int(r_cy))
    if style == "aviator":
        ux, uy = -dy, -dx
        bh = lens_ry * 0.93
        l_ts = (int(l_cx - dx * lens_rx * 1.05 + ux * bh),
                int(l_cy - dy * lens_rx * 1.05 + uy * bh))
        r_ts = (int(r_cx + dx * lens_rx * 1.05 + ux * bh),
                int(r_cy + dy * lens_rx * 1.05 + uy * bh))
    l_temple = (int(l_cx - dx * temple_len), int(l_cy - dy * temple_len))
    r_temple = (int(r_cx + dx * temple_len), int(r_cy + dy * temple_len))
    _draw_tapered_temple(overlay, l_ts, l_temple, frame_color, frame_thick)
    _draw_tapered_temple(overlay, r_ts, r_temple, frame_color, frame_thick)

    refl_mask = np.zeros((h, w), dtype=np.float32)
    if not is_clear:
        for ecx, ecy in ((l_cx, l_cy), (r_cx, r_cy)):
            _draw_specular(overlay, refl_mask, ecx, ecy, lens_rx, lens_ry, roll_deg, style)

    LENS_ALPHA = 0.28 if is_clear else 0.76
    if is_clear:
        out = _composite_clear_lenses(image, lens_mask)
    else:
        out = cv2.addWeighted(overlay, LENS_ALPHA, image, 1.0 - LENS_ALPHA, 0)

    for ecx, ecy in ((l_cx, l_cy), (r_cx, r_cy)):
        _draw_frame_outline(out, ecx, ecy, lens_rx, lens_ry, roll_deg, frame_color, frame_thick, style)
    _draw_bridge(out, l_cx, l_cy, r_cx, r_cy,
                 lens_rx, lens_ry, roll_deg, frame_color, frame_thick)
    if style == "aviator":
        _draw_aviator_brow_bar(out, l_cx, l_cy, r_cx, r_cy,
                               lens_rx, lens_ry, angle_rad, frame_color, frame_thick)
    _draw_tapered_temple(out, l_ts, l_temple, frame_color, frame_thick)
    _draw_tapered_temple(out, r_ts, r_temple, frame_color, frame_thick)

    if is_clear:
        for ecx, ecy in ((l_cx, l_cy), (r_cx, r_cy)):
            _draw_glass_highlights(out, refl_mask, ecx, ecy, lens_rx, lens_ry, roll_deg)

    return out


# ---------------------------------------------------------------------------
# apply_beard
# ---------------------------------------------------------------------------

def apply_beard(
    image: np.ndarray,
    landmarks: np.ndarray,
    asset_path: str | None = None,
    color: str = "black",
    scale_multiplier: float = 1.0,
    offset_x: int = 0,
    offset_y: int = 0,
    rotation_offset_deg: float = 0.0,
    color_match: bool = True,
    debug: bool = False,
) -> np.ndarray:
    """Landmark-accurate beard overlay.

    Placement (MediaPipe indices):
        164 = philtrum (under nose) — top anchor
        152 = chin tip              — bottom anchor
        172 = left jaw              397 = right jaw
        61  = left mouth corner     291 = right mouth corner

    Scaling follows the full cheek/jaw beard mask.
    The asset's central mustache band is aligned to the philtrum/top-lip anchor.
    """
    lm = landmarks

    p164 = _pt(lm, 164)
    p152 = _pt(lm, 152)
    p172 = _pt(lm, 172)
    p397 = _pt(lm, 397)
    p61  = _pt(lm, 61)
    p291 = _pt(lm, 291)
    p0   = _pt(lm, 0)

    jaw_span         = _dist(lm, 172, 397)
    philtrum_to_chin = _dist(lm, 164, 152)

    roll_deg  = math.degrees(math.atan2(p291[1] - p61[1], p291[0] - p61[0]))
    roll_deg += rotation_offset_deg

    paste_cx_base = (p172[0] + p397[0]) / 2.0

    # Yaw detection → horizontal squash
    squash = 1.0
    if len(lm) > 454:
        nose_x    = _pt(lm, 1)[0]
        ear_mid_x = (_pt(lm, 234)[0] + _pt(lm, 454)[0]) / 2.0
        face_w    = _dist(lm, 234, 454)
        if face_w > 1.0:
            yaw_ratio = (nose_x - ear_mid_x) / face_w
            if abs(yaw_ratio) > 0.3:
                squash = 1.0 - 0.4 * abs(yaw_ratio)
        squash = float(np.clip(squash, 0.3, 1.0))

    # Full-beard target covers cheek, jaw, chin, and mustache regions.
    region_mask = _beard_region_mask(image.shape[:2], lm)
    region_bbox = _mask_bbox(region_mask)
    if region_bbox is not None:
        rx1, ry1, rx2, ry2 = region_bbox
        paste_cx_base = (rx1 + rx2) / 2.0
        target_h = max(4, int((ry2 - ry1) * scale_multiplier))
        target_w = max(4, int((rx2 - rx1) * scale_multiplier))
    else:
        target_h = max(4, int(philtrum_to_chin * 1.18 * scale_multiplier))
        target_w = max(4, int(jaw_span * 1.25 * scale_multiplier * squash))

    resolved = Path(asset_path) if asset_path else (_ASSETS_DIR / "beard.png")
    if resolved.is_file():
        raw = cv2.imread(str(resolved), cv2.IMREAD_UNCHANGED)
        if raw is not None:
            if raw.ndim == 2:
                raw = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)

            # Build BGRA — derive alpha from white background if no genuine alpha
            if raw.shape[2] == 4 and raw[:, :, 3].mean() < 250:
                sg = raw.copy()
            else:
                bgr     = raw[:, :, :3]
                gray    = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                alpha_d = np.clip((255 - gray.astype(np.float32)) * 2.0, 0, 255).astype(np.uint8)
                sg      = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
                sg[:, :, 3] = alpha_d

            # Crop to content bounding box so whitespace doesn't skew scale
            a_ch = sg[:, :, 3]
            rows_nz = np.any(a_ch > 10, axis=1)
            cols_nz = np.any(a_ch > 10, axis=0)
            if rows_nz.any() and cols_nz.any():
                ct = int(np.where(rows_nz)[0][0])
                cb = int(np.where(rows_nz)[0][-1]) + 1
                cl = int(np.where(cols_nz)[0][0])
                cr = int(np.where(cols_nz)[0][-1]) + 1
                sg = sg[ct:cb, cl:cr]

            # Prefer explicit UI colors; fall back to scalp matching for custom callers.
            if color_match:
                if _resolve_beard_target_bgr(color) is not None:
                    sg = _tint_beard_to_color(sg, color)
                else:
                    sg = _tint_beard_from_scalp(sg, image, lm)

            # Feather alpha edges with 7×7 blur
            sg = sg.copy()
            sg[:, :, 3] = cv2.GaussianBlur(sg[:, :, 3], (7, 7), 0)

            # Height-dominant scale; floor prevents invisible beard for portrait assets
            ah, aw = sg.shape[:2]
            anchor_y_ratio = _beard_asset_anchor_y_ratio(sg)
            s_h = target_h / max(ah, 1)
            s_w = target_w / max(aw, 1)
            s   = max(s_h, s_w) * 1.02
            rw  = max(4, int(aw * s))
            rh  = max(4, int(ah * s))
            interp  = cv2.INTER_AREA if s < 1.0 else cv2.INTER_CUBIC
            resized = cv2.resize(sg, (rw, rh), interpolation=interp)

            rotated = _rotate_asset(resized, -roll_deg)

            # Align the source mustache band to the target upper-lip region.
            paste_cx = int(paste_cx_base + offset_x)
            mustache_target_y = p164[1] * 0.42 + p0[1] * 0.58
            paste_cy = int(mustache_target_y + rh * (0.5 - anchor_y_ratio) + offset_y)

            out = alpha_blend(image, rotated, (paste_cx, paste_cy), alpha_mask=region_mask)

            if debug:
                for idx in (164, 152, 172, 397):
                    px, py = int(_pt(lm, idx)[0]), int(_pt(lm, idx)[1])
                    cv2.circle(out, (px, py), 4, (0, 0, 255), -1, cv2.LINE_AA)
                rh_rot, rw_rot = rotated.shape[:2]
                bx1, by1 = paste_cx - rw_rot // 2, paste_cy - rh_rot // 2
                cv2.rectangle(out, (bx1, by1), (bx1 + rw_rot, by1 + rh_rot), (0, 255, 0), 2)

            return out

    # Procedural fallback
    return _procedural_beard(image, lm, scale_multiplier, squash, color)


def _tint_beard_from_scalp(
    beard_rgba: np.ndarray,
    image: np.ndarray,
    lm: np.ndarray,
) -> np.ndarray:
    """Tint beard toward scalp hair color at 60% HSV strength.

    Samples a 40-px radius patch above lm10 (forehead/hairline), removes skin
    tones via HSV thresholding, then blends H and S channels toward hair color.
    """
    h, w = image.shape[:2]
    if 10 >= len(lm):
        return beard_rgba

    cx = int(np.clip(lm[10, 0], 0, w - 1))
    cy = int(np.clip(lm[10, 1], 0, h - 1))
    r  = 40
    x0, x1 = max(0, cx - r), min(w, cx + r)
    y0 = max(0, cy - r)
    y1 = max(y0 + 1, cy)  # only above the forehead landmark

    if y1 <= y0:
        return beard_rgba

    patch = image[y0:y1, x0:x1]
    if patch.size == 0:
        return beard_rgba

    patch_hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    skin1 = cv2.inRange(patch_hsv, np.array([0, 30, 80]),   np.array([25, 170, 255]))
    skin2 = cv2.inRange(patch_hsv, np.array([160, 30, 80]), np.array([180, 170, 255]))
    hair_mask = ~(skin1.astype(bool) | skin2.astype(bool))

    hair_pixels_bgr = patch.reshape(-1, 3)[hair_mask.ravel()]
    if len(hair_pixels_bgr) < 10:
        return beard_rgba

    hair_bgr_mean = hair_pixels_bgr.mean(axis=0).astype(np.uint8)
    hair_pixel_img = np.full((1, 1, 3), hair_bgr_mean, dtype=np.uint8)
    hair_hsv = cv2.cvtColor(hair_pixel_img, cv2.COLOR_BGR2HSV)[0, 0].astype(np.float32)

    out = beard_rgba.copy()
    hsv = cv2.cvtColor(out[:, :, :3], cv2.COLOR_BGR2HSV).astype(np.float32)

    BLEND = 0.60
    hsv[:, :, 0] = hsv[:, :, 0] * (1.0 - BLEND) + hair_hsv[0] * BLEND
    hsv[:, :, 1] = hsv[:, :, 1] * (1.0 - BLEND) + hair_hsv[1] * BLEND
    # V (brightness) is preserved to keep beard contrast

    out[:, :, :3] = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
    return out


def _procedural_beard(
    image: np.ndarray,
    lm: np.ndarray,
    scale: float,
    squash: float,
    color: str = "black",
) -> np.ndarray:
    """Draw a beard with OpenCV fill + directional noise texture."""
    h, w = image.shape[:2]
    beard_mask = _beard_region_mask((h, w), lm)
    if float(beard_mask.max(initial=0.0)) <= 0.0:
        return image.copy()

    mouth_w = max(_dist(lm, 61, 291), 1.0)
    face_cx = (_pt(lm, 61)[0] + _pt(lm, 291)[0]) * 0.5
    lower_lip_y = _pt(lm, 17)[1]
    base_bgr = _resolve_beard_target_bgr(color)
    if base_bgr is None:
        base_bgr = np.array(_BEARD_COLOR_TARGETS_BGR["black"], dtype=np.uint8)
    base_color = base_bgr.astype(np.float32)

    feather = cv2.GaussianBlur(
        beard_mask,
        (0, 0),
        sigmaX=max(1.2, mouth_w * 0.018),
        sigmaY=max(1.2, mouth_w * 0.018),
    )
    alpha_strength = float(np.clip(0.62 + (scale - 1.0) * 0.12, 0.42, 0.82))
    out = image.astype(np.float32)
    alpha = (feather * alpha_strength)[:, :, np.newaxis]
    out = out * (1.0 - alpha) + base_color * alpha

    rng = np.random.default_rng(42)
    noise = rng.normal(0.0, 1.0, (h, w)).astype(np.float32)
    noise = cv2.GaussianBlur(noise, (1, 9), 0)
    noise -= float(noise.min())
    denom = float(noise.max())
    if denom > 1e-6:
        noise /= denom
    out -= noise[:, :, np.newaxis] * beard_mask[:, :, np.newaxis] * 22.0

    ys, xs = np.where(beard_mask > 0.22)
    if xs.size:
        count = min(2200, max(220, xs.size // 38))
        chosen = rng.choice(xs.size, size=count, replace=xs.size < count)
        strand_layer = np.zeros((h, w, 3), dtype=np.float32)
        strand_alpha = np.zeros((h, w), dtype=np.float32)
        min_len = max(2.0, mouth_w * 0.022)
        max_len = max(min_len + 1.0, mouth_w * 0.070)

        for idx in np.atleast_1d(chosen):
            x = float(xs[idx])
            y = float(ys[idx])
            if y < lower_lip_y:
                angle = 0.12 if x >= face_cx else math.pi - 0.12
            else:
                side = np.clip((x - face_cx) / max(mouth_w * 1.8, 1.0), -0.48, 0.48)
                angle = math.pi * 0.5 + side
            length = float(rng.uniform(min_len, max_len))
            dx = math.cos(angle) * length * 0.5
            dy = math.sin(angle) * length * 0.5
            p1 = (int(round(x - dx)), int(round(y - dy)))
            p2 = (int(round(x + dx)), int(round(y + dy)))
            color = np.clip(base_color + rng.normal(0.0, 8.0, 3), 0.0, 90.0)
            cv2.line(
                strand_layer,
                p1,
                p2,
                tuple(float(c) for c in color),
                1,
                cv2.LINE_AA,
            )
            cv2.line(
                strand_alpha,
                p1,
                p2,
                float(rng.uniform(0.18, 0.55)),
                1,
                cv2.LINE_AA,
            )

        strand_alpha = cv2.GaussianBlur(strand_alpha, (0, 0), 0.35)
        strand_a = np.clip(strand_alpha * beard_mask * 0.9, 0.0, 0.85)[:, :, np.newaxis]
        out = out * (1.0 - strand_a) + strand_layer * strand_a

    return np.clip(out, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# FaceOverlayEngine  (delegates to module-level functions)
# ---------------------------------------------------------------------------

class FaceOverlayEngine:
    """Applies declarative overlay effects to a face image."""

    _EFFECT_REGISTRY: dict[str, str] = {
        "sunglasses":     "_effect_sunglasses",
        "sunglasses_png": "_effect_sunglasses_png",
        "beard":          "_effect_beard",
        "beard_png":      "_effect_beard_png",
    }

    def __init__(self, landmarks, image: np.ndarray) -> None:
        if isinstance(landmarks, np.ndarray):
            self._pixels = landmarks
        else:
            self._pixels = landmarks.pixels
        self.image = image

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def apply(self, effects: list[str], params: dict) -> OverlayResult:
        current     = self.image.copy()
        stages:     list[np.ndarray] = []
        all_masks:  dict[str, np.ndarray] = {}
        all_meta:   dict[str, Any] = {}

        for name in effects:
            method_name = self._EFFECT_REGISTRY.get(name)
            if method_name is None:
                raise ValueError(
                    f"Unknown effect {name!r}. Available: {list(self._EFFECT_REGISTRY)}"
                )
            method  = getattr(self, method_name)
            current, masks, meta = method(current, **(params.get(name) or {}))
            stages.append(current.copy())
            all_masks.update(masks)
            all_meta[name] = meta

        return OverlayResult(
            final_image=current,
            intermediate_stages=stages,
            masks_used=all_masks,
            dsp_metadata=all_meta,
        )

    # ------------------------------------------------------------------
    # Effects
    # ------------------------------------------------------------------

    def _effect_sunglasses(
        self,
        image: np.ndarray,
        style: str = "classic",
        offset_x: float = 0.0,
        offset_y: float = 0.0,
        scale: float = 1.0,
        rotation: float = 0.0,
    ) -> tuple[np.ndarray, dict, dict]:
        out = apply_sunglasses(
            image, self._pixels,
            asset_path=None,
            style=style,
            scale_multiplier=float(scale),
            offset_x=int(offset_x),
            offset_y=int(offset_y),
            rotation_offset_deg=float(rotation),
        )
        lm       = self._pixels
        roll     = math.degrees(math.atan2(
            _pt(lm, 263)[1] - _pt(lm, 33)[1],
            _pt(lm, 263)[0] - _pt(lm, 33)[0],
        ))
        blank    = np.zeros(image.shape[:2], dtype=np.float32)
        meta     = {
            "lens_style": style,
            "inter_eye_angle": round(roll, 2),
            "color_space": "BGR",
            "dsp_technique": "landmark-accurate compositing (spec v2)",
        }
        return out, {"sunglasses_lens": blank, "sunglasses_reflection": blank.copy()}, meta

    def _effect_sunglasses_png(
        self,
        image: np.ndarray,
        png_path: str | None = None,
        offset_x: float = 0.0,
        offset_y: float = 0.0,
        scale: float = 1.0,
        rotation: float = 0.0,
    ) -> tuple[np.ndarray, dict, dict]:
        out   = apply_sunglasses(
            image, self._pixels,
            asset_path=png_path or str(_ASSETS_DIR / "sunglasses.png"),
            scale_multiplier=float(scale),
            offset_x=int(offset_x),
            offset_y=int(offset_y),
            rotation_offset_deg=float(rotation),
        )
        blank = np.zeros(image.shape[:2], dtype=np.float32)
        meta  = {"dsp_technique": "alpha-channel PNG compositing"}
        return out, {"sunglasses_lens": blank, "sunglasses_reflection": blank.copy()}, meta

    def _effect_beard(
        self,
        image: np.ndarray,
        color: str = "black",
        offset_x: float = 0.0,
        offset_y: float = 0.0,
        scale: float = 1.0,
    ) -> tuple[np.ndarray, dict, dict]:
        out   = apply_beard(
            image, self._pixels,
            asset_path=None,
            color=color,
            scale_multiplier=float(scale),
            offset_x=int(offset_x),
            offset_y=int(offset_y),
            color_match=True,
        )
        blank = np.zeros(image.shape[:2], dtype=np.float32)
        meta  = {
            "beard_color": color,
            "color_space": "BGR",
            "dsp_technique": "landmark-accurate compositing (spec v2)",
        }
        return out, {"beard_region": blank}, meta

    def _effect_beard_png(
        self,
        image: np.ndarray,
        png_path: str | None = None,
        color: str = "black",
        offset_x: float = 0.0,
        offset_y: float = 0.0,
        scale: float = 1.0,
        rotation: float = 0.0,
    ) -> tuple[np.ndarray, dict, dict]:
        out   = apply_beard(
            image, self._pixels,
            asset_path=png_path or str(_ASSETS_DIR / "beard.png"),
            color=color,
            scale_multiplier=float(scale),
            offset_x=int(offset_x),
            offset_y=int(offset_y),
            rotation_offset_deg=float(rotation),
        )
        blank = np.zeros(image.shape[:2], dtype=np.float32)
        meta  = {
            "beard_color": color,
            "dsp_technique": "white-background removal + landmark-accurate alpha compositing",
        }
        return out, {"beard_region": blank}, meta

    # Keep old names as aliases so any external callers still work
    _apply_sunglasses     = _effect_sunglasses
    _apply_sunglasses_png = _effect_sunglasses_png
    _apply_beard          = _effect_beard
    _apply_beard_png      = _effect_beard_png


# ---------------------------------------------------------------------------
# Style tables (procedural sunglasses)
# ---------------------------------------------------------------------------

_STYLE_LENS_X: dict[str, float] = {
    "classic": 0.34, "aviator": 0.36, "round": 0.32, "glasses": 0.34,
}
_STYLE_LENS_RATIO: dict[str, float] = {
    "classic": 0.68, "aviator": 1.20, "round": 1.08, "glasses": 0.72,
}
_STYLE_LENS_COLOR: dict[str, tuple[int, int, int]] = {
    "classic": (22, 22, 26), "aviator": (18, 34, 22),
    "round":   (28, 20, 22), "glasses": (235, 230, 220),
}
_STYLE_FRAME_COLOR: dict[str, tuple[int, int, int]] = {
    "classic": (10, 8, 8), "aviator": (22, 18, 12),
    "round":   (18, 12, 8), "glasses": (40, 35, 30),
}


# ---------------------------------------------------------------------------
# OpenCV drawing helpers (procedural sunglasses)
# ---------------------------------------------------------------------------

def _rotate_translate(points: np.ndarray, cx: float, cy: float, angle_deg: float) -> np.ndarray:
    angle = math.radians(angle_deg)
    rot   = np.array([
        [math.cos(angle), -math.sin(angle)],
        [math.sin(angle),  math.cos(angle)],
    ], dtype=np.float32)
    shifted = points @ rot.T
    shifted[:, 0] += cx
    shifted[:, 1] += cy
    return np.round(shifted).astype(np.int32)


def _draw_lens(canvas, mask, cx, cy, rx, ry, angle_deg, style, fill_color, frame_color, frame_thick):
    h, w = canvas.shape[:2]
    irx, iry = max(1, int(rx)), max(1, int(ry))

    if style == "aviator":
        outer = np.array([
            [-0.52*irx,-0.90*iry],[-0.18*irx,-0.98*iry],[ 0.18*irx,-0.98*iry],
            [ 0.52*irx,-0.90*iry],[ 1.02*irx,-0.40*iry],[ 1.10*irx, 0.18*iry],
            [ 0.96*irx, 0.72*iry],[ 0.68*irx, 1.04*iry],[ 0.28*irx, 1.16*iry],
            [ 0.00*irx, 1.20*iry],[-0.28*irx, 1.16*iry],[-0.68*irx, 1.04*iry],
            [-0.96*irx, 0.72*iry],[-1.10*irx, 0.18*iry],[-1.02*irx,-0.40*iry],
        ], dtype=np.float32)
        outer_pts = _rotate_translate(outer, cx, cy, angle_deg)
        cv2.fillPoly(canvas, [outer_pts], fill_color, cv2.LINE_AA)
        cv2.fillPoly(mask,   [outer_pts], 1.0,        cv2.LINE_AA)
        y_lo, y_hi = float(outer_pts[:,1].min()), float(outer_pts[:,1].max())
        if y_hi > y_lo:
            gm = np.zeros((h, w), dtype=np.float32)
            cv2.fillPoly(gm, [outer_pts], 1.0, cv2.LINE_AA)
            y_t = np.clip((np.arange(h, dtype=np.float32)[:,np.newaxis]-y_lo)/(y_hi-y_lo), 0, 1)
            c_f = canvas.astype(np.float32)
            c_f[:,:,0] = np.clip(c_f[:,:,0]+y_t*72.0*gm, 0, 255)
            c_f[:,:,1] = np.clip(c_f[:,:,1]+y_t*68.0*gm, 0, 255)
            c_f[:,:,2] = np.clip(c_f[:,:,2]+y_t*82.0*gm, 0, 255)
            canvas[:] = c_f.astype(np.uint8)
        cv2.polylines(canvas, [outer_pts], True, frame_color, max(1, frame_thick), cv2.LINE_AA)
        hi = tuple(min(255, c+48) for c in frame_color)
        cv2.polylines(canvas, [outer_pts], True, hi, 1, cv2.LINE_AA)
        return

    top_lift = -0.80*iry
    outer = np.array([
        [-1.16*irx, top_lift],[-0.55*irx,-0.96*iry],[ 1.14*irx,-0.80*iry],
        [ 1.10*irx,-0.14*iry],[ 0.80*irx, 0.84*iry],[ 0.22*irx, 1.04*iry],
        [-0.60*irx, 0.97*iry],[-1.04*irx, 0.52*iry],[-1.20*irx,-0.18*iry],
    ], dtype=np.float32)
    inner = outer * np.array([0.80, 0.76], dtype=np.float32)
    inner[:,1] += iry * 0.07
    bevel = outer * np.array([0.90, 0.88], dtype=np.float32)
    bevel[:,1] += iry * 0.03

    outer_pts = _rotate_translate(outer, cx, cy, angle_deg)
    bevel_pts = _rotate_translate(bevel, cx, cy, angle_deg)
    inner_pts = _rotate_translate(inner, cx, cy, angle_deg)

    cv2.fillPoly(canvas, [outer_pts], frame_color, cv2.LINE_AA)
    bevel_hi = tuple(min(255, c+55) for c in frame_color)
    cv2.polylines(canvas, [bevel_pts], True, bevel_hi, max(1, frame_thick//2), cv2.LINE_AA)
    cv2.fillPoly(canvas, [inner_pts], fill_color, cv2.LINE_AA)

    if fill_color[0] < 150:
        gm = np.zeros(canvas.shape[:2], dtype=np.float32)
        cv2.fillPoly(gm, [inner_pts], 1.0, cv2.LINE_AA)
        y_lo = float(inner_pts[:,1].min()); y_hi = float(inner_pts[:,1].max())
        if y_hi > y_lo:
            y_t    = np.clip((np.arange(h, dtype=np.float32)[:,np.newaxis]-y_lo)/(y_hi-y_lo), 0, 1)
            bright = (y_t*60.0*gm)[:,:,np.newaxis]
            canvas[:] = np.clip(canvas.astype(np.float32)+bright, 0, 255).astype(np.uint8)

    edge_shade = tuple(max(0, c-8) for c in frame_color)
    cv2.polylines(canvas, [outer_pts], True, edge_shade, max(2, frame_thick), cv2.LINE_AA)
    cv2.polylines(canvas, [inner_pts], True, (6, 6, 8), max(1, frame_thick-1), cv2.LINE_AA)
    cv2.fillPoly(mask, [inner_pts], 1.0, cv2.LINE_AA)

    rivet_local = np.array([[-0.98*irx, -0.54*iry]], dtype=np.float32)
    rivet = _rotate_translate(rivet_local, cx, cy, angle_deg)[0]
    rv = (max(4, irx//11), max(3, iry//14))
    cv2.ellipse(canvas, tuple(rivet), rv, angle_deg, 0, 360, (180,178,168), -1, cv2.LINE_AA)
    cv2.ellipse(canvas, tuple(rivet), rv, angle_deg, 0, 360, (15,15,15),    1,  cv2.LINE_AA)
    shine_local = np.array([[-0.94*irx, -0.58*iry]], dtype=np.float32)
    shine = _rotate_translate(shine_local, cx, cy, angle_deg)[0]
    cv2.ellipse(canvas, tuple(shine), (max(1, rv[0]//3), max(1, rv[1]//3)),
                angle_deg, 0, 360, (230,228,220), -1, cv2.LINE_AA)


def _draw_frame_outline(canvas, cx, cy, rx, ry, angle_deg, frame_color, frame_thick, style="classic"):
    irx, iry = max(1, int(rx)), max(1, int(ry))
    if style == "aviator":
        outer = np.array([
            [-0.52*irx,-0.90*iry],[-0.18*irx,-0.98*iry],[ 0.18*irx,-0.98*iry],
            [ 0.52*irx,-0.90*iry],[ 1.02*irx,-0.40*iry],[ 1.10*irx, 0.18*iry],
            [ 0.96*irx, 0.72*iry],[ 0.68*irx, 1.04*iry],[ 0.28*irx, 1.16*iry],
            [ 0.00*irx, 1.20*iry],[-0.28*irx, 1.16*iry],[-0.68*irx, 1.04*iry],
            [-0.96*irx, 0.72*iry],[-1.10*irx, 0.18*iry],[-1.02*irx,-0.40*iry],
        ], dtype=np.float32)
        pts = _rotate_translate(outer, cx, cy, angle_deg)
        cv2.polylines(canvas, [pts], True, frame_color, max(1, frame_thick), cv2.LINE_AA)
        hi = tuple(min(255, c+48) for c in frame_color)
        cv2.polylines(canvas, [pts], True, hi, 1, cv2.LINE_AA)
        return
    top_lift = -0.78*iry
    outer = np.array([
        [-1.14*irx, top_lift],[-0.62*irx,-0.92*iry],[ 1.12*irx,-0.78*iry],
        [ 1.08*irx,-0.18*iry],[ 0.78*irx, 0.82*iry],[ 0.28*irx, 1.02*iry],
        [-0.64*irx, 0.95*iry],[-1.02*irx, 0.50*iry],[-1.18*irx,-0.20*iry],
    ], dtype=np.float32)
    outer_pts = _rotate_translate(outer, cx, cy, angle_deg)
    cv2.polylines(canvas, [outer_pts], True, frame_color, max(2, frame_thick), cv2.LINE_AA)
    rivet_local = np.array([[-0.98*irx, -0.52*iry]], dtype=np.float32)
    rivet = _rotate_translate(rivet_local, cx, cy, angle_deg)[0]
    cv2.ellipse(canvas, tuple(rivet), (max(3, irx//13), max(2, iry//16)),
                angle_deg, 0, 360, (215,215,205), -1, cv2.LINE_AA)
    cv2.ellipse(canvas, tuple(rivet), (max(3, irx//13), max(2, iry//16)),
                angle_deg, 0, 360, (20,20,20), 1, cv2.LINE_AA)


def _draw_bridge(canvas, l_cx, l_cy, r_cx, r_cy, rx, ry, angle_deg, color, thick):
    angle_rad = math.radians(angle_deg)
    dx, dy    = math.cos(angle_rad), math.sin(angle_rad)
    perp_x, perp_y = -dy, dx
    l_ex = l_cx + dx*rx*0.82; l_ey = l_cy + dy*rx*0.82
    r_ex = r_cx - dx*rx*0.82; r_ey = r_cy - dy*rx*0.82
    ctrl_x = (l_ex+r_ex)/2.0 + perp_x*ry*0.54
    ctrl_y = (l_ey+r_ey)/2.0 + perp_y*ry*0.54
    n   = 22
    pts = []
    for i in range(n+1):
        t  = i/n
        bx = (1-t)**2*l_ex + 2*(1-t)*t*ctrl_x + t**2*r_ex
        by = (1-t)**2*l_ey + 2*(1-t)*t*ctrl_y + t**2*r_ey
        pts.append([int(round(bx)), int(round(by))])
    pts_arr = np.array(pts, dtype=np.int32)
    cv2.polylines(canvas, [pts_arr], False, color, max(2, thick), cv2.LINE_AA)
    hi = tuple(min(255, c+50) for c in color)
    cv2.polylines(canvas, [pts_arr], False, hi, 1, cv2.LINE_AA)


def _draw_aviator_brow_bar(canvas, l_cx, l_cy, r_cx, r_cy, rx, ry, angle_rad, color, thick):
    dx, dy  = math.cos(angle_rad), math.sin(angle_rad)
    ux, uy  = -math.sin(angle_rad), -math.cos(angle_rad)
    h_off   = ry * 0.94; span = rx * 1.06
    lx = int(l_cx - dx*span + ux*h_off); ly = int(l_cy - dy*span + uy*h_off)
    rx_ = int(r_cx + dx*span + ux*h_off); ry_ = int(r_cy + dy*span + uy*h_off)
    cv2.line(canvas, (lx, ly), (rx_, ry_), color, max(1, thick), cv2.LINE_AA)
    hi = tuple(min(255, c+52) for c in color)
    cv2.line(canvas, (lx, ly), (rx_, ry_), hi, 1, cv2.LINE_AA)


def _draw_tapered_temple(canvas, start, end, color, thick_start):
    n = 16
    for i in range(n):
        t1, t2 = i/n, (i+1)/n
        p1 = (int(start[0]+t1*(end[0]-start[0])), int(start[1]+t1*(end[1]-start[1])))
        p2 = (int(start[0]+t2*(end[0]-start[0])), int(start[1]+t2*(end[1]-start[1])))
        thick = max(1, int(round(thick_start*(1.0-((t1+t2)/2)*0.58))))
        cv2.line(canvas, p1, p2, color, thick, cv2.LINE_AA)


def _draw_specular(canvas, mask, cx, cy, rx, ry, angle_deg, style="classic"):
    h, w = canvas.shape[:2]
    if style == "aviator":
        hi_rx, hi_ry = max(2, int(rx*0.82)), max(1, int(ry*0.44))
        hi_cx  = int(np.clip(cx-rx*0.04, 0, w-1))
        hi_cy  = int(np.clip(cy-ry*0.26, 0, h-1))
        sigma_x, sigma_y = rx*0.44, ry*0.22
        band_alpha = 0.28
        glint_x = int(np.clip(cx-rx*0.38, 0, w-1))
        glint_y = int(np.clip(cy-ry*0.62, 0, h-1))
        glint_rx, glint_ry = max(1, int(rx*0.13)), max(1, int(ry*0.09))
        glint_alpha = 0.80
    else:
        hi_rx, hi_ry = max(2, int(rx*0.74)), max(1, int(ry*0.27))
        hi_cx  = int(np.clip(cx-rx*0.06, 0, w-1))
        hi_cy  = int(np.clip(cy-ry*0.43, 0, h-1))
        sigma_x, sigma_y = rx*0.30, ry*0.12
        band_alpha = 0.14
        glint_x = int(np.clip(cx-rx*0.46, 0, w-1))
        glint_y = int(np.clip(cy-ry*0.52, 0, h-1))
        glint_rx, glint_ry = max(1, int(rx*0.088)), max(1, int(ry*0.055))
        glint_alpha = 0.65

    band = np.zeros((h, w), dtype=np.float32)
    cv2.ellipse(band, (hi_cx, hi_cy), (hi_rx, hi_ry), angle_deg, 0, 360, 1.0, -1, cv2.LINE_AA)
    if sigma_x < 0.5 or sigma_y < 0.5:
        return
    band = cv2.GaussianBlur(band, (0, 0), sigmaX=sigma_x, sigmaY=sigma_y)
    canvas[:] = np.clip(canvas.astype(np.float32)+band[:,:,np.newaxis]*band_alpha*255, 0, 255).astype(np.uint8)

    glint = np.zeros((h, w), dtype=np.float32)
    cv2.ellipse(glint, (glint_x, glint_y), (glint_rx, glint_ry),
                angle_deg-32, 0, 360, 1.0, -1, cv2.LINE_AA)
    glint = cv2.GaussianBlur(glint, (0, 0), 1.2)
    canvas[:] = np.clip(canvas.astype(np.float32)+glint[:,:,np.newaxis]*glint_alpha*255, 0, 255).astype(np.uint8)
    mask += np.clip(band+glint, 0, 1)


def _composite_clear_lenses(image: np.ndarray, lens_mask: np.ndarray) -> np.ndarray:
    h, w  = image.shape[:2]
    mask  = np.clip(lens_mask.astype(np.float32), 0.0, 1.0)
    if float(mask.max()) <= 0.0:
        return image.copy()
    mask_u8 = np.clip(mask*255.0, 0, 255).astype(np.uint8)
    soft    = cv2.GaussianBlur(mask, (0, 0), 1.6)
    dist    = cv2.distanceTransform(mask_u8, cv2.DIST_L2, 5)
    if float(dist.max()) > 0.0:
        dist = dist / float(dist.max())
    grad_x = cv2.Sobel(dist, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(dist, cv2.CV_32F, 0, 1, ksize=3)
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    map_x  = np.clip(xs + grad_x*3.2, 0, w-1).astype(np.float32)
    map_y  = np.clip(ys + grad_y*2.0, 0, h-1).astype(np.float32)
    refracted   = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
    refracted_f = refracted.astype(np.float32)
    tint        = np.full_like(refracted_f, (8.0, 10.0, 12.0), dtype=np.float32)
    transmitted = np.clip(refracted_f*1.035+tint, 0, 255)
    out   = image.astype(np.float32)
    alpha = (soft*0.34)[:,:,np.newaxis]
    out   = out*(1.0-alpha) + transmitted*alpha
    edge  = cv2.morphologyEx(mask_u8, cv2.MORPH_GRADIENT, np.ones((5,5), np.uint8)).astype(np.float32)/255.0
    edge  = cv2.GaussianBlur(edge, (0, 0), 1.0)
    upper = np.clip((-grad_y*2.5)+0.18, 0.0, 1.0)*edge
    lower = np.clip(( grad_y*2.2)+0.12, 0.0, 1.0)*edge
    out  += upper[:,:,np.newaxis]*18.0
    out  -= lower[:,:,np.newaxis]*10.0
    ar_alpha = (edge*mask*0.72)[:,:,np.newaxis]
    out     += ar_alpha * np.array([22.0, 9.0, 5.0], dtype=np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def _draw_glass_highlights(canvas, mask, cx, cy, rx, ry, angle_deg):
    h, w  = canvas.shape[:2]
    layer = canvas.copy()
    combined = np.zeros((h, w), dtype=np.float32)
    for ox, oy, al, sx, sy, arc_s, arc_e in [
        (-0.23, -0.34, 0.18, 0.42, 0.12, 190, 335),
        ( 0.20,  0.20, 0.10, 0.34, 0.10,  15, 145),
        (-0.02, -0.04, 0.07, 0.64, 0.34, 212, 256),
    ]:
        hx  = int(np.clip(cx+rx*ox, 0, w-1))
        hy  = int(np.clip(cy+ry*oy, 0, h-1))
        hrx = max(2, int(rx*sx)); hry = max(1, int(ry*sy))
        patch = np.zeros((h, w), dtype=np.float32)
        cv2.ellipse(layer, (hx, hy), (hrx, hry), angle_deg-18.0,
                    arc_s, arc_e, (255,255,255), max(1, int(rx*0.018)), cv2.LINE_AA)
        cv2.ellipse(patch, (hx, hy), (hrx, hry), angle_deg-18.0,
                    arc_s, arc_e, 1.0, max(1, int(rx*0.018)), cv2.LINE_AA)
        patch = cv2.GaussianBlur(patch, (0, 0), 0.8)
        a3 = (patch*al)[:,:,np.newaxis]
        canvas[:] = np.clip(canvas.astype(np.float32)*(1.0-a3)+layer.astype(np.float32)*a3,
                            0, 255).astype(np.uint8)
        combined += patch
        layer = canvas.copy()
    mask += np.clip(combined, 0, 1)
