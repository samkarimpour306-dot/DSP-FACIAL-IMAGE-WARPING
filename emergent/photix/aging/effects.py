"""Composable classical-CV aging effects."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from functools import lru_cache

import cv2
import numpy as np

from . import masks as masklib
from landmarks.detector import FaceLandmarks

try:
    from core.flow_warper import apply_optical_flow_warp
except Exception:  # pragma: no cover - import fallback for package execution
    apply_optical_flow_warp = None

log = logging.getLogger(__name__)


class BaseAgingEffect(ABC):
    """Base class for a separately controllable aging effect."""

    name: str

    @abstractmethod
    def apply(
        self,
        image: np.ndarray,
        landmarks: FaceLandmarks,
        mask: np.ndarray,
        intensity: float,
    ) -> np.ndarray:
        """Apply the effect at ``intensity`` in ``[0, 1]``."""


def _blend(base: np.ndarray, effect: np.ndarray, mask: np.ndarray, amount: float) -> np.ndarray:
    alpha = np.clip(mask * amount, 0.0, 1.0).astype(np.float32)[:, :, np.newaxis]
    out = effect.astype(np.float32) * alpha + base.astype(np.float32) * (1.0 - alpha)
    return np.nan_to_num(np.clip(out, 0, 255), copy=False).astype(np.uint8)


def _safe_intensity(intensity: float) -> float:
    return float(np.clip(intensity, 0.0, 1.0))


@lru_cache(maxsize=24)
def _fractal_noise(height: int, width: int, seed: int, octaves: int = 4) -> np.ndarray:
    rng = np.random.default_rng(seed + height * 928371 + width * 1237)
    total = np.zeros((height, width), dtype=np.float32)
    amp = 1.0
    amp_sum = 0.0
    for octave in range(octaves):
        scale = 2 ** octave
        small_h = max(2, height // (18 // min(scale, 8) + 1))
        small_w = max(2, width // (18 // min(scale, 8) + 1))
        noise = rng.random((small_h, small_w), dtype=np.float32)
        noise = cv2.resize(noise, (width, height), interpolation=cv2.INTER_CUBIC)
        total += noise * amp
        amp_sum += amp
        amp *= 0.52
    total /= max(amp_sum, 1e-6)
    return cv2.GaussianBlur(total, (0, 0), 0.7).astype(np.float32)


@lru_cache(maxsize=24)
def _direction_map(height: int, width: int, region: str) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    if region == "forehead":
        coord = yy
    elif region == "eyes":
        coord = xx * 0.45 + yy * 0.95
    elif region == "nasolabial":
        coord = xx * 0.55 - yy * 0.95
    else:
        coord = yy * 0.75
    waves = np.maximum(0.0, np.sin(coord * 0.17))
    return cv2.GaussianBlur(waves, (0, 0), 0.9).astype(np.float32)


class WrinkleSynthesis(BaseAgingEffect):
    """Procedurally synthesize forehead, crow's-feet, fold, and mouth wrinkles."""

    name = "wrinkles"

    def __init__(self, seed: int | None = None) -> None:
        self.seed = 0 if seed is None else int(seed)

    def apply(
        self,
        image: np.ndarray,
        landmarks: FaceLandmarks,
        mask: np.ndarray,
        intensity: float,
    ) -> np.ndarray:
        """Apply multiply-style procedural wrinkles inside anatomical masks."""
        amount = _safe_intensity(intensity)
        if amount == 0.0:
            return image.copy()
        height, width = image.shape[:2]
        wrinkle = np.zeros((height, width), dtype=np.float32)
        regions = {
            "forehead": masklib.forehead_mask((height, width), landmarks),
            "eyes": masklib.eye_corner_mask((height, width), landmarks),
            "nasolabial": masklib.nasolabial_mask((height, width), landmarks),
            "mouth": masklib.lip_mask((height, width), landmarks),
        }
        for offset, (name, region_mask) in enumerate(regions.items()):
            direction = _direction_map(height, width, name)
            noise = _fractal_noise(height, width, self.seed + offset * 113)
            wrinkle += np.maximum(0.0, direction * 0.72 + noise * 0.38 - 0.48) * region_mask

        wrinkle = cv2.GaussianBlur(wrinkle, (0, 0), 0.55)
        darkened = image.astype(np.float32) * (1.0 - wrinkle[:, :, np.newaxis] * 0.36 * amount)
        highlight = cv2.GaussianBlur(wrinkle, (0, 0), 2.0)[:, :, np.newaxis] * 18.0 * amount
        return _blend(image, np.clip(darkened + highlight, 0, 255).astype(np.uint8), mask, 1.0)


class SkinTextureRoughening(BaseAgingEffect):
    """Add pores and high-frequency skin roughness."""

    name = "roughening"

    def __init__(self, seed: int | None = None) -> None:
        self.seed = 41 if seed is None else int(seed) + 41

    def apply(
        self,
        image: np.ndarray,
        landmarks: FaceLandmarks,
        mask: np.ndarray,
        intensity: float,
    ) -> np.ndarray:
        """Enhance pore texture with unsharp detail and deterministic noise."""
        amount = _safe_intensity(intensity)
        if amount == 0.0:
            return image.copy()
        height, width = image.shape[:2]
        blur = cv2.GaussianBlur(image, (0, 0), 1.0)
        sharp = cv2.addWeighted(image, 1.0 + 0.55 * amount, blur, -0.55 * amount, 0)
        noise = (_fractal_noise(height, width, self.seed, 5) - 0.5) * 34.0 * amount
        rough = sharp.astype(np.float32) + noise[:, :, np.newaxis] * mask[:, :, np.newaxis]
        return _blend(image, np.clip(rough, 0, 255).astype(np.uint8), mask, 0.88)


class SkinSmoothing(BaseAgingEffect):
    """Smooth skin for de-aging while preserving strong landmark edges."""

    name = "smoothing"

    def apply(
        self,
        image: np.ndarray,
        landmarks: FaceLandmarks,
        mask: np.ndarray,
        intensity: float,
    ) -> np.ndarray:
        """Apply OpenCV bilateral and edge-preserving filters."""
        amount = _safe_intensity(intensity)
        if amount == 0.0:
            return image.copy()
        smooth = cv2.bilateralFilter(image, 11, 54.0 + 40.0 * amount, 14.0 + 10.0 * amount)
        try:
            smooth = cv2.edgePreservingFilter(smooth, flags=1, sigma_s=42, sigma_r=0.18)
        except cv2.error:
            log.debug("edgePreservingFilter unavailable; bilateral result used.")
        edges = np.zeros(mask.shape, dtype=np.float32)
        for pts in (landmarks.left_eye, landmarks.right_eye, landmarks.lips, landmarks.nose):
            if len(pts) >= 3:
                edges = np.maximum(edges, masklib._soft_polygon(mask.shape, pts, 3.0))
        skin_mask = np.clip(mask * (1.0 - edges * 0.75), 0.0, 1.0)
        lab = cv2.cvtColor(smooth, cv2.COLOR_BGR2LAB).astype(np.float32)
        l_chan, a_chan, b_chan = cv2.split(lab)
        local_l = cv2.bilateralFilter(l_chan, 7, 22.0 + 22.0 * amount, 14.0)
        local_a = cv2.GaussianBlur(a_chan, (0, 0), 2.8 + 2.0 * amount)
        local_b = cv2.GaussianBlur(b_chan, (0, 0), 2.8 + 2.0 * amount)
        clear_lab = cv2.merge([
            np.clip(local_l * (1.0 + 0.016 * amount) + 1.4 * amount, 0, 255),
            np.clip(a_chan * (1.0 - 0.18 * amount) + local_a * (0.18 * amount), 0, 255),
            np.clip(b_chan * (1.0 - 0.13 * amount) + local_b * (0.13 * amount), 0, 255),
        ])
        clear = cv2.cvtColor(clear_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        low = cv2.GaussianBlur(gray, (0, 0), 5.0)
        spots = np.clip((low - gray - 3.0) / 22.0, 0.0, 1.0) * skin_mask
        clear = np.clip(clear.astype(np.float32) + spots[:, :, np.newaxis] * 22.0 * amount, 0, 255)

        gray_u8 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 15))
        thin_lines = np.maximum(
            cv2.morphologyEx(gray_u8, cv2.MORPH_BLACKHAT, kernel_h),
            cv2.morphologyEx(gray_u8, cv2.MORPH_BLACKHAT, kernel_v),
        ).astype(np.float32) / 255.0
        line_lift = cv2.GaussianBlur(thin_lines * skin_mask, (0, 0), 0.8)
        clear = np.clip(clear + line_lift[:, :, np.newaxis] * 24.0 * amount, 0, 255)

        detail = image.astype(np.float32) - cv2.GaussianBlur(image, (0, 0), 1.0).astype(np.float32)
        clear = clear + detail * mask[:, :, np.newaxis] * (0.20 + 0.08 * (1.0 - amount))
        return _blend(image, np.clip(clear, 0, 255).astype(np.uint8), skin_mask, 0.78 * amount)


class SkinToneShift(BaseAgingEffect):
    """Shift skin tone in LAB space for aging or de-aging."""

    name = "skin_tone"

    def __init__(self, direction: int) -> None:
        self.direction = 1 if direction >= 0 else -1

    def apply(
        self,
        image: np.ndarray,
        landmarks: FaceLandmarks,
        mask: np.ndarray,
        intensity: float,
    ) -> np.ndarray:
        """Apply LAB color shifts; LAB keeps luminance separate from color axes."""
        amount = _safe_intensity(intensity)
        if amount == 0.0:
            return image.copy()
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
        if self.direction > 0:
            lab[:, :, 0] = lab[:, :, 0] * (1.0 - 0.08 * amount)
            lab[:, :, 1] += 3.0 * amount
            lab[:, :, 2] += 8.0 * amount
        else:
            lab[:, :, 0] = lab[:, :, 0] * (1.0 + 0.070 * amount) + 3.5 * amount
            lab[:, :, 1] += 3.0 * amount
            lab[:, :, 2] += 1.2 * amount
        shifted = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
        return _blend(image, shifted, mask, 0.95)


class HairColorShift(BaseAgingEffect):
    """Gray hair for aging and enrich/darken it for de-aging."""

    name = "hair"

    def __init__(self, direction: int) -> None:
        self.direction = 1 if direction >= 0 else -1

    def apply(
        self,
        image: np.ndarray,
        landmarks: FaceLandmarks,
        mask: np.ndarray,
        intensity: float,
    ) -> np.ndarray:
        """Apply hair color shifts, skipping gracefully when mask is empty."""
        amount = _safe_intensity(intensity)
        if amount == 0.0 or float(mask.max()) < 0.03:
            return image.copy()
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        if self.direction > 0:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
            silver = np.stack([gray * 1.10 + 45.0] * 3, axis=2)
            hsv[:, :, 1] *= 1.0 - 0.70 * amount
            muted = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
            target = np.clip(0.55 * silver + 0.45 * muted.astype(np.float32), 0, 235).astype(np.uint8)
        else:
            hsv[:, :, 1] *= 1.0 + 0.45 * amount
            hsv[:, :, 2] *= 1.0 - 0.22 * amount
            target = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
        return _blend(image, target, mask, amount)


class EyeAreaDarkening(BaseAgingEffect):
    """Darken or brighten the under-eye region."""

    name = "eyes"

    def __init__(self, direction: int) -> None:
        self.direction = 1 if direction >= 0 else -1

    def apply(
        self,
        image: np.ndarray,
        landmarks: FaceLandmarks,
        mask: np.ndarray,
        intensity: float,
    ) -> np.ndarray:
        """Apply subtle eye-area luminance and chroma changes."""
        amount = _safe_intensity(intensity)
        if amount == 0.0:
            return image.copy()
        out = image.astype(np.float32)
        if self.direction > 0:
            out[:, :, 0] += 8.0 * amount
            out[:, :, 1] -= 10.0 * amount
            out[:, :, 2] -= 16.0 * amount
        else:
            out[:, :, 0] += 10.0 * amount
            out[:, :, 1] += 24.0 * amount
            out[:, :, 2] += 31.0 * amount
        return _blend(image, np.clip(out, 0, 255).astype(np.uint8), mask, 0.85)


class LipDesaturation(BaseAgingEffect):
    """Thin/desaturate lips for aging; subtly plump and enrich for de-aging."""

    name = "lips"

    def __init__(self, direction: int) -> None:
        self.direction = 1 if direction >= 0 else -1

    def apply(
        self,
        image: np.ndarray,
        landmarks: FaceLandmarks,
        mask: np.ndarray,
        intensity: float,
    ) -> np.ndarray:
        """Apply lip color and subtle local scale changes."""
        amount = _safe_intensity(intensity)
        if amount == 0.0:
            return image.copy()
        base = image
        if self.direction < 0 and len(landmarks.lips) >= 3:
            pts = landmarks.lips
            cx, cy = float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1]))
            rx = max(8.0, float(np.ptp(pts[:, 0])) * 0.55)
            ry = max(4.0, float(np.ptp(pts[:, 1])) * 0.95)
            base = _local_scale(image, (cx, cy), rx, ry, 1.0 + 0.08 * amount, 1.0 + 0.06 * amount, mask)

        hsv = cv2.cvtColor(base, cv2.COLOR_BGR2HSV).astype(np.float32)
        if self.direction > 0:
            hsv[:, :, 1] *= 1.0 - 0.45 * amount
            hsv[:, :, 2] *= 1.0 - 0.08 * amount
        else:
            hsv[:, :, 1] *= 1.0 + 0.24 * amount
            hsv[:, :, 2] *= 1.0 + 0.04 * amount
        target = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
        return _blend(image, target, mask, 0.85)


def _local_scale(
    image: np.ndarray,
    center: tuple[float, float],
    radius_x: float,
    radius_y: float,
    scale_x: float,
    scale_y: float,
    mask: np.ndarray,
) -> np.ndarray:
    height, width = image.shape[:2]
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    cx, cy = center
    map_x = cx + (xx - cx) / max(scale_x, 0.5)
    map_y = cy + (yy - cy) / max(scale_y, 0.5)
    warped = cv2.remap(
        image,
        np.clip(map_x, 0, width - 1).astype(np.float32),
        np.clip(map_y, 0, height - 1).astype(np.float32),
        cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    oval = masklib._soft_ellipse((height, width), center, radius_x, radius_y, 6.0)
    return _blend(image, warped, np.minimum(mask, oval), 1.0)


class GeometricSagging(BaseAgingEffect):
    """Subtle landmark-flow warp for sagging or lifted de-aging."""

    name = "sagging"

    def __init__(self, direction: int) -> None:
        self.direction = 1 if direction >= 0 else -1
        self.last_warp_field: np.ndarray | None = None

    def apply(
        self,
        image: np.ndarray,
        landmarks: FaceLandmarks,
        mask: np.ndarray,
        intensity: float,
    ) -> np.ndarray:
        """Reuse Photix landmark-flow warping instead of implementing a new warp."""
        amount = _safe_intensity(intensity)
        self.last_warp_field = np.zeros((*image.shape[:2], 2), dtype=np.float32)
        if amount == 0.0 or apply_optical_flow_warp is None:
            return image.copy()

        src = landmarks.pixels[:, :2].astype(np.float32)
        dst = src.copy()
        height = image.shape[0]
        pixels = landmarks.pixels
        face_h = max(24.0, float(landmarks.bbox[3] or height * 0.5))
        dy = face_h * (0.028 + 0.035 * amount) * (1.0 if self.direction > 0 else -1.0)

        jaw_candidates = set(range(len(pixels)))
        for pts in (landmarks.jaw, landmarks.left_eyebrow, landmarks.right_eyebrow):
            for point in pts:
                d = np.linalg.norm(pixels - point, axis=1)
                idx = int(np.argmin(d))
                if idx in jaw_candidates:
                    dst[idx, 1] = np.clip(dst[idx, 1] + dy, 0, height - 1)

        try:
            warped = apply_optical_flow_warp(image, src, dst, intensity=amount)
        except MemoryError:
            log.warning("GeometricSagging skipped because optical-flow allocation failed.")
            return image.copy()
        self.last_warp_field = (src - dst).reshape(-1, 2).mean(axis=0).astype(np.float32)
        field = np.zeros((*image.shape[:2], 2), dtype=np.float32)
        field[:, :, 1] = float(self.last_warp_field[1]) * mask
        self.last_warp_field = field
        return _blend(image, warped, mask, 0.95)
