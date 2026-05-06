"""Identity-preserving final blend for aging results."""

from __future__ import annotations

import cv2
import numpy as np


def _guided_filter_gray(guide: np.ndarray, src: np.ndarray, radius: int = 8, eps: float = 1e-3) -> np.ndarray:
    mean_i = cv2.boxFilter(guide, cv2.CV_32F, (radius, radius))
    mean_p = cv2.boxFilter(src, cv2.CV_32F, (radius, radius))
    corr_i = cv2.boxFilter(guide * guide, cv2.CV_32F, (radius, radius))
    corr_ip = cv2.boxFilter(guide * src, cv2.CV_32F, (radius, radius))
    var_i = corr_i - mean_i * mean_i
    cov_ip = corr_ip - mean_i * mean_p
    a = cov_ip / (var_i + eps)
    b = mean_p - a * mean_i
    mean_a = cv2.boxFilter(a, cv2.CV_32F, (radius, radius))
    mean_b = cv2.boxFilter(b, cv2.CV_32F, (radius, radius))
    return mean_a * guide + mean_b


def guided_identity_blend(
    original: np.ndarray,
    transformed: np.ndarray,
    masks: dict[str, np.ndarray],
    strength: float = 1.0,
) -> np.ndarray:
    """Constrain per-region changes while preserving guided-filtered detail.

    Args:
        original: Original BGR uint8 image.
        transformed: Fully transformed BGR uint8 image.
        masks: Region masks used by the effect pipeline.
        strength: Blend strength in ``[0, 1]``.

    Returns:
        Identity-preserved BGR uint8 image.
    """
    if strength <= 0.0:
        return transformed.copy()

    orig = original.astype(np.float32)
    trans = transformed.astype(np.float32)
    diff = trans - orig

    face = masks.get("face", np.ones(original.shape[:2], dtype=np.float32))
    hair = masks.get("hair", np.zeros(original.shape[:2], dtype=np.float32))
    lips = masks.get("lips", np.zeros(original.shape[:2], dtype=np.float32))
    eyes = np.clip(
        masks.get("under_eyes", 0.0) + masks.get("eye_corners", 0.0), 0.0, 1.0
    ).astype(np.float32)

    tolerance = np.full(original.shape[:2], 18.0, dtype=np.float32)
    tolerance += face * 20.0 + hair * 28.0 + lips * 18.0 + eyes * 14.0
    tolerance *= float(np.clip(strength, 0.35, 1.0))

    mag = np.linalg.norm(diff, axis=2)
    scale = np.minimum(1.0, tolerance / (mag + 1e-6))
    constrained = orig + diff * scale[:, :, np.newaxis]

    guide = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    filtered = np.empty_like(constrained, dtype=np.float32)
    for channel in range(3):
        src = constrained[:, :, channel].astype(np.float32) / 255.0
        filtered[:, :, channel] = _guided_filter_gray(guide, src, radius=9, eps=2e-3) * 255.0

    detail = orig - cv2.GaussianBlur(orig, (0, 0), 1.2)
    out = filtered + detail * (0.18 * face[:, :, np.newaxis])
    return np.nan_to_num(np.clip(out, 0, 255), copy=False).astype(np.uint8)
