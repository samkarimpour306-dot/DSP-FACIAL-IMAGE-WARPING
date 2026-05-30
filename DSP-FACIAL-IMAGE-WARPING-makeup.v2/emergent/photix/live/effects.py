"""
effects.py — shared single-effect dispatcher for live/video modes.

All effects reuse the existing static-image implementations under photix/core
and photix/effects so live behaviour matches the desktop app exactly.

Effect ids:
    "none"        passthrough
    "age"         apply_aging         (sigma = intensity * 100)
    "deage"       apply_deaging       (sigma = intensity * 100)
    "smile"       expression warp     (smile slider = intensity)
    "eyebrow"     expression warp     (eyebrow slider = intensity)
    "lip_wide"    expression warp     (lip_wide slider = intensity)
    "face_slim"   expression warp     (face_slim slider = intensity)
    "surprise"    expression warp     (eyebrow + lip_wide = intensity)
    "beard"       overlay engine      (color via kwargs)
    "sunglasses"  overlay engine      (style via kwargs)
    "hair"        overlay engine      (style + color via kwargs)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_PHOTIX_ROOT = Path(__file__).resolve().parent.parent
if str(_PHOTIX_ROOT) not in sys.path:
    sys.path.insert(0, str(_PHOTIX_ROOT))

from core.aging_filter      import apply_aging, apply_deaging
from core.expression_warper import compute_combined_displacement
from core.mesh_warper       import apply_delaunay_warp
from effects.overlays       import FaceOverlayEngine


def apply_effect(
    frame_bgr: np.ndarray,
    effect: str,
    intensity: float,
    landmarks: np.ndarray | None,
    *,
    beard_color: str = "black",
    sunglasses_style: str = "classic",
    hair_style: str = "long",
    hair_color: str = "original",
) -> np.ndarray:
    """Apply one of the supported effects to a single BGR frame.

    Returns a uint8 BGR frame the same shape as the input. If the effect
    cannot run (no landmarks, zero intensity, unknown id, or an internal
    error) the input is returned unchanged. The live loop must never crash
    on a single bad frame.
    """
    if frame_bgr is None or frame_bgr.size == 0:
        return frame_bgr

    intensity = float(np.clip(intensity, 0.0, 1.0))
    if effect == "none":
        return frame_bgr

    # Every effect we support requires landmarks.
    if landmarks is None:
        return frame_bgr

    # The aging / warp effects honour intensity; the overlay effects don't
    # have a meaningful "intensity" slider — they're on/off positioned via
    # landmarks. So we only gate on intensity for the warp/aging family.
    if effect in (
        "age", "deage", "smile", "eyebrow", "lip_wide", "face_slim",
        "surprise",
    ) and intensity < 1e-3:
        return frame_bgr

    try:
        if effect == "age":
            return apply_aging(frame_bgr, sigma=intensity * 100.0,
                               landmarks=landmarks)

        if effect == "deage":
            return apply_deaging(frame_bgr, sigma=intensity * 100.0,
                                 landmarks=landmarks)

        if effect == "smile":
            src, dst = compute_combined_displacement(landmarks, smile=intensity)
            return apply_delaunay_warp(frame_bgr, src, dst)

        if effect == "eyebrow":
            src, dst = compute_combined_displacement(landmarks, eyebrow=intensity)
            return apply_delaunay_warp(frame_bgr, src, dst)

        if effect == "lip_wide":
            src, dst = compute_combined_displacement(landmarks, lip_wide=intensity)
            return apply_delaunay_warp(frame_bgr, src, dst)

        if effect == "face_slim":
            src, dst = compute_combined_displacement(landmarks, face_slim=intensity)
            return apply_delaunay_warp(frame_bgr, src, dst)

        if effect == "surprise":
            src, dst = compute_combined_displacement(
                landmarks, eyebrow=intensity, lip_wide=intensity * 0.6,
            )
            return apply_delaunay_warp(frame_bgr, src, dst)

        if effect == "beard":
            res = FaceOverlayEngine(landmarks, frame_bgr).apply(
                ["beard"], {"beard": {"color": beard_color}},
            )
            return res.final_image

        if effect == "sunglasses":
            res = FaceOverlayEngine(landmarks, frame_bgr).apply(
                ["sunglasses"], {"sunglasses": {"style": sunglasses_style}},
            )
            return res.final_image

        if effect == "hair":
            res = FaceOverlayEngine(landmarks, frame_bgr).apply(
                ["hair"],
                {"hair": {"style": hair_style, "color": hair_color}},
            )
            return res.final_image
    except Exception:
        return frame_bgr

    return frame_bgr
