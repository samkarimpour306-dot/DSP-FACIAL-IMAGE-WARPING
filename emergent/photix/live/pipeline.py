"""
pipeline.py — full stacked-effect pipeline for live and video modes.

Mirrors PhotoixApp.apply_processing in main.py so live frames go through the
*same* sequence of operations as static images:

    1. Expression warp     (smile / eyebrow / lip-wide / face-slim, Delaunay)
    2. Aging  /  De-Aging  (apply_aging / apply_deaging on landmarks + sigma)
    3. Frequency filters   (low-pass smoothing, high-pass sharpening)
    4. Face overlays       (sunglasses + beard + hair, all stackable)

A single ``LivePipeline`` dataclass holds every parameter; ``apply_pipeline``
evaluates the whole stack on one frame. The webcam / video / Gradio entry
points all share this code path.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, asdict, field, fields
from pathlib import Path
from typing import Any

import cv2
import numpy as np

_PHOTIX_ROOT = Path(__file__).resolve().parent.parent
if str(_PHOTIX_ROOT) not in sys.path:
    sys.path.insert(0, str(_PHOTIX_ROOT))

from core.aging_filter      import apply_aging, apply_deaging
from core.expression_warper import compute_combined_displacement
from core.mesh_warper       import apply_delaunay_warp
from core.filters           import (apply_lowpass_filter,
                                    apply_highpass_filter,
                                    FilterType)
from effects.overlays       import FaceOverlayEngine


# ── Data ─────────────────────────────────────────────────────────────────────

@dataclass
class LivePipeline:
    """Every parameter the live/video pipeline consumes."""

    # Expression warp
    smile:       float = 0.0
    eyebrow:     float = 0.0
    lip_wide:    float = 0.0
    face_slim:   float = 0.0

    # Aging  /  De-Aging
    aging:       str   = "none"     # "none" | "age" | "deage"
    aging_sigma: float = 50.0       # 1.0–100.0

    # Frequency-domain filters
    lowpass_enabled:  bool  = False
    lowpass_sigma:    float = 0.0   # 0.0–40.0
    highpass_enabled: bool  = False
    highpass_sigma:   float = 0.0   # 0.0–30.0

    # Face overlays — multiple can be active simultaneously.
    beard:       str = "none"       # "none" | "black" | "brown" | "blonde"
    sunglasses:  str = "none"       # "none" | "classic" | "aviator" | "round" | "glasses"
    hair_style:  str = "none"       # "none" | "long"
    hair_color:  str = "original"   # "original" | "black" | … | "gray"

    # ── helpers ──────────────────────────────────────────────────────────
    def has_warp(self) -> bool:
        return (self.smile + self.eyebrow + self.lip_wide + self.face_slim) > 0.01

    def has_aging(self) -> bool:
        return self.aging in ("age", "deage")

    def has_overlays(self) -> bool:
        return (self.beard != "none" or self.sunglasses != "none"
                or self.hair_style != "none")

    def is_empty(self) -> bool:
        return not (self.has_warp() or self.has_aging() or self.has_overlays()
                    or (self.lowpass_enabled  and self.lowpass_sigma  > 0.5)
                    or (self.highpass_enabled and self.highpass_sigma > 0.5))

    def label(self) -> str:
        """Compact one-line summary for the live HUD (ASCII-safe)."""
        parts: list[str] = []
        if self.aging == "age":   parts.append(f"Age s{int(self.aging_sigma)}")
        if self.aging == "deage": parts.append(f"DeAge s{int(self.aging_sigma)}")
        if self.smile     > 0.01: parts.append(f"Smile {self.smile:.2f}")
        if self.eyebrow   > 0.01: parts.append(f"Brow {self.eyebrow:.2f}")
        if self.lip_wide  > 0.01: parts.append(f"Lip {self.lip_wide:.2f}")
        if self.face_slim > 0.01: parts.append(f"Slim {self.face_slim:.2f}")
        if self.beard      != "none": parts.append(f"Beard:{self.beard}")
        if self.sunglasses != "none": parts.append(f"Glasses:{self.sunglasses}")
        if self.hair_style != "none": parts.append(f"Hair:{self.hair_style}")
        if self.lowpass_enabled  and self.lowpass_sigma  > 0.5: parts.append("LP")
        if self.highpass_enabled and self.highpass_sigma > 0.5: parts.append("HP")
        return " | ".join(parts) if parts else "No Effect"

    # ── (de)serialization for subprocess / Gradio handoff ────────────────
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LivePipeline":
        valid = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in valid})

    @classmethod
    def from_json(cls, raw: str | bytes) -> "LivePipeline":
        return cls.from_dict(json.loads(raw))


# ── Per-frame evaluator ──────────────────────────────────────────────────────

def apply_pipeline(
    frame_bgr: np.ndarray,
    pipeline:  LivePipeline,
    landmarks: np.ndarray | None,
) -> np.ndarray:
    """Apply every enabled stage of ``pipeline`` to one BGR frame.

    Returns a uint8 BGR frame the same shape as the input. Falls back to a
    passthrough on any failure so the live loop never crashes on a single
    bad frame.
    """
    if frame_bgr is None or frame_bgr.size == 0:
        return frame_bgr
    if pipeline.is_empty():
        return frame_bgr

    img = frame_bgr

    try:
        # ── 1. Expression warp (Delaunay only; TPS/flow too slow for live) ──
        if landmarks is not None and pipeline.has_warp():
            src, dst = compute_combined_displacement(
                landmarks,
                smile=pipeline.smile,
                eyebrow=pipeline.eyebrow,
                lip_wide=pipeline.lip_wide,
                face_slim=pipeline.face_slim,
            )
            img = apply_delaunay_warp(img, src, dst)

        # ── 2. Aging / De-Aging ─────────────────────────────────────────
        if landmarks is not None and pipeline.aging == "age":
            img = apply_aging(img, sigma=pipeline.aging_sigma,
                              landmarks=landmarks)
        elif landmarks is not None and pipeline.aging == "deage":
            img = apply_deaging(img, sigma=pipeline.aging_sigma,
                                landmarks=landmarks)

        # ── 3. Frequency-domain filters ─────────────────────────────────
        if pipeline.lowpass_enabled and pipeline.lowpass_sigma > 0.5:
            img = apply_lowpass_filter(
                img, sigma=float(pipeline.lowpass_sigma),
                filter_type=FilterType.GAUSSIAN,
            )

        if pipeline.highpass_enabled and pipeline.highpass_sigma > 0.5:
            # Gaussian high-pass filter (FFT domain) — see core.filters.
            # Output is a normalized edge map: flat regions black, edges
            # bright.
            img = apply_highpass_filter(
                img, sigma=float(pipeline.highpass_sigma),
                filter_type=FilterType.GAUSSIAN,
            )

        # ── 4. Face overlays — all enabled overlays stack in one engine call ─
        if landmarks is not None and pipeline.has_overlays():
            effects_list: list[str] = []
            params: dict[str, dict[str, str]] = {}
            if pipeline.hair_style != "none":
                effects_list.append("hair")
                params["hair"] = {
                    "style": pipeline.hair_style,
                    "color": pipeline.hair_color,
                }
            if pipeline.sunglasses != "none":
                effects_list.append("sunglasses")
                params["sunglasses"] = {"style": pipeline.sunglasses}
            if pipeline.beard != "none":
                effects_list.append("beard")
                params["beard"] = {"color": pipeline.beard}
            res = FaceOverlayEngine(landmarks, img).apply(effects_list, params)
            img = res.final_image
    except Exception:
        return frame_bgr

    return np.clip(img, 0, 255).astype(np.uint8)
