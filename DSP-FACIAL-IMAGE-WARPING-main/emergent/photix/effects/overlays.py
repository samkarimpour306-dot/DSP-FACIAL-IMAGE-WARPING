"""
effects/overlays.py
Face overlay effects — drawn with OpenCV primitives or composited from PNG assets.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from landmarks.detector import FaceLandmarks

# Default PNG asset path (place sunglasses.png here to use the PNG mode)
_ASSETS_DIR = Path(__file__).parent / "assets"
_DEFAULT_PNG = _ASSETS_DIR / "sunglasses.png"


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class OverlayResult:
    """Output produced by FaceOverlayEngine.apply().

    Attributes:
        final_image:         BGR uint8 composite with all effects applied.
        intermediate_stages: One BGR image per effect, in application order.
        masks_used:          Float32 mask per named region actually touched.
        dsp_metadata:        Blend alphas, kernel sizes, color space used —
                             suitable for inclusion in a DSP report.
    """
    final_image: np.ndarray
    intermediate_stages: list[np.ndarray] = field(default_factory=list)
    masks_used: dict[str, np.ndarray] = field(default_factory=dict)
    dsp_metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class FaceOverlayEngine:
    """Applies declarative overlay effects to a face image.

    Args:
        landmarks: Detected face landmarks (pixels in image space).
        image:     Source BGR uint8 image.

    Example::

        engine = FaceOverlayEngine(landmarks, frame)
        result = engine.apply(["sunglasses"], {"sunglasses": {"style": "aviator"}})
    """

    _EFFECT_REGISTRY: dict[str, str] = {
        "sunglasses":     "_apply_sunglasses",
        "sunglasses_png": "_apply_sunglasses_png",
        "beard":          "_apply_beard",
        "beard_png":      "_apply_beard_png",
    }

    def __init__(self, landmarks, image: np.ndarray) -> None:
        # Accept either a FaceLandmarks dataclass or a raw (N, 2) ndarray
        if isinstance(landmarks, np.ndarray):
            self._pixels = landmarks
        else:
            self._pixels = landmarks.pixels
        self.image = image

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def apply(self, effects: list[str], params: dict) -> OverlayResult:
        """Apply a sequence of named effects and return an OverlayResult.

        Args:
            effects: Ordered list of effect names to apply.
            params:  Dict keyed by effect name; each value is a dict of
                     effect-specific keyword arguments.

        Returns:
            Populated :class:`OverlayResult`.
        """
        current = self.image.copy()
        stages: list[np.ndarray] = []
        all_masks: dict[str, np.ndarray] = {}
        all_meta: dict[str, Any] = {}

        for name in effects:
            method_name = self._EFFECT_REGISTRY.get(name)
            if method_name is None:
                raise ValueError(
                    f"Unknown effect {name!r}. "
                    f"Available: {list(self._EFFECT_REGISTRY)}"
                )
            method = getattr(self, method_name)
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
    # Effect: Sunglasses
    # ------------------------------------------------------------------

    def _apply_sunglasses(
        self,
        image: np.ndarray,
        style: str = "classic",
        offset_x: float = 0.0,
        offset_y: float = 0.0,
        scale: float = 1.0,
        rotation: float = 0.0,
    ) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, Any]]:
        """Render sunglasses over the eye region using OpenCV primitives.

        Args:
            image: BGR uint8 source (may already have prior effects applied).
            style: One of "classic", "aviator", "round".

        Returns:
            ``(composited_image, masks_used, dsp_metadata)``
        """
        _VALID = {"classic", "aviator", "round", "glasses"}
        if style not in _VALID:
            raise ValueError(f"style must be one of {_VALID}, got {style!r}")

        is_clear = style == "glasses"

        pts = self._pixels
        h, w = image.shape[:2]

        # --- key landmark positions ------------------------------------
        l_inner  = _pt(pts, 133)   # left eye inner corner
        l_outer  = _pt(pts, 33)    # left eye outer corner
        r_inner  = _pt(pts, 362)   # right eye inner corner
        r_outer  = _pt(pts, 263)   # right eye outer corner
        l_brow   = _pt(pts, 105)   # left brow top
        r_brow   = _pt(pts, 334)   # right brow top

        # eye centres and radii
        l_cx = (l_inner[0] + l_outer[0]) / 2.0
        l_cy = (l_inner[1] + l_outer[1]) / 2.0
        r_cx = (r_inner[0] + r_outer[0]) / 2.0
        r_cy = (r_inner[1] + r_outer[1]) / 2.0

        eye_span   = math.dist(l_outer, r_outer)          # full inter-eye span
        scale      = float(np.clip(scale, 0.25, 3.0))
        lens_rx    = eye_span * _STYLE_LENS_X[style] * scale
        lens_ry    = lens_rx  * _STYLE_LENS_RATIO[style]

        # inter-eye angle (roll compensation)
        angle_rad  = math.atan2(r_cy - l_cy, r_cx - l_cx)
        angle_deg  = math.degrees(angle_rad) + float(rotation)
        angle_rad  = math.radians(angle_deg)

        # vertical offset: sit glasses just above the eye centre line
        brow_y     = (l_brow[1] + r_brow[1]) / 2.0
        eye_y      = (l_cy + r_cy) / 2.0
        v_offset   = (eye_y - brow_y) * 0.25   # shift up 25 % toward brow

        l_cx -= v_offset * math.sin(angle_rad)
        l_cy -= v_offset * math.cos(angle_rad)
        r_cx -= v_offset * math.sin(angle_rad)
        r_cy -= v_offset * math.cos(angle_rad)

        l_cx += float(offset_x)
        r_cx += float(offset_x)
        l_cy += float(offset_y)
        r_cy += float(offset_y)

        # bridge midpoint
        mid_x = (l_cx + r_cx) / 2.0
        mid_y = (l_cy + r_cy) / 2.0

        # temple end points (extend outward from each outer-eye corner)
        temple_len = eye_span * 0.52 * scale
        dx = math.cos(angle_rad)
        dy = math.sin(angle_rad)
        l_temple = (int(l_cx - dx * temple_len), int(l_cy - dy * temple_len))
        r_temple = (int(r_cx + dx * temple_len), int(r_cy + dy * temple_len))

        # Aviator temples hinge at the outer ends of the brow bar, not at lens centre
        if style == "aviator":
            _ux = -math.sin(angle_rad)
            _uy = -math.cos(angle_rad)
            _bh = lens_ry * 0.93
            l_ts = (int(l_cx - dx * lens_rx * 1.05 + _ux * _bh),
                    int(l_cy - dy * lens_rx * 1.05 + _uy * _bh))
            r_ts = (int(r_cx + dx * lens_rx * 1.05 + _ux * _bh),
                    int(r_cy + dy * lens_rx * 1.05 + _uy * _bh))
        else:
            l_ts = (int(l_cx), int(l_cy))
            r_ts = (int(r_cx), int(r_cy))

        # --- build overlay canvas --------------------------------------
        overlay = image.copy()
        # Clear glasses use this fill only to create the lens mask. The final
        # lens look is built from a subtle refraction composite below.
        lens_color  = _STYLE_LENS_COLOR[style]
        frame_color = _STYLE_FRAME_COLOR[style]
        frame_thick = max(3, int(eye_span * 0.018 * scale))
        if style == "aviator":
            frame_thick = max(1, int(frame_thick * 0.55))  # thin metal wire

        # lens alpha mask (float32, for region reporting)
        lens_mask = np.zeros((h, w), dtype=np.float32)

        for cx, cy in ((l_cx, l_cy), (r_cx, r_cy)):
            _draw_lens(overlay, lens_mask, cx, cy, lens_rx, lens_ry,
                       angle_deg, style, lens_color, frame_color, frame_thick)

        # bridge / nose piece
        _draw_bridge(overlay, l_cx, l_cy, r_cx, r_cy,
                     lens_rx, lens_ry, angle_deg, frame_color, frame_thick)

        # aviator brow bar (top horizontal bar spanning both lenses)
        if style == "aviator":
            _draw_aviator_brow_bar(overlay, l_cx, l_cy, r_cx, r_cy,
                                    lens_rx, lens_ry, angle_rad, frame_color, frame_thick)

        # nose pads — small metallic ovals where bridge meets each lens
        pad_size = (max(3, int(lens_rx * 0.08)), max(2, int(lens_ry * 0.07)))
        for cx, cy, sign in ((l_cx, l_cy, 1), (r_cx, r_cy, -1)):
            px = int(cx + sign * lens_rx * 0.72 * math.cos(angle_rad))
            py = int(cy + sign * lens_rx * 0.72 * math.sin(angle_rad) + lens_ry * 0.22)
            if is_clear:
                cv2.ellipse(overlay, (px, py), pad_size, angle_deg, 0, 360,
                            (238, 236, 226), -1, cv2.LINE_AA)
                cv2.ellipse(overlay, (px, py), pad_size, angle_deg, 0, 360,
                            (95, 92, 86), 1, cv2.LINE_AA)
            else:
                cv2.ellipse(overlay, (px, py), pad_size, angle_deg, 0, 360,
                            (110, 108, 100), -1, cv2.LINE_AA)
                cv2.ellipse(overlay, (px, py), pad_size, angle_deg, 0, 360,
                            (20, 20, 20), 1, cv2.LINE_AA)

        # temple arms — taper from frame thickness to thin at ear end
        for start, end in ((l_ts, l_temple), (r_ts, r_temple)):
            _draw_tapered_temple(overlay, start, end, frame_color, frame_thick)

        # reflection highlights — skip for clear glasses
        refl_mask = np.zeros((h, w), dtype=np.float32)
        if not is_clear:
            for cx, cy in ((l_cx, l_cy), (r_cx, r_cy)):
                _draw_specular(overlay, refl_mask, cx, cy, lens_rx, lens_ry, angle_deg, style)

        # Clear glasses refract and lightly brighten the source instead of
        # laying a flat translucent color over the face.
        LENS_ALPHA = 0.28 if is_clear else 0.76
        if is_clear:
            out = _composite_clear_lenses(image, lens_mask)
        else:
            out = cv2.addWeighted(overlay, LENS_ALPHA, image, 1.0 - LENS_ALPHA, 0)

        # Redraw the frame at full opacity so the frame is crisp while the
        # lenses stay softly composited with the face underneath.
        redraw_frame = True
        if redraw_frame:
            for cx, cy in ((l_cx, l_cy), (r_cx, r_cy)):
                _draw_frame_outline(out, cx, cy, lens_rx, lens_ry,
                                    angle_deg, frame_color, frame_thick, style)
            _draw_bridge(out, l_cx, l_cy, r_cx, r_cy,
                         lens_rx, lens_ry, angle_deg, frame_color, frame_thick)
            if style == "aviator":
                _draw_aviator_brow_bar(out, l_cx, l_cy, r_cx, r_cy,
                                        lens_rx, lens_ry, angle_rad, frame_color, frame_thick)
            _draw_tapered_temple(out, l_ts, l_temple, frame_color, frame_thick)
            _draw_tapered_temple(out, r_ts, r_temple, frame_color, frame_thick)
            if is_clear:
                for cx, cy in ((l_cx, l_cy), (r_cx, r_cy)):
                    _draw_glass_highlights(out, refl_mask, cx, cy, lens_rx, lens_ry, angle_deg)
                for cx, cy, sign in ((l_cx, l_cy, 1), (r_cx, r_cy, -1)):
                    px = int(cx + sign * lens_rx * 0.72 * math.cos(angle_rad))
                    py = int(cy + sign * lens_rx * 0.72 * math.sin(angle_rad) + lens_ry * 0.22)
                    cv2.ellipse(out, (px, py), pad_size, angle_deg, 0, 360,
                                (228, 226, 216), -1, cv2.LINE_AA)
                    cv2.ellipse(out, (px, py), pad_size, angle_deg, 0, 360,
                                (82, 78, 72), 1, cv2.LINE_AA)

        masks_used = {
            "sunglasses_lens":       lens_mask,
            "sunglasses_reflection": refl_mask,
        }

        dsp_meta = {
            "blend_alpha":     LENS_ALPHA,
            "specular_alpha":  0.18 if is_clear else 0.11,
            "kernel_sizes":    [],
            "color_space":     "BGR",
            "lens_style":      style,
            "inter_eye_angle": round(angle_deg, 2),
            "lens_rx_px":      round(lens_rx, 1),
            "lens_ry_px":      round(lens_ry, 1),
            "offset_x":        round(float(offset_x), 1),
            "offset_y":        round(float(offset_y), 1),
            "scale":           round(scale, 2),
            "rotation":        round(float(rotation), 1),
            "frame_thickness": frame_thick,
            "dsp_technique":   "clear-lens refraction compositing" if is_clear else "specular highlight compositing",
        }

        return out, masks_used, dsp_meta

    # ------------------------------------------------------------------
    # Effect: Sunglasses PNG
    # ------------------------------------------------------------------

    def _apply_sunglasses_png(
        self,
        image: np.ndarray,
        png_path: str | None = None,
        offset_x: float = 0.0,
        offset_y: float = 0.0,
        scale: float = 1.0,
        rotation: float = 0.0,
    ) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, Any]]:
        """Overlay a sunglasses PNG (with alpha channel) onto the face.

        The PNG is scaled so its width spans the full inter-temple region,
        rotated to match head roll, then alpha-composited onto the image.

        Args:
            image:    BGR uint8 source.
            png_path: Path to a BGRA (or BGR) PNG. Defaults to
                      ``effects/assets/sunglasses.png``.

        Returns:
            ``(composited_image, masks_used, dsp_metadata)``
        """
        resolved = Path(png_path) if png_path else _DEFAULT_PNG
        if not resolved.is_file():
            raise FileNotFoundError(
                f"Sunglasses PNG not found at {resolved}. "
                "Save the PNG to effects/assets/sunglasses.png or pass png_path."
            )

        sg = cv2.imread(str(resolved), cv2.IMREAD_UNCHANGED)
        if sg is None:
            raise IOError(f"cv2.imread could not load: {resolved}")
        # Ensure 4-channel BGRA
        if sg.ndim == 2:
            sg = cv2.cvtColor(sg, cv2.COLOR_GRAY2BGRA)
        elif sg.shape[2] == 3:
            sg = cv2.cvtColor(sg, cv2.COLOR_BGR2BGRA)

        pts = self._pixels
        h, w = image.shape[:2]

        # Key landmarks
        l_inner = _pt(pts, 133)   # left eye inner corner
        l_outer = _pt(pts, 33)    # left eye outer corner
        r_inner = _pt(pts, 362)   # right eye inner corner
        r_outer = _pt(pts, 263)   # right eye outer corner
        l_brow  = _pt(pts, 105)   # left brow top
        r_brow  = _pt(pts, 334)   # right brow top

        # Eye centres
        l_cx = (l_inner[0] + l_outer[0]) / 2.0
        l_cy = (l_inner[1] + l_outer[1]) / 2.0
        r_cx = (r_inner[0] + r_outer[0]) / 2.0
        r_cy = (r_inner[1] + r_outer[1]) / 2.0

        # Face centre and inter-eye geometry
        center_x = (l_cx + r_cx) / 2.0
        center_y = (l_cy + r_cy) / 2.0
        eye_span = math.dist(l_outer, r_outer)

        # Roll angle + user rotation
        angle_deg = math.degrees(math.atan2(r_cy - l_cy, r_cx - l_cx)) + float(rotation)

        # Scale PNG: width covers temples (2.2× eye_span) plus user scale
        scale = float(np.clip(scale, 0.25, 3.0))
        target_w = max(1, int(eye_span * 2.2 * scale))
        aspect   = sg.shape[0] / sg.shape[1]
        target_h = max(1, int(target_w * aspect))

        sg_resized = cv2.resize(sg, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)

        # Rotate with expanded canvas so corners are not clipped
        rot_center = (target_w / 2.0, target_h / 2.0)
        M = cv2.getRotationMatrix2D(rot_center, angle_deg, 1.0)
        cos_a = abs(M[0, 0])
        sin_a = abs(M[0, 1])
        new_w = int(target_h * sin_a + target_w * cos_a)
        new_h = int(target_h * cos_a + target_w * sin_a)
        M[0, 2] += (new_w - target_w) / 2.0
        M[1, 2] += (new_h - target_h) / 2.0
        sg_rot = cv2.warpAffine(
            sg_resized, M, (new_w, new_h),
            flags=cv2.INTER_LANCZOS4,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0),
        )

        # Vertical nudge: shift glasses up 15 % of brow–eye distance
        brow_y   = (l_brow[1] + r_brow[1]) / 2.0
        v_offset = (center_y - brow_y) * 0.15

        # Top-left corner of pasted region
        place_x = int(center_x + float(offset_x) - new_w / 2.0)
        place_y = int(center_y + float(offset_y) - v_offset - new_h / 2.0)

        out       = image.copy()
        lens_mask = np.zeros((h, w), dtype=np.float32)

        # Clip to image bounds
        sg_x1 = max(0, place_x);          sg_y1 = max(0, place_y)
        sg_x2 = min(w, place_x + new_w);  sg_y2 = min(h, place_y + new_h)
        src_x1 = sg_x1 - place_x;         src_y1 = sg_y1 - place_y
        src_x2 = src_x1 + (sg_x2 - sg_x1); src_y2 = src_y1 + (sg_y2 - sg_y1)

        if sg_x2 > sg_x1 and sg_y2 > sg_y1:
            patch = sg_rot[src_y1:src_y2, src_x1:src_x2]
            alpha = patch[:, :, 3:4].astype(np.float32) / 255.0
            bg    = out[sg_y1:sg_y2, sg_x1:sg_x2].astype(np.float32)
            fg    = patch[:, :, :3].astype(np.float32)
            out[sg_y1:sg_y2, sg_x1:sg_x2] = np.clip(
                fg * alpha + bg * (1.0 - alpha), 0, 255
            ).astype(np.uint8)
            lens_mask[sg_y1:sg_y2, sg_x1:sg_x2] = alpha[:, :, 0]

        dsp_meta = {
            "blend_alpha":      1.0,
            "specular_alpha":   0.0,
            "kernel_sizes":     [],
            "color_space":      "BGR",
            "lens_style":       "png",
            "inter_eye_angle":  round(angle_deg, 2),
            "target_width_px":  target_w,
            "target_height_px": target_h,
            "offset_x":         round(float(offset_x), 1),
            "offset_y":         round(float(offset_y), 1),
            "scale":            round(scale, 2),
            "rotation":         round(float(rotation), 1),
            "png_path":         str(resolved),
            "dsp_technique":    "alpha-channel compositing",
        }

        return out, {"sunglasses_lens": lens_mask, "sunglasses_reflection": np.zeros_like(lens_mask)}, dsp_meta

    # ------------------------------------------------------------------
    # Effect: Beard
    # ------------------------------------------------------------------

    def _apply_beard_png(
        self,
        image: np.ndarray,
        png_path: str | None = None,
        color: str = "black",
        offset_x: float = 0.0,
        offset_y: float = 0.0,
        scale: float = 1.0,
        rotation: float = 0.0,
    ) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, Any]]:
        """Overlay a beard PNG (white or transparent background) onto the face.

        White backgrounds are automatically converted to transparency so the
        beard blends cleanly over any skin tone.
        """
        resolved = Path(png_path) if png_path else (_ASSETS_DIR / "beard.png")
        if not resolved.is_file():
            raise FileNotFoundError(
                f"Beard PNG not found at {resolved}. "
                "Save the PNG to effects/assets/beard.png"
            )

        raw = cv2.imread(str(resolved), cv2.IMREAD_UNCHANGED)
        if raw is None:
            raise IOError(f"cv2.imread could not load: {resolved}")

        if raw.ndim == 2:
            raw = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)

        if raw.shape[2] == 4 and raw[:, :, 3].mean() < 250:
            # Genuine transparency present — use as-is
            sg = raw
        else:
            # White background: derive alpha from inverse brightness
            bgr = raw[:, :, :3]
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            alpha = np.clip((255 - gray.astype(np.float32)) * 2.0, 0, 255).astype(np.uint8)
            sg = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
            sg[:, :, 3] = alpha

        hair_tones = {
            "black": (
                np.array((10, 9, 8), dtype=np.float32),
                np.array((28, 25, 23), dtype=np.float32),
                np.array((58, 52, 46), dtype=np.float32),
            ),
            "brown": (
                np.array((22, 28, 38), dtype=np.float32),
                np.array((45, 62, 86), dtype=np.float32),
                np.array((90, 105, 128), dtype=np.float32),
            ),
            "blonde": (
                np.array((58, 70, 82), dtype=np.float32),
                np.array((102, 126, 148), dtype=np.float32),
                np.array((162, 178, 190), dtype=np.float32),
            ),
        }
        if color not in hair_tones:
            color = "black"

        shadow, midtone, highlight = hair_tones[color]
        gray = cv2.cvtColor(sg[:, :, :3], cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        tone = np.clip(gray[:, :, np.newaxis] * 1.35, 0.0, 1.0)
        low = shadow * (1.0 - tone) + midtone * tone
        high = midtone * (1.0 - tone) + highlight * tone
        sg[:, :, :3] = np.clip(low * (1.0 - tone) + high * tone, 0, 255).astype(np.uint8)

        pts = self._pixels
        h, w = image.shape[:2]

        l_inner = _pt(pts, 133);  l_outer = _pt(pts, 33)
        r_inner = _pt(pts, 362);  r_outer = _pt(pts, 263)
        l_mouth = _pt(pts, 61);   r_mouth = _pt(pts, 291)
        chin    = _pt(pts, 152)
        l_jaw   = _pt(pts, 172);  r_jaw   = _pt(pts, 397)

        l_cx = (l_inner[0] + l_outer[0]) / 2.0
        l_cy = (l_inner[1] + l_outer[1]) / 2.0
        r_cx = (r_inner[0] + r_outer[0]) / 2.0
        r_cy = (r_inner[1] + r_outer[1]) / 2.0
        angle_deg = math.degrees(math.atan2(r_cy - l_cy, r_cx - l_cx)) + float(rotation)

        # Beard centre: horizontally between jaw points, vertically between
        # mouth and chin (shifted slightly down so the full beard fits)
        center_x = (l_jaw[0] + r_jaw[0]) / 2.0
        center_y = (l_mouth[1] + chin[1]) / 2.0 + (chin[1] - l_mouth[1]) * 0.2

        # Width spans the jaw; scale adjustable by user
        jaw_span  = math.dist(l_jaw, r_jaw)
        scale_f   = float(np.clip(scale, 0.3, 2.5))
        target_w  = max(1, int(jaw_span * 1.55 * scale_f))
        aspect    = sg.shape[0] / sg.shape[1]
        target_h  = max(1, int(target_w * aspect))

        sg_resized = cv2.resize(sg, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)

        # Rotate with expanded canvas
        M = cv2.getRotationMatrix2D((target_w / 2.0, target_h / 2.0), angle_deg, 1.0)
        cos_a, sin_a = abs(M[0, 0]), abs(M[0, 1])
        new_w = int(target_h * sin_a + target_w * cos_a)
        new_h = int(target_h * cos_a + target_w * sin_a)
        M[0, 2] += (new_w - target_w) / 2.0
        M[1, 2] += (new_h - target_h) / 2.0
        sg_rot = cv2.warpAffine(sg_resized, M, (new_w, new_h),
                                flags=cv2.INTER_LANCZOS4,
                                borderMode=cv2.BORDER_CONSTANT,
                                borderValue=(0, 0, 0, 0))

        place_x = int(center_x + float(offset_x) - new_w / 2.0)
        place_y = int(center_y + float(offset_y) - new_h / 2.0)

        out        = image.copy()
        beard_mask = np.zeros((h, w), dtype=np.float32)

        sg_x1 = max(0, place_x);            sg_y1 = max(0, place_y)
        sg_x2 = min(w, place_x + new_w);    sg_y2 = min(h, place_y + new_h)
        src_x1 = sg_x1 - place_x;           src_y1 = sg_y1 - place_y
        src_x2 = src_x1 + (sg_x2 - sg_x1); src_y2 = src_y1 + (sg_y2 - sg_y1)

        if sg_x2 > sg_x1 and sg_y2 > sg_y1:
            patch = sg_rot[src_y1:src_y2, src_x1:src_x2]
            alpha = patch[:, :, 3:4].astype(np.float32) / 255.0
            bg    = out[sg_y1:sg_y2, sg_x1:sg_x2].astype(np.float32)
            fg    = patch[:, :, :3].astype(np.float32)
            out[sg_y1:sg_y2, sg_x1:sg_x2] = np.clip(
                fg * alpha + bg * (1.0 - alpha), 0, 255
            ).astype(np.uint8)
            beard_mask[sg_y1:sg_y2, sg_x1:sg_x2] = alpha[:, :, 0]

        dsp_meta = {
            "blend_alpha":   1.0,
            "color_space":   "BGR",
            "beard_style":   "png",
            "beard_color":   color,
            "angle_deg":     round(angle_deg, 2),
            "target_width":  target_w,
            "offset_x":      round(float(offset_x), 1),
            "offset_y":      round(float(offset_y), 1),
            "scale":         round(scale_f, 2),
            "png_path":      str(resolved),
            "dsp_technique": "white-background removal + alpha compositing",
        }

        return out, {"beard_region": beard_mask}, dsp_meta

    def _apply_beard(
        self,
        image: np.ndarray,
        color: str = "black",
        offset_x: float = 0.0,
        offset_y: float = 0.0,
        scale: float = 1.0,
    ) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, Any]]:
        """Render a beard. Uses the PNG asset when available, drawn fallback otherwise.

        Args:
            image:    BGR uint8 source.
            color:    "black", "brown", or "blonde".
            offset_x/y: pixel nudge (movable by the user).
            scale:    size multiplier relative to the face.

        Returns:
            ``(composited_image, masks_used, dsp_metadata)``
        """
        # Auto-upgrade to PNG when the asset is present
        if (_ASSETS_DIR / "beard.png").is_file():
            return self._apply_beard_png(
                image, color=color, offset_x=offset_x, offset_y=offset_y, scale=scale
            )

        _COLORS = {
            "black":  (14, 13, 12),
            "brown":  (32, 45, 66),
            "blonde": (88, 112, 138),
        }
        if color not in _COLORS:
            color = "black"
        base_color = _COLORS[color]

        pts  = self._pixels
        h, w = image.shape[:2]
        ox, oy = float(offset_x), float(offset_y)
        scale  = float(np.clip(scale, 0.3, 2.5))

        def lp(idx: int) -> tuple[float, float]:
            x, y = _pt(pts, idx)
            return x + ox, y + oy

        # Key jaw/face landmarks
        chin      = lp(152)
        l_mouth   = lp(61)
        r_mouth   = lp(291)
        lip_bot   = lp(17)

        # Jaw outline points — left side (screen-left = face right)
        left_jaw  = [lp(i) for i in [172, 136, 150, 149, 176, 148]]
        # Jaw outline points — right side
        right_jaw = [lp(i) for i in [377, 400, 378, 379, 365, 397]]

        # Lower lip bottom boundary (top edge of beard)
        lip_edge  = [lp(i) for i in [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291]]

        # Expand the jaw outward slightly based on scale
        def _expand(pt, ref, s):
            """Push pt away from ref by (s-1) fraction."""
            dx, dy = pt[0] - ref[0], pt[1] - ref[1]
            return (pt[0] + dx * (s - 1.0), pt[1] + dy * (s - 1.0))

        face_cx = (l_mouth[0] + r_mouth[0]) / 2.0
        face_cy = (l_mouth[1] + chin[1]) / 2.0
        ref = (face_cx, face_cy)

        left_jaw  = [_expand(p, ref, scale) for p in left_jaw]
        right_jaw = [_expand(p, ref, scale) for p in right_jaw]
        chin_exp  = _expand(chin, ref, scale)

        # Build outer beard polygon: left mouth → left jaw → chin → right jaw → right mouth
        beard_poly = (
            [l_mouth] + left_jaw + [chin_exp] + list(reversed(right_jaw)) + [r_mouth]
        )
        # Close top with lower lip edge (reversed so polygon goes clockwise)
        beard_poly = beard_poly + list(reversed(lip_edge[1:-1]))

        poly_pts = np.array([[int(p[0]), int(p[1])] for p in beard_poly],
                             dtype=np.int32)

        # --- draw beard on an overlay canvas ---
        overlay   = image.copy()
        beard_mask = np.zeros((h, w), dtype=np.float32)

        cv2.fillPoly(overlay, [poly_pts], base_color, cv2.LINE_AA)
        cv2.fillPoly(beard_mask, [poly_pts], 1.0, cv2.LINE_AA)

        # Hair-streak texture: directional noise blurred vertically
        rng = np.random.default_rng(42)
        noise = rng.integers(0, 60, (h, w), dtype=np.uint8)
        noise_blur = cv2.GaussianBlur(noise, (1, 9), 0)
        noise_3ch  = cv2.merge([noise_blur, noise_blur, noise_blur]).astype(np.float32)
        mask_3ch   = beard_mask[:, :, np.newaxis]
        overlay_f  = overlay.astype(np.float32)
        # Darken the beard slightly where hair streaks are stronger
        overlay_f  = np.clip(overlay_f - noise_3ch * mask_3ch * 0.35, 0, 255)
        overlay    = overlay_f.astype(np.uint8)

        # Feather the top edge of the beard (soft blend into skin)
        feather_mask = cv2.GaussianBlur(beard_mask, (0, 0), sigmaX=6, sigmaY=3)

        out = image.copy().astype(np.float32)
        ov  = overlay.astype(np.float32)
        alpha_3ch = feather_mask[:, :, np.newaxis] * 0.88
        out = np.clip(ov * alpha_3ch + out * (1.0 - alpha_3ch), 0, 255).astype(np.uint8)

        dsp_meta = {
            "blend_alpha":    0.88,
            "color_space":    "BGR",
            "beard_color":    color,
            "offset_x":       round(ox, 1),
            "offset_y":       round(oy, 1),
            "scale":          round(scale, 2),
            "dsp_technique":  "polygon fill + directional noise texture + feathered alpha blend",
        }

        return out, {"beard_region": beard_mask}, dsp_meta


# ---------------------------------------------------------------------------
# Style tables
# ---------------------------------------------------------------------------

# horizontal-radius as a fraction of inter-eye span
_STYLE_LENS_X: dict[str, float] = {
    "classic":  0.34,
    "aviator":  0.36,
    "round":    0.32,
    "glasses":  0.34,   # same shape as classic, clear lens
}

# lens vertical-to-horizontal aspect ratio
_STYLE_LENS_RATIO: dict[str, float] = {
    "classic": 0.68,
    "aviator": 1.20,   # tall teardrop — real aviator height > width
    "round":   1.08,
    "glasses": 0.72,   # slightly taller than classic for a modern glasses look
}

# BGR lens fill colors — style-specific tints
_STYLE_LENS_COLOR: dict[str, tuple[int, int, int]] = {
    "classic": (22, 22, 26),    # cool neutral dark (very slight blue)
    "aviator": (18, 34, 22),    # G-15 green-gray (classic aviator tint)
    "round":   (28, 20, 22),    # slight warm dark gray
    "glasses": (235, 230, 220), # clear lens
}

# BGR frame body colors per style
_STYLE_FRAME_COLOR: dict[str, tuple[int, int, int]] = {
    "classic": (10,  8,  8),   # deep matte black acetate
    "aviator": (22, 18, 12),   # dark gunmetal
    "round":   (18, 12,  8),   # dark tortoiseshell
    "glasses": (40, 35, 30),   # warm tortoiseshell frame
}


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def _pt(pixels: np.ndarray, idx: int) -> tuple[float, float]:
    """Return (x, y) for landmark index, clamped to valid range."""
    if idx < len(pixels):
        return float(pixels[idx, 0]), float(pixels[idx, 1])
    # fallback: return array centre
    return float(pixels[:, 0].mean()), float(pixels[:, 1].mean())


def _rotate_translate(points: np.ndarray, cx: float, cy: float, angle_deg: float) -> np.ndarray:
    angle = math.radians(angle_deg)
    rot = np.array([
        [math.cos(angle), -math.sin(angle)],
        [math.sin(angle),  math.cos(angle)],
    ], dtype=np.float32)
    shifted = points @ rot.T
    shifted[:, 0] += cx
    shifted[:, 1] += cy
    return np.round(shifted).astype(np.int32)


def _draw_lens(
    canvas: np.ndarray,
    mask: np.ndarray,
    cx: float, cy: float,
    rx: float, ry: float,
    angle_deg: float,
    style: str,
    fill_color: tuple[int, int, int],
    frame_color: tuple[int, int, int],
    frame_thick: int,
) -> None:
    """Draw one lens shaped and framed according to style."""
    h, w = canvas.shape[:2]
    irx, iry = max(1, int(rx)), max(1, int(ry))

    # ── Aviator: teardrop shape with thin wire frame ──────────────────────────
    if style == "aviator":
        outer = np.array([
            [-0.52 * irx, -0.90 * iry],
            [-0.18 * irx, -0.98 * iry],
            [ 0.18 * irx, -0.98 * iry],
            [ 0.52 * irx, -0.90 * iry],
            [ 1.02 * irx, -0.40 * iry],
            [ 1.10 * irx,  0.18 * iry],
            [ 0.96 * irx,  0.72 * iry],
            [ 0.68 * irx,  1.04 * iry],
            [ 0.28 * irx,  1.16 * iry],
            [ 0.00 * irx,  1.20 * iry],
            [-0.28 * irx,  1.16 * iry],
            [-0.68 * irx,  1.04 * iry],
            [-0.96 * irx,  0.72 * iry],
            [-1.10 * irx,  0.18 * iry],
            [-1.02 * irx, -0.40 * iry],
        ], dtype=np.float32)
        outer_pts = _rotate_translate(outer, cx, cy, angle_deg)

        # Fill entire teardrop with lens tint
        cv2.fillPoly(canvas, [outer_pts], fill_color, cv2.LINE_AA)
        cv2.fillPoly(mask,   [outer_pts], 1.0,        cv2.LINE_AA)

        # Strong top-black → warm-gray bottom gradient (as seen on real aviators)
        y_lo = float(outer_pts[:, 1].min())
        y_hi = float(outer_pts[:, 1].max())
        if y_hi > y_lo:
            gm = np.zeros((h, w), dtype=np.float32)
            cv2.fillPoly(gm, [outer_pts], 1.0, cv2.LINE_AA)
            y_t = np.clip(
                (np.arange(h, dtype=np.float32)[:, np.newaxis] - y_lo) / (y_hi - y_lo),
                0, 1,
            )
            c_f = canvas.astype(np.float32)
            c_f[:, :, 0] = np.clip(c_f[:, :, 0] + y_t * 72.0 * gm, 0, 255)  # blue
            c_f[:, :, 1] = np.clip(c_f[:, :, 1] + y_t * 68.0 * gm, 0, 255)  # green
            c_f[:, :, 2] = np.clip(c_f[:, :, 2] + y_t * 82.0 * gm, 0, 255)  # red (warmer)
            canvas[:] = c_f.astype(np.uint8)

        # Thin wire frame outline (no thick acetate body)
        cv2.polylines(canvas, [outer_pts], True, frame_color, max(1, frame_thick), cv2.LINE_AA)
        hi = tuple(min(255, c + 48) for c in frame_color)
        cv2.polylines(canvas, [outer_pts], True, hi, 1, cv2.LINE_AA)
        return

    # ── Wayfarer / default styles ─────────────────────────────────────────────
    # Modern wayfarer silhouette — flat top, gently rounded bottom corners
    top_lift = -0.80 * iry
    outer = np.array([
        [-1.16 * irx,  top_lift],
        [-0.55 * irx, -0.96 * iry],
        [ 1.14 * irx, -0.80 * iry],
        [ 1.10 * irx, -0.14 * iry],
        [ 0.80 * irx,  0.84 * iry],
        [ 0.22 * irx,  1.04 * iry],
        [-0.60 * irx,  0.97 * iry],
        [-1.04 * irx,  0.52 * iry],
        [-1.20 * irx, -0.18 * iry],
    ], dtype=np.float32)
    inner = outer * np.array([0.80, 0.76], dtype=np.float32)
    inner[:, 1] += iry * 0.07

    # Bevel ring: slightly smaller than outer, for a 3-D frame effect
    bevel = outer * np.array([0.90, 0.88], dtype=np.float32)
    bevel[:, 1] += iry * 0.03

    outer_pts = _rotate_translate(outer, cx, cy, angle_deg)
    bevel_pts = _rotate_translate(bevel, cx, cy, angle_deg)
    inner_pts = _rotate_translate(inner, cx, cy, angle_deg)

    # Outer frame body
    cv2.fillPoly(canvas, [outer_pts], frame_color, cv2.LINE_AA)
    # Bevel highlight: lighter strip along inner bevel ring
    bevel_hi = tuple(min(255, c + 55) for c in frame_color)
    cv2.polylines(canvas, [bevel_pts], True, bevel_hi, max(1, frame_thick // 2), cv2.LINE_AA)
    # Lens fill
    cv2.fillPoly(canvas, [inner_pts], fill_color, cv2.LINE_AA)
    # Vertical gradient: top stays darkest, bottom grades lighter (gives glass depth)
    h_c, w_c = canvas.shape[:2]
    if fill_color[0] < 150:  # dark lenses only
        gm = np.zeros((h_c, w_c), dtype=np.float32)
        cv2.fillPoly(gm, [inner_pts], 1.0, cv2.LINE_AA)
        y_lo = float(inner_pts[:, 1].min())
        y_hi = float(inner_pts[:, 1].max())
        if y_hi > y_lo:
            y_t = np.clip(
                (np.arange(h_c, dtype=np.float32)[:, np.newaxis] - y_lo) / (y_hi - y_lo),
                0, 1,
            )
            bright = (y_t * 60.0 * gm)[:, :, np.newaxis]
            c_f = canvas.astype(np.float32)
            c_f += bright
            canvas[:] = np.clip(c_f, 0, 255).astype(np.uint8)
    # Crisp outer edge
    edge_shade = tuple(max(0, c - 8) for c in frame_color)
    cv2.polylines(canvas, [outer_pts], True, edge_shade, max(2, frame_thick), cv2.LINE_AA)
    # Inner shadow line
    cv2.polylines(canvas, [inner_pts], True, (6, 6, 8), max(1, frame_thick - 1), cv2.LINE_AA)
    cv2.fillPoly(mask, [inner_pts], 1.0, cv2.LINE_AA)

    # Rivet — metallic detail at hinge corner
    rivet_local = np.array([[-0.98 * irx, -0.54 * iry]], dtype=np.float32)
    rivet = _rotate_translate(rivet_local, cx, cy, angle_deg)[0]
    rv = (max(4, irx // 11), max(3, iry // 14))
    cv2.ellipse(canvas, tuple(rivet), rv, angle_deg, 0, 360, (180, 178, 168), -1, cv2.LINE_AA)
    cv2.ellipse(canvas, tuple(rivet), rv, angle_deg, 0, 360, (15, 15, 15),  1,  cv2.LINE_AA)
    # Rivet shine dot
    shine_local = np.array([[-0.94 * irx, -0.58 * iry]], dtype=np.float32)
    shine = _rotate_translate(shine_local, cx, cy, angle_deg)[0]
    cv2.ellipse(canvas, tuple(shine), (max(1, rv[0] // 3), max(1, rv[1] // 3)),
                angle_deg, 0, 360, (230, 228, 220), -1, cv2.LINE_AA)


def _draw_frame_outline(
    canvas: np.ndarray,
    cx: float, cy: float,
    rx: float, ry: float,
    angle_deg: float,
    frame_color: tuple[int, int, int],
    frame_thick: int,
    style: str = "classic",
) -> None:
    """Draw only the frame outline of a lens (no fill) — used for redraw pass."""
    irx, iry = max(1, int(rx)), max(1, int(ry))

    if style == "aviator":
        outer = np.array([
            [-0.52 * irx, -0.90 * iry],
            [-0.18 * irx, -0.98 * iry],
            [ 0.18 * irx, -0.98 * iry],
            [ 0.52 * irx, -0.90 * iry],
            [ 1.02 * irx, -0.40 * iry],
            [ 1.10 * irx,  0.18 * iry],
            [ 0.96 * irx,  0.72 * iry],
            [ 0.68 * irx,  1.04 * iry],
            [ 0.28 * irx,  1.16 * iry],
            [ 0.00 * irx,  1.20 * iry],
            [-0.28 * irx,  1.16 * iry],
            [-0.68 * irx,  1.04 * iry],
            [-0.96 * irx,  0.72 * iry],
            [-1.10 * irx,  0.18 * iry],
            [-1.02 * irx, -0.40 * iry],
        ], dtype=np.float32)
        pts = _rotate_translate(outer, cx, cy, angle_deg)
        cv2.polylines(canvas, [pts], True, frame_color, max(1, frame_thick), cv2.LINE_AA)
        hi = tuple(min(255, c + 48) for c in frame_color)
        cv2.polylines(canvas, [pts], True, hi, 1, cv2.LINE_AA)
        return

    top_lift = -0.78 * iry
    outer = np.array([
        [-1.14 * irx, top_lift],
        [-0.62 * irx, -0.92 * iry],
        [ 1.12 * irx, -0.78 * iry],
        [ 1.08 * irx, -0.18 * iry],
        [ 0.78 * irx,  0.82 * iry],
        [ 0.28 * irx,  1.02 * iry],
        [-0.64 * irx,  0.95 * iry],
        [-1.02 * irx,  0.50 * iry],
        [-1.18 * irx, -0.20 * iry],
    ], dtype=np.float32)
    outer_pts = _rotate_translate(outer, cx, cy, angle_deg)
    cv2.polylines(canvas, [outer_pts], True, frame_color,
                  max(2, frame_thick), cv2.LINE_AA)

    # corner rivet
    rivet_local = np.array([[-0.98 * irx, -0.52 * iry]], dtype=np.float32)
    rivet = _rotate_translate(rivet_local, cx, cy, angle_deg)[0]
    cv2.ellipse(canvas, tuple(rivet), (max(3, irx // 13), max(2, iry // 16)),
                angle_deg, 0, 360, (215, 215, 205), -1, cv2.LINE_AA)
    cv2.ellipse(canvas, tuple(rivet), (max(3, irx // 13), max(2, iry // 16)),
                angle_deg, 0, 360, (20, 20, 20), 1, cv2.LINE_AA)


def _draw_bridge(
    canvas: np.ndarray,
    l_cx: float, l_cy: float,
    r_cx: float, r_cy: float,
    rx: float, ry: float,
    angle_deg: float,
    color: tuple[int, int, int],
    thick: int,
) -> None:
    """Draw a curved keyhole arch bridge connecting the two lenses."""
    angle_rad = math.radians(angle_deg)
    dx, dy = math.cos(angle_rad), math.sin(angle_rad)
    perp_x, perp_y = -dy, dx  # perpendicular direction (toward forehead at 0° tilt)

    # Attachment points at the inner edge of each lens
    l_ex = l_cx + dx * rx * 0.82
    l_ey = l_cy + dy * rx * 0.82
    r_ex = r_cx - dx * rx * 0.82
    r_ey = r_cy - dy * rx * 0.82

    # Quadratic Bézier control point arching upward (nose-bridge shape)
    ctrl_x = (l_ex + r_ex) / 2.0 + perp_x * ry * 0.54
    ctrl_y = (l_ey + r_ey) / 2.0 + perp_y * ry * 0.54

    n = 22
    pts = []
    for i in range(n + 1):
        t = i / n
        bx = (1 - t) ** 2 * l_ex + 2 * (1 - t) * t * ctrl_x + t ** 2 * r_ex
        by = (1 - t) ** 2 * l_ey + 2 * (1 - t) * t * ctrl_y + t ** 2 * r_ey
        pts.append([int(round(bx)), int(round(by))])

    pts_arr = np.array(pts, dtype=np.int32)
    cv2.polylines(canvas, [pts_arr], False, color, max(2, thick), cv2.LINE_AA)
    # Lighter highlight strip along the top of the arch
    hi = tuple(min(255, c + 50) for c in color)
    cv2.polylines(canvas, [pts_arr], False, hi, 1, cv2.LINE_AA)


def _composite_clear_lenses(image: np.ndarray, lens_mask: np.ndarray) -> np.ndarray:
    """Composite clear lenses with mild refraction, transmission, and edge depth."""
    h, w = image.shape[:2]
    mask = np.clip(lens_mask.astype(np.float32), 0.0, 1.0)
    if float(mask.max()) <= 0.0:
        return image.copy()

    mask_u8 = np.clip(mask * 255.0, 0, 255).astype(np.uint8)
    soft = cv2.GaussianBlur(mask, (0, 0), 1.6)
    dist = cv2.distanceTransform(mask_u8, cv2.DIST_L2, 5)
    if float(dist.max()) > 0.0:
        dist = dist / float(dist.max())

    grad_x = cv2.Sobel(dist, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(dist, cv2.CV_32F, 0, 1, ksize=3)
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    map_x = np.clip(xs + grad_x * 3.2, 0, w - 1).astype(np.float32)
    map_y = np.clip(ys + grad_y * 2.0, 0, h - 1).astype(np.float32)
    refracted = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)

    refracted_f = refracted.astype(np.float32)
    tint = np.full_like(refracted_f, (8.0, 10.0, 12.0), dtype=np.float32)
    transmitted = np.clip(refracted_f * 1.035 + tint, 0, 255)

    out = image.astype(np.float32)
    alpha = (soft * 0.34)[:, :, np.newaxis]
    out = out * (1.0 - alpha) + transmitted * alpha

    edge = cv2.morphologyEx(mask_u8, cv2.MORPH_GRADIENT, np.ones((5, 5), np.uint8)).astype(np.float32) / 255.0
    edge = cv2.GaussianBlur(edge, (0, 0), 1.0)
    upper = np.clip((-grad_y * 2.5) + 0.18, 0.0, 1.0) * edge
    lower = np.clip((grad_y * 2.2) + 0.12, 0.0, 1.0) * edge
    out += upper[:, :, np.newaxis] * 18.0
    out -= lower[:, :, np.newaxis] * 10.0
    # Anti-reflective coating: subtle blue-purple shimmer at lens perimeter
    ar_alpha = (edge * mask * 0.72)[:, :, np.newaxis]
    out += ar_alpha * np.array([22.0, 9.0, 5.0], dtype=np.float32)  # BGR: blue tint
    return np.clip(out, 0, 255).astype(np.uint8)


def _draw_glass_highlights(
    canvas: np.ndarray,
    mask: np.ndarray,
    cx: float, cy: float,
    rx: float, ry: float,
    angle_deg: float,
) -> None:
    """Draw curved, low-opacity glints for clear prescription lenses."""
    h, w = canvas.shape[:2]
    layer = canvas.copy()
    combined = np.zeros((h, w), dtype=np.float32)

    for ox, oy, alpha, scale_x, scale_y, arc_start, arc_end in [
        (-0.23, -0.34, 0.18, 0.42, 0.12, 190, 335),
        (0.20, 0.20, 0.10, 0.34, 0.10, 15, 145),
        (-0.02, -0.04, 0.07, 0.64, 0.34, 212, 256),
    ]:
        hx = int(np.clip(cx + rx * ox, 0, w - 1))
        hy = int(np.clip(cy + ry * oy, 0, h - 1))
        hrx = max(2, int(rx * scale_x))
        hry = max(1, int(ry * scale_y))
        patch = np.zeros((h, w), dtype=np.float32)
        cv2.ellipse(layer, (hx, hy), (hrx, hry), angle_deg - 18.0,
                    arc_start, arc_end, (255, 255, 255), max(1, int(rx * 0.018)), cv2.LINE_AA)
        cv2.ellipse(patch, (hx, hy), (hrx, hry), angle_deg - 18.0,
                    arc_start, arc_end, 1.0, max(1, int(rx * 0.018)), cv2.LINE_AA)
        patch = cv2.GaussianBlur(patch, (0, 0), 0.8)
        a3 = (patch * alpha)[:, :, np.newaxis]
        canvas[:] = np.clip(canvas.astype(np.float32) * (1.0 - a3) + layer.astype(np.float32) * a3,
                            0, 255).astype(np.uint8)
        combined += patch
        layer = canvas.copy()

    mask += np.clip(combined, 0, 1)


def _draw_aviator_brow_bar(
    canvas: np.ndarray,
    l_cx: float, l_cy: float,
    r_cx: float, r_cy: float,
    rx: float, ry: float,
    angle_rad: float,
    color: tuple[int, int, int],
    thick: int,
) -> None:
    """Draw the signature straight brow bar above aviator lenses."""
    dx = math.cos(angle_rad)
    dy = math.sin(angle_rad)
    ux = -math.sin(angle_rad)  # unit vector toward forehead (up in image)
    uy = -math.cos(angle_rad)

    h_off = ry * 0.94   # height above lens centre
    span  = rx * 1.06   # extends just beyond lens outer edge

    lx = int(l_cx - dx * span + ux * h_off)
    ly = int(l_cy - dy * span + uy * h_off)
    rx_ = int(r_cx + dx * span + ux * h_off)
    ry_ = int(r_cy + dy * span + uy * h_off)

    cv2.line(canvas, (lx, ly), (rx_, ry_), color, max(1, thick), cv2.LINE_AA)
    hi = tuple(min(255, c + 52) for c in color)
    cv2.line(canvas, (lx, ly), (rx_, ry_), hi, 1, cv2.LINE_AA)


def _draw_tapered_temple(
    canvas: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
    color: tuple[int, int, int],
    thick_start: int,
) -> None:
    """Draw a temple arm that tapers from thick_start near the frame to 1 px at the ear end."""
    n = 16
    for i in range(n):
        t1, t2 = i / n, (i + 1) / n
        p1 = (int(start[0] + t1 * (end[0] - start[0])), int(start[1] + t1 * (end[1] - start[1])))
        p2 = (int(start[0] + t2 * (end[0] - start[0])), int(start[1] + t2 * (end[1] - start[1])))
        t_mid = (t1 + t2) / 2.0
        thick = max(1, int(round(thick_start * (1.0 - t_mid * 0.58))))
        cv2.line(canvas, p1, p2, color, thick, cv2.LINE_AA)


def _draw_specular(
    canvas: np.ndarray,
    mask: np.ndarray,
    cx: float, cy: float,
    rx: float, ry: float,
    angle_deg: float,
    style: str = "classic",
) -> None:
    """Realistic specular glare — large prominent band for aviator, standard for others."""
    h, w = canvas.shape[:2]

    if style == "aviator":
        # Very large, soft highlight covering the upper half of the lens
        hi_rx = max(2, int(rx * 0.82))
        hi_ry = max(1, int(ry * 0.44))
        hi_cx = int(np.clip(cx - rx * 0.04, 0, w - 1))
        hi_cy = int(np.clip(cy - ry * 0.26, 0, h - 1))
        sigma_x, sigma_y = rx * 0.44, ry * 0.22
        band_alpha = 0.28
        glint_x = int(np.clip(cx - rx * 0.38, 0, w - 1))
        glint_y = int(np.clip(cy - ry * 0.62, 0, h - 1))
        glint_rx, glint_ry = max(1, int(rx * 0.13)), max(1, int(ry * 0.09))
        glint_alpha = 0.80
    else:
        hi_rx = max(2, int(rx * 0.74))
        hi_ry = max(1, int(ry * 0.27))
        hi_cx = int(np.clip(cx - rx * 0.06, 0, w - 1))
        hi_cy = int(np.clip(cy - ry * 0.43, 0, h - 1))
        sigma_x, sigma_y = rx * 0.30, ry * 0.12
        band_alpha = 0.14
        glint_x = int(np.clip(cx - rx * 0.46, 0, w - 1))
        glint_y = int(np.clip(cy - ry * 0.52, 0, h - 1))
        glint_rx, glint_ry = max(1, int(rx * 0.088)), max(1, int(ry * 0.055))
        glint_alpha = 0.65

    band = np.zeros((h, w), dtype=np.float32)
    cv2.ellipse(band, (hi_cx, hi_cy), (hi_rx, hi_ry), angle_deg, 0, 360, 1.0, -1, cv2.LINE_AA)
    band = cv2.GaussianBlur(band, (0, 0), sigmaX=sigma_x, sigmaY=sigma_y)
    canvas[:] = np.clip(
        canvas.astype(np.float32) + band[:, :, np.newaxis] * band_alpha * 255, 0, 255
    ).astype(np.uint8)

    glint = np.zeros((h, w), dtype=np.float32)
    cv2.ellipse(glint, (glint_x, glint_y), (glint_rx, glint_ry),
                angle_deg - 32, 0, 360, 1.0, -1, cv2.LINE_AA)
    glint = cv2.GaussianBlur(glint, (0, 0), 1.2)
    canvas[:] = np.clip(
        canvas.astype(np.float32) + glint[:, :, np.newaxis] * glint_alpha * 255, 0, 255
    ).astype(np.uint8)

    mask += np.clip(band + glint, 0, 1)
