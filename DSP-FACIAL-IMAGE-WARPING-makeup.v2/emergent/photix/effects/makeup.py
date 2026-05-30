"""
effects/makeup.py — Photix virtual makeup engine
================================================

Landmark-accurate virtual makeup using the MediaPipe Face Mesh (478 points).

Supported makeup layers
------------------------
* **eyeshadow** (göz farı) — soft eyelid colour band above each eye.
      colours: blue, black, red, green, orange, brown, glitter (simli)
* **blush**     (allık)    — diffuse cheek colour on both cheeks.
      colours: pink, brown
* **lipstick**  (ruj)      — lip colouring with optional glossy highlight.
      colours: red, pink, brown, gloss (parlak)
* **eyeliner**  (eyeliner) — thin dark line along the upper lash line.
* **mascara**   (kirpik)   — darkened / extended upper lashes.

Design goals
------------
1. **Pose-stable.** Every region is rebuilt from the *current* landmarks on
   every frame, so it tracks the face when the head turns (live & video).
   Nothing is baked into a static texture, so makeup never "slides off".
2. **Per-eye / per-cheek geometry** computed independently → works under yaw.
3. **HSV / Lab tinting** that keeps the skin's own shading & pores, so the
   result reads as makeup on skin rather than flat paint.
4. Pure NumPy + OpenCV — same dependency footprint as the rest of Photix.

All public colour names accept English **and** Turkish synonyms
(e.g. "mavi" == "blue", "kahverengi" == "brown", "simli" == "glitter").
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# MediaPipe Face Mesh landmark index groups (478-point model)
# ---------------------------------------------------------------------------
# Upper-lid lash line (outer→inner) and the brow-side lid crease, used to
# build a closed polygon covering the mobile eyelid for eyeshadow.
_LEFT_EYE_UPPER  = [33, 246, 161, 160, 159, 158, 157, 173, 133]
_LEFT_EYE_LOWER  = [133, 155, 154, 153, 145, 144, 163, 7, 33]
# Crease landmarks a little higher than the lash line (toward the brow).
_LEFT_EYE_CREASE = [33, 247, 30, 29, 27, 28, 56, 190, 133]

_RIGHT_EYE_UPPER  = [263, 466, 388, 387, 386, 385, 384, 398, 362]
_RIGHT_EYE_LOWER  = [362, 382, 381, 380, 374, 373, 390, 249, 263]
_RIGHT_EYE_CREASE = [263, 467, 260, 259, 257, 258, 286, 414, 362]

# Lips: outer ring and inner ring (mouth opening).
_LIPS_OUTER = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
    291, 409, 270, 269, 267, 0, 37, 39, 40, 185,
]
_LIPS_INNER = [
    78, 95, 88, 178, 87, 14, 317, 402, 318, 324,
    308, 415, 310, 311, 312, 13, 82, 81, 80, 191,
]

# Cheek apple centre + a ring of nearby points to size the blush.
_LEFT_CHEEK_CENTER  = 50
_LEFT_CHEEK_RING    = [116, 123, 147, 187, 205, 36]
_RIGHT_CHEEK_CENTER = 280
_RIGHT_CHEEK_RING   = [345, 352, 376, 411, 425, 266]

# Reference points for global scale / pose.
_FACE_LEFT, _FACE_RIGHT = 234, 454      # ear-to-ear (face width)
_NOSE_TIP               = 1
_CHIN, _FOREHEAD        = 152, 10


# ---------------------------------------------------------------------------
# Colour tables  (all in BGR — OpenCV native order)
# ---------------------------------------------------------------------------
_EYESHADOW_BGR: dict[str, tuple[int, int, int]] = {
    "blue":    (200, 90, 30),     # mavi
    "black":   (40, 38, 40),      # siyah
    "red":     (50, 45, 180),     # kırmızı
    "green":   (70, 150, 40),     # yeşil
    "orange":  (30, 120, 235),    # turuncu
    "brown":   (55, 80, 120),     # kahverengi
    "glitter": (210, 200, 235),   # simli — shimmery champagne, gets sparkle
}

_BLUSH_BGR: dict[str, tuple[int, int, int]] = {
    "pink":  (150, 120, 235),     # pembe
    "brown": (90, 110, 165),      # kahverengi (warm bronze)
}

_LIPSTICK_BGR: dict[str, tuple[int, int, int]] = {
    "red":   (45, 40, 190),       # kırmızı
    "pink":  (140, 95, 225),      # pembe
    "brown": (40, 58, 108),       # kahverengi
    "gloss": (95, 80, 175),       # parlak — natural rose with strong gloss
}

# Turkish (and a few common) synonyms → canonical English keys.
_COLOR_SYNONYMS: dict[str, str] = {
    # eyeshadow / general
    "mavi": "blue", "siyah": "black", "kirmizi": "red", "kırmızı": "red",
    "yesil": "green", "yeşil": "green", "turuncu": "orange",
    "kahve": "brown", "kahverengi": "brown", "simli": "glitter",
    "parlak": "gloss", "pembe": "pink",
    # passthrough English
    "blue": "blue", "black": "black", "red": "red", "green": "green",
    "orange": "orange", "brown": "brown", "glitter": "glitter",
    "pink": "pink", "gloss": "gloss",
}


def _canon_color(name: str | None, default: str) -> str:
    if not name:
        return default
    return _COLOR_SYNONYMS.get(str(name).strip().lower(), str(name).strip().lower())


# ---------------------------------------------------------------------------
# Result dataclass (mirrors OverlayResult so callers can treat them alike)
# ---------------------------------------------------------------------------
@dataclass
class MakeupResult:
    final_image: np.ndarray
    masks_used: dict[str, np.ndarray] = field(default_factory=dict)
    dsp_metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Small landmark helpers
# ---------------------------------------------------------------------------
def _pt(lm: np.ndarray, idx: int) -> tuple[float, float]:
    if idx < len(lm):
        return float(lm[idx, 0]), float(lm[idx, 1])
    return float(lm[:, 0].mean()), float(lm[:, 1].mean())


def _dist(lm: np.ndarray, i: int, j: int) -> float:
    return math.dist(_pt(lm, i), _pt(lm, j))


def _points(lm: np.ndarray, indices: list[int]) -> np.ndarray:
    return np.array([_pt(lm, i) for i in indices], dtype=np.float32)


def _face_width(lm: np.ndarray) -> float:
    return max(_dist(lm, _FACE_LEFT, _FACE_RIGHT), 1.0)


def _poly_mask(shape: tuple[int, int], pts: np.ndarray) -> np.ndarray:
    """Filled antialiased polygon mask in [0,1]."""
    mask = np.zeros(shape, dtype=np.uint8)
    if len(pts) >= 3:
        cv2.fillPoly(mask, [np.round(pts).astype(np.int32)], 255, cv2.LINE_AA)
    return mask.astype(np.float32) / 255.0


# ---------------------------------------------------------------------------
# Core tinting primitive
# ---------------------------------------------------------------------------
def _apply_tint(
    image: np.ndarray,
    mask: np.ndarray,
    color_bgr: tuple[int, int, int],
    strength: float,
    preserve_luma: float = 0.55,
) -> np.ndarray:
    """Blend ``color_bgr`` into ``image`` wherever ``mask`` > 0.

    The blend is done in a luminance-preserving way: the colour's hue &
    chroma are applied while the original brightness variation (skin
    shading, lip texture, lash shadows) is largely retained.  This keeps
    the makeup looking like pigment on skin rather than a flat decal.

    Parameters
    ----------
    strength       : 0..1 overall opacity of the makeup.
    preserve_luma  : 0..1 how much of the original brightness to keep
                     (higher = more natural, lower = more opaque colour).
    """
    if mask.max() <= 0.0 or strength <= 0.0:
        return image

    out = image.astype(np.float32)
    color = np.array(color_bgr, dtype=np.float32)

    # Luminance of original (for preserving shading).
    luma = (0.114 * out[:, :, 0] + 0.587 * out[:, :, 1] + 0.299 * out[:, :, 2])
    luma_n = np.clip(luma / 200.0, 0.35, 1.25)[:, :, None]   # gentle normalisation

    # Shade the flat colour by the local brightness so folds/shadows survive.
    shaded = color[None, None, :] * (
        (1.0 - preserve_luma) + preserve_luma * luma_n
    )

    a = np.clip(mask * strength, 0.0, 1.0)[:, :, None]
    blended = out * (1.0 - a) + shaded * a
    return np.clip(blended, 0, 255).astype(np.uint8)


def _feather(mask: np.ndarray, sigma: float) -> np.ndarray:
    sigma = max(0.6, float(sigma))
    return cv2.GaussianBlur(mask, (0, 0), sigmaX=sigma, sigmaY=sigma)


# ===========================================================================
# EYESHADOW  (göz farı)
# ===========================================================================
def _eyelid_band(
    lm: np.ndarray,
    shape: tuple[int, int],
    upper_idx: list[int],
    crease_idx: list[int],
    lift: float,
) -> np.ndarray:
    """Polygon between the lash line and a lifted crease line (mobile lid)."""
    lash  = _points(lm, upper_idx)
    crease = _points(lm, crease_idx)

    # Push the crease line upward (toward the brow) by `lift`× eye-height so
    # the shadow extends onto the lid rather than only the lash line.
    # The crease landmarks already sit above the lash line; we add a modest
    # extra lift proportional to the eye height for a soft gradient band.
    eye_h = max(np.ptp(lash[:, 1]), 1.0)
    crease_lift = crease.copy()
    crease_lift[:, 1] -= lift * eye_h * 1.4

    # Build closed polygon: lash line (outer→inner) then crease (inner→outer).
    poly = np.vstack([lash, crease_lift[::-1]])
    return _poly_mask(shape, poly)


def apply_eyeshadow(
    image: np.ndarray,
    landmarks: np.ndarray,
    color: str = "brown",
    intensity: float = 0.55,
    lift: float = 0.42,
) -> tuple[np.ndarray, np.ndarray]:
    """Soft eyeshadow on both upper lids.

    Returns (image, mask).
    """
    color = _canon_color(color, "brown")
    bgr = _EYESHADOW_BGR.get(color, _EYESHADOW_BGR["brown"])
    h, w = image.shape[:2]
    fw = _face_width(lm := landmarks)

    left  = _eyelid_band(lm, (h, w), _LEFT_EYE_UPPER,  _LEFT_EYE_CREASE,  lift)
    right = _eyelid_band(lm, (h, w), _RIGHT_EYE_UPPER, _RIGHT_EYE_CREASE, lift)
    mask = np.clip(left + right, 0.0, 1.0)

    # Feather generously so it diffuses into the skin like real shadow.
    mask = _feather(mask, fw * 0.012)

    # Glitter / simli: add a sparkle layer on top of the colour.
    glitter = (color == "glitter")

    out = _apply_tint(
        image, mask, bgr,
        strength=float(np.clip(intensity, 0.0, 1.0)),
        preserve_luma=0.62 if not glitter else 0.5,
    )

    if glitter:
        out = _add_glitter(out, mask, intensity)

    return out, mask


def _add_glitter(image: np.ndarray, mask: np.ndarray, intensity: float) -> np.ndarray:
    """Sprinkle tiny bright specks within the masked lid region (shimmer)."""
    h, w = image.shape[:2]
    ys, xs = np.where(mask > 0.25)
    if xs.size == 0:
        return image
    out = image.copy()
    rng = np.random.default_rng(7)
    n = int(min(900, max(120, xs.size * 0.18)) * np.clip(intensity, 0.2, 1.0))
    pick = rng.choice(xs.size, size=min(n, xs.size), replace=False)
    speck = np.zeros((h, w), dtype=np.float32)
    for i in pick:
        x, y = int(xs[i]), int(ys[i])
        b = rng.uniform(0.4, 1.0)
        cv2.circle(speck, (x, y), rng.integers(0, 2) + 1, float(b), -1, cv2.LINE_AA)
    speck = cv2.GaussianBlur(speck, (0, 0), 0.6) * mask
    add = (speck[:, :, None] * np.array([245, 245, 255], np.float32)) * 0.8
    return np.clip(out.astype(np.float32) + add, 0, 255).astype(np.uint8)


# ===========================================================================
# BLUSH  (allık)
# ===========================================================================
def _cheek_mask(
    lm: np.ndarray,
    shape: tuple[int, int],
    center_idx: int,
    ring_idx: list[int],
    fw: float,
) -> np.ndarray:
    h, w = shape
    cx, cy = _pt(lm, center_idx)
    # Radius scales with face width; ring points give an organic apple shape.
    rx = max(6, int(fw * 0.11))
    ry = max(5, int(fw * 0.085))
    m = np.zeros((h, w), dtype=np.float32)
    cv2.ellipse(m, (int(cx), int(cy)), (rx, ry), 0, 0, 360, 1.0, -1, cv2.LINE_AA)
    return m


def apply_blush(
    image: np.ndarray,
    landmarks: np.ndarray,
    color: str = "pink",
    intensity: float = 0.30,
) -> tuple[np.ndarray, np.ndarray]:
    color = _canon_color(color, "pink")
    bgr = _BLUSH_BGR.get(color, _BLUSH_BGR["pink"])
    h, w = image.shape[:2]
    fw = _face_width(lm := landmarks)

    left  = _cheek_mask(lm, (h, w), _LEFT_CHEEK_CENTER,  _LEFT_CHEEK_RING,  fw)
    right = _cheek_mask(lm, (h, w), _RIGHT_CHEEK_CENTER, _RIGHT_CHEEK_RING, fw)
    mask = np.clip(left + right, 0.0, 1.0)

    # Very soft — blush is a diffuse flush, not a hard patch.
    mask = _feather(mask, fw * 0.05)

    out = _apply_tint(
        image, mask, bgr,
        strength=float(np.clip(intensity, 0.0, 0.8)),
        preserve_luma=0.72,   # keep lots of skin texture
    )
    return out, mask


# ===========================================================================
# LIPSTICK  (ruj)
# ===========================================================================
def apply_lipstick(
    image: np.ndarray,
    landmarks: np.ndarray,
    color: str = "red",
    intensity: float = 0.60,
    gloss: bool | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Colour the lips (outer ring minus inner mouth opening).

    ``gloss`` adds a wet specular highlight.  The "gloss"/"parlak" colour
    turns it on automatically; other colours can opt-in.
    """
    color = _canon_color(color, "red")
    bgr = _LIPSTICK_BGR.get(color, _LIPSTICK_BGR["red"])
    h, w = image.shape[:2]
    fw = _face_width(lm := landmarks)

    outer = _poly_mask((h, w), _points(lm, _LIPS_OUTER))
    inner = _poly_mask((h, w), _points(lm, _LIPS_INNER))
    mask = np.clip(outer - inner, 0.0, 1.0)
    mask = _feather(mask, fw * 0.006)

    is_gloss = (color == "gloss") if gloss is None else bool(gloss)

    out = _apply_tint(
        image, mask, bgr,
        strength=float(np.clip(intensity, 0.0, 1.0)),
        preserve_luma=0.5 if not is_gloss else 0.6,
    )

    if is_gloss:
        out = _add_lip_gloss(out, mask, lm, fw)

    return out, mask


def _add_lip_gloss(
    image: np.ndarray,
    mask: np.ndarray,
    lm: np.ndarray,
    fw: float,
) -> np.ndarray:
    """Add a soft specular highlight band on the lower lip for a wet look."""
    h, w = image.shape[:2]
    # Lower-lip centre ≈ landmark 17; place a small bright blob slightly above.
    cx, cy = _pt(lm, 17)
    cx2, cy2 = _pt(lm, 0)   # cupid's bow area for an upper subtle shine
    hl = np.zeros((h, w), dtype=np.float32)
    cv2.ellipse(hl, (int(cx), int(cy - fw * 0.012)),
                (max(2, int(fw * 0.045)), max(1, int(fw * 0.012))),
                0, 0, 360, 1.0, -1, cv2.LINE_AA)
    cv2.ellipse(hl, (int(cx2), int(cy2)),
                (max(2, int(fw * 0.02)), max(1, int(fw * 0.006))),
                0, 0, 360, 0.7, -1, cv2.LINE_AA)
    hl = cv2.GaussianBlur(hl, (0, 0), fw * 0.01) * mask
    add = hl[:, :, None] * np.array([255, 255, 255], np.float32) * 0.65
    return np.clip(image.astype(np.float32) + add, 0, 255).astype(np.uint8)


# ===========================================================================
# EYELINER  (eyeliner)
# ===========================================================================
def apply_eyeliner(
    image: np.ndarray,
    landmarks: np.ndarray,
    color: str = "black",
    intensity: float = 0.85,
    wing: float = 0.6,
) -> tuple[np.ndarray, np.ndarray]:
    """Thin dark line along the upper lash line, with a small outer wing."""
    color = _canon_color(color, "black")
    bgr = _EYESHADOW_BGR.get(color, _EYESHADOW_BGR["black"])
    h, w = image.shape[:2]
    fw = _face_width(lm := landmarks)
    thickness = max(1, int(round(fw * 0.006)))

    mask = np.zeros((h, w), dtype=np.float32)
    for upper, outer_idx, inner_idx in (
        (_LEFT_EYE_UPPER, 33, 133),
        (_RIGHT_EYE_UPPER, 263, 362),
    ):
        pts = _points(lm, upper)
        cv2.polylines(mask, [np.round(pts).astype(np.int32)],
                      False, 1.0, thickness, cv2.LINE_AA)
        # Wing: extend from the outer corner outward & slightly up.
        ox, oy = _pt(lm, outer_idx)
        ix, iy = _pt(lm, inner_idx)
        dirx, diry = ox - ix, oy - iy
        norm = math.hypot(dirx, diry) or 1.0
        wx = ox + (dirx / norm) * fw * 0.05 * wing
        wy = oy + (diry / norm) * fw * 0.05 * wing - fw * 0.02 * wing
        cv2.line(mask, (int(ox), int(oy)), (int(wx), int(wy)),
                 1.0, thickness, cv2.LINE_AA)

    mask = _feather(mask, 0.8)
    out = _apply_tint(image, mask, bgr,
                      strength=float(np.clip(intensity, 0.0, 1.0)),
                      preserve_luma=0.25)
    return out, mask


# ===========================================================================
# MASCARA / LASHES  (kirpik)
# ===========================================================================
def apply_mascara(
    image: np.ndarray,
    landmarks: np.ndarray,
    color: str = "black",
    intensity: float = 0.8,
    length: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw extended upper lashes (small upward strokes) for a mascara look."""
    color = _canon_color(color, "black")
    bgr = _EYESHADOW_BGR.get(color, _EYESHADOW_BGR["black"])
    h, w = image.shape[:2]
    fw = _face_width(lm := landmarks)
    base_len = fw * 0.035 * float(np.clip(length, 0.3, 2.0))

    mask = np.zeros((h, w), dtype=np.float32)
    for upper, outer_idx, inner_idx in (
        (_LEFT_EYE_UPPER, 33, 133),
        (_RIGHT_EYE_UPPER, 263, 362),
    ):
        pts = _points(lm, upper)
        ox, oy = _pt(lm, outer_idx)
        ix, iy = _pt(lm, inner_idx)
        # Outward direction of the eye (for fanning the lashes).
        ex, ey = ox - ix, oy - iy
        enorm = math.hypot(ex, ey) or 1.0
        ex, ey = ex / enorm, ey / enorm
        for i, (x, y) in enumerate(pts):
            t = i / max(len(pts) - 1, 1)
            # Longer toward the outer corner; lashes point up & slightly out.
            llen = base_len * (0.55 + 0.7 * t)
            ax = ex * 0.5 + 0.0
            ay = -1.0                      # upward in image coords
            an = math.hypot(ax, ay) or 1.0
            tx = x + (ax / an) * llen
            ty = y + (ay / an) * llen
            cv2.line(mask, (int(x), int(y)), (int(tx), int(ty)),
                     1.0, 1, cv2.LINE_AA)

    mask = _feather(mask, 0.7)
    out = _apply_tint(image, mask, bgr,
                      strength=float(np.clip(intensity, 0.0, 1.0)),
                      preserve_luma=0.2)
    return out, mask


# ===========================================================================
# High-level orchestrator
# ===========================================================================
# Default per-layer parameters. The application/UI overrides these.
DEFAULT_MAKEUP = {
    "eyeshadow": {"enabled": False, "color": "brown",  "intensity": 0.55},
    "blush":     {"enabled": False, "color": "pink",   "intensity": 0.30},
    "lipstick":  {"enabled": False, "color": "red",    "intensity": 0.60},
    "eyeliner":  {"enabled": False, "color": "black",  "intensity": 0.85},
    "mascara":   {"enabled": False, "color": "black",  "intensity": 0.80},
}

# Available colour choices, exported for building UIs (English + intent).
COLOR_CHOICES = {
    "eyeshadow": list(_EYESHADOW_BGR.keys()),   # blue black red green orange brown glitter
    "blush":     list(_BLUSH_BGR.keys()),       # pink brown
    "lipstick":  list(_LIPSTICK_BGR.keys()),    # red pink brown gloss
    "eyeliner":  ["black", "brown", "blue", "green"],
    "mascara":   ["black", "brown"],
}


def apply_makeup(
    image: np.ndarray,
    landmarks: np.ndarray,
    config: dict[str, dict[str, Any]] | None = None,
) -> MakeupResult:
    """Apply a full makeup look described by ``config``.

    ``config`` maps layer name → {"enabled": bool, "color": str,
    "intensity": float, ...}.  Missing layers fall back to DEFAULT_MAKEUP
    (disabled).  Order matters: blush → eyeshadow → eyeliner → mascara →
    lipstick gives the most natural compositing.

    Returns a :class:`MakeupResult`.  If no landmarks are supplied the
    original image is returned unchanged (so callers can pass ``None`` on
    frames where detection failed without special-casing).
    """
    if landmarks is None or len(landmarks) == 0:
        return MakeupResult(final_image=image.copy())

    cfg = {**{k: dict(v) for k, v in DEFAULT_MAKEUP.items()}}
    if config:
        for k, v in config.items():
            if k in cfg and isinstance(v, dict):
                cfg[k].update(v)
            elif isinstance(v, dict):
                cfg[k] = v

    out = image.copy()
    masks: dict[str, np.ndarray] = {}
    meta: dict[str, Any] = {}

    # 1) blush (under everything, on the cheeks)
    if cfg["blush"].get("enabled"):
        out, m = apply_blush(out, landmarks,
                             color=cfg["blush"].get("color", "pink"),
                             intensity=float(cfg["blush"].get("intensity", 0.30)))
        masks["blush"] = m
        meta["blush"] = {"color": _canon_color(cfg["blush"].get("color"), "pink")}

    # 2) eyeshadow
    if cfg["eyeshadow"].get("enabled"):
        out, m = apply_eyeshadow(out, landmarks,
                                color=cfg["eyeshadow"].get("color", "brown"),
                                intensity=float(cfg["eyeshadow"].get("intensity", 0.55)),
                                lift=float(cfg["eyeshadow"].get("lift", 0.42)))
        masks["eyeshadow"] = m
        meta["eyeshadow"] = {"color": _canon_color(cfg["eyeshadow"].get("color"), "brown")}

    # 3) eyeliner
    if cfg["eyeliner"].get("enabled"):
        out, m = apply_eyeliner(out, landmarks,
                               color=cfg["eyeliner"].get("color", "black"),
                               intensity=float(cfg["eyeliner"].get("intensity", 0.85)),
                               wing=float(cfg["eyeliner"].get("wing", 0.6)))
        masks["eyeliner"] = m
        meta["eyeliner"] = {"color": _canon_color(cfg["eyeliner"].get("color"), "black")}

    # 4) mascara / lashes (kirpik)
    if cfg["mascara"].get("enabled"):
        out, m = apply_mascara(out, landmarks,
                              color=cfg["mascara"].get("color", "black"),
                              intensity=float(cfg["mascara"].get("intensity", 0.80)),
                              length=float(cfg["mascara"].get("length", 1.0)))
        masks["mascara"] = m
        meta["mascara"] = {"color": _canon_color(cfg["mascara"].get("color"), "black")}

    # 5) lipstick (on top)
    if cfg["lipstick"].get("enabled"):
        out, m = apply_lipstick(out, landmarks,
                               color=cfg["lipstick"].get("color", "red"),
                               intensity=float(cfg["lipstick"].get("intensity", 0.60)),
                               gloss=cfg["lipstick"].get("gloss"))
        masks["lipstick"] = m
        meta["lipstick"] = {"color": _canon_color(cfg["lipstick"].get("color"), "red")}

    return MakeupResult(
        final_image=out,
        masks_used=masks,
        dsp_metadata={
            "layers": meta,
            "dsp_technique": "landmark-driven luminance-preserving HSV tinting",
            "pose_stable": True,
        },
    )


# ---------------------------------------------------------------------------
# Class wrapper (mirrors FaceOverlayEngine API style)
# ---------------------------------------------------------------------------
class MakeupEngine:
    """Stateless makeup applier bound to one image + landmark set."""

    def __init__(self, landmarks, image: np.ndarray) -> None:
        if isinstance(landmarks, np.ndarray):
            self._pixels = landmarks
        else:                              # FaceLandmarks-like object
            self._pixels = landmarks.pixels
        self.image = image

    def apply(self, config: dict[str, dict[str, Any]] | None = None) -> MakeupResult:
        return apply_makeup(self.image, self._pixels, config)
