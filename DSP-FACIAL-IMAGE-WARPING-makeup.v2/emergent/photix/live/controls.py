"""
controls.py — live-mode runtime state + keyboard shortcut handler.

The live runtime state is split into:
  * a :class:`LivePipeline` (what to apply to each frame) — see
    :mod:`live.pipeline`.
  * a :class:`LiveState` (LivePipeline + ephemeral UI flags: quit,
    snapshot request, landmark overlay toggle).

Keyboard map. Two selection modes:
  * Exclusive (default): each feature key clears all other features so
    one effect is visible at a time.
  * Multi (press ``m`` to toggle): each feature key only changes its own
    effect, so you can stack as many as you want.

    1   cycle aging  : none → age → deage → none
    2   bump smile   : 0 → 0.3 → 0.6 → 0.9 → 0
    3   bump eyebrow : 0 → 0.3 → 0.6 → 0.9 → 0
    4   bump face-slim : 0 → 0.3 → 0.6 → 0.9 → 0
    5   cycle beard      : none → black → brown → blonde → none
    6   cycle sunglasses : none → classic → aviator → round → glasses → none
    7   cycle hair       : none → long → none
    8   toggle low-pass smoothing
    9   toggle high-pass sharpening
    0   reset everything (full passthrough)
    m   toggle multi-effect (stacking) mode

    + / =   aging sigma +10  (or last-bumped warp +0.1 if aging is off)
    - / _   aging sigma -10  (or last-bumped warp -0.1)

    Makeup (independent of the 1-9 effect slots — always stackable):
        e / E   eyeshadow toggle / cycle colour
                  (blue, black, red, green, orange, brown, glitter)
        b / B   blush     toggle / cycle colour  (pink, brown)
        r / R   lipstick  toggle / cycle colour  (red, pink, brown, gloss)
        n / N   eyeliner  toggle / cycle colour
        k / K   mascara   toggle / cycle colour  (kirpik)

    s   save snapshot
    l   toggle landmark overlay on the left panel
    q / Esc   quit
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .pipeline import LivePipeline


# Effect ids preserved for ui.py / Gradio (single-effect mode).
EFFECTS: dict[str, str] = {
    "none":       "No Effect",
    "age":        "Aging",
    "deage":      "De-Aging",
    "smile":      "Smile Warp",
    "eyebrow":    "Eyebrow Warp",
    "lip_wide":   "Lip-Wide Warp",
    "face_slim":  "Face Slim Warp",
    "surprise":   "Surprised Warp",
    "beard":      "Beard",
    "sunglasses": "Sunglasses",
    "hair":       "Hair",
}

_BEARD_COLORS:      tuple[str, ...] = ("none", "black", "brown", "blonde")
_SUNGLASSES_STYLES: tuple[str, ...] = ("none", "classic", "aviator", "round", "glasses")
_HAIR_STYLES:       tuple[str, ...] = ("none", "long")
_AGING_MODES:       tuple[str, ...] = ("none", "age", "deage")
_WARP_LEVELS:       tuple[float, ...] = (0.0, 0.3, 0.6, 0.9)

# Makeup colour cycles (English keys; engine also accepts Turkish synonyms).
_EYESHADOW_COLORS: tuple[str, ...] = ("blue", "black", "red", "green", "orange", "brown", "glitter")
_BLUSH_COLORS:     tuple[str, ...] = ("pink", "brown")
_LIPSTICK_COLORS:  tuple[str, ...] = ("red", "pink", "brown", "gloss")
_EYELINER_COLORS:  tuple[str, ...] = ("black", "brown", "blue", "green")
_MASCARA_COLORS:   tuple[str, ...] = ("black", "brown")


@dataclass
class LiveState:
    """Mutable runtime state for the webcam / video loops."""

    pipeline: LivePipeline = field(default_factory=LivePipeline)

    # Ephemeral runtime flags (not part of the saved pipeline config).
    show_landmarks:     bool = False
    request_screenshot: bool = False
    quit:               bool = False

    # When True, feature keys add to the pipeline without wiping siblings —
    # lets the user stack beard + sunglasses + hair + filters at once.
    # Default False preserves the historical "one effect at a time" UX.
    multi_mode:         bool = False

    # Track the most-recently bumped warp slider so +/- knows which to adjust.
    _last_warp: str = "smile"

    def label(self) -> str:
        base = self.pipeline.label()
        if self.multi_mode:
            return f"[MULTI] {base}" if base != "No Effect" else "[MULTI]"
        return base


# ── Cycle / bump helpers ─────────────────────────────────────────────────────

def _next(values: tuple, current) -> object:
    """Return the value after ``current`` in ``values`` (wraps)."""
    try:
        i = values.index(current)
    except ValueError:
        return values[0]
    return values[(i + 1) % len(values)]


def _next_warp_level(current: float) -> float:
    """Cycle a warp slider 0 → 0.3 → 0.6 → 0.9 → 0."""
    # Snap to the nearest level then step.
    closest = min(range(len(_WARP_LEVELS)),
                  key=lambda i: abs(_WARP_LEVELS[i] - current))
    return _WARP_LEVELS[(closest + 1) % len(_WARP_LEVELS)]


# ── Key dispatch ─────────────────────────────────────────────────────────────

def _clear_selected_features(p: LivePipeline) -> None:
    """Clear live features so the next key shows one selected effect."""
    p.smile = 0.0
    p.eyebrow = 0.0
    p.lip_wide = 0.0
    p.face_slim = 0.0
    p.aging = "none"
    p.beard = "none"
    p.sunglasses = "none"
    p.hair_style = "none"
    p.lowpass_enabled = False
    p.highpass_enabled = False


def handle_key(state: LiveState, key: int) -> bool:
    """Mutate ``state`` based on an OpenCV waitKey return code.

    Returns True if the key was recognised. ``key`` of -1 is a no-op.
    """
    if key == -1:
        return False

    k = key & 0xFF
    p = state.pipeline

    if k in (ord("q"), ord("Q"), 27):
        state.quit = True
        return True

    if k in (ord("s"), ord("S")):
        state.request_screenshot = True
        return True

    if k in (ord("l"), ord("L")):
        state.show_landmarks = not state.show_landmarks
        return True

    # ── m — toggle multi-effect / stacking mode ─────────────────────────
    if k in (ord("m"), ord("M")):
        state.multi_mode = not state.multi_mode
        return True

    # ── 0 — clear everything ────────────────────────────────────────────
    if k == ord("0"):
        state.pipeline = LivePipeline()
        return True

    # ── 1 — cycle aging mode ────────────────────────────────────────────
    if k == ord("1"):
        next_aging = str(_next(_AGING_MODES, p.aging))
        if not state.multi_mode:
            _clear_selected_features(p)
        p.aging = next_aging
        return True

    # ── 2/3/4 — bump warp sliders ───────────────────────────────────────
    if k == ord("2"):
        next_smile = _next_warp_level(p.smile)
        if not state.multi_mode:
            _clear_selected_features(p)
        p.smile = next_smile
        state._last_warp = "smile"
        return True
    if k == ord("3"):
        next_eyebrow = _next_warp_level(p.eyebrow)
        if not state.multi_mode:
            _clear_selected_features(p)
        p.eyebrow = next_eyebrow
        state._last_warp = "eyebrow"
        return True
    if k == ord("4"):
        next_slim = _next_warp_level(p.face_slim)
        if not state.multi_mode:
            _clear_selected_features(p)
        p.face_slim = next_slim
        state._last_warp = "face_slim"
        return True

    # ── 5/6/7 — cycle overlay style/colour ──────────────────────────────
    if k == ord("5"):
        next_beard = str(_next(_BEARD_COLORS, p.beard))
        if not state.multi_mode:
            _clear_selected_features(p)
        p.beard = next_beard
        return True
    if k == ord("6"):
        next_sunglasses = str(_next(_SUNGLASSES_STYLES, p.sunglasses))
        if not state.multi_mode:
            _clear_selected_features(p)
        p.sunglasses = next_sunglasses
        return True
    if k == ord("7"):
        next_hair = str(_next(_HAIR_STYLES, p.hair_style))
        if not state.multi_mode:
            _clear_selected_features(p)
        p.hair_style = next_hair
        return True

    # ── 8/9 — toggle frequency filters ──────────────────────────────────
    if k == ord("8"):
        next_lowpass = not p.lowpass_enabled
        if not state.multi_mode:
            _clear_selected_features(p)
        p.lowpass_enabled = next_lowpass
        if p.lowpass_enabled and p.lowpass_sigma < 1.0:
            p.lowpass_sigma = 12.0
        return True
    if k == ord("9"):
        next_highpass = not p.highpass_enabled
        if not state.multi_mode:
            _clear_selected_features(p)
        p.highpass_enabled = next_highpass
        if p.highpass_enabled and p.highpass_sigma < 1.0:
            p.highpass_sigma = 8.0
        return True

    # ── +/- — adjust aging sigma, or the most-recent warp ───────────────
    if k in (ord("+"), ord("=")):
        if p.aging != "none":
            p.aging_sigma = min(100.0, p.aging_sigma + 10.0)
        else:
            cur = getattr(p, state._last_warp, 0.0)
            setattr(p, state._last_warp, min(1.0, cur + 0.1))
        return True
    if k in (ord("-"), ord("_")):
        if p.aging != "none":
            p.aging_sigma = max(1.0, p.aging_sigma - 10.0)
        else:
            cur = getattr(p, state._last_warp, 0.0)
            setattr(p, state._last_warp, max(0.0, cur - 0.1))
        return True

    # ── Makeup toggles & colour cycles ───────────────────────────────────
    # Lowercase  = toggle the layer on/off.
    # Uppercase  = cycle to the next colour (also enables the layer).
    if k == ord("e"):
        p.makeup_eyeshadow_enabled = not p.makeup_eyeshadow_enabled
        return True
    if k == ord("E"):
        p.makeup_eyeshadow_color = str(_next(_EYESHADOW_COLORS, p.makeup_eyeshadow_color))
        p.makeup_eyeshadow_enabled = True
        return True
    if k == ord("b"):
        p.makeup_blush_enabled = not p.makeup_blush_enabled
        return True
    if k == ord("B"):
        p.makeup_blush_color = str(_next(_BLUSH_COLORS, p.makeup_blush_color))
        p.makeup_blush_enabled = True
        return True
    if k == ord("r"):
        p.makeup_lipstick_enabled = not p.makeup_lipstick_enabled
        return True
    if k == ord("R"):
        p.makeup_lipstick_color = str(_next(_LIPSTICK_COLORS, p.makeup_lipstick_color))
        p.makeup_lipstick_enabled = True
        return True
    if k == ord("n"):
        p.makeup_eyeliner_enabled = not p.makeup_eyeliner_enabled
        return True
    if k == ord("N"):
        p.makeup_eyeliner_color = str(_next(_EYELINER_COLORS, p.makeup_eyeliner_color))
        p.makeup_eyeliner_enabled = True
        return True
    if k == ord("k"):
        p.makeup_mascara_enabled = not p.makeup_mascara_enabled
        return True
    if k == ord("K"):
        p.makeup_mascara_color = str(_next(_MASCARA_COLORS, p.makeup_mascara_color))
        p.makeup_mascara_enabled = True
        return True

    return False
