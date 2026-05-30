from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from live import pipeline as live_pipeline
from live.controls import LiveState, handle_key
from live.pipeline import LivePipeline, apply_pipeline


def test_frequency_filters_run_without_landmarks():
    rng = np.random.default_rng(7)
    img = rng.integers(0, 256, (96, 96, 3), dtype=np.uint8)
    pipe = LivePipeline(lowpass_enabled=True, lowpass_sigma=8.0)

    out = apply_pipeline(img, pipe, landmarks=None)

    assert out.shape == img.shape
    assert out.dtype == np.uint8
    assert not np.array_equal(out, img)


def test_live_overlays_stack_glasses_beard_and_hair(monkeypatch):
    calls = {}

    class FakeFaceOverlayEngine:
        def __init__(self, landmarks, image):
            self.landmarks = landmarks
            self.image = image

        def apply(self, effects, params):
            calls["effects"] = effects
            calls["params"] = params
            return SimpleNamespace(final_image=self.image + 7)

    monkeypatch.setattr(live_pipeline, "FaceOverlayEngine", FakeFaceOverlayEngine)

    img = np.zeros((64, 64, 3), dtype=np.uint8)
    landmarks = np.zeros((478, 2), dtype=np.float32)
    pipe = LivePipeline(
        beard="brown",
        sunglasses="glasses",
        hair_style="long",
        hair_color="black",
    )

    out = apply_pipeline(img, pipe, landmarks)

    assert calls["effects"] == ["hair", "sunglasses", "beard"]
    assert calls["params"] == {
        "hair": {"style": "long", "color": "black"},
        "sunglasses": {"style": "glasses"},
        "beard": {"color": "brown"},
    }
    assert np.all(out == 7)


def test_live_keyboard_selects_one_feature_at_a_time():
    state = LiveState()

    assert handle_key(state, ord("5"))
    assert state.pipeline.beard == "black"
    assert state.pipeline.sunglasses == "none"
    assert state.pipeline.hair_style == "none"

    assert handle_key(state, ord("6"))
    assert state.pipeline.beard == "none"
    assert state.pipeline.sunglasses == "classic"
    assert state.pipeline.hair_style == "none"

    assert handle_key(state, ord("7"))
    assert state.pipeline.beard == "none"
    assert state.pipeline.sunglasses == "none"
    assert state.pipeline.hair_style == "long"


def test_live_keyboard_multi_mode_stacks_features():
    """Pressing `m` enables stacking — feature keys add without wiping."""
    state = LiveState()
    assert not state.multi_mode

    assert handle_key(state, ord("m"))      # toggle stacking ON
    assert state.multi_mode

    assert handle_key(state, ord("5"))      # beard cycles to black
    assert handle_key(state, ord("6"))      # sunglasses cycles to classic
    assert handle_key(state, ord("7"))      # hair cycles to long
    assert handle_key(state, ord("8"))      # low-pass on
    assert handle_key(state, ord("2"))      # smile bumps to 0.3

    p = state.pipeline
    assert p.beard == "black"
    assert p.sunglasses == "classic"
    assert p.hair_style == "long"
    assert p.lowpass_enabled is True
    assert p.smile == 0.3

    # Toggling stacking OFF then pressing a key should clear siblings again.
    assert handle_key(state, ord("m"))
    assert not state.multi_mode
    assert handle_key(state, ord("5"))      # beard cycles black -> brown,
                                            # everything else cleared
    assert state.pipeline.beard == "brown"
    assert state.pipeline.sunglasses == "none"
    assert state.pipeline.hair_style == "none"
    assert state.pipeline.lowpass_enabled is False
    assert state.pipeline.smile == 0.0


def test_live_state_label_marks_multi_mode():
    state = LiveState()
    assert state.label() == "No Effect"
    state.multi_mode = True
    assert "MULTI" in state.label()
    state.pipeline.beard = "black"
    assert "MULTI" in state.label() and "Beard:black" in state.label()
