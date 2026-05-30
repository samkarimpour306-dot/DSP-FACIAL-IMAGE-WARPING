"""Tests for the production aging transformer."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from aging.effects import (
    EyeAreaDarkening,
    GeometricSagging,
    HairColorShift,
    LipDesaturation,
    SkinSmoothing,
    SkinTextureRoughening,
    SkinToneShift,
    WrinkleSynthesis,
)
from aging.masks import build_region_masks
from aging.transformer import AgeTransformer
from landmarks.detector import FaceLandmarks


@pytest.fixture()
def synthetic_image() -> np.ndarray:
    img = np.full((256, 256, 3), 132, dtype=np.uint8)
    cv2.ellipse(img, (128, 132), (66, 82), 0, 0, 360, (164, 177, 198), -1)
    cv2.circle(img, (101, 112), 9, (34, 34, 38), -1)
    cv2.circle(img, (155, 112), 9, (34, 34, 38), -1)
    cv2.ellipse(img, (128, 160), (26, 8), 0, 0, 180, (76, 78, 130), 2)
    cv2.rectangle(img, (67, 20), (189, 72), (44, 48, 62), -1)
    return img


@pytest.fixture()
def synthetic_landmarks() -> FaceLandmarks:
    h = w = 256
    pixels = np.zeros((478, 2), dtype=np.float32)
    angles = np.linspace(0, 2 * np.pi, 478, endpoint=False)
    pixels[:, 0] = 128 + 62 * np.cos(angles)
    pixels[:, 1] = 132 + 78 * np.sin(angles)
    fixed = {
        10: (128, 50), 338: (145, 58), 297: (160, 66), 332: (172, 80),
        284: (184, 96), 251: (190, 118), 389: (188, 142), 356: (178, 165),
        454: (190, 132), 323: (184, 165), 361: (174, 186), 288: (158, 202),
        397: (145, 210), 365: (136, 214), 379: (128, 216), 378: (120, 214),
        400: (111, 210), 377: (98, 202), 152: (128, 212), 148: (98, 202),
        176: (82, 186), 149: (72, 165), 150: (66, 142), 136: (66, 132),
        172: (72, 112), 58: (84, 96), 132: (96, 80), 93: (110, 66),
        234: (66, 132), 127: (84, 86), 162: (100, 66), 21: (72, 118),
        54: (88, 96), 103: (102, 72), 67: (116, 58), 109: (128, 50),
        33: (91, 112), 7: (96, 119), 163: (103, 122), 144: (111, 119),
        145: (114, 112), 153: (111, 105), 154: (103, 102), 155: (96, 105),
        133: (115, 112), 173: (100, 104), 157: (105, 104), 158: (109, 106),
        159: (103, 103), 160: (97, 105), 161: (94, 108), 246: (90, 112),
        263: (141, 112), 249: (146, 119), 390: (153, 122), 373: (161, 119),
        374: (164, 112), 380: (161, 105), 381: (153, 102), 382: (146, 105),
        362: (165, 112), 398: (150, 104), 384: (155, 104), 385: (159, 106),
        386: (153, 103), 387: (147, 105), 388: (144, 108), 466: (140, 112),
        46: (88, 96), 53: (96, 91), 52: (104, 89), 65: (112, 91),
        55: (118, 96), 276: (138, 96), 283: (146, 91), 282: (154, 89),
        295: (162, 91), 285: (168, 96), 1: (128, 132), 4: (128, 144),
        168: (128, 96), 6: (128, 108), 197: (128, 120), 116: (88, 143),
        345: (168, 143), 186: (107, 145), 92: (111, 154), 165: (117, 162),
        167: (121, 169), 410: (149, 145), 322: (145, 154), 391: (139, 162),
        393: (135, 169), 61: (104, 160), 146: (110, 168), 91: (119, 172),
        181: (128, 174), 84: (137, 172), 17: (128, 176), 314: (137, 172),
        405: (146, 168), 321: (152, 160), 375: (148, 155), 291: (152, 160),
        409: (144, 153), 270: (136, 150), 269: (128, 149), 267: (120, 150),
        0: (128, 153), 37: (120, 150), 39: (112, 153), 40: (108, 155),
        185: (104, 160), 13: (128, 160), 14: (128, 168),
    }
    for idx, point in fixed.items():
        pixels[idx] = point
    normalized = np.column_stack([pixels[:, 0] / w, pixels[:, 1] / h, np.zeros(478)])
    return FaceLandmarks(
        normalized=normalized.astype(np.float32),
        pixels=pixels,
        image_size=(h, w),
        bbox=(66, 50, 124, 166),
    )


def test_identity_at_zero_delta(synthetic_image, synthetic_landmarks):
    transformer = AgeTransformer(seed=7)
    result = transformer.transform(synthetic_image, synthetic_landmarks, 0)
    assert result.image.shape == synthetic_image.shape
    assert np.mean(np.abs(result.image.astype(np.int16) - synthetic_image.astype(np.int16))) < 0.5


def test_aging_then_deaging_is_approximately_reversible(synthetic_image, synthetic_landmarks):
    transformer = AgeTransformer(seed=7, preserve_identity=True)
    aged = transformer.age(synthetic_image, synthetic_landmarks, 20).image
    restored = transformer.deage(aged, synthetic_landmarks, 20).image
    assert np.mean(np.abs(restored.astype(np.int16) - synthetic_image.astype(np.int16))) < 42.0


@pytest.mark.parametrize(
    "effect",
    [
        WrinkleSynthesis(seed=1),
        SkinTextureRoughening(seed=1),
        SkinSmoothing(),
        SkinToneShift(1),
        HairColorShift(1),
        EyeAreaDarkening(1),
        LipDesaturation(1),
        GeometricSagging(1),
    ],
)
def test_effect_zero_intensity_is_identity(effect, synthetic_image, synthetic_landmarks):
    masks = build_region_masks(synthetic_image.shape[:2], synthetic_landmarks)
    out = effect.apply(synthetic_image, synthetic_landmarks, masks["face"], 0.0)
    np.testing.assert_array_equal(out, synthetic_image)


def test_region_masks_shape_and_range(synthetic_image, synthetic_landmarks):
    masks = build_region_masks(synthetic_image.shape[:2], synthetic_landmarks)
    expected = {
        "forehead", "eye_corners", "nasolabial", "cheeks",
        "jaw", "lips", "under_eyes", "hair",
    }
    assert expected <= set(masks)
    for mask in masks.values():
        assert mask.shape == synthetic_image.shape[:2]
        assert mask.dtype == np.float32
        assert float(mask.min()) >= 0.0
        assert float(mask.max()) <= 1.0


def test_batch_matches_single_frame(synthetic_image, synthetic_landmarks):
    transformer = AgeTransformer(seed=9, preserve_identity=False)
    single = transformer.transform(synthetic_image, synthetic_landmarks, 15).image
    batch = transformer.transform_batch([synthetic_image], [synthetic_landmarks], 15)[0].image
    np.testing.assert_array_equal(single, batch)


@pytest.mark.parametrize("delta", [-200, 0, 300])
def test_extreme_values_do_not_produce_nan(delta, synthetic_image, synthetic_landmarks):
    transformer = AgeTransformer(seed=3)
    result = transformer.transform(synthetic_image, synthetic_landmarks, delta)
    assert result.image.dtype == np.uint8
    assert np.isfinite(result.image.astype(np.float32)).all()
    for layer in result.intermediate_layers.values():
        assert np.isfinite(layer).all()
