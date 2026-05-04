"""High-level aging and de-aging transformer."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, Sequence

import cv2
import numpy as np

from aging.effects import (
    BaseAgingEffect,
    EyeAreaDarkening,
    GeometricSagging,
    HairColorShift,
    LipDesaturation,
    SkinSmoothing,
    SkinTextureRoughening,
    SkinToneShift,
    WrinkleSynthesis,
)
from aging.identity import guided_identity_blend
from aging.masks import build_region_masks
from aging.profiles import AgingProfile, get_profile
from landmarks.detector import FaceLandmarks

log = logging.getLogger(__name__)

_MIN_PROCESS_DIM = 256
_AGE_MIN = -25.0
_AGE_MAX = 40.0


@dataclass
class AgingResult:
    """Result bundle returned by :class:`AgeTransformer`.

    Attributes:
        image: Final transformed BGR uint8 image.
        intermediate_layers: Debug layers including texture, wrinkle, color, and warp maps.
        region_masks: Soft masks used by the effect pipeline.
        effects_applied: Per-effect intensity values after profile and global scaling.
    """

    image: np.ndarray
    intermediate_layers: dict[str, np.ndarray]
    region_masks: dict[str, np.ndarray]
    effects_applied: dict[str, float]


class AgeTransformer:
    """Production classical-CV aging and de-aging transformer."""

    def __init__(
        self,
        intensity: float = 1.0,
        preserve_identity: bool = True,
        seed: int | None = None,
        profile: str | AgingProfile = "natural",
    ) -> None:
        """Initialise the transformer.

        Args:
            intensity: Global intensity multiplier. Values are clipped to ``[0, 2]``.
            preserve_identity: Whether to run the final identity-preserving blend.
            seed: Seed for deterministic procedural texture generation.
            profile: Preset profile name or an ``AgingProfile`` instance.
        """
        self.intensity = float(np.clip(intensity, 0.0, 2.0))
        self.preserve_identity = bool(preserve_identity)
        self.seed = seed
        self.profile = get_profile(profile) if isinstance(profile, str) else profile

    def transform(self, image: np.ndarray, landmarks: FaceLandmarks, age_delta: float) -> AgingResult:
        """Transform a face image by a target age delta.

        Args:
            image: Input BGR/RGB-like uint8 image array, shape ``(H, W, 3)``.
            landmarks: Face landmarks in the same image coordinate space.
            age_delta: Continuous age delta in years; negative de-ages, positive ages.

        Returns:
            ``AgingResult`` with the transformed image and debug layers.
        """
        if not isinstance(image, np.ndarray):
            raise TypeError(f"image must be a numpy array, got {type(image).__name__}")
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("image must have shape (H, W, 3)")

        original_delta = float(age_delta)
        clamped_delta = float(np.clip(original_delta, _AGE_MIN, _AGE_MAX))
        if clamped_delta != original_delta:
            log.warning("age_delta %.2f clamped to %.2f", original_delta, clamped_delta)

        source = np.nan_to_num(image, copy=True).astype(np.uint8)
        proc_image, proc_landmarks, scale = self._prepare_resolution(source, landmarks)
        if abs(clamped_delta) < 1e-6:
            masks = build_region_masks(proc_image.shape[:2], proc_landmarks)
            final = self._restore_resolution(proc_image.copy(), source.shape[:2], scale)
            return AgingResult(
                image=final,
                intermediate_layers=self._empty_layers(source.shape[:2]),
                region_masks=self._resize_masks(masks, source.shape[:2]) if scale != 1.0 else masks,
                effects_applied={name: 0.0 for name in self.profile.aging},
            )

        direction = 1 if clamped_delta > 0.0 else -1
        intensities = self.profile.intensities(clamped_delta, self.intensity)
        masks = build_region_masks(proc_image.shape[:2], proc_landmarks)
        effects = self._build_effects(direction)

        current = proc_image.copy()
        before_texture = current.copy()
        before_wrinkle = current.copy()
        before_color = current.copy()
        warp_field = np.zeros((*proc_image.shape[:2], 2), dtype=np.float32)

        for effect in effects:
            amount = intensities.get(effect.name, 0.0)
            effect_mask = self._mask_for_effect(effect.name, masks)
            if effect.name == "wrinkles":
                before_wrinkle = current.copy()
            if effect.name in {"roughening", "smoothing"}:
                before_texture = current.copy()
            if effect.name in {"skin_tone", "hair", "eyes", "lips"}:
                before_color = current.copy()
            current = effect.apply(current, proc_landmarks, effect_mask, amount)
            if isinstance(effect, GeometricSagging) and effect.last_warp_field is not None:
                warp_field = effect.last_warp_field

        if self.preserve_identity:
            current = guided_identity_blend(proc_image, current, masks, strength=0.85)

        layers = {
            "texture_map": self._absdiff_gray(before_texture, current),
            "wrinkle_map": self._absdiff_gray(before_wrinkle, current),
            "color_shift_map": self._absdiff_gray(before_color, current),
            "warp_field": warp_field.astype(np.float32),
        }

        final = self._restore_resolution(current, source.shape[:2], scale)
        if scale != 1.0:
            layers = self._resize_layers(layers, source.shape[:2])
            masks = self._resize_masks(masks, source.shape[:2])

        final = np.nan_to_num(np.clip(final, 0, 255), copy=False).astype(np.uint8)
        return AgingResult(
            image=final,
            intermediate_layers=layers,
            region_masks=masks,
            effects_applied=intensities,
        )

    def age(self, image: np.ndarray, landmarks: FaceLandmarks, years: float = 20) -> AgingResult:
        """Age a face by ``years``.

        Args:
            image: Input image.
            landmarks: Face landmarks for the image.
            years: Positive number of years to add.

        Returns:
            ``AgingResult`` for the aged face.
        """
        return self.transform(image, landmarks, abs(float(years)))

    def deage(self, image: np.ndarray, landmarks: FaceLandmarks, years: float = 15) -> AgingResult:
        """De-age a face by ``years``.

        Args:
            image: Input image.
            landmarks: Face landmarks for the image.
            years: Positive number of years to remove.

        Returns:
            ``AgingResult`` for the de-aged face.
        """
        return self.transform(image, landmarks, -abs(float(years)))

    def transform_batch(
        self,
        images: Sequence[np.ndarray],
        landmarks_list: Sequence[FaceLandmarks],
        age_delta: float,
    ) -> list[AgingResult]:
        """Transform a sequence of images, suitable for video frame batches.

        Args:
            images: Sequence of image arrays.
            landmarks_list: Sequence of matching landmark objects.
            age_delta: Age delta applied to every frame.

        Returns:
            List of ``AgingResult`` objects in input order.
        """
        if len(images) != len(landmarks_list):
            raise ValueError("images and landmarks_list must have the same length")
        return [self.transform(img, lm, age_delta) for img, lm in zip(images, landmarks_list)]

    def _build_effects(self, direction: int) -> list[BaseAgingEffect]:
        return [
            GeometricSagging(direction),
            SkinSmoothing(),
            WrinkleSynthesis(self.seed),
            SkinTextureRoughening(self.seed),
            SkinToneShift(direction),
            HairColorShift(direction),
            EyeAreaDarkening(direction),
            LipDesaturation(direction),
        ]

    @staticmethod
    def _mask_for_effect(name: str, masks: dict[str, np.ndarray]) -> np.ndarray:
        if name == "wrinkles":
            return np.clip(
                masks["forehead"] + masks["eye_corners"] + masks["nasolabial"] + masks["lips"],
                0.0,
                1.0,
            )
        if name in {"roughening", "smoothing", "skin_tone", "sagging"}:
            return masks["face"]
        if name == "hair":
            return masks["hair"]
        if name == "eyes":
            return masks["under_eyes"]
        if name == "lips":
            return masks["lips"]
        return masks["face"]

    @staticmethod
    def _absdiff_gray(before: np.ndarray, after: np.ndarray) -> np.ndarray:
        diff = cv2.absdiff(before, after)
        return cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    @staticmethod
    def _empty_layers(shape: tuple[int, int]) -> dict[str, np.ndarray]:
        h, w = shape
        return {
            "texture_map": np.zeros((h, w), dtype=np.float32),
            "wrinkle_map": np.zeros((h, w), dtype=np.float32),
            "color_shift_map": np.zeros((h, w), dtype=np.float32),
            "warp_field": np.zeros((h, w, 2), dtype=np.float32),
        }

    @staticmethod
    def _prepare_resolution(
        image: np.ndarray,
        landmarks: FaceLandmarks,
    ) -> tuple[np.ndarray, FaceLandmarks, float]:
        h, w = image.shape[:2]
        min_dim = min(h, w)
        if min_dim >= _MIN_PROCESS_DIM:
            return image, landmarks, 1.0
        scale = _MIN_PROCESS_DIM / max(1, min_dim)
        new_size = (int(round(w * scale)), int(round(h * scale)))
        log.warning("Low-resolution aging input %dx%d upscaled to %dx%d", w, h, *new_size)
        up = cv2.resize(image, new_size, interpolation=cv2.INTER_CUBIC)
        lm = FaceLandmarks(
            normalized=landmarks.normalized.copy(),
            pixels=(landmarks.pixels * scale).astype(np.float32),
            image_size=(up.shape[0], up.shape[1]),
            bbox=tuple(int(round(v * scale)) for v in landmarks.bbox),
            yaw=landmarks.yaw,
            pitch=landmarks.pitch,
            roll=landmarks.roll,
        )
        return up, lm, scale

    @staticmethod
    def _restore_resolution(image: np.ndarray, original_shape: tuple[int, int], scale: float) -> np.ndarray:
        if scale == 1.0:
            return image
        h, w = original_shape
        return cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)

    @staticmethod
    def _resize_masks(masks: dict[str, np.ndarray], shape: tuple[int, int]) -> dict[str, np.ndarray]:
        h, w = shape
        return {
            name: cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_AREA)
            for name, mask in masks.items()
        }

    @staticmethod
    def _resize_layers(layers: dict[str, np.ndarray], shape: tuple[int, int]) -> dict[str, np.ndarray]:
        h, w = shape
        out: dict[str, np.ndarray] = {}
        for name, layer in layers.items():
            if layer.ndim == 3:
                out[name] = cv2.resize(layer.astype(np.float32), (w, h), interpolation=cv2.INTER_AREA)
            else:
                out[name] = cv2.resize(layer.astype(np.float32), (w, h), interpolation=cv2.INTER_AREA)
        return out
