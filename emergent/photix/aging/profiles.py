"""Configurable age-effect intensity profiles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class EffectRamp:
    """Smooth ramp mapping an age delta to one effect intensity."""

    max_intensity: float
    full_years: float
    start_years: float = 0.0
    power: float = 1.0

    def value(self, years: float) -> float:
        """Return an intensity in ``[0, 1]`` for the given age distance."""
        if years <= self.start_years:
            return 0.0
        denom = max(1e-6, self.full_years - self.start_years)
        t = np.clip((years - self.start_years) / denom, 0.0, 1.0)
        return float(np.clip(self.max_intensity * (t ** self.power), 0.0, 1.0))


@dataclass(frozen=True)
class AgingProfile:
    """Named set of effect ramps for aging and de-aging."""

    name: str
    aging: Mapping[str, EffectRamp]
    deaging: Mapping[str, EffectRamp]

    def intensities(self, age_delta: float, global_intensity: float = 1.0) -> dict[str, float]:
        """Map ``age_delta`` to per-effect intensities.

        Args:
            age_delta: Positive values age the face; negative values de-age it.
            global_intensity: Overall multiplier supplied by ``AgeTransformer``.

        Returns:
            Dict from effect name to intensity in ``[0, 1]``.
        """
        ramps = self.aging if age_delta >= 0.0 else self.deaging
        years = abs(float(age_delta))
        scale = float(np.clip(global_intensity, 0.0, 2.0))
        return {
            name: float(np.clip(ramp.value(years) * scale, 0.0, 1.0))
            for name, ramp in ramps.items()
        }


PROFILES: dict[str, AgingProfile] = {
    "natural": AgingProfile(
        name="natural",
        aging={
            "wrinkles": EffectRamp(0.84, 40.0),
            "roughening": EffectRamp(1.00, 40.0),
            "skin_tone": EffectRamp(0.96, 40.0),
            "hair": EffectRamp(0.68, 40.0),
            "eyes": EffectRamp(0.88, 40.0),
            "lips": EffectRamp(0.76, 40.0),
            "sagging": EffectRamp(0.36, 40.0),
            "smoothing": EffectRamp(0.0, 1.0),
        },
        deaging={
            "smoothing": EffectRamp(0.92, 25.0),
            "skin_tone": EffectRamp(0.70, 25.0),
            "hair": EffectRamp(0.42, 25.0),
            "eyes": EffectRamp(0.62, 25.0),
            "lips": EffectRamp(0.55, 25.0),
            "sagging": EffectRamp(0.18, 25.0),
            "wrinkles": EffectRamp(0.0, 1.0),
            "roughening": EffectRamp(0.0, 1.0),
        },
    ),
    "cinematic": AgingProfile(
        name="cinematic",
        aging={
            "wrinkles": EffectRamp(1.00, 35.0, power=0.9),
            "roughening": EffectRamp(1.00, 30.0),
            "skin_tone": EffectRamp(1.00, 35.0),
            "hair": EffectRamp(0.92, 35.0),
            "eyes": EffectRamp(1.00, 35.0),
            "lips": EffectRamp(0.90, 35.0),
            "sagging": EffectRamp(0.62, 40.0),
            "smoothing": EffectRamp(0.0, 1.0),
        },
        deaging={
            "smoothing": EffectRamp(1.00, 22.0),
            "skin_tone": EffectRamp(0.90, 22.0),
            "hair": EffectRamp(0.55, 22.0),
            "eyes": EffectRamp(0.78, 22.0),
            "lips": EffectRamp(0.72, 22.0),
            "sagging": EffectRamp(0.26, 25.0),
            "wrinkles": EffectRamp(0.0, 1.0),
            "roughening": EffectRamp(0.0, 1.0),
        },
    ),
    "subtle": AgingProfile(
        name="subtle",
        aging={
            "wrinkles": EffectRamp(0.45, 40.0),
            "roughening": EffectRamp(0.35, 40.0),
            "skin_tone": EffectRamp(0.52, 35.0),
            "hair": EffectRamp(0.25, 40.0),
            "eyes": EffectRamp(0.35, 40.0),
            "lips": EffectRamp(0.28, 40.0),
            "sagging": EffectRamp(0.12, 40.0),
            "smoothing": EffectRamp(0.0, 1.0),
        },
        deaging={
            "smoothing": EffectRamp(0.48, 25.0),
            "skin_tone": EffectRamp(0.42, 25.0),
            "hair": EffectRamp(0.20, 25.0),
            "eyes": EffectRamp(0.30, 25.0),
            "lips": EffectRamp(0.25, 25.0),
            "sagging": EffectRamp(0.08, 25.0),
            "wrinkles": EffectRamp(0.0, 1.0),
            "roughening": EffectRamp(0.0, 1.0),
        },
    ),
}


def get_profile(name: str = "natural") -> AgingProfile:
    """Return a preset profile by name."""
    if name not in PROFILES:
        raise ValueError(f"Unknown aging profile {name!r}; expected one of {sorted(PROFILES)}")
    return PROFILES[name]
