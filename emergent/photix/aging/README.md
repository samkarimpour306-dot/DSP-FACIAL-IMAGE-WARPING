# Photix Aging Transformer

`aging.transformer.AgeTransformer` is a classical-CV aging/de-aging pipeline built from composable effects. It does not download models or use a GAN.

## API

```python
from aging.transformer import AgeTransformer

transformer = AgeTransformer(intensity=1.0, preserve_identity=True, seed=7)
result = transformer.transform(image, landmarks, age_delta=20)
older = transformer.age(image, landmarks, years=20)
younger = transformer.deage(image, landmarks, years=15)
batch = transformer.transform_batch(images, landmarks_list, age_delta=10)
```

`AgingResult` contains:

- `image`: final BGR `uint8` image
- `intermediate_layers`: `texture_map`, `wrinkle_map`, `color_shift_map`, `warp_field`
- `region_masks`: soft masks for face, forehead, eye corners, nasolabial folds, cheeks, jaw, lips, under-eyes, and hair
- `effects_applied`: per-effect intensity values after profile scaling

## Pipeline

The transformer maps `age_delta` to effect intensities through an `AgingProfile`, then applies:

1. `GeometricSagging`
2. `SkinSmoothing` for de-aging
3. `WrinkleSynthesis`
4. `SkinTextureRoughening`
5. `SkinToneShift`
6. `HairColorShift`
7. `EyeAreaDarkening`
8. `LipDesaturation`
9. Optional guided identity blend

Low-resolution inputs are upscaled for processing and restored to the original size. Extreme deltas are clamped to `[-25, +40]`.

## Profiles

Preset profiles live in `aging/profiles.py`:

- `natural`: balanced default for still images
- `cinematic`: stronger texture, hair, and color shifts
- `subtle`: lower-amplitude changes for video sequences

For tuning, adjust each `EffectRamp(max_intensity, full_years, start_years, power)`. Lower `full_years` makes an effect ramp faster. Higher `power` delays the effect until larger deltas.

At `age_delta=+20`, the natural profile applies approximately:

- wrinkles `0.42`
- roughening `0.50`
- skin tone `0.48`
- hair `0.34`
- eyes `0.44`
- lips `0.38`
- sagging `0.18`

No extra requirements are added. Procedural wrinkle noise is implemented with deterministic cached fractal value noise using NumPy and OpenCV, avoiding an additional `noise` or `opensimplex` dependency.
