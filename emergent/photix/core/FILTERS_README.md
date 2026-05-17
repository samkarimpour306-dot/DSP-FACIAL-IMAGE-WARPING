# Filter Module — DSP Filter Implementations

This module provides comprehensive **frequency-domain** and **spatial-domain** filtering functions for image processing in the Photix DSP project.

---

## Overview

### Frequency-Domain Filters (FFT-based)

Implemented using **2D Fast Fourier Transform (FFT)** for optimal performance with large kernels:

| Filter | Function | Use Case |
|--------|----------|----------|
| **Gaussian Low-Pass** | `apply_lowpass_filter()` | Smoothing, noise reduction, skin softening |
| **Gaussian High-Pass** | `apply_highpass_filter()` | Edge enhancement, texture sharpening, aging effects |
| **Butterworth Low-Pass** | `apply_lowpass_filter(..., filter_type=FilterType.BUTTERWORTH)` | Smoother transitions than Gaussian |
| **Butterworth High-Pass** | `apply_highpass_filter(..., filter_type=FilterType.BUTTERWORTH)` | Sharper edge enhancement than Gaussian |
| **Ideal Low-Pass** | `apply_lowpass_filter(..., filter_type=FilterType.IDEAL)` | Theoretical reference (sharp cutoff) |
| **Ideal High-Pass** | `apply_highpass_filter(..., filter_type=FilterType.IDEAL)` | Theoretical reference (sharp cutoff) |
| **Band-Pass** | `apply_bandpass_filter()` | Isolate specific frequency ranges |
| **Band-Stop (Notch)** | `apply_bandstop_filter()` | Remove specific frequency ranges (e.g., powerline noise) |

### Spatial-Domain Filters

Fast filters for small kernels:

| Filter | Function | Use Case |
|--------|----------|----------|
| **Gaussian Blur** | `apply_spatial_lowpass()` | Fast smoothing for real-time applications |
| **Laplacian** | `apply_spatial_highpass()` | Edge detection, zero-crossing detection |
| **Unsharp Mask** | `apply_unsharp_mask()` | Selective sharpening with controlled strength |

---

## API Reference

### Frequency-Domain Low-Pass Filter

```python
from core import apply_lowpass_filter, FilterType

# Apply Gaussian low-pass (sigma-based)
filtered = apply_lowpass_filter(
    img,
    sigma=15.0,              # Standard deviation (larger = more blur)
    filter_type=FilterType.GAUSSIAN
)

# Apply Butterworth low-pass (smoother transition)
filtered = apply_lowpass_filter(
    img,
    sigma=20.0,              # Cutoff frequency
    filter_type=FilterType.BUTTERWORTH,
    order=2                  # Sharpness of cutoff (higher = sharper)
)

# Apply Ideal low-pass (brick-wall, theoretical)
filtered = apply_lowpass_filter(
    img,
    sigma=25.0,              # Cutoff radius
    filter_type=FilterType.IDEAL
)
```

**Parameters:**
- `img` (np.ndarray): Input image (BGR or grayscale)
- `sigma` (float): Filter parameter; interpretation depends on `filter_type`
- `filter_type` (FilterType): `GAUSSIAN`, `BUTTERWORTH`, or `IDEAL`
- `**kwargs`: Additional parameters (e.g., `order=2` for Butterworth)

**Returns:**
- Filtered image (same dtype and shape as input)

**Performance:**
- Gaussian LP (512×512, RGB): ~15–20 ms
- Ideal LP (512×512, RGB): ~15–20 ms
- Butterworth LP (512×512, RGB): ~15–20 ms

---

### Frequency-Domain High-Pass Filter

```python
from core import apply_highpass_filter, FilterType

# Apply Gaussian high-pass
enhanced = apply_highpass_filter(
    img,
    sigma=15.0,
    filter_type=FilterType.GAUSSIAN
)

# Apply Butterworth high-pass
enhanced = apply_highpass_filter(
    img,
    sigma=20.0,
    filter_type=FilterType.BUTTERWORTH,
    order=2
)
```

**Parameters:** Same as `apply_lowpass_filter()`

**Returns:** High-pass filtered image (result is centered at 128 for visibility)

**Performance:** Same as low-pass filters

---

### Band-Pass Filter

```python
from core import apply_bandpass_filter

# Isolate mid-range frequencies (e.g., facial features)
midrange = apply_bandpass_filter(
    img,
    low_freq=10.0,           # Lower cutoff
    high_freq=30.0,          # Upper cutoff
    filter_type=FilterType.GAUSSIAN
)
```

**Use Cases:**
- Extract texture at specific scales
- Enhance specific facial features
- Frequency-domain analysis

---

### Band-Stop (Notch) Filter

```python
from core import apply_bandstop_filter

# Remove specific frequency range (e.g., powerline noise at 50 Hz)
denoised = apply_bandstop_filter(
    img,
    low_freq=48.0,
    high_freq=52.0,
    filter_type=FilterType.GAUSSIAN
)
```

---

### Spatial-Domain Filters

#### Gaussian Blur (Spatial)

```python
from core import apply_spatial_lowpass

blurred = apply_spatial_lowpass(
    img,
    kernel_size=5,           # Must be odd (auto-corrected)
    sigma=1.0                # Standard deviation
)
```

**Performance:**
- Spatial LP (512×512, kernel=5): ~2–3 ms (much faster than FFT for small kernels)

---

#### Laplacian (Spatial High-Pass)

```python
from core import apply_spatial_highpass

edges = apply_spatial_highpass(
    img,
    kernel_size=5            # Typical: 3, 5, or 7
)
```

**Performance:**
- Spatial HP (512×512, kernel=5): ~1–2 ms

---

#### Unsharp Mask

```python
from core import apply_unsharp_mask

sharpened = apply_unsharp_mask(
    img,
    kernel_size=5,           # Blur kernel size
    sigma=1.0,               # Blur sigma
    strength=1.5             # Sharpening intensity (0.5–2.0 typical)
)
```

**Formula:**
```
Output = Original + strength × (Original - Blur)
```

**Use Cases:**
- Selective sharpening for facial features
- Texture enhancement for aging effects
- High-frequency detail preservation

---

## Practical Examples

### Example 1: Image Smoothing + Sharpening Pipeline

```python
import cv2
from core import apply_lowpass_filter, apply_unsharp_mask, FilterType

# Load image
img = cv2.imread('face.jpg')

# Smooth skin while preserving edges
smooth = apply_lowpass_filter(img, sigma=20.0, filter_type=FilterType.GAUSSIAN)

# Enhance facial features
enhanced = apply_unsharp_mask(smooth, kernel_size=5, sigma=1.0, strength=1.2)

cv2.imshow('Original', img)
cv2.imshow('Smoothed', smooth)
cv2.imshow('Enhanced', enhanced)
cv2.waitKey(0)
```

### Example 2: Aging Effect using High-Pass Filter

```python
from core import apply_highpass_filter, FilterType

# Apply Gaussian high-pass to enhance wrinkles/texture
aged = apply_highpass_filter(img, sigma=10.0, filter_type=FilterType.GAUSSIAN)

# Blend with original
result = cv2.addWeighted(img, 0.7, aged, 0.3, 0)
```

### Example 3: De-Aging using Low-Pass Filter

```python
from core import apply_lowpass_filter, FilterType

# Apply Gaussian low-pass to smooth skin
deaged = apply_lowpass_filter(img, sigma=25.0, filter_type=FilterType.GAUSSIAN)

# Blend with original (face area only)
result = cv2.addWeighted(img, 0.8, deaged, 0.2, 0)
```

### Example 4: Multi-Scale Texture Analysis

```python
from core import apply_bandpass_filter, FilterType

# Extract coarse texture (low frequencies)
coarse = apply_bandpass_filter(img, low_freq=5.0, high_freq=15.0)

# Extract fine texture (high frequencies)
fine = apply_bandpass_filter(img, low_freq=20.0, high_freq=40.0)

# Extract very fine texture (very high frequencies)
detail = apply_bandpass_filter(img, low_freq=45.0, high_freq=70.0)
```

---

## Filter Comparison

### Gaussian vs. Butterworth vs. Ideal

| Aspect | Gaussian | Butterworth | Ideal |
|--------|----------|-------------|-------|
| **Smoothness** | Very smooth | Smooth | Brick-wall (sharp) |
| **Ringing** | None | Minimal | Severe (Gibbs phenomenon) |
| **Natural Look** | Best | Good | Unnatural artifacts |
| **Computational Cost** | ~20ms | ~20ms | ~20ms |
| **Use Case** | General purpose | Controlled transition | Theoretical reference |

**Recommendation:** Use **Gaussian** for most applications; use **Butterworth** for more control over frequency transition sharpness.

---

## Performance Notes

### FFT-Based Filters (Frequency Domain)

| Image Size | Time (ms) |
|------------|-----------|
| 256×256 RGB | ~3–5 |
| 512×512 RGB | ~15–20 |
| 1024×1024 RGB | ~60–80 |

**Characteristics:**
- Independent of kernel size
- Excellent for large kernels
- Small overhead per image

### Spatial Filters

| Filter | 512×512 RGB | Notes |
|--------|------------|-------|
| Gaussian Blur (k=5) | 2–3 ms | Very fast for small kernels |
| Laplacian (k=5) | 1–2 ms | Edge detection |
| Unsharp Mask | 5–7 ms | Blur + subtraction |

**Rule of Thumb:**
- **FFT filters**: Use when `sigma > 10` (large blur)
- **Spatial filters**: Use for real-time (`kernel_size ≤ 7`)

---

## Testing

Run the comprehensive test suite:

```bash
cd photix/
python tests/test_filters.py
```

Tests include:
- Gaussian, Butterworth, and Ideal filters (LP, HP)
- Band-pass and band-stop filters
- Spatial-domain filters
- Grayscale and RGB input
- Edge cases (empty images, invalid parameters)
- Performance benchmarking

---

## Integration with Aging Effects

The filters module seamlessly integrates with the existing `aging_filter.py`:

```python
from core.aging_filter import apply_aging, apply_deaging
from core.filters import apply_highpass_filter, FilterType

# Aging uses Gaussian high-pass internally
aged = apply_aging(img, sigma=30.0)

# De-aging uses Gaussian low-pass internally
deaged = apply_deaging(img, sigma=30.0)

# Can also apply filters independently
custom_aged = apply_highpass_filter(img, sigma=8.0, filter_type=FilterType.GAUSSIAN)
```

---

## Design Notes

### Why FFT-Based Filtering?

1. **Kernel Size Independence**: 512×512 σ=15 and σ=100 take ~same time
2. **Precision**: Double-precision complex arithmetic
3. **Flexibility**: Easy to combine multiple filters in frequency domain
4. **Standards**: Aligns with DSP literature and theory

### Frequency Domain Interpretation

- **Low frequencies** (center): Overall brightness and large shapes
- **Mid frequencies**: Facial features, contours
- **High frequencies** (edges): Texture, fine details, noise

For a 512×512 image:
- Center pixel (0,0) in frequency domain = DC (average brightness)
- Radius 0–50 pixels = Low frequencies (smoothing)
- Radius 50–150 pixels = Mid frequencies (features)
- Radius 150+ pixels = High frequencies (texture, noise)

---

## References

- Gonzalez & Woods: "Digital Image Processing" (3rd ed.)
- FFT algorithm: Cooley-Tukey, O(n log n)
- Butterworth filter design: standard digital signal processing
- Unsharp masking: classical image enhancement technique

---

## Future Enhancements

- [ ] Morphological filters (opening, closing, gradient)
- [ ] Wavelet-based filtering
- [ ] Directional filters (Gabor, gradient orientation)
- [ ] Adaptive filters (bilateral, non-local means)
- [ ] GPU acceleration (CUDA/OpenCL)
- [ ] Filter cascade optimization

