# Filter Module — Quick Reference Card

## Import

```python
from core.filters import (
    FilterType,
    apply_lowpass_filter,
    apply_highpass_filter,
    apply_bandpass_filter,
    apply_bandstop_filter,
    apply_spatial_lowpass,
    apply_spatial_highpass,
    apply_unsharp_mask,
)
```

---

## Frequency-Domain Filters (FFT-Based)

### Low-Pass (Smoothing)

```python
# Gaussian (best for most uses)
smooth = apply_lowpass_filter(img, sigma=20.0, filter_type=FilterType.GAUSSIAN)

# Butterworth (smoother transition)
smooth = apply_lowpass_filter(img, sigma=20.0, filter_type=FilterType.BUTTERWORTH, order=2)

# Ideal (sharp cutoff)
smooth = apply_lowpass_filter(img, sigma=20.0, filter_type=FilterType.IDEAL)
```

### High-Pass (Enhancement)

```python
# Gaussian
enhanced = apply_highpass_filter(img, sigma=15.0, filter_type=FilterType.GAUSSIAN)

# Butterworth
enhanced = apply_highpass_filter(img, sigma=15.0, filter_type=FilterType.BUTTERWORTH, order=2)

# Ideal
enhanced = apply_highpass_filter(img, sigma=15.0, filter_type=FilterType.IDEAL)
```

### Band-Pass (Mid-Range Frequencies)

```python
midrange = apply_bandpass_filter(img, low_freq=10.0, high_freq=30.0)
```

### Band-Stop (Notch/Remove Frequencies)

```python
denoised = apply_bandstop_filter(img, low_freq=48.0, high_freq=52.0)
```

---

## Spatial-Domain Filters (Fast)

### Gaussian Blur

```python
blurred = apply_spatial_lowpass(img, kernel_size=5, sigma=1.0)
```

### Laplacian (Edge Detection)

```python
edges = apply_spatial_highpass(img, kernel_size=5)
```

### Unsharp Mask (Sharpening)

```python
sharp = apply_unsharp_mask(img, kernel_size=5, sigma=1.0, strength=1.5)
```

---

## Common Use Cases

### Skin Smoothing
```python
smoothed = apply_lowpass_filter(img, sigma=25.0)
```

### Texture Enhancement
```python
enhanced = apply_highpass_filter(img, sigma=10.0)
```

### Aging Effect
```python
aged = apply_highpass_filter(img, sigma=8.0)
# Result: More visible wrinkles/texture
```

### De-Aging Effect
```python
deaged = apply_lowpass_filter(img, sigma=25.0)
# Result: Smoother skin, reduced wrinkles
```

### Selective Sharpening
```python
sharpened = apply_unsharp_mask(img, kernel_size=5, strength=1.5)
```

### Multi-Scale Enhancement
```python
low = apply_lowpass_filter(img, sigma=30.0)          # Overall shape
mid = apply_bandpass_filter(img, 15.0, 40.0)        # Features
high = apply_highpass_filter(img, sigma=10.0)       # Texture
enhanced = low + 1.5*mid + 0.5*high
```

---

## Parameter Guide

### Sigma (σ) — Standard Deviation

| Value | Effect | Use Case |
|-------|--------|----------|
| 5–10 | Subtle smoothing | Detail preservation |
| 10–20 | Moderate smoothing | Balanced effect |
| 20–40 | Strong smoothing | Skin softening, de-aging |
| 40+ | Very strong smoothing | Extreme blur |

### Filter Type Comparison

| Type | Smoothness | Sharpness | Ringing | Speed |
|------|-----------|-----------|---------|-------|
| **Gaussian** | Very smooth | Gradual | None | 70-85ms |
| **Butterworth** | Smooth | Controllable | Minimal | 70-85ms |
| **Ideal** | Sharp | Brick-wall | Severe (Gibbs) | 70-85ms |

**Recommendation**: Use **Gaussian** for most cases; use **Butterworth** for fine control.

### Kernel Size (Spatial Filters)

| Size | Speed | Use Case |
|------|-------|----------|
| 3×3 | Fastest (1-2ms) | Quick operations |
| 5×5 | Fast (2-5ms) | Balanced |
| 7×7 | Slower (5-10ms) | Strong effect |
| 11×11+ | Slowest (10+ms) | Extreme effect |

---

## Performance Benchmark (512×512 RGB)

| Operation | Time | Notes |
|-----------|------|-------|
| Gaussian LP | ~77ms | Independent of σ |
| Gaussian HP | ~85ms | Independent of σ |
| Butterworth LP | ~70ms | Includes Fourier transforms |
| Spatial LP (5×5) | 0.26ms | Very fast |
| Spatial HP (5×5) | 8.80ms | OpenCV Laplacian |
| Unsharp Mask | 7.88ms | Blur + subtract |

**Rule**: Use spatial filters for real-time; FFT for large kernels.

---

## Blending Results

Combine filter outputs for effects:

```python
# 70% original + 30% high-pass (subtle enhancement)
enhanced = cv2.addWeighted(img.astype(np.float32), 0.7, hp.astype(np.float32), 0.3, 0)

# 80% original + 20% smoothed (subtle de-aging)
deaged = cv2.addWeighted(img.astype(np.float32), 0.8, smooth.astype(np.float32), 0.2, 0)

# Multi-scale synthesis
composite = low + 1.5*mid + 0.5*high
```

---

## Input/Output

- **Input**: uint8 BGR or grayscale, any size
- **Output**: Same dtype, same size
- **Clipping**: Automatically applied to [0, 255]

```python
# Works with both:
img_rgb = cv2.imread('photo.jpg')          # BGR uint8
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)  # Grayscale uint8

result_rgb = apply_lowpass_filter(img_rgb)
result_gray = apply_lowpass_filter(img_gray)
```

---

## Testing & Examples

Run tests:
```bash
cd emergent/photix/
python tests/test_filters.py
```

Run demonstrations:
```bash
python examples/filter_demo.py
```

---

## Additional Resources

- Full documentation: `core/FILTERS_README.md`
- Test suite: `tests/test_filters.py`
- Examples: `examples/filter_demo.py`
- Source: `core/filters.py`

---

## Common Issues

**Q: Result looks washed out**  
A: High σ values blur too much. Try σ = 10–20.

**Q: High-pass result too dark**  
A: Result is centered at 128. Use with blending: `0.7×orig + 0.3×hp`

**Q: Need very fast results**  
A: Use spatial filters (`apply_spatial_lowpass`) instead of FFT.

**Q: Result has artifacts**  
A: Ideal filter has Gibbs ringing. Use Gaussian or Butterworth instead.

---

## Version Info

- **Module**: `core/filters.py`
- **Functions**: 8 (4 frequency-domain, 3 spatial-domain, 1 FFT helper)
- **Test Coverage**: 50+ test cases
- **Dependencies**: numpy, opencv-python (cv2)
- **Status**: ✓ Production-ready

