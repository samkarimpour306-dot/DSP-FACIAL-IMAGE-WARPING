"""
test_filters.py — Unit tests and examples for frequency-domain filters

Run with: python -m pytest tests/test_filters.py
Or directly: python tests/test_filters.py
"""
import numpy as np
import cv2
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

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


def create_test_image() -> np.ndarray:
    """Create a synthetic test image (512x512) with text and patterns."""
    img = np.ones((512, 512, 3), dtype=np.uint8) * 200
    
    # Add some patterns
    img[100:150, 100:400] = 50
    img[200:250, 50:450] = 150
    img[300:350, 150:350] = 100
    
    # Add noise
    noise = np.random.normal(0, 15, img.shape)
    img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    
    # Add text
    cv2.putText(img, "Test Image", (150, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
    
    return img


def test_lowpass_gaussian():
    """Test Gaussian low-pass filter."""
    print("Testing Gaussian Low-pass Filter...")
    img = create_test_image()
    
    # Apply with different sigma values
    for sigma in [5.0, 15.0, 30.0]:
        result = apply_lowpass_filter(img, sigma=sigma, filter_type=FilterType.GAUSSIAN)
        assert result.shape == img.shape, f"Shape mismatch: {result.shape} vs {img.shape}"
        assert result.dtype == img.dtype, f"Dtype mismatch: {result.dtype} vs {img.dtype}"
        print(f"  ✓ Gaussian LP (σ={sigma}) — output range: [{result.min()}, {result.max()}]")


def test_highpass_gaussian():
    """Test Gaussian high-pass filter."""
    print("Testing Gaussian High-pass Filter...")
    img = create_test_image()
    
    for sigma in [5.0, 15.0, 30.0]:
        result = apply_highpass_filter(img, sigma=sigma, filter_type=FilterType.GAUSSIAN)
        assert result.shape == img.shape
        assert result.dtype == img.dtype
        print(f"  ✓ Gaussian HP (σ={sigma}) — output range: [{result.min()}, {result.max()}]")


def test_lowpass_butterworth():
    """Test Butterworth low-pass filter."""
    print("Testing Butterworth Low-pass Filter...")
    img = create_test_image()
    
    for cutoff in [10.0, 20.0, 40.0]:
        result = apply_lowpass_filter(
            img,
            sigma=cutoff,
            filter_type=FilterType.BUTTERWORTH,
            order=2
        )
        assert result.shape == img.shape
        print(f"  ✓ Butterworth LP (cutoff={cutoff}, order=2) OK")


def test_highpass_butterworth():
    """Test Butterworth high-pass filter."""
    print("Testing Butterworth High-pass Filter...")
    img = create_test_image()
    
    for cutoff in [10.0, 20.0, 40.0]:
        result = apply_highpass_filter(
            img,
            sigma=cutoff,
            filter_type=FilterType.BUTTERWORTH,
            order=2
        )
        assert result.shape == img.shape
        print(f"  ✓ Butterworth HP (cutoff={cutoff}, order=2) OK")


def test_bandpass_filter():
    """Test band-pass filter."""
    print("Testing Band-pass Filter...")
    img = create_test_image()
    
    result = apply_bandpass_filter(img, low_freq=10.0, high_freq=30.0)
    assert result.shape == img.shape
    assert result.dtype == img.dtype
    print(f"  ✓ Band-pass (10-30 Hz) — output range: [{result.min()}, {result.max()}]")


def test_bandstop_filter():
    """Test band-stop (notch) filter."""
    print("Testing Band-stop (Notch) Filter...")
    img = create_test_image()
    
    result = apply_bandstop_filter(img, low_freq=15.0, high_freq=25.0)
    assert result.shape == img.shape
    assert result.dtype == img.dtype
    print(f"  ✓ Band-stop (15-25 Hz) — output range: [{result.min()}, {result.max()}]")


def test_spatial_lowpass():
    """Test spatial-domain low-pass filter."""
    print("Testing Spatial Low-pass Filter...")
    img = create_test_image()
    
    for kernel_size in [3, 5, 11]:
        result = apply_spatial_lowpass(img, kernel_size=kernel_size, sigma=1.0)
        assert result.shape == img.shape
        print(f"  ✓ Spatial LP (kernel={kernel_size}x{kernel_size}) OK")


def test_spatial_highpass():
    """Test spatial-domain high-pass filter (Laplacian)."""
    print("Testing Spatial High-pass Filter (Laplacian)...")
    img = create_test_image()
    
    for kernel_size in [3, 5, 7]:
        result = apply_spatial_highpass(img, kernel_size=kernel_size)
        assert result.shape == img.shape
        print(f"  ✓ Spatial HP / Laplacian (kernel={kernel_size}) OK")


def test_unsharp_mask():
    """Test unsharp mask (sharpening)."""
    print("Testing Unsharp Mask (Sharpening)...")
    img = create_test_image()
    
    for strength in [0.5, 1.0, 2.0]:
        result = apply_unsharp_mask(img, kernel_size=5, sigma=1.0, strength=strength)
        assert result.shape == img.shape
        print(f"  ✓ Unsharp Mask (strength={strength}) OK")


def test_grayscale_input():
    """Test filters with grayscale input."""
    print("Testing Filters with Grayscale Input...")
    img_color = create_test_image()
    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    
    result_lp = apply_lowpass_filter(img_gray, sigma=15.0)
    assert result_lp.shape == img_gray.shape
    assert len(result_lp.shape) == 2
    
    result_hp = apply_highpass_filter(img_gray, sigma=15.0)
    assert result_hp.shape == img_gray.shape
    
    print(f"  ✓ Grayscale low-pass OK (shape: {result_lp.shape})")
    print(f"  ✓ Grayscale high-pass OK (shape: {result_hp.shape})")


def test_edge_cases():
    """Test edge cases and error handling."""
    print("Testing Edge Cases...")
    
    # Empty image
    empty = np.zeros((0, 0, 3), dtype=np.uint8)
    result = apply_lowpass_filter(empty)
    assert result.size == 0
    print("  ✓ Empty image handled")
    
    # Single-channel
    img_gray = create_test_image()[:, :, 0]
    result = apply_lowpass_filter(img_gray)
    assert result.shape == img_gray.shape
    print("  ✓ Single-channel image handled")
    
    # Invalid band-pass (low > high)
    img = create_test_image()
    result = apply_bandpass_filter(img, low_freq=30.0, high_freq=10.0)
    assert result.shape == img.shape
    print("  ✓ Invalid band-pass range handled gracefully")


def benchmark_filters():
    """Benchmark filter performance."""
    print("\nBenchmarking Filters (512x512 RGB image)...")
    import time
    
    img = create_test_image()
    
    filters_to_test = [
        ("Gaussian LP (σ=15)", lambda: apply_lowpass_filter(img, sigma=15.0, filter_type=FilterType.GAUSSIAN)),
        ("Gaussian HP (σ=15)", lambda: apply_highpass_filter(img, sigma=15.0, filter_type=FilterType.GAUSSIAN)),
        ("Butterworth LP", lambda: apply_lowpass_filter(img, sigma=20.0, filter_type=FilterType.BUTTERWORTH, order=2)),
        ("Spatial LP (k=5)", lambda: apply_spatial_lowpass(img, kernel_size=5)),
        ("Spatial HP (k=5)", lambda: apply_spatial_highpass(img, kernel_size=5)),
        ("Unsharp Mask", lambda: apply_unsharp_mask(img, kernel_size=5, strength=1.5)),
    ]
    
    for name, func in filters_to_test:
        start = time.perf_counter()
        for _ in range(3):
            func()
        elapsed = (time.perf_counter() - start) / 3.0
        print(f"  {name:25s} — {elapsed*1000:6.2f} ms")


if __name__ == "__main__":
    print("=" * 70)
    print("DSP Filter Module — Comprehensive Test Suite")
    print("=" * 70)
    
    test_lowpass_gaussian()
    test_highpass_gaussian()
    test_lowpass_butterworth()
    test_highpass_butterworth()
    test_bandpass_filter()
    test_bandstop_filter()
    test_spatial_lowpass()
    test_spatial_highpass()
    test_unsharp_mask()
    test_grayscale_input()
    test_edge_cases()
    
    benchmark_filters()
    
    print("\n" + "=" * 70)
    print("✓ All tests passed!")
    print("=" * 70)
