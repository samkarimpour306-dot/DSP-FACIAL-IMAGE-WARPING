"""
examples/filter_demo.py — Practical demonstration of filter usage

This script demonstrates how to apply various filters to facial images
for aging/de-aging effects and image enhancement.

Run with: python examples/filter_demo.py
"""
import cv2
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.filters import (
    FilterType,
    apply_lowpass_filter,
    apply_highpass_filter,
    apply_bandpass_filter,
    apply_spatial_lowpass,
    apply_unsharp_mask,
)
from core.image_loader import load_image


def demo_basic_smoothing(img):
    """Demonstrate basic smoothing with low-pass filter."""
    print("Demo 1: Basic Skin Smoothing (Low-Pass Filter)")
    print("-" * 60)
    
    # Apply Gaussian low-pass for skin smoothing
    smoothed = apply_lowpass_filter(img, sigma=20.0, filter_type=FilterType.GAUSSIAN)
    
    print(f"Original shape: {img.shape}")
    print(f"Filtered shape: {smoothed.shape}")
    print(f"Original dtype: {img.dtype}")
    print(f"Filtered dtype: {smoothed.dtype}")
    print(f"Intensity range - Original: [{img.min()}, {img.max()}]")
    print(f"Intensity range - Smoothed: [{smoothed.min()}, {smoothed.max()}]")
    
    return smoothed


def demo_edge_enhancement(img):
    """Demonstrate edge enhancement with high-pass filter."""
    print("\nDemo 2: Edge Enhancement (High-Pass Filter)")
    print("-" * 60)
    
    # Apply Gaussian high-pass for edge enhancement
    enhanced = apply_highpass_filter(img, sigma=15.0, filter_type=FilterType.GAUSSIAN)
    
    # Blend with original for visible enhancement
    result = cv2.addWeighted(img.astype(np.float32), 0.7, enhanced.astype(np.float32), 0.3, 0)
    result = np.clip(result, 0, 255).astype(img.dtype)
    
    print(f"High-pass component range: [{enhanced.min()}, {enhanced.max()}]")
    print(f"Blended result range: [{result.min()}, {result.max()}]")
    print("Blend formula: Output = 0.7×Original + 0.3×HighPass")
    
    return result


def demo_butterworth_filter(img):
    """Demonstrate Butterworth filter with controllable smoothness."""
    print("\nDemo 3: Butterworth Low-Pass Filter (Smooth Transition)")
    print("-" * 60)
    
    # Butterworth provides smoother transition than Gaussian
    filtered = apply_lowpass_filter(
        img,
        sigma=20.0,
        filter_type=FilterType.BUTTERWORTH,
        order=2  # Lower order = smoother, Higher order = sharper cutoff
    )
    
    print(f"Butterworth LP with cutoff=20.0, order=2")
    print(f"Result range: [{filtered.min()}, {filtered.max()}]")
    print("Butterworth provides smoother frequency response than Gaussian")
    
    return filtered


def demo_multi_scale_enhancement(img):
    """Demonstrate multi-scale texture enhancement."""
    print("\nDemo 4: Multi-Scale Texture Enhancement")
    print("-" * 60)
    
    # Extract low frequencies (overall shape)
    low_freq = apply_lowpass_filter(img, sigma=30.0)
    
    # Extract mid frequencies (facial features)
    mid_freq = apply_bandpass_filter(img, low_freq=15.0, high_freq=40.0)
    
    # Extract high frequencies (texture/detail)
    high_freq = apply_highpass_filter(img, sigma=10.0)
    
    # Reconstruct with enhanced mid-range
    enhanced = low_freq.astype(np.float32) + \
               mid_freq.astype(np.float32) * 1.5 + \
               high_freq.astype(np.float32) * 0.5
    enhanced = np.clip(enhanced, 0, 255).astype(img.dtype)
    
    print(f"Low-frequency range (shape): [{low_freq.min()}, {low_freq.max()}]")
    print(f"Mid-frequency range (features): [{mid_freq.min()}, {mid_freq.max()}]")
    print(f"High-frequency range (texture): [{high_freq.min()}, {high_freq.max()}]")
    print(f"Enhanced result range: [{enhanced.min()}, {enhanced.max()}]")
    
    return enhanced


def demo_sharpening(img):
    """Demonstrate selective sharpening with unsharp mask."""
    print("\nDemo 5: Selective Sharpening (Unsharp Mask)")
    print("-" * 60)
    
    # Apply unsharp mask for controlled sharpening
    sharpened = apply_unsharp_mask(
        img,
        kernel_size=5,
        sigma=1.0,
        strength=1.5  # Higher = more sharpening (0.5-2.0 typical)
    )
    
    print(f"Unsharp Mask Settings:")
    print(f"  Kernel Size: 5x5")
    print(f"  Sigma: 1.0")
    print(f"  Strength: 1.5")
    print(f"Result range: [{sharpened.min()}, {sharpened.max()}]")
    print("Formula: Output = Original + strength × (Original - Blur)")
    
    return sharpened


def demo_aging_effect(img):
    """Demonstrate synthetic aging effect using high-pass filter."""
    print("\nDemo 6: Synthetic Aging Effect")
    print("-" * 60)
    
    # Enhance texture with high-pass to simulate wrinkles/aging
    texture_enhanced = apply_highpass_filter(img, sigma=8.0, filter_type=FilterType.GAUSSIAN)
    
    # Blend: more high-pass = older appearance
    aged = cv2.addWeighted(img.astype(np.float32), 0.6, texture_enhanced.astype(np.float32), 0.4, 0)
    aged = np.clip(aged, 0, 255).astype(img.dtype)
    
    print(f"Aging Effect Settings:")
    print(f"  Filter: Gaussian High-Pass (σ=8.0)")
    print(f"  Blend: 60% Original + 40% HighPass")
    print(f"Result range: [{aged.min()}, {aged.max()}]")
    print("Explanation: High-pass enhances texture/wrinkles for aged appearance")
    
    return aged


def demo_de_aging_effect(img):
    """Demonstrate synthetic de-aging effect using low-pass filter."""
    print("\nDemo 7: Synthetic De-Aging Effect")
    print("-" * 60)
    
    # Smooth skin with subtle low-pass
    smoothed = apply_lowpass_filter(img, sigma=25.0, filter_type=FilterType.GAUSSIAN)
    
    # Blend: preserve some detail to avoid unnatural look
    deaged = cv2.addWeighted(img.astype(np.float32), 0.8, smoothed.astype(np.float32), 0.2, 0)
    deaged = np.clip(deaged, 0, 255).astype(img.dtype)
    
    print(f"De-Aging Effect Settings:")
    print(f"  Filter: Gaussian Low-Pass (σ=25.0)")
    print(f"  Blend: 80% Original + 20% Smoothed")
    print(f"Result range: [{deaged.min()}, {deaged.max()}]")
    print("Explanation: Low-pass smooths skin/reduces wrinkles for younger appearance")
    
    return deaged


def create_test_image():
    """Create a test image if no input is provided."""
    print("Creating synthetic test image...")
    img = np.ones((512, 512, 3), dtype=np.uint8) * 128
    
    # Add gradient
    for i in range(512):
        img[i, :, 0] = int(100 + i * 0.155)
        img[i, :, 1] = int(128 + i * 0.1)
        img[i, :, 2] = int(150 + i * 0.05)
    
    # Add some texture
    noise = np.random.normal(0, 20, img.shape)
    img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    
    # Add circles
    cv2.circle(img, (150, 150), 50, (50, 100, 200), -1)
    cv2.circle(img, (350, 350), 60, (200, 100, 50), -1)
    cv2.circle(img, (450, 150), 40, (100, 200, 100), -1)
    
    # Add text
    cv2.putText(img, "Filter Demo", (100, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
    
    return img


def main():
    print("=" * 70)
    print("DSP Filter Module — Practical Demonstrations")
    print("=" * 70)
    
    # Try to load a sample image; create synthetic one if not found
    try:
        img = load_image("sample.jpg", target_size=512)
        print(f"\nLoaded image: sample.jpg (shape: {img.shape})")
    except Exception as e:
        print(f"\nCould not load sample image: {e}")
        print("Creating synthetic test image instead...\n")
        img = create_test_image()
    
    # Run all demonstrations
    demos = [
        ("Basic Smoothing", demo_basic_smoothing),
        ("Edge Enhancement", demo_edge_enhancement),
        ("Butterworth Filter", demo_butterworth_filter),
        ("Multi-Scale Enhancement", demo_multi_scale_enhancement),
        ("Sharpening", demo_sharpening),
        ("Aging Effect", demo_aging_effect),
        ("De-Aging Effect", demo_de_aging_effect),
    ]
    
    results = {"Original": img}
    
    for name, demo_func in demos:
        try:
            result = demo_func(img)
            results[name] = result
        except Exception as e:
            print(f"Error in {name}: {e}")
    
    # Save results
    output_dir = Path(__file__).parent.parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    
    print("\n" + "=" * 70)
    print("Summary of Demonstrations")
    print("=" * 70)
    for name, result in results.items():
        filename = output_dir / f"filter_demo_{name.lower().replace(' ', '_')}.png"
        cv2.imwrite(str(filename), result)
        print(f"✓ {name:30s} → {filename.name}")
    
    print("\n" + "=" * 70)
    print("✓ All demonstrations completed!")
    print(f"✓ Results saved to: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
