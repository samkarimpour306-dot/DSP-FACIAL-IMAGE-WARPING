"""
filters.py — Frequency-domain and spatial-domain filter implementations
Provides low-pass, high-pass, band-pass, and band-stop filters for DSP applications.
"""
import numpy as np
import cv2
from enum import Enum


class FilterType(Enum):
    """Filter kernel types for frequency-domain filtering."""
    GAUSSIAN = "gaussian"
    BUTTERWORTH = "butterworth"
    IDEAL = "ideal"


# ── FREQUENCY-DOMAIN FILTER KERNELS ─────────────────────────────────────────

def _gaussian_kernel(shape: tuple, sigma: float, center: tuple[int, int] | None = None) -> np.ndarray:
    """Generate 2D Gaussian kernel in frequency domain.
    
    Args:
        shape: (height, width) of the output kernel
        sigma: standard deviation of the Gaussian
        center: (cy, cx) center point; defaults to image center if None
    
    Returns:
        2D Gaussian kernel (float32) with values in [0, 1]
    """
    H, W = shape
    if center is None:
        center = (H // 2, W // 2)
    cy, cx = center
    
    y = (np.arange(H) - cy).astype(np.float32)
    x = (np.arange(W) - cx).astype(np.float32)
    X, Y = np.meshgrid(x, y)
    
    return np.exp(-(X ** 2 + Y ** 2) / (2.0 * sigma ** 2))


def _butterworth_kernel(shape: tuple, cutoff: float, order: int = 2, center: tuple[int, int] | None = None) -> np.ndarray:
    """Generate 2D Butterworth kernel in frequency domain.
    
    Args:
        shape: (height, width) of the output kernel
        cutoff: cutoff frequency (normalized to image dimensions)
        order: filter order (higher = sharper transition)
        center: (cy, cx) center point; defaults to image center if None
    
    Returns:
        2D Butterworth kernel (float32) with values in [0, 1]
    """
    H, W = shape
    if center is None:
        center = (H // 2, W // 2)
    cy, cx = center
    
    y = (np.arange(H) - cy).astype(np.float32)
    x = (np.arange(W) - cx).astype(np.float32)
    X, Y = np.meshgrid(x, y)
    D = np.sqrt(X ** 2 + Y ** 2)
    
    # Avoid division by zero
    D = np.maximum(D, 1e-6)
    cutoff = max(cutoff, 1e-6)
    
    return 1.0 / (1.0 + (D / cutoff) ** (2.0 * order))


def _ideal_kernel(shape: tuple, radius: float, center: tuple[int, int] | None = None) -> np.ndarray:
    """Generate 2D ideal (brick-wall) low-pass kernel.
    
    Args:
        shape: (height, width) of the output kernel
        radius: cutoff radius
        center: (cy, cx) center point; defaults to image center if None
    
    Returns:
        Binary kernel (float32) with values 0 (block) or 1 (pass)
    """
    H, W = shape
    if center is None:
        center = (H // 2, W // 2)
    cy, cx = center
    
    y = (np.arange(H) - cy).astype(np.float32)
    x = (np.arange(W) - cx).astype(np.float32)
    X, Y = np.meshgrid(x, y)
    D = np.sqrt(X ** 2 + Y ** 2)
    
    return (D <= radius).astype(np.float32)


# ── FREQUENCY-DOMAIN FILTERING ───────────────────────────────────────────────

def _fft_filter_channel(channel: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Apply frequency-domain filter to a single channel.
    
    Args:
        channel: 2D image channel (grayscale)
        kernel: frequency-domain filter kernel
    
    Returns:
        Filtered channel (same shape as input)
    """
    # Ensure float64 for precision
    ch = channel.astype(np.float64)
    
    # Compute FFT with shift
    F = np.fft.fftshift(np.fft.fft2(ch))
    
    # Apply filter in frequency domain
    F_filtered = F * kernel
    
    # Inverse FFT
    filtered = np.fft.ifft2(np.fft.ifftshift(F_filtered)).real
    
    return filtered


def apply_lowpass_filter(
    img: np.ndarray,
    sigma: float = 15.0,
    filter_type: FilterType = FilterType.GAUSSIAN,
    **kwargs
) -> np.ndarray:
    """Apply low-pass (smoothing) filter to an image.
    
    Uses frequency-domain filtering (FFT) for optimal performance.
    
    Args:
        img: Input image (BGR or grayscale)
        sigma: Filter parameter (interpretation depends on filter_type):
               - GAUSSIAN: standard deviation
               - BUTTERWORTH: cutoff frequency
               - IDEAL: cutoff radius
        filter_type: Type of low-pass filter kernel
        **kwargs: Additional arguments passed to kernel generators
                 (order for Butterworth)
    
    Returns:
        Low-pass filtered image (same shape and dtype as input)
    """
    if img.size == 0:
        return img
    
    H, W = img.shape[:2]
    is_color = len(img.shape) == 3
    
    # Generate filter kernel
    if filter_type == FilterType.GAUSSIAN:
        kernel = _gaussian_kernel((H, W), sigma)
    elif filter_type == FilterType.BUTTERWORTH:
        order = kwargs.get('order', 2)
        kernel = _butterworth_kernel((H, W), sigma, order)
    elif filter_type == FilterType.IDEAL:
        kernel = _ideal_kernel((H, W), sigma)
    else:
        raise ValueError(f"Unknown filter type: {filter_type}")
    
    # Apply to each channel
    if is_color:
        result = np.zeros_like(img, dtype=np.float32)
        for c in range(min(3, img.shape[2])):
            result[:, :, c] = _fft_filter_channel(img[:, :, c], kernel)
    else:
        result = _fft_filter_channel(img, kernel)
    
    # Clip and return in original dtype
    result = np.clip(result, 0, 255).astype(img.dtype)
    return result


def apply_highpass_filter(
    img: np.ndarray,
    sigma: float = 15.0,
    filter_type: FilterType = FilterType.GAUSSIAN,
    **kwargs
) -> np.ndarray:
    """Apply high-pass (edge enhancement) filter to an image.
    
    Uses frequency-domain filtering (FFT) for optimal performance.
    High-pass = Original - Low-pass
    
    Args:
        img: Input image (BGR or grayscale)
        sigma: Filter parameter (interpretation depends on filter_type):
               - GAUSSIAN: standard deviation
               - BUTTERWORTH: cutoff frequency
               - IDEAL: cutoff radius
        filter_type: Type of high-pass filter kernel
        **kwargs: Additional arguments passed to kernel generators
                 (order for Butterworth)
    
    Returns:
        High-pass filtered image (same shape and dtype as input)
    """
    if img.size == 0:
        return img
    
    H, W = img.shape[:2]
    is_color = len(img.shape) == 3
    
    # Generate high-pass kernel (complement of low-pass)
    if filter_type == FilterType.GAUSSIAN:
        lp_kernel = _gaussian_kernel((H, W), sigma)
    elif filter_type == FilterType.BUTTERWORTH:
        order = kwargs.get('order', 2)
        lp_kernel = _butterworth_kernel((H, W), sigma, order)
    elif filter_type == FilterType.IDEAL:
        lp_kernel = _ideal_kernel((H, W), sigma)
    else:
        raise ValueError(f"Unknown filter type: {filter_type}")
    
    hp_kernel = 1.0 - lp_kernel
    
    # Apply to each channel
    if is_color:
        result = np.zeros_like(img, dtype=np.float32)
        for c in range(min(3, img.shape[2])):
            result[:, :, c] = _fft_filter_channel(img[:, :, c], hp_kernel)
    else:
        result = _fft_filter_channel(img, hp_kernel)
    
    # Center the result around 128 (high-pass typically has zero DC component)
    result = np.clip(result + 128, 0, 255).astype(img.dtype)
    return result


def apply_bandpass_filter(
    img: np.ndarray,
    low_freq: float = 5.0,
    high_freq: float = 20.0,
    filter_type: FilterType = FilterType.GAUSSIAN,
    **kwargs
) -> np.ndarray:
    """Apply band-pass filter (passes frequencies between low_freq and high_freq).
    
    Args:
        img: Input image (BGR or grayscale)
        low_freq: Lower frequency cutoff
        high_freq: Upper frequency cutoff
        filter_type: Type of filter kernel
        **kwargs: Additional arguments
    
    Returns:
        Band-pass filtered image
    """
    if img.size == 0 or low_freq >= high_freq:
        return img
    
    H, W = img.shape[:2]
    is_color = len(img.shape) == 3
    
    # Band-pass = Low-pass(high_freq) - Low-pass(low_freq)
    if filter_type == FilterType.GAUSSIAN:
        lp_high = _gaussian_kernel((H, W), high_freq)
        lp_low = _gaussian_kernel((H, W), low_freq)
    elif filter_type == FilterType.BUTTERWORTH:
        order = kwargs.get('order', 2)
        lp_high = _butterworth_kernel((H, W), high_freq, order)
        lp_low = _butterworth_kernel((H, W), low_freq, order)
    elif filter_type == FilterType.IDEAL:
        lp_high = _ideal_kernel((H, W), high_freq)
        lp_low = _ideal_kernel((H, W), low_freq)
    else:
        raise ValueError(f"Unknown filter type: {filter_type}")
    
    bp_kernel = lp_high - lp_low
    
    # Apply to each channel
    if is_color:
        result = np.zeros_like(img, dtype=np.float32)
        for c in range(min(3, img.shape[2])):
            result[:, :, c] = _fft_filter_channel(img[:, :, c], bp_kernel)
    else:
        result = _fft_filter_channel(img, bp_kernel)
    
    result = np.clip(result + 128, 0, 255).astype(img.dtype)
    return result


def apply_bandstop_filter(
    img: np.ndarray,
    low_freq: float = 5.0,
    high_freq: float = 20.0,
    filter_type: FilterType = FilterType.GAUSSIAN,
    **kwargs
) -> np.ndarray:
    """Apply band-stop (notch) filter (blocks frequencies between low_freq and high_freq).
    
    Args:
        img: Input image (BGR or grayscale)
        low_freq: Lower frequency cutoff
        high_freq: Upper frequency cutoff
        filter_type: Type of filter kernel
        **kwargs: Additional arguments
    
    Returns:
        Band-stop filtered image
    """
    if img.size == 0 or low_freq >= high_freq:
        return img
    
    H, W = img.shape[:2]
    is_color = len(img.shape) == 3
    
    # Band-stop = 1 - Band-pass
    if filter_type == FilterType.GAUSSIAN:
        lp_high = _gaussian_kernel((H, W), high_freq)
        lp_low = _gaussian_kernel((H, W), low_freq)
    elif filter_type == FilterType.BUTTERWORTH:
        order = kwargs.get('order', 2)
        lp_high = _butterworth_kernel((H, W), high_freq, order)
        lp_low = _butterworth_kernel((H, W), low_freq, order)
    elif filter_type == FilterType.IDEAL:
        lp_high = _ideal_kernel((H, W), high_freq)
        lp_low = _ideal_kernel((H, W), low_freq)
    else:
        raise ValueError(f"Unknown filter type: {filter_type}")
    
    bp_kernel = lp_high - lp_low
    bs_kernel = 1.0 - bp_kernel
    
    # Apply to each channel
    if is_color:
        result = np.zeros_like(img, dtype=np.float32)
        for c in range(min(3, img.shape[2])):
            result[:, :, c] = _fft_filter_channel(img[:, :, c], bs_kernel)
    else:
        result = _fft_filter_channel(img, bs_kernel)
    
    result = np.clip(result, 0, 255).astype(img.dtype)
    return result


# ── SPATIAL-DOMAIN CONVENIENCE FILTERS ───────────────────────────────────────

def apply_spatial_lowpass(img: np.ndarray, kernel_size: int = 5, sigma: float = 1.0) -> np.ndarray:
    """Apply Gaussian low-pass filter in spatial domain (faster for small kernels).
    
    Args:
        img: Input image
        kernel_size: Kernel size (must be odd)
        sigma: Gaussian standard deviation
    
    Returns:
        Spatially low-pass filtered image
    """
    if kernel_size % 2 == 0:
        kernel_size += 1  # Ensure odd
    
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma)


def apply_spatial_highpass(img: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Apply Laplacian high-pass filter in spatial domain.
    
    Args:
        img: Input image
        kernel_size: Kernel size for Laplacian (must be odd, typically 3, 5, or 7)
    
    Returns:
        Spatially high-pass filtered image (edge-enhanced)
    """
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    is_color = len(img.shape) == 3
    result = np.zeros_like(img, dtype=np.float32)
    
    if is_color:
        for c in range(3):
            laplacian = cv2.Laplacian(img[:, :, c].astype(np.float32), cv2.CV_32F, ksize=kernel_size)
            result[:, :, c] = img[:, :, c].astype(np.float32) - laplacian
    else:
        laplacian = cv2.Laplacian(img.astype(np.float32), cv2.CV_32F, ksize=kernel_size)
        result = img.astype(np.float32) - laplacian
    
    result = np.clip(result, 0, 255).astype(img.dtype)
    return result


def apply_unsharp_mask(
    img: np.ndarray,
    kernel_size: int = 5,
    sigma: float = 1.0,
    strength: float = 1.0
) -> np.ndarray:
    """Apply unsharp mask (high-pass enhancement).
    
    Formula: Output = Original + strength * (Original - Blur)
    
    Args:
        img: Input image
        kernel_size: Kernel size for the blur
        sigma: Gaussian standard deviation
        strength: Enhancement strength (higher = more sharpening)
    
    Returns:
        Unsharp masked image
    """
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    is_color = len(img.shape) == 3
    img_float = img.astype(np.float32)
    
    blurred = cv2.GaussianBlur(img_float, (kernel_size, kernel_size), sigma)
    highpass = img_float - blurred
    
    result = img_float + strength * highpass
    result = np.clip(result, 0, 255).astype(img.dtype)
    
    return result
