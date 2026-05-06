"""
fft_analyzer.py — FR-18, FR-19, FR-20, FR-21, FR-29
2D FFT analysis: magnitude spectrum, energy ratio, side-by-side comparison.
"""
import cv2
import numpy as np


def _analyze_spectrum(img: np.ndarray, low_radius_frac: float = 0.25) -> tuple:
    """Compute magnitude spectrum + energy ratio with a single FFT pass."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)
    gray -= np.mean(gray)

    F     = np.fft.fftshift(np.fft.fft2(gray))
    mag   = np.abs(F)
    spectrum = (20.0 * np.log1p(mag)).astype(np.float32)

    power     = mag ** 2
    H, W      = gray.shape
    cy, cx    = H // 2, W // 2
    Y, X      = np.ogrid[:H, :W]
    dist      = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    threshold = min(H, W) * low_radius_frac

    low   = float(np.sum(power[dist <= threshold]))
    high  = float(np.sum(power[dist >  threshold]))
    total = low + high

    energy = {"total": total, "low": low, "high": high,
              "ratio": high / (total + 1e-12)}
    return spectrum, energy


def compute_magnitude_spectrum(img: np.ndarray) -> np.ndarray:
    """FR-18, FR-19: log-scaled magnitude spectrum (float32, display-ready)."""
    spectrum, _ = _analyze_spectrum(img)
    return spectrum


def compute_energy_ratio(img: np.ndarray, low_radius_frac: float = 0.25) -> dict:
    """FR-20: Spectral energy decomposition (total / low / high / ratio)."""
    _, energy = _analyze_spectrum(img, low_radius_frac)
    return energy


def compare_spectra(orig: np.ndarray, processed: np.ndarray) -> dict:
    """FR-21, FR-29: Compare FFT spectra — two FFTs total (was four)."""
    orig_sp, orig_e = _analyze_spectrum(orig)
    proc_sp, proc_e = _analyze_spectrum(processed)
    return {
        "orig_spectrum": orig_sp,
        "proc_spectrum": proc_sp,
        "orig_energy":   orig_e,
        "proc_energy":   proc_e,
    }
