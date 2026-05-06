"""
image_loader.py — FR-01, FR-02
Image loading and preprocessing.
"""
import cv2
import numpy as np
from pathlib import Path

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
TARGET_SIZE = 512


def load_image(path: str) -> np.ndarray:
    """FR-01: Load JPEG/PNG/BMP image from disk."""
    if not Path(path).exists():
        raise FileNotFoundError(f"File not found: {path}")
    ext = Path(path).suffix.lower()
    if ext not in SUPPORTED_EXTS:
        raise ValueError(f"Unsupported format: {ext}. Use JPEG, PNG or BMP.")
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise OSError(f"OpenCV could not read: {path}")
    return img


def preprocess(img: np.ndarray, size: int = TARGET_SIZE) -> np.ndarray:
    """FR-02: Resize + center-crop to size×size (preserves aspect ratio).
    Uses INTER_CUBIC for quality. Output is guaranteed 0-255 uint8."""
    h, w = img.shape[:2]
    scale = size / min(h, w)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    y0 = (new_h - size) // 2
    x0 = (new_w - size) // 2
    cropped = resized[y0: y0 + size, x0: x0 + size]
    # Boundary safety — NFR-R02
    return np.clip(cropped, 0, 255).astype(np.uint8)
