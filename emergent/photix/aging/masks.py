"""Landmark-derived soft masks for aging effects."""

from __future__ import annotations

import cv2
import numpy as np

from landmarks.detector import FaceLandmarks


def _empty(shape: tuple[int, int]) -> np.ndarray:
    return np.zeros(shape, dtype=np.float32)


def _soft_polygon(shape: tuple[int, int], points: np.ndarray, feather: float = 15.0) -> np.ndarray:
    if points is None or len(points) < 3:
        return _empty(shape)
    height, width = shape
    pts = points[:, :2].astype(np.float32)
    pts[:, 0] = np.clip(pts[:, 0], 0, width - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, height - 1)
    hull = cv2.convexHull(pts.astype(np.int32))
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)
    soft = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), feather) / 255.0
    return np.clip(soft, 0.0, 1.0).astype(np.float32)


def _soft_ellipse(
    shape: tuple[int, int],
    center: tuple[float, float],
    radius_x: float,
    radius_y: float,
    feather: float = 10.0,
) -> np.ndarray:
    height, width = shape
    mask = np.zeros((height, width), dtype=np.uint8)
    cx = int(np.clip(center[0], 0, width - 1))
    cy = int(np.clip(center[1], 0, height - 1))
    rx = max(1, int(radius_x))
    ry = max(1, int(radius_y))
    cv2.ellipse(mask, (cx, cy), (rx, ry), 0, 0, 360, 255, -1, cv2.LINE_AA)
    soft = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), feather) / 255.0
    return np.clip(soft, 0.0, 1.0).astype(np.float32)


def _face_width(landmarks: FaceLandmarks, width: int) -> float:
    pts = landmarks.pixels
    if len(pts) > 454:
        return max(24.0, abs(float(pts[454, 0]) - float(pts[234, 0])))
    return max(24.0, landmarks.bbox[2] or width * 0.55)


def face_mask(shape: tuple[int, int], landmarks: FaceLandmarks, feather: float = 18.0) -> np.ndarray:
    """Return a soft full-face mask."""
    return _soft_polygon(shape, landmarks.face_oval, feather)


def forehead_mask(shape: tuple[int, int], landmarks: FaceLandmarks) -> np.ndarray:
    """Return a feathered forehead mask."""
    x, y, w, h = landmarks.bbox
    top = max(0.0, y - h * 0.04)
    bottom = y + h * 0.32
    points = np.array(
        [[x + w * 0.20, top], [x + w * 0.80, top], [x + w * 0.74, bottom], [x + w * 0.26, bottom]],
        dtype=np.float32,
    )
    return _soft_polygon(shape, points, 16.0) * face_mask(shape, landmarks, 22.0)


def eye_corner_mask(shape: tuple[int, int], landmarks: FaceLandmarks) -> np.ndarray:
    """Return soft masks around outer eye corners."""
    fw = _face_width(landmarks, shape[1])
    masks = []
    for eye, side in ((landmarks.left_eye, -1.0), (landmarks.right_eye, 1.0)):
        if len(eye) == 0:
            continue
        outer = eye[np.argmin(eye[:, 0])] if side < 0 else eye[np.argmax(eye[:, 0])]
        center = (float(outer[0] + side * fw * 0.035), float(outer[1]))
        masks.append(_soft_ellipse(shape, center, fw * 0.12, fw * 0.09, 9.0))
    return np.clip(sum(masks, _empty(shape)), 0.0, 1.0)


def nasolabial_mask(shape: tuple[int, int], landmarks: FaceLandmarks) -> np.ndarray:
    """Return soft masks for the nasolabial folds."""
    pts = landmarks.pixels
    if len(pts) < 426:
        return _empty(shape)
    fw = _face_width(landmarks, shape[1])
    centers = [(pts[186, 0], pts[186, 1] + fw * 0.10), (pts[410, 0], pts[410, 1] + fw * 0.10)]
    masks = [_soft_ellipse(shape, (float(x), float(y)), fw * 0.10, fw * 0.23, 10.0) for x, y in centers]
    return np.clip(sum(masks, _empty(shape)), 0.0, 1.0) * face_mask(shape, landmarks, 18.0)


def cheek_mask(shape: tuple[int, int], landmarks: FaceLandmarks) -> np.ndarray:
    """Return soft cheek masks."""
    fw = _face_width(landmarks, shape[1])
    pts = landmarks.pixels
    if len(pts) < 346:
        return _empty(shape)
    centers = [(pts[116, 0], pts[116, 1] + fw * 0.05), (pts[345, 0], pts[345, 1] + fw * 0.05)]
    masks = [_soft_ellipse(shape, (float(x), float(y)), fw * 0.19, fw * 0.18, 13.0) for x, y in centers]
    return np.clip(sum(masks, _empty(shape)), 0.0, 1.0) * face_mask(shape, landmarks, 18.0)


def jaw_mask(shape: tuple[int, int], landmarks: FaceLandmarks) -> np.ndarray:
    """Return a soft jaw and lower-face mask."""
    jaw = landmarks.jaw
    if len(jaw) < 3:
        return _empty(shape)
    lower = jaw[jaw[:, 1] >= np.percentile(jaw[:, 1], 45)]
    return _soft_polygon(shape, lower, 18.0)


def lip_mask(shape: tuple[int, int], landmarks: FaceLandmarks) -> np.ndarray:
    """Return a feathered lip mask."""
    return _soft_polygon(shape, landmarks.lips, 5.0)


def under_eye_mask(shape: tuple[int, int], landmarks: FaceLandmarks) -> np.ndarray:
    """Return soft under-eye masks."""
    fw = _face_width(landmarks, shape[1])
    masks = []
    for eye in (landmarks.left_eye, landmarks.right_eye):
        if len(eye) == 0:
            continue
        center = (float(np.mean(eye[:, 0])), float(np.max(eye[:, 1]) + fw * 0.055))
        masks.append(_soft_ellipse(shape, center, fw * 0.14, fw * 0.065, 7.0))
    return np.clip(sum(masks, _empty(shape)), 0.0, 1.0)


def hair_mask(shape: tuple[int, int], landmarks: FaceLandmarks) -> np.ndarray:
    """Estimate a soft hair mask; returns zeros when the region is ambiguous."""
    height, width = shape
    x, y, w, h = landmarks.bbox
    if h <= 0 or w <= 0:
        return _empty(shape)
    hairline = max(0.0, y + h * 0.10)
    fade = max(18.0, h * 0.20)
    yy = np.arange(height, dtype=np.float32)[:, np.newaxis]
    y_weight = np.clip((hairline + fade - yy) / fade, 0.0, 1.0)
    x_mask = np.zeros((height, width), dtype=np.float32)
    x1 = max(0, int(x - w * 0.18))
    x2 = min(width, int(x + w * 1.18))
    x_mask[:, x1:x2] = 1.0
    mask = y_weight * x_mask
    return np.clip(cv2.GaussianBlur(mask, (0, 0), 16.0), 0.0, 1.0).astype(np.float32)


def build_region_masks(shape: tuple[int, int], landmarks: FaceLandmarks) -> dict[str, np.ndarray]:
    """Build all aging masks for a frame."""
    return {
        "face": face_mask(shape, landmarks),
        "forehead": forehead_mask(shape, landmarks),
        "eye_corners": eye_corner_mask(shape, landmarks),
        "nasolabial": nasolabial_mask(shape, landmarks),
        "cheeks": cheek_mask(shape, landmarks),
        "jaw": jaw_mask(shape, landmarks),
        "lips": lip_mask(shape, landmarks),
        "under_eyes": under_eye_mask(shape, landmarks),
        "hair": hair_mask(shape, landmarks),
    }
