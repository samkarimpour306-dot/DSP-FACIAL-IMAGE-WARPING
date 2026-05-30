"""
landmark_detector.py — FR-03, FR-04
MediaPipe Face Landmarker (Tasks API): 478-point landmark detection.
Replaces the legacy mp.solutions API removed in mediapipe >= 0.10.14.
"""
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp

_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)
_MODEL_PATH = Path(__file__).parent / "face_landmarker.task"
_DOT_COLOUR     = (50, 255, 120)
_CONTOUR_COLOUR = (30, 210, 90)

_BaseOptions        = mp.tasks.BaseOptions
_FaceLandmarker     = mp.tasks.vision.FaceLandmarker
_FaceLandmarkerOpts = mp.tasks.vision.FaceLandmarkerOptions
_RunningMode        = mp.tasks.vision.RunningMode

# Singleton — created once, reused across all detect_landmarks() calls.
_landmarker: _FaceLandmarker | None = None


def _ensure_model() -> str:
    """Download the face landmarker model on first use (~29 MB)."""
    if not _MODEL_PATH.exists():
        print("Downloading face_landmarker.task model (~29 MB) — one-time setup…")
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
        print("Model downloaded.")
    return str(_MODEL_PATH)


def _get_landmarker() -> _FaceLandmarker:
    global _landmarker
    if _landmarker is None:
        opts = _FaceLandmarkerOpts(
            base_options=_BaseOptions(model_asset_path=_ensure_model()),
            running_mode=_RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        _landmarker = _FaceLandmarker.create_from_options(opts)
    return _landmarker


def detect_landmarks(img: np.ndarray) -> np.ndarray | None:
    """FR-03: Detect facial landmarks.
    Returns float32 ndarray (N, 2) in pixel coordinates, or None."""
    h, w = img.shape[:2]
    rgb    = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    result = _get_landmarker().detect(mp_img)

    if not result.face_landmarks:
        return None

    lm  = result.face_landmarks[0]
    pts = np.array([[p.x * w, p.y * h] for p in lm], dtype=np.float32)
    pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
    return pts


# Load face mesh contour connections for the overlay (indices match the 478-pt model).
try:
    from mediapipe.python.solutions.face_mesh_connections import FACEMESH_CONTOURS
    _CONNECTIONS: frozenset[tuple[int, int]] = frozenset(FACEMESH_CONTOURS)
except ImportError:
    _CONNECTIONS = frozenset()


def draw_landmarks(
    img: np.ndarray,
    landmarks: np.ndarray | None,
    visible: bool = True,
) -> np.ndarray:
    """FR-04: Toggle landmark overlay. Returns a copy (original unchanged)."""
    out = img.copy()
    if not visible or landmarks is None:
        return out

    n = len(landmarks)
    H, W = img.shape[:2]

    # Draw face contour connections beneath the dots.
    for (i, j) in _CONNECTIONS:
        if i >= n or j >= n:
            continue
        p1 = (int(np.clip(landmarks[i, 0], 0, W - 1)),
              int(np.clip(landmarks[i, 1], 0, H - 1)))
        p2 = (int(np.clip(landmarks[j, 0], 0, W - 1)),
              int(np.clip(landmarks[j, 1], 0, H - 1)))
        cv2.line(out, p1, p2, _CONTOUR_COLOUR, 1, cv2.LINE_AA)

    # Landmark dots on top of the lines.
    for pt in landmarks:
        x = int(np.clip(pt[0], 0, W - 1))
        y = int(np.clip(pt[1], 0, H - 1))
        cv2.circle(out, (x, y), 1, _DOT_COLOUR, -1, cv2.LINE_AA)

    return out
