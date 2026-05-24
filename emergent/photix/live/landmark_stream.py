"""
landmark_stream.py — real-time 478-point FaceMesh landmark detection.

Wraps the MediaPipe Tasks API in RunningMode.VIDEO so each detect() call
reuses tracker state from the previous frame. This is materially faster
than core.landmark_detector.detect_landmarks (which uses IMAGE mode and
restarts detection every call).

Falls back to None when no face is detected; callers must handle that
case (skip processing, show overlay, etc.).
"""
from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp

# Reuse the model file already downloaded by the static-image detector.
_MODEL_PATH = Path(__file__).resolve().parent.parent / "core" / "face_landmarker.task"

_BaseOptions        = mp.tasks.BaseOptions
_FaceLandmarker     = mp.tasks.vision.FaceLandmarker
_FaceLandmarkerOpts = mp.tasks.vision.FaceLandmarkerOptions
_RunningMode        = mp.tasks.vision.RunningMode


class StreamingLandmarker:
    """Per-frame 478-point landmark detector for video / webcam streams."""

    def __init__(self, min_detection_conf: float = 0.5,
                       min_presence_conf:  float = 0.5,
                       min_tracking_conf:  float = 0.5):
        if not _MODEL_PATH.exists():
            # Defer to the existing one-time downloader if the model is missing.
            from core.landmark_detector import _ensure_model
            _ensure_model()

        opts = _FaceLandmarkerOpts(
            base_options=_BaseOptions(model_asset_path=str(_MODEL_PATH)),
            running_mode=_RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=min_detection_conf,
            min_face_presence_confidence=min_presence_conf,
            min_tracking_confidence=min_tracking_conf,
        )
        self._lm = _FaceLandmarker.create_from_options(opts)
        self._t0 = time.monotonic()

    def detect(self, frame_bgr: np.ndarray) -> np.ndarray | None:
        """Return (N, 2) float32 pixel coordinates, or None if no face."""
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # Tasks API requires monotonically increasing millisecond timestamps.
        ts_ms = int((time.monotonic() - self._t0) * 1000)
        result = self._lm.detect_for_video(mp_img, ts_ms)

        if not result.face_landmarks:
            return None

        lm = result.face_landmarks[0]
        pts = np.array([[p.x * w, p.y * h] for p in lm], dtype=np.float32)
        pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
        return pts

    def close(self) -> None:
        try:
            self._lm.close()
        except Exception:
            pass

    def __enter__(self) -> "StreamingLandmarker":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
