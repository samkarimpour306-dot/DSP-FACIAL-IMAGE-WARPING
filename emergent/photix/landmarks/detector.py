"""
landmarks/detector.py
Facial landmark detection with a pluggable backend architecture.

Usage::

    from landmarks.detector import LandmarkDetector

    det = LandmarkDetector(model="mediapipe", refine=True)
    faces = det.detect(bgr_image)          # List[FaceLandmarks]
    overlay = det.draw(bgr_image, faces[0], style="mesh")
"""

from __future__ import annotations

import logging
import sys
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import List

import cv2
import numpy as np

from utils.landmark_indices import (
    FACE_OVAL,
    JAW,
    LEFT_EYE,
    LEFT_EYEBROW,
    LEFT_IRIS,
    LIPS,
    NOSE,
    POSE_KEYPOINTS,
    POSE_MODEL_3D,
    RIGHT_EYE,
    RIGHT_EYEBROW,
    RIGHT_IRIS,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Minimum image dimension before detection upscaling kicks in
# ---------------------------------------------------------------------------
_MIN_DETECT_DIM = 64
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)
_MODEL_PATH = Path(__file__).resolve().parent.parent / "core" / "face_landmarker.task"


def _ensure_task_model() -> str:
    if not _MODEL_PATH.exists():
        log.info("Downloading MediaPipe face landmarker model to %s.", _MODEL_PATH)
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
    return str(_MODEL_PATH)


# ---------------------------------------------------------------------------
# FaceLandmarks dataclass
# ---------------------------------------------------------------------------

@dataclass
class FaceLandmarks:
    """All landmark data for a single detected face.

    Attributes:
        normalized: Raw landmark coordinates, shape ``(N, 3)`` float32,
            values in ``[0, 1]`` (x, y, z).  N is 468 without iris
            refinement or 478 with it.
        pixels: Landmark positions projected to pixel space, shape
            ``(N, 2)`` float32.
        image_size: ``(height, width)`` of the source image.
        bbox: Axis-aligned bounding box ``(x, y, w, h)`` in pixels,
            derived from the convex extent of all landmarks.
        yaw: Head rotation around the vertical axis (degrees).
            Positive → face turned left from the camera's perspective.
        pitch: Head rotation around the horizontal axis (degrees).
            Positive → face tilted up.
        roll: Head rotation around the depth axis (degrees).
            Positive → face tilted right (clockwise).
    """

    normalized: np.ndarray
    pixels: np.ndarray
    image_size: tuple[int, int]
    bbox: tuple[int, int, int, int]
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0

    # ------------------------------------------------------------------
    # Named region accessors — return pixel coords for the region
    # ------------------------------------------------------------------

    def _region(self, indices: list[int]) -> np.ndarray:
        """Return pixel coordinates for *indices*, skipping out-of-range."""
        n = len(self.pixels)
        valid = [i for i in indices if i < n]
        return self.pixels[valid]

    @property
    def left_eye(self) -> np.ndarray:
        """Pixel coords of the left-eye contour, shape ``(M, 2)``."""
        return self._region(LEFT_EYE)

    @property
    def right_eye(self) -> np.ndarray:
        """Pixel coords of the right-eye contour, shape ``(M, 2)``."""
        return self._region(RIGHT_EYE)

    @property
    def left_eyebrow(self) -> np.ndarray:
        """Pixel coords of the left eyebrow, shape ``(M, 2)``."""
        return self._region(LEFT_EYEBROW)

    @property
    def right_eyebrow(self) -> np.ndarray:
        """Pixel coords of the right eyebrow, shape ``(M, 2)``."""
        return self._region(RIGHT_EYEBROW)

    @property
    def lips(self) -> np.ndarray:
        """Pixel coords of all lip landmarks (outer + inner), shape ``(M, 2)``."""
        return self._region(LIPS)

    @property
    def nose(self) -> np.ndarray:
        """Pixel coords of nose region landmarks, shape ``(M, 2)``."""
        return self._region(NOSE)

    @property
    def jaw(self) -> np.ndarray:
        """Pixel coords of the lower jawline ring, shape ``(M, 2)``."""
        return self._region(JAW)

    @property
    def face_oval(self) -> np.ndarray:
        """Pixel coords of the full face silhouette contour, shape ``(M, 2)``."""
        return self._region(FACE_OVAL)

    @property
    def left_iris(self) -> np.ndarray:
        """Pixel coords of the left iris (requires ``refine=True``), shape ``(5, 2)``
        or empty array when iris points are absent."""
        return self._region(LEFT_IRIS)

    @property
    def right_iris(self) -> np.ndarray:
        """Pixel coords of the right iris (requires ``refine=True``), shape ``(5, 2)``
        or empty array when iris points are absent."""
        return self._region(RIGHT_IRIS)


# ---------------------------------------------------------------------------
# Pose estimation helper
# ---------------------------------------------------------------------------

def _estimate_pose(
    pixels: np.ndarray,
    image_size: tuple[int, int],
) -> tuple[float, float, float]:
    """Estimate head yaw, pitch, and roll from 6 stable landmarks.

    Args:
        pixels: All landmark pixel coordinates, shape ``(N, 2)``.
        image_size: ``(height, width)`` of the source image.

    Returns:
        ``(yaw, pitch, roll)`` in degrees.  Returns ``(0.0, 0.0, 0.0)``
        if ``solvePnP`` fails or indices are out of range.
    """
    H, W = image_size
    n = len(pixels)

    if any(i >= n for i in POSE_KEYPOINTS):
        log.debug("Pose estimation skipped: landmark indices out of range.")
        return 0.0, 0.0, 0.0

    img_pts = pixels[POSE_KEYPOINTS].astype(np.float64)

    focal = float(W)
    cam = np.array([[focal, 0.0, W / 2.0],
                    [0.0, focal, H / 2.0],
                    [0.0,  0.0,     1.0]], dtype=np.float64)

    ok, rvec, _ = cv2.solvePnP(
        POSE_MODEL_3D, img_pts, cam, np.zeros((4, 1)),
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return 0.0, 0.0, 0.0

    rmat, _ = cv2.Rodrigues(rvec)

    # Tait-Bryan ZYX decomposition
    sy = float(np.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2))
    if sy > 1e-6:
        roll  = float(np.degrees(np.arctan2( rmat[2, 1], rmat[2, 2])))
        pitch = float(np.degrees(np.arctan2(-rmat[2, 0], sy)))
        yaw   = float(np.degrees(np.arctan2( rmat[1, 0], rmat[0, 0])))
    else:
        roll  = float(np.degrees(np.arctan2(-rmat[1, 2], rmat[1, 1])))
        pitch = float(np.degrees(np.arctan2(-rmat[2, 0], sy)))
        yaw   = 0.0

    return yaw, pitch, roll


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _get_connections() -> dict[str, frozenset[tuple[int, int]]]:
    """Load MediaPipe face-mesh connection sets (cached, import-once).

    Returns:
        Dict mapping ``"tesselation"``, ``"contours"``, ``"irises"`` to
        their respective frozensets of ``(start, end)`` index pairs.
    """
    try:
        from mediapipe.python.solutions.face_mesh_connections import (
            FACEMESH_CONTOURS,
            FACEMESH_IRISES,
            FACEMESH_TESSELATION,
        )
        return {
            "tesselation": frozenset(FACEMESH_TESSELATION),
            "contours":    frozenset(FACEMESH_CONTOURS),
            "irises":      frozenset(FACEMESH_IRISES),
        }
    except ImportError:
        log.warning("face_mesh_connections not available; mesh/contour drawing disabled.")
        return {"tesselation": frozenset(), "contours": frozenset(), "irises": frozenset()}


def _draw_connections(
    out: np.ndarray,
    pixels: np.ndarray,
    connections: frozenset[tuple[int, int]],
    color: tuple[int, int, int],
    thickness: int = 1,
) -> None:
    """Draw a set of landmark connections onto *out* in-place."""
    n = len(pixels)
    H, W = out.shape[:2]
    for i, j in connections:
        if i >= n or j >= n:
            continue
        p1 = (int(np.clip(pixels[i, 0], 0, W - 1)),
              int(np.clip(pixels[i, 1], 0, H - 1)))
        p2 = (int(np.clip(pixels[j, 0], 0, W - 1)),
              int(np.clip(pixels[j, 1], 0, H - 1)))
        cv2.line(out, p1, p2, color, thickness, cv2.LINE_AA)


def _draw_dots(
    out: np.ndarray,
    pixels: np.ndarray,
    color: tuple[int, int, int],
    radius: int = 1,
) -> None:
    """Draw a filled dot at each landmark position onto *out* in-place."""
    H, W = out.shape[:2]
    for pt in pixels:
        x = int(np.clip(pt[0], 0, W - 1))
        y = int(np.clip(pt[1], 0, H - 1))
        cv2.circle(out, (x, y), radius, color, -1, cv2.LINE_AA)


# Region → BGR color for "regions" style
_REGION_COLORS: list[tuple[list[int], tuple[int, int, int]]] = [
    (FACE_OVAL,     (160, 160, 160)),  # grey
    (LEFT_EYE,      (255,  80,  80)),  # blue
    (RIGHT_EYE,     (255,  80,  80)),  # blue
    (LEFT_EYEBROW,  ( 60, 180,  60)),  # green
    (RIGHT_EYEBROW, ( 60, 180,  60)),  # green
    (NOSE,          (  0, 220, 220)),  # yellow-cyan
    (LIPS,          ( 60,  60, 220)),  # red
    (LEFT_IRIS,     (200, 100,   0)),  # orange
    (RIGHT_IRIS,    (200, 100,   0)),  # orange
]


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BaseDetector(ABC):
    """Abstract interface every backend must implement.

    Concrete subclasses should initialise their model in ``__init__``
    exactly once and keep it alive for the lifetime of the object.
    """

    @abstractmethod
    def detect(self, image: np.ndarray) -> List[FaceLandmarks]:
        """Detect all faces in *image* and return their landmarks.

        Args:
            image: BGR, RGB, grayscale, or RGBA uint8 array.

        Returns:
            A possibly-empty list of :class:`FaceLandmarks`, one per face.
        """

    @abstractmethod
    def draw(
        self,
        image: np.ndarray,
        landmarks: FaceLandmarks,
        style: str = "mesh",
    ) -> np.ndarray:
        """Render a landmark overlay on a copy of *image*.

        Args:
            image: Source BGR image.
            landmarks: Detection result for the face to annotate.
            style: One of ``"mesh"``, ``"points"``, ``"contours"``,
                ``"regions"``.

        Returns:
            Annotated BGR image (original is not modified).
        """


# ---------------------------------------------------------------------------
# MediaPipe backend
# ---------------------------------------------------------------------------

class MediaPipeDetector(BaseDetector):
    """MediaPipe Face Landmarker Tasks backend (primary implementation).

    The FaceLandmarker model is created once in ``__init__`` and reused across
    all ``detect()`` calls — no per-image re-initialisation overhead.

    Args:
        refine: If ``True``, enables iris refinement (478 points instead
            of 468).  Adds iris + pupil landmarks useful for eye-gaze.
        max_faces: Maximum number of faces to detect per image.
        min_detection_confidence: Detection confidence threshold in
            ``[0, 1]``.
    """

    def __init__(
        self,
        refine: bool = True,
        max_faces: int = 1,
        min_detection_confidence: float = 0.5,
    ) -> None:
        try:
            import mediapipe as mp
        except ImportError as exc:
            raise ImportError(
                "mediapipe is required for MediaPipeDetector. "
                "Install it with: pip install mediapipe"
            ) from exc

        self._refine = refine
        base_options = mp.tasks.BaseOptions(model_asset_path=_ensure_task_model())
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_faces=max_faces,
            min_face_detection_confidence=min_detection_confidence,
            min_face_presence_confidence=min_detection_confidence,
            min_tracking_confidence=0.5,
        )
        self._face_landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)
        log.debug(
            "MediaPipeDetector initialised (refine=%s, max_faces=%d, conf=%.2f)",
            refine, max_faces, min_detection_confidence,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, image: np.ndarray) -> List[FaceLandmarks]:
        """Detect faces in *image*.

        Handles grayscale, RGBA, BGR, and RGB inputs transparently.
        Images smaller than 64 px on either dimension are upscaled
        before detection and results are mapped back to original coords.

        Args:
            image: Input image as a numpy array.

        Returns:
            List of :class:`FaceLandmarks`, empty when no face is found.
        """
        if not isinstance(image, np.ndarray):
            raise TypeError(f"image must be a numpy array, got {type(image).__name__}")

        orig_h, orig_w = image.shape[:2]
        rgb, scale = self._preprocess(image)
        proc_h, proc_w = rgb.shape[:2]

        import mediapipe as mp

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._face_landmarker.detect(mp_image)

        if not result.face_landmarks:
            log.debug("No face detected in image of size %dx%d.", orig_w, orig_h)
            return []

        faces = []
        for face_lm in result.face_landmarks:
            lm = self._build_landmarks(
                face_lm,
                proc_h, proc_w,
                orig_h, orig_w,
                scale,
            )
            faces.append(lm)

        log.debug("Detected %d face(s).", len(faces))
        return faces

    def draw(
        self,
        image: np.ndarray,
        landmarks: FaceLandmarks,
        style: str = "mesh",
    ) -> np.ndarray:
        """Draw a landmark overlay on a copy of *image*.

        Args:
            image: BGR source image.
            landmarks: Face to annotate.
            style: ``"mesh"`` — full tesselation (grey lines + green dots);
                ``"points"`` — dots only;
                ``"contours"`` — facial contour lines + dots;
                ``"regions"`` — colour-coded anatomical regions.

        Returns:
            Annotated BGR copy.

        Raises:
            ValueError: If *style* is not one of the supported values.
        """
        _VALID_STYLES = {"mesh", "points", "contours", "regions"}
        if style not in _VALID_STYLES:
            raise ValueError(f"style must be one of {_VALID_STYLES}, got {style!r}")

        out  = image.copy()
        pts  = landmarks.pixels
        conn = _get_connections()

        if style == "mesh":
            _draw_connections(out, pts, conn["tesselation"], (80, 80, 80))
            if self._refine:
                _draw_connections(out, pts, conn["irises"], (0, 200, 200))
            _draw_dots(out, pts, (50, 255, 120), radius=1)

        elif style == "points":
            _draw_dots(out, pts, (50, 255, 120), radius=2)

        elif style == "contours":
            _draw_connections(out, pts, conn["contours"], (30, 210, 90), thickness=1)
            if self._refine:
                _draw_connections(out, pts, conn["irises"], (0, 200, 200))
            _draw_dots(out, pts, (50, 255, 120), radius=1)

        elif style == "regions":
            for indices, color in _REGION_COLORS:
                n = len(pts)
                valid = [i for i in indices if i < n]
                if valid:
                    _draw_dots(out, pts[valid], color, radius=2)

        return out

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _preprocess(image: np.ndarray) -> tuple[np.ndarray, float]:
        """Normalise channel order and upscale tiny images.

        Args:
            image: Raw input (any channel order, any dtype).

        Returns:
            ``(rgb_image, scale)`` where *scale* is the upscaling factor
            applied (1.0 when no upscaling was needed).
        """
        # Normalise to uint8
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)

        # Normalise channels → RGB
        if image.ndim == 2:
            log.debug("Grayscale input; converting to RGB.")
            rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            log.debug("RGBA input; stripping alpha channel.")
            rgb = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        else:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Upscale very small images so MediaPipe can find a face
        h, w = rgb.shape[:2]
        scale = 1.0
        if min(h, w) < _MIN_DETECT_DIM:
            scale = _MIN_DETECT_DIM / min(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            rgb = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            log.warning(
                "Input image is very small (%dx%d); upscaled to %dx%d for detection.",
                w, h, new_w, new_h,
            )

        return rgb, scale

    @staticmethod
    def _build_landmarks(
        face_lm,
        proc_h: int, proc_w: int,
        orig_h: int, orig_w: int,
        scale: float,
    ) -> FaceLandmarks:
        """Convert a raw MediaPipe NormalizedLandmarkList to FaceLandmarks.

        Args:
            face_lm: MediaPipe landmark list for one face.
            proc_h: Height of the (possibly upscaled) image used for detection.
            proc_w: Width of the (possibly upscaled) image used for detection.
            orig_h: Height of the original input image.
            orig_w: Width of the original input image.
            scale: Upscaling factor; pixel coords are divided by this.

        Returns:
            Populated :class:`FaceLandmarks` instance.
        """
        pts = face_lm
        n = len(pts)

        normalized = np.array(
            [[p.x, p.y, p.z] for p in pts], dtype=np.float32
        )

        # Project to original image pixel space (undo any upscale)
        pixels = np.column_stack([
            normalized[:, 0] * proc_w / scale,
            normalized[:, 1] * proc_h / scale,
        ]).astype(np.float32)

        pixels[:, 0] = np.clip(pixels[:, 0], 0.0, orig_w - 1)
        pixels[:, 1] = np.clip(pixels[:, 1], 0.0, orig_h - 1)

        # Bounding box from landmark extents
        x0 = int(pixels[:, 0].min())
        y0 = int(pixels[:, 1].min())
        x1 = int(pixels[:, 0].max())
        y1 = int(pixels[:, 1].max())
        bbox = (x0, y0, x1 - x0, y1 - y0)

        yaw, pitch, roll = _estimate_pose(pixels, (orig_h, orig_w))

        return FaceLandmarks(
            normalized=normalized,
            pixels=pixels,
            image_size=(orig_h, orig_w),
            bbox=bbox,
            yaw=yaw,
            pitch=pitch,
            roll=roll,
        )


# ---------------------------------------------------------------------------
# Public facade
# ---------------------------------------------------------------------------

class LandmarkDetector:
    """Pluggable facial landmark detector.

    This class is the intended public entry-point.  It delegates all
    detection and drawing work to a backend that implements
    :class:`BaseDetector`.  Adding a new backend (e.g. dlib) only
    requires registering it in ``_BACKENDS``.

    Args:
        model: Backend identifier string.  Currently ``"mediapipe"``
            is the only supported value.
        refine: Pass ``True`` to enable iris refinement (478 pts).
        max_faces: Maximum number of faces to detect per call.
        min_detection_confidence: Confidence threshold in ``[0, 1]``.

    Example::

        det = LandmarkDetector()
        faces = det.detect(bgr_image)
        if faces:
            overlay = det.draw(bgr_image, faces[0], style="regions")
    """

    _BACKENDS: dict[str, type[BaseDetector]] = {
        "mediapipe": MediaPipeDetector,
        # "dlib": DlibDetector,  # ← plug in here when implemented
    }

    def __init__(
        self,
        model: str = "mediapipe",
        refine: bool = True,
        max_faces: int = 1,
        min_detection_confidence: float = 0.5,
    ) -> None:
        if model not in self._BACKENDS:
            raise ValueError(
                f"Unknown model {model!r}. "
                f"Available backends: {list(self._BACKENDS)}"
            )
        self._detector: BaseDetector = self._BACKENDS[model](
            refine=refine,
            max_faces=max_faces,
            min_detection_confidence=min_detection_confidence,
        )
        log.info("LandmarkDetector initialised with backend %r.", model)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def detect(self, image: np.ndarray) -> List[FaceLandmarks]:
        """Detect faces in a numpy image array.

        Args:
            image: BGR, grayscale, or RGBA uint8 array.

        Returns:
            List of :class:`FaceLandmarks` (empty if no face found).
        """
        return self._detector.detect(image)

    def detect_from_path(self, path: str | Path) -> List[FaceLandmarks]:
        """Load an image from *path* and detect faces.

        Args:
            path: Filesystem path to a JPEG, PNG, or BMP image.

        Returns:
            List of :class:`FaceLandmarks` (empty if no face found).

        Raises:
            FileNotFoundError: If *path* does not exist.
            OSError: If OpenCV cannot decode the file.
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Image not found: {p}")

        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            raise OSError(f"OpenCV could not decode image: {p}")

        return self.detect(img)

    def draw(
        self,
        image: np.ndarray,
        landmarks: FaceLandmarks,
        style: str = "mesh",
    ) -> np.ndarray:
        """Draw a landmark overlay on a copy of *image*.

        Args:
            image: BGR source image.
            landmarks: A :class:`FaceLandmarks` result to annotate.
            style: One of ``"mesh"``, ``"points"``, ``"contours"``,
                ``"regions"``.

        Returns:
            Annotated BGR copy (original is unchanged).
        """
        return self._detector.draw(image, landmarks, style)
