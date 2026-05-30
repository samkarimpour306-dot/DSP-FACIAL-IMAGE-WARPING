"""
tests/test_detector.py
pytest suite for landmarks.detector.

Run with::

    cd emergent/photix
    pytest tests/test_detector.py -v

Tests that require an actual face image use the ``face_image`` fixture.
If the fixture cannot load or download a face, those tests are skipped
automatically — all robustness tests run unconditionally.
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import pytest

# Ensure the photix package root is on sys.path when running directly.
sys.path.insert(0, str(Path(__file__).parent.parent))

from landmarks.detector import FaceLandmarks, LandmarkDetector

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def detector() -> LandmarkDetector:
    """Session-scoped detector — initialised once, reused across all tests."""
    return LandmarkDetector(model="mediapipe", refine=True, max_faces=2)


@pytest.fixture(scope="session")
def face_image() -> np.ndarray:
    """Try to load a real face image for detection-positive tests.

    Checks ``tests/data/face.jpg`` first; falls back to a small public
    domain portrait from Wikimedia Commons.  Skips if neither is available.
    """
    local = Path(__file__).parent / "data" / "face.jpg"
    if local.exists():
        img = cv2.imread(str(local), cv2.IMREAD_COLOR)
        if img is not None:
            return img

    _URL = (
        "https://upload.wikimedia.org/wikipedia/commons/thumb/"
        "1/14/Gatto_europeo4.jpg/200px-Gatto_europeo4.jpg"
    )
    # Fallback: try to fetch a portrait-style image for detection.
    # Using a known CC0 portrait crop stored on raw GitHub.
    _PORTRAIT_URL = (
        "https://raw.githubusercontent.com/ageitgey/face_recognition/"
        "master/examples/obama.jpg"
    )
    try:
        with urllib.request.urlopen(_PORTRAIT_URL, timeout=8) as resp:
            data = resp.read()
        arr = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is not None:
            return img
    except Exception:
        pass

    pytest.skip("No face image available (place one at tests/data/face.jpg).")


@pytest.fixture(scope="session")
def blank_image() -> np.ndarray:
    """Plain black 256×256 BGR image — contains no face."""
    return np.zeros((256, 256, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Robustness tests (never skipped)
# ---------------------------------------------------------------------------

class TestNoFace:
    def test_blank_returns_empty_list(self, detector, blank_image):
        results = detector.detect(blank_image)
        assert isinstance(results, list)
        assert len(results) == 0

    def test_random_noise_returns_empty_or_list(self, detector):
        rng   = np.random.default_rng(42)
        noise = rng.integers(0, 256, (128, 128, 3), dtype=np.uint8)
        results = detector.detect(noise)
        assert isinstance(results, list)  # must not crash

    def test_tiny_image_does_not_crash(self, detector):
        tiny = np.zeros((16, 16, 3), dtype=np.uint8)
        results = detector.detect(tiny)
        assert isinstance(results, list)


class TestInputNormalisation:
    def test_grayscale_input_does_not_crash(self, detector, blank_image):
        gray = cv2.cvtColor(blank_image, cv2.COLOR_BGR2GRAY)
        assert gray.ndim == 2
        results = detector.detect(gray)
        assert isinstance(results, list)

    def test_rgba_input_does_not_crash(self, detector, blank_image):
        rgba = cv2.cvtColor(blank_image, cv2.COLOR_BGR2BGRA)
        assert rgba.shape[2] == 4
        results = detector.detect(rgba)
        assert isinstance(results, list)

    def test_non_ndarray_raises_type_error(self, detector):
        with pytest.raises(TypeError):
            detector.detect([[0, 0, 0], [0, 0, 0]])  # type: ignore[arg-type]


class TestInvalidPath:
    def test_nonexistent_path_raises_file_not_found(self, detector):
        with pytest.raises(FileNotFoundError):
            detector.detect_from_path("/no/such/file/face.jpg")

    def test_nonexistent_pathlib_path_raises(self, detector):
        with pytest.raises(FileNotFoundError):
            detector.detect_from_path(Path("/no/such/file.png"))


class TestUnknownBackend:
    def test_unknown_model_raises_value_error(self):
        with pytest.raises(ValueError, match="dlib"):
            LandmarkDetector(model="dlib")


# ---------------------------------------------------------------------------
# Detection-positive tests (require face_image fixture)
# ---------------------------------------------------------------------------

class TestSingleFace:
    def test_returns_at_least_one_face(self, detector, face_image):
        results = detector.detect(face_image)
        assert len(results) >= 1

    def test_result_type_is_face_landmarks(self, detector, face_image):
        results = detector.detect(face_image)
        assert isinstance(results[0], FaceLandmarks)

    def test_pixels_shape(self, detector, face_image):
        lm = detector.detect(face_image)[0]
        assert lm.pixels.ndim == 2
        assert lm.pixels.shape[1] == 2
        assert lm.pixels.dtype == np.float32

    def test_normalized_shape(self, detector, face_image):
        lm = detector.detect(face_image)[0]
        assert lm.normalized.ndim == 2
        assert lm.normalized.shape[1] == 3
        # Values should be in [0, 1] for x and y (z may exceed slightly)
        assert lm.normalized[:, 0].min() >= 0.0
        assert lm.normalized[:, 0].max() <= 1.0

    def test_bbox_is_valid(self, detector, face_image):
        lm = detector.detect(face_image)[0]
        x, y, w, h = lm.bbox
        assert w > 0 and h > 0
        assert x >= 0 and y >= 0

    def test_image_size_matches(self, detector, face_image):
        lm = detector.detect(face_image)[0]
        H, W = face_image.shape[:2]
        assert lm.image_size == (H, W)

    def test_pose_angles_are_finite(self, detector, face_image):
        lm = detector.detect(face_image)[0]
        for angle in (lm.yaw, lm.pitch, lm.roll):
            assert np.isfinite(angle)

    def test_pixels_within_image_bounds(self, detector, face_image):
        lm = detector.detect(face_image)[0]
        H, W = face_image.shape[:2]
        assert lm.pixels[:, 0].max() <= W
        assert lm.pixels[:, 1].max() <= H


class TestRegionAccessors:
    """Each named region must return a 2-D float32 array with 2 columns."""

    _REGIONS = [
        "left_eye", "right_eye", "left_eyebrow", "right_eyebrow",
        "lips", "nose", "jaw", "face_oval",
    ]

    @pytest.fixture(autouse=True)
    def _lm(self, detector, face_image):
        results = detector.detect(face_image)
        self.lm = results[0]

    @pytest.mark.parametrize("region", _REGIONS)
    def test_region_shape(self, region):
        pts = getattr(self.lm, region)
        assert pts.ndim == 2, f"{region}: expected 2-D array"
        assert pts.shape[1] == 2, f"{region}: expected 2 columns (x, y)"
        assert pts.dtype == np.float32, f"{region}: expected float32"

    @pytest.mark.parametrize("region", _REGIONS)
    def test_region_non_empty(self, region):
        pts = getattr(self.lm, region)
        assert len(pts) > 0, f"{region}: expected non-empty"

    def test_iris_present_with_refine(self):
        # detector fixture has refine=True → 478 points → iris accessible
        assert len(self.lm.left_iris) == 5
        assert len(self.lm.right_iris) == 5


class TestMultipleFaces:
    def test_max_faces_respected(self, face_image):
        """With max_faces=1, at most 1 face is returned even in multi-face images."""
        det1 = LandmarkDetector(max_faces=1)
        results = det1.detect(face_image)
        assert len(results) <= 1

    def test_multi_face_detector_initialises(self):
        det = LandmarkDetector(max_faces=5)
        assert det is not None


class TestDrawStyles:
    _STYLES = ["mesh", "points", "contours", "regions"]

    @pytest.fixture(autouse=True)
    def _setup(self, detector, face_image):
        self.detector    = detector
        self.face_image  = face_image
        self.lm          = detector.detect(face_image)[0]

    @pytest.mark.parametrize("style", _STYLES)
    def test_draw_returns_same_shape(self, style):
        out = self.detector.draw(self.face_image, self.lm, style=style)
        assert out.shape == self.face_image.shape

    @pytest.mark.parametrize("style", _STYLES)
    def test_draw_does_not_modify_original(self, style):
        original = self.face_image.copy()
        self.detector.draw(self.face_image, self.lm, style=style)
        np.testing.assert_array_equal(self.face_image, original)

    def test_invalid_style_raises(self):
        with pytest.raises(ValueError, match="style"):
            self.detector.draw(self.face_image, self.lm, style="heatmap")
