"""
MediaPipe FaceMesh landmark index groups.

All indices reference the 478-point model produced when
``refine_landmarks=True`` (468 face-mesh points + 10 iris points).
With ``refine_landmarks=False`` only the first 468 points are available;
the LEFT_IRIS / RIGHT_IRIS groups will be out-of-range and are ignored
automatically by FaceLandmarks region accessors.

Index visualiser: https://github.com/google/mediapipe/blob/master/docs/solutions/face_mesh.md
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Eyes
# ---------------------------------------------------------------------------

#: Full left-eye contour (upper + lower lid, tear ducts, corners).
LEFT_EYE: list[int] = [
    33, 7, 163, 144, 145, 153, 154, 155, 133,
    173, 157, 158, 159, 160, 161, 246,
]

#: Full right-eye contour (mirrored equivalent of LEFT_EYE).
RIGHT_EYE: list[int] = [
    263, 249, 390, 373, 374, 380, 381, 382, 362,
    398, 384, 385, 386, 387, 388, 466,
]

# ---------------------------------------------------------------------------
# Eyebrows
# ---------------------------------------------------------------------------

#: Left eyebrow, inner → outer arch.
LEFT_EYEBROW: list[int] = [46, 53, 52, 65, 55, 70, 63, 105, 66, 107]

#: Right eyebrow, inner → outer arch.
RIGHT_EYEBROW: list[int] = [276, 283, 282, 295, 285, 300, 293, 334, 296, 336]

# ---------------------------------------------------------------------------
# Lips
# ---------------------------------------------------------------------------

#: Outer lip boundary (Cupid's bow, philtrum columns, corners, lower lip).
LIPS_OUTER: list[int] = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
    291, 409, 270, 269, 267, 0, 37, 39, 40, 185,
]

#: Inner lip boundary (teeth-line ring).
LIPS_INNER: list[int] = [
    78, 95, 88, 178, 87, 14, 317, 402, 318, 324,
    308, 415, 310, 311, 312, 13, 82, 81, 80, 191,
]

#: Combined outer + inner lip points.
LIPS: list[int] = LIPS_OUTER + LIPS_INNER

# ---------------------------------------------------------------------------
# Nose
# ---------------------------------------------------------------------------

#: Nose bridge (root → tip) and lateral alar base.
NOSE: list[int] = [
    168, 6, 197, 4, 1, 2, 5,          # bridge and tip
    19, 94, 195,                        # septum / columella
    64, 60, 99, 97, 2,                  # left nostril contour
    326, 327, 294, 278, 344,            # right nostril contour
    45, 51, 281, 363,                   # alar attachment
]

# ---------------------------------------------------------------------------
# Jaw and face oval
# ---------------------------------------------------------------------------

#: Lower jawline landmark ring (chin + mandible edges).
JAW: list[int] = [
    152, 148, 176, 149, 150, 136, 172, 58, 132, 93,
    234, 127, 162, 21, 54, 103, 67, 109,
    338, 297, 332, 284, 251, 389, 356, 454, 323,
    361, 288, 397, 365, 379, 378, 400, 377,
]

#: Full face oval / silhouette contour (forehead → chin ring).
FACE_OVAL: list[int] = [
    10, 338, 297, 332, 284, 251, 389, 356, 454,
    323, 361, 288, 397, 365, 379, 378, 400, 377,
    152, 148, 176, 149, 150, 136, 172, 58, 132,
    93, 234, 127, 162, 21, 54, 103, 67, 109,
]

# ---------------------------------------------------------------------------
# Iris  (only present when refine_landmarks=True)
# ---------------------------------------------------------------------------

#: Left iris centre + four cardinal points (indices 468–472).
LEFT_IRIS: list[int] = [468, 469, 470, 471, 472]

#: Right iris centre + four cardinal points (indices 473–477).
RIGHT_IRIS: list[int] = [473, 474, 475, 476, 477]

# ---------------------------------------------------------------------------
# Pose-estimation anchor set
# ---------------------------------------------------------------------------

#: Six stable 2-D ↔ 3-D correspondences used by ``cv2.solvePnP``.
#: Order: nose-tip, chin, left-eye-corner, right-eye-corner,
#:        left-mouth-corner, right-mouth-corner.
POSE_KEYPOINTS: list[int] = [1, 152, 33, 263, 61, 291]

#: Corresponding 3-D points in a generic face model (millimetres).
import numpy as np  # noqa: E402 — kept with the data it describes

POSE_MODEL_3D: np.ndarray = np.array([
    [0.0,     0.0,     0.0],       # 1   nose tip
    [0.0,  -330.0,   -65.0],       # 152 chin
    [-225.0,  170.0, -135.0],      # 33  left eye left corner
    [225.0,   170.0, -135.0],      # 263 right eye right corner
    [-150.0, -150.0, -125.0],      # 61  left mouth corner
    [150.0,  -150.0, -125.0],      # 291 right mouth corner
], dtype=np.float64)
