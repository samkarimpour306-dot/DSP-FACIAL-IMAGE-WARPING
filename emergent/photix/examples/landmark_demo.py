"""
examples/landmark_demo.py
Side-by-side landmark visualisation demo.

Usage::

    cd emergent/photix
    python examples/landmark_demo.py path/to/face.jpg
    python examples/landmark_demo.py path/to/face.jpg --out result.png --style regions

The script saves a 1×3 Matplotlib figure:
    [original]  |  [mesh overlay]  |  [regions coloured]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Allow running from the examples/ directory directly
sys.path.insert(0, str(Path(__file__).parent.parent))

from landmarks.detector import LandmarkDetector

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    """Convert OpenCV BGR image to Matplotlib-compatible RGB."""
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _annotate_pose(ax: plt.Axes, yaw: float, pitch: float, roll: float) -> None:
    """Add a small pose annotation box to an Axes."""
    text = f"yaw {yaw:+.1f}°\npitch {pitch:+.1f}°\nroll {roll:+.1f}°"
    ax.text(
        0.02, 0.97, text,
        transform=ax.transAxes,
        va="top", ha="left",
        fontsize=7,
        color="white",
        bbox=dict(boxstyle="round,pad=0.3", fc="black", alpha=0.55),
    )


def _annotate_bbox(ax: plt.Axes, bbox: tuple[int, int, int, int]) -> None:
    """Draw the face bounding box as a dashed rectangle."""
    x, y, w, h = bbox
    rect = mpatches.Rectangle(
        (x, y), w, h,
        linewidth=1.5, edgecolor="lime", facecolor="none",
        linestyle="--",
    )
    ax.add_patch(rect)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_figure(
    original: np.ndarray,
    mesh_overlay: np.ndarray,
    regions_overlay: np.ndarray,
    face_info: str,
) -> plt.Figure:
    """Build and return the side-by-side figure.

    Args:
        original: BGR source image.
        mesh_overlay: BGR image with mesh landmarks drawn.
        regions_overlay: BGR image with region-colour landmarks drawn.
        face_info: String summary displayed in the figure title.

    Returns:
        Matplotlib Figure (not yet shown or saved).
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    fig.suptitle(face_info, fontsize=10, color="dimgray")

    panels = [
        (original,        "Original"),
        (mesh_overlay,    "Mesh overlay"),
        (regions_overlay, "Regions coloured"),
    ]

    for ax, (img, title) in zip(axes, panels):
        ax.imshow(_bgr_to_rgb(img))
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    return fig


def run(args: argparse.Namespace) -> None:
    """Load image, run detection, build figure, save / show.

    Args:
        args: Parsed command-line arguments.
    """
    det = LandmarkDetector(
        model="mediapipe",
        refine=True,
        max_faces=args.max_faces,
    )

    log.info("Loading image: %s", args.image)
    faces = det.detect_from_path(args.image)

    original = cv2.imread(str(args.image), cv2.IMREAD_COLOR)

    if not faces:
        log.warning("No face detected in %s", args.image)
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.imshow(_bgr_to_rgb(original))
        ax.set_title("No face detected", color="red")
        ax.axis("off")
    else:
        face = faces[0]
        log.info(
            "Face detected — bbox %s  yaw=%.1f°  pitch=%.1f°  roll=%.1f°",
            face.bbox, face.yaw, face.pitch, face.roll,
        )
        log.info("Landmark count: %d", len(face.pixels))

        mesh_img    = det.draw(original, face, style="mesh")
        regions_img = det.draw(original, face, style="regions")

        # Overlay bounding box on the mesh panel
        _annotate_bbox(plt.gca(), face.bbox)          # placeholder; attached below

        face_info = (
            f"Landmarks: {len(face.pixels)}  |  "
            f"Bbox: {face.bbox}  |  "
            f"Yaw {face.yaw:+.1f}°  Pitch {face.pitch:+.1f}°  Roll {face.roll:+.1f}°"
        )
        fig = build_figure(original, mesh_img, regions_img, face_info)

        # Attach bbox rectangle to the mesh panel (axes[1])
        ax_mesh = fig.axes[1]
        _annotate_bbox(ax_mesh, face.bbox)
        _annotate_pose(ax_mesh, face.yaw, face.pitch, face.roll)

        # Print region sizes
        for region in ("left_eye", "right_eye", "lips", "nose", "jaw", "face_oval"):
            pts = getattr(face, region)
            log.info("  %-15s  %d pts", region, len(pts))

    out_path = Path(args.out) if args.out else None
    if out_path:
        fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
        log.info("Saved to %s", out_path)
    else:
        plt.show()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Parsed namespace.
    """
    p = argparse.ArgumentParser(
        description="Facial landmark visualisation demo.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("image", type=Path, help="Path to input face image.")
    p.add_argument(
        "--out", default=None,
        help="Output path for saved figure (e.g. result.png). "
             "Displays interactively if omitted.",
    )
    p.add_argument(
        "--max-faces", type=int, default=1, dest="max_faces",
        help="Maximum number of faces to detect.",
    )
    return p.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
