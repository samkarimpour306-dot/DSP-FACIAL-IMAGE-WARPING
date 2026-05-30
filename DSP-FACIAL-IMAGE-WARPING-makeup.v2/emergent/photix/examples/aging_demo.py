"""Generate a 1x5 aging/de-aging demo figure.

Usage:
    python examples/aging_demo.py path/to/face.jpg --out aging_demo.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aging.transformer import AgeTransformer
from landmarks.detector import LandmarkDetector


def main() -> None:
    """Load a face image, transform it, and save a labelled comparison figure."""
    parser = argparse.ArgumentParser(description="Photix aging transformer demo.")
    parser.add_argument("image", type=Path, nargs="?", default=Path(__file__).with_name("reference_face.png"))
    parser.add_argument("--out", type=Path, default=Path("aging_demo.png"))
    args = parser.parse_args()

    image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if image is None:
        raise OSError(f"Could not read image: {args.image}")

    detector = LandmarkDetector(model="mediapipe", refine=True)
    faces = detector.detect(image)
    if not faces:
        raise RuntimeError("No face detected in input image.")

    transformer = AgeTransformer(seed=11, preserve_identity=True)
    deltas = [-20, -10, 0, 15, 30]
    labels = ["-20y", "-10y", "original", "+15y", "+30y"]
    outputs = [
        image if delta == 0 else transformer.transform(image, faces[0], delta).image
        for delta in deltas
    ]

    fig, axes = plt.subplots(1, 5, figsize=(14, 3.2))
    for ax, out, label in zip(axes, outputs, labels):
        ax.imshow(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
        ax.set_title(label)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(args.out, dpi=150, facecolor="white")


if __name__ == "__main__":
    main()
