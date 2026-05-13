"""Run Replicate SAM aging and de-aging on one image.

Usage:
    python examples/replicate_age_transform.py path/to/face.jpg
    python examples/replicate_age_transform.py path/to/face.jpg --younger-age 20 --older-age 70
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.replicate_aging import transform_age_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Replicate SAM age transformation demo.")
    parser.add_argument("image", type=Path, help="Input face image.")
    parser.add_argument("--younger-age", type=int, default=20, help="Target age for de-aging.")
    parser.add_argument("--older-age", type=int, default=70, help="Target age for aging.")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs"), help="Output directory.")
    args = parser.parse_args()

    stem = args.image.stem
    younger_path = args.out_dir / f"{stem}_younger_{args.younger_age}.jpg"
    older_path = args.out_dir / f"{stem}_older_{args.older_age}.jpg"

    print(f"Processing {args.image} -> target age {args.younger_age}...")
    transform_age_file(args.image, args.younger_age, younger_path)
    print(f"Saved -> {younger_path}")

    print(f"Processing {args.image} -> target age {args.older_age}...")
    transform_age_file(args.image, args.older_age, older_path)
    print(f"Saved -> {older_path}")


if __name__ == "__main__":
    main()
