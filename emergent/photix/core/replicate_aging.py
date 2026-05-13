"""
core/replicate_aging.py
Cloud-backed age transformation via the Replicate SAM model.

The SAM model uses a target apparent age, so the same function supports
both aging and de-aging:
    - lower target_age -> younger result
    - higher target_age -> older result
"""

from __future__ import annotations

import io
import os
from collections.abc import Iterable
from pathlib import Path

import cv2
import numpy as np

_MODEL = "yuval-alaluf/sam:e947ce080d4c8c9a5769e7c7aadfb5df143aba7c972309c01c671dca8049087d"


def _load_dotenv_if_available() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    # Always load from photix/.env regardless of CWD
    load_dotenv(Path(__file__).parent.parent / ".env")


def _require_replicate():
    try:
        import replicate
    except ImportError as exc:
        raise ImportError(
            "replicate is required for SAM aging modes. "
            "Install project dependencies with: pip install -r requirements.txt"
        ) from exc
    return replicate


def _require_token() -> None:
    _load_dotenv_if_available()
    token = os.getenv("REPLICATE_API_TOKEN")
    if not token:
        raise EnvironmentError(
            "REPLICATE_API_TOKEN is not set. "
            "Add it to photix/.env as REPLICATE_API_TOKEN=r8_..."
        )
    os.environ["REPLICATE_API_TOKEN"] = token


def _first_output_url(output) -> str:
    if output is None:
        raise RuntimeError("Replicate returned no output.")

    if isinstance(output, str):
        return output

    if hasattr(output, "url"):
        return str(output.url)

    if isinstance(output, Iterable):
        for item in output:
            if hasattr(item, "url"):
                return str(item.url)
            return str(item)

    return str(output)


def _download_image(url: str, timeout: int = 120) -> bytes:
    try:
        import requests
    except ImportError as exc:
        raise ImportError(
            "requests is required for SAM aging modes. "
            "Install project dependencies with: pip install -r requirements.txt"
        ) from exc

    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return response.content


def transform_age_file(
    image_path: str | Path,
    target_age: int,
    output_path: str | Path,
) -> Path:
    """Transform an image file to a target apparent age and save the result."""
    _require_token()
    replicate = _require_replicate()

    image_path = Path(image_path)
    output_path = Path(output_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Input image not found: {image_path}")

    with image_path.open("rb") as image_file:
        output = replicate.run(
            _MODEL,
            input={
                "image": image_file,
                "target_age": str(int(target_age)),
            },
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(_download_image(_first_output_url(output)))
    return output_path


def apply_age_replicate(
    img: np.ndarray,
    target_age: int,
) -> np.ndarray:
    """Transform a BGR image to a target apparent age using Replicate SAM."""
    _require_token()
    replicate = _require_replicate()

    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise RuntimeError("cv2.imencode failed; could not convert image to JPEG.")
    image_bytes = io.BytesIO(buf.tobytes())

    output = replicate.run(
        _MODEL,
        input={
            "image": image_bytes,
            "target_age": str(int(target_age)),
        },
    )

    result_bytes = np.frombuffer(_download_image(_first_output_url(output)), dtype=np.uint8)
    result = cv2.imdecode(result_bytes, cv2.IMREAD_COLOR)
    if result is None:
        raise RuntimeError("Could not decode the image returned by Replicate.")

    if result.shape[:2] != img.shape[:2]:
        result = cv2.resize(
            result,
            (img.shape[1], img.shape[0]),
            interpolation=cv2.INTER_LANCZOS4,
        )

    return result


def apply_deaging_replicate(
    img: np.ndarray,
    target_age: int = 25,
) -> np.ndarray:
    """Backward-compatible helper for callers that only need de-aging."""
    return apply_age_replicate(img, target_age)
