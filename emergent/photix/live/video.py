"""
video.py — apply the full Photix pipeline to every frame of a video file.

Reads the input with cv2.VideoCapture, runs landmark detection +
``apply_pipeline`` per frame, and writes the result to ``results/`` with a
timestamped filename. Shows a tqdm progress bar.

This module never touches cv2.imshow — it's pure batch processing so it
can run head-less or be called from the Gradio UI / the desktop app.
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

_HERE = Path(__file__).resolve().parent
_PHOTIX_ROOT = _HERE.parent
if str(_PHOTIX_ROOT) not in sys.path:
    sys.path.insert(0, str(_PHOTIX_ROOT))

from live.pipeline        import LivePipeline, apply_pipeline
from live.landmark_stream import StreamingLandmarker


PROC_W, PROC_H = 640, 480
RESULTS_DIR = _PHOTIX_ROOT / "results"

_DEFAULT_FOURCC = cv2.VideoWriter_fourcc(*"mp4v")


def _resolve_output_path(
    input_path: Path,
    pipeline:   LivePipeline,
    out_dir:    Path | None,
) -> Path:
    """Build ``results/photix_<tag>_<stem>_<timestamp>.mp4``."""
    base = Path(out_dir) if out_dir else RESULTS_DIR
    base.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_stem = "".join(c if c.isalnum() or c in "-_" else "_"
                        for c in input_path.stem)[:40] or "video"

    tag_parts: list[str] = []
    if pipeline.aging == "age":   tag_parts.append("age")
    if pipeline.aging == "deage": tag_parts.append("deage")
    if pipeline.beard      != "none": tag_parts.append(f"beard-{pipeline.beard}")
    if pipeline.sunglasses != "none": tag_parts.append(f"glasses-{pipeline.sunglasses}")
    if pipeline.hair_style != "none": tag_parts.append(f"hair-{pipeline.hair_style}")
    if pipeline.smile      > 0.01: tag_parts.append("smile")
    if pipeline.eyebrow    > 0.01: tag_parts.append("brow")
    if pipeline.lip_wide   > 0.01: tag_parts.append("lip")
    if pipeline.face_slim  > 0.01: tag_parts.append("slim")
    if pipeline.lowpass_enabled and pipeline.lowpass_sigma > 0.5:
        tag_parts.append("lowpass")
    if pipeline.highpass_enabled and pipeline.highpass_sigma > 0.5:
        tag_parts.append("highpass")
    tag = "_".join(tag_parts) if tag_parts else "passthrough"

    return base / f"photix_{tag}_{safe_stem}_{ts}.mp4"


def process_video(
    input_path: str | Path,
    pipeline:   LivePipeline | None = None,
    output_path: str | Path | None = None,
    results_dir: str | Path | None = None,
    show_progress: bool = True,
) -> Path:
    """Process ``input_path`` frame-by-frame through ``pipeline``.

    Returns the path of the written video.
    """
    pipeline = pipeline or LivePipeline()

    in_path = Path(input_path)
    if not in_path.exists():
        raise FileNotFoundError(f"Input video not found: {in_path}")

    cap = cv2.VideoCapture(str(in_path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {in_path}")

    src_w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  or PROC_W
    src_h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or PROC_H
    fps    = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    out_size = (src_w, src_h)
    out_path = Path(output_path) if output_path else _resolve_output_path(
        in_path, pipeline, results_dir,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(str(out_path), _DEFAULT_FOURCC, fps, out_size)
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(
            f"OpenCV could not open VideoWriter for: {out_path}. "
            "Your build may be missing the mp4v encoder — try a .avi output."
        )

    bar = (tqdm(total=n_frames or None,
                desc=f"Photix · {pipeline.label()[:40]}",
                unit="f", dynamic_ncols=True)
           if show_progress else None)

    try:
        with StreamingLandmarker() as detector:
            while True:
                ok, frame = cap.read()
                if not ok or frame is None or not frame.size:
                    break

                proc = cv2.resize(frame, (PROC_W, PROC_H),
                                  interpolation=cv2.INTER_AREA)
                lm = detector.detect(proc)
                proc = apply_pipeline(proc, pipeline, lm)

                if (proc.shape[1], proc.shape[0]) != out_size:
                    proc = cv2.resize(proc, out_size,
                                      interpolation=cv2.INTER_CUBIC)
                writer.write(np.clip(proc, 0, 255).astype(np.uint8))
                if bar is not None:
                    bar.update(1)
    finally:
        if bar is not None:
            bar.close()
        writer.release()
        cap.release()

    return out_path


if __name__ == "__main__":
    import argparse, json

    ap = argparse.ArgumentParser(description="Photix offline video pipeline")
    ap.add_argument("input", help="Path to source video")
    ap.add_argument("--config", default=None,
                    help="Path to a JSON LivePipeline (overrides flags)")
    ap.add_argument("--output", default=None,
                    help="Output path (defaults to results/<auto>.mp4)")

    ap.add_argument("--smile",      type=float, default=0.0)
    ap.add_argument("--eyebrow",    type=float, default=0.0)
    ap.add_argument("--lip-wide",   dest="lip_wide",  type=float, default=0.0)
    ap.add_argument("--face-slim",  dest="face_slim", type=float, default=0.0)
    ap.add_argument("--aging", choices=["none", "age", "deage"], default="none")
    ap.add_argument("--aging-sigma", dest="aging_sigma", type=float, default=50.0)
    ap.add_argument("--lowpass",       action="store_true")
    ap.add_argument("--lowpass-sigma", dest="lowpass_sigma",
                    type=float, default=0.0)
    ap.add_argument("--highpass",       action="store_true")
    ap.add_argument("--highpass-sigma", dest="highpass_sigma",
                    type=float, default=0.0)
    ap.add_argument("--beard",      choices=["none", "black", "brown", "blonde"],
                    default="none")
    ap.add_argument("--sunglasses",
                    choices=["none", "classic", "aviator", "round", "glasses"],
                    default="none")
    ap.add_argument("--hair-style", dest="hair_style",
                    choices=["none", "long"], default="none")
    ap.add_argument("--hair-color", dest="hair_color",
                    choices=["original", "black", "brown", "blonde",
                             "red", "blue", "gray"],
                    default="original")
    args = ap.parse_args()

    if args.config:
        pipeline = LivePipeline.from_json(
            Path(args.config).read_text(encoding="utf-8")
        )
    else:
        pipeline = LivePipeline(
            smile=args.smile, eyebrow=args.eyebrow,
            lip_wide=args.lip_wide, face_slim=args.face_slim,
            aging=args.aging, aging_sigma=args.aging_sigma,
            lowpass_enabled=args.lowpass, lowpass_sigma=args.lowpass_sigma,
            highpass_enabled=args.highpass, highpass_sigma=args.highpass_sigma,
            beard=args.beard, sunglasses=args.sunglasses,
            hair_style=args.hair_style, hair_color=args.hair_color,
        )

    out = process_video(args.input, pipeline=pipeline,
                        output_path=args.output)
    print(f"Wrote: {out}")
