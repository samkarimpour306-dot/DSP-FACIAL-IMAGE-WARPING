"""
webcam.py — live webcam warping loop with side-by-side preview.

Design:
  * One background thread runs cv2.VideoCapture.read() in a tight loop and
    publishes the latest frame via a single-slot, lock-protected slot. The
    GIL releases inside cv2.VideoCapture.read(), so this materially reduces
    UI lag when landmark detection is the bottleneck.
  * The main thread snapshots the latest frame, runs FaceMesh + effect, then
    draws a labelled side-by-side panel via cv2.imshow.
  * The cv2 window is rendered on EVERY tick — even before the first frame
    arrives or while the camera is stalled — so the user always sees status.

Performance budget:
  * Processing frames are downscaled to 640x480 before any work.
  * Landmark detection is skipped (overlay shown) if no face is present.

The loop is fully driven by keyboard shortcuts wired through
``controls.handle_key`` — see that module for the full key map.
"""
from __future__ import annotations

import sys
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

_HERE = Path(__file__).resolve().parent
_PHOTIX_ROOT = _HERE.parent
if str(_PHOTIX_ROOT) not in sys.path:
    sys.path.insert(0, str(_PHOTIX_ROOT))

from core.landmark_detector import draw_landmarks

# Absolute imports so this file works both as `python -m live.webcam`
# (package context) and `python live/webcam.py` (script context).
from live.controls         import LiveState, handle_key
from live.pipeline         import LivePipeline, apply_pipeline
from live.landmark_stream  import StreamingLandmarker


PROC_W, PROC_H = 640, 480
RESULTS_DIR = _PHOTIX_ROOT / "results"
WINDOW_NAME = "Photix · Live"

_FONT = cv2.FONT_HERSHEY_SIMPLEX


def _log(msg: str) -> None:
    """Diagnostic print to stderr — visible in the parent terminal.

    Falls back to ASCII if the host stderr encoding can't represent a char
    (cp1252 on default Windows terminals trips on Greek/box-drawing glyphs).
    """
    line = f"[photix.live] {msg}"
    try:
        print(line, file=sys.stderr, flush=True)
    except UnicodeEncodeError:
        enc = getattr(sys.stderr, "encoding", "ascii") or "ascii"
        print(line.encode(enc, errors="replace").decode(enc, errors="replace"),
              file=sys.stderr, flush=True)


# ── Capture thread ───────────────────────────────────────────────────────────

class _CameraThread(threading.Thread):
    """Background camera reader. Holds only the most recent frame."""

    def __init__(self, cap: cv2.VideoCapture, label: str):
        super().__init__(daemon=True, name=f"PhotixCamReader[{label}]")
        self._cap = cap
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._stop = threading.Event()
        self._reads = 0
        self._bad_reads = 0

    def run(self) -> None:
        while not self._stop.is_set():
            try:
                ok, frame = self._cap.read()
            except cv2.error as exc:
                _log(f"VideoCapture.read raised: {exc}")
                time.sleep(0.05)
                continue
            self._reads += 1
            if not ok or frame is None or not frame.size:
                self._bad_reads += 1
                time.sleep(0.01)
                continue
            with self._lock:
                self._frame = frame

    def latest(self) -> np.ndarray | None:
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def stats(self) -> tuple[int, int]:
        return self._reads, self._bad_reads

    def stop(self) -> None:
        self._stop.set()


# ── Camera open with platform-aware backend + index fallback ─────────────────

def _platform_backends() -> tuple[tuple[int, str], ...]:
    """Backend ids paired with a human label, in trial order."""
    if sys.platform.startswith("win"):
        return (
            (cv2.CAP_DSHOW, "DirectShow"),
            (cv2.CAP_MSMF,  "Media Foundation"),
            (cv2.CAP_ANY,   "default"),
        )
    if sys.platform == "darwin":
        return (
            (getattr(cv2, "CAP_AVFOUNDATION", cv2.CAP_ANY), "AVFoundation"),
            (cv2.CAP_ANY,   "default"),
        )
    return (
        (getattr(cv2, "CAP_V4L2", cv2.CAP_ANY), "V4L2"),
        (cv2.CAP_ANY,   "default"),
    )


def _probe_capture(index: int, backend: int) -> cv2.VideoCapture | None:
    """Open one (index, backend) combo and return it only if it produces frames."""
    cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        cap.release()
        return None

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  PROC_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, PROC_H)
    cap.set(cv2.CAP_PROP_FPS, 30)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass

    # Drivers often return empty reads while warming up.
    for _ in range(40):
        ok, frame = cap.read()
        if ok and frame is not None and frame.size:
            return cap
        time.sleep(0.02)
    cap.release()
    return None


def _open_capture(source: int | str) -> tuple[cv2.VideoCapture, str]:
    """Open the requested source, returning (cap, human-label).

    For int sources, tries the requested index first across all platform
    backends, then iterates through indices 0..4 to find a working camera.
    """
    if isinstance(source, str):
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video source: {source!r}")
        return cap, f"file:{source}"

    requested = int(source)
    indices: list[int] = [requested] + [i for i in range(5) if i != requested]

    tried: list[str] = []
    for idx in indices:
        for backend, name in _platform_backends():
            cap = _probe_capture(idx, backend)
            tag = f"index={idx} via {name}"
            tried.append(tag)
            if cap is not None:
                _log(f"opened camera: {tag}")
                return cap, tag

    raise RuntimeError(
        "No working webcam found.\n"
        "Tried:\n  " + "\n  ".join(tried) +
        "\n\nThings to check:\n"
        "  • Close other apps using the camera (Teams, Zoom, Camera, OBS).\n"
        "  • Windows Settings → Privacy → Camera → allow desktop apps.\n"
        "  • If using a USB webcam, replug it.\n"
    )


# ── Overlay rendering ────────────────────────────────────────────────────────

def _put_label(img: np.ndarray, text: str, org=(12, 28),
               scale: float = 0.7, color=(255, 255, 255), thick: int = 2) -> None:
    """Draw text with a dark outline for readability over any background."""
    cv2.putText(img, text, org, _FONT, scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
    cv2.putText(img, text, org, _FONT, scale, color,       thick,     cv2.LINE_AA)


def _status_panel(message: str, sub: str = "") -> np.ndarray:
    """Build a placeholder panel shown while waiting for camera frames."""
    panel = np.full((PROC_H, PROC_W * 2 + 4, 3), 12, dtype=np.uint8)
    cv2.rectangle(panel, (0, 0), (panel.shape[1] - 1, panel.shape[0] - 1),
                  (40, 60, 90), 1)

    (tw, th), _ = cv2.getTextSize(message, _FONT, 0.9, 2)
    x = (panel.shape[1] - tw) // 2
    y = panel.shape[0] // 2
    _put_label(panel, message, org=(x, y), scale=0.9,
               color=(120, 220, 255), thick=2)

    if sub:
        for i, line in enumerate(sub.split("\n")):
            (tw2, _), _ = cv2.getTextSize(line, _FONT, 0.5, 1)
            xs = (panel.shape[1] - tw2) // 2
            ys = y + 30 + i * 22
            _put_label(panel, line, org=(xs, ys), scale=0.5,
                       color=(170, 170, 180), thick=1)
    return panel


def _compose_panel(
    left: np.ndarray,
    right: np.ndarray,
    state: LiveState,
    fps: float,
    face_found: bool,
) -> np.ndarray:
    """Build the side-by-side preview frame with overlays."""
    h, w = left.shape[:2]
    panel = np.zeros((h, w * 2 + 4, 3), dtype=np.uint8)
    panel[:, :w]       = left
    panel[:, w + 4:]   = right
    cv2.line(panel, (w + 2, 0), (w + 2, h), (60, 60, 70), 1)

    _put_label(panel, "ORIGINAL", org=(12, 28), color=(200, 200, 220))
    summary = state.label()
    # Wrap the active-effect summary onto the processed panel.
    _put_label(panel, "PROCESSED", org=(w + 16, 28), color=(140, 220, 255))
    _put_label(panel, summary[:64], org=(w + 16, 52),
               scale=0.45, color=(180, 220, 255), thick=1)

    _put_label(panel, f"FPS {fps:5.1f}",
               org=(12, h - 18), scale=0.6, color=(120, 255, 180), thick=2)
    p = state.pipeline
    bumper = (f"σ {int(p.aging_sigma)}" if p.aging != "none"
              else f"{state._last_warp[:4]} {getattr(p, state._last_warp, 0.0):.2f}")
    _put_label(panel, bumper,
               org=(w + 16, h - 18), scale=0.6,
               color=(255, 200, 120), thick=2)

    if not face_found:
        msg = "No face detected"
        (tw, th), _ = cv2.getTextSize(msg, _FONT, 0.8, 2)
        x = w + 4 + (w - tw) // 2
        y = h // 2
        cv2.rectangle(panel, (x - 10, y - th - 8), (x + tw + 10, y + 10),
                      (0, 0, 0), -1)
        _put_label(panel, msg, org=(x, y), scale=0.8,
                   color=(80, 160, 255), thick=2)

    help_text = ("1·Aging  2·Smile  3·Brow  4·Slim  5·Beard  6·Glasses  "
                 "7·Hair  8·LP  9·HP  0·Reset   +/- adjust   S·snap   Q·quit")
    _put_label(panel, help_text, org=(12, h - 44),
               scale=0.38, color=(160, 160, 170), thick=1)
    return panel


# ── Public entry point ───────────────────────────────────────────────────────

def run_webcam(
    source: int = 0,
    pipeline: LivePipeline | None = None,
    results_dir: Path | None = None,
) -> None:
    """Open the webcam and run the live warping loop until the user quits.

    ``pipeline`` seeds every effect/parameter; keyboard shortcuts in the
    window mutate it from there. When ``None`` the loop starts in
    passthrough mode.
    """
    pipeline = pipeline or LivePipeline()

    out_dir = Path(results_dir) if results_dir else RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    _log(f"booting · pipeline={pipeline.label()}")

    # ── Open camera (raises with detailed message on failure) ────────────
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, PROC_W * 2 + 4, PROC_H)
    # Force the window above other windows on Windows so users see it.
    try:
        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_TOPMOST, 1)
    except cv2.error:
        pass
    cv2.imshow(WINDOW_NAME, _status_panel("Opening camera…",
                                          "Probing backends and indices."))
    cv2.waitKey(1)

    try:
        cap, cam_label = _open_capture(source)
    except RuntimeError as exc:
        cv2.imshow(WINDOW_NAME, _status_panel(
            "Camera not available",
            "See the terminal for details. Press any key to close."))
        cv2.waitKey(5000)
        cv2.destroyAllWindows()
        _log(str(exc))
        return

    reader = _CameraThread(cap, label=cam_label)
    reader.start()

    state = LiveState(pipeline=pipeline)

    fps_window: deque[float] = deque(maxlen=30)
    last_tick = time.monotonic()
    wait_start = time.monotonic()
    seen_first_frame = False

    try:
        with StreamingLandmarker() as detector:
            while not state.quit:
                frame = reader.latest()

                # ── No frame yet → show a "waiting" panel and keep pumping. ──
                if frame is None:
                    elapsed = time.monotonic() - wait_start
                    reads, bad = reader.stats()
                    sub = f"reads={reads}  empty={bad}  elapsed={elapsed:.1f}s"
                    msg = ("Waiting for first frame…" if elapsed < 5.0
                           else "Camera opened but no frames yet")
                    cv2.imshow(WINDOW_NAME, _status_panel(msg, sub))
                    handle_key(state, cv2.waitKey(40))
                    try:
                        if cv2.getWindowProperty(WINDOW_NAME,
                                                 cv2.WND_PROP_VISIBLE) < 1:
                            state.quit = True
                    except cv2.error:
                        state.quit = True
                    continue

                if not seen_first_frame:
                    seen_first_frame = True
                    _log(f"first frame received from {cam_label}")

                # Mirror for a natural "selfie" view.
                frame = cv2.flip(frame, 1)
                if frame.shape[1] != PROC_W or frame.shape[0] != PROC_H:
                    frame = cv2.resize(frame, (PROC_W, PROC_H),
                                       interpolation=cv2.INTER_AREA)

                landmarks = detector.detect(frame)

                if state.pipeline.is_empty() or landmarks is None:
                    processed = frame.copy()
                else:
                    processed = apply_pipeline(frame, state.pipeline, landmarks)

                left_view = (draw_landmarks(frame, landmarks, True)
                             if state.show_landmarks else frame)

                now = time.monotonic()
                dt = now - last_tick
                last_tick = now
                if dt > 0:
                    fps_window.append(1.0 / dt)
                fps = sum(fps_window) / len(fps_window) if fps_window else 0.0

                panel = _compose_panel(
                    left_view, processed, state, fps,
                    face_found=landmarks is not None,
                )
                cv2.imshow(WINDOW_NAME, panel)

                if state.request_screenshot:
                    state.request_screenshot = False
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    path = out_dir / f"photix_live_{ts}.png"
                    cv2.imwrite(str(path), panel)
                    _log(f"saved screenshot → {path}")

                handle_key(state, cv2.waitKey(1))

                try:
                    if cv2.getWindowProperty(WINDOW_NAME,
                                             cv2.WND_PROP_VISIBLE) < 1:
                        state.quit = True
                except cv2.error:
                    state.quit = True
    finally:
        reader.stop()
        reader.join(timeout=1.0)
        cap.release()
        cv2.destroyAllWindows()
        _log("shutdown complete")


def _pipeline_from_args(ns) -> LivePipeline:
    """Build a LivePipeline from argparse Namespace."""
    if ns.config:
        return LivePipeline.from_json(Path(ns.config).read_text(encoding="utf-8"))

    return LivePipeline(
        smile=ns.smile,
        eyebrow=ns.eyebrow,
        lip_wide=ns.lip_wide,
        face_slim=ns.face_slim,
        aging=ns.aging,
        aging_sigma=ns.aging_sigma,
        lowpass_enabled=ns.lowpass,
        lowpass_sigma=ns.lowpass_sigma,
        highpass_enabled=ns.highpass,
        highpass_sigma=ns.highpass_sigma,
        beard=ns.beard,
        sunglasses=ns.sunglasses,
        hair_style=ns.hair_style,
        hair_color=ns.hair_color,
    )


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Photix live webcam warping")
    ap.add_argument("--source", default=0, type=int,
                    help="OpenCV camera index (default: 0)")
    ap.add_argument("--config", default=None,
                    help="Path to a JSON file containing a LivePipeline. "
                         "Overrides every other flag when present.")

    ap.add_argument("--smile",      type=float, default=0.0)
    ap.add_argument("--eyebrow",    type=float, default=0.0)
    ap.add_argument("--lip-wide",   dest="lip_wide",  type=float, default=0.0)
    ap.add_argument("--face-slim",  dest="face_slim", type=float, default=0.0)

    ap.add_argument("--aging",      choices=["none", "age", "deage"],
                    default="none")
    ap.add_argument("--aging-sigma", dest="aging_sigma",
                    type=float, default=50.0)

    ap.add_argument("--lowpass",       action="store_true")
    ap.add_argument("--lowpass-sigma", dest="lowpass_sigma",
                    type=float, default=0.0)
    ap.add_argument("--highpass",       action="store_true")
    ap.add_argument("--highpass-sigma", dest="highpass_sigma",
                    type=float, default=0.0)

    ap.add_argument("--beard",
                    choices=["none", "black", "brown", "blonde"],
                    default="none")
    ap.add_argument("--sunglasses",
                    choices=["none", "classic", "aviator", "round", "glasses"],
                    default="none")
    ap.add_argument("--hair-style",  dest="hair_style",
                    choices=["none", "long"], default="none")
    ap.add_argument("--hair-color",  dest="hair_color",
                    choices=["original", "black", "brown", "blonde",
                             "red", "blue", "gray"],
                    default="original")

    args = ap.parse_args()
    run_webcam(source=args.source, pipeline=_pipeline_from_args(args))
