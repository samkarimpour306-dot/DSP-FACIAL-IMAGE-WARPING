"""
photix.live — real-time facial warping pipeline.

Public surface:
    run_webcam(...)        live webcam loop with side-by-side preview
    process_video(...)     batch-process a video file with progress bar
    LiveState              effect / intensity state used by both modes
    apply_effect(...)      shared frame-level effect dispatch
"""
"""Public surface of the live/ package.

We intentionally do NOT eagerly import ``live.webcam`` or ``live.video`` here.
Both modules are entry points runnable via ``python -m live.webcam`` /
``python -m live.video``; importing them from ``__init__`` triggers a
``RuntimeWarning`` from runpy when those subprocess commands are later used.
Callers that need them just do ``from live.webcam import run_webcam``.
"""
from .controls import LiveState, EFFECTS, handle_key
from .effects  import apply_effect
from .pipeline import LivePipeline, apply_pipeline

__all__ = [
    "LiveState",
    "LivePipeline",
    "EFFECTS",
    "handle_key",
    "apply_effect",
    "apply_pipeline",
]
