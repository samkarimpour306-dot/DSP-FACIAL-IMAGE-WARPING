"""
Start Photix live webcam directly, without the Tk desktop shell.

This is useful on Python installs where tkinter/Tcl is broken. The same
LivePipeline is used by the desktop, web UI, video processor, and this
launcher.
"""
from __future__ import annotations

from live.pipeline import LivePipeline
from live.webcam import run_webcam


def main() -> None:
    pipeline = LivePipeline()
    run_webcam(source=0, pipeline=pipeline)


if __name__ == "__main__":
    main()
