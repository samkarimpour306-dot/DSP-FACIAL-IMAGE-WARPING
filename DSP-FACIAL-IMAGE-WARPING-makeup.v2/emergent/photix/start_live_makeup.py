"""
start_live_makeup.py — Photix live webcam, started with makeup already on.

Same plumbing as start_live.py but seeds the LivePipeline with a complete
makeup look so you can see the effect immediately. Once the window opens
you can still toggle / recolour every layer with the keyboard:

    e/E   eyeshadow toggle / cycle colour
    b/B   blush     toggle / cycle colour
    r/R   lipstick  toggle / cycle colour
    n/N   eyeliner  toggle / cycle colour
    k/K   mascara   toggle / cycle colour
    0     reset everything
    s     snapshot         q / Esc  quit
"""
from __future__ import annotations

from live.pipeline import LivePipeline
from live.webcam import run_webcam


def main() -> None:
    pipeline = LivePipeline(
        makeup_eyeshadow_enabled=True, makeup_eyeshadow_color="brown",  makeup_eyeshadow_intensity=0.55,
        makeup_blush_enabled=True,    makeup_blush_color="pink",       makeup_blush_intensity=0.30,
        makeup_lipstick_enabled=True, makeup_lipstick_color="red",     makeup_lipstick_intensity=0.62,
        makeup_eyeliner_enabled=True, makeup_eyeliner_color="black",   makeup_eyeliner_intensity=0.85,
        makeup_mascara_enabled=True,  makeup_mascara_color="black",    makeup_mascara_intensity=0.75,
    )
    run_webcam(source=0, pipeline=pipeline)


if __name__ == "__main__":
    main()
