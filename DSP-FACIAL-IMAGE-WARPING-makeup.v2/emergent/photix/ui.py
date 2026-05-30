"""
ui.py - Gradio front-end for Photix live/video features.

Run:
    python ui.py

Then open http://127.0.0.1:7860 in a browser.
"""
from __future__ import annotations

import sys
import threading
import time
import socket
import os
from pathlib import Path

import cv2
import gradio as gr
import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from live.controls import EFFECTS
from live.pipeline import LivePipeline
from live.video import process_video
from live.webcam import run_webcam
from core.landmark_detector import detect_landmarks
from effects.overlays import FaceOverlayEngine


_WARP_EFFECT_KEYS = [
    "none",
    "age",
    "deage",
    "smile",
    "eyebrow",
    "lip_wide",
    "face_slim",
    "surprise",
    "beard",
    "sunglasses",
    "hair",
]
_EFFECT_CHOICES: list[tuple[str, str]] = [
    (EFFECTS[key], key) for key in _WARP_EFFECT_KEYS
]


def _build_pipeline(
    effect_key: str,
    intensity: float,
    beard: str,
    sunglasses: str,
    hair_style: str,
    hair_color: str,
    mk_eye_on: bool = False,   mk_eye_color: str = "brown",   mk_eye_int: float = 0.55,
    mk_blush_on: bool = False, mk_blush_color: str = "pink",  mk_blush_int: float = 0.30,
    mk_lip_on: bool = False,   mk_lip_color: str = "red",     mk_lip_int: float = 0.62,
    mk_liner_on: bool = False, mk_liner_color: str = "black", mk_liner_int: float = 0.85,
    mk_masc_on: bool = False,  mk_masc_color: str = "black",  mk_masc_int: float = 0.75,
) -> LivePipeline:
    """Translate the selected live/video feature into a one-effect pipeline."""
    p = LivePipeline()
    intensity = max(0.0, min(1.0, float(intensity)))

    if effect_key == "age":
        p.aging = "age"
        p.aging_sigma = max(1.0, intensity * 100.0)
    elif effect_key == "deage":
        p.aging = "deage"
        p.aging_sigma = max(1.0, intensity * 100.0)
    elif effect_key == "smile":
        p.smile = intensity
    elif effect_key == "eyebrow":
        p.eyebrow = intensity
    elif effect_key == "lip_wide":
        p.lip_wide = intensity
    elif effect_key == "face_slim":
        p.face_slim = intensity
    elif effect_key == "surprise":
        p.eyebrow = intensity
        p.lip_wide = intensity * 0.6
    elif effect_key == "beard":
        p.beard = beard if beard != "none" else "black"
    elif effect_key == "sunglasses":
        p.sunglasses = sunglasses if sunglasses != "none" else "classic"
    elif effect_key == "hair":
        p.hair_style = hair_style if hair_style != "none" else "long"
        p.hair_color = hair_color or "original"

    # Makeup layers stack on top of whichever warp/overlay is selected.
    p.makeup_eyeshadow_enabled  = bool(mk_eye_on)
    p.makeup_eyeshadow_color    = str(mk_eye_color or "brown")
    p.makeup_eyeshadow_intensity = float(mk_eye_int)
    p.makeup_blush_enabled      = bool(mk_blush_on)
    p.makeup_blush_color        = str(mk_blush_color or "pink")
    p.makeup_blush_intensity    = float(mk_blush_int)
    p.makeup_lipstick_enabled   = bool(mk_lip_on)
    p.makeup_lipstick_color     = str(mk_lip_color or "red")
    p.makeup_lipstick_intensity = float(mk_lip_int)
    p.makeup_eyeliner_enabled   = bool(mk_liner_on)
    p.makeup_eyeliner_color     = str(mk_liner_color or "black")
    p.makeup_eyeliner_intensity = float(mk_liner_int)
    p.makeup_mascara_enabled    = bool(mk_masc_on)
    p.makeup_mascara_color      = str(mk_masc_color or "black")
    p.makeup_mascara_intensity  = float(mk_masc_int)

    return p


def _process_video_ui(
    video_path: str | None,
    effect_key: str,
    intensity: float,
    beard: str,
    sunglasses: str,
    hair_style: str,
    hair_color: str,
    mk_eye_on: bool, mk_eye_color: str, mk_eye_int: float,
    mk_blush_on: bool, mk_blush_color: str, mk_blush_int: float,
    mk_lip_on: bool, mk_lip_color: str, mk_lip_int: float,
    mk_liner_on: bool, mk_liner_color: str, mk_liner_int: float,
    mk_masc_on: bool, mk_masc_color: str, mk_masc_int: float,
    progress: gr.Progress = gr.Progress(track_tqdm=True),
) -> tuple[str | None, str]:
    """Gradio handler: run process_video and return (output_path, log)."""
    if not video_path:
        return None, "No video uploaded. Pick a file first."

    in_path = Path(video_path)
    if not in_path.exists():
        return None, f"File not found: {in_path}"

    try:
        pipeline = _build_pipeline(
            effect_key, intensity, beard, sunglasses, hair_style, hair_color,
            mk_eye_on, mk_eye_color, mk_eye_int,
            mk_blush_on, mk_blush_color, mk_blush_int,
            mk_lip_on, mk_lip_color, mk_lip_int,
            mk_liner_on, mk_liner_color, mk_liner_int,
            mk_masc_on, mk_masc_color, mk_masc_int,
        )
        progress(0.0, desc=f"Processing {pipeline.label()}...")
        out_path = process_video(
            in_path,
            pipeline=pipeline,
            show_progress=True,
        )
        return str(out_path), (
            f"Done.\n"
            f"Pipeline: {pipeline.label()}\n"
            f"Output: {out_path}"
        )
    except Exception as exc:  # noqa: BLE001
        return None, f"Processing failed: {exc}"


def _process_image_ui(
    image_rgb: np.ndarray | None,
    beard: str,
    sunglasses: str,
    hair_style: str,
    hair_color: str,
    mk_eye_on: bool, mk_eye_color: str, mk_eye_int: float,
    mk_blush_on: bool, mk_blush_color: str, mk_blush_int: float,
    mk_lip_on: bool, mk_lip_color: str, mk_lip_int: float,
    mk_liner_on: bool, mk_liner_color: str, mk_liner_int: float,
    mk_masc_on: bool, mk_masc_color: str, mk_masc_int: float,
) -> tuple[np.ndarray | None, str]:
    """Apply still-image overlays + makeup."""
    if image_rgb is None:
        return None, "No image uploaded. Pick a face image first."

    try:
        img_rgb = np.asarray(image_rgb, dtype=np.uint8)
        if img_rgb.ndim == 2:
            img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_GRAY2RGB)
        if img_rgb.shape[2] == 4:
            img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_RGBA2RGB)

        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        landmarks = detect_landmarks(img_bgr)
        if landmarks is None:
            return image_rgb, "No face detected, so overlays/makeup could not be placed."

        applied: list[str] = []

        # 1) Hair / sunglasses / beard overlays
        effects: list[str] = []
        params: dict[str, dict[str, str]] = {}
        if hair_style != "none":
            effects.append("hair")
            params["hair"] = {"style": hair_style, "color": hair_color}
        if sunglasses != "none":
            effects.append("sunglasses")
            params["sunglasses"] = {"style": sunglasses}
        if beard != "none":
            effects.append("beard")
            params["beard"] = {"color": beard}
        if effects:
            img_bgr = FaceOverlayEngine(landmarks, img_bgr).apply(effects, params).final_image
            applied.extend(effects)

        # 2) Makeup layers
        from effects.makeup import apply_makeup
        mk_cfg = {
            "eyeshadow": {"enabled": bool(mk_eye_on),   "color": mk_eye_color,   "intensity": float(mk_eye_int)},
            "blush":     {"enabled": bool(mk_blush_on), "color": mk_blush_color, "intensity": float(mk_blush_int)},
            "lipstick":  {"enabled": bool(mk_lip_on),   "color": mk_lip_color,   "intensity": float(mk_lip_int)},
            "eyeliner":  {"enabled": bool(mk_liner_on), "color": mk_liner_color, "intensity": float(mk_liner_int)},
            "mascara":   {"enabled": bool(mk_masc_on),  "color": mk_masc_color,  "intensity": float(mk_masc_int)},
        }
        active = [name for name, v in mk_cfg.items() if v["enabled"]]
        if active:
            img_bgr = apply_makeup(img_bgr, landmarks, mk_cfg).final_image
            applied.extend(f"makeup:{n}" for n in active)

        if not applied:
            return image_rgb, "No overlay or makeup selected."

        out_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return out_rgb, "Applied: " + ", ".join(applied)
    except Exception as exc:  # noqa: BLE001
        return image_rgb, f"Image processing failed: {exc}"


def _launch_webcam(
    effect_key: str,
    intensity: float,
    beard: str,
    sunglasses: str,
    hair_style: str,
    hair_color: str,
    mk_eye_on: bool, mk_eye_color: str, mk_eye_int: float,
    mk_blush_on: bool, mk_blush_color: str, mk_blush_int: float,
    mk_lip_on: bool, mk_lip_color: str, mk_lip_int: float,
    mk_liner_on: bool, mk_liner_color: str, mk_liner_int: float,
    mk_masc_on: bool, mk_masc_color: str, mk_masc_int: float,
) -> str:
    """Open the OpenCV live window from a background thread."""

    def _runner() -> None:
        try:
            pipeline = _build_pipeline(
                effect_key, intensity, beard, sunglasses, hair_style, hair_color,
                mk_eye_on, mk_eye_color, mk_eye_int,
                mk_blush_on, mk_blush_color, mk_blush_int,
                mk_lip_on, mk_lip_color, mk_lip_int,
                mk_liner_on, mk_liner_color, mk_liner_int,
                mk_masc_on, mk_masc_color, mk_masc_int,
            )
            run_webcam(source=0, pipeline=pipeline)
        except Exception as exc:  # noqa: BLE001
            print(f"[photix.ui] webcam loop ended: {exc}")

    threading.Thread(target=_runner, daemon=True).start()
    return (
        "Live webcam window opened in a separate OpenCV window.\n"
        "Effect keys: 1-7 cycles  ·  +/- intensity  ·  S snapshot  ·  Q quit\n"
        "Makeup keys: e/E eyeshadow  b/B blush  r/R lipstick  n/N eyeliner  k/K mascara\n"
        "(lowercase = on/off, UPPERCASE = cycle colour)"
    )


def _build_makeup_widgets(prefix: str) -> dict:
    """Return a dict of Gradio components for a five-layer makeup panel.

    The same widget set is used in the Image tab and the Live/Video tab —
    keep the IDs separate so Gradio doesn't dedupe them.
    """
    w: dict = {}
    with gr.Accordion("Makyaj / Makeup", open=False):
        gr.Markdown("Her katmanı **ayrı** aç/kapa, rengini ve yoğunluğunu seç.")
        with gr.Group():
            w["eye_on"]    = gr.Checkbox(value=False, label="Göz Farı (eyeshadow)")
            w["eye_color"] = gr.Dropdown(
                choices=["blue", "black", "red", "green", "orange", "brown", "glitter"],
                value="brown", label="Eyeshadow colour")
            w["eye_int"]   = gr.Slider(0.0, 1.0, value=0.55, step=0.05, label="Eyeshadow intensity")
        with gr.Group():
            w["blush_on"]    = gr.Checkbox(value=False, label="Allık (blush)")
            w["blush_color"] = gr.Dropdown(choices=["pink", "brown"], value="pink", label="Blush colour")
            w["blush_int"]   = gr.Slider(0.0, 1.0, value=0.30, step=0.05, label="Blush intensity")
        with gr.Group():
            w["lip_on"]    = gr.Checkbox(value=False, label="Ruj (lipstick)")
            w["lip_color"] = gr.Dropdown(
                choices=["red", "pink", "brown", "gloss"], value="red", label="Lipstick colour")
            w["lip_int"]   = gr.Slider(0.0, 1.0, value=0.62, step=0.05, label="Lipstick intensity")
        with gr.Group():
            w["liner_on"]    = gr.Checkbox(value=False, label="Eyeliner")
            w["liner_color"] = gr.Dropdown(
                choices=["black", "brown", "blue", "green"], value="black", label="Eyeliner colour")
            w["liner_int"]   = gr.Slider(0.0, 1.0, value=0.85, step=0.05, label="Eyeliner intensity")
        with gr.Group():
            w["masc_on"]    = gr.Checkbox(value=False, label="Kirpik (mascara)")
            w["masc_color"] = gr.Dropdown(choices=["black", "brown"], value="black", label="Mascara colour")
            w["masc_int"]   = gr.Slider(0.0, 1.0, value=0.75, step=0.05, label="Mascara intensity")
    return w


def _makeup_inputs(w: dict) -> list:
    """Flatten the dict above into the positional input order the handlers expect."""
    return [
        w["eye_on"],   w["eye_color"],   w["eye_int"],
        w["blush_on"], w["blush_color"], w["blush_int"],
        w["lip_on"],   w["lip_color"],   w["lip_int"],
        w["liner_on"], w["liner_color"], w["liner_int"],
        w["masc_on"],  w["masc_color"],  w["masc_int"],
    ]


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Photix Live / Video") as demo:
        gr.Markdown(
            "# Photix\n"
            "Apply Photix face overlays to still images, uploaded videos, "
            "or the live webcam loop."
        )

        with gr.Tabs():
            with gr.Tab("Image"):
                with gr.Row():
                    with gr.Column(scale=1):
                        image_in = gr.Image(
                            label="Input face image",
                            type="numpy",
                            image_mode="RGB",
                        )
                        image_beard = gr.Dropdown(
                            choices=["none", "black", "brown", "blonde"],
                            value="none",
                            label="Beard",
                        )
                        image_glasses = gr.Dropdown(
                            choices=["none", "classic", "aviator", "round", "glasses"],
                            value="none",
                            label="Glasses",
                        )
                        image_hair_style = gr.Dropdown(
                            choices=["none", "long"],
                            value="none",
                            label="Hair",
                        )
                        image_hair_color = gr.Dropdown(
                            choices=[
                                "original",
                                "black",
                                "brown",
                                "blonde",
                                "red",
                                "blue",
                                "gray",
                            ],
                            value="original",
                            label="Hair Color",
                        )
                        image_mk = _build_makeup_widgets("img")
                        image_btn = gr.Button("Process Image", variant="primary")

                    with gr.Column(scale=1):
                        image_out = gr.Image(label="Processed image", type="numpy")
                        image_log = gr.Textbox(
                            label="Image status",
                            interactive=False,
                            lines=4,
                        )

                image_btn.click(
                    _process_image_ui,
                    inputs=[
                        image_in,
                        image_beard,
                        image_glasses,
                        image_hair_style,
                        image_hair_color,
                        *_makeup_inputs(image_mk),
                    ],
                    outputs=[image_out, image_log],
                )

            with gr.Tab("Live / Video"):
                with gr.Row():
                    with gr.Column(scale=1):
                        video_in = gr.Video(
                            label="Input video (.mp4, .avi, .mov)",
                            sources=["upload"],
                            format=None,
                        )
                        effect_dd = gr.Dropdown(
                            choices=_EFFECT_CHOICES,
                            value="none",
                            label="Warp / Aging",
                        )
                        intensity = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=0.6,
                            step=0.05,
                            label="Intensity",
                        )
                        beard_dd = gr.Dropdown(
                            choices=["none", "black", "brown", "blonde"],
                            value="none",
                            label="Beard",
                        )
                        glasses_dd = gr.Dropdown(
                            choices=["none", "classic", "aviator", "round", "glasses"],
                            value="none",
                            label="Glasses",
                        )
                        hair_style_dd = gr.Dropdown(
                            choices=["none", "long"],
                            value="none",
                            label="Hair",
                        )
                        hair_color_dd = gr.Dropdown(
                            choices=[
                                "original",
                                "black",
                                "brown",
                                "blonde",
                                "red",
                                "blue",
                                "gray",
                            ],
                            value="original",
                            label="Hair Color",
                        )
                        live_mk = _build_makeup_widgets("live")
                        process_btn = gr.Button(
                            "Process Video",
                            variant="primary",
                        )
                        gr.Markdown("---")
                        webcam_btn = gr.Button(
                            "Open Live Webcam Window",
                            variant="secondary",
                        )
                        webcam_log = gr.Textbox(
                            label="Webcam status",
                            interactive=False,
                            lines=3,
                        )

                    with gr.Column(scale=1):
                        video_out = gr.Video(
                            label="Processed video (saved under /results)",
                        )
                        log = gr.Textbox(
                            label="Run log",
                            interactive=False,
                            lines=6,
                        )

                pipeline_inputs = [
                    effect_dd,
                    intensity,
                    beard_dd,
                    glasses_dd,
                    hair_style_dd,
                    hair_color_dd,
                    *_makeup_inputs(live_mk),
                ]
                process_btn.click(
                    _process_video_ui,
                    inputs=[video_in, *pipeline_inputs],
                    outputs=[video_out, log],
                )
                webcam_btn.click(
                    _launch_webcam,
                    inputs=pipeline_inputs,
                    outputs=[webcam_log],
                )

                gr.Markdown(
                    "**Keyboard shortcuts in the webcam window:** "
                    "`1` Aging, `2` Smile, `3` Brow, `4` Slim, "
                    "`5` Beard, `6` Glasses, `7` Hair, `0` Off, "
                    "`+/-` Intensity, `S` Snapshot, `L` Landmarks, `Q` Quit  ·  "
                    "**Makeup:** `e/E` eyeshadow, `b/B` blush, `r/R` lipstick, "
                    "`n/N` eyeliner, `k/K` mascara (lowercase = on/off, UPPERCASE = next colour)"
                )

    return demo


def _resolve_port(preferred: int = 7860) -> int:
    """Return GRADIO_SERVER_PORT or the first free localhost port."""
    raw_port = os.environ.get("GRADIO_SERVER_PORT")
    if raw_port:
        return int(raw_port)

    for port in range(preferred, preferred + 50):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.2)
            if sock.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise RuntimeError("No free localhost port found for the Photix web UI.")


def main() -> None:
    demo = build_ui()
    port = _resolve_port()
    _app, local_url, _share_url = demo.queue().launch(
        server_name="127.0.0.1",
        server_port=port,
        prevent_thread_lock=True,
        show_error=True,
        ssr_mode=False,
        theme=gr.themes.Soft(primary_hue="violet"),
    )
    print(f"Photix web UI running at {local_url}", flush=True)
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
