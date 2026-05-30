# DSP Facial Image Warping — Photix Studio

A desktop application for real-time facial image processing and analysis, built with Python, OpenCV, MediaPipe, and tkinter.

## Features

- **Facial landmark detection** — 478-point MediaPipe face mesh with pose estimation (yaw, pitch, roll)
- **Expression warping** — Smile, eyebrow raise, lip widening, and face slimming via three warp methods:
  - Delaunay Affine Triangulation
  - Thin-Plate Spline (TPS)
  - Optical Flow
- **Aging / De-aging filters** — Gaussian-based aging and de-aging effects with adjustable strength
- **Face overlays** — Sunglasses (classic, aviator, round, clear) and beard effects with realistic compositing
- **Virtual makeup** — Five pose-stable layers, recomputed from landmarks every frame so they track the face when the head turns. Works in `main.py` (Tk sidebar), `ui.py` (Gradio web UI), `start_live.py` (live webcam) and on video files via `live/video.py`.
  - Göz farı (eyeshadow): blue, black, red, green, orange, brown, **glitter** (simli)
  - Allık (blush): pink, brown
  - Ruj (lipstick): red, pink, brown, **gloss** (parlak)
  - Eyeliner: black, brown, blue, green
  - Kirpik (mascara): black, brown
- **FFT analysis** — Frequency-domain spectral energy comparison between original and processed images
- **Quality metrics** — MSE, PSNR, and SSIM measurements
- **Export** — CSV reports, PDF summaries, and PNG image saving
- **Live camera** — Real-time camera preview with single-click capture

## Project Structure

```
emergent/photix/
├── main.py                  # Application entry point (tkinter GUI)
├── requirements.txt
├── core/                    # Core DSP processing modules
│   ├── landmark_detector.py # MediaPipe 478-point detector
│   ├── expression_warper.py # Displacement field computation
│   ├── mesh_warper.py       # Delaunay triangulation warp
│   ├── tps_warper.py        # Thin-plate spline warp
│   ├── flow_warper.py       # Optical flow warp
│   ├── aging_filter.py      # Aging / de-aging filters
│   ├── fft_analyzer.py      # FFT spectral analysis
│   ├── metrics.py           # MSE / PSNR / SSIM
│   ├── report_exporter.py   # CSV + PDF export
│   └── replicate_aging.py   # Cloud aging via Replicate SAM (optional)
├── landmarks/               # Modular landmark detector (pluggable backends)
│   └── detector.py
├── effects/                 # Face overlay effects
│   ├── overlays.py          # Sunglasses + beard compositing engine
│   └── assets/beard.png
├── aging/                   # Aging pipeline modules
├── utils/                   # Landmark index constants
├── examples/                # Demo scripts
├── tests/                   # Unit tests
├── save_beard.py            # Helper: copy beard PNG to assets/
└── save_beard_clipboard.py  # Helper: paste beard PNG from clipboard
```

## Setup

```bash
pip install -r emergent/photix/requirements.txt
```

The MediaPipe face landmarker model (~29 MB) downloads automatically on first run.

## Run

```bash
cd emergent/photix
python main.py
```

## Cloud Aging (optional)

For Replicate SAM-based age transformation, add your API token to `emergent/photix/.env`:

```
REPLICATE_API_TOKEN=r8_...
```

Then use `examples/replicate_age_transform.py`:

```bash
python examples/replicate_age_transform.py path/to/face.jpg
```
