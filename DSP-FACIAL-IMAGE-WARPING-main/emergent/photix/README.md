# Photix — Facial Image Processing Studio

> **Academic DSP project** covering all 29 functional requirements (FR-01 … FR-29)  
> from the Photix SRS/SDD specification.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the application
python main.py
```

---

## Functional Requirements Coverage

| FR | Description | Module |
|----|-------------|--------|
| FR-01 | JPEG/PNG/BMP image loading via file dialog | `core/image_loader.py` |
| FR-02 | Resize + centre-crop to 512×512 | `core/image_loader.py` |
| FR-03 | MediaPipe Face Mesh — 468 landmarks | `core/landmark_detector.py` |
| FR-04 | Toggle landmark visualisation | `core/landmark_detector.py` |
| FR-05 | Smile expression warp | `core/expression_warper.py` |
| FR-06 | Eyebrow raise warp | `core/expression_warper.py` |
| FR-07 | Lip widening warp | `core/expression_warper.py` |
| FR-08 | Face slimming warp | `core/expression_warper.py` |
| FR-09 | Smooth interpolation (INTER_CUBIC) | `core/mesh_warper.py` |
| FR-10 | Delaunay triangulation | `core/mesh_warper.py` |
| FR-11 | Per-triangle affine inverse warp | `core/mesh_warper.py` |
| FR-12 | Sparse optical-flow estimation | `core/flow_warper.py` |
| FR-13 | Dense flow via `scipy.griddata` + remap | `core/flow_warper.py` |
| FR-14 | TPS coefficients via `scipy.linalg.solve` | `core/tps_warper.py` |
| FR-15 | TPS global displacement field + remap | `core/tps_warper.py` |
| FR-16 | Aging — Gaussian High-pass FFT filter | `core/aging_filter.py` |
| FR-17 | De-aging — Gaussian Low-pass FFT filter | `core/aging_filter.py` |
| FR-18 | 2D FFT computation | `core/fft_analyzer.py` |
| FR-19 | Log-scaled magnitude spectrum | `core/fft_analyzer.py` |
| FR-20 | H/L energy ratio | `core/fft_analyzer.py` |
| FR-21 | Side-by-side spectrum comparison | `core/fft_analyzer.py` + `main.py` |
| FR-22 | MSE metric | `core/metrics.py` |
| FR-23 | PSNR metric | `core/metrics.py` |
| FR-24 | SSIM metric (skimage) | `core/metrics.py` |
| FR-25 | CSV + PDF report export | `core/report_exporter.py` |
| FR-26 | Side-by-side visualisation | `main.py` |
| FR-27 | matplotlib.widgets.Slider controls | `main.py` |
| FR-28 | Save PNG output images | `core/report_exporter.py` |
| FR-29 | FFT spectrum for filtered images | `main.py` (FFT view) |

---

## Usage Guide

1. **Load Image** (FR-01) — Click *Browse Image…* and select a JPEG/PNG/BMP file.  
   The image is automatically resized to 512×512 (FR-02) and landmarks detected (FR-03).

2. **Adjust Sliders** (FR-27) — Use the five matplotlib sliders in the *Images* view:  
   `Smile · Eyebrow · Lip Wide · Face Slim · Aging σ`

3. **Select Warping Method** (FR-10..15):  
   - **Delaunay Affine** — triangulation mesh warp (FR-10, 11)  
   - **Thin-Plate Spline** — smooth global warp (FR-14, 15) *(~10 s on CPU)*  
   - **Optical Flow** — dense interpolated flow warp (FR-12, 13)

4. **Select Aging Mode** (FR-16, 17):  
   - **Age (+)** — Gaussian high-pass enhances texture  
   - **De-Age (-)** — Gaussian low-pass smooths skin

5. **Apply Processing** — Click the green *Apply Processing* button.  
   Metrics (FR-22..24) and FFT analysis (FR-18..21) are computed automatically.

6. **Switch Views**:  
   - *Images* — Original vs Processed (FR-26)  
   - *FFT Spectra* — Log-magnitude spectra + energy table (FR-18..21, FR-29)  
   - *Metrics* — MSE / PSNR / SSIM bar charts + summary table (FR-22..24)

7. **Export**:  
   - *Export CSV Report* (FR-25) — metrics + FFT energy  
   - *Export PDF Report* (FR-25) — visual report with images & charts  
   - *Save Images PNG* (FR-28) — saves original + processed as PNG

---

## Project Structure

```
photix/
├── main.py                  # GUI entry point (tkinter + matplotlib)
├── requirements.txt
├── README.md
└── core/
    ├── image_loader.py      # FR-01, FR-02
    ├── landmark_detector.py # FR-03, FR-04
    ├── expression_warper.py # FR-05..08
    ├── mesh_warper.py       # FR-09..11
    ├── flow_warper.py       # FR-12, FR-13
    ├── tps_warper.py        # FR-14, FR-15
    ├── aging_filter.py      # FR-16, FR-17
    ├── fft_analyzer.py      # FR-18..21, FR-29
    ├── metrics.py           # FR-22..24
    └── report_exporter.py   # FR-25, FR-28
```

---

## System Requirements

- Python 3.10+
- Windows 10/11, macOS 12+, or Ubuntu 20.04+
- CPU-only (no GPU required)
- RAM: ≥ 4 GB recommended (TPS warp is memory-intensive)
- No internet connection required

---

## NFR Compliance

| NFR | Status |
|-----|--------|
| NFR-P01: Pipeline < 30 s on i5/Ryzen 5 | ✔ (Delaunay/Flow), ~8-15 s (TPS) |
| NFR-P02: FFT < 2 s | ✔ |
| NFR-R01: Graceful error on no-face | ✔ — warning shown, no crash |
| NFR-R02: `np.clip` boundary safety | ✔ — all output arrays clipped |
| NFR-U01: Single-launch GUI | ✔ — `python main.py` |
 
---

## Replicate SAM Aging / De-aging

This project also includes a cloud-backed SAM age transformer:

- `core/replicate_aging.py` exposes `transform_age_file(...)` and `apply_age_replicate(...)`.
- `examples/replicate_age_transform.py` creates both younger and older outputs for one image.

Set `REPLICATE_API_TOKEN` in `.env`, then run:

```bash
python examples/replicate_age_transform.py path/to/face.jpg
```

By default it saves:

- `outputs/<name>_younger_20.jpg`
- `outputs/<name>_older_70.jpg`

Use `--younger-age` and `--older-age` to choose different target ages.
