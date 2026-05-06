"""
report_exporter.py — FR-25, FR-28
Export metrics to CSV (and optionally PDF) and save processed images as PNG.
"""
import csv
import cv2
import numpy as np
from datetime import datetime
from pathlib import Path


def export_csv(
    metrics: dict,
    fft_data: dict | None,
    params: dict | None,
    filepath: str,
) -> str:
    """FR-25: Export all metrics and parameters to a CSV file."""
    with open(filepath, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["Photix Analysis Report", datetime.now().isoformat()])
        writer.writerow([])

        # Quality metrics
        writer.writerow(["=== Image Quality Metrics ==="])
        writer.writerow(["Metric", "Value"])
        for key, val in metrics.items():
            if val == float("inf"):
                writer.writerow([key, "inf (identical images)"])
            else:
                writer.writerow([key, f"{val:.6f}"])
        writer.writerow([])

        # FFT energy
        if fft_data:
            writer.writerow(["=== Spectral Energy Analysis ==="])
            writer.writerow(["Channel", "Total", "Low", "High", "H/L Ratio"])
            for label, key in [("Original", "orig_energy"), ("Processed", "proc_energy")]:
                e = fft_data.get(key, {})
                writer.writerow([
                    label,
                    f"{e.get('total', 0):.2f}",
                    f"{e.get('low',   0):.2f}",
                    f"{e.get('high',  0):.2f}",
                    f"{e.get('ratio', 0):.4f}",
                ])
            writer.writerow([])

        # Processing parameters
        if params:
            writer.writerow(["=== Processing Parameters ==="])
            writer.writerow(["Parameter", "Value"])
            for k, v in params.items():
                writer.writerow([k, v])

    return filepath


def export_pdf(
    orig_img: np.ndarray,
    proc_img: np.ndarray,
    metrics: dict,
    fft_data: dict | None,
    filepath: str,
) -> str:
    """FR-25 (bonus): Export a summary PDF using matplotlib's PDF backend."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.pyplot as plt

    with PdfPages(filepath) as pdf:
        fig, axes = plt.subplots(2, 3, figsize=(14, 9))
        fig.suptitle("Photix — Analysis Report", fontsize=16, fontweight="bold")

        # Images
        axes[0, 0].imshow(cv2.cvtColor(orig_img,  cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title("Original")
        axes[0, 0].axis("off")

        axes[0, 1].imshow(cv2.cvtColor(proc_img, cv2.COLOR_BGR2RGB))
        axes[0, 1].set_title("Processed")
        axes[0, 1].axis("off")

        # Metrics bar chart
        if metrics:
            keys = [k for k in metrics if metrics[k] != float("inf")]
            vals = [metrics[k] for k in keys]
            axes[0, 2].barh(keys, vals, color=["#1f77b4", "#ff7f0e", "#2ca02c"])
            axes[0, 2].set_title("Quality Metrics")

        # FFT spectra
        if fft_data and "orig_spectrum" in fft_data:
            axes[1, 0].imshow(fft_data["orig_spectrum"], cmap="inferno", origin="lower")
            axes[1, 0].set_title("FFT — Original")
            axes[1, 0].axis("off")

            axes[1, 1].imshow(fft_data["proc_spectrum"], cmap="inferno", origin="lower")
            axes[1, 1].set_title("FFT — Processed")
            axes[1, 1].axis("off")

        # Energy table
        if fft_data:
            axes[1, 2].axis("off")
            oe = fft_data.get("orig_energy", {})
            pe = fft_data.get("proc_energy", {})
            tdata = [
                ["", "Original", "Processed"],
                ["H/L Ratio", f"{oe.get('ratio',0):.4f}", f"{pe.get('ratio',0):.4f}"],
                ["Low",  f"{oe.get('low',0):.1e}", f"{pe.get('low',0):.1e}"],
                ["High", f"{oe.get('high',0):.1e}", f"{pe.get('high',0):.1e}"],
            ]
            tbl = axes[1, 2].table(cellText=tdata, loc="center", cellLoc="center")
            tbl.auto_set_font_size(True)
            axes[1, 2].set_title("Spectral Energy Table")

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    return filepath


def save_image(img: np.ndarray, filepath: str) -> str:
    """FR-28: Save image array as PNG."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(filepath, np.clip(img, 0, 255).astype(np.uint8))
    return filepath
