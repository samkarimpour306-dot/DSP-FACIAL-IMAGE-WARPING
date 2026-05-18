"""
main.py — Photix v2  ·  Facial Image Processing Studio
Native-tkinter image display + tk.Scale sliders (no matplotlib overhead for images).
"""
import sys
import logging
import threading
from pathlib import Path
from datetime import datetime

import numpy as np
import cv2
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))
_log = logging.getLogger(__name__)

from core.image_loader      import load_image, preprocess
from core.landmark_detector import detect_landmarks, draw_landmarks
from core.expression_warper import compute_combined_displacement
from core.mesh_warper       import apply_delaunay_warp
from core.flow_warper       import apply_optical_flow_warp
from core.tps_warper        import apply_tps_warp
from core.aging_filter      import apply_aging, apply_deaging
from core.filters           import apply_lowpass_filter, apply_highpass_filter, FilterType
from core.fft_analyzer      import compare_spectra, compute_magnitude_spectrum
from core.metrics           import compute_all_metrics
from core.report_exporter   import export_csv, export_pdf, save_image
from effects.overlays       import FaceOverlayEngine

# ── DESIGN TOKENS ─────────────────────────────────────────────────────────────
BG0    = "#06080F"
BG1    = "#0B1018"
BG2    = "#101520"
BG3    = "#16202E"
BORDER = "#1A2A3F"

VIOLET = "#7C5CF6"
CYAN   = "#22D3EE"
GREEN  = "#10B981"
AMBER  = "#F59E0B"
RED    = "#F43F5E"
PINK   = "#EC4899"

T0     = "#F8FAFC"
T1     = "#94A3B8"
T2     = "#3F5068"

FS  = ("Segoe UI", 8)
FN  = ("Segoe UI", 9)
FB  = ("Segoe UI", 9,  "bold")
FL  = ("Segoe UI", 13, "bold")
FM  = ("Consolas", 8)

SPIN = ["⠋","⠙","⠹","⠸","⠼","⠴","⠦","⠧","⠇","⠏"]

MPL_RC = {
    "figure.facecolor": BG2,   "axes.facecolor":   BG0,
    "axes.edgecolor":   BORDER,"axes.labelcolor":  T1,
    "xtick.color":      T2,    "ytick.color":      T2,
    "text.color":       T1,    "grid.color":       BORDER,
    "font.family":      "sans-serif",
    "font.size":        9,     "axes.titlecolor":  T0,
}

# ── MODULE-LEVEL HELPERS ───────────────────────────────────────────────────────

def _lighten(hex_color: str, amount: float) -> str:
    try:
        h = hex_color.lstrip("#")
        r, g, b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
        r = min(255, int(r+(255-r)*amount))
        g = min(255, int(g+(255-g)*amount))
        b = min(255, int(b+(255-b)*amount))
        return f"#{r:02x}{g:02x}{b:02x}"
    except Exception:
        return hex_color

def _sep(parent: tk.Widget, color: str = BORDER) -> None:
    tk.Frame(parent, bg=color, height=1).pack(fill=tk.X, pady=5)

def _sect(parent: tk.Widget, title: str) -> None:
    f = tk.Frame(parent, bg=BG1)
    f.pack(fill=tk.X, padx=10, pady=(7, 1))
    tk.Label(f, text=title, bg=BG1, fg=T2,
             font=("Segoe UI", 7, "bold")).pack(side=tk.LEFT)
    tk.Frame(f, bg=BORDER, height=1).pack(
        side=tk.LEFT, fill=tk.X, expand=True, padx=(6, 0), pady=(3, 0))


# ── CUSTOM WIDGETS ─────────────────────────────────────────────────────────────

class _Btn(tk.Button):
    def __init__(self, parent, text, cmd, accent=BG3, fg=T0, icon="", **kw):
        label = f"{icon}  {text}" if icon else text
        super().__init__(parent, text=label, command=cmd,
                         bg=accent, fg=fg, relief=tk.FLAT,
                         font=FN, padx=10, pady=5,
                         cursor="hand2", bd=0,
                         activebackground=_lighten(accent, 0.18),
                         activeforeground=fg,
                         highlightthickness=0, **kw)
        self._acc = accent
        self.bind("<Enter>", lambda _: self.config(bg=_lighten(accent, 0.18)))
        self.bind("<Leave>", lambda _: self.config(bg=accent))

    def recolor(self, accent: str, fg: str = T0):
        self._acc = accent
        self.config(bg=accent, fg=fg,
                    activebackground=_lighten(accent, 0.18),
                    activeforeground=fg)
        self.bind("<Enter>", lambda _: self.config(bg=_lighten(accent, 0.18)))
        self.bind("<Leave>", lambda _: self.config(bg=accent))


class _ImgPane(tk.Frame):
    """Fast PIL/ImageTk image display — no matplotlib overhead."""

    def __init__(self, parent, title: str, **kw):
        super().__init__(parent, bg=BG2, **kw)
        self._photo   = None
        self._last    = None

        hdr = tk.Frame(self, bg=BG1, height=28)
        hdr.pack(fill=tk.X)
        hdr.pack_propagate(False)
        tk.Label(hdr, text=f"  {title}", bg=BG1, fg=T1,
                 font=("Segoe UI", 8, "bold")).pack(side=tk.LEFT, pady=5)
        self._badge = tk.Label(hdr, text="", bg=BG1, fg=VIOLET, font=FM)
        self._badge.pack(side=tk.RIGHT, padx=8)

        self.cv = tk.Canvas(self, bg=BG0, highlightthickness=0, cursor="crosshair")
        self.cv.pack(fill=tk.BOTH, expand=True)
        self.cv.bind("<Configure>", lambda _: self._rerender())
        self._show_placeholder()

    def _show_placeholder(self):
        self.cv.delete("all")
        w = max(self.cv.winfo_width(), 100)
        h = max(self.cv.winfo_height(), 100)
        self.cv.create_rectangle(0, 0, w, h, fill=BG0, outline="")
        self.cv.create_text(w//2, h//2 - 14, text="⬜",
                            font=("Segoe UI", 32), fill=BG3)
        self.cv.create_text(w//2, h//2 + 22, text="No image loaded",
                            fill=T2, font=FS)

    def _rerender(self):
        if self._last is not None:
            self._render(self._last)
        else:
            self._show_placeholder()

    def _render(self, img_bgr: np.ndarray):
        cw = self.cv.winfo_width()
        ch = self.cv.winfo_height()
        if cw < 10 or ch < 10:
            return
        ih, iw = img_bgr.shape[:2]
        scale = min(cw / iw, ch / ih)
        nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))

        rgb   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil   = Image.fromarray(rgb).resize((nw, nh), Image.LANCZOS)
        photo = ImageTk.PhotoImage(pil)

        self.cv.delete("all")
        self.cv.create_image(cw // 2, ch // 2, image=photo, anchor="center")
        self._photo = photo
        self._badge.config(text=f"{iw}×{ih}")

    def show(self, img_bgr: np.ndarray):
        self._last = img_bgr
        self._render(img_bgr)

    def clear(self):
        self._last  = None
        self._photo = None
        self._badge.config(text="")
        self._show_placeholder()


class _SliderRow(tk.Frame):
    """Compact label + tk.Scale + live value display."""

    def __init__(self, parent, label, key, vmin, vmax, init, color, sv: dict, on_change_cmd=None, **kw):
        super().__init__(parent, bg=BG2, **kw)
        self._key = key
        self._sv  = sv
        self._color = color
        self._on_change_cmd = on_change_cmd
        self._is_sigma = (key in ("sigma", "lowpass_sigma", "highpass_sigma"))

        tk.Label(self, text=label, bg=BG2, fg=T1, font=FS,
                 width=11, anchor="w").pack(side=tk.LEFT, padx=(6, 0))

        self._val_var = tk.StringVar(value=self._fmt(init))
        tk.Label(self, textvariable=self._val_var, bg=BG2, fg=color,
                 font=FM, width=5, anchor="e").pack(side=tk.RIGHT, padx=(0, 8))

        self._scale = tk.Scale(
            self, from_=vmin, to=vmax, orient=tk.HORIZONTAL,
            bg=BG2, fg=T1, troughcolor=BG3,
            activebackground=color, highlightthickness=0,
            bd=0, relief=tk.FLAT, showvalue=False, width=8,
            resolution=1.0 if self._is_sigma else 0.01,
        )
        self._scale.set(init)
        self._scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self._scale.config(command=self._on_change)

    def _fmt(self, v: float) -> str:
        return f"{v:.0f}" if self._is_sigma else f"{v:.2f}"

    def _on_change(self, val):
        v = float(val)
        self._sv[self._key] = v
        self._val_var.set(self._fmt(v))
        # Auto-trigger processing for filter sliders
        if self._on_change_cmd and self._key in ("lowpass_sigma", "highpass_sigma"):
            self._on_change_cmd()

    def reset(self, val: float):
        self._scale.set(val)
        self._sv[self._key] = float(val)
        self._val_var.set(self._fmt(float(val)))


# ── MAIN APPLICATION ───────────────────────────────────────────────────────────

class PhotoixApp:

    def __init__(self, root: tk.Tk):
        self.root = root

        # state
        self.orig_img     : np.ndarray | None = None
        self.proc_img     : np.ndarray | None = None
        self.landmarks    : np.ndarray | None = None
        self.show_lm      : bool = False
        self.current_file : str  = ""
        self.metrics      : dict = {}
        self.fft_data     : dict = {}
        self.params_log   : dict = {}
        self._processing  : bool = False
        self.current_view : str  = "images"
        self._spin_idx    : int  = 0
        self._spin_stop   : bool = True

        # camera
        self.camera_cap     = None
        self.camera_running = False
        self.camera_job     = None
        self.camera_frame   = None
        self._cam_photo     = None
        self._cam_source    = None
        self._cam_failures  = 0
        self._cam_candidates = self._build_camera_candidates()
        self._cam_candidate_idx = 0

        # slider values
        self._sv           = dict(smile=0.0, eyebrow=0.0, lip=0.0, slim=0.0, sigma=50.0, 
                                  lowpass_sigma=0.0, highpass_sigma=0.0)
        self._slider_rows  : list[_SliderRow] = []

        # control vars
        self.warp_var           = tk.StringVar(value="delaunay")
        self.aging_var          = tk.StringVar(value="none")
        self.overlay_active     = tk.StringVar(value="none")
        self.lowpass_enabled    = tk.BooleanVar(value=False)
        self.highpass_enabled   = tk.BooleanVar(value=False)
        self.hair_style_var     = tk.StringVar(value="none")
        self.hair_color_var     = tk.StringVar(value="original")


        self._setup_window()
        self._setup_styles()
        self._build_sidebar()
        self._build_main_area()
        self._build_statusbar()
        self._set_status("Ready  ·  load a face image to begin.")

    # ── window ────────────────────────────────────────────────────────────────
    def _setup_window(self):
        self.root.title("Photix  ·  Facial Processing Studio")
        self.root.geometry("1420x840")
        self.root.minsize(1100, 660)
        self.root.configure(bg=BG0)
        self.root.resizable(True, True)
        try:
            self.root.iconbitmap("")
        except Exception:
            pass
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _setup_styles(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure("TRadiobutton",
                        background=BG1, foreground=T1,
                        font=FN, focuscolor=BG1)
        style.map("TRadiobutton",
                  foreground=[("active", VIOLET), ("selected", T0)],
                  background=[("active", BG1)],
                  indicatorcolor=[("selected", VIOLET), ("!selected", BG3)])

    # ── SIDEBAR ───────────────────────────────────────────────────────────────
    def _build_sidebar(self):
        outer = tk.Frame(self.root, bg=BORDER, width=262)
        outer.pack(side=tk.LEFT, fill=tk.Y)
        outer.pack_propagate(False)

        inner = tk.Frame(outer, bg=BG1, width=260)
        inner.pack(fill=tk.BOTH, expand=True, padx=(0, 1))
        inner.pack_propagate(False)

        canv = tk.Canvas(inner, bg=BG1, highlightthickness=0, bd=0)
        sb   = ttk.Scrollbar(inner, orient=tk.VERTICAL, command=canv.yview)
        canv.configure(yscrollcommand=sb.set)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        canv.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        frame = tk.Frame(canv, bg=BG1)
        _win  = canv.create_window((0, 0), window=frame, anchor="nw")
        canv.bind("<Configure>", lambda e: canv.itemconfig(_win, width=e.width))
        frame.bind("<Configure>",
                   lambda e: canv.config(scrollregion=canv.bbox("all")))
        canv.bind_all("<MouseWheel>",
                      lambda e: canv.yview_scroll(int(-e.delta / 120), "units"))

        self._fill_sidebar(frame)

    def _fill_sidebar(self, p: tk.Frame):
        # ── logo ──────────────────────────────────────────────────────────
        hdr = tk.Frame(p, bg=BG0)
        hdr.pack(fill=tk.X)
        lf = tk.Frame(hdr, bg=BG0)
        lf.pack(fill=tk.X, padx=12, pady=11)
        tk.Label(lf, text="◈  PHOTIX", bg=BG0, fg=VIOLET, font=FL).pack(side=tk.LEFT)
        tk.Label(lf, text="DSP Studio", bg=BG0, fg=T2, font=FS).pack(side=tk.RIGHT, pady=2)
        tk.Frame(p, bg=BORDER, height=1).pack(fill=tk.X)

        # ── file ──────────────────────────────────────────────────────────
        _sect(p, "FILE")
        _Btn(p, "Browse Image…", self.load_image,
             accent=VIOLET, fg=T0, icon="📂").pack(fill=tk.X, padx=8, pady=(2,1))

        cam_row = tk.Frame(p, bg=BG1)
        cam_row.pack(fill=tk.X, padx=8, pady=(0, 2))
        _Btn(cam_row, "Open",    self.open_camera,   accent=BG3).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 1))
        _Btn(cam_row, "Capture", self.capture_photo, accent=GREEN, fg=BG0).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 1))
        _Btn(cam_row, "Close",   self.close_camera,  accent=RED).pack(
            side=tk.LEFT, fill=tk.X, expand=True)

        self._cam_cv = tk.Canvas(p, width=242, height=136, bg=BG0,
                                  highlightthickness=1,
                                  highlightbackground=BORDER)
        self._cam_cv.pack(padx=8, pady=4)
        self._cam_cv.create_text(121, 68, text="Camera preview", fill=T2, font=FS)

        self._file_lbl = tk.Label(p, text="No file loaded", bg=BG1, fg=T2,
                                   font=FS, wraplength=230, justify=tk.LEFT)
        self._file_lbl.pack(padx=10, anchor="w", pady=(0, 4))
        _sep(p)

        # ── landmarks ─────────────────────────────────────────────────────
        _sect(p, "LANDMARKS")
        self._lm_btn = _Btn(p, "Show Landmarks", self.toggle_landmarks,
                             accent=BG3, icon="◎")
        self._lm_btn.pack(fill=tk.X, padx=8, pady=3)
        _sep(p)

        # ── warp ──────────────────────────────────────────────────────────
        _sect(p, "WARP METHOD")
        wf = tk.Frame(p, bg=BG1)
        wf.pack(fill=tk.X, padx=14, pady=2)
        for txt, val in [("Delaunay Affine",   "delaunay"),
                          ("Thin-Plate Spline", "tps"),
                          ("Optical Flow",      "flow")]:
            ttk.Radiobutton(wf, text=txt, variable=self.warp_var,
                            value=val).pack(anchor="w", pady=1)
        _sep(p)

        # ── aging ─────────────────────────────────────────────────────────
        _sect(p, "AGING  /  DE-AGING")
        af = tk.Frame(p, bg=BG1)
        af.pack(fill=tk.X, padx=14, pady=2)
        for txt, val in [("None",       "none"),
                          ("Age  (+)",   "age"),
                          ("De-Age (−)", "deage")]:
            ttk.Radiobutton(af, text=txt, variable=self.aging_var,
                            value=val).pack(anchor="w", pady=1)
        tk.Label(p, text="  σ controls strength  →  slider below",
                 bg=BG1, fg=T2, font=FS).pack(padx=10, anchor="w", pady=(0, 4))
        _sep(p)

        # ── frequency filters ─────────────────────────────────────────────
        _sect(p, "FREQUENCY FILTERS")
        ff = tk.Frame(p, bg=BG1)
        ff.pack(fill=tk.X, padx=14, pady=(2, 4))
        
        # Low-pass checkbox - auto-apply on toggle
        cb_lp = ttk.Checkbutton(ff, text="✓ Low-Pass (Smoothing)", 
                       variable=self.lowpass_enabled,
                       command=self.apply_processing)
        cb_lp.pack(anchor="w", pady=2)
        tk.Label(ff, text="  Use slider below to adjust strength", 
                bg=BG1, fg=T2, font=FS).pack(padx=10, anchor="w", pady=(0, 1))
        
        # High-pass checkbox - auto-apply on toggle
        cb_hp = ttk.Checkbutton(ff, text="✓ High-Pass (Sharpening)", 
                       variable=self.highpass_enabled,
                       command=self.apply_processing)
        cb_hp.pack(anchor="w", pady=2)
        tk.Label(ff, text="  Use slider below to adjust strength", 
                bg=BG1, fg=T2, font=FS).pack(padx=10, anchor="w", pady=(0, 4))
        _sep(p)

        # ── face overlays ─────────────────────────────────────────────────
        _sect(p, "FACE OVERLAYS")
        of = tk.Frame(p, bg=BG1)
        of.pack(fill=tk.X, padx=14, pady=(0, 4))
        tk.Label(of, text="Off", bg=BG1, fg=T1,
                 font=("Segoe UI", 8, "bold")).pack(anchor="w", pady=(2, 0))
        ttk.Radiobutton(of, text="None", variable=self.overlay_active,
                        value="none").pack(anchor="w", pady=1)
        tk.Label(of, text="Sunglasses", bg=BG1, fg=T1,
                 font=("Segoe UI", 8, "bold")).pack(anchor="w", pady=(6, 0))
        for txt, val in [("Classic", "sg_classic"), ("Aviator", "sg_aviator"),
                          ("Round", "sg_round"), ("Clear", "sg_glasses")]:
            ttk.Radiobutton(of, text=txt, variable=self.overlay_active,
                            value=val).pack(anchor="w", pady=1)
        tk.Label(of, text="Beard", bg=BG1, fg=T1,
                 font=("Segoe UI", 8, "bold")).pack(anchor="w", pady=(6, 0))
        for txt, val in [("Black", "bd_black"), ("Brown", "bd_brown"),
                          ("Blonde", "bd_blonde")]:
            ttk.Radiobutton(of, text=txt, variable=self.overlay_active,
                            value=val).pack(anchor="w", pady=1)
            
        tk.Label(of, text="Hair Style", bg=BG1, fg=T1,
         font=("Segoe UI", 8, "bold")).pack(anchor="w", pady=(6, 0))

        hair_style_box = ttk.Combobox(
            of,
            textvariable=self.hair_style_var,
            values=["none", "long"],
            state="readonly",
            width=18,
        )
        hair_style_box.pack(anchor="w", pady=2)
        hair_style_box.bind("<<ComboboxSelected>>", lambda _e: self.apply_processing())

        tk.Label(of, text="Hair Color", bg=BG1, fg=T1,
                font=("Segoe UI", 8, "bold")).pack(anchor="w", pady=(6, 0))

        hair_color_box = ttk.Combobox(
            of,
            textvariable=self.hair_color_var,
            values=["original", "black", "brown", "blonde", "red", "blue", "gray"],
            state="readonly",
            width=18,
        )
        hair_color_box.pack(anchor="w", pady=2)
        hair_color_box.bind("<<ComboboxSelected>>", lambda _e: self.apply_processing())
        _sep(p)

        # ── view ──────────────────────────────────────────────────────────
        _sect(p, "VIEW")
        vf = tk.Frame(p, bg=BG1)
        vf.pack(fill=tk.X, padx=8, pady=2)
        self._view_btns: dict[str, _Btn] = {}
        for lbl, key in [("  Images",      "images"),
                          ("∿  FFT Spectra", "fft"),
                          ("▦  Metrics",    "metrics")]:
            b = _Btn(vf, lbl, lambda k=key: self.switch_view(k), accent=BG3)
            b.pack(fill=tk.X, pady=1)
            self._view_btns[key] = b
        self._highlight_view_btn()
        _sep(p)

        # ── actions ───────────────────────────────────────────────────────
        _sect(p, "ACTIONS")
        _Btn(p, "Apply Processing", self.apply_processing,
             accent=GREEN, fg=BG0, icon="▶").pack(fill=tk.X, padx=8, pady=(2, 1))
        _Btn(p, "Reset to Original", self.reset,
             accent=BG3).pack(fill=tk.X, padx=8, pady=(0, 2))
        _sep(p)

        # ── export ────────────────────────────────────────────────────────
        _sect(p, "EXPORT")
        ef = tk.Frame(p, bg=BG1)
        ef.pack(fill=tk.X, padx=8, pady=2)
        _Btn(ef, "Export CSV",  self.export_csv,  accent=AMBER, fg=BG0).pack(fill=tk.X, pady=1)
        _Btn(ef, "Export PDF",  self.export_pdf,  accent=VIOLET).pack(fill=tk.X, pady=1)
        _Btn(ef, "Save PNG",    self.save_images, accent=BG3).pack(fill=tk.X, pady=1)

        tk.Frame(p, height=20, bg=BG1).pack()

    # ── MAIN AREA ─────────────────────────────────────────────────────────────
    def _build_main_area(self):
        self._main = tk.Frame(self.root, bg=BG0)
        self._main.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # ── images view (native — fast) ───────────────────────────────────
        self._img_view = tk.Frame(self._main, bg=BG0)

        panes = tk.Frame(self._img_view, bg=BG0)
        panes.pack(fill=tk.BOTH, expand=True, padx=6, pady=(6, 3))

        self._pane_orig = _ImgPane(panes, "ORIGINAL")
        self._pane_orig.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 3))

        self._pane_proc = _ImgPane(panes, "PROCESSED")
        self._pane_proc.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(3, 0))

        # sliders panel
        sl_wrap = tk.Frame(self._img_view, bg=BG2)
        sl_wrap.pack(fill=tk.X, padx=6, pady=(0, 6))
        tk.Frame(sl_wrap, bg=BORDER, height=1).pack(fill=tk.X)

        sl_inner = tk.Frame(sl_wrap, bg=BG2)
        sl_inner.pack(fill=tk.BOTH, expand=True, padx=10, pady=6)

        specs = [
            ("Smile",          "smile",            0.0,   1.0,  0.0,  VIOLET),
            ("Eyebrow",        "eyebrow",          0.0,   1.0,  0.0,  CYAN),
            ("Lip Wide",       "lip",              0.0,   1.0,  0.0,  GREEN),
            ("Face Slim",      "slim",             0.0,   1.0,  0.0,  PINK),
            ("Aging σ",        "sigma",            1.0, 100.0, 50.0,  AMBER),
            ("LP Filter σ",    "lowpass_sigma",    0.0,  40.0,  0.0,  CYAN),
            ("HP Filter σ",    "highpass_sigma",   0.0,  30.0,  0.0,  VIOLET),
        ]
        self._slider_rows = []
        for lbl, key, vmin, vmax, init, color in specs:
            row = _SliderRow(sl_inner, lbl, key, vmin, vmax, init, color, self._sv, 
                           on_change_cmd=self.apply_processing)
            row.pack(fill=tk.X, pady=1)
            self._slider_rows.append(row)

        # ── matplotlib view (FFT / Metrics) ──────────────────────────────
        self._mpl_view = tk.Frame(self._main, bg=BG0)

        matplotlib.rcParams.update(MPL_RC)
        self.fig = Figure(figsize=(11, 7.5), dpi=100)
        self.fig.patch.set_facecolor(BG2)

        self.mpl_canvas = FigureCanvasTkAgg(self.fig, master=self._mpl_view)
        self.mpl_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True,
                                             padx=6, pady=6)

        # start in images view
        self._img_view.pack(fill=tk.BOTH, expand=True)

    # ── STATUS BAR ────────────────────────────────────────────────────────────
    def _build_statusbar(self):
        bar = tk.Frame(self.root, bg=BG1, height=24)
        bar.pack(side=tk.BOTTOM, fill=tk.X)
        bar.pack_propagate(False)
        tk.Frame(bar, bg=BORDER, height=1).pack(fill=tk.X, side=tk.TOP)
        self._status_var = tk.StringVar(value="")
        self._info_var   = tk.StringVar(value="")
        tk.Label(bar, textvariable=self._status_var, bg=BG1, fg=T2,
                 font=FM, anchor="w", padx=12).pack(side=tk.LEFT)
        tk.Label(bar, textvariable=self._info_var, bg=BG1, fg=T2,
                 font=FM, anchor="e", padx=12).pack(side=tk.RIGHT)

    def _set_status(self, msg: str, info: str = ""):
        self._status_var.set(f"  {msg}")
        self._info_var.set(info)
        self.root.update_idletasks()

    # ── SPINNER ───────────────────────────────────────────────────────────────
    def _start_spinner(self, base: str):
        self._spin_base  = base
        self._spin_stop  = False
        self._spin_idx   = 0
        self._tick_spinner()

    def _tick_spinner(self):
        if self._spin_stop:
            return
        c = SPIN[self._spin_idx % len(SPIN)]
        self._status_var.set(f"  {c}  {self._spin_base}")
        self._spin_idx += 1
        self._spin_job = self.root.after(80, self._tick_spinner)

    def _stop_spinner(self):
        self._spin_stop = True
        if hasattr(self, "_spin_job"):
            try:
                self.root.after_cancel(self._spin_job)
            except Exception:
                pass

    # ── VIEW SWITCHING ────────────────────────────────────────────────────────
    def switch_view(self, view: str):
        self.current_view = view
        self._highlight_view_btn()
        if view == "images":
            self._mpl_view.pack_forget()
            self._img_view.pack(fill=tk.BOTH, expand=True)
            self._refresh_images()
        else:
            self._img_view.pack_forget()
            self._mpl_view.pack(fill=tk.BOTH, expand=True)
            self._draw_mpl_view()

    def _highlight_view_btn(self):
        for k, b in self._view_btns.items():
            if k == self.current_view:
                b.recolor(VIOLET)
            else:
                b.recolor(BG3)

    # ── IMAGE REFRESH ─────────────────────────────────────────────────────────
    def _refresh_images(self):
        if self.orig_img is not None:
            disp = draw_landmarks(self.orig_img, self.landmarks, self.show_lm)
            self._pane_orig.show(disp)
        else:
            self._pane_orig.clear()

        if self.proc_img is not None:
            self._pane_proc.show(self.proc_img)
        else:
            self._pane_proc.clear()

    # ── MATPLOTLIB VIEWS ──────────────────────────────────────────────────────
    def _draw_mpl_view(self):
        self.fig.clear()
        if self.current_view == "fft":
            self._draw_fft_view()
        elif self.current_view == "metrics":
            self._draw_metrics_view()
        self.mpl_canvas.draw_idle()

    def _draw_fft_view(self):
        if self.orig_img is None:
            ax = self.fig.add_subplot(1, 1, 1)
            ax.text(0.5, 0.5, "Load an image first", ha="center", va="center",
                    color=T2, fontsize=13, transform=ax.transAxes)
            ax.axis("off")
            return
        gs = GridSpec(2, 3, figure=self.fig, wspace=0.35, hspace=0.40,
                      left=0.06, right=0.97, top=0.90, bottom=0.08)

        ax_o = self.fig.add_subplot(gs[0, 0])
        ax_p = self.fig.add_subplot(gs[0, 1])
        ax_o.imshow(cv2.cvtColor(self.orig_img, cv2.COLOR_BGR2RGB))
        ax_o.set_title("Original", color=T0); ax_o.axis("off")

        if self.proc_img is not None:
            ax_p.imshow(cv2.cvtColor(self.proc_img, cv2.COLOR_BGR2RGB))
            ax_p.set_title("Processed", color=T0)
        else:
            ax_p.text(0.5, 0.5, "Not processed yet", ha="center", va="center",
                      color=T2, transform=ax_p.transAxes)
        ax_p.axis("off")

        oe  = self.fft_data.get("orig_energy", {})
        pe  = self.fft_data.get("proc_energy", {})
        tbl_ax = self.fig.add_subplot(gs[0, 2])
        tbl_ax.axis("off")
        rows = [
            ["",      "Original",                  "Processed"],
            ["H/L",   f"{oe.get('ratio',0):.4f}",  f"{pe.get('ratio',0):.4f}"],
            ["Low",   f"{oe.get('low',0):.2e}",    f"{pe.get('low',0):.2e}"],
            ["High",  f"{oe.get('high',0):.2e}",   f"{pe.get('high',0):.2e}"],
            ["Total", f"{oe.get('total',0):.2e}",  f"{pe.get('total',0):.2e}"],
        ]
        tbl = tbl_ax.table(cellText=rows, loc="center", cellLoc="center",
                            bbox=[0, 0.1, 1.0, 0.85])
        tbl.auto_set_font_size(False); tbl.set_fontsize(9)
        for (r, c), cell in tbl.get_celld().items():
            cell.set_facecolor(BG3 if r == 0 else BG2)
            cell.set_edgecolor(BORDER)
            cell.set_text_props(color=T0 if r == 0 else T1)
        tbl_ax.set_title("Spectral Energy", color=T0, pad=6)

        orig_spec = compute_magnitude_spectrum(self.orig_img)
        ax1 = self.fig.add_subplot(gs[1, 0])
        im1 = ax1.imshow(orig_spec, cmap="inferno", origin="lower", aspect="auto")
        ax1.set_title("FFT — Original", color=T0); ax1.axis("off")
        self.fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04).ax.yaxis.label.set_color(T2)

        if self.proc_img is not None:
            proc_spec = self.fft_data.get("proc_spectrum",
                                          compute_magnitude_spectrum(self.proc_img))
            ax2 = self.fig.add_subplot(gs[1, 1])
            im2 = ax2.imshow(proc_spec, cmap="inferno", origin="lower", aspect="auto")
            ax2.set_title("FFT — Processed", color=T0); ax2.axis("off")
            self.fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04).ax.yaxis.label.set_color(T2)

            ax3 = self.fig.add_subplot(gs[1, 2])
            imd = ax3.imshow(orig_spec - proc_spec, cmap="RdBu_r",
                             origin="lower", aspect="auto")
            ax3.set_title("Spectral Δ", color=T0); ax3.axis("off")
            self.fig.colorbar(imd, ax=ax3, fraction=0.046, pad=0.04).ax.yaxis.label.set_color(T2)

        self.fig.suptitle("Frequency Domain Analysis",
                          color=T0, fontsize=11, fontweight="bold")

    def _draw_metrics_view(self):
        gs = GridSpec(2, 2, figure=self.fig, wspace=0.35, hspace=0.45,
                      left=0.08, right=0.96, top=0.88, bottom=0.08)
        has = bool(self.metrics)

        ax_bar = self.fig.add_subplot(gs[0, :])
        if has:
            labels = ["MSE", "PSNR (dB)", "SSIM × 100"]
            vals   = [self.metrics.get("MSE", 0),
                      min(self.metrics.get("PSNR", 0), 60),
                      self.metrics.get("SSIM", 0) * 100]
            bars = ax_bar.bar(labels, vals, color=[RED, VIOLET, GREEN],
                              width=0.45, edgecolor=BORDER, linewidth=0.6)
            for bar, val in zip(bars, vals):
                ax_bar.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.5,
                            f"{val:.4f}", ha="center", va="bottom",
                            color=T0, fontsize=10, fontweight="bold")
            ax_bar.set_facecolor(BG0)
            ax_bar.set_title("Quality Metrics", color=T0, pad=8)
            ax_bar.set_ylabel("Value", color=T2)
            ax_bar.tick_params(colors=T2)
            ax_bar.set_ylim(0, max(vals) * 1.25 + 1)
            for sp in ax_bar.spines.values():
                sp.set_edgecolor(BORDER)
        else:
            ax_bar.text(0.5, 0.5, "Run  Apply Processing  to compute metrics",
                        ha="center", va="center", color=T2, fontsize=12,
                        transform=ax_bar.transAxes)
            ax_bar.set_title("Quality Metrics", color=T0)
            ax_bar.axis("off")

        ax_hl = self.fig.add_subplot(gs[1, 0])
        if self.fft_data:
            oe = self.fft_data.get("orig_energy", {})
            pe = self.fft_data.get("proc_energy", {})
            x  = np.arange(2); w = 0.32
            ax_hl.bar(x - w/2, [oe.get("low",0), oe.get("high",0)], w,
                      label="Original",  color=VIOLET, alpha=0.85, edgecolor=BORDER)
            ax_hl.bar(x + w/2, [pe.get("low",0), pe.get("high",0)], w,
                      label="Processed", color=AMBER,  alpha=0.85, edgecolor=BORDER)
            ax_hl.set_xticks(x); ax_hl.set_xticklabels(["Low Energy", "High Energy"])
            ax_hl.legend(framealpha=0.3, labelcolor=T1,
                         facecolor=BG2, edgecolor=BORDER)
            ax_hl.set_title("Spectral Energy", color=T0)
            ax_hl.set_facecolor(BG0)
            ax_hl.set_ylabel("Energy", color=T2)
            ax_hl.tick_params(colors=T2)
            for sp in ax_hl.spines.values():
                sp.set_edgecolor(BORDER)
        else:
            ax_hl.text(0.5, 0.5, "No FFT data yet", ha="center", va="center",
                       color=T2, transform=ax_hl.transAxes)
            ax_hl.axis("off")

        ax_tbl = self.fig.add_subplot(gs[1, 1])
        ax_tbl.axis("off")
        rows = [["Metric", "Value", "Interpretation"]]
        if has:
            hl = self.fft_data.get("proc_energy", {}).get("ratio", float("nan"))
            rows += [
                ["MSE",       f"{self.metrics.get('MSE',float('nan')):.2f}",     "0 = identical"],
                ["PSNR",      f"{self.metrics.get('PSNR',float('nan')):.2f} dB", "Higher = better"],
                ["SSIM",      f"{self.metrics.get('SSIM',float('nan')):.4f}",    "1.0 = identical"],
                ["H/L Ratio", f"{hl:.4f}",                                       "Higher = texture"],
            ]
        tbl = ax_tbl.table(cellText=rows, loc="center", cellLoc="left",
                            bbox=[0, 0.05, 1.0, 0.90])
        tbl.auto_set_font_size(False); tbl.set_fontsize(9)
        for (r, c), cell in tbl.get_celld().items():
            cell.set_facecolor(BG3 if r == 0 else BG2)
            cell.set_edgecolor(BORDER)
            cell.set_text_props(color=CYAN if r == 0 else T1)
        ax_tbl.set_title("Numeric Summary", color=T0, pad=6)
        self.fig.suptitle("Metrics & Evaluation",
                          color=T0, fontsize=11, fontweight="bold")

    # ── LOAD IMAGE (FR-01, FR-02) ─────────────────────────────────────────────
    def load_image(self):
        path = filedialog.askopenfilename(
            title="Open face image",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp"),
                       ("JPEG", "*.jpg *.jpeg"), ("PNG", "*.png"), ("BMP", "*.bmp")])
        if not path:
            return
        try:
            self._set_status("Loading…")
            raw           = load_image(path)
            self.orig_img = preprocess(raw)
            self.proc_img = self.orig_img.copy()
            self.current_file = path
            self._file_lbl.configure(text=Path(path).name, fg=VIOLET)

            self._set_status("Detecting landmarks…")
            self.landmarks = detect_landmarks(self.orig_img)
            lm_txt = (f"✔  {len(self.landmarks)} landmarks"
                      if self.landmarks is not None else "⚠  No face detected")

            self.metrics = {}; self.fft_data = {}
            if self.current_view == "images":
                self._refresh_images()
            self._set_status(f"{Path(path).name}  ·  {lm_txt}")
        except Exception as exc:
            messagebox.showerror("Load error", str(exc))
            self._set_status("Load failed.")

    # ── CAMERA ────────────────────────────────────────────────────────────────
    def _show_camera_message(self, text: str):
        self._cam_photo = None
        self._cam_cv.delete("all")
        self._cam_cv.create_text(121, 68, text=text, fill=T2, font=FS,
                                 width=210, justify=tk.CENTER)

    def _build_camera_candidates(self) -> list[tuple[int, int]]:
        indices = range(5)
        if sys.platform.startswith("win"):
            backends = (cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY)
        elif sys.platform == "darwin":
            backends = (getattr(cv2, "CAP_AVFOUNDATION", cv2.CAP_ANY), cv2.CAP_ANY)
        else:
            backends = (getattr(cv2, "CAP_V4L2", cv2.CAP_ANY), cv2.CAP_ANY)

        seen: set[tuple[int, int]] = set()
        candidates: list[tuple[int, int]] = []
        for backend in backends:
            for index in indices:
                candidate = (index, backend)
                if candidate not in seen:
                    candidates.append(candidate)
                    seen.add(candidate)
        return candidates

    def _camera_backend_name(self, backend: int) -> str:
        names = {
            cv2.CAP_DSHOW: "DirectShow",
            cv2.CAP_MSMF: "Media Foundation",
            getattr(cv2, "CAP_AVFOUNDATION", -1): "AVFoundation",
            getattr(cv2, "CAP_V4L2", -2): "V4L2",
            cv2.CAP_ANY: "default backend",
        }
        return names.get(backend, f"backend {backend}")

    def _open_camera_candidate(self, index: int, backend: int):
        trial = cv2.VideoCapture(index, backend)
        if not trial.isOpened():
            trial.release()
            return None, None

        trial.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        trial.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
        trial.set(cv2.CAP_PROP_FPS, 30)
        try:
            trial.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        # Warm up: many drivers return no frame for the first few reads.
        frame = None
        ok = False
        for _ in range(40):
            ok, frame = trial.read()
            if ok and frame is not None and frame.size:
                return trial, frame

        trial.release()
        return None, None

    def open_camera(self):
        if self.camera_running:
            return
        self._show_camera_message("Opening camera...")
        candidates = list(self._cam_candidates)
        cap = None
        source = None
        first_frame = None
        self._cam_candidate_idx = 0

        for idx, (index, backend) in enumerate(candidates):
            trial, frame = self._open_camera_candidate(index, backend)
            if trial is not None and frame is not None:
                cap = trial
                source = (index, backend)
                first_frame = frame
                self._cam_candidate_idx = idx
                break

        if cap is None or first_frame is None:
            self._show_camera_message("Camera not available")
            messagebox.showerror(
                "Camera Error",
                "Camera could not be opened or did not return a preview frame.\n\n"
                "Close other apps using the camera, check Windows camera privacy "
                "permissions, then try again."
            )
            return

        self.camera_cap = cap
        self._cam_source = source
        self.camera_frame = first_frame.copy()
        self._cam_failures = 0
        self.camera_running = True
        idx, backend = source
        self._set_status(f"Camera live: device {idx} ({self._camera_backend_name(backend)}).")
        self._update_camera_preview()

    def _reopen_camera_next_backend(self):
        """Try the next backend/index combo when the stream is blank."""
        if self.camera_cap:
            try:
                self.camera_cap.release()
            except Exception:
                pass
        self.camera_cap = None
        self.camera_frame = None
        self._cam_photo = None
        self._cam_failures = 0

        self._show_camera_message("Re-opening camera...")
        total = len(self._cam_candidates)
        for offset in range(1, total + 1):
            candidate_idx = (self._cam_candidate_idx + offset) % total
            index, backend = self._cam_candidates[candidate_idx]
            trial, frame = self._open_camera_candidate(index, backend)
            if trial is None or frame is None:
                continue
            self._cam_candidate_idx = candidate_idx
            self.camera_cap = trial
            self._cam_source = (index, backend)
            self.camera_frame = frame.copy()
            self._set_status(f"Camera live: device {index} ({self._camera_backend_name(backend)}).")
            return

        self._show_camera_message("Camera not available")
        self.camera_running = False
        self._cam_source = None
        self._set_status("Camera stream unavailable.")

    def _update_camera_preview(self):
        if not self.camera_running or self.camera_cap is None:
            return
        ok, frame = self.camera_cap.read()
        if ok and frame is not None and frame.size:
            self._cam_failures = 0
            self.camera_frame = frame.copy()
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb   = cv2.resize(rgb, (242, 136))
            photo = ImageTk.PhotoImage(Image.fromarray(rgb))
            self._cam_photo = photo
            self._cam_cv.delete("all")
            self._cam_cv.create_image(0, 0, image=photo, anchor="nw")
        else:
            self._cam_failures += 1
            if self._cam_failures == 10:
                self._show_camera_message("Waiting for camera frame...")
                self._set_status("Camera opened, waiting for frames.")
            # If we keep getting blank frames, try another backend/index automatically.
            if self._cam_failures == 60:
                self._set_status("Camera stream blank; retrying…")
                self._reopen_camera_next_backend()
        self.camera_job = self.root.after(30, self._update_camera_preview)

    def capture_photo(self):
        if self.camera_frame is None:
            messagebox.showwarning("Camera", "No frame available.")
            return
        if self._processing:
            return
        self._processing = True
        captured = self.camera_frame.copy()
        self._start_spinner("Detecting landmarks…")

        def _run():
            try:
                img = preprocess(captured)
                lm  = detect_landmarks(img)
                self.root.after(0, lambda: self._on_capture_done(img, lm))
            except Exception as exc:
                err = str(exc)
                self.root.after(0, lambda m=err: self._on_processing_error(m))

        threading.Thread(target=_run, daemon=True).start()

    def _on_capture_done(self, img: np.ndarray, lm: "np.ndarray | None"):
        self._processing = False
        self._stop_spinner()
        self.orig_img     = img
        self.proc_img     = img.copy()
        self.landmarks    = lm
        self.current_file = "Camera Capture"
        self.metrics = {}; self.fft_data = {}
        self._file_lbl.configure(text="Camera Capture", fg=VIOLET)
        if self.current_view == "images":
            self._refresh_images()
        lm_txt = (f"✔  {len(lm)} landmarks" if lm is not None else "⚠  No face detected")  # type: ignore[arg-type]
        self._set_status(f"Camera Capture  ·  {lm_txt}")

    def close_camera(self):
        self.camera_running = False
        if self.camera_job:
            try:
                self.root.after_cancel(self.camera_job)
            except Exception:
                pass
            self.camera_job = None
        if self.camera_cap:
            self.camera_cap.release()
            self.camera_cap = None
        self.camera_frame = None
        self._cam_source = None
        self._cam_failures = 0
        self._show_camera_message("Camera preview")
        self._set_status("Camera closed.")

    def _on_close(self):
        self.close_camera()
        self.root.destroy()

    # ── LANDMARKS (FR-04) ─────────────────────────────────────────────────────
    def toggle_landmarks(self):
        self.show_lm = not self.show_lm
        if self.show_lm:
            self._lm_btn.configure(text="◉  Hide Landmarks")
            self._lm_btn.recolor(GREEN, BG0)
        else:
            self._lm_btn.configure(text="◎  Show Landmarks")
            self._lm_btn.recolor(BG3, T0)
        if self.current_view == "images":
            self._refresh_images()

    # ── APPLY PROCESSING (FR-05..24) ─────────────────────────────────────────
    def apply_processing(self):
        if self.orig_img is None:
            messagebox.showwarning("No image", "Please load an image first.")
            return
        if self._processing:
            return
        self._processing = True
        self._start_spinner("Processing…")

        def _run():
            try:
                img     = self.orig_img.copy()
                lm      = self.landmarks
                smile   = self._sv["smile"]
                eyebrow = self._sv["eyebrow"]
                lip     = self._sv["lip"]
                slim    = self._sv["slim"]
                sigma   = self._sv["sigma"]
                warp    = self.warp_var.get()
                aging   = self.aging_var.get()
                _ov     = self.overlay_active.get()
                glasses = _ov[3:] if _ov.startswith("sg_") else "none"
                beard   = _ov[3:] if _ov.startswith("bd_") else "none"
                hair_style = self.hair_style_var.get()
                hair_color = self.hair_color_var.get()


                if lm is not None and (smile + eyebrow + lip + slim) > 0.01:
                    src_pts, dst_pts = compute_combined_displacement(
                        lm, smile, eyebrow, lip, slim)
                    if   warp == "delaunay": img = apply_delaunay_warp(img, src_pts, dst_pts)
                    elif warp == "tps":      img = apply_tps_warp(img, src_pts, dst_pts)
                    elif warp == "flow":     img = apply_optical_flow_warp(img, src_pts, dst_pts)

                if   aging == "age":   img = apply_aging(img, sigma, lm)
                elif aging == "deage": img = apply_deaging(img, sigma, lm)

                # ── Apply frequency-domain filters ───────────────────────
                # Low-pass (smoothing)
                if self.lowpass_enabled.get() and self._sv["lowpass_sigma"] > 0.5:
                    try:
                        lp_sigma = float(self._sv["lowpass_sigma"])
                        img = apply_lowpass_filter(img, sigma=lp_sigma, filter_type=FilterType.GAUSSIAN)
                    except Exception as e:
                        _log.warning(f"Low-pass filter failed: {e}")
                
                # High-pass (sharpening) - blend for visible enhancement
                if self.highpass_enabled.get() and self._sv["highpass_sigma"] > 0.5:
                    try:
                        hp_sigma = float(self._sv["highpass_sigma"])
                        hp_result = apply_highpass_filter(img, sigma=hp_sigma, filter_type=FilterType.GAUSSIAN)
                        # Blend: keep 70% original, add 30% high-pass for enhancement
                        blend_amt = min(0.5, hp_sigma / 100.0)  # Scale blend by sigma
                        img = cv2.addWeighted(img.astype(np.float32), 0.7, 
                                             hp_result.astype(np.float32), 0.3 * blend_amt, 0)
                        img = np.clip(img, 0, 255).astype(np.uint8)
                    except Exception as e:
                        _log.warning(f"High-pass filter failed: {e}")

                if lm is not None and (glasses != "none" or beard != "none" or hair_style != "none"):
                    effects_list, params = [], {}
                    if glasses != "none":
                        effects_list.append("sunglasses")
                        params["sunglasses"] = {"style": glasses}
                    if beard != "none":
                        effects_list.append("beard")
                        params["beard"] = {"color": beard}
                    if hair_style != "none":
                        effects_list.append("hair")
                        params["hair"] = {
                            "style": hair_style,
                            "color": hair_color,
                        }
                    result = FaceOverlayEngine(lm, img).apply(effects_list, params)
                    img = result.final_image

                self.proc_img   = np.clip(img, 0, 255).astype(np.uint8)
                self.fft_data   = compare_spectra(self.orig_img, self.proc_img)
                self.metrics    = compute_all_metrics(self.orig_img, self.proc_img)
                self.params_log = {
                    "File": Path(self.current_file).name, "Warp": warp,
                    "Aging": aging, "Smile": smile, "Eyebrow": eyebrow,
                    "Lip Wide": lip, "Face Slim": slim, "Sigma": sigma,
                    "LowPass": self.lowpass_enabled.get(), "LowPass σ": self._sv["lowpass_sigma"],
                    "HighPass": self.highpass_enabled.get(), "HighPass σ": self._sv["highpass_sigma"],
                    "Overlay": _ov,
                    "Timestamp": datetime.now().isoformat(),
                }
                self.root.after(0, self._on_processing_done)
            except Exception as exc:
                err = str(exc)
                self.root.after(0, lambda m=err: self._on_processing_error(m))

        threading.Thread(target=_run, daemon=True).start()

    def _on_processing_done(self):
        self._processing = False
        self._stop_spinner()
        if self.current_view == "images":
            self._refresh_images()
        else:
            self._draw_mpl_view()
        mse  = self.metrics.get("MSE",  0)
        psnr = self.metrics.get("PSNR", 0)
        ssim = self.metrics.get("SSIM", 0)
        hl   = self.fft_data.get("proc_energy", {}).get("ratio", 0)
        self._set_status(
            f"Done  ·  MSE={mse:.2f}  PSNR={psnr:.2f} dB  SSIM={ssim:.4f}",
            info=f"H/L={hl:.4f}")

    def _on_processing_error(self, msg: str):
        self._processing = False
        self._stop_spinner()
        messagebox.showerror("Processing error", msg)
        self._set_status("Processing failed.")

    # ── RESET ─────────────────────────────────────────────────────────────────
    def reset(self):
        if self.orig_img is None:
            return
        self.proc_img = self.orig_img.copy()
        self.metrics  = {}
        self.fft_data = {}
        defaults = dict(
            smile=0.0,
            eyebrow=0.0,
            lip=0.0,
            slim=0.0,
            sigma=50.0,
            lowpass_sigma=0.0,
            highpass_sigma=0.0,
        )
        for k, v in defaults.items():
            self._sv[k] = v
        for row in self._slider_rows:
            row.reset(defaults[row._key])
        if self.current_view == "images":
            self._refresh_images()
        self._set_status("Reset to original.")

    # ── EXPORT (FR-25, FR-28) ─────────────────────────────────────────────────
    def export_csv(self):
        if not self.metrics:
            messagebox.showwarning("No data", "Apply processing first.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".csv", filetypes=[("CSV", "*.csv")],
            initialfile="photix_report.csv")
        if not path:
            return
        try:
            out = export_csv(self.metrics, self.fft_data, self.params_log, path)
            self._set_status(f"CSV saved: {Path(out).name}")
            messagebox.showinfo("Saved", f"CSV saved to:\n{out}")
        except Exception as exc:
            messagebox.showerror("Export error", str(exc))

    def export_pdf(self):
        if self.orig_img is None:
            messagebox.showwarning("No image", "Load an image first.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".pdf", filetypes=[("PDF", "*.pdf")],
            initialfile="photix_report.pdf")
        if not path:
            return
        try:
            proc = self.proc_img if self.proc_img is not None else self.orig_img
            out  = export_pdf(self.orig_img, proc, self.metrics, self.fft_data, path)
            self._set_status(f"PDF saved: {Path(out).name}")
            messagebox.showinfo("Saved", f"PDF saved to:\n{out}")
        except Exception as exc:
            messagebox.showerror("Export error", str(exc))

    def save_images(self):
        if self.orig_img is None:
            messagebox.showwarning("No image", "Load an image first.")
            return
        folder = filedialog.askdirectory(title="Select output folder")
        if not folder:
            return
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        try:
            p1   = save_image(self.orig_img,
                              str(Path(folder) / f"photix_orig_{ts}.png"))
            proc = self.proc_img if self.proc_img is not None else self.orig_img
            p2   = save_image(proc,
                              str(Path(folder) / f"photix_proc_{ts}.png"))
            self._set_status(f"Images saved  →  {folder}")
            messagebox.showinfo("Saved",
                f"Original : {Path(p1).name}\nProcessed: {Path(p2).name}\n\n{folder}")
        except Exception as exc:
            messagebox.showerror("Save error", str(exc))


# ── ENTRY POINT ───────────────────────────────────────────────────────────────
def main():
    root = tk.Tk()
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass
    PhotoixApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
