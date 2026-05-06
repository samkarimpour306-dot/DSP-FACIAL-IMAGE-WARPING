"""
tps_warper.py — FR-14, FR-15
Thin-Plate Spline global warp with smooth C∞ displacement field.

ARCHITECTURE (v5):
  Same active-landmark-only selection as mesh_warper.py, but uses a TPS
  kernel (r² log r²) instead of piecewise-linear interpolation.
  TPS is naturally C∞, so no additional Gaussian smoothing is required.

  Backward mapping (correct for cv2.remap):
    Control points in DESTINATION space.
    TPS interpolates backward offsets (src − dst) at every output pixel.

  Key fix vs. v3:
    Old: uniform subsample of all 468 landmarks every 9-th step.
         Most sampled points have zero displacement → TPS collapses to flat.
    New: filter to actively displaced landmarks only → coarse, wide-triangle
         control mesh → deformation propagates across the full face.
"""
import cv2
import numpy as np
import scipy.linalg

_MIN_DISP = 0.8    # px — below this a landmark is considered inactive
_CHUNK    = 32768  # pixels per evaluation batch (memory safety)
_LAMBDA   = 1e-3   # Tikhonov regularisation

# Anatomical anchors — same set as mesh_warper.py.
# Zero-displacement interior face points that break up large TPS triangles
# and prevent the smooth C∞ field from leaking deformation to unrelated
# face regions (e.g., smile must not pull the nose or eyebrows).
_ANATOM_ANCHORS = [
    168, 6, 197, 4,      # nose bridge + tip
    159, 386,            # eye upper lids
    130, 359,            # eye outer corners
    133, 362,            # eye inner corners
    10, 67, 297, 54, 284,# forehead
    152,                 # chin centre
    116, 345,            # cheekbones
]


# ── TPS mathematics ────────────────────────────────────────────────────────

def _tps_kernel(r2: np.ndarray) -> np.ndarray:
    return np.where(r2 < 1e-12, 0.0, r2 * np.log(r2 + 1e-12))


def _build_system(ctrl: np.ndarray) -> tuple:
    """Build the (n+3)×(n+3) TPS linear system for control points ctrl."""
    n = len(ctrl)
    d = ctrl[:, None, :] - ctrl[None, :, :]
    K = _tps_kernel(np.sum(d ** 2, axis=2)) + _LAMBDA * np.eye(n)
    P = np.hstack([np.ones((n, 1)), ctrl])
    L = np.vstack([np.hstack([K, P]),
                   np.hstack([P.T, np.zeros((3, 3))])])
    return L, n


def _perimeter(H: int, W: int, n: int = 14) -> np.ndarray:
    """56 evenly-spaced anchor points around image perimeter (zero disp)."""
    ts = np.linspace(0, 1, n, endpoint=True)
    pts = np.vstack([
        np.c_[ts * (W - 1),  np.zeros(n)],
        np.c_[ts * (W - 1),  np.full(n, H - 1.)],
        np.c_[np.zeros(n),   ts * (H - 1)],
        np.c_[np.full(n, W - 1.), ts * (H - 1)],
    ])
    return np.unique(pts, axis=0).astype(np.float64)


def _solve_tps_backward(dst_ctrl: np.ndarray,
                        back_offsets: np.ndarray) -> tuple:
    """FR-14: solve TPS coefficients for the backward displacement field.

    dst_ctrl      — control point positions in destination (output) space.
    back_offsets  — src − dst for each control point (0 for anchors).
    Returns (cx, cy, n) where cx/cy are coefficient vectors.
    """
    L, n = _build_system(dst_ctrl)
    cx = scipy.linalg.solve(
        L, np.concatenate([back_offsets[:, 0], np.zeros(3)]))
    cy = scipy.linalg.solve(
        L, np.concatenate([back_offsets[:, 1], np.zeros(3)]))
    return cx, cy, n


def _eval_batch(coords: np.ndarray, ctrl: np.ndarray,
                n: int, cx: np.ndarray, cy: np.ndarray) -> tuple:
    """FR-15: evaluate TPS backward displacement at coords (M×2) in batches."""
    total  = len(coords)
    dx_out = np.empty(total, dtype=np.float64)
    dy_out = np.empty(total, dtype=np.float64)
    wx, ax = cx[:n], cx[n:]
    wy, ay = cy[:n], cy[n:]
    for s in range(0, total, _CHUNK):
        e  = min(s + _CHUNK, total)
        ch = coords[s:e]
        d  = ch[:, None, :] - ctrl[None, :, :]
        r2 = np.sum(d ** 2, axis=2)
        Kc = _tps_kernel(r2)
        Pc = np.hstack([np.ones((len(ch), 1)), ch])
        dx_out[s:e] = Kc @ wx + Pc @ ax
        dy_out[s:e] = Kc @ wy + Pc @ ay
    return dx_out, dy_out


# ── Public API ─────────────────────────────────────────────────────────────

def apply_tps_warp(
    src_img: np.ndarray,
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
    intensity: float = 1.0,
) -> np.ndarray:
    """FR-14, FR-15: TPS warp — active landmarks only + perimeter anchors.

    Naturally smooth (C∞) field — no Gaussian post-processing required.
    Backward mapping: for each output pixel p, the TPS gives the source
    position to sample: source = p + TPS_backward(p).
    """
    H, W = src_img.shape[:2]

    # ── Active landmark filter ─────────────────────────────────────────────
    disp   = dst_pts.astype(np.float64) - src_pts.astype(np.float64)
    active = np.linalg.norm(disp, axis=1) > _MIN_DISP

    if not np.any(active):
        return src_img.copy()

    a_src = src_pts[active].astype(np.float64)
    a_dst = dst_pts[active].astype(np.float64)
    back_active = a_src - a_dst     # backward offsets for active pts

    # Perimeter anchors keep the background fixed
    peri = _perimeter(H, W)

    # Anatomical interior anchors — prevent TPS from leaking deformation
    # to unrelated face regions (nose, eyes, chin, forehead).
    if len(src_pts) >= 468:
        anc_idx = [i for i in _ANATOM_ANCHORS if not active[i]]
        anc_pts = src_pts[np.array(anc_idx)].astype(np.float64) if anc_idx else np.empty((0, 2))
    else:
        anc_pts = np.empty((0, 2))

    # All control points in destination space
    all_dst  = np.vstack([a_dst, peri] + ([anc_pts] if len(anc_pts) else []))
    all_back = np.vstack([back_active,
                          np.zeros((len(peri),    2)),
                          np.zeros((len(anc_pts), 2))])

    # Solve TPS backward system (FR-14)
    cx, cy, n = _solve_tps_backward(all_dst, all_back)

    # Evaluate on 1/8 grid (4 096 pts) then bicubic-upscale — ~64× faster.
    # TPS is C∞ so the upscaled field is indistinguishable from full-res eval.
    _S = 8
    hs, ws = max(1, H // _S), max(1, W // _S)
    gy_s, gx_s = np.mgrid[0:hs, 0:ws]
    coords_s = np.column_stack([(gx_s.ravel() * _S),
                                  (gy_s.ravel() * _S)]).astype(np.float64)
    ddx_s, ddy_s = _eval_batch(coords_s, all_dst, n, cx, cy)

    ddx_img = cv2.resize(ddx_s.reshape(hs, ws).astype(np.float32),
                         (W, H), interpolation=cv2.INTER_CUBIC)
    ddy_img = cv2.resize(ddy_s.reshape(hs, ws).astype(np.float32),
                         (W, H), interpolation=cv2.INTER_CUBIC)

    gy, gx = np.mgrid[0:H, 0:W]
    # Build backward remap maps — FR-09 INTER_CUBIC, NFR-R02 clamp
    map_x = np.clip(gx.astype(np.float32) + ddx_img, 0, W - 1).astype(np.float32)
    map_y = np.clip(gy.astype(np.float32) + ddy_img, 0, H - 1).astype(np.float32)

    result = cv2.remap(src_img, map_x, map_y,
                       cv2.INTER_CUBIC,
                       borderMode=cv2.BORDER_REPLICATE)
    return np.clip(result, 0, 255).astype(np.uint8)
