"""
mesh_warper.py — FR-10, FR-11
Sparse Delaunay inverse-displacement-field warp.

ARCHITECTURE (v5 — smooth deformation):
  1. Filter landmarks to ONLY the actively displaced ones (|disp| > _MIN_DISP).
     Typical active set: 10–40 points per expression.
  2. Add 56 perimeter anchors (zero backward offset) to keep background fixed.
  3. Build LinearNDInterpolator (= Delaunay + per-triangle affine) on the
     merged set in destination space  →  backward offset field.
  4. GAUSSIAN SMOOTHING of the raw displacement field (sigma=_SMOOTH_SIGMA).
     This converts the piecewise-linear (C0) field into a smooth C∞ field,
     eliminating the "kink" artefacts that appear at triangle boundaries and
     that users perceive as "face tearing".
  5. cv2.remap(INTER_CUBIC, BORDER_REPLICATE) for pixel-perfect output.
"""
import cv2
import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from scipy.ndimage import gaussian_filter

_MIN_DISP     = 0.8    # px — threshold to classify a landmark as active
_SMOOTH_SIGMA = 12.0   # px — Gaussian smoothing applied to raw disp field

# Anatomical anchors — zero-displacement interior face points.
# These break up the large Delaunay triangles that previously stretched
# from the mouth/eyebrow region all the way to the image perimeter,
# causing the deformation to spread globally across the face.
#
# Landmark indices (MediaPipe 468-point model):
#   nose bridge/tip : 168, 6, 197, 4
#   eye upper lids  : 159, 386
#   eye outer corners: 130, 359
#   eye inner corners: 133, 362
#   forehead        : 10, 67, 297, 54, 284
#   chin centre     : 152
#   cheekbones      : 116, 345
_ANATOM_ANCHORS = [
    168, 6, 197, 4,
    159, 386,
    130, 359,
    133, 362,
    10, 67, 297, 54, 284,
    152,
    116, 345,
]


def _perimeter(H: int, W: int, n: int = 14) -> np.ndarray:
    """56 evenly-spaced anchor points around the image perimeter (zero disp)."""
    ts = np.linspace(0, 1, n, endpoint=True)
    pts = np.vstack([
        np.c_[ts * (W - 1),  np.zeros(n)],
        np.c_[ts * (W - 1),  np.full(n, H - 1.)],
        np.c_[np.zeros(n),   ts * (H - 1)],
        np.c_[np.full(n, W - 1.), ts * (H - 1)],
    ])
    return np.unique(pts, axis=0).astype(np.float64)


def apply_delaunay_warp(
    src_img: np.ndarray,
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
) -> np.ndarray:
    """FR-10, FR-11: Sparse Delaunay warp with smooth displacement field.

    Backward mapping semantics (correct for cv2.remap):
        For each output pixel p, the backward offset tells us which source
        pixel to sample: source = p + backward_offset(p).
    Control points live in destination space:
        - Active displaced points: backward_offset = src − dst
        - Perimeter anchors:       backward_offset = 0
    """
    H, W = src_img.shape[:2]

    # ── Step 1: active landmark filter ────────────────────────────────────
    disp   = dst_pts.astype(np.float64) - src_pts.astype(np.float64)
    active = np.linalg.norm(disp, axis=1) > _MIN_DISP

    if not np.any(active):
        return src_img.copy()

    a_src = src_pts[active].astype(np.float64)
    a_dst = dst_pts[active].astype(np.float64)
    back_ctrl = a_src - a_dst                       # backward offsets

    # ── Step 2: perimeter anchors + anatomical interior anchors ──────────
    peri = _perimeter(H, W)

    # Extract anatomical anchors from src_pts (only non-active ones).
    # These enforce zero displacement at nose, eyes, chin, forehead —
    # preventing the interpolation from spreading the effect globally.
    if len(src_pts) >= 468:
        anc_idx = [i for i in _ANATOM_ANCHORS if not active[i]]
        anc_pts = src_pts[np.array(anc_idx)].astype(np.float64) if anc_idx else np.empty((0, 2))
    else:
        anc_pts = np.empty((0, 2))

    all_dst  = np.vstack([a_dst, peri] + ([anc_pts] if len(anc_pts) else []))
    all_back = np.vstack([back_ctrl,
                          np.zeros((len(peri),    2)),
                          np.zeros((len(anc_pts), 2))])

    # ── Step 3: piecewise-linear interpolation (FR-10 Delaunay, FR-11 affine)
    lin = LinearNDInterpolator(all_dst, all_back, fill_value=np.nan)
    nn  = NearestNDInterpolator(all_dst, all_back)

    # Query at 1/8 resolution (4 096 pts) then bicubic-upscale — ~64× faster,
    # visually identical because displacement fields are smooth at this scale.
    _S = 8
    hs, ws = max(1, H // _S), max(1, W // _S)
    gy_s, gx_s = np.mgrid[0:hs, 0:ws]
    xi_s = np.column_stack([(gx_s.ravel() * _S).astype(np.float64),
                              (gy_s.ravel() * _S).astype(np.float64)])

    field_s = lin(xi_s)
    bad_s   = np.isnan(field_s[:, 0])
    if bad_s.any():
        field_s[bad_s] = nn(xi_s[bad_s])

    bdx_raw = cv2.resize(field_s[:, 0].reshape(hs, ws).astype(np.float32),
                         (W, H), interpolation=cv2.INTER_CUBIC)
    bdy_raw = cv2.resize(field_s[:, 1].reshape(hs, ws).astype(np.float32),
                         (W, H), interpolation=cv2.INTER_CUBIC)

    gy, gx = np.mgrid[0:H, 0:W]

    # ── Step 4: Gaussian smoothing — C0 → C∞ (eliminates triangle kinks) ─
    bdx = gaussian_filter(bdx_raw, sigma=_SMOOTH_SIGMA).astype(np.float32)
    bdy = gaussian_filter(bdy_raw, sigma=_SMOOTH_SIGMA).astype(np.float32)

    # ── Step 5: remap — FR-09 INTER_CUBIC, NFR-R02 clamp ─────────────────
    map_x = np.clip(gx.astype(np.float32) + bdx, 0, W - 1)
    map_y = np.clip(gy.astype(np.float32) + bdy, 0, H - 1)

    result = cv2.remap(src_img, map_x, map_y,
                       cv2.INTER_CUBIC,
                       borderMode=cv2.BORDER_REPLICATE)
    return np.clip(result, 0, 255).astype(np.uint8)


# ── Per-triangle reference (FR-11 documentation) ──────────────────────────
def _warp_triangle(src_img, dst_img, src_tri, dst_tri):
    """FR-11 reference: per-triangle affine inverse warp (INTER_CUBIC)."""
    H, W = src_img.shape[:2]
    sr = cv2.boundingRect(src_tri.reshape(-1, 1, 2).astype(np.float32))
    dr = cv2.boundingRect(dst_tri.reshape(-1, 1, 2).astype(np.float32))
    sx, sy, sw, sh = sr
    dx, dy, dw, dh = dr
    if sw <= 0 or sh <= 0 or dw <= 0 or dh <= 0:
        return
    M = cv2.getAffineTransform(
        (dst_tri - [dx, dy]).astype(np.float32),
        (src_tri - [sx, sy]).astype(np.float32))
    sx1, sy1 = max(0, sx), max(0, sy)
    sx2, sy2 = min(sx + sw, W), min(sy + sh, H)
    if sx2 <= sx1 or sy2 <= sy1:
        return
    patch = src_img[sy1:sy2, sx1:sx2]
    pt, pb = sy1 - sy, sy + sh - sy2
    pl, pr = sx1 - sx, sx + sw - sx2
    if pt or pb or pl or pr:
        patch = cv2.copyMakeBorder(patch, pt, pb, pl, pr, cv2.BORDER_REPLICATE)
    warped = cv2.warpAffine(patch, M, (dw, dh),
                            flags=cv2.INTER_CUBIC,
                            borderMode=cv2.BORDER_REPLICATE)
    mask = np.zeros((dh, dw), dtype=np.uint8)
    cv2.fillConvexPoly(mask, (dst_tri - [dx, dy]).astype(np.int32), 255)
    dx1, dy1 = max(0, dx), max(0, dy)
    dx2, dy2 = min(dx + dw, W), min(dy + dh, H)
    if dx2 <= dx1 or dy2 <= dy1:
        return
    wc = warped[dy1 - dy:dy2 - dy, dx1 - dx:dx2 - dx]
    mc = mask[dy1 - dy:dy2 - dy, dx1 - dx:dx2 - dx]
    if wc.size == 0:
        return
    m3  = mc[:, :, np.newaxis]
    roi = dst_img[dy1:dy2, dx1:dx2].astype(np.float32)
    dst_img[dy1:dy2, dx1:dx2] = np.clip(
        np.where(m3 > 0, wc.astype(np.float32), roi), 0, 255
    ).astype(np.uint8)
