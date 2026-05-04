import numpy as np
import cv2

# ── 1. MATH HELPERS ──────────────────────────────────────────────────────────

def _gauss_lp(shape: tuple, sigma: float) -> np.ndarray:
    H, W = shape
    cy, cx = H // 2, W // 2
    y = (np.arange(H) - cy).astype(np.float64)
    x = (np.arange(W) - cx).astype(np.float64)
    X, Y = np.meshgrid(x, y)
    return np.exp(-(X ** 2 + Y ** 2) / (2.0 * sigma ** 2))

def _fft_filter(ch: np.ndarray, filt: np.ndarray) -> np.ndarray:
    F = np.fft.fftshift(np.fft.fft2(ch))
    return np.fft.ifft2(np.fft.ifftshift(F * filt)).real

def _face_mask(landmarks: np.ndarray, H: int, W: int, feather: float = 25.0) -> np.ndarray:
    pts = landmarks[:, :2].astype(np.int32)
    hull = cv2.convexHull(pts)
    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)
    soft = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), sigmaX=feather) / 255.0
    return soft

def _safe_pts(landmarks: np.ndarray, idx: list[int]) -> np.ndarray | None:
    valid = [i for i in idx if i < len(landmarks)]
    if len(valid) < 2:
        return None
    return landmarks[valid, :2].astype(np.float32)

def _face_width(landmarks: np.ndarray, W: int) -> float:
    if len(landmarks) > 454:
        return max(24.0, abs(float(landmarks[454, 0]) - float(landmarks[234, 0])))
    return W * 0.58

def _soft_ellipse_mask(
    shape: tuple[int, int],
    center: tuple[float, float],
    radius_x: float,
    radius_y: float,
    feather: float = 13.0,
) -> np.ndarray:
    H, W = shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    cx, cy = center
    dist = ((xx - cx) / max(radius_x, 1.0)) ** 2 + ((yy - cy) / max(radius_y, 1.0)) ** 2
    mask = np.clip(1.0 - dist, 0.0, 1.0)
    return cv2.GaussianBlur(mask, (0, 0), feather)

def _blend_mask(base: np.ndarray, effect: np.ndarray, mask: np.ndarray, amount: float = 1.0) -> np.ndarray:
    m3d = np.clip(mask * amount, 0.0, 1.0)[:, :, np.newaxis].astype(np.float32)
    out = m3d * effect.astype(np.float32) + (1.0 - m3d) * base.astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)

def _local_scale(
    img: np.ndarray,
    center: tuple[float, float],
    radius_x: float,
    radius_y: float,
    scale_x: float,
    scale_y: float,
    amount: float,
) -> np.ndarray:
    """Scale a local oval region with soft blending, useful for face shape edits."""
    H, W = img.shape[:2]
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    cx, cy = center
    local = _soft_ellipse_mask((H, W), center, radius_x, radius_y, feather=max(5.0, min(radius_x, radius_y) * 0.12))
    local = np.clip(local * amount, 0.0, 1.0)

    sx = max(0.55, scale_x)
    sy = max(0.55, scale_y)
    map_x = cx + (xx - cx) / sx
    map_y = cy + (yy - cy) / sy
    warped = cv2.remap(
        img,
        np.clip(map_x, 0, W - 1).astype(np.float32),
        np.clip(map_y, 0, H - 1).astype(np.float32),
        cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return _blend_mask(img, warped, local, 1.0)

def _deterministic_noise(shape: tuple[int, int], salt: int = 0) -> np.ndarray:
    rng = np.random.default_rng(shape[0] * 73856093 ^ shape[1] * 19349663 ^ salt)
    return rng.random(shape, dtype=np.float32)

# ── 2. AGING — WRINKLE, HAIR-GRAYING & SAG ENGINE ───────────────────────────

def _apply_wrinkle_texture(img: np.ndarray, landmarks: np.ndarray, intensity: float) -> np.ndarray:
    """Overlay directional wrinkle texture on strategic face regions."""
    H, W = img.shape[:2]
    overlay = np.zeros_like(img, dtype=np.float32)

    regions = [
        landmarks[[10, 338, 297, 332, 284, 251, 21, 54, 103, 67, 109]],  # forehead
        landmarks[[33, 133, 157, 158, 159, 160, 161, 246]],               # left eye
        landmarks[[263, 362, 384, 385, 386, 387, 388, 466]],              # right eye
        landmarks[[186, 92, 165, 167]],                                    # nasolabial left
        landmarks[[410, 322, 391, 393]],                                   # nasolabial right
    ]

    for pts in regions:
        rect = cv2.boundingRect(pts[:, :2].astype(np.int32))
        x, y, w, h = rect
        if w < 5 or h < 5:
            continue

        # Horizontal blur with capped kernel so noise variance is preserved
        noise = np.random.normal(0, 1, (h, w)).astype(np.float32)
        kw = min(51, w | 1)
        noise = cv2.GaussianBlur(noise, (kw, 3), 0)
        # Only darken (wrinkle valleys) — avoids random lightening artifacts
        noise = np.abs(noise)

        # Build region mask via convex hull so points are properly ordered
        local_pts = (pts[:, :2] - [x, y]).astype(np.int32)
        hull = cv2.convexHull(local_pts)
        mask_u8 = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(mask_u8, hull, 255)
        mask_f = cv2.GaussianBlur(mask_u8.astype(np.float32), (11, 11), 0) / 255.0

        for c in range(3):
            overlay[y:y+h, x:x+w, c] += noise * mask_f * intensity * 55.0

    result = img.astype(np.float32) - overlay
    return np.clip(result, 0, 255).astype(np.uint8)

def _apply_hair_graying(img: np.ndarray, landmarks: np.ndarray, intensity: float) -> np.ndarray:
    """Desaturate hair toward white/silver to simulate age-related colour loss.

    Strategy:
      1. Estimate hairline Y from the topmost forehead landmarks.
      2. Build a spatial weight that is 1.0 above the hairline and fades to 0
         over a short band below it (so hair roots are included but the face is not).
      3. Multiply by a dark-pixel weight — hair is typically darker than skin or sky,
         which suppresses false positives in bright backgrounds.
      4. Blend matched pixels toward a desaturated silver/white target.
    """
    H, W = img.shape[:2]

    # Hairline estimate: lowest 10th-percentile Y among top forehead landmarks
    forehead_idx = [10, 338, 297, 332, 284, 251, 21, 54, 103, 67, 109]
    hairline_y = float(np.percentile(landmarks[forehead_idx, 1], 10))

    # Spatial mask: 1.0 above hairline, linear fade to 0 over `fade_zone` px below
    fade_zone = max(20.0, hairline_y * 0.20)
    yy = np.arange(H, dtype=np.float32)[:, np.newaxis]
    y_weight = np.clip((hairline_y + fade_zone - yy) / fade_zone, 0.0, 1.0)
    y_weight = np.broadcast_to(y_weight, (H, W)).copy()

    # Colour mask: darker pixels → more likely to be hair (not bright skin/sky)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    brightness = hsv[:, :, 2]                            # 0–255
    dark_weight = np.clip((160.0 - brightness) / 120.0, 0.0, 1.0)

    # Combined hair probability map — smooth to avoid hard edges
    hair_mask = y_weight * dark_weight
    hair_mask = cv2.GaussianBlur(hair_mask, (21, 21), 0)
    hair_mask = np.clip(hair_mask * 2.0, 0.0, 1.0)

    # Silver/white target: desaturate + boost luminance
    gray_lum = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    silver = np.clip(gray_lum * 1.2 + 55.0, 0.0, 235.0)
    target = np.stack([silver, silver, silver], axis=2)

    m3d = (hair_mask * intensity)[:, :, np.newaxis]
    result = m3d * target + (1.0 - m3d) * img.astype(np.float32)
    return np.clip(result, 0, 255).astype(np.uint8)

def _apply_deep_wrinkle_lines(img: np.ndarray, landmarks: np.ndarray, intensity: float) -> np.ndarray:
    """Add stable expression-line shadows/highlights instead of random noise only."""
    H, W = img.shape[:2]
    overlay = np.zeros((H, W), dtype=np.float32)

    regions = [
        ([10, 338, 297, 332, 284, 251, 21, 54, 103, 67, 109], 0.0, 7, 0.95),
        ([33, 133, 157, 158, 159, 160, 161, 246], -0.45, 5, 0.75),
        ([263, 362, 384, 385, 386, 387, 388, 466], 0.45, 5, 0.75),
        ([186, 92, 165, 167, 205, 50], 0.95, 4, 0.85),
        ([410, 322, 391, 393, 425, 280], -0.95, 4, 0.85),
        ([61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291], 0.0, 4, 0.65),
    ]

    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    for idx, angle, bands, strength in regions:
        pts = _safe_pts(landmarks, idx)
        if pts is None:
            continue
        x, y, w, h = cv2.boundingRect(pts.astype(np.int32))
        if w < 5 or h < 5:
            continue
        pad = max(10, int(max(w, h) * 0.35))
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2, y2 = min(W, x + w + pad), min(H, y + h + pad)
        cx, cy = float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1]))
        rx, ry = max(10.0, w * 0.85 + pad), max(8.0, h * 0.75 + pad)
        mask = _soft_ellipse_mask((H, W), (cx, cy), rx, ry, feather=9.0)

        lx = xx[y1:y2, x1:x2] - cx
        ly = yy[y1:y2, x1:x2] - cy
        coord = np.cos(angle) * ly + np.sin(angle) * lx
        waves = np.maximum(0.0, np.sin(coord * (np.pi * bands / max(ry * 2.0, 1.0))))
        fine = cv2.GaussianBlur(waves.astype(np.float32), (0, 0), 0.8)
        overlay[y1:y2, x1:x2] += fine * mask[y1:y2, x1:x2] * strength

    overlay = cv2.GaussianBlur(overlay, (0, 0), 0.65)
    dark = img.astype(np.float32) - overlay[:, :, np.newaxis] * (42.0 * intensity)
    light = dark + cv2.GaussianBlur(overlay, (0, 0), 2.0)[:, :, np.newaxis] * (12.0 * intensity)
    face_m = _face_mask(landmarks, H, W, feather=18.0)
    return _blend_mask(img, np.clip(light, 0, 255).astype(np.uint8), face_m, min(1.0, 0.85 + intensity * 0.15))

def _apply_age_spots_and_pores(img: np.ndarray, landmarks: np.ndarray, intensity: float) -> np.ndarray:
    H, W = img.shape[:2]
    face_m = _face_mask(landmarks, H, W, feather=18.0)
    noise = _deterministic_noise((H, W), salt=17)
    coarse = cv2.GaussianBlur(noise, (0, 0), 2.2)
    spot_mask = ((coarse > (0.69 - intensity * 0.08)).astype(np.float32) * face_m)
    spot_mask = cv2.GaussianBlur(spot_mask, (0, 0), 1.4)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] + spot_mask * 34.0 * intensity, 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] - spot_mask * 42.0 * intensity, 0, 255)
    spotted = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)

    pores = cv2.GaussianBlur(_deterministic_noise((H, W), salt=29), (0, 0), 0.55) - 0.5
    pore_layer = spotted.astype(np.float32) + pores[:, :, np.newaxis] * face_m[:, :, np.newaxis] * 20.0 * intensity
    return np.clip(pore_layer, 0, 255).astype(np.uint8)

def _apply_eye_bags_and_hollows(img: np.ndarray, landmarks: np.ndarray, intensity: float) -> np.ndarray:
    H, W = img.shape[:2]
    result = img.astype(np.float32)
    for idx in ([33, 7, 163, 144, 145, 153, 154, 155, 133],
                [263, 249, 390, 373, 374, 380, 381, 382, 362]):
        pts = _safe_pts(landmarks, idx)
        if pts is None:
            continue
        cx, cy = float(np.mean(pts[:, 0])), float(np.max(pts[:, 1]) + 8.0)
        w = max(18.0, float(np.ptp(pts[:, 0])) * 0.72)
        h = max(8.0, float(np.ptp(pts[:, 1])) * 1.1)
        mask = _soft_ellipse_mask((H, W), (cx, cy), w, h, feather=7.0)
        result[:, :, 0] += mask * 5.0 * intensity
        result[:, :, 1] -= mask * 12.0 * intensity
        result[:, :, 2] -= mask * 17.0 * intensity
    return np.clip(result, 0, 255).astype(np.uint8)

def _apply_mature_color_grade(img: np.ndarray, landmarks: np.ndarray, intensity: float) -> np.ndarray:
    H, W = img.shape[:2]
    face_m = _face_mask(landmarks, H, W, feather=22.0)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab[:, :, 0] = np.clip(lab[:, :, 0] * (1.0 - 0.07 * intensity), 0, 255)
    lab[:, :, 1] = np.clip(lab[:, :, 1] + 3.0 * intensity, 0, 255)
    lab[:, :, 2] = np.clip(lab[:, :, 2] + 7.0 * intensity, 0, 255)
    graded = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)

    hsv = cv2.cvtColor(graded, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (1.0 - 0.22 * intensity), 0, 255)
    graded = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
    return _blend_mask(img, graded, face_m, 1.0)

def _apply_mature_geometry(img: np.ndarray, landmarks: np.ndarray, intensity: float) -> np.ndarray:
    H, W = img.shape[:2]
    out = img
    fw = _face_width(landmarks, W)
    if len(landmarks) > 152:
        chin = tuple(landmarks[152, :2].astype(float))
        out = _local_scale(out, (chin[0], chin[1] - fw * 0.05), fw * 0.36, fw * 0.30, 0.96, 1.10, intensity * 0.55)
    for idx in ([50, 117, 123, 147, 187, 205], [280, 346, 352, 376, 411, 425]):
        pts = _safe_pts(landmarks, idx)
        if pts is None:
            continue
        center = (float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1]) + fw * 0.05))
        out = _local_scale(out, center, fw * 0.22, fw * 0.28, 0.98, 1.08, intensity * 0.45)
    return out


def apply_aging(img: np.ndarray, sigma: float = 20.0, landmarks: np.ndarray | None = None) -> np.ndarray:
    H, W = img.shape[:2]
    norm_val = float(np.clip(sigma / 100.0, 0.0, 1.0))
    
    # 1. FFT high-pass: enhance skin texture / micro-contrast
    lp = _gauss_lp((H, W), max(2.0, 7.0 - (norm_val * 4.5)))
    hp = 1.0 - lp
    temp = img.astype(np.float64)
    for c in range(3):
        hp_comp = _fft_filter(temp[:, :, c], hp)
        temp[:, :, c] += np.clip(hp_comp, -20, 20) * norm_val * 3.5

    res = np.clip(temp, 0, 255).astype(np.uint8)

    if landmarks is not None:
        # 2. Mature geometry: heavier lower face and softer cheek sag.
        res = _apply_mature_geometry(res, landmarks, norm_val)

        # 3. Wrinkle detail: random fine texture plus deterministic expression lines.
        if len(landmarks) > 466:
            res = _apply_wrinkle_texture(res, landmarks, norm_val)
        res = _apply_deep_wrinkle_lines(res, landmarks, norm_val)

        # 4. Gravitational sag - pull content downward.
        face_m = _face_mask(landmarks, H, W)
        gy, gx = np.mgrid[0:H, 0:W].astype(np.float32)
        dy = (norm_val * 28.0 * face_m).astype(np.float32)

        map_y = np.clip(gy - dy, 0.0, H - 1.0)
        res = cv2.remap(res, gx, map_y, cv2.INTER_CUBIC)

        # Blend back: only the face area gets the effect; background restored here
        m3d = face_m[:, :, np.newaxis]
        res = m3d * res.astype(np.float64) + (1.0 - m3d) * img.astype(np.float64)
        res = np.clip(res, 0, 255).astype(np.uint8)

        # 5. Real-world age cues: duller skin, eye hollows, spots/pores, gray hair.
        res = _apply_mature_color_grade(res, landmarks, norm_val)
        res = _apply_eye_bags_and_hollows(res, landmarks, norm_val)
        res = _apply_age_spots_and_pores(res, landmarks, norm_val)
        res = _apply_hair_graying(res, landmarks, norm_val)

    return np.clip(res, 0, 255).astype(np.uint8)

# ── 3. CHILDISH HELPERS ──────────────────────────────────────────────────────

def _apply_rosy_cheeks(img: np.ndarray, landmarks: np.ndarray, intensity: float) -> np.ndarray:
    """Blend a warm rosy tint onto each cheek — the most distinctive child feature."""
    H, W = img.shape[:2]
    result = img.astype(np.float32)

    left_cheek_idx  = [116, 123, 147, 187, 205, 36, 137]
    right_cheek_idx = [345, 352, 376, 411, 425, 266, 366]

    # Warm pink/coral blush (BGR)
    blush = np.array([140.0, 130.0, 215.0], dtype=np.float32)

    n = len(landmarks)
    if 454 < n and 234 < n:
        face_w = abs(float(landmarks[454, 0]) - float(landmarks[234, 0]))
    else:
        face_w = W * 0.6
    radius = max(20.0, face_w * 0.18)

    y_g, x_g = np.mgrid[0:H, 0:W].astype(np.float32)

    for cheek_idx in (left_cheek_idx, right_cheek_idx):
        valid = [i for i in cheek_idx if i < n]
        if not valid:
            continue
        pts = landmarks[valid, :2]
        cx = float(np.mean(pts[:, 0]))
        cy = float(np.mean(pts[:, 1]))

        gauss = np.exp(-((x_g - cx) ** 2 + (y_g - cy) ** 2) / (2.0 * radius ** 2))
        alpha = np.clip(gauss * intensity * 0.60, 0.0, 0.55)
        m3d   = alpha[:, :, np.newaxis]

        tinted = result * 0.80 + blush * 0.20
        result = (1.0 - m3d) * result + m3d * tinted

    return np.clip(result, 0, 255).astype(np.uint8)


def _apply_baby_smooth(img: np.ndarray, landmarks: np.ndarray, intensity: float) -> np.ndarray:
    """Full-face multi-pass bilateral filter for flawless baby-skin texture."""
    face_m = _face_mask(landmarks, img.shape[0], img.shape[1], feather=30.0)

    sigma_c = 34.0 + intensity * 30.0
    sigma_s = 11.0 + intensity * 12.0
    d       = 9 if intensity < 0.5 else 13
    passes  = 1 + int(intensity > 0.75)

    smooth = img.copy()
    for _ in range(passes):
        smooth = cv2.bilateralFilter(smooth, d, sigma_c, sigma_s)

    m3d    = (face_m * min(intensity * 0.52, 0.62))[:, :, np.newaxis]
    result = m3d * smooth.astype(np.float32) + (1.0 - m3d) * img.astype(np.float32)
    return np.clip(result, 0, 255).astype(np.uint8)


def _apply_bright_eyes(img: np.ndarray, landmarks: np.ndarray, intensity: float) -> np.ndarray:
    """Brighten the eye region for a wide, childlike appearance."""
    H, W = img.shape[:2]
    result = img.astype(np.float32)

    left_eye_idx  = [33,  7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
    right_eye_idx = [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466]

    n = len(landmarks)
    for eye_idx in (left_eye_idx, right_eye_idx):
        valid = [i for i in eye_idx if i < n]
        if not valid:
            continue
        pts = landmarks[valid, :2]
        x, y, w, h = cv2.boundingRect(pts.astype(np.int32))
        pad = max(12, int(h * 0.6))
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2, y2 = min(W, x + w + pad), min(H, y + h + pad)

        local_pts = (pts - np.array([x1, y1])).astype(np.int32)
        hull      = cv2.convexHull(local_pts)
        mask_u8   = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
        cv2.fillConvexPoly(mask_u8, hull, 255)
        mask_f = cv2.GaussianBlur(mask_u8.astype(np.float32), (21, 21), 0) / 255.0
        m3d    = (mask_f * intensity * 0.70)[:, :, np.newaxis]

        roi        = result[y1:y2, x1:x2]
        brightened = np.clip(roi + 25.0 * intensity, 0, 255)
        result[y1:y2, x1:x2] = (1.0 - m3d) * roi + m3d * brightened

    return np.clip(result, 0, 255).astype(np.uint8)

def _apply_child_face_geometry(img: np.ndarray, landmarks: np.ndarray, intensity: float) -> np.ndarray:
    """Round the face and enlarge the eyes for a more childlike read."""
    H, W = img.shape[:2]
    out = img
    fw = _face_width(landmarks, W)

    for idx in ([33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
                [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466]):
        pts = _safe_pts(landmarks, idx)
        if pts is None:
            continue
        cx, cy = float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1]))
        rx = max(14.0, float(np.ptp(pts[:, 0])) * 0.90)
        ry = max(9.0, float(np.ptp(pts[:, 1])) * 1.55)
        out = _local_scale(out, (cx, cy), rx, ry, 1.08 + 0.08 * intensity, 1.10 + 0.10 * intensity, intensity)

    for idx in ([116, 123, 147, 187, 205, 36, 137], [345, 352, 376, 411, 425, 266, 366]):
        pts = _safe_pts(landmarks, idx)
        if pts is None:
            continue
        center = (float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1]) + fw * 0.02))
        out = _local_scale(out, center, fw * 0.24, fw * 0.22, 1.05 + 0.07 * intensity, 1.04 + 0.05 * intensity, intensity * 0.70)

    if len(landmarks) > 152:
        chin = tuple(landmarks[152, :2].astype(float))
        out = _local_scale(out, (chin[0], chin[1] - fw * 0.03), fw * 0.32, fw * 0.28, 1.04, 0.92, intensity * 0.55)

    return out

def _apply_child_skin_finish(img: np.ndarray, landmarks: np.ndarray, intensity: float) -> np.ndarray:
    """High-end app style skin finish: smooth color, preserve key edges."""
    H, W = img.shape[:2]
    face_m = _face_mask(landmarks, H, W, feather=24.0)

    smooth = img.copy()
    passes = 1 + int(intensity > 0.65)
    for _ in range(passes):
        smooth = cv2.bilateralFilter(smooth, 11, 58.0 + 36.0 * intensity, 14.0 + 10.0 * intensity)

    try:
        smooth = cv2.edgePreservingFilter(smooth, flags=1, sigma_s=42, sigma_r=0.18)
    except cv2.error:
        pass

    base = cv2.GaussianBlur(smooth, (0, 0), 0.65 + 0.45 * intensity)
    detail = img.astype(np.float32) - cv2.GaussianBlur(img, (0, 0), 1.2).astype(np.float32)
    finished = base.astype(np.float32) + detail * (0.42 - 0.12 * intensity)
    return _blend_mask(img, np.clip(finished, 0, 255).astype(np.uint8), face_m, min(0.72, 0.42 + intensity * 0.28))

def _apply_clear_complexion(img: np.ndarray, landmarks: np.ndarray, intensity: float) -> np.ndarray:
    """Even blotchy skin and suppress small blemishes without flattening features."""
    H, W = img.shape[:2]
    face_m = _face_mask(landmarks, H, W, feather=22.0)

    protected = np.zeros((H, W), dtype=np.float32)
    for idx in (
        [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
        [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466],
        [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 0, 13, 14],
    ):
        pts = _safe_pts(landmarks, idx)
        if pts is not None and len(pts) >= 3:
            local = np.zeros((H, W), dtype=np.uint8)
            cv2.fillConvexPoly(local, cv2.convexHull(pts.astype(np.int32)), 255)
            protected = np.maximum(
                protected,
                cv2.GaussianBlur(local.astype(np.float32), (0, 0), 7.0) / 255.0,
            )
    skin_m = np.clip(face_m * (1.0 - protected * 0.82), 0.0, 1.0)

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    l_chan, a_chan, b_chan = cv2.split(lab)
    even_l = cv2.bilateralFilter(l_chan, 7, 22.0 + 22.0 * intensity, 14.0)
    even_a = cv2.GaussianBlur(a_chan, (0, 0), 2.8 + 2.0 * intensity)
    even_b = cv2.GaussianBlur(b_chan, (0, 0), 2.8 + 2.0 * intensity)

    # Reduce local red/yellow blotches by pulling chroma gently toward the local average.
    lab_even = cv2.merge([
        np.clip(even_l * (1.0 + 0.018 * intensity) + 1.5 * intensity, 0, 255),
        np.clip(a_chan * (1.0 - 0.22 * intensity) + even_a * (0.22 * intensity), 0, 255),
        np.clip(b_chan * (1.0 - 0.16 * intensity) + even_b * (0.16 * intensity), 0, 255),
    ])
    even = cv2.cvtColor(lab_even.astype(np.uint8), cv2.COLOR_LAB2BGR)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    low = cv2.GaussianBlur(gray, (0, 0), 5.0)
    spots = np.clip((low - gray - 3.0) / 22.0, 0.0, 1.0) * skin_m
    spots = cv2.GaussianBlur(spots, (0, 0), 1.6)
    spot_fixed = even.astype(np.float32)
    spot_fixed += spots[:, :, np.newaxis] * (18.0 + 14.0 * intensity)

    cleared = _blend_mask(img, np.clip(spot_fixed, 0, 255).astype(np.uint8), skin_m, 0.44 + 0.30 * intensity)

    # Put back a little true edge detail so clear skin does not become plastic.
    detail = img.astype(np.float32) - cv2.GaussianBlur(img, (0, 0), 1.0).astype(np.float32)
    restored = cleared.astype(np.float32) + detail * face_m[:, :, np.newaxis] * (0.18 + 0.08 * (1.0 - intensity))
    return np.clip(restored, 0, 255).astype(np.uint8)

def _apply_fine_line_reducer(img: np.ndarray, landmarks: np.ndarray, intensity: float) -> np.ndarray:
    """Lift thin wrinkle lines locally while preserving pores and facial edges."""
    H, W = img.shape[:2]
    line_mask = np.zeros((H, W), dtype=np.float32)

    regions = [
        _safe_pts(landmarks, [10, 338, 297, 332, 284, 251, 21, 54, 103, 67, 109]),
        _safe_pts(landmarks, [33, 133, 157, 158, 159, 160, 161, 246]),
        _safe_pts(landmarks, [263, 362, 384, 385, 386, 387, 388, 466]),
        _safe_pts(landmarks, [186, 92, 165, 167, 205, 50]),
        _safe_pts(landmarks, [410, 322, 391, 393, 425, 280]),
    ]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 15))
    dark_h = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel_h).astype(np.float32)
    dark_v = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel_v).astype(np.float32)
    dark_lines = np.maximum(dark_h, dark_v)
    dark_lines = cv2.GaussianBlur(dark_lines, (0, 0), 0.9) / 255.0

    for pts in regions:
        if pts is None or len(pts) < 3:
            continue
        x, y, w, h = cv2.boundingRect(pts.astype(np.int32))
        pad = max(12, int(max(w, h) * 0.35))
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2, y2 = min(W, x + w + pad), min(H, y + h + pad)
        local_pts = (pts - [x1, y1]).astype(np.int32)
        local = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
        cv2.fillConvexPoly(local, cv2.convexHull(local_pts), 255)
        local = cv2.GaussianBlur(local.astype(np.float32), (0, 0), 8.0) / 255.0
        line_mask[y1:y2, x1:x2] = np.maximum(line_mask[y1:y2, x1:x2], local)

    line_mask = np.clip(line_mask * dark_lines * (2.2 + intensity), 0.0, 1.0)
    line_mask = cv2.GaussianBlur(line_mask, (0, 0), 0.8)

    lifted = img.astype(np.float32)
    lifted += line_mask[:, :, np.newaxis] * (22.0 + 24.0 * intensity)
    return _blend_mask(img, np.clip(lifted, 0, 255).astype(np.uint8), line_mask, 0.45 + 0.35 * intensity)

def _preserve_natural_detail(
    original: np.ndarray,
    processed: np.ndarray,
    landmarks: np.ndarray,
    intensity: float,
) -> np.ndarray:
    """Reinject natural pores/edges and cap extreme pixel shifts for identity."""
    H, W = original.shape[:2]
    face_m = _face_mask(landmarks, H, W, feather=18.0)
    orig_f = original.astype(np.float32)
    proc_f = processed.astype(np.float32)

    orig_detail = orig_f - cv2.GaussianBlur(orig_f, (0, 0), 1.05)
    proc_detail = proc_f - cv2.GaussianBlur(proc_f, (0, 0), 1.05)
    detail_restore = orig_detail * 0.28 + proc_detail * 0.18
    sharp = proc_f + detail_restore * face_m[:, :, np.newaxis] * (0.55 + 0.20 * (1.0 - intensity))

    diff = sharp - orig_f
    limit = 34.0 + 24.0 * intensity
    mag = np.linalg.norm(diff, axis=2)
    scale = np.minimum(1.0, limit / (mag + 1e-6))
    constrained = orig_f + diff * scale[:, :, np.newaxis]
    return np.clip(constrained, 0, 255).astype(np.uint8)

def _apply_child_color_grade(img: np.ndarray, landmarks: np.ndarray, intensity: float) -> np.ndarray:
    H, W = img.shape[:2]
    face_m = _face_mask(landmarks, H, W, feather=22.0)

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab[:, :, 0] = np.clip(lab[:, :, 0] * (1.0 + 0.055 * intensity) + 2.0 * intensity, 0, 255)
    lab[:, :, 1] = np.clip(lab[:, :, 1] + 4.5 * intensity, 0, 255)
    lab[:, :, 2] = np.clip(lab[:, :, 2] + 2.5 * intensity, 0, 255)
    warm = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)

    hsv = cv2.cvtColor(warm, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (1.0 + 0.10 * intensity), 0, 255)
    warm = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
    return _blend_mask(img, warm, face_m, 1.0)

def _apply_soft_child_mouth(img: np.ndarray, landmarks: np.ndarray, intensity: float) -> np.ndarray:
    pts = _safe_pts(landmarks, [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 0, 13, 14])
    if pts is None:
        return img
    H, W = img.shape[:2]
    cx, cy = float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1]))
    rx = max(18.0, float(np.ptp(pts[:, 0])) * 0.72)
    ry = max(8.0, float(np.ptp(pts[:, 1])) * 1.20)
    out = _local_scale(img, (cx, cy), rx, ry, 1.03 + 0.05 * intensity, 1.06, intensity * 0.55)

    mask = _soft_ellipse_mask((H, W), (cx, cy), rx, ry, feather=7.0)
    hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (1.0 + mask * 0.10 * intensity), 0, 255)
    lip = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
    return _blend_mask(out, lip, mask, intensity * 0.45)


# ── 4. DE-AGING — WRINKLE / HAIR / LIFT HELPERS ─────────────────────────────

def _apply_wrinkle_smooth(img: np.ndarray, landmarks: np.ndarray, intensity: float) -> np.ndarray:
    """Bilateral-filter the same zones that aging adds wrinkles to."""
    H, W = img.shape[:2]
    result = img.astype(np.float32)

    regions = [
        landmarks[[10, 338, 297, 332, 284, 251, 21, 54, 103, 67, 109]],  # forehead
        landmarks[[33, 133, 157, 158, 159, 160, 161, 246]],               # left eye
        landmarks[[263, 362, 384, 385, 386, 387, 388, 466]],              # right eye
        landmarks[[186, 92, 165, 167]],                                    # nasolabial left
        landmarks[[410, 322, 391, 393]],                                   # nasolabial right
    ]

    sigma_color = 30.0 + intensity * 60.0
    sigma_space = 12.0 + intensity * 18.0

    for pts in regions:
        rect = cv2.boundingRect(pts[:, :2].astype(np.int32))
        x, y, w, h = rect
        if w < 5 or h < 5:
            continue

        pad = 18
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2, y2 = min(W, x + w + pad), min(H, y + h + pad)

        roi      = img[y1:y2, x1:x2]
        smoothed = cv2.bilateralFilter(roi, 13, sigma_color, sigma_space)

        local_pts = (pts[:, :2] - [x1, y1]).astype(np.int32)
        hull      = cv2.convexHull(local_pts)
        mask_u8   = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
        cv2.fillConvexPoly(mask_u8, hull, 255)
        mask_f = cv2.GaussianBlur(mask_u8.astype(np.float32), (19, 19), 0) / 255.0
        m3d    = (mask_f * intensity)[:, :, np.newaxis]

        result[y1:y2, x1:x2] = (m3d * smoothed.astype(np.float32)
                                  + (1.0 - m3d) * result[y1:y2, x1:x2])

    return np.clip(result, 0, 255).astype(np.uint8)


def _apply_undereye_brighten(img: np.ndarray, landmarks: np.ndarray, intensity: float) -> np.ndarray:
    """Lighten under-eye triangles to reduce dark circles."""
    H, W = img.shape[:2]
    result = img.astype(np.float32)

    # Lower-lid landmark rings for each eye
    left_lower  = landmarks[[33, 7, 163, 144, 145, 153, 154, 155, 133]]
    right_lower = landmarks[[263, 249, 390, 373, 374, 380, 381, 382, 362]]

    for pts in (left_lower, right_lower):
        rect = cv2.boundingRect(pts[:, :2].astype(np.int32))
        x, y, w, h = rect
        if w < 4 or h < 4:
            continue

        local_pts = (pts[:, :2] - [x, y]).astype(np.int32)
        hull      = cv2.convexHull(local_pts)
        mask_u8   = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(mask_u8, hull, 255)
        mask_f = cv2.GaussianBlur(mask_u8.astype(np.float32), (11, 11), 0) / 255.0
        m3d    = (mask_f * intensity * 0.65)[:, :, np.newaxis]

        roi        = result[y:y+h, x:x+w]
        brightened = roi.astype(np.float32)
        brightened[:, :, 0] = np.clip(brightened[:, :, 0] + 10.0 * intensity, 0, 255)
        brightened[:, :, 1] = np.clip(brightened[:, :, 1] + 24.0 * intensity, 0, 255)
        brightened[:, :, 2] = np.clip(brightened[:, :, 2] + 30.0 * intensity, 0, 255)
        result[y:y+h, x:x+w] = m3d * brightened + (1.0 - m3d) * roi

    return np.clip(result, 0, 255).astype(np.uint8)


def _apply_skin_vibrance(img: np.ndarray, landmarks: np.ndarray, intensity: float) -> np.ndarray:
    """Boost saturation inside face region — younger skin reads as more vivid."""
    face_m = _face_mask(landmarks, img.shape[0], img.shape[1], feather=20.0)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (1.0 + intensity * 0.35), 0, 255)
    vibrant = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8),
                           cv2.COLOR_HSV2BGR).astype(np.float32)

    m3d    = (face_m * intensity)[:, :, np.newaxis]
    result = m3d * vibrant + (1.0 - m3d) * img.astype(np.float32)
    return np.clip(result, 0, 255).astype(np.uint8)


def _apply_hair_darkening(img: np.ndarray, landmarks: np.ndarray, intensity: float) -> np.ndarray:
    """Enrich hair colour — reverse of graying (deepen + re-saturate)."""
    H, W = img.shape[:2]

    forehead_idx = [10, 338, 297, 332, 284, 251, 21, 54, 103, 67, 109]
    hairline_y   = float(np.percentile(landmarks[forehead_idx, 1], 10))
    fade_zone    = max(20.0, hairline_y * 0.20)

    yy       = np.arange(H, dtype=np.float32)[:, np.newaxis]
    y_weight = np.clip((hairline_y + fade_zone - yy) / fade_zone, 0.0, 1.0)
    y_weight = np.broadcast_to(y_weight, (H, W)).copy()

    hsv        = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    brightness = hsv[:, :, 2]
    hair_wt    = np.clip((160.0 - brightness) / 120.0, 0.0, 1.0)

    hair_mask = y_weight * hair_wt
    hair_mask = cv2.GaussianBlur(hair_mask, (21, 21), 0)
    hair_mask = np.clip(hair_mask * 2.0, 0.0, 1.0)

    # Deepen (darken slightly) and re-saturate
    hsv_mod = hsv.copy()
    hsv_mod[:, :, 1] = np.clip(hsv[:, :, 1] * (1.0 + intensity * 0.40), 0, 255)
    hsv_mod[:, :, 2] = np.clip(hsv[:, :, 2] * (1.0 - intensity * 0.22), 0, 255)
    enriched = cv2.cvtColor(np.clip(hsv_mod, 0, 255).astype(np.uint8),
                            cv2.COLOR_HSV2BGR).astype(np.float32)

    m3d    = (hair_mask * intensity)[:, :, np.newaxis]
    result = m3d * enriched + (1.0 - m3d) * img.astype(np.float32)
    return np.clip(result, 0, 255).astype(np.uint8)


# ── 5. DE-AGING — FULL PIPELINE ──────────────────────────────────────────────

def apply_deaging(img: np.ndarray, sigma: float = 20.0, landmarks: np.ndarray | None = None) -> np.ndarray:
    H, W = img.shape[:2]
    norm_val = float(np.clip(sigma / 100.0, 0.0, 1.0))

    # IMPORTANT: avoid "whole-image blur".
    # Any global low-pass must be face-masked (background untouched) and kept subtle.
    temp = img.copy()

    if landmarks is not None:
        original_ref = img.copy()
        face_m = _face_mask(landmarks, H, W, feather=22.0)

        # 1) Subtle low-pass ONLY inside the face (reduces micro-contrast/wrinkle noise),
        # keeping blend low so pores/edges aren't wiped out.
        freq_sigma = max(18.0, 44.0 - (sigma * 0.18))
        blend      = float(np.clip(0.06 + norm_val * 0.22, 0.06, 0.30))
        lp_f = _gauss_lp((H, W), freq_sigma)
        lp_img = np.zeros_like(img, dtype=np.float64)
        for c in range(3):
            ch = img[:, :, c].astype(np.float64)
            lp_img[:, :, c] = _fft_filter(ch, lp_f)
        lp_img_u8 = np.clip(lp_img, 0, 255).astype(np.uint8)
        temp = _blend_mask(img, lp_img_u8, face_m, blend)

        # 2. Very subtle youthful geometry while preserving the real face shape.
        temp = _apply_child_face_geometry(temp, landmarks, norm_val * 0.38)

        # 3. App-style child skin finish: strong smoothing with edge retention.
        temp = _apply_child_skin_finish(temp, landmarks, norm_val)
        try:
            temp = _apply_clear_complexion(temp, landmarks, norm_val)
        except MemoryError:
            pass
        temp = _apply_baby_smooth(temp, landmarks, norm_val * 0.48)

        # 4. Targeted bilateral smoothing on wrinkle zones.
        if len(landmarks) > 466:
            temp = _apply_wrinkle_smooth(temp, landmarks, norm_val * 0.70)
        try:
            temp = _apply_fine_line_reducer(temp, landmarks, norm_val)
        except MemoryError:
            pass

        # 5. Under-eye brightening.
        temp = _apply_undereye_brighten(temp, landmarks, norm_val)

        # 6. Geometric lifting - pull face content upward.
        face_m = _face_mask(landmarks, H, W, feather=15.0)
        gy, gx = np.mgrid[0:H, 0:W].astype(np.float32)
        dy     = (norm_val * 8.0 * face_m).astype(np.float32)
        map_y  = np.clip(gy + dy, 0.0, H - 1.0)
        lifted = cv2.remap(temp, gx, map_y, cv2.INTER_CUBIC)

        m3d  = face_m[:, :, np.newaxis]
        temp = np.clip(m3d * lifted.astype(np.float64)
                       + (1.0 - m3d) * temp.astype(np.float64),
                       0, 255).astype(np.uint8)

        # 7. Warm childlike color and skin vibrance.
        temp = _apply_child_color_grade(temp, landmarks, norm_val)
        temp = _apply_skin_vibrance(temp, landmarks, norm_val)

        # 8. Hair darkening (reverse of graying).
        temp = _apply_hair_darkening(temp, landmarks, norm_val)

        # 9. Subtle face glow.
        glow_f = temp.astype(np.float32) * (1.0 + norm_val * 0.06)
        temp   = np.clip(m3d * glow_f + (1.0 - m3d) * temp.astype(np.float32),
                         0, 255).astype(np.uint8)

        # 10. Rosy cheeks and softer mouth color.
        temp = _apply_rosy_cheeks(temp, landmarks, norm_val)
        temp = _apply_soft_child_mouth(temp, landmarks, norm_val)

        # 11. Bright wide eyes - childlike wide-eyed appearance.
        temp = _apply_bright_eyes(temp, landmarks, norm_val * 0.70)
        try:
            temp = _preserve_natural_detail(original_ref, temp, landmarks, norm_val)
        except MemoryError:
            pass
        return temp

    # Fallback when landmarks are missing: keep it gentle and edge-preserving (no FFT blur).
    try:
        temp = cv2.edgePreservingFilter(
            temp,
            flags=1,
            sigma_s=int(34 + 26 * norm_val),
            sigma_r=float(0.14 + 0.10 * norm_val),
        )
    except cv2.error:
        pass
    temp = cv2.bilateralFilter(temp, 9, 40.0 + 35.0 * norm_val, 12.0 + 10.0 * norm_val)
    return np.clip(temp, 0, 255).astype(np.uint8)
