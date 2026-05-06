import numpy as np

# ── Anatomical Control-Point Subsets (Sadece Hedef Bölgeler) ─────────────

# Sadece dudak çevresi ve gülümseme hatları
SMILE_CTRL = [61, 291, 57, 287, 38, 268, 84, 314, 91, 321]

# Sadece sol ve sağ kaşlar
BROW_L_CTRL = [70, 105, 107, 66, 63, 55]
BROW_R_CTRL = [300, 334, 336, 296, 293, 285]
BROW_CTRL   = BROW_L_CTRL + BROW_R_CTRL

# 🔥 FIX: Sadece köşeler bırakıldı (çene etkisini keser)
# 🔥 YENİ FIX: Tüm dış ve iç dudak noktaları eklendi
LIP_CTRL = [
    # Dış dudak
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185,
    # İç dudak
    78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191
]

# Sadece yüz konturu (çene ve yanak sınırları)
SLIM_CTRL = [234, 93, 132, 58, 172, 136, 149, 150, 176,
             454, 323, 361, 288, 397, 365, 379, 378, 400]

_CAP = 40.0


# ── Helpers (Çalışan Matematik Korundu) ───────────────────────────────────

def _cap_displacements(delta: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(delta, axis=1, keepdims=True)
    s = np.where(n > _CAP, _CAP / (n + 1e-9), 1.0)
    return delta * s

def _nonlinear_scale(x: float) -> float:
    return float(np.tanh(2.0 * x) / np.tanh(2.0))

def _guard_top(y: float, amount: float, margin: float = 6.0) -> float:
    return min(amount, max(0.0, y - margin))


# ── Bölgesel İşlemler (Sadece İlgili Bölgeyi Etkiler) ──────────────────────

# ── Smile ──────────────────────────────────────────────────────────

def _smile_delta(lm: np.ndarray, intensity: float) -> np.ndarray:
    delta = np.zeros_like(lm, dtype=np.float32)
    
    # Ağız merkezi
    cx = (lm[61, 0] + lm[291, 0]) * 0.5
    
    # Ağzın genişliğini referans alıyoruz (kişiden kişiye değişebilir)
    mouth_width = abs(lm[291, 0] - lm[61, 0]) + 1e-6

    # Güç dengesi: Yatayda biraz daha toleranslı, dikeyde kontrollü kalkış
    dh = 512.0 * 0.025 * intensity  
    dv = 512.0 * 0.045 * intensity  

    for idx in SMILE_CTRL:
        if idx >= len(lm):
            continue

        # Noktanın merkeze olan yatay mesafesi
        dx = lm[idx, 0] - cx
        sign = np.sign(dx) if abs(dx) > 0.5 else 0.0

        # Nokta merkezden ne kadar uzak? (0 ile 1 arası bir oran çıkarıyoruz)
        # 0.55 değeri köşelerin biraz dışına kadar pürüzsüz geçiş sağlar.
        ratio = abs(dx) / (mouth_width * 0.55)
        ratio = float(np.clip(ratio, 0.0, 1.0))
        
        # Parabolik güç eğrisi (ratio ** 1.5):
        # Bu sayede köşeler tam güç (1.0) alırken, dudağın ortasına doğru 
        # güç aniden değil, kavisli ve doğal bir şekilde sıfıra iner.
        power = ratio ** 1.5 

        delta[idx, 0] = sign * dh * power
        delta[idx, 1] = -_guard_top(lm[idx, 1], dv * power)

    return _cap_displacements(delta)
def _eyebrow_delta(lm: np.ndarray, intensity: float) -> np.ndarray:
    delta = np.zeros_like(lm, dtype=np.float32)

    # 🔥 FIX: gözleri bozmayacak seviye
    disp = 512.0 * 0.040 * intensity

    for idx in BROW_CTRL:
        if idx >= len(lm):
            continue
        delta[idx, 1] = -_guard_top(lm[idx, 1], disp)

    return _cap_displacements(delta)


# ── Lip Wide ──────────────────────────────────────────────────────────

def _lip_wide_delta(lm: np.ndarray, intensity: float) -> np.ndarray:
    delta = np.zeros_like(lm, dtype=np.float32)

    # Ağız merkezi ve genişliği
    cx = (lm[61, 0] + lm[291, 0]) * 0.5
    mouth_width = abs(lm[291, 0] - lm[61, 0]) + 1e-6

    # Gücü biraz kıstık çünkü artık tüm dudak noktaları hareket ediyor
    disp = 512.0 * 0.035 * intensity

    for idx in LIP_CTRL:
        if idx >= len(lm):
            continue

        # Noktanın merkeze olan yatay mesafesi
        dx = lm[idx, 0] - cx
        sign = np.sign(dx) if abs(dx) > 0.5 else 0.0

        # Nokta merkezden ne kadar uzak? (0: tam merkez, 1: tam köşe)
        # Yarı genişliğe bölüyoruz ki oran 0 ile 1 arasında çıksın.
        ratio = abs(dx) / (mouth_width * 0.5)
        ratio = float(np.clip(ratio, 0.0, 1.0))

        # Merkez noktaları (örn. dudağın tam ortası) 0 ile çarpılıp sabit kalacak,
        # Köşelere doğru gittikçe esneme katsayısı artacak.
        delta[idx, 0] = sign * disp * ratio

    return _cap_displacements(delta)


def _face_slim_delta(lm: np.ndarray, intensity: float) -> np.ndarray:
    delta = np.zeros_like(lm, dtype=np.float32)

    # 1. Hareket edecek noktalar (Çene hattı)
    oval_pts = np.array([lm[i] for i in SLIM_CTRL if i < len(lm)], dtype=np.float64)
    if len(oval_pts) == 0: return delta

    cx = float(np.mean(oval_pts[:, 0]))
    cy = float(np.mean(oval_pts[:, 1]))
    vy = float(np.ptp(oval_pts[:, 1])) + 1e-6

    # 2. 🔥 Sabit Çapalar (Anchors) - Bu noktalar TPS'in yayılmasını durdurur
    # Kulak önleri, elmacık üstü ve alın kenarları
    ANCHOR_PTS = [127, 234, 93, 356, 454, 323, 10, 152] 

    disp = 512.0 * 0.045 * intensity

    # Çene hattını içeri çek
    for idx in SLIM_CTRL:
        if idx >= len(lm): continue
        dx = lm[idx, 0] - cx
        dy = lm[idx, 1] - cy
        vert_supp = 1.0 - float(np.clip(abs(dy) / (vy * 0.55), 0, 0.65))
        
        # Sadece X ekseninde merkeze doğru çekim
        delta[idx, 0] = -np.sign(dx) * disp * vert_supp if abs(dx) > 0.5 else 0.0

    # 3. 🔥 Çapa noktalarına 0.0 hareketi zorla (TPS duvarı örüyoruz)
    for idx in ANCHOR_PTS:
        if idx < len(lm):
            delta[idx] = [0.0, 0.0]

    return _cap_displacements(delta)

# ── Public API ────────────────────────────────────────────────────────────

def compute_combined_displacement(
    landmarks: np.ndarray,
    smile: float = 0.0,
    eyebrow: float = 0.0,
    lip_wide: float = 0.0,
    face_slim: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    
    # İşlem yapılmayan noktalar (gözler, burun vb.) bu 0'lar sayesinde
    # sabit kalacak (anchor olacak) ve warp sırasında yüzü germeyecek.
    delta = np.zeros_like(landmarks, dtype=np.float32)

    # Parametre 0 ise o bölgenin fonksiyonunu hiç yormuyoruz
    if abs(smile) > 1e-3:
        delta += _smile_delta(landmarks, _nonlinear_scale(smile))

    if abs(eyebrow) > 1e-3:
        delta += _eyebrow_delta(landmarks, _nonlinear_scale(eyebrow))

    if abs(lip_wide) > 1e-3:
        delta += _lip_wide_delta(landmarks, _nonlinear_scale(lip_wide))

    if abs(face_slim) > 1e-3:
        delta += _face_slim_delta(landmarks, _nonlinear_scale(face_slim))

    # 🔥 FIX: global stabilite
    delta = _cap_displacements(delta)
    delta = np.clip(delta, -25.0, 25.0)

    # Tip dönüşümleri garantileniyor
    src = landmarks.copy().astype(np.float32)
    dst = np.clip(landmarks + delta, 4.0, 508.0).astype(np.float32)

    return src, dst