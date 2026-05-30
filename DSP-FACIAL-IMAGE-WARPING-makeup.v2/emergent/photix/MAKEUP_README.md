# Photix — Sanal Makyaj / Virtual Makeup

Yüz işaret noktalarına (MediaPipe Face Mesh, 478 nokta) dayalı, **poz kararlı**
sanal makyaj. Photix'in mevcut **masaüstü uygulamasına (`main.py`)**, **canlı
webcam döngüsüne (`start_live.py`)** ve **Gradio web arayüzüne (`ui.py`)**
entegre edilmiştir. Yüzünü çevirdiğinde makyaj kaymaz — her karede landmark'tan
yeniden çizilir.

> Landmark-driven, **pose-stable** virtual makeup, integrated into the existing
> Photix Tk desktop app, the live webcam loop and the Gradio web UI. Works
> on photos, live camera and video files. Makeup tracks the face when you
> turn your head because every region is rebuilt from the current landmarks
> on every frame — nothing is baked into a static texture.

---

## Katmanlar / Layers

| Katman | Açıklama | Renkler |
|--------|----------|---------|
| **Göz farı** (eyeshadow) | Göz kapağında yumuşak renk bandı | mavi, siyah, kırmızı, yeşil, turuncu, kahverengi, **simli** |
| **Allık** (blush) | Yanaklarda diffuse renk | pembe, kahverengi |
| **Ruj** (lipstick) | Dudak boyama (+ parlaklık) | kırmızı, pembe, kahverengi, **parlak** |
| **Eyeliner** | Üst kirpik dibinde ince çizgi + kanat | siyah, kahverengi, mavi, yeşil |
| **Kirpik** (mascara) | Uzatılmış/koyulaştırılmış üst kirpikler | siyah, kahverengi |

Renk adlarını Türkçe de yazabilirsin (`mavi`, `kırmızı`, `kahverengi`, `simli`,
`parlak`, `pembe`). Her katmanın **rengi**, **yoğunluğu** ve **açık/kapalı**
durumu birbirinden bağımsız — istediğin kombinasyonu yapabilirsin.

---

## Nasıl çalıştırılır

### 1) Masaüstü GUI (en kolay yol)

```bash
pip install -r requirements.txt
python main.py
```

Sol kenar çubuğunda **MAKEUP** bölümü görünür. Her katmanın yanındaki kutuyu
işaretle, açılır menüden rengini seç, kaydırma çubuğuyla yoğunluğunu ayarla.
Fotoğraf yüklediğinde otomatik uygulanır. **LIVE WARPING** panelinden canlı
kamerayı açtığında aynı makyaj canlı görüntüye de gider.

### 2) Sadece canlı webcam (Tk yok)

```bash
python start_live.py            # boş başlar, tuşlarla makyaj ekle
python start_live_makeup.py     # tam makyajla başlar
```

Pencerede klavye:

```
e / E    göz farı toggle / sonraki renk
b / B    allık     toggle / sonraki renk
r / R    ruj       toggle / sonraki renk
n / N    eyeliner  toggle / sonraki renk
k / K    kirpik    toggle / sonraki renk

1-9      mevcut warp / aging / overlay efektleri
+ / -    yoğunluk           0    her şeyi sıfırla
s        anlık görüntü      l    landmark'ları göster
q / Esc  çıkış
```

### 3) Gradio web arayüzü

```bash
python ui.py
# Tarayıcıda: http://127.0.0.1:7860
```

Hem **Image** hem **Live / Video** sekmesinde "**Makyaj / Makeup**" başlığı
altında bir akordion var — açtığında 5 katman görünür. Webcam butonuna
bastığında seçtiğin makyaj canlı pencereye taşınır.

---

## Programatik kullanım

### Doğrudan motor

```python
import cv2
from core.landmark_detector import detect_landmarks
from effects.makeup import apply_makeup

img = cv2.imread("yuz.jpg")
lm  = detect_landmarks(img)

config = {
    "eyeshadow": {"enabled": True, "color": "blue",  "intensity": 0.6},
    "blush":     {"enabled": True, "color": "pink",  "intensity": 0.3},
    "lipstick":  {"enabled": True, "color": "red",   "intensity": 0.65},
    "eyeliner":  {"enabled": True, "color": "black", "intensity": 0.8},
    "mascara":   {"enabled": True, "color": "black", "intensity": 0.75},
}
cv2.imwrite("cikti.png", apply_makeup(img, lm, config).final_image)
```

### LivePipeline ile (foto/canlı/video aynı kod yolu)

```python
from live.pipeline import LivePipeline, apply_pipeline
from live.video import process_video
from pathlib import Path

pipe = LivePipeline(
    makeup_eyeshadow_enabled=True, makeup_eyeshadow_color="blue",
    makeup_blush_enabled=True,
    makeup_lipstick_enabled=True, makeup_lipstick_color="gloss",
    makeup_eyeliner_enabled=True,
    makeup_mascara_enabled=True,
)

out = apply_pipeline(frame_bgr, pipe, landmarks)              # tek kare
out_path = process_video(Path("klip.mp4"), pipeline=pipe)     # tüm video
```

`LivePipeline` JSON serialize edilebilir, böylece `main.py` → webcam
subprocess handoff'u ve Gradio state ile sorunsuz çalışır.

### FaceOverlayEngine kayıt sistemi üzerinden

Makyaj, mevcut overlay motoruna da `"makeup"` adıyla kayıtlıdır:

```python
from effects import FaceOverlayEngine
engine = FaceOverlayEngine(landmarks, image)
out = engine.apply(["makeup"], {"makeup": {"config": config}}).final_image
```

---

## Nasıl çalışır (DSP)

1. **İşaret noktaları** — MediaPipe Face Landmarker, 478 nokta. Statik foto
   için `IMAGE` modu (`core/landmark_detector.py`), canlı/video için `VIDEO`
   modu (`live/landmark_stream.py`) ile zamansal tracking.
2. **Bölge maskeleri** — göz kapağı (lash line + crease), dudak halkası
   (outer − inner), yanak elması ve kirpik hattı her karede landmark'lardan
   poligon/elips olarak kurulur, Gauss ile yumuşatılır.
3. **Aydınlık-koruyan tonlama** — renk yerel parlaklığa göre gölgelenir
   (`shaded = color · ((1−k) + k·luma)`); tenin dokusu/gölgeleri korunur,
   sonuç boya değil pigment gibi okur.
4. **Poz kararlılığı** — hiçbir şey statik dokuya pişirilmez. Yüz dönünce
   landmark'lar değişir, maskeler yeniden doğar → makyaj kaymaz.
5. **Özel efektler** — simli farda parıltı (rastgele parlak benekler),
   parlak (gloss) rujda alt dudakta speküler vurgu, eyeliner kanadı,
   kirpik yelpazesi.

---

## Yapılan değişiklikler

```
effects/makeup.py            (yeni)  Makyaj motoru — tüm katmanlar + renkler
effects/__init__.py          guncellendi  apply_makeup, COLOR_CHOICES export
effects/overlays.py          guncellendi  'makeup' efekti FaceOverlayEngine'e kayitli
live/pipeline.py             guncellendi  LivePipeline + 15 makyaj alani
                                          + has_makeup() + makeup_config()
                                          + apply_pipeline'da 5. asama
live/controls.py             guncellendi  Klavye kisayollari (e/E b/B r/R n/N k/K)
main.py                      guncellendi  Tk vars + sidebar MAKEUP paneli
                                          + apply_processing'e makyaj
                                          + _build_live_pipeline'da makyaj
                                          + reset() makyajı temizler
ui.py                        guncellendi  Image & Live/Video sekmelerinde
                                          makyaj akordionu + handler argumanlari
start_live_makeup.py         (yeni)  Tam makyajla baslayan webcam launcher
MAKEUP_README.md             (bu)    Belge
```
