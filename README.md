# 👁️ VigilanceAI — Surveillance Comportementale + GPS Réel

Système de surveillance IA complet : détection comportementale, score de menace,
tracking GPS réel multi-sources, dashboard web et interface Gradio.

---

## 📁 Structure

```
vigilance_ai/
├── detector.py              ← Pipeline principal (webcam/vidéo + Flask)
├── gps_engine.py            ← Moteur GPS réel multi-sources
├── app.py                   ← Interface Gradio (Hugging Face Spaces)
├── requirements.txt         ← Dépendances HuggingFace
├── requirements_local.txt   ← Dépendances locales
├── .env.example             ← Template variables d'environnement
└── README.md
```

---

## ⚙️ Installation locale

```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Linux/Mac

pip install -r requirements_local.txt
```

---

## 🔑 APIs GPS gratuites

| API | Gratuit | Clé requise | Usage |
|-----|---------|-------------|-------|
| **ipinfo.io** | ✅ 50k/mois | ❌ Non | Position caméra par IP |
| **OSM Nominatim** | ✅ Illimité | ❌ Non | Adresse depuis GPS |
| **OpenCage** | ✅ 2500/jour | ✅ Oui | Geocoding précis |
| **What3Words** | ✅ Gratuit | ✅ Oui | Adresse en 3 mots |

### Obtenir les clés gratuites :
- **OpenCage** → https://opencagedata.com/ (inscription gratuite)
- **What3Words** → https://developer.what3words.com/ (inscription gratuite)

### Configurer les clés :
```bash
# Copier le template
cp .env.example .env
# Éditer .env et remplir les clés
```

Ou directement en argument :
```bash
python detector.py --opencage-key VOTRE_CLE --w3w-key VOTRE_CLE
```

---

## 🚀 Lancement

### Webcam (GPS auto-détecté par IP)
```bash
python detector.py
```

### Vidéo + GPS + clés API
```bash
python detector.py \
  --source video.mp4 \
  --opencage-key VOTRE_CLE \
  --w3w-key VOTRE_CLE \
  --save output.mp4
```

### Caméra fixe avec position GPS connue (plus précis)
```bash
python detector.py \
  --source rtsp://192.168.1.10:554/stream \
  --cam-lat 48.8566 \
  --cam-lon 2.3522 \
  --cam-alt 6.0 \
  --cam-fov-h 80.0 \
  --cam-tilt 50.0
```

### Test du GPS Engine seul
```bash
python gps_engine.py
```

---

## 🎮 Touches clavier

| Touche | Action |
|--------|--------|
| `q` | Quitter |
| `SPACE` | Pause / Reprise |
| `s` | Screenshot |
| `z` | Zone interdite (clic gauche=point, clic droit=fermer) |
| `d` | Dashboard Flask → http://localhost:5000 |
| `c` | Effacer zones |

---

## 🗺️ GPS Réel — Comment ça marche

### Sans calibration (auto)
1. **ipinfo.io** détecte la position GPS de la caméra via son IP
2. La position des personnes est calculée par **projection géométrique**
   en tenant compte du FOV, de l'altitude et de l'inclinaison
3. **OSM Nominatim** convertit les coords en adresse réelle

### Avec calibration BEV (précision maximale)
```python
from gps_engine import GPSEngine

gps = GPSEngine()
gps.calibrate_bev(
    pixel_points=[(100,200), (800,200), (800,700), (100,700)],
    gps_points=[(48.8567,2.3521), (48.8567,2.3525),
                (48.8563,2.3525), (48.8563,2.3521)]
)
```

### Export GeoJSON (QGIS / Google Earth)
À la fin de chaque session, un fichier `logs/gps_tracking.json` est généré.
Il est lisible dans :
- 🗺️ **QGIS** (import GeoJSON)
- 🌍 **Google Earth** (import KML/GeoJSON)
- 🌐 **geojson.io** (visualisation en ligne)

---

## 🎯 Fonctionnalités complètes

### Détection & Analyse
- ✅ Détection personnes YOLOv8
- ✅ Score de menace 0–100% par individu
- ✅ Analyse expression faciale (DeepFace)
- ✅ Analyse posturale (fuite, agitation, surveillance)
- ✅ Détection objets suspects (couteau, sacs...)
- ✅ Tracking trajectoires + trainées colorées

### Alertes
- ✅ Flash rouge si suspect détecté
- ✅ Alerte zone interdite
- ✅ Alerte surpopulation (foule)
- ✅ Identification véhicule de fuite

### GPS Réel
- ✅ Auto-localisation caméra (ipinfo.io)
- ✅ Coordonnées GPS par personne en temps réel
- ✅ Adresse réelle (OSM Nominatim, OpenCage)
- ✅ Adresse W3W (What3Words)
- ✅ Export GeoJSON (QGIS/Google Earth)
- ✅ Calibration BEV 4 points pour précision maximale

### Interfaces
- ✅ Fenêtre OpenCV locale
- ✅ Dashboard Flask (http://localhost:5000)
- ✅ Interface Gradio (Hugging Face)
- ✅ Logs CSV + JSON complets

---

## 🔧 Arguments complets

| Argument | Défaut | Description |
|----------|--------|-------------|
| `--source` | `0` | Webcam / fichier / RTSP |
| `--no-show` | off | Mode headless |
| `--save` | aucun | Vidéo annotée sortie |
| `--log-every` | `3` | Logger 1 frame sur N |
| `--opencage-key` | `` | Clé OpenCage |
| `--w3w-key` | `` | Clé What3Words |
| `--cam-lat` | auto | Latitude caméra |
| `--cam-lon` | auto | Longitude caméra |
| `--cam-alt` | `8.0` | Altitude caméra (m) |
| `--cam-fov-h` | `70.0` | FOV horizontal (°) |
| `--cam-fov-v` | `45.0` | FOV vertical (°) |
| `--cam-tilt` | `45.0` | Inclinaison (°) |

---

## 🌐 Déploiement Hugging Face

Uploader sur HuggingFace Spaces :
- `app.py`
- `requirements.txt`
- `README_HF.md` (renommer en README.md)

Les clés API se configurent dans **Settings → Repository secrets** :
- `OPENCAGE_KEY`
- `W3W_KEY`

---

*Fait par **Mayele** · YOLOv8 + OpenCV + GPS Réel*
