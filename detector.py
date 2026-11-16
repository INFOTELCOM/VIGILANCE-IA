"""
VigilanceAI — Système de Surveillance Comportementale
======================================================
Fonctionnalités :
  ✅ Détection personnes YOLOv8
  ✅ Analyse expression faciale (DeepFace)
  ✅ Analyse posture & comportement
  ✅ Score de menace par personne (0-100%)
  ✅ Tracking trajectoires + détection de fuite
  ✅ Retraçage chemin de fuite + véhicule de fuite
  ✅ Analyse de foule (densité, suspects, neutres)
  ✅ Détection objets suspects (sacs, armes...)
  ✅ Alertes temps réel + logs CSV/JSON
  ✅ Dashboard Flask + Interface Gradio
  ✅ Coordonnées GPS simulées (extensible satellite)

Auteur : Mayele
Touches :
  q        → Quitter
  SPACE    → Pause
  s        → Screenshot
  z        → Zone interdite
  d        → Dashboard Flask :5000
  c        → Effacer zones
"""

import cv2
import numpy as np
import time
import os
import csv
import json
import threading
from collections import defaultdict, deque
from datetime import datetime
from ultralytics import YOLO

# ══════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════
PERSON_CLASS    = 0
VEHICLE_CLASSES = {2:"Car", 3:"Motorcycle", 5:"Bus", 7:"Truck"}
OBJECT_CLASSES  = {
    24:"Backpack", 26:"Handbag", 28:"Suitcase",
    39:"Bottle",   43:"Knife",   76:"Scissors",
}
SUSPICIOUS_OBJECTS = {43, 76}   # couteau, ciseaux → toujours suspect

CONF_THRESHOLD  = 0.40
TRAIL_LEN       = 50
MAX_HISTORY     = 30
CROWD_ALERT_N   = 10    # alerte si + de N personnes dans la zone

# Seuils score menace
SCORE_GENTIL    = 35
SCORE_SUSPECT   = 65

# Couleurs BGR
C_GREEN   = (0, 255, 100)
C_YELLOW  = (0, 200, 255)
C_RED     = (0, 0, 255)
C_ORANGE  = (0, 140, 255)
C_WHITE   = (240, 240, 240)
C_BLACK   = (10, 12, 16)
C_CYAN    = (255, 200, 0)
C_PURPLE  = (255, 0, 200)

TRAIL_PALETTE = [
    (0,255,100),(0,170,255),(255,200,0),(255,80,0),
    (200,0,255),(0,255,255),(255,0,150),(100,255,0),
    (0,100,255),(255,150,50),
]

# ══════════════════════════════════════════════════════
#  ÉTAT PARTAGÉ FLASK
# ══════════════════════════════════════════════════════
class SharedState:
    def __init__(self):
        self.lock         = threading.Lock()
        self.fps          = 0.0
        self.frame_num    = 0
        self.total        = 0
        self.suspects     = 0
        self.neutres      = 0
        self.gentils      = 0
        self.crowd_alert  = False
        self.alerts       = []
        self.persons      = []
        self.frame_jpg    = None
        self.fugitifs     = []
        self.vehicles_fuite = []
        self.session_start = datetime.now().isoformat()

    def snapshot(self):
        with self.lock:
            return {
                "fps": round(self.fps,1),
                "frame": self.frame_num,
                "session_start": self.session_start,
                "persons": {
                    "total":   self.total,
                    "suspects":self.suspects,
                    "neutres": self.neutres,
                    "gentils": self.gentils,
                },
                "crowd_alert":    self.crowd_alert,
                "alerts":         self.alerts[-20:],
                "persons_list":   self.persons[:30],
                "fugitifs":       self.fugitifs[-10:],
                "vehicles_fuite": self.vehicles_fuite[-5:],
            }

shared = SharedState()

# ══════════════════════════════════════════════════════
#  DASHBOARD FLASK
# ══════════════════════════════════════════════════════
def make_flask_app():
    try:
        from flask import Flask, Response, jsonify
    except ImportError:
        return None

    app = Flask(__name__)

    HTML = """<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<title>VigilanceAI Dashboard</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#080a0d;color:#e0e0e0;font-family:'Courier New',monospace;padding:14px}
h1{font-size:1.1rem;letter-spacing:3px;color:#ff4444;margin-bottom:14px;
   border-bottom:1px solid #2a0a0a;padding-bottom:8px;text-align:center}
.grid{display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px;margin-bottom:12px}
.card{background:#0e1015;border:1px solid #1a1a2a;border-radius:6px;padding:12px}
.card h2{font-size:0.6rem;letter-spacing:2px;color:#555;margin-bottom:10px;text-transform:uppercase}
.stat{display:flex;justify-content:space-between;margin-bottom:6px;font-size:0.78rem}
.val{font-weight:bold}
.green{color:#00ff88}.yellow{color:#ffcc00}.red{color:#ff4444}.orange{color:#ff8800}
.stream{width:100%;border:1px solid #2a1a1a;border-radius:6px;margin-bottom:12px}
table{width:100%;border-collapse:collapse;font-size:0.68rem}
th{color:#555;text-align:left;padding:3px 6px;border-bottom:1px solid #1a1a1a}
td{padding:3px 6px;border-bottom:1px solid #0d0d12}
.alert-row{color:#ff4444}.suspect-row{color:#ff8800}.gentil-row{color:#00ff88}
.crowd-banner{background:#3a0000;border:1px solid #ff4444;border-radius:4px;
  padding:8px;text-align:center;color:#ff4444;font-weight:bold;margin-bottom:10px;
  display:none}
.crowd-banner.active{display:block}
</style>
</head>
<body>
<h1>👁️ VIGILANCE AI — SURVEILLANCE COMPORTEMENTALE</h1>
<img class="stream" src="/video_feed" alt="Stream">
<div class="crowd-banner" id="crowdBanner">⚠ ALERTE FOULE — SURPOPULATION DÉTECTÉE</div>
<div id="data">Chargement...</div>
<script>
async function refresh(){
  try{
    const r=await fetch('/api/stats');
    const d=await r.json();
    document.getElementById('crowdBanner').className='crowd-banner'+(d.crowd_alert?' active':'');
    document.getElementById('data').innerHTML=renderDash(d);
  }catch(e){}
}
function threatColor(score){
  if(score>=65) return 'red';
  if(score>=35) return 'orange';
  return 'green';
}
function renderDash(d){
  const p=d.persons;
  return `
  <div class="grid">
    <div class="card">
      <h2>Personnes</h2>
      <div class="stat"><span>Total</span><span class="val green">${p.total}</span></div>
      <div class="stat"><span>Gentils</span><span class="val green">${p.gentils}</span></div>
      <div class="stat"><span>Neutres</span><span class="val yellow">${p.neutres}</span></div>
      <div class="stat"><span>Suspects</span><span class="val red">${p.suspects}</span></div>
    </div>
    <div class="card">
      <h2>Système</h2>
      <div class="stat"><span>FPS</span><span class="val green">${d.fps}</span></div>
      <div class="stat"><span>Frame</span><span class="val">${d.frame}</span></div>
      <div class="stat"><span>Fugitifs</span><span class="val red">${d.fugitifs.length}</span></div>
      <div class="stat"><span>Véh. fuite</span><span class="val orange">${d.vehicles_fuite.length}</span></div>
    </div>
    <div class="card">
      <h2>Alertes récentes</h2>
      ${d.alerts.slice(-4).reverse().map(a=>`
        <div class="stat" style="font-size:0.65rem">
          <span style="color:#888">${a.time?.slice(11,19)||''}</span>
          <span class="red">${a.msg}</span>
        </div>`).join('')}
    </div>
  </div>
  ${d.persons_list.length?`
  <div class="card" style="margin-bottom:10px">
    <h2>Individus détectés</h2>
    <table>
      <tr><th>ID</th><th>Score</th><th>Statut</th><th>Expression</th><th>Posture</th><th>Objet</th></tr>
      ${d.persons_list.map(p=>`
        <tr class="${p.score>=65?'alert-row':p.score>=35?'suspect-row':'gentil-row'}">
          <td>#${p.id}</td>
          <td>${p.score}%</td>
          <td>${p.statut}</td>
          <td>${p.expression||'—'}</td>
          <td>${p.posture||'—'}</td>
          <td>${p.objet||'—'}</td>
        </tr>`).join('')}
    </table>
  </div>`:''}
  ${d.fugitifs.length?`
  <div class="card" style="margin-bottom:10px;border-color:#ff4444">
    <h2 style="color:#ff4444">⚠ FUGITIFS TRACÉS</h2>
    <table>
      <tr><th>ID</th><th>Dernière position</th><th>Heure</th><th>GPS simulé</th></tr>
      ${d.fugitifs.map(f=>`
        <tr class="alert-row">
          <td>#${f.id}</td>
          <td>${f.last_pos}</td>
          <td>${f.time?.slice(11,19)||''}</td>
          <td>${f.gps}</td>
        </tr>`).join('')}
    </table>
  </div>`:''}
  ${d.vehicles_fuite.length?`
  <div class="card">
    <h2 style="color:#ff8800">🚗 VÉHICULES DE FUITE</h2>
    <table>
      <tr><th>ID</th><th>Type</th><th>Proximité fugitif</th><th>Heure</th></tr>
      ${d.vehicles_fuite.map(v=>`
        <tr class="suspect-row">
          <td>#${v.id}</td><td>${v.type}</td><td>${v.proximity}</td>
          <td>${v.time?.slice(11,19)||''}</td>
        </tr>`).join('')}
    </table>
  </div>`:''}`;
}
setInterval(refresh,1500);
refresh();
</script>
</body>
</html>"""

    @app.route('/')
    def index(): return HTML

    @app.route('/api/stats')
    def stats(): return jsonify(shared.snapshot())

    @app.route('/video_feed')
    def video_feed():
        def gen():
            while True:
                with shared.lock:
                    jpg = shared.frame_jpg
                if jpg:
                    yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpg + b'\r\n'
                time.sleep(0.05)
        return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

    return app

# ══════════════════════════════════════════════════════
#  LOGGER
# ══════════════════════════════════════════════════════
class Logger:
    def __init__(self, out_dir="logs"):
        import numpy as np

        os.makedirs(out_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path  = os.path.join(out_dir, f"vigilance_{ts}.csv")
        self.json_path = os.path.join(out_dir, f"vigilance_{ts}.json")
        self.records   = []

        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                "timestamp","frame","person_id","score","statut",
                "expression","posture","objet_suspect",
                "x1","y1","x2","y2","gps_lat","gps_lon","fugitif"
            ])

        print(f"[LOG] {self.csv_path}")

    # ─────────────────────────────
    # CLEAN SAFE VALUES
    # ─────────────────────────────
    def _clean(self, v):
        import numpy as np

        if isinstance(v, (np.bool_,)):
            return bool(v)
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (np.floating,)):
            return float(v)
        if isinstance(v, (tuple, list)):
            return [self._clean(x) for x in v]
        return v

    # ─────────────────────────────
    # LOG CSV + MEMORY
    # ─────────────────────────────
    def log(self, frame_num, pid, score, statut, expression,
            posture, objet, box, gps, fugitif=False):

        ts = datetime.now().isoformat(timespec="milliseconds")

        x1, y1, x2, y2 = box

        row = {
            "timestamp": ts,
            "frame": int(frame_num),
            "person_id": int(pid),
            "score": float(round(score, 1)),
            "statut": str(statut),
            "expression": str(expression),
            "posture": str(posture),
            "objet_suspect": str(objet),

            "x1": int(x1),
            "y1": int(y1),
            "x2": int(x2),
            "y2": int(y2),

            "gps_lat": float(round(gps[0], 6)),
            "gps_lon": float(round(gps[1], 6)),

            "fugitif": bool(fugitif)
        }

        self.records.append(row)

        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(list(row.values()))

    # ─────────────────────────────
    # JSON SAFE EXPORT
    # ─────────────────────────────
    def flush(self):
        safe_records = []

        for r in self.records:
            clean_row = {k: self._clean(v) for k, v in r.items()}
            safe_records.append(clean_row)

        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump(
                {"detections": safe_records},
                f,
                indent=2,
                ensure_ascii=False
            )
# ══════════════════════════════════════════════════════
#  GPS SIMULÉ (extensible satellite/API)
# ══════════════════════════════════════════════════════
class GPSSimulator:
    """
    Simule des coordonnées GPS à partir des coords pixels.
    En production : remplacer par une API GPS réelle ou
    intégration satellite (ex: OpenStreetMap + caméra IP géolocalisée).
    """
    def __init__(self, origin_lat=48.8566, origin_lon=2.3522,
                 scale_lat=0.00001, scale_lon=0.00001):
        self.lat0 = origin_lat   # Paris par défaut
        self.lon0 = origin_lon
        self.slat = scale_lat
        self.slon = scale_lon

    def pixel_to_gps(self, cx, cy):
        lat = self.lat0 + cy * self.slat
        lon = self.lon0 + cx * self.slon
        return round(lat, 6), round(lon, 6)

# ══════════════════════════════════════════════════════
#  ANALYSEUR DE COMPORTEMENT
# ══════════════════════════════════════════════════════
class BehaviorAnalyzer:
    """
    Analyse le comportement d'une personne à partir de :
    - Son historique de positions (vitesse, direction, erratisme)
    - Les objets détectés sur elle
    - (optionnel) expression faciale DeepFace

    Retourne un score de menace 0-100 et des labels.
    """

    def __init__(self):
        self.deepface_ok = self._check_deepface()
        if self.deepface_ok:
            print("[INFO] DeepFace disponible — analyse faciale activée")
        else:
            print("[INFO] DeepFace non installé — analyse posturale uniquement")

    def _check_deepface(self):
        try:
            import deepface
            return True
        except ImportError:
            return False

    def analyze_face(self, face_roi):
        """Analyse l'expression faciale via DeepFace."""
        if not self.deepface_ok or face_roi is None or face_roi.size == 0:
            return None, 0
        try:
            from deepface import DeepFace
            result = DeepFace.analyze(
                face_roi, actions=['emotion'],
                enforce_detection=False, silent=True
            )
            emotion  = result[0]['dominant_emotion']
            emotions = result[0]['emotion']
            # Score menace basé sur l'émotion
            threat_emotions = {
                'angry': 70, 'fear': 55, 'disgust': 50,
                'sad': 20,   'neutral': 10, 'happy': 0, 'surprise': 30
            }
            score = threat_emotions.get(emotion, 10)
            return emotion, score
        except Exception:
            return None, 0

    def analyze_posture(self, history, box):
        """
        Analyse le comportement postural :
        - Vitesse de déplacement (fuite ?)
        - Erratisme (mouvement imprévisible ?)
        - Stationnement suspect (reste immobile longtemps ?)
        """
        if len(history) < 3:
            return "inconnu", 0, False

        positions = [(h[0], h[1]) for h in history]
        times     = [h[2] for h in history]

        # Vitesse moyenne (pixels/s)
        speeds = []
        for i in range(1, len(positions)):
            dt = times[i] - times[i-1]
            if dt < 1e-4: continue
            dx = positions[i][0] - positions[i-1][0]
            dy = positions[i][1] - positions[i-1][1]
            speeds.append(np.sqrt(dx**2 + dy**2) / dt)

        avg_speed   = np.mean(speeds) if speeds else 0
        max_speed   = np.max(speeds)  if speeds else 0

        # Erratisme : variance de direction
        directions = []
        for i in range(1, len(positions)):
            dx = positions[i][0] - positions[i-1][0]
            dy = positions[i][1] - positions[i-1][1]
            if abs(dx)+abs(dy) > 2:
                directions.append(np.arctan2(dy, dx))

        erratisme = np.std(directions) if len(directions) > 2 else 0

        # Détection de fuite (vitesse élevée + direction cohérente)
        fuite = avg_speed > 80 and erratisme < 0.8

        # Score posture
        score = 0
        posture = "normal"

        if avg_speed > 120:
            score += 50; posture = "FUITE"
        elif avg_speed > 60:
            score += 30; posture = "course"
        elif avg_speed < 5 and len(history) > 15:
            score += 20; posture = "surveillance"
        else:
            posture = "marche"

        if erratisme > 1.2:
            score += 20; posture += "+agité"

        return posture, min(score, 80), fuite

    def compute_threat_score(self, face_emotion, face_score,
                              posture, posture_score,
                              has_suspicious_object):
        """Calcule le score de menace final (0-100)."""
        score = 0

        # Composante faciale (40% du score si disponible)
        if face_emotion:
            score += face_score * 0.4

        # Composante posturale (40%)
        score += posture_score * 0.4

        # Objet suspect (+30 points direct)
        if has_suspicious_object:
            score += 30

        return min(round(score), 100)

    def get_statut(self, score):
        if score >= SCORE_SUSPECT:
            return "SUSPECT", C_RED
        elif score >= SCORE_GENTIL:
            return "NEUTRE",  C_ORANGE
        else:
            return "GENTIL",  C_GREEN

# ══════════════════════════════════════════════════════
#  TRACKER PERSONNES
# ══════════════════════════════════════════════════════
class PersonTracker:
    def __init__(self):
        self.next_id  = 0
        self.tracks   = {}   # id → track data
        self.fugitifs = {}   # id → dernière position connue

    @staticmethod
    def iou(b1, b2):
        xa=max(b1[0],b2[0]); ya=max(b1[1],b2[1])
        xb=min(b1[2],b2[2]); yb=min(b1[3],b2[3])
        inter=max(0,xb-xa)*max(0,yb-ya)
        a1=(b1[2]-b1[0])*(b1[3]-b1[1])
        a2=(b2[2]-b2[0])*(b2[3]-b2[1])
        return inter/(a1+a2-inter+1e-6)

    def update(self, detections, t_now):
        for tid in list(self.tracks):
            self.tracks[tid]['missing'] += 1

        matched = set()
        results = []

        for det in detections:
            x1,y1,x2,y2 = det
            box = (x1,y1,x2,y2)
            best_iou, best_id = 0, None
            for tid, t in self.tracks.items():
                if tid in matched: continue
                sc = self.iou(box, t['box'])
                if sc > best_iou: best_iou,best_id = sc,tid

            if best_iou >= 0.25 and best_id is not None:
                tid = best_id
            else:
                tid = self.next_id; self.next_id += 1
                self.tracks[tid] = {
                    'box': box, 'history': [], 'missing': 0,
                    'trail': deque(maxlen=TRAIL_LEN),
                    'threat_history': deque(maxlen=10),
                    'was_suspect': False,
                }

            t  = self.tracks[tid]
            cx = (x1+x2)/2; cy = (y1+y2)/2
            t['history'].append((cx, cy, t_now))
            if len(t['history']) > MAX_HISTORY: t['history'].pop(0)
            t['trail'].append((int(cx), int(cy)))
            t['box'] = box; t['missing'] = 0
            matched.add(tid)
            results.append((tid, box, cx, cy, t['history'], t['trail']))

        # Détecter fugitifs (suspects disparus)
        for tid in list(self.tracks):
            t = self.tracks[tid]
            if t['missing'] > 15 and t.get('was_suspect'):
                if tid not in self.fugitifs and t['history']:
                    last = t['history'][-1]
                    self.fugitifs[tid] = {
                        'last_pos': (int(last[0]), int(last[1])),
                        'time': datetime.now().isoformat(timespec="seconds"),
                        'trail': list(t['trail']),
                    }
            if t['missing'] > 30:
                del self.tracks[tid]

        return results

    def mark_suspect(self, tid):
        if tid in self.tracks:
            self.tracks[tid]['was_suspect'] = True

# ══════════════════════════════════════════════════════
#  ZONE INTERDITE
# ══════════════════════════════════════════════════════
class ForbiddenZone:
    def __init__(self):
        self.points=[]; self.active=False; self.closed=False

    def start(self):
        self.points=[]; self.active=True; self.closed=False
        print("[ZONE] Clic gauche=point | Clic droit=fermer")

    def add_point(self,x,y):
        if self.active: self.points.append([x,y])

    def close(self):
        if len(self.points)>=3:
            self.active=False; self.closed=True
            print("[ZONE] Zone fermée ✓")

    def clear(self):
        self.points=[]; self.active=False; self.closed=False

    def contains(self,cx,cy):
        if not self.closed or len(self.points)<3: return False
        poly=np.array(self.points,np.int32)
        return cv2.pointPolygonTest(poly,(float(cx),float(cy)),False)>=0

    def draw(self,frame,flash=False):
        if not self.points: return
        pts=np.array(self.points,np.int32).reshape((-1,1,2))
        if self.closed:
            ov=frame.copy()
            cv2.fillPoly(ov,[pts],(0,0,140))
            cv2.addWeighted(ov,0.55 if flash else 0.25,frame,
                            0.45 if flash else 0.75,0,frame)
            cv2.polylines(frame,[pts],True,(0,0,255) if not flash else (0,200,255),2)
            cx=int(np.mean([p[0] for p in self.points]))
            cy=int(np.mean([p[1] for p in self.points]))
            cv2.putText(frame,"ZONE INTERDITE",(cx-60,cy),
                        cv2.FONT_HERSHEY_SIMPLEX,0.55,C_WHITE,2)
        else:
            cv2.polylines(frame,[pts],False,(0,0,255),2)
            for p in self.points: cv2.circle(frame,tuple(p),4,(0,0,255),-1)
        if self.active:
            cv2.putText(frame,"ZONE: clic gauche=pt | clic droit=fermer",
                        (10,frame.shape[0]-20),
                        cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2)

# ══════════════════════════════════════════════════════
#  DESSIN
# ══════════════════════════════════════════════════════
def draw_person(frame, x1,y1,x2,y2, pid, score, statut, color,
                expression, posture, objet, in_zone, fugitif=False):
    x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)
    bgr = color

    # Boîte principale
    cv2.rectangle(frame,(x1,y1),(x2,y2),bgr,2)

    # Coins accentués
    cs,lw=10,3
    for px,py in [(x1,y1),(x2,y1),(x1,y2),(x2,y2)]:
        dx=cs if px==x1 else -cs; dy=cs if py==y1 else -cs
        cv2.line(frame,(px,py),(px+dx,py),bgr,lw)
        cv2.line(frame,(px,py),(px,py+dy),bgr,lw)

    # Icône statut
    icon = "⚠" if statut=="SUSPECT" else ("?" if statut=="NEUTRE" else "✓")
    flags = ""
    if in_zone:   flags += " [ZONE]"
    if fugitif:   flags += " [FUITE]"
    if objet:     flags += f" [{objet}]"

    # Label
    txt = f"#{pid} {statut} {score}%{flags}"
    (tw,th),_=cv2.getTextSize(txt,cv2.FONT_HERSHEY_SIMPLEX,0.46,1)
    lx,ly=max(x1,0),max(y1-20,0)
    cv2.rectangle(frame,(lx,ly),(lx+tw+6,ly+th+4),bgr,-1)
    cv2.putText(frame,txt,(lx+3,ly+th+1),
                cv2.FONT_HERSHEY_SIMPLEX,0.46,(0,0,0),1,cv2.LINE_AA)

    # Barre de score
    bw=x2-x1; ratio=score/100
    bar_col = C_RED if ratio>.65 else C_ORANGE if ratio>.35 else C_GREEN
    cv2.rectangle(frame,(x1,y2+2),(x2,y2+4),(40,40,40),-1)
    cv2.rectangle(frame,(x1,y2+2),(x1+int(bw*ratio),y2+4),bar_col,-1)

    # Expression + posture (sous la boîte)
    if expression or posture:
        info = f"{expression or ''}  {posture or ''}"
        cv2.putText(frame,info,(x1,y2+16),
                    cv2.FONT_HERSHEY_SIMPLEX,0.35,(180,180,180),1,cv2.LINE_AA)

def draw_trails(frame, tracker, suspects_ids):
    for tid, t in tracker.tracks.items():
        pts = list(t['trail'])
        if len(pts) < 2: continue
        color = TRAIL_PALETTE[tid % len(TRAIL_PALETTE)]
        if tid in suspects_ids:
            color = (0, 0, 255)  # rouge pour suspects
        for i in range(1, len(pts)):
            alpha = i/len(pts)
            c = tuple(int(v*alpha) for v in color)
            cv2.line(frame, pts[i-1], pts[i], c, max(1,int(alpha*3)), cv2.LINE_AA)
        if pts:
            cv2.circle(frame, pts[-1], 4, color, -1)

def draw_fugitif_trails(frame, fugitifs):
    """Dessine les trajectoires des fugitifs en rouge pointillé."""
    for fid, fdata in fugitifs.items():
        trail = fdata.get('trail', [])
        if len(trail) < 2: continue
        for i in range(1, len(trail)):
            cv2.line(frame, trail[i-1], trail[i], (0,0,255), 2, cv2.LINE_AA)
            if i % 3 == 0:
                cv2.circle(frame, trail[i], 3, (0,200,255), -1)
        if trail:
            lp = trail[-1]
            cv2.putText(frame, f"FUGITIF #{fid}",
                        (lp[0]+5, lp[1]-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,255), 2)
            cv2.circle(frame, lp, 8, (0,0,255), 2)

def draw_hud(frame, fps, total, suspects, neutres, gentils,
             crowd_alert, frame_num, alerts_n, dash_on, paused):
    H,W=frame.shape[:2]
    ov=frame.copy()
    cv2.rectangle(ov,(6,6),(240,210),C_BLACK,-1)
    cv2.addWeighted(ov,0.80,frame,0.20,0,frame)
    cv2.rectangle(frame,(6,6),(240,210),(0,0,200),1)

    lines=[
        ("VIGILANCE AI",           (0,100,255),0.50,2),
        (f"FRAME  : {frame_num}",  C_GREEN,    0.37,1),
        (f"FPS    : {fps:.1f}",    C_GREEN,    0.37,1),
        (f"TOTAL  : {total}",      C_WHITE,    0.37,1),
        (f"GENTILS: {gentils}",    C_GREEN,    0.37,1),
        (f"NEUTRES: {neutres}",    C_YELLOW,   0.37,1),
        (f"SUSPECTS:{suspects}",   C_RED,      0.37,1),
        (f"ALERTES: {alerts_n}",
         C_RED if alerts_n>0 else C_GREEN,     0.37,1),
        (f"FOULE  : {'ALERTE!' if crowd_alert else 'OK'}",
         C_RED if crowd_alert else C_GREEN,    0.37,1),
        (f"DASH   : {'ON:5000' if dash_on else 'OFF'}",
         (160,160,160),0.33,1),
    ]
    for i,(txt,col,sc,th) in enumerate(lines):
        cv2.putText(frame,txt,(13,26+i*18),
                    cv2.FONT_HERSHEY_SIMPLEX,sc,col,th,cv2.LINE_AA)

    if crowd_alert:
        cv2.rectangle(frame,(0,0),(W,38),(0,0,140),-1)
        cv2.putText(frame,"⚠  ALERTE FOULE — SURPOPULATION DÉTECTÉE  ⚠",
                    (8,26),cv2.FONT_HERSHEY_SIMPLEX,0.60,C_WHITE,2)

    if paused:
        cv2.rectangle(frame,(W-90,6),(W-6,26),(0,140,255),-1)
        cv2.putText(frame,"PAUSE",(W-78,20),
                    cv2.FONT_HERSHEY_SIMPLEX,0.48,C_WHITE,1)
    else:
        cv2.circle(frame,(W-18,15),7,C_RED,-1)
        cv2.putText(frame,"LIVE",(W-60,20),
                    cv2.FONT_HERSHEY_SIMPLEX,0.42,C_RED,1)

    help_txt="q=Quit SPACE=Pause s=Screenshot z=Zone d=Dashboard c=Clear"
    cv2.putText(frame,help_txt,(8,H-6),
                cv2.FONT_HERSHEY_SIMPLEX,0.28,(100,100,100),1,cv2.LINE_AA)

# ══════════════════════════════════════════════════════
#  SOURIS
# ══════════════════════════════════════════════════════
class MouseHandler:
    def __init__(self, zone):
        self.zone = zone

    def callback(self, event, x, y, flags, param):
        if self.zone.active:
            if event==cv2.EVENT_LBUTTONDOWN: self.zone.add_point(x,y)
            elif event==cv2.EVENT_RBUTTONDOWN: self.zone.close()

# ══════════════════════════════════════════════════════
#  PIPELINE PRINCIPAL
# ══════════════════════════════════════════════════════
def run(source, show, save_path, log_every=3):
    print(f"\n[INFO] Chargement YOLOv8...")
    model    = YOLO("yolov8n.pt")
    analyzer = BehaviorAnalyzer()
    tracker  = PersonTracker()
    zone     = ForbiddenZone()
    logger   = Logger()
    gps_sim  = GPSSimulator()
    mouse    = MouseHandler(zone)
    flask_app= make_flask_app()

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Source inaccessible : {source}")

    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_src = cap.get(cv2.CAP_PROP_FPS) or 30

    writer = None
    if save_path:
        writer = cv2.VideoWriter(save_path,
                    cv2.VideoWriter_fourcc(*'mp4v'), fps_src, (W,H))

    if show:
        cv2.namedWindow("VigilanceAI", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("VigilanceAI", mouse.callback)

    frame_num    = 0
    fps_disp     = 0.0
    t_prev       = time.time()
    paused       = False
    dash_running = False
    alerts       = []
    frame        = None
    suspects_ids = set()

    print(f"[INFO] {W}×{H} | source={source}")
    print("[INFO] q=Quitter SPACE=Pause s=Screenshot z=Zone d=Dashboard c=Effacer\n")

    while True:
        if not paused:
            ret, frame_orig = cap.read()
            if not ret:
                print("[INFO] Fin vidéo.")
                break
            frame_num += 1
            t_now = time.time()
            fps_disp = 0.9*fps_disp + 0.1*(1/max(t_now-t_prev,1e-6))
            t_prev = t_now
            frame = frame_orig.copy()

            # ── Inférence YOLO (personnes + objets + véhicules) ──
            all_classes = [PERSON_CLASS] + list(VEHICLE_CLASSES.keys()) + list(OBJECT_CLASSES.keys())
            results = model(frame, classes=all_classes,
                           conf=CONF_THRESHOLD, iou=0.45, verbose=False)[0]

            person_dets  = []
            vehicle_dets = []
            object_dets  = []

            for box in results.boxes:
                x1,y1,x2,y2 = box.xyxy[0].tolist()
                cls  = int(box.cls[0])
                conf = float(box.conf[0])
                if cls == PERSON_CLASS:
                    person_dets.append((x1,y1,x2,y2))
                elif cls in VEHICLE_CLASSES:
                    vehicle_dets.append((x1,y1,x2,y2,cls,conf))
                elif cls in OBJECT_CLASSES:
                    object_dets.append((x1,y1,x2,y2,cls,conf))

            # ── Tracker personnes ──
            tracked = tracker.update(person_dets, t_now)

            # ── Dessiner trajectoires ──
            draw_trails(frame, tracker, suspects_ids)
            draw_fugitif_trails(frame, tracker.fugitifs)

            total_persons = len(tracked)
            n_suspects=0; n_neutres=0; n_gentils=0
            zone_flash = False
            suspects_ids = set()
            persons_snap = []
            vehicles_fuite_snap = []

            for pid, box, cx, cy, history, trail in tracked:
                x1,y1,x2,y2 = box

                # ROI visage (tiers supérieur de la boîte)
                face_h = int((y2-y1)*0.35)
                face_roi = frame_orig[int(y1):int(y1)+face_h, int(x1):int(x2)]

                # Analyse faciale
                expression, face_score = analyzer.analyze_face(face_roi)

                # Analyse posturale
                posture, posture_score, is_fleeing = analyzer.analyze_posture(history, box)

                # Objets sur/près de la personne
                objet_suspect = None
                for ox1,oy1,ox2,oy2,ocls,_ in object_dets:
                    # Vérifier si l'objet est proche de la personne
                    ocx=(ox1+ox2)/2; ocy=(oy1+oy2)/2
                    if x1<=ocx<=x2 and y1<=ocy<=y2:
                        oname = OBJECT_CLASSES[ocls]
                        if ocls in SUSPICIOUS_OBJECTS:
                            objet_suspect = oname
                        # Dessiner l'objet
                        col = (0,0,255) if ocls in SUSPICIOUS_OBJECTS else (0,200,255)
                        cv2.rectangle(frame,(int(ox1),int(oy1)),(int(ox2),int(oy2)),col,1)
                        cv2.putText(frame,oname,(int(ox1),int(oy1)-4),
                                    cv2.FONT_HERSHEY_SIMPLEX,0.35,col,1)

                # Score final
                score = analyzer.compute_threat_score(
                    expression, face_score,
                    posture, posture_score,
                    objet_suspect is not None
                )

                statut, color = analyzer.get_statut(score)

                if statut == "SUSPECT":
                    n_suspects += 1
                    suspects_ids.add(pid)
                    tracker.mark_suspect(pid)
                elif statut == "NEUTRE":
                    n_neutres += 1
                else:
                    n_gentils += 1

                # Zone interdite
                in_zone = zone.contains(cx, cy)
                if in_zone: zone_flash = True

                # GPS simulé
                gps = gps_sim.pixel_to_gps(cx, cy)

                # Alerte fuite + véhicule de fuite
                if is_fleeing and statut in ("SUSPECT","NEUTRE"):
                    # Chercher véhicule à proximité
                    for vx1,vy1,vx2,vy2,vcls,_ in vehicle_dets:
                        vcx=(vx1+vx2)/2; vcy=(vy1+vy2)/2
                        dist=np.sqrt((vcx-cx)**2+(vcy-cy)**2)
                        if dist < 200:
                            vtype = VEHICLE_CLASSES[vcls]
                            evt = {
                                "id":pid,"type":vtype,
                                "proximity":f"{int(dist)}px",
                                "time":datetime.now().isoformat(timespec="seconds")
                            }
                            vehicles_fuite_snap.append(evt)
                            # Dessiner le véhicule de fuite
                            cv2.rectangle(frame,(int(vx1),int(vy1)),(int(vx2),int(vy2)),(0,0,255),3)
                            cv2.putText(frame,f"VEHICULE FUITE #{pid}",
                                        (int(vx1),int(vy1)-8),
                                        cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2)

                draw_person(frame,x1,y1,x2,y2,pid,score,statut,color,
                           expression,posture,objet_suspect,in_zone,is_fleeing)

                # Alertes
                if statut=="SUSPECT" or in_zone or objet_suspect:
                    msg = f"#{pid} {statut} score={score}%"
                    if in_zone: msg += " ZONE"
                    if objet_suspect: msg += f" {objet_suspect}"
                    alerts.append({"msg":msg,
                                   "time":datetime.now().isoformat(timespec="seconds")})

                if frame_num % log_every == 0:
                    logger.log(frame_num,pid,score,statut,
                               expression or "",posture,
                               objet_suspect or "",box,gps,is_fleeing)

                persons_snap.append({
                    "id":pid,"score":score,"statut":statut,
                    "expression":expression or "","posture":posture,
                    "objet":objet_suspect or ""
                })

            # Zone interdite
            zone.draw(frame, flash=zone_flash)

            # Véhicules normaux
            for vx1,vy1,vx2,vy2,vcls,_ in vehicle_dets:
                vtype=VEHICLE_CLASSES[vcls]
                cv2.rectangle(frame,(int(vx1),int(vy1)),(int(vx2),int(vy2)),(255,200,0),1)
                cv2.putText(frame,vtype,(int(vx1),int(vy1)-4),
                            cv2.FONT_HERSHEY_SIMPLEX,0.35,(255,200,0),1)

            # Alerte foule
            crowd_alert = total_persons >= CROWD_ALERT_N

            # Flash alerte suspect
            if n_suspects > 0:
                ov=frame.copy()
                cv2.rectangle(ov,(0,0),(W,H),(0,0,80),-1)
                cv2.addWeighted(ov,0.08,frame,0.92,0,frame)

            draw_hud(frame,fps_disp,total_persons,n_suspects,n_neutres,
                     n_gentils,crowd_alert,frame_num,len(alerts),
                     dash_running,paused)

            if frame_num % 300 == 0: logger.flush()

            # Mise à jour Flask
            fugitifs_snap = [
                {"id":fid,
                 "last_pos":f"{fd['last_pos'][0]},{fd['last_pos'][1]}",
                 "time":fd['time'],
                 "gps":str(gps_sim.pixel_to_gps(*fd['last_pos']))}
                for fid,fd in tracker.fugitifs.items()
            ]
            with shared.lock:
                shared.fps=fps_disp; shared.frame_num=frame_num
                shared.total=total_persons; shared.suspects=n_suspects
                shared.neutres=n_neutres; shared.gentils=n_gentils
                shared.crowd_alert=crowd_alert; shared.alerts=alerts
                shared.persons=persons_snap; shared.fugitifs=fugitifs_snap
                shared.vehicles_fuite=vehicles_fuite_snap
                if dash_running:
                    _,jpg=cv2.imencode('.jpg',frame,[cv2.IMWRITE_JPEG_QUALITY,60])
                    shared.frame_jpg=jpg.tobytes()

            if writer: writer.write(frame)

        if show and frame is not None:
            cv2.imshow("VigilanceAI", frame)
            key=cv2.waitKey(1 if not paused else 50) & 0xFF
            if   key==ord('q'): break
            elif key==ord(' '): paused=not paused
            elif key==ord('s'):
                fn=f"screenshot_{frame_num}.jpg"
                cv2.imwrite(fn,frame); print(f"[INFO] Screenshot → {fn}")
            elif key==ord('z'): zone.start()
            elif key==ord('c'): zone.clear()
            elif key==ord('d'):
                if flask_app and not dash_running:
                    threading.Thread(target=lambda:flask_app.run(
                        host='0.0.0.0',port=5000,debug=False,use_reloader=False
                    ),daemon=True).start()
                    dash_running=True
                    print("[DASH] http://localhost:5000")

    cap.release()
    if writer: writer.release()
    cv2.destroyAllWindows()
    logger.flush()
    print(f"\n[INFO] Terminé — {frame_num} frames")
    print(f"[LOG]  {logger.csv_path}")

# ══════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════
if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser(description="VigilanceAI")
    p.add_argument("--source",   default=0)
    p.add_argument("--no-show",  action="store_true")
    p.add_argument("--save",     default=None)
    p.add_argument("--log-every",type=int,default=3)
    args=p.parse_args()
    source=int(args.source) if str(args.source).isdigit() else args.source
    run(source=source, show=not args.no_show,
        save_path=args.save, log_every=args.log_every)

# ══════════════════════════════════════════════════════
#  INTÉGRATION GPS RÉEL DANS LE PIPELINE
#  (patch appliqué sur le run() existant)
# ══════════════════════════════════════════════════════
def run_with_gps(source, show=True, save_path=None, log_every=3,
                 opencage_key="", w3w_key="",
                 camera_lat=None, camera_lon=None,
                 camera_alt=8.0, camera_fov_h=70.0,
                 camera_fov_v=45.0, camera_tilt=45.0):
    """
    Version complète avec GPS réel intégré.
    Remplace run() pour un usage en production.
    """
    from gps_engine import GPSEngine

    print(f"\n[INFO] Chargement YOLOv8 + GPS Engine...")
    model    = YOLO("yolov8n.pt")
    analyzer = BehaviorAnalyzer()
    tracker  = PersonTracker()
    zone     = ForbiddenZone()
    logger   = Logger()
    mouse    = MouseHandler(zone)
    flask_app= make_flask_app()

    # ── GPS Engine ──
    gps = GPSEngine(
        opencage_key=opencage_key,
        w3w_key=w3w_key,
        camera_lat=camera_lat,
        camera_lon=camera_lon,
        camera_alt=camera_alt,
        camera_fov_h=camera_fov_h,
        camera_fov_v=camera_fov_v,
        camera_tilt=camera_tilt,
    )
    gps.auto_locate_camera()
    cam_info = gps.get_camera_info()
    print(f"[GPS] Caméra : {cam_info['lat']}, {cam_info['lon']}")
    if cam_info.get("osm_url"):
        print(f"[GPS] Carte caméra : {cam_info['osm_url']}")

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Source inaccessible : {source}")

    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_src = cap.get(cv2.CAP_PROP_FPS) or 30

    writer = None
    if save_path:
        writer = cv2.VideoWriter(save_path,
                    cv2.VideoWriter_fourcc(*'mp4v'), fps_src, (W, H))

    if show:
        cv2.namedWindow("VigilanceAI", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("VigilanceAI", mouse.callback)

    frame_num    = 0
    fps_disp     = 0.0
    t_prev       = time.time()
    paused       = False
    dash_running = False
    alerts       = []
    suspects_ids = set()
    frame        = None
    gps_tracks   = {}    # pid → {statut, score, gps_history}

    print(f"[INFO] {W}×{H} | GPS actif\n")

    while True:
        if not paused:
            ret, frame_orig = cap.read()
            if not ret: break
            frame_num += 1
            t_now = time.time()
            fps_disp = 0.9*fps_disp + 0.1*(1/max(t_now-t_prev, 1e-6))
            t_prev = t_now
            frame = frame_orig.copy()

            all_classes = ([PERSON_CLASS]
                          + list(VEHICLE_CLASSES.keys())
                          + list(OBJECT_CLASSES.keys()))
            results = model(frame, classes=all_classes,
                           conf=CONF_THRESHOLD, iou=0.45,
                           verbose=False)[0]

            person_dets=[]; vehicle_dets=[]; object_dets=[]
            for box in results.boxes:
                x1,y1,x2,y2=box.xyxy[0].tolist()
                cls=int(box.cls[0]); conf=float(box.conf[0])
                if cls==PERSON_CLASS:
                    person_dets.append((x1,y1,x2,y2))
                elif cls in VEHICLE_CLASSES:
                    vehicle_dets.append((x1,y1,x2,y2,cls,conf))
                elif cls in OBJECT_CLASSES:
                    object_dets.append((x1,y1,x2,y2,cls,conf))

            tracked = tracker.update(person_dets, t_now)
            draw_trails(frame, tracker, suspects_ids)
            draw_fugitif_trails(frame, tracker.fugitifs)

            total_persons=len(tracked); n_suspects=0; n_neutres=0; n_gentils=0
            zone_flash=False; suspects_ids=set()
            persons_snap=[]; vehicles_fuite_snap=[]

            for pid, box, cx, cy, history, trail in tracked:
                x1,y1,x2,y2=box

                # ── GPS réel ──
                gps_info = gps.track_person(
                    pid, cx, cy, W, H,
                    force_geocode=(frame_num % 150 == 0)  # geocode toutes les ~5s
                )
                lat, lon = gps_info["lat"], gps_info["lon"]

                # Stocker historique GPS
                if pid not in gps_tracks:
                    gps_tracks[pid] = {"statut":"?","score":0,"gps_history":[]}
                gps_tracks[pid]["gps_history"].append(gps_info)
                if len(gps_tracks[pid]["gps_history"]) > 100:
                    gps_tracks[pid]["gps_history"].pop(0)

                # Analyse comportementale
                face_roi = frame_orig[int(y1):int(y1)+int((y2-y1)*0.35), int(x1):int(x2)]
                expression, face_score = analyzer.analyze_face(face_roi)
                posture, posture_score, is_fleeing = analyzer.analyze_posture(history, box)

                objet_suspect=None
                for ox1,oy1,ox2,oy2,ocls,_ in object_dets:
                    ocx=(ox1+ox2)/2; ocy=(oy1+oy2)/2
                    if x1<=ocx<=x2 and y1<=ocy<=y2:
                        oname=OBJECT_CLASSES[ocls]
                        if ocls in SUSPICIOUS_OBJECTS: objet_suspect=oname
                        col=(0,0,255) if ocls in SUSPICIOUS_OBJECTS else (0,200,255)
                        cv2.rectangle(frame,(int(ox1),int(oy1)),(int(ox2),int(oy2)),col,1)
                        cv2.putText(frame,oname,(int(ox1),int(oy1)-4),
                                    cv2.FONT_HERSHEY_SIMPLEX,0.35,col,1)

                score=analyzer.compute_threat_score(
                    expression,face_score,posture,posture_score,objet_suspect is not None)
                statut,color=analyzer.get_statut(score)

                gps_tracks[pid]["statut"]=statut
                gps_tracks[pid]["score"]=score

                if statut=="SUSPECT": n_suspects+=1; suspects_ids.add(pid); tracker.mark_suspect(pid)
                elif statut=="NEUTRE": n_neutres+=1
                else: n_gentils+=1

                in_zone=zone.contains(cx,cy)
                if in_zone: zone_flash=True

                # ── Afficher GPS sur la frame ──
                gps_txt = f"GPS:{lat:.4f},{lon:.4f}"
                if gps_info.get("w3w"):
                    gps_txt = gps_info["w3w"]
                elif gps_info.get("city") and gps_info["city"] != "?":
                    gps_txt = f"{gps_info['city']}"
                cv2.putText(frame, gps_txt,
                            (int(x1), int(y2)+28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.30,
                            (200, 255, 200), 1, cv2.LINE_AA)

                # Véhicule de fuite
                if is_fleeing and statut in ("SUSPECT","NEUTRE"):
                    for vx1,vy1,vx2,vy2,vcls,_ in vehicle_dets:
                        vcx=(vx1+vx2)/2; vcy=(vy1+vy2)/2
                        dist=np.sqrt((vcx-cx)**2+(vcy-cy)**2)
                        if dist<200:
                            vtype=VEHICLE_CLASSES[vcls]
                            vgps=gps.pixel_to_gps(vcx,vcy,W,H)
                            evt={"id":pid,"type":vtype,
                                 "proximity":f"{int(dist)}px",
                                 "gps":f"{vgps[0]:.5f},{vgps[1]:.5f}",
                                 "time":datetime.now().isoformat(timespec="seconds")}
                            vehicles_fuite_snap.append(evt)
                            cv2.rectangle(frame,(int(vx1),int(vy1)),(int(vx2),int(vy2)),(0,0,255),3)
                            cv2.putText(frame,f"VEHICULE FUITE #{pid} GPS:{vgps[0]:.4f},{vgps[1]:.4f}",
                                        (int(vx1),int(vy1)-8),
                                        cv2.FONT_HERSHEY_SIMPLEX,0.45,(0,0,255),2)

                draw_person(frame,x1,y1,x2,y2,pid,score,statut,color,
                           expression,posture,objet_suspect,in_zone,is_fleeing)

                if statut=="SUSPECT" or in_zone or objet_suspect:
                    msg=f"#{pid} {statut} score={score}% GPS:{lat:.4f},{lon:.4f}"
                    if gps_info.get("city","?")!="?": msg+=f" {gps_info['city']}"
                    alerts.append({"msg":msg,
                                   "time":datetime.now().isoformat(timespec="seconds")})

                if frame_num % log_every == 0:
                    logger.log(frame_num,pid,score,statut,
                               expression or "",posture,
                               objet_suspect or "",box,
                               (lat,lon),is_fleeing)

                persons_snap.append({
                    "id":pid,"score":score,"statut":statut,
                    "expression":expression or "","posture":posture,
                    "objet":objet_suspect or "",
                    "gps":f"{lat:.5f},{lon:.5f}",
                    "address":gps_info.get("address","")[:40],
                })

            zone.draw(frame, flash=zone_flash)

            for vx1,vy1,vx2,vy2,vcls,_ in vehicle_dets:
                vtype=VEHICLE_CLASSES[vcls]
                cv2.rectangle(frame,(int(vx1),int(vy1)),(int(vx2),int(vy2)),(255,200,0),1)
                cv2.putText(frame,vtype,(int(vx1),int(vy1)-4),
                            cv2.FONT_HERSHEY_SIMPLEX,0.35,(255,200,0),1)

            crowd_alert=total_persons>=CROWD_ALERT_N
            if n_suspects>0:
                ov=frame.copy()
                cv2.rectangle(ov,(0,0),(W,H),(0,0,80),-1)
                cv2.addWeighted(ov,0.08,frame,0.92,0,frame)

            draw_hud(frame,fps_disp,total_persons,n_suspects,n_neutres,
                     n_gentils,crowd_alert,frame_num,len(alerts),
                     dash_running,paused)

            # Info caméra GPS (coin bas droit)
            cam_txt=f"CAM GPS:{cam_info['lat']:.4f},{cam_info['lon']:.4f}"
            cv2.putText(frame,cam_txt,(8,H-22),
                        cv2.FONT_HERSHEY_SIMPLEX,0.32,(100,200,100),1,cv2.LINE_AA)

            if frame_num%300==0:
                logger.flush()
                gps.export_tracking_report(gps_tracks)

            fugitifs_snap=[
                {"id":fid,
                 "last_pos":f"{fd['last_pos'][0]},{fd['last_pos'][1]}",
                 "time":fd['time'],
                 "gps":str(gps.pixel_to_gps(*fd['last_pos'],W,H))}
                for fid,fd in tracker.fugitifs.items()
            ]

            with shared.lock:
                shared.fps=fps_disp; shared.frame_num=frame_num
                shared.total=total_persons; shared.suspects=n_suspects
                shared.neutres=n_neutres; shared.gentils=n_gentils
                shared.crowd_alert=crowd_alert; shared.alerts=alerts
                shared.persons=persons_snap; shared.fugitifs=fugitifs_snap
                shared.vehicles_fuite=vehicles_fuite_snap
                if dash_running:
                    _,jpg=cv2.imencode('.jpg',frame,[cv2.IMWRITE_JPEG_QUALITY,60])
                    shared.frame_jpg=jpg.tobytes()

            if writer: writer.write(frame)

        if show and frame is not None:
            cv2.imshow("VigilanceAI", frame)
            key=cv2.waitKey(1 if not paused else 50) & 0xFF
            if   key==ord('q'): break
            elif key==ord(' '): paused=not paused
            elif key==ord('s'):
                fn=f"screenshot_{frame_num}.jpg"
                cv2.imwrite(fn,frame); print(f"[INFO] {fn}")
            elif key==ord('z'): zone.start()
            elif key==ord('c'): zone.clear()
            elif key==ord('d'):
                if flask_app and not dash_running:
                    threading.Thread(target=lambda:flask_app.run(
                        host='0.0.0.0',port=5000,debug=False,use_reloader=False
                    ),daemon=True).start()
                    dash_running=True
                    print("[DASH] http://localhost:5000")

    cap.release()
    if writer: writer.release()
    cv2.destroyAllWindows()
    logger.flush()
    report_path = gps.export_tracking_report(gps_tracks)
    print(f"\n[INFO] Session terminée — {frame_num} frames")
    print(f"[GPS]  Rapport GeoJSON → {report_path}")
    print(f"[GPS]  Appels API — ipinfo:{gps.calls_ipinfo} OSM:{gps.calls_osm} OpenCage:{gps.calls_opencage} W3W:{gps.calls_w3w}")


# ── Remplace le main pour utiliser GPS réel ──
if __name__ == "__main__":
    import argparse
    p=argparse.ArgumentParser(description="VigilanceAI avec GPS réel")
    p.add_argument("--source",      default=0)
    p.add_argument("--no-show",     action="store_true")
    p.add_argument("--save",        default=None)
    p.add_argument("--log-every",   type=int,   default=3)
    p.add_argument("--opencage-key",default="",  help="Clé API OpenCage (optionnel)")
    p.add_argument("--w3w-key",     default="",  help="Clé API What3Words (optionnel)")
    p.add_argument("--cam-lat",     type=float,  default=None, help="Latitude caméra")
    p.add_argument("--cam-lon",     type=float,  default=None, help="Longitude caméra")
    p.add_argument("--cam-alt",     type=float,  default=8.0,  help="Altitude caméra (m)")
    p.add_argument("--cam-fov-h",   type=float,  default=70.0, help="FOV horizontal (°)")
    p.add_argument("--cam-fov-v",   type=float,  default=45.0, help="FOV vertical (°)")
    p.add_argument("--cam-tilt",    type=float,  default=45.0, help="Inclinaison (°)")
    args=p.parse_args()
    source=int(args.source) if str(args.source).isdigit() else args.source
    run_with_gps(
        source=source, show=not args.no_show, save_path=args.save,
        log_every=args.log_every,
        opencage_key=args.opencage_key, w3w_key=args.w3w_key,
        camera_lat=args.cam_lat, camera_lon=args.cam_lon,
        camera_alt=args.cam_alt, camera_fov_h=args.cam_fov_h,
        camera_fov_v=args.cam_fov_v, camera_tilt=args.cam_tilt,
    )
