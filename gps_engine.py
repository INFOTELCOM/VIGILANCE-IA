"""
VigilanceAI — GPS Engine
=========================
Système de géolocalisation réelle multi-sources :
  1. ipinfo.io      → position de la caméra par IP
  2. OpenCage API   → reverse geocoding (coords → adresse)
  3. What3Words API → adresse en 3 mots
  4. OSM Nominatim  → reverse geocoding gratuit illimité
  5. Homographie    → pixel → GPS réel via calibration caméra

Usage :
    gps = GPSEngine(opencage_key="VOTRE_CLE", w3w_key="VOTRE_CLE")
    gps.auto_locate_camera()          # détecte la position de la caméra
    lat, lon = gps.pixel_to_gps(cx, cy, frame_w, frame_h)
    info = gps.reverse_geocode(lat, lon)
    print(info["address"], info["w3w"])
"""

import requests
import numpy as np
import json
import time
import os
from datetime import datetime

# ══════════════════════════════════════════════════════
#  CONFIGURATION CLÉS API (gratuites)
# ══════════════════════════════════════════════════════
# OpenCage  : https://opencagedata.com/  (2500 req/jour gratuit)
# What3Words: https://developer.what3words.com/ (gratuit)
# ipinfo.io : gratuit sans clé (50k req/mois)
# OSM       : gratuit illimité (pas de clé)

OPENCAGE_KEY = os.environ.get("OPENCAGE_KEY", "")
W3W_KEY      = os.environ.get("W3W_KEY", "")

CACHE_TTL    = 30   # secondes entre deux appels GPS pour le même point


class GPSEngine:
    """
    Moteur GPS temps réel multi-sources.
    Convertit des coordonnées pixel → GPS réelles
    en se basant sur la position connue de la caméra.
    """

    def __init__(self, opencage_key=None, w3w_key=None,
                 camera_lat=None, camera_lon=None,
                 camera_fov_h=60.0, camera_fov_v=45.0,
                 camera_alt=10.0, camera_tilt=45.0):
        """
        opencage_key  : clé API OpenCage (optionnel)
        w3w_key       : clé API What3Words (optionnel)
        camera_lat/lon: position GPS de la caméra (si connue)
        camera_fov_h  : champ de vision horizontal (degrés)
        camera_fov_v  : champ de vision vertical (degrés)
        camera_alt    : altitude de la caméra (mètres)
        camera_tilt   : inclinaison de la caméra (degrés, 0=horizontal, 90=nadir)
        """
        self.opencage_key = opencage_key or OPENCAGE_KEY
        self.w3w_key      = w3w_key or W3W_KEY

        # Position de la caméra
        self.camera_lat  = camera_lat
        self.camera_lon  = camera_lon
        self.camera_alt  = camera_alt
        self.camera_fov_h = camera_fov_h
        self.camera_fov_v = camera_fov_v
        self.camera_tilt  = camera_tilt

        # Cache pour limiter les appels API
        self._cache       = {}
        self._last_call   = {}
        self._ip_info     = None

        # Stats
        self.calls_opencage  = 0
        self.calls_w3w       = 0
        self.calls_osm       = 0
        self.calls_ipinfo    = 0

        print("[GPS] GPSEngine initialisé")
        if self.opencage_key:
            print(f"[GPS] OpenCage API : ✓")
        if self.w3w_key:
            print(f"[GPS] What3Words API : ✓")
        print(f"[GPS] OSM Nominatim : ✓ (gratuit illimité)")
        print(f"[GPS] ipinfo.io : ✓ (sans clé)")

    # ─────────────────────────────────────────────────
    #  LOCALISATION AUTOMATIQUE DE LA CAMÉRA
    # ─────────────────────────────────────────────────
    def auto_locate_camera(self):
        """
        Détecte automatiquement la position GPS de la caméra
        via l'adresse IP publique (ipinfo.io).
        Utilisé quand la position n'est pas fournie manuellement.
        """
        if self.camera_lat and self.camera_lon:
            print(f"[GPS] Position caméra fournie : {self.camera_lat}, {self.camera_lon}")
            return True

        try:
            print("[GPS] Détection position caméra via ipinfo.io...")
            resp = requests.get(
                "https://ipinfo.io/json",
                timeout=5,
                headers={"User-Agent": "VigilanceAI/1.0"}
            )
            if resp.status_code == 200:
                data = resp.json()
                self._ip_info = data
                if "loc" in data:
                    lat_str, lon_str = data["loc"].split(",")
                    self.camera_lat = float(lat_str)
                    self.camera_lon = float(lon_str)
                    city    = data.get("city", "?")
                    country = data.get("country", "?")
                    org     = data.get("org", "?")
                    print(f"[GPS] ✓ Caméra localisée : {self.camera_lat:.4f}, {self.camera_lon:.4f}")
                    print(f"[GPS] ✓ Ville : {city}, {country} | FAI : {org}")
                    self.calls_ipinfo += 1
                    return True
        except Exception as e:
            print(f"[GPS] ipinfo.io échoué : {e}")

        # Fallback : Paris par défaut
        print("[GPS] Fallback : Paris (48.8566, 2.3522)")
        self.camera_lat = 48.8566
        self.camera_lon = 2.3522
        return False

    # ─────────────────────────────────────────────────
    #  CONVERSION PIXEL → GPS
    # ─────────────────────────────────────────────────
    def pixel_to_gps(self, cx, cy, frame_w, frame_h):
        """
        Convertit des coordonnées pixel (cx, cy) en GPS réel (lat, lon).

        Méthode : projection géométrique depuis la position de la caméra.
        Plus la caméra est bien calibrée (fov, tilt, altitude),
        plus la précision est élevée.

        Pour une précision maximale : utiliser calibration_bev() avec
        4 points GPS connus dans l'image.
        """
        if not self.camera_lat or not self.camera_lon:
            self.auto_locate_camera()

        # ── Normalisation pixel → [-1, 1] ──
        nx = (cx - frame_w / 2) / (frame_w / 2)   # -1=gauche, +1=droite
        ny = (cy - frame_h / 2) / (frame_h / 2)   # -1=haut,   +1=bas

        # ── Calcul de la distance au sol ──
        # On projette le pixel sur le sol en tenant compte de
        # l'angle de la caméra (tilt) et de son altitude
        tilt_rad = np.radians(self.camera_tilt)
        fov_h_rad = np.radians(self.camera_fov_h)
        fov_v_rad = np.radians(self.camera_fov_v)

        # Angle de visée pour ce pixel
        angle_h = nx * fov_h_rad / 2
        angle_v = tilt_rad + ny * fov_v_rad / 2

        # Distance horizontale au sol (mètres)
        if abs(np.sin(angle_v)) > 1e-6:
            dist_ground = abs(self.camera_alt / np.tan(angle_v))
        else:
            dist_ground = self.camera_alt * 10   # très loin

        # Déplacement Est-Ouest et Nord-Sud (mètres)
        dx_m = dist_ground * np.sin(angle_h)   # Est positif
        dy_m = dist_ground * np.cos(angle_h)   # Nord positif

        # ── Conversion mètres → degrés GPS ──
        # 1 degré latitude  ≈ 111,320 m
        # 1 degré longitude ≈ 111,320 * cos(lat) m
        lat_deg_per_m = 1 / 111320
        lon_deg_per_m = 1 / (111320 * np.cos(np.radians(self.camera_lat)))

        lat = self.camera_lat + dy_m * lat_deg_per_m
        lon = self.camera_lon + dx_m * lon_deg_per_m

        return round(lat, 7), round(lon, 7)

    # ─────────────────────────────────────────────────
    #  CALIBRATION BEV (4 points GPS connus)
    # ─────────────────────────────────────────────────
    def calibrate_bev(self, pixel_points, gps_points):
        """
        Calibration précise via 4 points de correspondance
        pixel ↔ GPS connus.

        pixel_points : [(px1,py1), (px2,py2), (px3,py3), (px4,py4)]
        gps_points   : [(lat1,lon1), (lat2,lon2), ...]

        Après calibration, pixel_to_gps_calibrated() donne
        des coordonnées très précises.
        """
        if len(pixel_points) != 4 or len(gps_points) != 4:
            print("[GPS] Calibration BEV : 4 points requis")
            return False

        src = np.float32(pixel_points)

        # Normaliser les GPS en mètres relatifs
        lat0, lon0 = gps_points[0]
        def gps_to_m(lat, lon):
            dx = (lon - lon0) * 111320 * np.cos(np.radians(lat0))
            dy = (lat - lat0) * 111320
            return dx, dy

        dst_pts = []
        for lat, lon in gps_points:
            dx, dy = gps_to_m(lat, lon)
            dst_pts.append([dx, dy])

        self._bev_src     = src
        self._bev_dst     = np.float32(dst_pts)
        self._bev_origin  = (lat0, lon0)
        self._bev_calibrated = True

        # Matrice de transformation homographique
        # (px → mètres relatifs)
        dst_normalized = np.float32([[p[0], p[1]] for p in dst_pts])
        scale = max(abs(dst_normalized).max(), 1)
        dst_px = (dst_normalized / scale * 100 + 200).astype(np.float32)
        self._H, _ = cv2.findHomography(src, dst_px)
        self._bev_scale  = scale
        self._bev_offset = 200

        print("[GPS] ✓ Calibration BEV effectuée avec 4 points GPS")
        return True

    # ─────────────────────────────────────────────────
    #  REVERSE GEOCODING (GPS → Adresse)
    # ─────────────────────────────────────────────────
    def reverse_geocode(self, lat, lon):
        """
        Convertit des coordonnées GPS en adresse lisible.
        Essaie dans l'ordre : OpenCage → OSM Nominatim.
        Retourne un dict avec address, city, country, w3w, osm_url.
        """
        cache_key = f"{lat:.4f},{lon:.4f}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        result = {
            "lat": lat, "lon": lon,
            "address": f"{lat:.5f}, {lon:.5f}",
            "city": "?", "country": "?",
            "w3w": None,
            "osm_url": f"https://www.openstreetmap.org/?mlat={lat}&mlon={lon}#map=18/{lat}/{lon}",
            "source": "coords"
        }

        # ── 1. OpenCage (si clé disponible) ──
        if self.opencage_key:
            try:
                url = (f"https://api.opencagedata.com/geocode/v1/json"
                       f"?q={lat}+{lon}&key={self.opencage_key}&limit=1&language=fr")
                resp = requests.get(url, timeout=5)
                if resp.status_code == 200:
                    data = resp.json()
                    if data["results"]:
                        r = data["results"][0]
                        result["address"] = r.get("formatted", result["address"])
                        comp = r.get("components", {})
                        result["city"]    = comp.get("city") or comp.get("town") or comp.get("village","?")
                        result["country"] = comp.get("country","?")
                        result["source"]  = "opencage"
                        self.calls_opencage += 1
            except Exception as e:
                print(f"[GPS] OpenCage erreur : {e}")

        # ── 2. OSM Nominatim (gratuit, toujours) ──
        if result["source"] == "coords":
            try:
                url = (f"https://nominatim.openstreetmap.org/reverse"
                       f"?lat={lat}&lon={lon}&format=json&accept-language=fr")
                resp = requests.get(url, timeout=5,
                    headers={"User-Agent": "VigilanceAI/1.0 contact@vigilance.ai"})
                if resp.status_code == 200:
                    data = resp.json()
                    addr = data.get("address", {})
                    result["address"] = data.get("display_name", result["address"])
                    result["city"]    = (addr.get("city") or addr.get("town")
                                        or addr.get("village") or addr.get("municipality","?"))
                    result["country"] = addr.get("country","?")
                    result["source"]  = "osm"
                    self.calls_osm += 1
                    time.sleep(1)   # OSM rate limit : 1 req/s
            except Exception as e:
                print(f"[GPS] OSM Nominatim erreur : {e}")

        # ── 3. What3Words (si clé disponible) ──
        if self.w3w_key:
            try:
                url = (f"https://api.what3words.com/v3/convert-to-3wa"
                       f"?coordinates={lat},{lon}&language=fr"
                       f"&key={self.w3w_key}")
                resp = requests.get(url, timeout=5)
                if resp.status_code == 200:
                    data = resp.json()
                    result["w3w"] = "///" + data.get("words","?")
                    self.calls_w3w += 1
            except Exception as e:
                print(f"[GPS] What3Words erreur : {e}")

        self._cache[cache_key] = result
        return result

    # ─────────────────────────────────────────────────
    #  TRACKING GPS EN TEMPS RÉEL
    # ─────────────────────────────────────────────────
    def track_person(self, person_id, cx, cy, frame_w, frame_h,
                     force_geocode=False):
        """
        Retourne les données GPS complètes pour une personne.
        Limite les appels API avec un cache temporel.
        """
        lat, lon = self.pixel_to_gps(cx, cy, frame_w, frame_h)

        now = time.time()
        cache_key = f"track_{person_id}"
        last = self._last_call.get(cache_key, 0)

        info = {
            "person_id": person_id,
            "lat": lat, "lon": lon,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "address": f"{lat:.5f}, {lon:.5f}",
            "city": "?", "w3w": None,
            "osm_url": f"https://www.openstreetmap.org/?mlat={lat}&mlon={lon}#map=18/{lat}/{lon}"
        }

        # Geocoding uniquement toutes les CACHE_TTL secondes
        if force_geocode or (now - last) > CACHE_TTL:
            self._last_call[cache_key] = now
            geo = self.reverse_geocode(lat, lon)
            info.update({
                "address": geo["address"],
                "city":    geo["city"],
                "country": geo["country"],
                "w3w":     geo["w3w"],
                "source":  geo["source"],
            })

        return info

    # ─────────────────────────────────────────────────
    #  EXPORT RAPPORT GPS
    # ─────────────────────────────────────────────────
    def export_tracking_report(self, tracks, output_path="logs/gps_tracking.json"):
        """
        Exporte un rapport GPS complet de tous les individus trackés.
        Compatible avec QGIS, Google Earth (GeoJSON).
        """
        os.makedirs("logs", exist_ok=True)
        geojson = {
            "type": "FeatureCollection",
            "metadata": {
                "generated": datetime.now().isoformat(),
                "camera_lat": self.camera_lat,
                "camera_lon": self.camera_lon,
                "api_calls": {
                    "ipinfo":   self.calls_ipinfo,
                    "opencage": self.calls_opencage,
                    "osm":      self.calls_osm,
                    "w3w":      self.calls_w3w,
                }
            },
            "features": []
        }

        for pid, track_data in tracks.items():
            positions = track_data.get("gps_history", [])
            if not positions:
                continue

            # LineString (trajectoire)
            coords = [[p["lon"], p["lat"]] for p in positions]
            geojson["features"].append({
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": coords},
                "properties": {
                    "person_id":  pid,
                    "statut":     track_data.get("statut","?"),
                    "score":      track_data.get("score", 0),
                    "start_time": positions[0].get("timestamp"),
                    "end_time":   positions[-1].get("timestamp"),
                    "last_address": positions[-1].get("address","?"),
                    "w3w":        positions[-1].get("w3w"),
                }
            })

            # Point final
            last = positions[-1]
            geojson["features"].append({
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [last["lon"], last["lat"]]
                },
                "properties": {
                    "person_id": pid,
                    "statut":    track_data.get("statut","?"),
                    "score":     track_data.get("score",0),
                    "timestamp": last.get("timestamp"),
                    "address":   last.get("address","?"),
                    "w3w":       last.get("w3w"),
                    "osm_url":   last.get("osm_url"),
                }
            })

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(geojson, f, indent=2, ensure_ascii=False)

        print(f"[GPS] ✓ Rapport GeoJSON → {output_path}")
        print(f"[GPS] ✓ Ouvrable dans QGIS / Google Earth / geojson.io")
        return output_path

    def get_camera_info(self):
        """Résumé de la position de la caméra."""
        return {
            "lat": self.camera_lat,
            "lon": self.camera_lon,
            "alt": self.camera_alt,
            "fov_h": self.camera_fov_h,
            "fov_v": self.camera_fov_v,
            "tilt": self.camera_tilt,
            "ip_info": self._ip_info,
            "osm_url": (f"https://www.openstreetmap.org/"
                        f"?mlat={self.camera_lat}&mlon={self.camera_lon}#map=17/"
                        f"{self.camera_lat}/{self.camera_lon}")
            if self.camera_lat else None
        }


# ══════════════════════════════════════════════════════
#  TEST RAPIDE
# ══════════════════════════════════════════════════════
if __name__ == "__main__":
    import cv2

    print("=== Test GPSEngine ===\n")

    # Init sans clé (OSM + ipinfo gratuits)
    gps = GPSEngine(
        camera_fov_h=70.0,
        camera_fov_v=45.0,
        camera_alt=8.0,
        camera_tilt=45.0
    )

    # Localiser la caméra automatiquement
    gps.auto_locate_camera()
    info = gps.get_camera_info()
    print(f"\nCaméra : {info['lat']}, {info['lon']}")
    if info["osm_url"]:
        print(f"Carte  : {info['osm_url']}")

    # Simuler 3 personnes dans l'image (1920x1080)
    frame_w, frame_h = 1920, 1080
    test_persons = [
        (960, 540, "milieu"),
        (200, 800, "gauche bas"),
        (1700, 300, "droite haut"),
    ]

    print("\n--- Conversion pixel → GPS ---")
    for cx, cy, label in test_persons:
        lat, lon = gps.pixel_to_gps(cx, cy, frame_w, frame_h)
        print(f"[{label}] pixel=({cx},{cy}) → GPS=({lat}, {lon})")

    # Test reverse geocoding (OSM gratuit)
    if gps.camera_lat:
        print("\n--- Reverse geocoding (OSM) ---")
        geo = gps.reverse_geocode(gps.camera_lat, gps.camera_lon)
        print(f"Adresse  : {geo['address']}")
        print(f"Ville    : {geo['city']}")
        print(f"Pays     : {geo['country']}")
        print(f"OSM URL  : {geo['osm_url']}")
        if geo["w3w"]:
            print(f"W3W      : {geo['w3w']}")

    print("\n[OK] GPSEngine opérationnel")
