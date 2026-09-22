"""
Detecteur VIGIL : tourne dans son propre thread, jamais bloque par le
serveur HTTP. Capture la webcam en continu, fait tourner YOLO-World sur
chaque frame, dessine les detections, et garde en memoire la derniere
image annotee (consommee par le flux MJPEG /video_feed).

Chaque nouvelle detection est transmise aux "listeners" enregistres
(utilise par le serveur API pour l'ecrire en base et la pousser sur le
websocket), sans dependance directe a FastAPI.
"""

import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone

import cv2
from ultralytics import YOLO

from .camera import RobustCamera

# Uniquement des objets du theme "debris spatial" : le pilote ne veut voir
# que ca, pas les personnes, outils ou autres objets de la scene.
# "rock" seul ne matchait quasiment jamais en test reel (webcam, objet tenu
# a la main) : cette combinaison, validee en conditions reelles, marche
# beaucoup mieux pour le meme objet.
DEFAULT_CLASSES = ["stone", "pebble", "grey rock", "small rock", "piece of rock"]

# YOLO-World fonctionne mieux avec un vocabulaire anglais (CLIP), mais le
# dashboard doit afficher un libelle clair en francais pour le pilote.
LABEL_FR = {
    "stone": "Rocher",
    "pebble": "Rocher",
    "grey rock": "Rocher",
    "small rock": "Rocher",
    "piece of rock": "Rocher",
    "meteorite": "Meteorite",
    "asteroid": "Asteroide",
}


def _label_fr(label):
    return LABEL_FR.get(label, label.capitalize())


@dataclass
class Detection:
    ts: str
    label: str
    confidence: float
    source: str  # "yolo"
    object_id: int
    zone: str  # position dans le champ : GAUCHE, DROITE, HAUT, BAS, CENTRE, ou combinaison


def _now_iso():
    return datetime.now(timezone.utc).isoformat()


def _iou(box_a, box_b):
    """Intersection-over-union entre deux boites (x1, y1, x2, y2)."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    inter_area = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area

    return inter_area / union if union > 0 else 0.0


def _zone_label(box_xyxy, frame_width, frame_height):
    """Position de l'objet dans le champ de la camera, utile au pilote pour
    savoir de quel cote devier (gauche/droite/haut/bas)."""
    x1, y1, x2, y2 = box_xyxy
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

    if cx < frame_width / 3:
        horizontal = "GAUCHE"
    elif cx > frame_width * 2 / 3:
        horizontal = "DROITE"
    else:
        horizontal = None

    if cy < frame_height / 3:
        vertical = "HAUT"
    elif cy > frame_height * 2 / 3:
        vertical = "BAS"
    else:
        vertical = None

    if horizontal and vertical:
        return f"{vertical} {horizontal}"
    return horizontal or vertical or "CENTRE"


class Detector:
    def __init__(
        self,
        classes=None,
        camera_index=0,
        model_name="yolov8s-worldv2.pt",
        conf_threshold=0.03,
        detect_every_n_frames=1,
    ):
        self.classes = classes or list(DEFAULT_CLASSES)
        self.camera_index = camera_index
        self.model_name = model_name
        self.conf_threshold = conf_threshold
        self.detect_every_n_frames = max(1, detect_every_n_frames)

        self._model = None
        self._camera = None

        self._thread = None
        self._running = False

        self._frame_lock = threading.Lock()
        self._latest_jpeg = None

        self._listeners = []
        self._frame_count = 0

        # Memoire spatiale : on retient la position des objets deja alertes
        # pour ne pas realerter en boucle sur le meme rocher qui reste dans
        # le champ. Un nouveau rocher, a une position differente, garde son
        # propre cadre et declenche bien sa propre alerte. Mise a jour a
        # chaque frame (meme les objets deja connus) pour alimenter le radar
        # en temps reel, protegee par un verrou car lue depuis l'API (autre
        # thread) via get_live_objects().
        self._objects_lock = threading.Lock()
        self._known_objects = []  # liste de dicts : id, box, label, rel_x, rel_y, rel_size, last_seen
        self._iou_match_threshold = 0.2  # au-dela, on considere que c'est le meme objet
        self._object_memory_seconds = 8.0  # objet non revu depuis ce delai = oublie
        self._next_object_id = 1

    def add_listener(self, callback):
        """callback(Detection) est appele depuis le thread du detecteur
        a chaque nouvelle detection. Doit rester rapide (pas d'IO bloquante)."""
        self._listeners.append(callback)

    def start(self):
        if self._running:
            return
        print(f"[DETECTOR] Chargement du modele {self.model_name} ...")
        self._model = YOLO(self.model_name)
        self._model.set_classes(self.classes)
        print(f"[DETECTOR] Vocabulaire actif : {self.classes}")

        self._camera = RobustCamera(self.camera_index)
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=5)
        if self._camera is not None:
            self._camera.release()

    def get_latest_jpeg(self):
        with self._frame_lock:
            return self._latest_jpeg

    def get_live_objects(self):
        """Snapshot thread-safe des objets actuellement dans le champ,
        utilise par l'API pour alimenter le radar en temps reel."""
        with self._objects_lock:
            return [
                {
                    "object_id": obj["id"],
                    "label": obj["label"],
                    "zone": obj["zone"],
                    "rel_x": obj["rel_x"],
                    "rel_size": obj["rel_size"],
                }
                for obj in self._known_objects
            ]

    def _emit(self, detection):
        for callback in self._listeners:
            try:
                callback(detection)
            except Exception as exc:
                print(f"[DETECTOR] Erreur dans un listener : {exc}")

    def _run_loop(self):
        while self._running:
            try:
                self._tick()
            except Exception as exc:
                # Une erreur inattendue dans la boucle ne doit jamais tuer le thread.
                print(f"[DETECTOR] Erreur inattendue dans la boucle : {exc}")
                time.sleep(0.5)

    def _process_boxes(self, boxes, frame_width, frame_height):
        now = time.time()
        frame_area = frame_width * frame_height

        with self._objects_lock:
            for box in boxes:
                box_xyxy = tuple(float(v) for v in box.xyxy[0])
                x1, y1, x2, y2 = box_xyxy
                rel_x = ((x1 + x2) / 2) / frame_width
                rel_y = ((y1 + y2) / 2) / frame_height
                rel_size = ((x2 - x1) * (y2 - y1)) / frame_area if frame_area > 0 else 0.0
                label = _label_fr(self._model.names[int(box.cls[0])])

                known = self._find_known_object(box_xyxy)
                if known is not None:
                    known.update(box=box_xyxy, last_seen=now, rel_x=rel_x, rel_y=rel_y, rel_size=rel_size)
                    continue

                object_id = self._next_object_id
                self._next_object_id += 1
                zone = _zone_label(box_xyxy, frame_width, frame_height)
                self._known_objects.append(
                    {
                        "id": object_id,
                        "box": box_xyxy,
                        "last_seen": now,
                        "label": label,
                        "zone": zone,
                        "rel_x": rel_x,
                        "rel_y": rel_y,
                        "rel_size": rel_size,
                    }
                )

                confidence = float(box.conf[0])
                detection = Detection(
                    ts=_now_iso(),
                    label=label,
                    confidence=confidence,
                    source="yolo",
                    object_id=object_id,
                    zone=zone,
                )
                self._emit(detection)

            self._forget_stale_objects(now)

    def _find_known_object(self, box_xyxy):
        for known in self._known_objects:
            if _iou(box_xyxy, known["box"]) >= self._iou_match_threshold:
                return known
        return None

    def _forget_stale_objects(self, now):
        self._known_objects = [
            known for known in self._known_objects if now - known["last_seen"] <= self._object_memory_seconds
        ]

    def _tick(self):
        ok, frame = self._camera.read()
        if not ok or frame is None:
            time.sleep(0.05)
            return

        self._frame_count += 1
        annotated = frame

        if self._frame_count % self.detect_every_n_frames == 0:
            results = self._model.predict(frame, conf=self.conf_threshold, verbose=False)
            annotated = results[0].plot()
            frame_height, frame_width = frame.shape[:2]
            self._process_boxes(results[0].boxes, frame_width, frame_height)

        ok, buffer = cv2.imencode(".jpg", annotated)
        if ok:
            with self._frame_lock:
                self._latest_jpeg = buffer.tobytes()
