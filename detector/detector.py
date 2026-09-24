"""
Detecteur VIGIL : tourne dans son propre thread, jamais bloque par le
serveur HTTP. Capture la webcam en continu, fait tourner YOLO-World sur
chaque frame, dessine les detections, et garde en memoire la derniere
image annotee (consommee par le flux MJPEG /video_feed).

Chaque nouvelle detection est transmise aux "listeners" enregistres
(utilise par le serveur API pour l'ecrire en base et la pousser sur le
websocket), sans dependance directe a FastAPI.
"""

import colorsys
import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone

import cv2
import numpy as np
from ultralytics import YOLO

from .camera import RobustCamera

SNAPSHOTS_DIR = os.path.join(os.path.dirname(__file__), "..", "snapshots")

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
    level: str  # CRITIQUE, HAUT, MOYEN, ou BAS
    snapshot: str = None  # nom de fichier dans snapshots/, ou None si echec de sauvegarde


# Seuil separant une confiance "forte" d'une confiance "faible" pour le
# calcul du niveau d'alerte. Les scores reels sur ce vocabulaire tournent
# le plus souvent entre 3% et 30% : 15% coupe a peu pres au milieu.
HIGH_CONFIDENCE_THRESHOLD = 0.15


def _alert_level(zone, confidence):
    """Niveau d'alerte a partir de donnees deja disponibles (zone, confiance),
    sans nouveau capteur :
    - CRITIQUE : dans l'axe direct (CENTRE) avec une confiance forte.
    - HAUT : CENTRE avec confiance plus faible, ou lateral avec confiance forte.
    - MOYEN : lateral avec confiance plus faible.
    - BAS : uniquement HAUT/BAS, hors de la trajectoire directe.
    """
    is_high_conf = confidence >= HIGH_CONFIDENCE_THRESHOLD

    if zone == "CENTRE":
        return "CRITIQUE" if is_high_conf else "HAUT"

    if zone in ("HAUT", "BAS"):
        return "BAS"

    # Lateral : GAUCHE, DROITE, ou combinaisons (HAUT GAUCHE, BAS DROITE, ...)
    return "HAUT" if is_high_conf else "MOYEN"


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


def _same_object(box_a, box_b, iou_threshold, center_dist_factor, elapsed_seconds=0.0):
    """Un objet tenu a la main bouge un peu d'une frame a l'autre : se fier
    uniquement a l'IoU (chevauchement strict) faisait perdre l'identite du
    rocher (et donc sa couleur) au moindre mouvement. On accepte aussi une
    correspondance si les centres restent proches, relativement a la taille
    des boites.

    La detection ne traite plus forcement des frames consecutives (elle saute
    les images si elle est occupee) : plus il s'est ecoule de temps depuis le
    dernier passage sur cet objet, plus il a pu se deplacer entre-temps. La
    tolerance grandit donc avec elapsed_seconds, sinon un simple saut de
    frames suffit a faire perdre l'identite (et la couleur) d'un rocher qui
    n'a pourtant pas vraiment bouge plus vite qu'avant."""
    if _iou(box_a, box_b) >= iou_threshold:
        return True

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    acx, acy = (ax1 + ax2) / 2, (ay1 + ay2) / 2
    bcx, bcy = (bx1 + bx2) / 2, (by1 + by2) / 2
    center_dist = ((acx - bcx) ** 2 + (acy - bcy) ** 2) ** 0.5

    a_diag = ((ax2 - ax1) ** 2 + (ay2 - ay1) ** 2) ** 0.5
    b_diag = ((bx2 - bx1) ** 2 + (by2 - by1) ** 2) ** 0.5
    avg_diag = (a_diag + b_diag) / 2

    dynamic_factor = center_dist_factor * (1 + elapsed_seconds * 1.5)
    return avg_diag > 0 and center_dist <= avg_diag * dynamic_factor


def _color_for_id(object_id):
    """Couleur stable et distincte par objet (angle dore : deux ids
    consecutifs ont toujours des teintes bien differentes). Renvoie
    (r, g, b) 0-255, utilisable a la fois en BGR (OpenCV) et en CSS."""
    hue = (object_id * 137.508) % 360
    r, g, b = colorsys.hls_to_rgb(hue / 360, 0.55, 0.85)
    return int(r * 255), int(g * 255), int(b * 255)


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
        confirm_hits=2,
        detect_every_n_frames=1,
        imgsz=640,
    ):
        self.classes = classes or list(DEFAULT_CLASSES)
        self.camera_index = camera_index
        self.model_name = model_name
        # Seuil bas volontairement : le vocabulaire "rocher" donne des scores
        # naturellement faibles (souvent 3 a 15%) sur un objet reel tenu a la
        # main. Remonter ce seuil pour filtrer le bruit ratait aussi le vrai
        # rocher. A la place, le bruit de fond (reflets, coins d'objets) est
        # filtre par persistance : voir confirm_hits ci-dessous.
        self.conf_threshold = conf_threshold
        # Un objet doit etre revu (meme position) ce nombre de frames de
        # suite avant d'etre confirme et alerte. Le bruit ponctuel disparait
        # d'une frame a l'autre et n'atteint jamais ce seuil ; un vrai rocher,
        # lui, reste visible en continu.
        self.confirm_hits = confirm_hits
        self.detect_every_n_frames = max(1, detect_every_n_frames)
        # Si l'inference reste trop lente malgre la separation capture/detection
        # (voir plus bas), baisser cette valeur (ex: 416) accelere predict().
        self.imgsz = imgsz

        self._model = None
        self._camera = None
        self._camera_lock = threading.Lock()
        self._paused = False
        self._last_frame_shape = (480, 640, 3)

        self._thread = None
        self._capture_thread = None
        self._running = False

        # Image brute la plus recente, ecrite en continu par le thread de
        # capture (jamais de file d'attente qui grossit : une seule variable,
        # toujours ecrasee). frame_id incremente a chaque nouvelle image,
        # pour que le thread de detection sache s'il y a du neuf sans jamais
        # retraiter deux fois la meme image.
        self._raw_frame_lock = threading.Lock()
        self._raw_frame = None
        self._raw_frame_id = 0
        self._last_processed_frame_id = 0

        self._frame_lock = threading.Lock()
        self._latest_jpeg = None

        self._listeners = []
        self._disappearance_listeners = []
        self._frame_count = 0

        # Memoire spatiale : on retient la position des objets deja alertes
        # pour ne pas realerter en boucle sur le meme rocher qui reste dans
        # le champ. Un nouveau rocher, a une position differente, garde son
        # propre cadre et declenche bien sa propre alerte. Mise a jour a
        # chaque frame (meme les objets deja connus) pour alimenter le radar
        # en temps reel, protegee par un verrou car lue depuis l'API (autre
        # thread) via get_live_objects().
        self._objects_lock = threading.Lock()
        self._known_objects = []  # liste de dicts : id, box, label, hit_count, confirmed, ...
        self._iou_match_threshold = 0.05  # au-dela, on considere que c'est le meme objet
        # Tolerance resserree : quand il n'y a qu'un seul objet connu,
        # _find_known_object court-circuite cette comparaison (voir plus bas)
        # - elle ne sert donc plus qu'a distinguer plusieurs rochers reels
        # entre eux, ou une tolerance large ferait justement l'inverse de ce
        # qu'on veut (fusionner un deuxieme rocher dans l'identite du premier).
        self._center_dist_factor = 0.5
        self._confirmed_memory_seconds = 8.0  # objet confirme non revu depuis ce delai = oublie
        self._pending_memory_seconds = 1.0  # candidat non confirme non revu depuis ce delai = oublie (bruit)
        self._next_object_id = 1

    def add_listener(self, callback):
        """callback(Detection) est appele depuis le thread du detecteur
        a chaque nouvelle detection. Doit rester rapide (pas d'IO bloquante)."""
        self._listeners.append(callback)

    def add_disappearance_listener(self, callback):
        """callback(object_id, duration_seconds) appele quand un objet
        confirme quitte durablement le champ - utilise pour completer son
        entree en base avec sa duree totale de presence."""
        self._disappearance_listeners.append(callback)

    def start(self):
        if self._running:
            return
        os.makedirs(SNAPSHOTS_DIR, exist_ok=True)
        print(f"[DETECTOR] Chargement du modele {self.model_name} ...")
        self._model = YOLO(self.model_name)
        self._model.set_classes(self.classes)
        print(f"[DETECTOR] Vocabulaire actif : {self.classes}")

        self._camera = RobustCamera(self.camera_index)
        self._running = True
        self._thread = threading.Thread(target=self._detection_loop, daemon=True)
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        self._capture_thread.start()

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=5)
        if self._capture_thread is not None:
            self._capture_thread.join(timeout=5)
        if self._camera is not None:
            self._camera.release()

    def disconnect_camera(self):
        """Coupe la camera sans recharger le modele (instantane), pour un
        vrai bouton connecter/deconnecter en un clic."""
        with self._camera_lock:
            if self._camera is not None:
                self._camera.release()
                self._camera = None
            self._paused = True

    def connect_camera(self):
        with self._camera_lock:
            if self._camera is None:
                self._camera = RobustCamera(self.camera_index)
            self._paused = False

    def is_camera_connected(self):
        return not self._paused

    def get_latest_jpeg(self):
        with self._frame_lock:
            return self._latest_jpeg

    def get_live_objects(self):
        """Snapshot thread-safe des objets confirmes actuellement dans le
        champ, utilise par l'API pour alimenter le radar en temps reel. Les
        candidats pas encore confirmes (potentiel bruit) ne sont pas exposes."""
        with self._objects_lock:
            return [
                {
                    "object_id": obj["id"],
                    "label": obj["label"],
                    "zone": obj["zone"],
                    "rel_x": obj["rel_x"],
                    "rel_size": obj["rel_size"],
                    "level": obj["level"],
                }
                for obj in self._known_objects
                if obj["confirmed"]
            ]

    def _snapshot_draw_items(self):
        """Derniers cadres de detection connus (objets confirmes), lus par
        le thread de capture pour les dessiner sur l'image la plus recente -
        meme s'ils datent d'une fraction de seconde."""
        with self._objects_lock:
            return [
                {"box": obj["box"], "id": obj["id"], "label": obj["label"], "level": obj["level"]}
                for obj in self._known_objects
                if obj["confirmed"]
            ]

    def _emit(self, detection):
        for callback in self._listeners:
            try:
                callback(detection)
            except Exception as exc:
                print(f"[DETECTOR] Erreur dans un listener : {exc}")

    def _detection_loop(self):
        while self._running:
            try:
                self._detection_tick()
            except Exception as exc:
                # Une erreur inattendue dans la boucle ne doit jamais tuer le thread.
                print(f"[DETECTOR] Erreur inattendue dans la boucle de detection : {exc}")
                time.sleep(0.5)

    def _capture_loop(self):
        """Tourne dans son propre thread, aussi vite que la camera le permet.
        N'accumule jamais rien dans une file : la derniere image lue ecrase
        toujours la precedente. C'est ce qui empeche le retard de grandir -
        avant, la capture et la detection (lente) partageaient la meme
        boucle, et le tampon interne de la camera se remplissait pendant
        que l'inference tournait."""
        while self._running:
            try:
                self._capture_tick()
            except Exception as exc:
                print(f"[DETECTOR] Erreur inattendue dans la boucle de capture : {exc}")
                time.sleep(0.5)

    def _process_boxes(self, boxes, frame_width, frame_height):
        """Met a jour la memoire des objets et renvoie (to_draw, newly_confirmed).
        to_draw : tous les objets confirmes a dessiner sur cette frame.
        newly_confirmed : ceux qui viennent juste de passer la confirmation
        (utilise par _tick pour prendre une miniature et emettre l'alerte,
        une fois l'image annotee disponible)."""
        now = time.time()
        frame_area = frame_width * frame_height
        to_draw = []
        newly_confirmed = []

        # Le raccourci "un seul objet" (voir _find_known_object) ne doit
        # etre decide qu'une fois par frame, sur l'etat AVANT traitement -
        # sinon un deuxieme rocher qui apparait dans la meme image que le
        # premier profiterait aussi du raccourci (encore un seul objet connu
        # au moment ou on le traite) et se ferait fusionner avec lui.
        single_object_shortcut = len(self._known_objects) == 1 and len(boxes) == 1

        with self._objects_lock:
            for box in boxes:
                box_xyxy = tuple(float(v) for v in box.xyxy[0])
                x1, y1, x2, y2 = box_xyxy
                rel_x = ((x1 + x2) / 2) / frame_width
                rel_y = ((y1 + y2) / 2) / frame_height
                rel_size = ((x2 - x1) * (y2 - y1)) / frame_area if frame_area > 0 else 0.0
                label = _label_fr(self._model.names[int(box.cls[0])])

                known = self._find_known_object(box_xyxy, now, single_object_shortcut)
                if known is None:
                    known = {
                        "id": self._next_object_id,
                        "box": box_xyxy,
                        "label": label,
                        "zone": _zone_label(box_xyxy, frame_width, frame_height),
                        "hit_count": 0,
                        "confirmed": False,
                        "first_seen": now,
                    }
                    self._next_object_id += 1
                    self._known_objects.append(known)

                known.update(
                    box=box_xyxy,
                    last_seen=now,
                    rel_x=rel_x,
                    rel_y=rel_y,
                    rel_size=rel_size,
                    hit_count=known["hit_count"] + 1,
                )

                if not known["confirmed"] and known["hit_count"] >= self.confirm_hits:
                    known["confirmed"] = True
                    confidence = float(box.conf[0])
                    known["level"] = _alert_level(known["zone"], confidence)
                    newly_confirmed.append(
                        {
                            "box": box_xyxy,
                            "id": known["id"],
                            "label": known["label"],
                            "zone": known["zone"],
                            "confidence": confidence,
                            "level": known["level"],
                        }
                    )

                if known["confirmed"]:
                    to_draw.append(
                        {"box": box_xyxy, "id": known["id"], "label": known["label"], "level": known["level"]}
                    )

            self._forget_stale_objects(now)

        return to_draw, newly_confirmed

    def _draw_boxes(self, frame, to_draw):
        """Dessine chaque objet confirme avec une couleur qui lui est propre
        (stable dans le temps), pour que le pilote associe visuellement le
        meme rocher entre la video, le radar et l'historique."""
        annotated = frame.copy()
        for item in to_draw:
            x1, y1, x2, y2 = (int(v) for v in item["box"])
            r, g, b = _color_for_id(item["id"])
            color_bgr = (b, g, r)

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color_bgr, 2)
            caption = f"{item['label']} #{item['id']} [{item['level']}]"
            (tw, th), _ = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 6, y1), color_bgr, -1)
            cv2.putText(
                annotated, caption, (x1 + 3, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA
            )
        return annotated

    def _find_known_object(self, box_xyxy, now, use_shortcut):
        # Cas le plus courant en demo : un seul rocher, seul dans le champ.
        # La ou il se trouve dans l'image, ca ne peut etre que lui - aucune
        # ambiguite possible. La comparaison de position (fragile face a un
        # objet tenu a la main, avec une detection qui ne tourne pas a
        # cadence fixe) ne sert qu'a distinguer plusieurs rochers reellement
        # distincts. use_shortcut est decide une fois par frame par l'appelant
        # (pas ici) pour ne jamais fusionner un deuxieme rocher qui apparait
        # dans la meme image que le premier.
        if use_shortcut:
            return self._known_objects[0]

        for known in self._known_objects:
            elapsed = now - known["last_seen"]
            if _same_object(box_xyxy, known["box"], self._iou_match_threshold, self._center_dist_factor, elapsed):
                return known
        return None

    def _forget_stale_objects(self, now):
        kept = []
        for known in self._known_objects:
            timeout = self._confirmed_memory_seconds if known["confirmed"] else self._pending_memory_seconds
            if now - known["last_seen"] <= timeout:
                kept.append(known)
            elif known["confirmed"]:
                duration = known["last_seen"] - known["first_seen"]
                for callback in self._disappearance_listeners:
                    try:
                        callback(known["id"], duration)
                    except Exception as exc:
                        print(f"[DETECTOR] Erreur dans un listener de disparition : {exc}")
        self._known_objects = kept

    def _save_snapshot(self, annotated_frame, item):
        try:
            x1, y1, x2, y2 = (int(v) for v in item["box"])
            h, w = annotated_frame.shape[:2]
            pad = 20
            crop = annotated_frame[max(0, y1 - pad) : min(h, y2 + pad), max(0, x1 - pad) : min(w, x2 + pad)]
            if crop.size == 0:
                return None
            filename = f"obj{item['id']}_{int(time.time() * 1000)}.jpg"
            cv2.imwrite(os.path.join(SNAPSHOTS_DIR, filename), crop)
            return filename
        except Exception as exc:
            print(f"[DETECTOR] Erreur en sauvegardant la miniature : {exc}")
            return None

    def _placeholder_frame(self):
        h, w = self._last_frame_shape[:2]
        frame = np.zeros((h, w, 3), dtype="uint8")
        text = "CAMERA DECONNECTEE"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        cv2.putText(
            frame, text, ((w - tw) // 2, (h + th) // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (90, 90, 90), 2, cv2.LINE_AA
        )
        return frame

    def _capture_tick(self):
        """Lit une image, la rend immediatement disponible pour la detection
        (variable partagee unique) et publie tout de suite le flux MJPEG
        avec les derniers cadres de detection connus dessus - sans jamais
        attendre la fin d'une inference."""
        with self._camera_lock:
            camera = self._camera

        if camera is None:
            self._raw_frame = None
            with self._frame_lock:
                ok, buffer = cv2.imencode(".jpg", self._placeholder_frame())
                if ok:
                    self._latest_jpeg = buffer.tobytes()
            time.sleep(0.2)
            return

        ok, frame = camera.read()
        if not ok or frame is None:
            time.sleep(0.02)
            return

        self._last_frame_shape = frame.shape

        with self._raw_frame_lock:
            self._raw_frame = frame
            self._raw_frame_id += 1

        annotated = self._draw_boxes(frame, self._snapshot_draw_items())
        ok, buffer = cv2.imencode(".jpg", annotated)
        if ok:
            with self._frame_lock:
                self._latest_jpeg = buffer.tobytes()

    def _detection_tick(self):
        """Traite la derniere image disponible, quitte a en sauter certaines
        si l'inference est plus lente que la cadence de la camera - jamais
        l'inverse (jamais de retard qui s'accumule)."""
        with self._raw_frame_lock:
            frame = self._raw_frame
            frame_id = self._raw_frame_id

        if frame is None or frame_id == self._last_processed_frame_id:
            time.sleep(0.01)
            return
        self._last_processed_frame_id = frame_id

        self._frame_count += 1
        if self._frame_count % self.detect_every_n_frames != 0:
            return

        frame_height, frame_width = frame.shape[:2]
        results = self._model.predict(frame, conf=self.conf_threshold, imgsz=self.imgsz, verbose=False)

        to_draw, newly_confirmed = self._process_boxes(results[0].boxes, frame_width, frame_height)

        # La miniature est prise sur cette meme frame (avec le cadre colore),
        # une fois qu'on est sur que l'objet est confirme.
        for item in newly_confirmed:
            annotated = self._draw_boxes(frame, to_draw)
            snapshot = self._save_snapshot(annotated, item)
            detection = Detection(
                ts=_now_iso(),
                label=item["label"],
                confidence=item["confidence"],
                source="yolo",
                object_id=item["id"],
                zone=item["zone"],
                level=item["level"],
                snapshot=snapshot,
            )
            self._emit(detection)
