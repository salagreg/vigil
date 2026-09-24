"""
Serveur API VIGIL (FastAPI).

Expose le flux video annote, un websocket temps reel pour les
detections, l'historique en base, et sert le tableau de bord statique.
"""

import asyncio
import json
from datetime import datetime, timezone

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
import os

from detector import Detector
from . import cloud_client, db

app = FastAPI(title="VIGIL")

# imgsz abaisse de 640 (defaut) a 416 : inference plus rapide, donc le cadre
# de detection colle mieux a un rocher qu'on deplace vite. Le flux camera
# lui-meme reste fluide quoi qu'il arrive (thread de capture separe) ; c'est
# la frequence de mise a jour du cadre qui depend de la vitesse d'inference.
detector = Detector(imgsz=416)
_main_loop = None
_ws_clients = set()


def _now_iso():
    return datetime.now(timezone.utc).isoformat()


def _on_detection(detection):
    """Appele depuis le thread du detecteur : ecrit en base puis
    planifie la diffusion websocket sur la boucle asyncio principale."""
    db.insert_event(
        detection.ts,
        detection.label,
        detection.confidence,
        detection.source,
        detection.object_id,
        detection.zone,
        detection.snapshot,
        detection.level,
    )

    message = json.dumps(
        {
            "type": "detection",
            "ts": detection.ts,
            "label": detection.label,
            "confidence": detection.confidence,
            "source": detection.source,
            "object_id": detection.object_id,
            "zone": detection.zone,
            "snapshot": detection.snapshot,
            "level": detection.level,
        }
    )
    if _main_loop is not None:
        asyncio.run_coroutine_threadsafe(_broadcast(message), _main_loop)
        # Tentative de synchronisation immediate (en plus de la boucle
        # periodique) pour que la demo reste reactive : sans effet si le
        # cloud est coupe ou desactive, l'evenement reste "non synchronise".
        asyncio.run_coroutine_threadsafe(cloud_client.sync_once(), _main_loop)


def _on_disappearance(object_id, duration_seconds):
    """Appele depuis le thread du detecteur quand un objet confirme quitte
    durablement le champ : complete son entree en base avec la duree totale
    de presence."""
    db.update_duration(object_id, duration_seconds)


async def _broadcast(message):
    dead = []
    for ws in list(_ws_clients):
        try:
            await ws.send_text(message)
        except Exception:
            dead.append(ws)
    for ws in dead:
        _ws_clients.discard(ws)


async def _heartbeat_loop():
    while True:
        await asyncio.sleep(2)
        message = json.dumps({"type": "heartbeat", "ts": _now_iso()})
        await _broadcast(message)


async def _radar_loop():
    """Diffuse la position de tous les objets actuellement dans le champ,
    pas seulement les nouvelles detections : le radar doit suivre un rocher
    en temps reel tant qu'il reste visible, sans repeter d'alerte."""
    while True:
        await asyncio.sleep(0.2)
        objects = detector.get_live_objects()
        message = json.dumps({"type": "radar", "ts": _now_iso(), "objects": objects})
        await _broadcast(message)


async def _cloud_status_loop():
    """Diffuse l'etat du cloud simule (joignable / mode autonome) toutes
    les secondes, pour que la pastille du dashboard reste a jour en temps
    reel sans que le pilote ait besoin de recharger la page."""
    while True:
        await asyncio.sleep(1)
        message = json.dumps({"type": "cloud_status", **cloud_client.status()})
        await _broadcast(message)


@app.on_event("startup")
async def on_startup():
    global _main_loop
    _main_loop = asyncio.get_event_loop()
    db.init_db()
    detector.add_listener(_on_detection)
    detector.add_disappearance_listener(_on_disappearance)
    detector.start()
    asyncio.create_task(_heartbeat_loop())
    asyncio.create_task(_radar_loop())
    asyncio.create_task(_cloud_status_loop())
    asyncio.create_task(cloud_client.sync_loop())


@app.on_event("shutdown")
async def on_shutdown():
    detector.stop()


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/cloud/status")
async def cloud_status():
    return cloud_client.status()


@app.post("/cloud/toggle")
async def cloud_toggle():
    cloud_client.toggle()
    return cloud_client.status()


@app.get("/camera/status")
async def camera_status():
    return {"connected": detector.is_camera_connected()}


@app.post("/camera/connect")
async def camera_connect():
    detector.connect_camera()
    return {"connected": detector.is_camera_connected()}


@app.post("/camera/disconnect")
async def camera_disconnect():
    detector.disconnect_camera()
    return {"connected": detector.is_camera_connected()}


@app.get("/events")
async def events(since: str | None = None):
    return db.get_events(since)


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    _ws_clients.add(websocket)
    try:
        while True:
            # On ne s'attend pas a recevoir de messages du client,
            # mais on doit lire pour detecter la deconnexion.
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        _ws_clients.discard(websocket)


async def _mjpeg_generator():
    boundary = b"--frame"
    while True:
        frame = detector.get_latest_jpeg()
        if frame is not None:
            yield (
                boundary
                + b"\r\nContent-Type: image/jpeg\r\nContent-Length: "
                + str(len(frame)).encode()
                + b"\r\n\r\n"
                + frame
                + b"\r\n"
            )
        await asyncio.sleep(1 / 15)


@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(
        _mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


snapshots_dir = os.path.join(os.path.dirname(__file__), "..", "snapshots")
os.makedirs(snapshots_dir, exist_ok=True)
app.mount("/snapshots", StaticFiles(directory=snapshots_dir), name="snapshots")

dashboard_dir = os.path.join(os.path.dirname(__file__), "..", "dashboard")
app.mount("/", StaticFiles(directory=dashboard_dir, html=True), name="dashboard")
