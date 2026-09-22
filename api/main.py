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
from . import db

app = FastAPI(title="VIGIL")

detector = Detector()
_main_loop = None
_ws_clients = set()


def _now_iso():
    return datetime.now(timezone.utc).isoformat()


def _on_detection(detection):
    """Appele depuis le thread du detecteur : ecrit en base puis
    planifie la diffusion websocket sur la boucle asyncio principale."""
    db.insert_event(
        detection.ts, detection.label, detection.confidence, detection.source, detection.object_id, detection.zone
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
        }
    )
    if _main_loop is not None:
        asyncio.run_coroutine_threadsafe(_broadcast(message), _main_loop)


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


@app.on_event("startup")
async def on_startup():
    global _main_loop
    _main_loop = asyncio.get_event_loop()
    db.init_db()
    detector.add_listener(_on_detection)
    detector.start()
    asyncio.create_task(_heartbeat_loop())
    asyncio.create_task(_radar_loop())


@app.on_event("shutdown")
async def on_shutdown():
    detector.stop()


@app.get("/health")
async def health():
    return {"status": "ok"}


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


dashboard_dir = os.path.join(os.path.dirname(__file__), "..", "dashboard")
app.mount("/", StaticFiles(directory=dashboard_dir, html=True), name="dashboard")
