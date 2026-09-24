"""
Serveur "cloud" simule pour VIGIL.

Tourne en local sur un port different du serveur principal (ex: 9000),
pour simuler un service distant sans dependance a un hebergement externe.
Recoit et stocke les evenements confirmes que le serveur principal
synchronise, et permet de les relire pour verifier ce qui est arrive.

Stockage en memoire uniquement (pas de base persistante) : c'est un
simulateur pour la demo, pas un vrai service cloud.
"""

from fastapi import FastAPI

app = FastAPI(title="VIGIL Cloud (simule)")

_events = []
_seen_ids = set()


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/events")
async def receive_event(event: dict):
    # Dedoublonnage par l'id de l'evenement source (cote serveur principal) :
    # si le meme evenement est renvoye deux fois (retry apres coupure), on
    # ne le stocke qu'une fois.
    event_id = event.get("id")
    if event_id is None or event_id not in _seen_ids:
        _events.append(event)
        if event_id is not None:
            _seen_ids.add(event_id)
    return {"status": "ok", "stored": len(_events)}


@app.get("/events")
async def list_events():
    return _events
