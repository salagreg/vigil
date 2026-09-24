"""
Client de synchronisation vers le cloud simule (vigil/cloud).

Boucle periodique : verifie la joignabilite reelle du cloud, et si actif
+ joignable, envoie les evenements pas encore synchronises (les plus
critiques d'abord). Si le cloud est coupe (ou desactive volontairement
via /cloud/toggle), les evenements restent simplement marques "non
synchronise" en base au lieu de se perdre - ils repartent au prochain
passage ou le cloud redevient joignable.

Le toggle est un vrai blocage des requetes : quand desactive, aucun appel
reseau n'est meme tente, pas seulement une pastille visuelle.
"""

import asyncio
import time
from datetime import datetime, timezone

import httpx

from . import db

CLOUD_URL = "http://127.0.0.1:9000"
CHECK_INTERVAL_SECONDS = 2.0

_enabled = True
_reachable = True
_autonomous_since = None  # epoch, ou None si actuellement connecte


def _now_iso():
    return datetime.now(timezone.utc).isoformat()


def toggle():
    """Coupe ou retablit reellement les appels vers le cloud simule."""
    global _enabled, _autonomous_since
    _enabled = not _enabled
    if not _enabled and _autonomous_since is None:
        _autonomous_since = time.time()
    return _enabled


def status():
    autonomous = (not _enabled) or (not _reachable)
    since_iso = None
    elapsed_seconds = None
    if autonomous and _autonomous_since is not None:
        since_iso = datetime.fromtimestamp(_autonomous_since, tz=timezone.utc).isoformat()
        elapsed_seconds = time.time() - _autonomous_since
    return {
        "enabled": _enabled,
        "reachable": _reachable,
        "autonomous": autonomous,
        "autonomous_since": since_iso,
        "elapsed_seconds": elapsed_seconds,
    }


async def _mark_unreachable():
    global _reachable, _autonomous_since
    _reachable = False
    if _autonomous_since is None:
        _autonomous_since = time.time()


async def _mark_reachable():
    global _reachable, _autonomous_since
    _reachable = True
    _autonomous_since = None


async def sync_once():
    """Un passage de synchronisation. Appele en boucle par l'API, et
    directement apres chaque nouvelle detection pour une demo reactive."""
    global _autonomous_since

    if not _enabled:
        # Blocage reel : on ne tente meme pas la requete de sante.
        if _autonomous_since is None:
            _autonomous_since = time.time()
        return

    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            response = await client.get(f"{CLOUD_URL}/health")
            if response.status_code != 200:
                await _mark_unreachable()
                return
    except Exception:
        await _mark_unreachable()
        return

    await _mark_reachable()

    pending = db.get_unsynced_events()
    if not pending:
        return

    async with httpx.AsyncClient(timeout=3.0) as client:
        for event in pending:
            try:
                response = await client.post(f"{CLOUD_URL}/events", json=event)
                if response.status_code == 200:
                    db.mark_synced(event["id"])
                else:
                    break
            except Exception:
                await _mark_unreachable()
                break


async def sync_loop():
    while True:
        try:
            await sync_once()
        except Exception as exc:
            print(f"[CLOUD] Erreur inattendue dans la boucle de synchronisation : {exc}")
        await asyncio.sleep(CHECK_INTERVAL_SECONDS)
