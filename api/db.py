"""
Acces SQLite pour les evenements de detection.

Une seule connexion partagee, mode WAL (lectures concurrentes possibles
pendant une ecriture), et un verrou pour garantir une seule ecriture a
la fois comme demande par le cahier des charges.
"""

import os
import sqlite3
import threading

DB_PATH = os.path.join(os.path.dirname(__file__), "vigil.db")

_write_lock = threading.Lock()
_connection = None


def get_connection():
    global _connection
    if _connection is None:
        _connection = sqlite3.connect(DB_PATH, check_same_thread=False)
        _connection.execute("PRAGMA journal_mode=WAL;")
        _connection.row_factory = sqlite3.Row
    return _connection


def init_db():
    conn = get_connection()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            label TEXT NOT NULL,
            confidence REAL NOT NULL,
            source TEXT NOT NULL,
            object_id INTEGER,
            zone TEXT,
            snapshot TEXT,
            duration_seconds REAL,
            level TEXT,
            synced INTEGER NOT NULL DEFAULT 0
        )
        """
    )
    # Migration douce pour une base creee avant l'ajout de ces colonnes.
    for column, coltype in [
        ("object_id", "INTEGER"),
        ("zone", "TEXT"),
        ("snapshot", "TEXT"),
        ("duration_seconds", "REAL"),
        ("level", "TEXT"),
        ("synced", "INTEGER NOT NULL DEFAULT 0"),
    ]:
        try:
            conn.execute(f"ALTER TABLE events ADD COLUMN {column} {coltype}")
        except sqlite3.OperationalError:
            pass  # colonne deja presente
    conn.commit()


def insert_event(ts, label, confidence, source, object_id, zone, snapshot, level):
    conn = get_connection()
    with _write_lock:
        cursor = conn.execute(
            "INSERT INTO events (ts, label, confidence, source, object_id, zone, snapshot, level) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (ts, label, confidence, source, object_id, zone, snapshot, level),
        )
        conn.commit()
        return cursor.lastrowid


LEVEL_PRIORITY = {"CRITIQUE": 0, "HAUT": 1, "MOYEN": 2, "BAS": 3}


def get_unsynced_events():
    """Evenements jamais envoyes au cloud simule, tries CRITIQUE d'abord
    (puis par ordre chronologique) pour la resynchronisation."""
    conn = get_connection()
    rows = conn.execute(
        "SELECT id, ts, label, confidence, source, object_id, zone, snapshot, duration_seconds, level "
        "FROM events WHERE synced = 0"
    ).fetchall()
    events = [dict(row) for row in rows]
    events.sort(key=lambda e: (LEVEL_PRIORITY.get(e["level"], 99), e["ts"]))
    return events


def mark_synced(event_id):
    conn = get_connection()
    with _write_lock:
        conn.execute("UPDATE events SET synced = 1 WHERE id = ?", (event_id,))
        conn.commit()


def update_duration(object_id, duration_seconds):
    conn = get_connection()
    with _write_lock:
        conn.execute(
            "UPDATE events SET duration_seconds = ? WHERE object_id = ?",
            (duration_seconds, object_id),
        )
        conn.commit()


def get_events(since=None):
    conn = get_connection()
    columns = "id, ts, label, confidence, source, object_id, zone, snapshot, duration_seconds, level, synced"
    if since:
        rows = conn.execute(
            f"SELECT {columns} FROM events WHERE ts >= ? ORDER BY ts DESC",
            (since,),
        ).fetchall()
    else:
        rows = conn.execute(f"SELECT {columns} FROM events ORDER BY ts DESC").fetchall()
    return [dict(row) for row in rows]


def count_events_since(since):
    conn = get_connection()
    row = conn.execute(
        "SELECT COUNT(*) AS n FROM events WHERE ts >= ?", (since,)
    ).fetchone()
    return row["n"]
