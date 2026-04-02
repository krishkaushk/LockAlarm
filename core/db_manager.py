"""
db_manager.py — all SQLite reads and writes for LockAlarm.

Design rules:
- Each caller creates its own connection (sqlite3 connections are NOT thread-safe).
- All public functions accept an optional `db_path` so tests can use a temp file.
- Schema is created automatically on first connection via _ensure_schema().
"""

import sqlite3
from pathlib import Path
from typing import Optional

# Default database location — sits in the data/ folder next to this package.
_DEFAULT_DB = Path(__file__).parent.parent / "data" / "lockalarm.db"


def _connect(db_path: Path) -> sqlite3.Connection:
    """Open a connection and ensure the schema exists."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row   # rows behave like dicts: row["focus_score"]
    conn.execute("PRAGMA foreign_keys = ON")  # enforce session_id references
    _ensure_schema(conn)
    return conn


def _ensure_schema(conn: sqlite3.Connection) -> None:
    """Create tables if they don't already exist. Safe to call every startup."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS sessions (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            started_at        TEXT    NOT NULL,   -- ISO 8601 e.g. "2026-04-01T09:00:00"
            ended_at          TEXT    NOT NULL,
            total_seconds     INTEGER NOT NULL,
            focus_seconds     INTEGER NOT NULL,
            slack_seconds     INTEGER NOT NULL,
            distraction_count INTEGER NOT NULL,
            focus_score       REAL    NOT NULL    -- 0.0 – 100.0
        );

        CREATE TABLE IF NOT EXISTS distractions (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id       INTEGER NOT NULL REFERENCES sessions(id),
            occurred_at      TEXT    NOT NULL,   -- ISO 8601 timestamp
            duration_seconds REAL    NOT NULL
        );
    """)
    conn.commit()


# ---------------------------------------------------------------------------
# Write path (called from CameraWorker thread — creates its own connection)
# ---------------------------------------------------------------------------

def save_session(
    started_at: str,
    ended_at: str,
    total_seconds: int,
    focus_seconds: int,
    slack_seconds: int,
    distraction_count: int,
    focus_score: float,
    distractions: list[dict],          # [{"occurred_at": str, "duration_seconds": float}, ...]
    db_path: Path = _DEFAULT_DB,
) -> int:
    """
    Persist one completed session plus all its distraction events.
    Returns the new session's id so callers can reference it.
    """
    conn = _connect(db_path)
    try:
        cursor = conn.execute(
            """
            INSERT INTO sessions
                (started_at, ended_at, total_seconds, focus_seconds,
                 slack_seconds, distraction_count, focus_score)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (started_at, ended_at, total_seconds, focus_seconds,
             slack_seconds, distraction_count, focus_score),
        )
        session_id = cursor.lastrowid

        # Bulk-insert all distraction events for this session.
        conn.executemany(
            "INSERT INTO distractions (session_id, occurred_at, duration_seconds) VALUES (?, ?, ?)",
            [(session_id, d["occurred_at"], d["duration_seconds"]) for d in distractions],
        )
        conn.commit()
        return session_id
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Read path (called from UI thread — creates its own connection)
# ---------------------------------------------------------------------------

def get_recent_sessions(limit: int = 30, db_path: Path = _DEFAULT_DB) -> list[dict]:
    """
    Return the most recent `limit` sessions, oldest-first, for the history line graph.
    Each item has: id, started_at, focus_score, total_seconds, distraction_count.
    """
    conn = _connect(db_path)
    try:
        rows = conn.execute(
            """
            SELECT id, started_at, focus_score, total_seconds, distraction_count
            FROM sessions
            ORDER BY started_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        # Reverse so the line graph reads left-to-right chronologically.
        return [dict(r) for r in reversed(rows)]
    finally:
        conn.close()


def get_daily_averages(db_path: Path = _DEFAULT_DB) -> list[dict]:
    """
    Return average focus score per calendar day for the heatmap.
    Each item has: day (YYYY-MM-DD), avg_score.
    """
    conn = _connect(db_path)
    try:
        rows = conn.execute(
            """
            SELECT date(started_at) AS day, AVG(focus_score) AS avg_score
            FROM sessions
            GROUP BY day
            ORDER BY day
            """
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_distractions_for_session(
    session_id: int, db_path: Path = _DEFAULT_DB
) -> list[dict]:
    """
    Return all distraction events for one session, ordered by time.
    Each item has: occurred_at, duration_seconds.
    Used by the post-session distraction timeline chart.
    """
    conn = _connect(db_path)
    try:
        rows = conn.execute(
            """
            SELECT occurred_at, duration_seconds
            FROM distractions
            WHERE session_id = ?
            ORDER BY occurred_at
            """,
            (session_id,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()
