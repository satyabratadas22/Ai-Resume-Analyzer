"""SQLite storage for analysis history."""
import sqlite3
from datetime import datetime, timezone

DB_PATH = "analyses.db"


def _conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with _conn() as c:
        c.execute(
            """CREATE TABLE IF NOT EXISTS analyses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                backend TEXT NOT NULL,
                score REAL NOT NULL,
                matched TEXT NOT NULL,
                missing TEXT NOT NULL
            )"""
        )


def save_analysis(backend, score, matched, missing) -> int:
    with _conn() as c:
        cur = c.execute(
            "INSERT INTO analyses (created_at, backend, score, matched, missing) VALUES (?,?,?,?,?)",
            (datetime.now(timezone.utc).isoformat(), backend, score,
             ", ".join(matched), ", ".join(missing)),
        )
        return cur.lastrowid


def list_analyses(limit: int = 20):
    with _conn() as c:
        rows = c.execute(
            "SELECT * FROM analyses ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]
