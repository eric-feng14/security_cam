#!/usr/bin/env python3
"""
face_db.py — SQLite storage for enrolled face embeddings.

Each enrolled person can have several embedding samples (one row each),
which makes matching more robust to angle/lighting. Embeddings are SFace
128-d float32 vectors stored as raw bytes.
"""

import os
import sqlite3
import numpy as np

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "faces.db")


def _connect(db_path: str = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS faces (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            name       TEXT    NOT NULL,
            embedding  BLOB    NOT NULL,
            created_at REAL    NOT NULL DEFAULT (strftime('%s','now'))
        )
        """
    )
    return conn


def init_db(db_path: str = DB_PATH) -> None:
    """Create the database/table if they don't exist yet."""
    _connect(db_path).close()


def add_embedding(name: str, embedding: np.ndarray, db_path: str = DB_PATH) -> None:
    """Store one 128-d embedding sample for `name`."""
    blob = np.asarray(embedding, dtype=np.float32).flatten().tobytes()
    conn = _connect(db_path)
    with conn:
        conn.execute(
            "INSERT INTO faces (name, embedding) VALUES (?, ?)", (name, blob)
        )
    conn.close()


def load_all(db_path: str = DB_PATH) -> list[tuple[str, np.ndarray]]:
    """Return every stored (name, embedding) pair."""
    conn = _connect(db_path)
    rows = conn.execute("SELECT name, embedding FROM faces").fetchall()
    conn.close()
    return [
        (name, np.frombuffer(blob, dtype=np.float32)) for name, blob in rows
    ]


def counts(db_path: str = DB_PATH) -> dict[str, int]:
    """Return {name: number_of_samples} for all enrolled people."""
    conn = _connect(db_path)
    rows = conn.execute(
        "SELECT name, COUNT(*) FROM faces GROUP BY name ORDER BY name"
    ).fetchall()
    conn.close()
    return {name: n for name, n in rows}


def delete_person(name: str, db_path: str = DB_PATH) -> int:
    """Remove all samples for `name`. Returns number of rows deleted."""
    conn = _connect(db_path)
    with conn:
        cur = conn.execute("DELETE FROM faces WHERE name = ?", (name,))
    conn.close()
    return cur.rowcount


if __name__ == "__main__":
    # Quick CLI: `python face_db.py` lists enrolled people.
    init_db()
    people = counts()
    if not people:
        print("No faces enrolled yet. Run: python enroll_faces.py \"Your Name\"")
    else:
        print("Enrolled people:")
        for name, n in people.items():
            print(f"  {name:20s} {n} sample(s)")
