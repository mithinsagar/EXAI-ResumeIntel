"""
EXAI-ResumeIntel: SQLite analysis history persistence
=====================================================

Module: app.backend.history_store
Author: Mithin Sagar S
GitHub: https://github.com/mithinsagar
Institution: Vellore Institute of Technology (VIT)
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional


APP_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = APP_DIR.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = PROCESSED_DIR / "analysis_history.db"


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_history_db() -> None:
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS analysis_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                file_name TEXT NOT NULL,
                role TEXT NOT NULL,
                match_percentage REAL NOT NULL,
                semantic_similarity REAL NOT NULL,
                payload_json TEXT NOT NULL
            )
            """
        )
        conn.commit()


def save_history_entry(
    created_at: str,
    file_name: str,
    role: str,
    match_percentage: float,
    semantic_similarity: float,
    payload: Dict[str, Any],
) -> int:
    payload_json = json.dumps(payload, ensure_ascii=True)
    with _connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO analysis_history
            (created_at, file_name, role, match_percentage, semantic_similarity, payload_json)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                created_at,
                file_name,
                role,
                float(match_percentage),
                float(semantic_similarity),
                payload_json,
            ),
        )
        conn.commit()
        return int(cur.lastrowid)


def list_history_entries(limit: int = 50) -> List[Dict[str, Any]]:
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT id, created_at, file_name, role, match_percentage, semantic_similarity
            FROM analysis_history
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_history_entry(entry_id: int) -> Optional[Dict[str, Any]]:
    with _connect() as conn:
        row = conn.execute(
            """
            SELECT id, created_at, file_name, role, match_percentage, semantic_similarity, payload_json
            FROM analysis_history
            WHERE id = ?
            """,
            (int(entry_id),),
        ).fetchone()
    if row is None:
        return None
    item = dict(row)
    try:
        item["payload"] = json.loads(item["payload_json"])
    except Exception:
        item["payload"] = {}
    return item

