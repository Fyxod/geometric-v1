from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any


class HistoryDB:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _init(self) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS ui_runs (
                    run_id TEXT PRIMARY KEY,
                    run_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    finished_at TEXT,
                    config_paths TEXT NOT NULL,
                    output_path TEXT NOT NULL,
                    report_path TEXT,
                    min_score REAL,
                    max_score REAL,
                    progress_summary TEXT NOT NULL
                )
                """
            )
            connection.commit()

    def create_run(
        self,
        *,
        run_id: str,
        run_type: str,
        status: str,
        started_at: str,
        config_paths: dict[str, str],
        output_path: str,
    ) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO ui_runs (
                    run_id, run_type, status, started_at, config_paths, output_path, progress_summary
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    run_type,
                    status,
                    started_at,
                    json.dumps(config_paths),
                    output_path,
                    "{}",
                ),
            )
            connection.commit()

    def update_run(self, run_id: str, **fields: Any) -> None:
        if not fields:
            return
        serialized = dict(fields)
        for key in ("config_paths", "progress_summary"):
            if key in serialized and not isinstance(serialized[key], str):
                serialized[key] = json.dumps(serialized[key])
        assignments = ", ".join(f"{key} = ?" for key in serialized)
        values = list(serialized.values()) + [run_id]
        with self._lock, self._connect() as connection:
            connection.execute(f"UPDATE ui_runs SET {assignments} WHERE run_id = ?", values)
            connection.commit()

    def list_runs(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM ui_runs ORDER BY started_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [self._normalize(dict(row)) for row in rows]

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        with self._lock, self._connect() as connection:
            row = connection.execute("SELECT * FROM ui_runs WHERE run_id = ?", (run_id,)).fetchone()
        return self._normalize(dict(row)) if row else None

    def _normalize(self, row: dict[str, Any]) -> dict[str, Any]:
        for key in ("config_paths", "progress_summary"):
            try:
                row[key] = json.loads(row.get(key) or "{}")
            except json.JSONDecodeError:
                row[key] = {}
        return row
