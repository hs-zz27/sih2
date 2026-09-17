"""SQLite run store (plan task 1.5).

Persists every run's query, trace and status so a run can be inspected after
the SSE stream has closed. Deliberately plain SQL with parameter binding - no
ORM, and no string interpolation of values anywhere.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from satquery.contracts.trace import Trace

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id       TEXT PRIMARY KEY,
    created_utc  TEXT NOT NULL,
    query        TEXT NOT NULL,
    status       TEXT NOT NULL,
    task         TEXT,
    answer       TEXT,
    confidence   REAL,
    band         TEXT,
    abstained    INTEGER,
    error        TEXT,
    trace_json   TEXT
);
CREATE INDEX IF NOT EXISTS idx_runs_created ON runs (created_utc DESC);
"""


class RunStore:
    """Thread-safe SQLite store for runs."""

    def __init__(self, path: str | Path = "satquery_runs.db"):
        self.path = str(path)
        # check_same_thread=False plus an explicit lock: FastAPI serves
        # requests from a thread pool, and SQLite connections are not
        # thread-safe without serialising access ourselves.
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._conn.executescript(SCHEMA)
            self._conn.commit()

    def create(self, run_id: str, query: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO runs "
                "(run_id, created_utc, query, status, abstained) "
                "VALUES (?, ?, ?, ?, 0)",
                (run_id, datetime.now(timezone.utc).isoformat(), query, "running"),
            )
            self._conn.commit()

    def complete(self, run_id: str, trace: Trace) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE runs SET status=?, task=?, answer=?, confidence=?, "
                "band=?, abstained=?, trace_json=? WHERE run_id=?",
                (
                    "complete",
                    trace.routing.selected_task,
                    trace.answer,
                    trace.confidence.final,
                    trace.confidence.band,
                    int(trace.abstained),
                    trace.model_dump_json(),
                    run_id,
                ),
            )
            self._conn.commit()

    def fail(self, run_id: str, error: str) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE runs SET status=?, error=? WHERE run_id=?",
                ("failed", error, run_id),
            )
            self._conn.commit()

    def get(self, run_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM runs WHERE run_id=?", (run_id,)
            ).fetchone()
        if row is None:
            return None
        record = dict(row)
        if record.get("trace_json"):
            record["trace"] = json.loads(record.pop("trace_json"))
        else:
            record.pop("trace_json", None)
            record["trace"] = None
        return record

    def list(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT run_id, created_utc, query, status, task, answer, "
                "confidence, band, abstained, error FROM runs "
                "ORDER BY created_utc DESC LIMIT ?",
                (int(limit),),
            ).fetchall()
        return [dict(r) for r in rows]

    def close(self) -> None:
        with self._lock:
            self._conn.close()
