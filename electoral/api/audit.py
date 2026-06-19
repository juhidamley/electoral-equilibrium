"""Audit logger for Electoral Equilibrium estimate requests.

Backed by DuckDB at data/audit.duckdb.  Designed for low-volume logging
(one write per estimate call) — single connection, guarded by threading.Lock
for safe use across the ThreadPoolExecutor workers in shock_endpoint.py.

DuckDB connections are single-writer; a lock is sufficient because estimates
are infrequent and the process pool is max_workers=1.
"""

from __future__ import annotations

import json
import logging
import math
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb

log = logging.getLogger(__name__)


class AuditLogger:
    def __init__(self, db_path: str = "data/audit.duckdb") -> None:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._db_path = db_path
        self._con = duckdb.connect(db_path)
        self._lock = threading.Lock()
        self._init_schema()

    def _init_schema(self) -> None:
        # Sequence for auto-increment id (DuckDB pattern: no AUTOINCREMENT keyword).
        self._con.execute("CREATE SEQUENCE IF NOT EXISTS estimates_id_seq START 1;")
        self._con.execute(
            """
            CREATE TABLE IF NOT EXISTS estimates (
                id             INTEGER PRIMARY KEY DEFAULT nextval('estimates_id_seq'),
                timestamp      TIMESTAMP,
                event_text     VARCHAR,
                intensity      FLOAT,
                deltas_json    VARCHAR,
                feasible       BOOLEAN,
                target_met     BOOLEAN,
                win_prob       FLOAT,
                llm_ms         INTEGER,
                optimizer_ms   INTEGER,
                montecarlo_ms  INTEGER,
                backend        VARCHAR,
                party          VARCHAR
            );
        """
        )
        # Idempotent migration: add party to databases created before this column existed.
        try:
            self._con.execute("ALTER TABLE estimates ADD COLUMN IF NOT EXISTS party VARCHAR;")
        except Exception:
            pass

    def log_estimate(
        self,
        *,
        event_text: str,
        intensity: float,
        deltas: Any,
        feasible: bool | None,
        target_met: bool | None,
        win_prob: float | None,
        llm_ms: int,
        optimizer_ms: int,
        montecarlo_ms: int,
        backend: str,
        party: str | None = None,
    ) -> None:
        """Insert one audit row.  Non-finite floats are stored as NULL."""

        def _safe(x: Any) -> Any:
            return None if (isinstance(x, float) and not math.isfinite(x)) else x

        try:
            with self._lock:
                self._con.execute(
                    "INSERT INTO estimates "
                    "(timestamp, event_text, intensity, deltas_json, "
                    "feasible, target_met, win_prob, "
                    "llm_ms, optimizer_ms, montecarlo_ms, backend, party) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                    [
                        datetime.now(timezone.utc),
                        event_text,
                        intensity,
                        json.dumps(deltas),
                        feasible,
                        target_met,
                        _safe(win_prob),
                        llm_ms,
                        optimizer_ms,
                        montecarlo_ms,
                        backend,
                        party,
                    ],
                )
        except Exception:
            # Audit failures must never surface to the user.
            log.warning("AuditLogger.log_estimate failed", exc_info=True)

    def count(self) -> int:
        """Return total number of estimate rows (lightweight — no full scan)."""
        with self._lock:
            result = self._con.execute("SELECT COUNT(*) FROM estimates").fetchone()
            return int(result[0]) if result else 0

    def recent(self, limit: int = 100, search: str | None = None) -> list[dict[str, Any]]:
        """Return the most recent rows as a list of dicts.

        If search is given, filters with WHERE event_text ILIKE ? (case-insensitive).
        The search value is passed as a bound parameter — "%" + search + "%" —
        never f-string interpolated into SQL (SQL injection guard).

        limit is clamped to 500 as a defence-in-depth measure; the route layer
        also clamps before calling here.

        DuckDB NULL floats arrive as float('nan') from fetch_df();
        convert them back to None so the JSON response is well-formed.
        """
        limit = min(max(1, limit), 500)
        with self._lock:
            if search:
                rows = self._con.execute(
                    "SELECT * FROM estimates WHERE event_text ILIKE ? "
                    "ORDER BY timestamp DESC LIMIT ?",
                    ["%" + search + "%", limit],
                ).fetch_df()
            else:
                rows = self._con.execute(
                    "SELECT * FROM estimates ORDER BY timestamp DESC LIMIT ?", [limit]
                ).fetch_df()
        records = rows.to_dict(orient="records")
        for row in records:
            for k, v in row.items():
                if isinstance(v, float) and math.isnan(v):
                    row[k] = None
        return records
