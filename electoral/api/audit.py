"""Audit logger for Electoral Equilibrium estimate requests.

═══════════════════════════════════════════════════════════════════════════════
WHAT THIS IS (beginner orientation)
═══════════════════════════════════════════════════════════════════════════════
"Audit logging" = keeping a permanent record of every estimate the API runs:
the event text, the inputs, the win probability, how long each stage took. This
powers the dashboard's audit table and lets you answer later questions like
"what did we predict for that shock last week, and how fast was it?"

It's backed by DuckDB — a lightweight SQL database that lives in a single file
(data/audit.duckdb), no server to run. Think "SQLite, but built for analytics."
We talk to it with ordinary SQL strings.

THREAD-SAFETY (the threading.Lock): the web server handles requests on multiple
threads at once. A single DuckDB connection must NOT be used by two threads
simultaneously, so every database operation below is wrapped in `with self._lock:`
— that ensures only one thread touches the connection at a time (the others wait
their turn). Because estimates are infrequent and quick, this serialization costs
nothing noticeable. (See shock_endpoint.py for the thread/process pools.)
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
    """Append-only logger of estimate runs, backed by a single DuckDB file.

    Created once at app startup (see shock_endpoint.py's lifespan) and shared by
    all requests. Three public methods: log_estimate() (write one row),
    recent() (read back recent rows, optionally filtered), count() (total rows).
    """

    def __init__(self, db_path: str = "data/audit.duckdb") -> None:
        # Make sure the data/ directory exists before DuckDB tries to create the
        # file inside it.
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._db_path = db_path
        self._con = duckdb.connect(db_path)  # opens (or creates) the .duckdb file
        self._lock = threading.Lock()  # serializes all access to _con (see module docstring)
        self._init_schema()  # create the table on first run (idempotent)

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
