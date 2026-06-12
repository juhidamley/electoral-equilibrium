"""Endpoint latency test — 5 POST /estimate calls, timing recorded but p95 not hard-asserted.

Skipped when ADAPTER_PATH env var is not set.
Starts the FastAPI app as a subprocess on API_PORT (default 8315).
"""

from __future__ import annotations

import os
import statistics
import subprocess
import sys
import time

import pytest
import requests

from electoral.artifacts import ShockResponseData
from electoral.core.types import BIN_MIDPOINTS, CANONICAL_RACES

# ── Config ────────────────────────────────────────────────────────────────────

PORT = os.environ.get("API_PORT", "8315")
BASE_URL = f"http://127.0.0.1:{PORT}"

_ADAPTER_PATH = os.environ.get("ADAPTER_PATH", "")
if not _ADAPTER_PATH:
    pytest.skip(
        "ADAPTER_PATH env var not set — skipping endpoint latency test.",
        allow_module_level=True,
    )

_REQUEST_BODY = {
    "event": {
        "description": (
            "Ayatollah Khamenei assassinated in Tehran, "
            "Iran enters open conflict with Israel and US forces"
        )
    },
    "intensity": 1.0,
}

_N_CALLS = 5
_N_RACE = len(CANONICAL_RACES)
_IDENTITY_COV = [
    [1.0 if i == j else 0.0 for j in range(_N_RACE)]
    for i in range(_N_RACE)
]

# ── Helpers ───────────────────────────────────────────────────────────────────


def _await_health(base_url: str, timeout: float = 120.0) -> bool:
    """Poll GET /health until 200 OK or timeout."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            r = requests.get(f"{base_url}/health", timeout=2)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(1)
    return False


def _validate_response_data(data: dict) -> None:
    """Construct ShockResponseData from /estimate response and call validate()."""
    bins_race: dict[str, str] = data["delta_bins_race"]
    bins_religion: dict[str, str] = data["delta_bins_religion"]
    bins_gender: dict[str, str] = data["delta_bins_gender"]
    result = ShockResponseData(
        shock="ayatollah_assassination",
        cycle=2026,
        party="democrat",
        delta_bins_race=bins_race,
        delta_bins_religion=bins_religion,
        delta_bins_gender=bins_gender,
        deltas_race={k: BIN_MIDPOINTS[v] for k, v in bins_race.items()},
        deltas_religion={k: BIN_MIDPOINTS[v] for k, v in bins_religion.items()},
        deltas_gender={k: BIN_MIDPOINTS[v] for k, v in bins_gender.items()},
        delta_eff=float(data["delta_eff"]),
        covariance=_IDENTITY_COV,
        source="llm_unified",
    )
    result.validate()


# ── Test ──────────────────────────────────────────────────────────────────────


@pytest.mark.timeout(600)
def test_endpoint_latency() -> None:
    """Start API server, send 5 /estimate requests, assert correctness, log latency."""
    env = {**os.environ, "ADAPTER_PATH": _ADAPTER_PATH, "API_PORT": PORT}
    proc = subprocess.Popen(
        [sys.executable, "-m", "electoral.api.shock_endpoint"],
        env=env,
    )
    try:
        if not _await_health(BASE_URL, timeout=120.0):
            pytest.fail(
                f"Server at {BASE_URL} did not respond to GET /health within 120s"
            )

        latencies: list[float] = []

        for i in range(_N_CALLS):
            t0 = time.perf_counter()
            response = requests.post(
                f"{BASE_URL}/estimate", json=_REQUEST_BODY, timeout=120
            )
            elapsed = time.perf_counter() - t0
            latencies.append(elapsed)

            # (i) HTTP 200
            assert response.status_code == 200, (
                f"Call {i + 1}: expected 200, got {response.status_code}: "
                f"{response.text[:300]}"
            )

            # (ii) valid JSON
            data = response.json()

            # (iii) ShockResponseData.validate()
            _validate_response_data(data)

            print(f"  call {i + 1}/{_N_CALLS}: {elapsed:.2f}s")

        # ── Latency summary (no hard assert on p95) ───────────────────────────
        mean_t = statistics.mean(latencies)
        median_t = statistics.median(latencies)
        p95_t = sorted(latencies)[int(0.95 * len(latencies))]

        print(
            f"\nLatency — {_N_CALLS} calls: "
            f"mean={mean_t:.1f}s  median={median_t:.1f}s  p95={p95_t:.1f}s"
        )
        if p95_t > 3.0:
            print(
                f"NOTE: p95={p95_t:.1f}s exceeds 3s target — "
                "deferred to vLLM post-SRP (see docs/latency_benchmark.md)"
            )

    finally:
        proc.terminate()
        proc.wait()
