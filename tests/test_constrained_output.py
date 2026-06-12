"""Constrained-output integration test: 10 calls across 10 events (1 rep each).

Requires the trained LoRA adapter at /Volumes/JUHIDRIVE/electoralData/models/mistral-r16.
All tests are skipped when the path is absent (CI / machines without the drive).

Test structure:
  - ShockEstimator loaded once at module scope (fixture).
  - 10 event dicts built from the first 10 active shocks in configs/shocks.json.
  - estimate(event, intensity=0.5) called 1× per event = 10 total.
  - Per call: (i) validate() passes, (ii) all deltas finite and in [-0.15, 0.15],
    (iii) all 15 canonical bloc keys present, (iv) no exception raised.
  - Summary printed: N/10 succeeded; failures logged per shock_id.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import os

import pytest

from electoral.core.types import CANONICAL_GENDERS, CANONICAL_RACES, CANONICAL_RELIGIONS

# ── Skip guard ────────────────────────────────────────────────────────────────

_ADAPTER_PATH = Path(os.environ.get(
    "ADAPTER_PATH",
    "/Volumes/JUHIDRIVE/electoralData/models/mistral-r16",
))

if not _ADAPTER_PATH.exists():
    pytest.skip(
        f"Trained adapter not found at {_ADAPTER_PATH} — "
        "attach JUHIDRIVE or copy adapter to run constrained-output tests.",
        allow_module_level=True,
    )

from electoral.llm.inference import ShockEstimator  # noqa: E402 (after skip guard)

# ── Event fixtures ────────────────────────────────────────────────────────────

_SHOCKS_PATH = Path(__file__).parent.parent / "configs" / "shocks.json"


def _build_events() -> list[dict]:
    shocks: list[dict] = json.loads(_SHOCKS_PATH.read_text(encoding="utf-8"))
    active = [s for s in shocks if s.get("active", False)]
    events = []
    for shock in active[:10]:
        events.append({
            "shock_id": shock["id"],
            "cycle": int(shock["date"][:4]),
            "party": "democrat",
            "description": shock["description"],
            "news_roberta_scores": {},
            "social_roberta_scores": {},
        })
    return events


_EVENTS: list[dict] = _build_events()
assert len(_EVENTS) == 10, f"Expected 10 active shocks, got {len(_EVENTS)}"

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def estimator() -> ShockEstimator:
    return ShockEstimator(str(_ADAPTER_PATH))


# ── Helpers ───────────────────────────────────────────────────────────────────


def _check_result(result) -> list[str]:
    """Return a list of failure messages; empty list means all assertions passed."""
    failures: list[str] = []

    # (i) validate() — catches canonical key violations, bin token validity, etc.
    try:
        result.validate()
    except Exception as exc:
        failures.append(f"validate() raised: {exc}")

    # (ii) all numeric deltas finite and in [-0.15, 0.15]
    for stratum, d in (
        ("race", result.deltas_race),
        ("religion", result.deltas_religion),
        ("gender", result.deltas_gender),
    ):
        for bloc, v in d.items():
            if not math.isfinite(v):
                failures.append(f"deltas_{stratum}[{bloc!r}] = {v!r} is not finite")
            elif not (-0.15 <= v <= 0.15):
                failures.append(
                    f"deltas_{stratum}[{bloc!r}] = {v:.4f} outside [-0.15, 0.15]"
                )

    # (iii) all canonical bloc keys present
    for canon, bins_dict, label in (
        (CANONICAL_RACES, result.delta_bins_race, "race"),
        (CANONICAL_RELIGIONS, result.delta_bins_religion, "religion"),
        (CANONICAL_GENDERS, result.delta_bins_gender, "gender"),
    ):
        for bloc in canon:
            if bloc not in bins_dict:
                failures.append(f"delta_bins_{label} missing key {bloc!r}")

    return failures


# ── Main test ─────────────────────────────────────────────────────────────────


@pytest.mark.timeout(1800)
def test_10_calls(estimator: ShockEstimator) -> None:
    """10 calls (10 events × 1 rep) — all must pass all four assertions."""
    n_success = 0
    n_calls = 0
    failures: list[tuple[str, list[str]]] = []  # (shock_id, [messages])

    for event in _EVENTS:
        shock_id = event["shock_id"]
        n_calls += 1
        try:
            result = estimator.estimate(event, intensity=0.5)
        except Exception as exc:
            failures.append((shock_id, [f"estimate() raised {type(exc).__name__}: {exc}"]))
            continue

        msgs = _check_result(result)
        if msgs:
            failures.append((shock_id, msgs))
        else:
            n_success += 1

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'─' * 60}")
    print(f"Constrained-output: {n_success}/{n_calls} calls succeeded")
    if failures:
        print(f"{len(failures)} failure group(s):")
        for shock_id, msgs in failures:
            for msg in msgs:
                print(f"  FAIL [{shock_id}] {msg}")
    else:
        print("All assertions passed.")
    print(f"{'─' * 60}")

    assert not failures, (
        f"{len(failures)} failure group(s) out of {n_calls} calls. "
        f"First: {failures[0]}"
    )
