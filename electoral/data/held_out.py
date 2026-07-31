"""Held-out shock enforcement -- the single source of truth for which shocks
are validation-only, permanently.

Phase 1, Step 1.6. Reads configs/held_out_shocks.json (frozen 2026-07-31,
append-only in spirit -- see that file's `_meta.principle`). Every corpus
builder, grounding/injection script, and resampling path in this project MUST
call assert_not_held_out()/assert_none_held_out() before writing a training
record or deriving a label from a real shock. A held-out shock appearing in
training data is a hard failure, not a warning -- see
scripts/check_held_out_leakage.py for why that distinction matters (existing
corpora already leaked held-out shocks once, silently, between two versions
of the grounding script).
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Iterable

_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "held_out_shocks.json"


class HeldOutShockError(Exception):
    """Raised when a held-out shock is detected somewhere it must never be:
    training data, label derivation, sentiment injection, or model selection.
    """


@lru_cache(maxsize=1)
def _load_config() -> dict:
    if not _CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"held-out shock config not found at {_CONFIG_PATH} -- this is required "
            "infrastructure, not optional; refusing to silently treat everything as trainable"
        )
    with open(_CONFIG_PATH, encoding="utf-8") as f:
        return json.load(f)


def held_out_shock_ids() -> frozenset[str]:
    """The frozen set of validation-only shock IDs. Cached -- config is read
    once per process; call held_out_shock_ids.cache_clear() (via
    _load_config) in tests that swap the config file.
    """
    cfg = _load_config()
    return frozenset(s["id"] for s in cfg["held_out_shocks"])


def held_out_shock_records() -> list[dict]:
    """Full records (id, category, ground_truth_coverage, elicitation,
    rationale) for callers that want the rationale, not just the id set.
    """
    return list(_load_config()["held_out_shocks"])


def is_held_out(shock_id: str) -> bool:
    return shock_id in held_out_shock_ids()


def assert_not_held_out(shock_id: str, context: str = "") -> None:
    """Raise HeldOutShockError if `shock_id` is held out. Call this at the
    point a training record is about to be written, or a label is about to be
    derived from `shock_id` and attached to another record.
    """
    if is_held_out(shock_id):
        where = f" in {context}" if context else ""
        raise HeldOutShockError(
            f"Held-out shock {shock_id!r} detected{where}. This shock is frozen for "
            f"validation-only use (configs/held_out_shocks.json, frozen "
            f"{_load_config()['_meta']['frozen_date']}) and must never appear in training "
            f"data, label derivation, or sentiment injection -- for any model, retroactively "
            f"and going forward. This is a hard failure, not a warning; do not catch and "
            f"continue without removing the offending shock from the pipeline input."
        )


def assert_none_held_out(shock_ids: Iterable[str], context: str = "") -> None:
    """Same as assert_not_held_out(), but checks a whole batch at once and
    reports every offender found, not just the first -- useful right before
    writing out a full corpus file.
    """
    leaked = sorted({sid for sid in shock_ids if is_held_out(sid)})
    if leaked:
        where = f" in {context}" if context else ""
        raise HeldOutShockError(
            f"{len(leaked)} held-out shock(s) detected{where}: {leaked}. These shocks are "
            f"frozen for validation-only use (configs/held_out_shocks.json) and must never "
            f"appear in training data, label derivation, or sentiment injection. This is a "
            f"hard failure, not a warning."
        )


def filter_out_held_out(shock_ids: Iterable[str]) -> list[str]:
    """Non-raising variant: returns `shock_ids` with any held-out entries
    removed, preserving order. Use this where the correct behavior is to
    silently exclude held-out shocks from a REFERENCE/CONTEXT list (e.g. the
    'real shock examples' slice embedded in a synthetic-generation prompt)
    rather than fail the whole run -- excluding is safe there because no
    label is being derived FROM the held-out shock, it's just being kept out
    of the model's view. Prefer assert_none_held_out() wherever a label or
    training record is actually being produced.
    """
    held = held_out_shock_ids()
    return [sid for sid in shock_ids if sid not in held]
