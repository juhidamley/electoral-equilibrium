"""Excluded-source enforcement -- the single source of truth for which social
archive sources are ineligible for bloc-level training.

Phase 4, Step 4.1. Reads configs/excluded_sources.json (frozen 2026-08-01,
append-only in spirit -- see that file's `_meta.principle`). Every resampling
script, archive loader, and corpus/label-aggregation path in this project MUST
call filter_excluded()/is_excluded_source() before a record contributes to
bloc-level training data (a per-bloc sentiment aggregate, a delta-bin label,
or any downstream fine-tuning record derived from a per-bloc signal). This
mirrors electoral/data/held_out.py's structure deliberately, so the two read
alike and a reader who understands one understands the other.

EXCLUDE, DO NOT DELETE: excluded sources stay on disk in full. This module
only governs eligibility for bloc-level training, not storage. Excluded
sources may still be used for aggregate event-level context, future
classifier work, or provenance -- see configs/excluded_sources.json's
`scope` field for the exact boundary.

Matching is on `archive_id` ONLY, never on `platform` -- `platform` is
overloaded and stores subreddit names for Reddit records (Step 0.1); matching
on it would both miss records and accidentally exclude unrelated content that
happens to share a platform string.
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "excluded_sources.json"


class ExcludedSourceError(Exception):
    """Raised when an excluded source is detected somewhere it must never be:
    bloc-level training data or per-bloc label derivation.
    """


@lru_cache(maxsize=1)
def _load_config() -> dict:
    if not _CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"excluded-sources config not found at {_CONFIG_PATH} -- this is required "
            "infrastructure, not optional; refusing to silently treat every source as eligible"
        )
    with open(_CONFIG_PATH, encoding="utf-8") as f:
        return json.load(f)


def excluded_archive_ids() -> frozenset[str]:
    """The frozen set of archive_ids ineligible for bloc-level training.
    Cached -- config is read once per process; call
    excluded_archive_ids.cache_clear() (via _load_config) in tests that swap
    the config file.
    """
    cfg = _load_config()
    return frozenset(s["archive_id"] for s in cfg["excluded_sources"])


def excluded_source_records() -> list[dict]:
    """Full records (archive_id, category, reason, phase_0_numbers,
    reinstatement_condition) for callers that want the rationale, not just
    the id set.
    """
    return list(_load_config()["excluded_sources"])


def _extract_archive_id(record: dict[str, Any]) -> str | None:
    """Pull archive_id out of a record, unwrapping a {"payload": {...}}
    envelope if present. Returns None if no archive_id is present at all --
    callers should treat that as "not excluded" (nothing to match against),
    not as an error; a missing archive_id is a data-quality issue for the
    caller to handle separately, not this module's concern.
    """
    payload = record.get("payload", record) if isinstance(record, dict) else None
    if not isinstance(payload, dict):
        return None
    aid = payload.get("archive_id")
    return aid if isinstance(aid, str) else None


def is_excluded_source(record: dict[str, Any]) -> bool:
    """True if `record`'s payload.archive_id is in the excluded set.

    Accepts either a raw payload dict or an {"schema_version"/"payload": {...}}
    envelope -- unwraps automatically. Matches on archive_id ONLY, never on
    platform (see module docstring).
    """
    aid = _extract_archive_id(record)
    return aid is not None and aid in excluded_archive_ids()


def assert_not_excluded_source(record: dict[str, Any], context: str = "") -> None:
    """Raise ExcludedSourceError if `record` is from an excluded source. Call
    this at the point a record is about to contribute to bloc-level training
    data or a per-bloc label derivation.
    """
    if is_excluded_source(record):
        aid = _extract_archive_id(record)
        where = f" in {context}" if context else ""
        raise ExcludedSourceError(
            f"Excluded source {aid!r} detected{where}. This archive_id is ineligible "
            f"for bloc-level training (configs/excluded_sources.json, frozen "
            f"{_load_config()['_meta']['frozen_date']}) and must never contribute to "
            f"bloc-level training data or per-bloc label derivation. This is a hard "
            f"failure, not a warning; do not catch and continue without removing the "
            f"offending record from the pipeline input."
        )


def filter_excluded(
    records: Iterable[dict[str, Any]], context: str = ""
) -> list[dict[str, Any]]:
    """Return `records` with excluded-source entries removed, preserving
    order. Logs the dropped count explicitly (never a silent drop) -- this is
    the primary entry point for pipeline code: filter loudly, then proceed
    with what remains.
    """
    records = list(records)
    kept = [r for r in records if not is_excluded_source(r)]
    n_dropped = len(records) - len(kept)
    if n_dropped:
        where = f" ({context})" if context else ""
        dropped_by_archive: dict[str, int] = {}
        for r in records:
            aid = _extract_archive_id(r)
            if aid is not None and aid in excluded_archive_ids():
                dropped_by_archive[aid] = dropped_by_archive.get(aid, 0) + 1
        logger.info(
            "filter_excluded%s: dropped %d/%d record(s) from excluded source(s) %s -- "
            "%d record(s) remain",
            where,
            n_dropped,
            len(records),
            dropped_by_archive,
            len(kept),
        )
    return kept
