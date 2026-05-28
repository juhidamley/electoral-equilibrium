"""Shared schema normalization for all social media collectors.

Every collector writes JSONL records in the canonical envelope format:
    {
      "schema_version": "1.0",
      "stage": "collect",
      "collected_at": "ISO8601",    ← when the collector wrote this record
      "seed": <int|null>,           ← global pipeline seed (from configs/base.json)
      "payload": {
        "id":                 str,  ← platform-specific unique post URI
        "text":               str,  ← post text (cleaned of truncation markers)
        "created_at":         str,  ← when the post was ORIGINALLY created (ISO8601)
        "lang":               str,  ← BCP-47 language code (e.g. "en") or ""
        "source":             str,  ← "live_stream" | "live_scrape" | "archive"
        "archive_id":         str,  ← "bluesky" | "apify_x" | dataset slug
        "platform":           str,  ← "bluesky" | "apify_x" | "reddit" | ...
        "shock_id":           str|null,  ← matched shock from shocks.json
        "author_did":         str|null,  ← platform user identifier
        "author_handle":      str|null,  ← @handle (null until batch resolve)
        "author_description": str|null,  ← bio text (null until bio classify pass)
        "inference_method":   str|null,  ← null until bio classifier assigns
      }
    }

After the merge_posts() Prefect task, the scorer reads from rawdata/merged/
which contains the flattened payload dicts (envelope stripped).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "1.0"
STAGE = "collect"

# Languages we keep.  Posts with no lang tag are retained (may be English).
KEEP_LANGS: frozenset[str] = frozenset(
    ["en", "en-gb", "en-us", "en-au", "en-ca", ""]
)


def normalize_timestamp(ts: str | None) -> str:
    """Return a UTC ISO-8601 string for any common timestamp format.

    Handles:
      - Already-valid ISO strings  (atproto, Apify ISO output)
      - Unix epoch integers / floats (some Apify actors)
      - Truncated ISO without timezone (assumes UTC)
    Falls back to the current UTC time if parsing fails.
    """
    if not ts:
        return datetime.now(timezone.utc).isoformat()

    # Strip common noise and ensure timezone
    ts = str(ts).strip()

    # Unix epoch (numeric)
    if ts.lstrip("-").replace(".", "", 1).isdigit():
        try:
            epoch = float(ts)
            # If looks like milliseconds (> year 2100 in seconds), convert
            if epoch > 4_102_444_800:
                epoch /= 1000.0
            return datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat()
        except (ValueError, OSError):
            pass

    # ISO string — normalize to UTC
    try:
        # Handle trailing Z
        clean = ts.replace("Z", "+00:00")
        # Handle space-separated (MySQL style)
        clean = clean.replace(" ", "T", 1)
        dt = datetime.fromisoformat(clean)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).isoformat()
    except (ValueError, TypeError):
        pass

    # Fallback: return as-is (downstream validation will catch it)
    return ts


def extract_primary_lang(langs: list[str] | str | None) -> str:
    """Return the primary BCP-47 language code, lowercased and stripped."""
    if not langs:
        return ""
    if isinstance(langs, str):
        return langs.lower().strip()
    for lang in langs:
        if lang:
            return lang.lower().strip()
    return ""


def is_english(langs: list[str] | str | None) -> bool:
    """Return True if the post is English or has no language tag."""
    lang = extract_primary_lang(langs)
    return not lang or lang[:2] == "en"


def build_post_payload(
    *,
    post_id: str,
    text: str,
    created_at: str | None,
    lang: str | list[str] | None,
    source: str,
    archive_id: str,
    platform: str,
    shock_id: str | None = None,
    author_did: str | None = None,
    author_handle: str | None = None,
    author_description: str | None = None,
    inference_method: str | None = None,
) -> dict[str, Any]:
    """Build the canonical payload dict for a single post record."""
    return {
        "id": post_id,
        "text": text.strip() if text else "",
        "created_at": normalize_timestamp(created_at),
        "lang": extract_primary_lang(lang),
        "source": source,
        "archive_id": archive_id,
        "platform": platform,
        "shock_id": shock_id,
        "author_did": author_did,
        "author_handle": author_handle,
        "author_description": author_description,
        "inference_method": inference_method,
    }


def wrap_envelope(payload: dict[str, Any], seed: int | None = None) -> dict[str, Any]:
    """Wrap a payload dict in the CLAUDE.md canonical envelope."""
    return {
        "schema_version": SCHEMA_VERSION,
        "stage": STAGE,
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "payload": payload,
    }


def append_post_record(
    path: Path,
    payload: dict[str, Any],
    seed: int | None = None,
) -> None:
    """Append one post record to a JSONL file in append-only mode.

    Write-once ownership: only the Intel Mac should call this for rawdata/social/.
    Uses append mode to prevent memory overflow on high-volume firehoses.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    envelope = wrap_envelope(payload, seed=seed)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(envelope, ensure_ascii=False))
        f.write("\n")


def load_shocks(shocks_path: str | Path) -> list[dict[str, Any]]:
    """Load shock registry from configs/shocks.json."""
    with open(shocks_path, encoding="utf-8") as f:
        return json.load(f)


def build_keyword_index(shocks: list[dict[str, Any]]) -> dict[str, list[str]]:
    """Build a case-insensitive keyword → [shock_id, ...] lookup table.

    A post text is matched by checking if any keyword is a substring of the
    lowercased text. O(K) per post where K = total number of unique keywords.
    """
    index: dict[str, list[str]] = {}
    for shock in shocks:
        shock_id = shock["id"]
        for kw in shock.get("keywords", []):
            normalized = kw.lower().strip()
            if normalized:
                index.setdefault(normalized, []).append(shock_id)
    return index


def match_shocks(text: str, keyword_index: dict[str, list[str]]) -> set[str]:
    """Return the set of shock IDs whose keywords appear in text."""
    text_lower = text.lower()
    matched: set[str] = set()
    for kw, shock_ids in keyword_index.items():
        if kw in text_lower:
            matched.update(shock_ids)
    return matched
