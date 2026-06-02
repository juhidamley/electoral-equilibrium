"""Panel cleaning — coerce, normalise, drop, sort, and dedup a raw survey panel."""

from __future__ import annotations

import logging
import re

import pandas as pd

log = logging.getLogger(__name__)

# Columns that must be non-null to keep a row.
_REQUIRED: tuple[str, ...] = ("cycle", "bloc")

# Ordered substitution rules applied to raw bloc strings.
_BLOC_SUBS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"[\s/\-\.]+"), "_"),  # whitespace / slashes / hyphens / dots → _
    (re.compile(r"[^a-z0-9_]"), ""),  # strip remaining non-word chars
    (re.compile(r"_+"), "_"),  # collapse runs of underscores
    (re.compile(r"^_+|_+$"), ""),  # strip leading/trailing underscores
]


def _normalise_bloc(s: pd.Series) -> pd.Series:
    """Lowercase and snake_case a string Series; preserve NA as pd.NA."""
    null_mask = s.isna()
    # fillna("") so str operations see an empty string for nulls, not "nan".
    out = s.fillna("").astype(str).str.lower().str.strip()
    for pattern, repl in _BLOC_SUBS:
        out = out.str.replace(pattern, repl, regex=True)
    out = out.astype("string")
    # Restore original nulls and values that normalised to empty string.
    # Use .eq("") rather than == "" to avoid Python-level comparison on StringArray.
    out[null_mask | out.eq("")] = pd.NA
    return out


def clean_raw_panel(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all cleaning steps to a raw survey panel DataFrame.

    Steps (in order):
    1. Coerce ``cycle`` to nullable Int64 (YYYY); non-numeric → NA.
    2. Normalise ``bloc`` to lowercase snake_case nullable string.
    3. Coerce ``vote_share`` and ``turnout`` to nullable Float64; non-numeric → NA.
    4. Drop rows where ``cycle`` or ``bloc`` is null; log count at INFO.
    5. Sort by (cycle, bloc); reset index.
    6. Raise ValueError on duplicate (cycle, bloc, source) tuples.
       Falls back to (cycle, bloc) when ``source`` column is absent.
       Note: ``pd.NA`` in ``source`` is treated as equal by pandas
       ``duplicated()`` — rows with missing source can be flagged as
       duplicates of each other even if they represent distinct provenance.

    Returns a new DataFrame; the input is not modified.
    """
    df = df.copy()

    # ── Step 1: cycle → Int64 ────────────────────────────────────────────────
    if "cycle" in df.columns:
        df["cycle"] = pd.to_numeric(df["cycle"], errors="coerce").astype("Int64")

    # ── Step 2: bloc → lowercase snake_case ──────────────────────────────────
    if "bloc" in df.columns:
        df["bloc"] = _normalise_bloc(df["bloc"])

    # ── Step 3: vote_share / turnout → Float64 ────────────────────────────────
    for col in ("vote_share", "turnout"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Float64")

    # ── Step 4: drop rows missing cycle or bloc ───────────────────────────────
    required = [c for c in _REQUIRED if c in df.columns]
    n_before = len(df)
    df = df.dropna(subset=required).reset_index(drop=True)
    n_dropped = n_before - len(df)
    if n_dropped:
        log.info(
            "clean_raw_panel: dropped %d row(s) with null %s",
            n_dropped,
            required,
        )

    # ── Step 5: sort by (cycle, bloc) ────────────────────────────────────────
    sort_keys = [c for c in ("cycle", "bloc") if c in df.columns]
    if sort_keys:
        df = df.sort_values(sort_keys).reset_index(drop=True)

    # ── Step 6: raise on duplicate key tuples ────────────────────────────────
    dup_keys = [c for c in ("cycle", "bloc", "source") if c in df.columns]
    if len(dup_keys) >= 2:
        dupes = df.duplicated(subset=dup_keys, keep=False)
        if dupes.any():
            examples = df.loc[dupes, dup_keys].drop_duplicates().head(3).to_dict(orient="records")
            raise ValueError(
                f"clean_raw_panel: duplicate ({', '.join(dup_keys)}) tuples detected — "
                f"deduplicate before cleaning. Examples: {examples}"
            )

    return df
