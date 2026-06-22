"""Panel DataFrame validation — five structural invariants.

This is the GATE the voter panel must pass before any downstream stage trusts it
(called at the end of kernels/data.py). It checks the shape/typing rules that the
rest of the pipeline assumes — required columns present and non-null, cycles are
sane YYYY integers, bloc IDs are canonical snake_case, and shares are in [0, 1].
It RAISES on the first violation (naming the column) rather than returning a
flag, so bad data stops the pipeline loudly instead of corrupting results later.
"""

from __future__ import annotations

import re

import pandas as pd

# Lowercase snake_case: starts with a-z, body is [a-z0-9_], no leading/trailing
# underscores, no consecutive underscores.
_SNAKE_RE = re.compile(r"^[a-z][a-z0-9]*(_[a-z0-9]+)*$")

# Columns that must lie in [0, 1] when present.
_SHARE_COLS: frozenset[str] = frozenset(["vote_share", "turnout"])

_CYCLE_MIN = 1900
_CYCLE_MAX = 2100


def validate_panel(
    df: pd.DataFrame,
    required_cols: list[str],
    context: str,
) -> None:
    """Enforce five structural invariants on a panel DataFrame.

    Raises ValueError — naming ``context`` and the offending field — on the
    first detected invariant violation.
    Invariants
    ----------
    1. Required columns present  — all names in *required_cols* exist in df.
    2. No nulls in required cols — every required column is fully populated.
    3. cycle YYYY range          — integer values in [1900, 2100].
    4. bloc snake_case           — all bloc IDs are lowercase snake_case.
    5. Share columns in [0, 1]  — vote_share and turnout (when present) ∈ [0, 1].
    """
    # ── Invariant 1: required columns present ────────────────────────────────
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"{context}.columns: required columns missing: {missing}. "
            f"DataFrame has: {sorted(df.columns.tolist())}"
        )

    # ── Invariant 2: no nulls in required columns ─────────────────────────
    for col in required_cols:
        n_null = int(df[col].isna().sum())
        if n_null:
            raise ValueError(f"{context}.{col}: {n_null} null value(s) in required column")

    # ── Invariant 3: cycle is int YYYY in [1900, 2100] ───────────────────
    if "cycle" in df.columns:
        cycle_num = pd.to_numeric(df["cycle"], errors="coerce")
        non_numeric = df["cycle"].notna() & cycle_num.isna()
        if non_numeric.any():
            bad_vals = sorted(df.loc[non_numeric, "cycle"].astype(str).unique().tolist())
            raise ValueError(f"{context}.cycle: non-numeric value(s): {bad_vals}")
        non_int = cycle_num.notna() & (cycle_num % 1 != 0)
        if non_int.any():
            bad_vals = sorted(cycle_num[non_int].unique().tolist())
            raise ValueError(f"{context}.cycle: non-integer value(s): {bad_vals}")
        non_null = cycle_num.dropna()
        bad = non_null[(non_null < _CYCLE_MIN) | (non_null > _CYCLE_MAX)]
        if len(bad):
            raise ValueError(
                f"{context}.cycle: values outside [{_CYCLE_MIN}, {_CYCLE_MAX}]: "
                f"{sorted(bad.astype(int).unique().tolist())}"
            )

    # ── Invariant 4: bloc IDs are lowercase snake_case ────────────────────
    if "bloc" in df.columns:
        bad_blocs = [v for v in df["bloc"].dropna().unique() if not _SNAKE_RE.match(str(v))]
        if bad_blocs:
            raise ValueError(f"{context}.bloc: non-snake_case bloc ID(s): {sorted(bad_blocs)}")

    # ── Invariant 5: share columns in [0, 1] ─────────────────────────────
    for col in _SHARE_COLS:
        if col not in df.columns:
            continue
        share_num = pd.to_numeric(df[col], errors="coerce")
        non_numeric = df[col].notna() & share_num.isna()
        if non_numeric.any():
            bad_vals = sorted(df.loc[non_numeric, col].astype(str).unique().tolist())
            raise ValueError(f"{context}.{col}: non-numeric value(s): {bad_vals}")
        non_null = share_num.dropna()
        out = non_null[(non_null < 0.0) | (non_null > 1.0)]
        if len(out):
            raise ValueError(f"{context}.{col}: values outside [0, 1]: {sorted(out.unique().tolist())}")
