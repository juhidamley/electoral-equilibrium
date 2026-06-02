"""Simple benchmark portfolio weights.

Two baselines used as sanity checks and paper comparisons against the
CVXPY optimizer:

  equal_weight_baseline  — uniform 1/n allocation across blocs
  value_weight_baseline  — proportional to each bloc's share of the
                           voting electorate

These are intentionally trivial: they require no solver and make no
assumptions beyond the bloc list and, optionally, a panel DataFrame
with turnout data.  The paper reports optimizer gains over both baselines.
"""

from __future__ import annotations

import logging

import pandas as pd

log = logging.getLogger(__name__)

# Documented approximate shares of the *eligible voting-age population*
# per demographic bloc (CLAUDE.md §Demographic architecture + Pew 2020).
# Used by value_weight_baseline when panel turnout data is absent.
# These are population shares, not vote shares.
_APPROX_ELECTORATE_SHARE: dict[str, float] = {
    # Race/ethnicity (CLAUDE.md)
    "african_american": 0.12,
    "latino": 0.11,
    "asian": 0.05,
    "white": 0.62,
    "other_race": 0.10,
    # Religion (Pew Religious Landscape 2020)
    "evangelical": 0.24,
    "catholic": 0.21,
    "protestant": 0.13,
    "secular": 0.26,
    "jewish": 0.02,
    "muslim": 0.01,
    "other_rel": 0.13,
    # Gender (U.S. Census / CPS 2020)
    "women": 0.52,
    "men": 0.47,
    "other_gender": 0.01,
}


def equal_weight_baseline(blocs: list[str]) -> dict[str, float]:
    """Return uniform 1/n coalition weights across *blocs*.

    Parameters
    ----------
    blocs:
        Ordered list of bloc identifiers.  Must be non-empty.

    Returns
    -------
    dict[str, float]
        Each key maps to exactly 1 / len(blocs); values sum to 1.0.

    Raises
    ------
    ValueError
        If *blocs* is empty.
    """
    if not blocs:
        raise ValueError("equal_weight_baseline: blocs must be non-empty.")
    w = 1.0 / len(blocs)
    return {b: w for b in blocs}


def value_weight_baseline(
    df: pd.DataFrame,
    blocs: list[str],
) -> dict[str, float]:
    """Return coalition weights proportional to each bloc's electorate share.

    Raw weight for each bloc is determined by the following priority:

    1. ``mean(turnout)`` × ``approx_pop_share``  — when *df* has a non-null
       ``turnout`` column **and** the bloc appears in the documented
       electorate-share table.  Captures both relative bloc size and its
       actual participation rate.
    2. ``approx_pop_share`` alone — when turnout data is absent for the bloc
       but the bloc is in the documented table.
    3. ``mean(turnout)`` alone — when the bloc has turnout data in *df* but
       is not in the documented table (e.g. novel or custom blocs).
    4. Uniform fallback (weight = 1) — when neither source is available.
       A WARNING is logged for each such bloc.

    All raw weights are normalized so the returned values sum to 1.0.

    Parameters
    ----------
    df:
        Panel DataFrame.  Only the ``bloc`` and ``turnout`` columns are
        consulted; all others are ignored.  May be empty or lack a
        ``turnout`` column entirely (falls through to priority 2).
    blocs:
        Ordered list of bloc identifiers for which to compute weights.
        Must be non-empty.

    Returns
    -------
    dict[str, float]
        Bloc coalition weights, values in [0, 1] summing to 1.0 ± 1e-9.
        A bloc with ``turnout=0`` and no pop_share entry receives weight 0.

    Raises
    ------
    ValueError
        If *blocs* is empty, if *blocs* contains duplicate entries, or if
        all computed raw weights are zero (e.g. every bloc has turnout=0
        and no documented electorate share).
    """
    if not blocs:
        raise ValueError("value_weight_baseline: blocs must be non-empty.")
    if len(blocs) != len(set(blocs)):
        dupes = [b for i, b in enumerate(blocs) if b in blocs[:i]]
        raise ValueError(
            f"value_weight_baseline: blocs contains duplicate(s): {sorted(set(dupes))}. "
            "Each bloc must appear at most once."
        )

    # ── Derive mean turnout per bloc from df ──────────────────────────────────
    turnout_by_bloc: dict[str, float] = {}
    if "turnout" in df.columns and not df.empty:
        t_col = pd.to_numeric(df["turnout"], errors="coerce")
        for bloc, grp_idx in df.groupby("bloc").groups.items():
            vals = t_col.loc[grp_idx].dropna()
            if not vals.empty:
                turnout_by_bloc[str(bloc)] = float(vals.mean())

    # ── Compute raw weight per bloc using priority order ──────────────────────
    raw: dict[str, float] = {}
    unknown: list[str] = []

    for bloc in blocs:
        pop_share = _APPROX_ELECTORATE_SHARE.get(bloc)
        turnout = turnout_by_bloc.get(bloc)

        if pop_share is not None and turnout is not None:
            raw[bloc] = pop_share * turnout  # priority 1
        elif pop_share is not None:
            raw[bloc] = pop_share  # priority 2
        elif turnout is not None:
            raw[bloc] = turnout  # priority 3
        else:
            raw[bloc] = 1.0  # priority 4 — uniform fallback
            unknown.append(bloc)

    if unknown:
        log.warning(
            "value_weight_baseline: no electorate share or turnout data for "
            "bloc(s) %s; assigning uniform fallback weight before normalization.",
            unknown,
        )

    # ── Normalize ─────────────────────────────────────────────────────────────
    total = sum(raw.values())
    if total <= 0:
        raise ValueError(
            "value_weight_baseline: all raw weights are zero or negative "
            f"(blocs={blocs}). This can happen when every bloc has turnout=0 "
            "and none appears in the documented electorate-share table."
        )
    return {b: raw[b] / total for b in blocs}
