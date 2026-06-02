"""Moment estimation for the baseline portfolio kernel.

Provides estimate_moments() which computes:
  mu^(P)  — mean within-bloc vote share for party P across winning cycles
  Sigma   — 5×5 empirical race-bloc covariance across all available cycles

These are the primary inputs to the CVXPY DQCP optimizer (portfolios/cvx.py).
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd

from electoral.core.types import (
    CANONICAL_GENDERS,
    CANONICAL_RACES,
    CANONICAL_RELIGIONS,
    Party,
)

# Approximate national electorate share per race bloc (CLAUDE.md §Demographic
# architecture). Used ONLY to compute a per-cycle national two-party vote share
# when identifying winning cycles — never used as optimizer decision variables.
_APPROX_RACE_SHARE: dict[str, float] = {
    "african_american": 0.12,
    "latino": 0.11,
    "asian": 0.05,
    "white": 0.62,
    "other_race": 0.10,
}

_EPS_DEFAULT: float = 1e-6


@dataclasses.dataclass(frozen=True)
class MomentEstimates:
    """Output of estimate_moments().

    Attributes
    ----------
    mu_race:        race_id → mean vote share (party P, winning cycles only)
    mu_religion:    religion_id → mean vote share (party P, winning cycles only)
    mu_gender:      gender_id → mean vote share (party P, winning cycles only)
    Sigma:          5×5 race-bloc empirical covariance (all cycles, party P)
    winning_cycles: cycles where national race-weighted share > 0.50
    race_blocs:     ordered race bloc labels for Sigma rows/cols (= CANONICAL_RACES)
    """

    mu_race: dict[str, float]
    mu_religion: dict[str, float]
    mu_gender: dict[str, float]
    Sigma: np.ndarray  # shape (5, 5), dtype float64
    winning_cycles: list[int]
    race_blocs: list[str]


def _national_vote_share(
    race_pivot: pd.DataFrame,
    approx_share: dict[str, float],
) -> pd.Series:
    """Return approximate national two-party vote share per cycle.

    Weights race-bloc vote shares by approximate electorate shares.
    Renormalizes weights per row to handle cycles where some blocs are absent.
    Returns NaN for cycles where no race-bloc data is available.
    """
    weights = pd.Series(approx_share).reindex(race_pivot.columns, fill_value=0.0)

    def _weighted_mean(row: pd.Series) -> float:
        valid = row.dropna()
        if valid.empty:
            return float("nan")
        w = weights.loc[valid.index]
        total_w = float(w.sum())
        if total_w == 0.0:
            return float("nan")
        return float((valid * w).sum() / total_w)

    return race_pivot.apply(_weighted_mean, axis=1)


def estimate_moments(
    df: pd.DataFrame,
    party: Party,
    *,
    epsilon: float = _EPS_DEFAULT,
) -> MomentEstimates:
    """Estimate bloc-level vote-share moments for the given party.

    Parameters
    ----------
    df:
        Cleaned voter panel with columns: cycle (int), bloc (str),
        vote_share (float in [0, 1]). Religion and gender blocs may be
        present alongside race blocs. Rows with null vote_share are
        excluded from all computations.
    party:
        "democrat" or "republican". For "democrat", vote_share is used
        directly; for "republican", 1 - vote_share is used throughout.
    epsilon:
        PSD safeguard magnitude. If the minimum eigenvalue of Sigma is
        negative, (-min_eig + epsilon) * I is added to Sigma.

    Returns
    -------
    MomentEstimates
        mu_race/religion/gender: mean share over winning cycles only.
        Sigma: 5×5 race empirical covariance over ALL available cycles.
        winning_cycles: cycles where national race-weighted share > 0.50.
        race_blocs: CANONICAL_RACES (authoritative row/col order for Sigma).

    Raises
    ------
    ValueError
        If df is missing required columns or party is invalid.
    """
    if party not in ("democrat", "republican"):
        raise ValueError(f"party must be 'democrat' or 'republican', got {party!r}")

    required = {"cycle", "bloc", "vote_share"}
    missing_cols = required - set(df.columns)
    if missing_cols:
        raise ValueError(f"estimate_moments: missing columns {sorted(missing_cols)}")

    # ── 1. Prepare: coerce types, drop null vote_shares ───────────────────────
    df = df.copy()
    df["cycle"] = pd.to_numeric(df["cycle"], errors="coerce")
    df["vote_share"] = pd.to_numeric(df["vote_share"], errors="coerce")
    df = df.dropna(subset=["cycle", "vote_share"]).copy()
    df["cycle"] = df["cycle"].astype(int)

    # ── 2. Flip vote_share for Republican ─────────────────────────────────────
    if party == "republican":
        df["vote_share"] = 1.0 - df["vote_share"]

    # ── 3. Identify winning cycles ─────────────────────────────────────────────
    # National vote share per cycle = weighted mean across race blocs.
    # Winning: national > 0.50 (absolute majority of two-party vote).
    race_df = df[df["bloc"].isin(CANONICAL_RACES)]
    race_pivot = race_df.pivot_table(
        index="cycle", columns="bloc", values="vote_share", aggfunc="mean"
    ).reindex(columns=CANONICAL_RACES)

    national = _national_vote_share(race_pivot, _APPROX_RACE_SHARE)
    winning_cycles: list[int] = sorted(int(c) for c in national.index[national > 0.50])

    # ── 4. mu^(P) per stratum — mean over winning cycles only ─────────────────
    def _mean_over_winning(blocs: list[str]) -> dict[str, float]:
        sub = df[df["bloc"].isin(blocs) & df["cycle"].isin(winning_cycles)]
        if sub.empty:
            return {b: float("nan") for b in blocs}
        means = sub.groupby("bloc")["vote_share"].mean().reindex(blocs)
        return {b: float(v) for b, v in means.items()}

    mu_race = _mean_over_winning(CANONICAL_RACES)
    mu_religion = _mean_over_winning(CANONICAL_RELIGIONS)
    mu_gender = _mean_over_winning(CANONICAL_GENDERS)

    # ── 5. Sigma — 5×5 race empirical covariance over ALL cycles ──────────────
    # ddof=1 (unbiased); blocs present in only one cycle produce NaN variance.
    cov_df = race_pivot.cov(ddof=1).reindex(index=CANONICAL_RACES, columns=CANONICAL_RACES)
    Sigma = cov_df.to_numpy(dtype=float)
    # Zero-fill NaN entries (arise when a bloc appears in fewer than 2 cycles).
    # This can break PSD; the safeguard below restores it.
    Sigma = np.where(np.isnan(Sigma), 0.0, Sigma)

    # ── 6. PSD safeguard ───────────────────────────────────────────────────────
    min_eig = float(np.linalg.eigvalsh(Sigma).min())
    if min_eig < 0.0:
        Sigma = Sigma + (-min_eig + epsilon) * np.eye(len(CANONICAL_RACES))

    return MomentEstimates(
        mu_race=mu_race,
        mu_religion=mu_religion,
        mu_gender=mu_gender,
        Sigma=Sigma,
        winning_cycles=winning_cycles,
        race_blocs=list(CANONICAL_RACES),
    )
