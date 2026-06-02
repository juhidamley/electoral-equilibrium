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

# ── Ground-truth presidential election results ────────────────────────────────
# Democrat popular-vote two-party share per cycle (dem / (dem + rep)).
# Source: National Archives certified FEC results. Third-party votes excluded
# from the denominator so the fraction is always in (0, 1).
#
# Ambiguous cycles (popular vote ≠ Electoral College outcome):
#   2000: Gore won popular vote (50.3%) but Bush won EC.
#   2016: H. Clinton won popular vote (51.1%) but Trump won EC.
#   Both are treated as Democratic wins under the popular-vote criterion used
#   here. Switch to EC-based ground truth once Prof. Espinosa confirms which
#   criterion V_eq should reflect (ESPINOSA.md §Q2.3).
_PRES_DEM_2P_SHARE: dict[int, float] = {
    1948: 0.524,  # Truman     D win
    1952: 0.446,  # Stevenson  R win — panel incorrectly flags as D
    1956: 0.422,  # Stevenson  R win
    1960: 0.501,  # Kennedy    D win (margin: 0.1 pp)
    1964: 0.613,  # Johnson    D win
    1968: 0.496,  # Humphrey   R win (Nixon; Wallace split)
    1972: 0.382,  # McGovern   R win
    1976: 0.511,  # Carter     D win
    1980: 0.447,  # Carter     R win
    1984: 0.406,  # Mondale    R win
    1988: 0.461,  # Dukakis    R win — panel incorrectly flags as D
    1992: 0.535,  # Clinton    D win — panel misses due to 1−dem_share Perot flip
    1996: 0.547,  # Clinton    D win
    2000: 0.503,  # Gore       D win (popular); Bush won EC
    2004: 0.488,  # Kerry      R win — panel incorrectly flags as D
    2008: 0.537,  # Obama      D win
    2012: 0.520,  # Obama      D win
    2016: 0.511,  # H. Clinton D win (popular); Trump won EC
    2020: 0.523,  # Biden      D win
    2024: 0.492,  # Harris     R win
}


def ground_truth_winning_cycles(party: Party, threshold: float = 0.50) -> list[int]:
    """Return cycles where *party* won the popular-vote two-party share.

    Uses the hardcoded _PRES_DEM_2P_SHARE table rather than deriving from the
    voter panel, fixing four known panel-derived misclassifications:

      - 1952, 1988, 2004 : panel overestimates Dem share → false D wins
      - 1992             : Perot 1-flip artifact → Clinton win missed

    Parameters
    ----------
    party:
        "democrat" or "republican".
    threshold:
        Win threshold (default 0.50 = majority of two-party vote).

    Returns
    -------
    Sorted list of cycle years where party exceeded *threshold*.
    """
    if party not in ("democrat", "republican"):
        raise ValueError(f"party must be 'democrat' or 'republican', got {party!r}")
    if party == "democrat":
        return sorted(c for c, s in _PRES_DEM_2P_SHARE.items() if s > threshold)
    return sorted(c for c, s in _PRES_DEM_2P_SHARE.items() if s <= threshold)


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


def psd_repair(cov: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Return a copy of *cov* that is positive semi-definite.

    If the minimum eigenvalue of *cov* is negative, adds
    ``(-min_eig + eps) * I`` so that all eigenvalues are >= ``eps``.
    Returns *cov* unchanged (zero-copy) when it is already PSD.

    Parameters
    ----------
    cov:
        Square, symmetric matrix (e.g. a sample covariance matrix).
    eps:
        Minimum eigenvalue guarantee after repair.  Defaults to 1e-6.
    """
    min_eig = float(np.linalg.eigvalsh(cov).min())
    if min_eig < 0.0:
        return cov + (-min_eig + eps) * np.eye(cov.shape[0])
    return cov


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
    winning_cycles: list[int] | None = None,
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
    winning_cycles:
        Optional explicit list of cycle years to treat as wins for *party*.
        When provided, the panel-derived national vote share heuristic is
        skipped entirely.  Pass ``ground_truth_winning_cycles(party)`` to
        use certified election results and fix the four known panel-derived
        misclassifications (1952, 1988, 1992, 2004).
        When None (default), winning cycles are derived from the panel —
        acceptable for synthetic/test panels that lack a real-data anchor.
    epsilon:
        PSD safeguard magnitude. If the minimum eigenvalue of Sigma is
        negative, (-min_eig + epsilon) * I is added to Sigma.

    Returns
    -------
    MomentEstimates
        mu_race/religion/gender: mean share over winning cycles only.
        Sigma: 5×5 race empirical covariance over ALL available cycles.
        winning_cycles: cycles used for mu estimation.
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

    # ── 3. Race pivot — needed for both winning-cycle derivation and Sigma ───────
    race_df = df[df["bloc"].isin(CANONICAL_RACES)]
    race_pivot = race_df.pivot_table(
        index="cycle", columns="bloc", values="vote_share", aggfunc="mean"
    ).reindex(columns=CANONICAL_RACES)

    # ── 4. Identify winning cycles ─────────────────────────────────────────────
    # Prefer an explicit ground-truth list (ground_truth_winning_cycles(party))
    # over panel-derived estimation, which produces four known errors caused by
    # survey-weighting artifacts and the 1992 Perot two-party-flip.
    if winning_cycles is not None:
        _winning: list[int] = sorted(int(c) for c in winning_cycles)
    else:
        national = _national_vote_share(race_pivot, _APPROX_RACE_SHARE)
        _winning = sorted(int(c) for c in national.index[national > 0.50])

    # ── 5. mu^(P) per stratum — mean over winning cycles only ─────────────────
    def _mean_over_winning(blocs: list[str]) -> dict[str, float]:
        sub = df[df["bloc"].isin(blocs) & df["cycle"].isin(_winning)]
        if sub.empty:
            return {b: float("nan") for b in blocs}
        means = sub.groupby("bloc")["vote_share"].mean().reindex(blocs)
        return {b: float(v) for b, v in means.items()}

    mu_race = _mean_over_winning(CANONICAL_RACES)
    mu_religion = _mean_over_winning(CANONICAL_RELIGIONS)
    mu_gender = _mean_over_winning(CANONICAL_GENDERS)

    # ddof=1 (unbiased); blocs present in only one cycle produce NaN variance.
    cov_df = race_pivot.cov(ddof=1).reindex(index=CANONICAL_RACES, columns=CANONICAL_RACES)
    Sigma = cov_df.to_numpy(dtype=float)
    # Zero-fill NaN entries (arise when a bloc appears in fewer than 2 cycles).
    # This can break PSD; the safeguard below restores it.
    Sigma = np.where(np.isnan(Sigma), 0.0, Sigma)

    # ── 7. PSD safeguard ───────────────────────────────────────────────────────
    Sigma = psd_repair(Sigma, eps=epsilon)

    return MomentEstimates(
        mu_race=mu_race,
        mu_religion=mu_religion,
        mu_gender=mu_gender,
        Sigma=Sigma,
        winning_cycles=_winning,
        race_blocs=list(CANONICAL_RACES),
    )
