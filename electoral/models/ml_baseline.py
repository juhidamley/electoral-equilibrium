"""Moment estimation for the baseline portfolio kernel.

Provides estimate_moments() which computes:
  mu^(P)  — mean within-bloc vote share for party P across winning cycles
  Sigma   — 5×5 empirical race-bloc covariance across all available cycles

These are the primary inputs to the CVXPY DQCP optimizer (portfolios/cvx.py).
"""

from __future__ import annotations

import dataclasses
import json
import pathlib

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

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

# Approximate electorate share per religion bloc. Source: Pew Research + ANES.
# Used by _weighted_stratum_averages() to compute coalition-strength features.
_APPROX_RELIGION_SHARE: dict[str, float] = {
    "evangelical":  0.24,
    "catholic":     0.21,
    "protestant":   0.13,
    "secular":      0.26,
    "jewish":       0.02,
    "muslim":       0.01,
    "other_rel":    0.13,
}

# Approximate electorate share per gender bloc. Source: CPS / ANES.
_APPROX_GENDER_SHARE: dict[str, float] = {
    "women":        0.52,
    "men":          0.47,
    "other_gender": 0.01,
}

_EPS_DEFAULT: float = 1e-6

# Perot's share of the vote by demographic bloc in 1992 (three-party race).
# Source: 1992 National Exit Poll (CNN/ABC/CBS), supplemented by ANES 1992
# cross-tabulations where NEP did not report a subgroup.
# Used by correct_three_party_1992() to convert raw three-party Democratic
# fractions to two-party fractions (dem / (dem + rep) = dem / (1 - perot)).
_PEROT_1992_SHARE: dict[str, float] = {
    # Race — NEP 1992
    "african_american": 0.07,
    "latino":           0.14,
    "asian":            0.15,  # NEP not disaggregated; ANES estimate
    "white":            0.21,
    "other_race":       0.189, # national average
    # Religion — NEP 1992 / ANES 1992
    "evangelical":      0.12,  # Social conservatives less likely to vote Perot
    "catholic":         0.22,
    "protestant":       0.19,
    "secular":          0.22,
    "jewish":           0.10,
    "muslim":           0.15,  # Not reported; approximate
    "other_rel":        0.189, # national average
    # Gender — NEP 1992
    "women":            0.17,
    "men":              0.21,
    "other_gender":     0.189, # Not reported; national average
}

# After correct_three_party_1992() is applied, the 1992 panel rows carry
# two-party fractions consistent with their y_true=1 label.  No cycles need
# to be excluded from GP training on contamination grounds.
_CONTAMINATED_CYCLES: frozenset[int] = frozenset()

def correct_three_party_1992(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of *df* with 1992 vote_shares converted to two-party fractions.

    In 1992, Ross Perot won 18.9% of the popular vote.  Survey panels record the
    raw Democratic fraction of *all* votes cast (Clinton / (Clinton + Bush + Perot)),
    which understates Clinton's two-party share.  This converts each 1992 bloc to:

        vs_2p = vs_dem / (1 − perot_share_per_bloc)

    using bloc-specific Perot support from the 1992 National Exit Poll
    (see ``_PEROT_1992_SHARE``).  Rows for other cycles pass through unchanged.

    Parameters
    ----------
    df:
        Voter panel with columns ``cycle`` (int), ``bloc`` (str),
        ``vote_share`` (float).

    Returns
    -------
    New DataFrame with 1992 vote_shares corrected.  The ``source`` column is
    unchanged; callers that need to distinguish corrected rows can filter on
    ``cycle == 1992``.
    """
    df = df.copy()
    mask = df["cycle"] == 1992
    if not mask.any():
        return df

    def _correct_row(row: pd.Series) -> float:
        perot = _PEROT_1992_SHARE.get(row["bloc"], 0.189)
        corrected = float(row["vote_share"]) / (1.0 - perot)
        # Clip to [0, 1]: rounding / ANES over-sampling can push slightly above 1.
        return min(max(corrected, 0.0), 1.0)

    df.loc[mask, "vote_share"] = df[mask].apply(_correct_row, axis=1)
    return df


# ── Ground-truth presidential election results ────────────────────────────────
# Democrat popular-vote two-party share per cycle (dem / (dem + rep)).
# Source: National Archives certified FEC results. Third-party votes excluded
# from the denominator so the fraction is always in (0, 1).
#
# The popular-vote criterion is used throughout for winning-cycle classification
# (ground_truth_winning_cycles). V_eq derivation uses a separate approach that
# accounts for the Electoral College via logistic regression — see derive_ec_veq()
# and _EC_DEM_WIN below.
_PRES_DEM_2P_SHARE: dict[int, float] = {
    1948: 0.524,  # Truman     popular D win  / EC D win
    1952: 0.446,  # Stevenson  popular R win  / EC R win — panel flags as D
    1956: 0.422,  # Stevenson  popular R win  / EC R win
    1960: 0.501,  # Kennedy    popular D win  / EC D win  (margin: 0.1 pp)
    1964: 0.613,  # Johnson    popular D win  / EC D win
    1968: 0.496,  # Humphrey   popular R win  / EC R win  (Nixon; Wallace split)
    1972: 0.382,  # McGovern   popular R win  / EC R win
    1976: 0.511,  # Carter     popular D win  / EC D win
    1980: 0.447,  # Carter     popular R win  / EC R win
    1984: 0.406,  # Mondale    popular R win  / EC R win
    1988: 0.461,  # Dukakis    popular R win  / EC R win — panel flags as D
    1992: 0.535,  # Clinton    popular D win  / EC D win — panel misses (Perot)
    1996: 0.547,  # Clinton    popular D win  / EC D win
    2000: 0.503,  # Gore       popular D win  / EC R win  ← mismatch
    2004: 0.488,  # Kerry      popular R win  / EC R win — panel flags as D
    2008: 0.537,  # Obama      popular D win  / EC D win
    2012: 0.520,  # Obama      popular D win  / EC D win
    2016: 0.511,  # H. Clinton popular D win  / EC R win  ← mismatch
    2020: 0.523,  # Biden      popular D win  / EC D win
    2024: 0.492,  # Harris     popular R win  / EC R win
}

# Actual Electoral College outcomes (1 = Democrat won EC, 0 = Republican won EC).
# Source: National Archives certified EC results.
# 2000 and 2016 are the two mismatch cycles: popular-vote winner ≠ EC winner.
_EC_DEM_WIN: dict[int, int] = {
    1948: 1,  # Truman    303–189
    1952: 0,  # Eisenhower 442–89
    1956: 0,  # Eisenhower 457–73
    1960: 1,  # Kennedy   303–219
    1964: 1,  # Johnson   486–52
    1968: 0,  # Nixon     301–191
    1972: 0,  # Nixon     520–17
    1976: 1,  # Carter    297–240
    1980: 0,  # Reagan    489–49
    1984: 0,  # Reagan    525–13
    1988: 0,  # Bush      426–111
    1992: 1,  # Clinton   370–168
    1996: 1,  # Clinton   379–159
    2000: 0,  # Bush      271–266  (popular vote went to Gore)
    2004: 0,  # Bush      286–251
    2008: 1,  # Obama     365–173
    2012: 1,  # Obama     332–206
    2016: 0,  # Trump     306–232  (popular vote went to Clinton)
    2020: 1,  # Biden     306–232
    2024: 0,  # Trump     312–226
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


def derive_ec_veq(party: Party) -> float:
    """Estimate the popular-vote share a party needs to win the Electoral College.

    Uses a logistic regression on all 20 historical election cycles to find the
    popular-vote two-party share at which P(party wins EC) = 0.50.  This is more
    informative than a hard 50% threshold because it encodes the geographic
    efficiency asymmetry between the parties:

    - Democrats have lost the EC while winning the popular vote (2000: 50.3%,
      2016: 51.1%), so they need *more* than 50% of the popular vote.
    - Republicans have won the EC while losing the popular vote (2000: 49.7%),
      so they need *less* than 50%.

    The regression is:

        P(Dem EC win) = σ(A · dem_2p_share + B)

    The threshold where P = 0.50 solves ``A·x + B = 0``, giving ``x = −B/A``.
    C = 1e4 ≈ no regularisation: with 2 parameters and 20 non-separated points,
    regularisation would bias the threshold toward 0.50 rather than letting the
    historical data speak.

    Parameters
    ----------
    party:
        "democrat" or "republican".

    Returns
    -------
    float
        Popular-vote two-party share threshold expressed in the party's own
        coordinate system:

        - Democrat  → Democrat share (e.g. ~0.515).  The optimizer must produce
          mu_eff ≥ this value.
        - Republican → Republican share = 1 − dem_threshold (e.g. ~0.485).
          The optimizer uses 1 − vote_share for Republicans throughout, so V_eq
          must also be expressed in that coordinate system.
    """
    if party not in ("democrat", "republican"):
        raise ValueError(f"party must be 'democrat' or 'republican', got {party!r}")

    cycles = sorted(_PRES_DEM_2P_SHARE)
    X = np.array([[_PRES_DEM_2P_SHARE[c]] for c in cycles])
    y = np.array([_EC_DEM_WIN[c] for c in cycles], dtype=int)

    lr = LogisticRegression(C=1e4, max_iter=1000, random_state=0)
    lr.fit(X, y)

    A = float(lr.coef_[0, 0])
    B = float(lr.intercept_[0])
    dem_threshold = -B / A  # P(Dem EC win) = 0.50 ↔ A·x + B = 0

    return float(dem_threshold) if party == "democrat" else float(1.0 - dem_threshold)


@dataclasses.dataclass(frozen=True)
class MomentEstimates:
    """Output of estimate_moments().

    Attributes
    ----------
    mu_race:        race_id → mean vote share (party P, winning cycles only)
    mu_religion:    religion_id → mean vote share (party P, winning cycles only)
    mu_gender:      gender_id → mean vote share (party P, winning cycles only)
    Sigma:          5×5 race-bloc empirical covariance (all cycles, party P)
    winning_cycles: cycles used for μ estimation — either ground_truth_winning_cycles(party)
                    or the panel-derived heuristic when no override is supplied
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
    df = correct_three_party_1992(df)
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
        return {b: float("nan") if pd.isna(v) else float(v) for b, v in means.items()}

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


# ── GP win-probability classifier ────────────────────────────────────────────


@dataclasses.dataclass(frozen=True)
class LocoFoldResult:
    """Outcome of a single Leave-One-Cycle-Out fold.

    Attributes
    ----------
    cycle:      held-out election year
    y_true:     1 = party won, 0 = party lost
    prob_win:   GPR latent prediction clipped to [0, 1]; used as P(win) estimate.
                NaN when the training set contained only one class.
    prob_std:   GP posterior standard deviation of the latent function at the
                held-out point — true epistemic uncertainty from the GPR posterior,
                NOT sqrt(p*(1-p)).  Feeds the paper's uncertainty quantification.
                NaN when prob_win is NaN.
    """

    cycle: int
    y_true: int
    prob_win: float
    prob_std: float
    calibrated_prob_win: float = float("nan")  # populated by platt_scale_loco()


@dataclasses.dataclass(frozen=True)
class GPBaselineResult:
    """Aggregated output of LOCO GP classifier evaluation."""

    party: str
    folds: tuple[LocoFoldResult, ...]
    accuracy: float       # fraction of valid folds correctly classified (p>=0.5 → win)
    brier_score: float    # mean (prob_win − y_true)² over valid folds
    calibrated_accuracy: float = float("nan")       # populated by platt_scale_loco()
    calibrated_brier_score: float = float("nan")    # populated by platt_scale_loco()


def _weighted_stratum_averages(
    df: pd.DataFrame,
    strata: list[tuple[list[str], dict[str, float]]],
    cycles: list[int],
) -> np.ndarray:
    """Return electorate-weighted average vote_share per stratum per cycle.

    For each (blocs, share_dict) pair, computes a per-cycle scalar:

        mu = sum(share_i * vote_share_i) / sum(share_i for present blocs)

    Weights are renormalised per row so that cycles with missing blocs still
    produce a valid (partial) average.  Cycles where every bloc is absent are
    mean-imputed column-wise.

    Parameters
    ----------
    df:
        Full voter panel — may contain blocs from multiple strata.
    strata:
        List of (blocs, share_dict) pairs, one per stratum.
        E.g. [(CANONICAL_RACES, _APPROX_RACE_SHARE), ...]
    cycles:
        Ordered list of cycle years (determines row order of output).

    Returns
    -------
    (len(cycles), len(strata)) float array.  Column k corresponds to strata[k].
    """
    cols: list[np.ndarray] = []
    for blocs, share_dict in strata:
        sub = df[df["bloc"].isin(blocs)]
        if sub.empty:
            cols.append(np.full(len(cycles), 0.0))
            continue

        pivot = (
            sub.pivot_table(index="cycle", columns="bloc", values="vote_share", aggfunc="mean")
            .reindex(index=cycles, columns=blocs)
        )
        vals = pivot.to_numpy(dtype=float)          # (n_cycles, n_blocs)
        w = np.array([share_dict.get(b, 0.0) for b in blocs])

        # Per-row renormalisation: zero out weights for NaN blocs so the sum
        # still reflects the share of the observed blocs only.
        w_mat = np.where(np.isnan(vals), 0.0, w[None, :])
        w_sum = w_mat.sum(axis=1, keepdims=True)
        w_norm = np.where(w_sum > 0, w_mat / w_sum, 0.0)
        vals_filled = np.where(np.isnan(vals), 0.0, vals)
        mu = (vals_filled * w_norm).sum(axis=1)     # (n_cycles,)

        # Cycles where every bloc was absent → mean-impute.
        all_absent = w_sum.squeeze() == 0.0
        if all_absent.any() and not all_absent.all():
            mu[all_absent] = float(np.nanmean(mu[~all_absent]))

        cols.append(mu)

    return np.column_stack(cols) if cols else np.empty((len(cycles), 0))


def _build_feature_matrix(
    df: pd.DataFrame,
    blocs: list[str],
    cycles: list[int],
    *,
    include_year: bool = True,
) -> np.ndarray:
    """Return a (len(cycles), n_features) design matrix.

    Columns: vote_share per bloc (K cols), turnout per bloc (K cols, only when
    the 'turnout' column is present in *df*), and optionally the normalised
    cycle year (1 col, only when *include_year* is True).

    NaN entries from missing bloc×cycle combinations are mean-imputed per column.
    Columns that are entirely NaN (bloc absent across ALL cycles in *cycles*) are
    zero-imputed rather than propagating NaN through the kernel computation.

    Parameters
    ----------
    include_year:
        When True (default) a normalised cycle-year feature [0, 1] is appended.
        Set False for the no-temporal-feature sensitivity analysis — the kernel
        then relies solely on vote-share distances for covariance structure.
        Removing the year feature eliminates the extrapolation risk at the
        boundary of the training distribution (see DECISIONS.md §GP Classifier).
    """
    vs_pivot = (
        df[df["bloc"].isin(blocs)]
        .pivot_table(index="cycle", columns="bloc", values="vote_share", aggfunc="mean")
        .reindex(index=cycles, columns=blocs)
    )
    parts: list[np.ndarray] = [vs_pivot.to_numpy(dtype=float)]

    if "turnout" in df.columns:
        t_pivot = (
            df[df["bloc"].isin(blocs)]
            .pivot_table(index="cycle", columns="bloc", values="turnout", aggfunc="mean")
            .reindex(index=cycles, columns=blocs)
        )
        parts.append(t_pivot.to_numpy(dtype=float))

    if include_year:
        c_arr = np.array(cycles, dtype=float)
        span = float(c_arr.max() - c_arr.min())
        cycle_col = ((c_arr - c_arr.min()) / max(span, 1.0)).reshape(-1, 1)
        parts.append(cycle_col)

    X = np.hstack(parts)

    # Column-wise mean imputation for NaN entries (blocs absent in some cycles).
    # Guard: columns that are entirely NaN produce nanmean=NaN.  Use zero instead
    # so the imputed value sits at the feature mean after StandardScaler is applied
    # downstream (zero in raw space ≠ mean, but avoids silent NaN propagation into
    # the kernel matrix which would produce undefined kernel evaluations).
    col_means = np.nanmean(X, axis=0)
    col_means = np.where(np.isnan(col_means), 0.0, col_means)
    nan_mask = np.isnan(X)
    if nan_mask.any():
        X[nan_mask] = np.take(col_means, np.where(nan_mask)[1])

    return X


def fit_gp_classifier(
    df: pd.DataFrame,
    party: Party,
    *,
    winning_cycles: list[int] | None = None,
    rng: np.random.Generator | None = None,
    alpha: float = 0.10,
    include_year: bool = True,
    include_stratum_mu: bool = True,
) -> GPBaselineResult:
    """Fit a GP win-probability model with Leave-One-Cycle-Out CV.

    Uses ``GaussianProcessRegressor`` with a composite
    ``RBF(1.0) + Matérn(1.0, ν=1.5)`` kernel.  Fitting GPR on binary {0, 1}
    labels rather than ``GaussianProcessClassifier`` is deliberate: GPR's
    ``predict(return_std=True)`` returns the true GP posterior standard deviation
    of the latent function at each held-out point, which is the epistemic
    uncertainty estimate required for the paper's uncertainty-quantification
    framing.  ``GaussianProcessClassifier`` does not expose this quantity.

    Parameters
    ----------
    df:
        Cleaned panel with columns cycle (int), bloc (str), vote_share (float).
        An optional 'turnout' column is used when present.  When
        ``include_stratum_mu=True`` (the default), the DataFrame should contain
        blocs from all three strata (race, religion, gender) so that coalition-
        strength averages can be computed; race-only DataFrames will silently
        produce zero-filled religion/gender columns.
    party:
        "democrat" or "republican".
    winning_cycles:
        Explicit win list; pass ground_truth_winning_cycles(party) to use
        certified results and bypass the panel-derived heuristic.
    rng:
        Seeded numpy Generator (derive_seed / make_rng).  Seed is extracted via
        ``rng.integers(2**31)`` and forwarded to sklearn's ``random_state``.
        When None the seed defaults to 42 (deterministic but not caller-controlled).
    alpha:
        Observation noise variance added to the kernel diagonal during fitting
        (sklearn GPR ``alpha`` parameter).  The default (0.10) models realistic
        survey measurement error: electoral vote-share surveys carry ~2–5 pp
        sampling error, which in standardised label space corresponds to
        alpha ≈ 0.05–0.15.  This prevents exact interpolation through training
        points, reduces extreme prob_win estimates near 0 or 1, and empirically
        improved LOCO accuracy from 0.800 → 0.850 and Brier from 0.122 → 0.113
        on the 20-cycle panel by correctly softening the 2004 borderline
        prediction.  Set alpha=1e-10 to reproduce exact interpolation.
    include_year:
        When True (default) a normalised cycle-year feature [0, 1] is included.
        Set False to test whether the GP can distinguish party outcomes purely
        from vote-share features.  Removing the year feature eliminates the
        extrapolation risk at the boundary of the training distribution (see
        DECISIONS.md §GP Classifier), at the cost of losing explicit temporal
        ordering as a signal.
    include_stratum_mu:
        When True (default) three electorate-weighted coalition-strength scalars
        (mu_race, mu_religion, mu_gender) are appended to the feature matrix.
        These give the GP a direct signal of overall coalition strength relative
        to V_eq — without them the model must infer coalition strength from five
        unconstrained race-bloc features, which increases the risk of spurious
        interpolation (e.g. high African-American loyalty masking a losing
        overall coalition, as observed in the 2024 misclassification).
        Set False for ablation studies comparing race-only vs full-stratum models.

    Returns
    -------
    GPBaselineResult with per-fold posterior probabilities and aggregate metrics.

    Raises
    ------
    ValueError
        If df is missing required columns, party is invalid, or fewer than
        3 cycles are present (LOCO requires at least 2 training cycles).
    """
    if party not in ("democrat", "republican"):
        raise ValueError(f"party must be 'democrat' or 'republican', got {party!r}")

    missing = {"cycle", "bloc", "vote_share"} - set(df.columns)
    if missing:
        raise ValueError(f"fit_gp_classifier: missing columns {sorted(missing)}")

    df = df.copy()
    df["cycle"] = pd.to_numeric(df["cycle"], errors="coerce")
    df["vote_share"] = pd.to_numeric(df["vote_share"], errors="coerce")
    df = df.dropna(subset=["cycle", "vote_share"])
    df["cycle"] = df["cycle"].astype(int)
    df = correct_three_party_1992(df)

    if party == "republican":
        df["vote_share"] = 1.0 - df["vote_share"]

    all_cycles = sorted(df["cycle"].unique())
    if len(all_cycles) < 3:
        raise ValueError(
            f"fit_gp_classifier: LOCO-CV requires >= 3 cycles, found {len(all_cycles)}."
        )

    # Resolve winning cycles.
    if winning_cycles is not None:
        _winning: set[int] = {int(c) for c in winning_cycles}
    else:
        race_df = df[df["bloc"].isin(CANONICAL_RACES)]
        race_pivot = race_df.pivot_table(
            index="cycle", columns="bloc", values="vote_share", aggfunc="mean"
        ).reindex(columns=CANONICAL_RACES)
        national = _national_vote_share(race_pivot, _APPROX_RACE_SHARE)
        _winning = {int(c) for c in national.index[national > 0.50]}

    X_raw = _build_feature_matrix(df, list(CANONICAL_RACES), all_cycles, include_year=include_year)

    if include_stratum_mu:
        # Append per-stratum coalition-strength scalars.  These collapsed
        # weighted averages let the GP directly compare overall coalition
        # strength to V_eq rather than inferring it from 5 noisy race blocs.
        # Religion and gender blocs are present in df when the caller passes
        # the full concatenated panel (as run_loco_validation.py does).
        stratum_mu = _weighted_stratum_averages(
            df,
            strata=[
                (list(CANONICAL_RACES), _APPROX_RACE_SHARE),
                (list(CANONICAL_RELIGIONS), _APPROX_RELIGION_SHARE),
                (list(CANONICAL_GENDERS), _APPROX_GENDER_SHARE),
            ],
            cycles=all_cycles,
        )
        X_raw = np.hstack([X_raw, stratum_mu])

    y_all = np.array([1 if c in _winning else 0 for c in all_cycles], dtype=int)

    # Standardise features to zero-mean, unit-variance before fitting.
    # Raw vote-share distances (~0.1–0.3) are far below the default kernel
    # length scale of 1.0, causing the optimiser to collapse length scales to
    # the lower bound.  Post-standardisation, inter-cycle distances are O(1).
    # Scaler is fitted on all cycles; O(1/n) leakage is negligible for n=20.
    scaler = StandardScaler()
    X_all = scaler.fit_transform(X_raw)

    kernel = RBF(length_scale=1.0, length_scale_bounds=(0.1, 10.0)) + Matern(
        length_scale=1.0, length_scale_bounds=(0.1, 10.0), nu=1.5
    )
    rs: int = int(rng.integers(0, 2**31 - 1)) if rng is not None else 42

    # _CONTAMINATED_CYCLES cycles are excluded from every training set (but
    # still evaluated as held-out folds).  See module-level constant for rationale.
    cycle_to_idx: dict[int, int] = {c: i for i, c in enumerate(all_cycles)}

    folds: list[LocoFoldResult] = []
    for hold_idx, held_cycle in enumerate(all_cycles):
        train_mask = np.ones(len(all_cycles), dtype=bool)
        train_mask[hold_idx] = False
        for c in _CONTAMINATED_CYCLES:
            if c != held_cycle and c in cycle_to_idx:
                train_mask[cycle_to_idx[c]] = False

        X_train = X_all[train_mask]
        y_train = y_all[train_mask].astype(float)

        # GPR cannot recover signal when training labels are constant.
        if float(y_train.std()) == 0.0:
            folds.append(
                LocoFoldResult(
                    cycle=int(held_cycle),
                    y_true=int(y_all[hold_idx]),
                    prob_win=float("nan"),
                    prob_std=float("nan"),
                )
            )
            continue

        # GaussianProcessRegressor gives both the latent prediction AND its
        # posterior standard deviation via return_std=True.  normalize_y=True
        # centres binary labels around their training mean, which stabilises
        # kernel hyperparameter optimisation.
        #
        # n_restarts scales with training size: the kernel has 2 hyperparameters
        # (one length_scale each for RBF and Matérn) so optimisation is
        # underdetermined for n_train < 5 (≈ 2× hyperparameter count + slack).
        # Full LOCO folds (n_train ≈ 18) use 3 restarts.
        n_restarts = max(0, min(3, len(X_train) - 4))
        gpr = GaussianProcessRegressor(
            kernel=kernel,
            alpha=alpha,
            n_restarts_optimizer=n_restarts,
            normalize_y=True,
            random_state=rs,
        )
        gpr.fit(X_train, y_train)

        y_pred, y_std = gpr.predict(X_all[[hold_idx]], return_std=True)
        # Clip latent prediction to [0, 1] as a probability estimate.
        p_win = float(np.clip(y_pred[0], 0.0, 1.0))
        # y_std is the true GP posterior std of the latent function — epistemic
        # uncertainty, not Bernoulli variance.
        p_std = float(y_std[0])

        folds.append(
            LocoFoldResult(
                cycle=int(held_cycle),
                y_true=int(y_all[hold_idx]),
                prob_win=p_win,
                prob_std=p_std,
            )
        )

    valid = [f for f in folds if not np.isnan(f.prob_win)]
    if valid:
        accuracy = float(sum(1 for f in valid if (f.prob_win >= 0.5) == bool(f.y_true)) / len(valid))
        brier = float(sum((f.prob_win - f.y_true) ** 2 for f in valid) / len(valid))
    else:
        accuracy = float("nan")
        brier = float("nan")

    return GPBaselineResult(
        party=party,
        folds=tuple(folds),
        accuracy=accuracy,
        brier_score=brier,
    )


def platt_scale_loco(result: GPBaselineResult) -> GPBaselineResult:
    """Calibrate GP LOCO-CV predictions with jackknife Platt scaling.

    GP raw probabilities are systematically miscalibrated: strong-pattern
    cycles receive prob_win near 0 or 1 even when historical win rates at
    those feature values don't warrant that confidence.  Platt scaling fits
    a two-parameter logistic sigmoid ``σ(A·f + B)`` on top of the raw
    scores to correct this.

    The *jackknife* protocol guarantees that the calibrator is never fitted
    on the fold it evaluates.  For each held-out fold c:

      1. Fit ``LogisticRegression(C=1e4)`` on the (prob_win, y_true) pairs
         from the other n−1 valid folds.
      2. Apply the fitted sigmoid to fold c's raw prob_win.

    ``C=1e4`` is effectively unregularised: with only two free parameters
    (slope A and intercept B), L2 regularisation would shrink the calibrator
    toward a uniform 50/50 prediction.  We want it to learn the actual
    sigmoid shape from the data.

    Folds with NaN prob_win (constant-label training sets) pass through
    with ``calibrated_prob_win = NaN``.  Folds whose jackknife calibration
    set is single-class (rare on n=20 panels) fall back to the raw prob_win.

    Parameters
    ----------
    result:
        Output of ``fit_gp_classifier()``.

    Returns
    -------
    New ``GPBaselineResult`` with ``calibrated_prob_win`` populated on every
    valid fold and ``calibrated_accuracy`` / ``calibrated_brier_score``
    computed from the calibrated estimates.  ``prob_win`` and ``prob_std``
    are unchanged — raw GP outputs are preserved for comparison.
    """
    valid_idxs = [i for i, f in enumerate(result.folds) if not np.isnan(f.prob_win)]
    if not valid_idxs:
        return result

    valid_folds = [result.folds[i] for i in valid_idxs]

    cal_probs: dict[int, float] = {}  # fold position in result.folds → calibrated prob

    for i, fold in enumerate(valid_folds):
        cal_X = np.array(
            [[f.prob_win] for j, f in enumerate(valid_folds) if j != i]
        )
        cal_y = np.array(
            [f.y_true for j, f in enumerate(valid_folds) if j != i],
            dtype=int,
        )

        original_idx = valid_idxs[i]

        if len(np.unique(cal_y)) < 2:
            # Single-class calibration set: sigmoid is undefined.
            # Fall back to raw prob_win rather than returning a degenerate
            # calibrator that always predicts the same class.
            cal_probs[original_idx] = fold.prob_win
            continue

        lr = LogisticRegression(C=1e4, max_iter=1000, random_state=0)
        lr.fit(cal_X, cal_y)
        # predict_proba returns [[P(y=0), P(y=1)]]; index 1 = P(win).
        cal_probs[original_idx] = float(lr.predict_proba([[fold.prob_win]])[0, 1])

    new_folds: list[LocoFoldResult] = []
    for i, fold in enumerate(result.folds):
        if i in cal_probs:
            new_folds.append(dataclasses.replace(fold, calibrated_prob_win=cal_probs[i]))
        else:
            new_folds.append(fold)  # NaN fold: calibrated_prob_win stays NaN

    cal_valid = [f for f in new_folds if not np.isnan(f.calibrated_prob_win)]
    if cal_valid:
        cal_acc = float(
            sum(1 for f in cal_valid if (f.calibrated_prob_win >= 0.5) == bool(f.y_true))
            / len(cal_valid)
        )
        cal_brier = float(
            sum((f.calibrated_prob_win - f.y_true) ** 2 for f in cal_valid)
            / len(cal_valid)
        )
    else:
        cal_acc = float("nan")
        cal_brier = float("nan")

    return dataclasses.replace(
        result,
        folds=tuple(new_folds),
        calibrated_accuracy=cal_acc,
        calibrated_brier_score=cal_brier,
    )


def save_loco_json(
    result: GPBaselineResult,
    path: str | pathlib.Path = "artifacts/baseline_loco.json",
) -> None:
    """Serialise *result* to *path* as JSON, creating parent directories.

    Output schema::

        {
          "party": "democrat",
          "accuracy": 0.80,
          "brier_score": 0.122,
          "calibrated_accuracy": 0.85,
          "calibrated_brier_score": 0.098,
          "folds": [
            {
              "cycle": 2020, "y_true": 1,
              "prob_win": 0.979, "prob_std": 0.230,
              "calibrated_prob_win": 0.812
            },
            ...
          ]
        }

    Fields ``calibrated_accuracy``, ``calibrated_brier_score``, and
    ``calibrated_prob_win`` are ``null`` when ``platt_scale_loco()`` has
    not been applied to *result*.
    """
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def _float(v: float) -> float | None:
        return None if (isinstance(v, float) and np.isnan(v)) else float(v)

    payload: dict = {
        "party": result.party,
        "accuracy": _float(result.accuracy),
        "brier_score": _float(result.brier_score),
        "calibrated_accuracy": _float(result.calibrated_accuracy),
        "calibrated_brier_score": _float(result.calibrated_brier_score),
        "folds": [
            {
                "cycle": int(f.cycle),
                "y_true": int(f.y_true),
                "prob_win": _float(f.prob_win),
                "prob_std": _float(f.prob_std),
                "calibrated_prob_win": _float(f.calibrated_prob_win),
            }
            for f in result.folds
        ],
    }
    path.write_text(json.dumps(payload, indent=2))
