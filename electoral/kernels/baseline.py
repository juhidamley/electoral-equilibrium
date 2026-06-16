"""Baseline portfolio kernel — Week 2.

build_baseline_portfolio() orchestrates:
  1. Moment estimation (mu_race/religion/gender, Sigma) from the voter panel
  2. NEP×loyalty population-weighted coalition shares → baseline race weights
  3. mu_eff scalar via the three-stratum formula from CLAUDE.md
  4. BaselinePortfolioData payload construction

w0_i = (nep_share_i × loyalty_i) / Σ_j(nep_share_j × loyalty_j)
solve_baseline (min-variance QP) is for post-shock rebalancing only
(kernels/shock.py → optimization/cvx.py), not for computing the baseline.

Equal stratum weights (1/N) are used for religion and gender in the mu_eff
formula until kernels/raking.py produces IPF-calibrated v_R and g_G values.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd

from electoral.artifacts import BaselinePortfolioData
from electoral.config import PipelineConfig
from electoral.core.types import (
    CANONICAL_GENDERS,
    CANONICAL_RACES,
    CANONICAL_RELIGIONS,
    LAYER_WEIGHT_KEYS,
)
from electoral.models.ml_baseline import estimate_moments, ground_truth_winning_cycles

log = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_LAYER_WEIGHTS_PATH = _REPO_ROOT / "configs" / "layer_weights.json"

# Neutral prior assigned to blocs absent from the panel.
# 0.50 = maximum uncertainty; does not bias the optimizer toward or away from the bloc.
_ABSENT_MU: float = 0.50

# Source: Edison Research National Exit Poll 2024.
# These are shares of actual voters, not census population.
# Non-citizens and non-voters are excluded — correct for
# electoral modeling. Do not substitute census shares here.
DEFAULT_NEP_SHARES: dict[str, float] = {
    "african_american": 0.12,  # NEP 2024: 12% of electorate
    "asian": 0.05,             # NEP 2024: 5% of electorate
    "latino": 0.15,            # NEP 2024: 15% of electorate
    "other_race": 0.05,        # NEP 2024: 5% of electorate
    "white": 0.63,             # NEP 2024: 63% of electorate
}


def _load_layer_weights() -> dict[str, float]:
    with _LAYER_WEIGHTS_PATH.open() as f:
        raw = json.load(f)
    return {k: float(raw[k]) for k in LAYER_WEIGHT_KEYS}


def _impute_nan(mu: dict[str, float], context: str) -> dict[str, float]:
    """Replace NaN entries with _ABSENT_MU and emit one WARNING per bloc."""
    out: dict[str, float] = {}
    for bloc, v in mu.items():
        if isinstance(v, float) and math.isnan(v):
            log.warning(
                "baseline: %s[%r] is NaN (absent from panel); imputing %.2f",
                context,
                bloc,
                _ABSENT_MU,
            )
            out[bloc] = _ABSENT_MU
        else:
            out[bloc] = float(v)
    return out


def build_baseline_portfolio(
    config: PipelineConfig,
    panel_df: pd.DataFrame,
) -> BaselinePortfolioData:
    """Estimate moments, optimise race coalition, and construct BaselinePortfolioData.

    Parameters
    ----------
    config:
        Pipeline configuration.  ``config.party`` determines the vote-share
        direction (Republican flips to 1 − vote_share).  ``config.target`` is
        used as V_eq.
    panel_df:
        Cleaned voter panel with columns ``cycle`` (int), ``bloc`` (str),
        ``vote_share`` (float).  All three strata (race, religion, gender) should
        be present as bloc values; absent strata produce NaN mu entries that are
        imputed with the neutral prior (0.50) before validation.

    Returns
    -------
    BaselinePortfolioData
        Fully validated artifact.  ``method`` is ``"nep_loyalty_weighted"``.
    """
    # ── 1. Layer weights ───────────────────────────────────────────────────────
    try:
        layer_weights = _load_layer_weights()
    except FileNotFoundError:
        log.warning(
            "baseline: %s not found; using placeholder λ 0.50/0.30/0.20",
            _LAYER_WEIGHTS_PATH,
        )
        layer_weights = {"lambda_1": 0.50, "lambda_2": 0.30, "lambda_3": 0.20}

    # ── 2. Moment estimation ───────────────────────────────────────────────────
    winning = ground_truth_winning_cycles(config.party)
    moments = estimate_moments(panel_df, config.party, winning_cycles=winning)
    log.info(
        "baseline: party=%s  winning_cycles=%s  (n=%d)",
        config.party,
        moments.winning_cycles,
        len(moments.winning_cycles),
    )

    # ── 3. NaN imputation ─────────────────────────────────────────────────────
    mu_race = _impute_nan(moments.mu_race, "mu_race")
    mu_religion = _impute_nan(moments.mu_religion, "mu_religion")
    mu_gender = _impute_nan(moments.mu_gender, "mu_gender")

    # ── 4. NEP×loyalty population-weighted coalition shares ───────────────────
    # w0_i = (nep_share_i × loyalty_i) / Σ_j(nep_share_j × loyalty_j)
    # Population-weighted coalition shares. solve_baseline is for
    # post-shock rebalancing only, not for computing the baseline.
    target = config.target
    pop_shares = getattr(config, "bloc_population_shares", None) or DEFAULT_NEP_SHARES
    raw = {r: pop_shares[r] * mu_race[r] for r in CANONICAL_RACES}
    total = sum(raw.values())
    weights = {r: raw[r] / total for r in raw}
    method = "nep_loyalty_weighted"
    log.info("w0 from NEP×loyalty: %s", {k: round(v, 4) for k, v in weights.items()})

    # ── 5. mu_eff — three-stratum formula (CLAUDE.md §Demographic architecture) ─
    # v_R = 1/N_R and g_G = 1/N_G (equal stratum weights) until raking.py
    # produces IPF-calibrated weights.
    lam = layer_weights
    n_rel = len(CANONICAL_RELIGIONS)
    n_gen = len(CANONICAL_GENDERS)
    mu_eff = float(
        np.clip(
            lam["lambda_1"] * sum(weights[b] * mu_race[b] for b in CANONICAL_RACES)
            + lam["lambda_2"] * sum(mu_religion[r] / n_rel for r in CANONICAL_RELIGIONS)
            + lam["lambda_3"] * sum(mu_gender[g] / n_gen for g in CANONICAL_GENDERS),
            0.0,
            1.0,
        )
    )

    payload = BaselinePortfolioData(
        method=method,
        party=config.party,
        weights=weights,
        mu_race=mu_race,
        mu_religion=mu_religion,
        mu_gender=mu_gender,
        mu_eff=mu_eff,
        layer_weights={k: lam[k] for k in LAYER_WEIGHT_KEYS},
        target=target,
    )
    payload.validate()
    return payload
