"""Baseline portfolio kernel — Week 2.

build_baseline_portfolio() orchestrates:
  1. Moment estimation (mu_race/religion/gender, Sigma) from the voter panel
  2. Min-variance QP via solve_baseline → optimal race coalition weights
  3. mu_eff scalar via the three-stratum formula from CLAUDE.md
  4. BaselinePortfolioData payload construction

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
from electoral.portfolios.cvx import solve_baseline

log = logging.getLogger(__name__)

_LAYER_WEIGHTS_PATH = Path("configs/layer_weights.json")

# Neutral prior assigned to blocs absent from the panel.
# 0.50 = maximum uncertainty; does not bias the optimizer toward or away from the bloc.
_ABSENT_MU: float = 0.50


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
        Fully validated artifact.  ``method`` is ``"cvxpy_minvar"`` on success
        or ``"equal_weight_fallback"`` if the QP is infeasible after retries.
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

    # ── 4. Min-variance QP ────────────────────────────────────────────────────
    target = config.target
    try:
        weights = solve_baseline(mu_race, moments.Sigma, target=target)
        method = "cvxpy_minvar"
    except ValueError as exc:
        log.warning(
            "baseline: solve_baseline infeasible at target=%.4f (%s); "
            "falling back to equal weights",
            target,
            exc,
        )
        weights = {b: 1.0 / len(CANONICAL_RACES) for b in CANONICAL_RACES}
        method = "equal_weight_fallback"

    # ── 5. mu_eff — three-stratum formula (CLAUDE.md §Demographic architecture) ─
    # v_R = 1/N_R and g_G = 1/N_G (equal stratum weights) until raking.py
    # produces IPF-calibrated weights.
    lam = layer_weights
    n_rel = len(CANONICAL_RELIGIONS)
    n_gen = len(CANONICAL_GENDERS)
    mu_eff = float(np.clip(
        lam["lambda_1"] * sum(weights[b] * mu_race[b] for b in CANONICAL_RACES)
        + lam["lambda_2"] * sum(mu_religion[r] / n_rel for r in CANONICAL_RELIGIONS)
        + lam["lambda_3"] * sum(mu_gender[g] / n_gen for g in CANONICAL_GENDERS),
        0.0,
        1.0,
    ))

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
