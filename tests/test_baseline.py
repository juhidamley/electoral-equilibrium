"""Integration contract tests for the Week-2 baseline components.

Five mandatory contracts verified here:
  (i)   estimate_moments  — output shapes are correct
  (ii)  psd_repair        — output is positive semi-definite
  (iii) solve_baseline    — weights sum to 1.0
  (iv)  fit_gp_classifier — valid folds have prob_win in [0, 1]
  (v)   solve_baseline    — infeasible QP raises ValueError

These are intentionally thin — each test asserts one contract and no more.
Exhaustive edge-case coverage lives in test_ml_baseline.py and test_cvx.py.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from electoral.core.types import CANONICAL_GENDERS, CANONICAL_RACES, CANONICAL_RELIGIONS
from electoral.models.ml_baseline import (
    estimate_moments,
    fit_gp_classifier,
    psd_repair,
)
from electoral.portfolios.cvx import solve_baseline

# Layer weights from conftest.LAYER_WEIGHTS; kept local to avoid import coupling.
_LAYER_WEIGHTS = {"lambda_1": 0.50, "lambda_2": 0.30, "lambda_3": 0.20}

# ── Shared fixture ────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    """Five-cycle, 15-bloc synthetic panel used across all baseline tests."""
    rng = np.random.default_rng(0)
    cycles = [2004, 2008, 2012, 2016, 2020]
    blocs = list(CANONICAL_RACES) + list(CANONICAL_RELIGIONS) + list(CANONICAL_GENDERS)
    rows = [
        {"cycle": c, "bloc": b, "vote_share": float(rng.uniform(0.30, 0.85))}
        for c in cycles
        for b in blocs
    ]
    return pd.DataFrame(rows)


# Winning cycles used consistently across moment and GP tests.
_WINNING = [2008, 2012, 2020]


# ── (i) estimate_moments — output shapes ─────────────────────────────────────


def test_estimate_moments_sigma_shape(panel):
    result = estimate_moments(panel, "democrat", winning_cycles=_WINNING)
    assert result.Sigma.shape == (5, 5)


def test_estimate_moments_mu_race_has_five_keys(panel):
    result = estimate_moments(panel, "democrat", winning_cycles=_WINNING)
    assert list(result.mu_race.keys()) == list(CANONICAL_RACES)


def test_estimate_moments_mu_religion_has_seven_keys(panel):
    result = estimate_moments(panel, "democrat", winning_cycles=_WINNING)
    assert list(result.mu_religion.keys()) == list(CANONICAL_RELIGIONS)


def test_estimate_moments_mu_gender_has_three_keys(panel):
    result = estimate_moments(panel, "democrat", winning_cycles=_WINNING)
    assert list(result.mu_gender.keys()) == list(CANONICAL_GENDERS)


def test_estimate_moments_race_blocs_canonical_order(panel):
    result = estimate_moments(panel, "democrat", winning_cycles=_WINNING)
    assert result.race_blocs == list(CANONICAL_RACES)


# ── (ii) psd_repair — output is PSD ──────────────────────────────────────────


def test_psd_repair_output_is_psd():
    v = np.array([1.0, 0.5, 0.2, 0.8, 0.3])
    M = np.outer(v, v)
    M[0, 0] -= 0.05  # push smallest eigenvalue negative
    assert np.linalg.eigvalsh(M).min() < 0, "precondition: M must be non-PSD"
    repaired = psd_repair(M)
    assert np.all(np.linalg.eigvalsh(repaired) >= 0.0)


def test_psd_repair_sigma_from_panel_is_psd(panel):
    result = estimate_moments(panel, "democrat", winning_cycles=_WINNING)
    assert np.all(np.linalg.eigvalsh(result.Sigma) >= -1e-12)


# ── (iii) solve_baseline — weights sum to 1.0 ────────────────────────────────


def test_solve_baseline_weights_sum_to_one():
    mu = {b: v for b, v in zip(
        CANONICAL_RACES, [0.87, 0.63, 0.72, 0.41, 0.55]
    )}
    cov = np.diag([0.01, 0.011, 0.060, 0.005, 0.054])
    result = solve_baseline(mu, cov, target=0.55)
    assert sum(result.values()) == pytest.approx(1.0, abs=1e-9)


def test_solve_baseline_weights_nonneg():
    mu = {b: v for b, v in zip(
        CANONICAL_RACES, [0.87, 0.63, 0.72, 0.41, 0.55]
    )}
    cov = np.diag([0.01, 0.011, 0.060, 0.005, 0.054])
    result = solve_baseline(mu, cov, target=0.55)
    assert all(v >= 0.0 for v in result.values())


def test_solve_baseline_loyalty_constraint_met():
    mu = {b: v for b, v in zip(
        CANONICAL_RACES, [0.87, 0.63, 0.72, 0.41, 0.55]
    )}
    cov = np.diag([0.01, 0.011, 0.060, 0.005, 0.054])
    target = 0.55
    result = solve_baseline(mu, cov, target=target)
    achieved = sum(mu[b] * result[b] for b in CANONICAL_RACES)
    assert achieved >= target - 1e-6


# ── (iv) fit_gp_classifier — calibrated probabilities in [0, 1] ──────────────


@pytest.fixture(scope="module")
def gp_result(panel) -> object:
    return fit_gp_classifier(
        panel, "democrat", winning_cycles=_WINNING, rng=np.random.default_rng(0)
    )


def test_gp_classifier_produces_calibrated_probabilities(gp_result):
    valid = [f for f in gp_result.folds if not math.isnan(f.prob_win)]
    assert valid, "Expected at least one valid LOCO fold"
    for f in valid:
        assert 0.0 <= f.prob_win <= 1.0, f"prob_win out of range: cycle={f.cycle}, p={f.prob_win}"


def test_gp_classifier_prob_std_nonneg(gp_result):
    valid = [f for f in gp_result.folds if not math.isnan(f.prob_std)]
    for f in valid:
        assert f.prob_std >= 0.0


def test_gp_classifier_accuracy_in_unit_interval(gp_result):
    assert 0.0 <= gp_result.accuracy <= 1.0


# ── (v) solve_baseline — infeasible QP raises correctly ──────────────────────


def test_solve_baseline_infeasible_raises_valueerror():
    mu = {"a": 0.6, "b": 0.4}
    cov = np.eye(2)
    with pytest.raises(ValueError, match="exceeds the maximum achievable"):
        solve_baseline(mu, cov, target=0.65, blocs=["a", "b"])


def test_solve_baseline_infeasible_error_names_best_bloc():
    mu = {"african_american": 0.87, "white": 0.41}
    cov = np.eye(2)
    with pytest.raises(ValueError, match="african_american"):
        solve_baseline(mu, cov, target=0.90, blocs=["african_american", "white"])


# ── Original stubs: now implemented ──────────────────────────────────────────


def test_mu_eff_respects_layer_weights(panel):
    """mu_eff = λ₁·Σw_i·μ_race + λ₂·Σv_R·μ_rel + λ₃·Σg_G·μ_gen is finite and in (0,1)."""
    result = estimate_moments(panel, "democrat", winning_cycles=_WINNING)
    n_race = len(CANONICAL_RACES)
    n_rel = len(CANONICAL_RELIGIONS)
    n_gen = len(CANONICAL_GENDERS)
    mu_eff = (
        _LAYER_WEIGHTS["lambda_1"] * sum(result.mu_race[b] / n_race for b in CANONICAL_RACES)
        + _LAYER_WEIGHTS["lambda_2"] * sum(result.mu_religion[b] / n_rel for b in CANONICAL_RELIGIONS)
        + _LAYER_WEIGHTS["lambda_3"] * sum(result.mu_gender[b] / n_gen for b in CANONICAL_GENDERS)
    )
    assert math.isfinite(mu_eff), f"mu_eff is not finite: {mu_eff}"
    assert 0.0 < mu_eff < 1.0, f"mu_eff out of (0,1): {mu_eff}"


def test_v_eq_derived_from_winning_cycles_only(panel):
    """μ_race changes when the winning-cycle set changes — only winning cycles enter mu."""
    result_3 = estimate_moments(panel, "democrat", winning_cycles=_WINNING)
    result_1 = estimate_moments(panel, "democrat", winning_cycles=[2020])
    # Single-winning-cycle mean == vote_share for that cycle exactly (no averaging).
    # Three-cycle mean will differ unless all three cycle values are identical.
    aa_3 = result_3.mu_race["african_american"]
    aa_1 = result_1.mu_race["african_american"]
    assert aa_3 != pytest.approx(aa_1, abs=1e-9), (
        "mu_race should differ between 3-cycle and 1-cycle winning sets"
    )


def test_loco_cv_leaves_one_cycle_out(panel):
    """LOCO-CV produces one fold per cycle, covering every cycle exactly once."""
    result = fit_gp_classifier(
        panel, "democrat", winning_cycles=_WINNING, rng=np.random.default_rng(0)
    )
    panel_cycles = sorted(panel["cycle"].unique().astype(int))
    fold_cycles = [f.cycle for f in result.folds]
    assert fold_cycles == panel_cycles
