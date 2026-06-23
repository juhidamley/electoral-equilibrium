import time

import numpy as np
import pytest
from unittest.mock import MagicMock

from electoral.artifacts import EquilibriumData
from electoral.core.types import CANONICAL_RACES
from electoral.simulation.montecarlo import _load_layer_weights, run_ilr_montecarlo

RACES = list(CANONICAL_RACES)


def _make_equilibrium(mu_val: float, target: float = 0.535) -> EquilibriumData:
    return EquilibriumData(
        method="test",
        party="democrat",
        shock="test_shock",
        weights={r: 0.2 for r in RACES},
        mu_shifted={r: mu_val for r in RACES},
        feasible=True,
        target_met=mu_val >= target,
        target=target,
    )


def _make_config(seed: int = 42) -> MagicMock:
    config = MagicMock()
    config.derive_seed.return_value = seed
    return config


def test_smoke_completes_fast():
    t0 = time.time()
    run_ilr_montecarlo(_make_equilibrium(0.55), _make_config(), n_simulations=100)
    assert time.time() - t0 < 2.0


# ── cov_delta wiring (real Σ_Δ → Monte Carlo) ─────────────────────────────────


def test_cov_delta_none_matches_explicit_diagonal():
    """cov_delta=None must reproduce the historical isotropic-diagonal behavior."""
    eq = _make_equilibrium(0.55)
    k = len(RACES)
    diag = [[(0.02**2) if i == j else 0.0 for j in range(k)] for i in range(k)]
    a = run_ilr_montecarlo(eq, _make_config(7), n_simulations=2000)  # default diagonal
    b = run_ilr_montecarlo(eq, _make_config(7), n_simulations=2000, cov_delta=diag)
    # Same seed + mathematically identical covariance → identical result.
    assert a.to_dict() == b.to_dict()


def test_cov_delta_actually_used():
    """A larger covariance must widen the per-bloc spread — proving Σ_Δ is consumed."""
    eq = _make_equilibrium(0.55)
    k = len(RACES)
    big = [[(0.10**2) if i == j else 0.0 for j in range(k)] for i in range(k)]
    small = run_ilr_montecarlo(eq, _make_config(1), n_simulations=4000, sigma_default=0.02)
    large = run_ilr_montecarlo(eq, _make_config(1), n_simulations=4000, cov_delta=big)
    # p95 - p5 spread for the first bloc should be wider under the larger covariance.
    b = RACES[0]
    small_spread = small.percentiles[b][4] - small.percentiles[b][0]
    large_spread = large.percentiles[b][4] - large.percentiles[b][0]
    assert large_spread > small_spread


def test_cov_delta_wrong_shape_raises():
    with pytest.raises(ValueError, match="cov_delta shape"):
        run_ilr_montecarlo(
            _make_equilibrium(0.55), _make_config(), n_simulations=100, cov_delta=[[1.0, 0.0]]
        )


# ── fixed_loyalty: derived from the equilibrium vs explicit ───────────────────


def test_mc_derives_fixed_loyalty_from_equilibrium():
    """With no explicit fixed_loyalty, the MC must recover it from mu_eff_shifted
    so its win condition matches the optimizer's exactly."""
    lambda_1 = _load_layer_weights()["lambda_1"]
    w = {r: 0.2 for r in RACES}
    mu = {r: 0.55 for r in RACES}
    chosen_fl = 0.27
    race_part = lambda_1 * sum(w[r] * mu[r] for r in RACES)
    eq = EquilibriumData(
        method="t",
        party="democrat",
        shock="t",
        weights=w,
        mu_shifted=mu,
        feasible=True,
        target_met=True,
        target=0.5,
        mu_eff_shifted=race_part + chosen_fl,  # encodes fixed_loyalty = 0.27
    )
    derived = run_ilr_montecarlo(eq, _make_config(7), n_simulations=3000)
    explicit = run_ilr_montecarlo(eq, _make_config(7), n_simulations=3000, fixed_loyalty=chosen_fl)
    # Deriving from the equilibrium reproduces passing fixed_loyalty explicitly.
    assert derived.to_dict() == explicit.to_dict()


def test_mc_fixed_loyalty_falls_back_to_neutral_when_unset():
    """An equilibrium with mu_eff_shifted=0.0 (unset/fallback) → neutral (1-λ₁)·0.5."""
    lambda_1 = _load_layer_weights()["lambda_1"]
    eq = _make_equilibrium(0.55)  # mu_eff_shifted defaults to 0.0
    a = run_ilr_montecarlo(eq, _make_config(7), n_simulations=3000)
    b = run_ilr_montecarlo(
        eq, _make_config(7), n_simulations=3000, fixed_loyalty=(1.0 - lambda_1) * 0.50
    )
    assert a.to_dict() == b.to_dict()


def test_win_probability_in_range():
    result = run_ilr_montecarlo(_make_equilibrium(0.55), _make_config(), n_simulations=1000)
    assert 0.0 <= result.win_probability <= 1.0


def test_high_mu_high_win_prob():
    result = run_ilr_montecarlo(_make_equilibrium(0.8), _make_config(), n_simulations=1000)
    assert result.win_probability > 0.95


def test_low_mu_low_win_prob():
    result = run_ilr_montecarlo(_make_equilibrium(0.2), _make_config(), n_simulations=1000)
    assert result.win_probability < 0.05


def test_same_seed_deterministic():
    eq = _make_equilibrium(0.55)
    result1 = run_ilr_montecarlo(eq, _make_config(seed=42), n_simulations=500)
    result2 = run_ilr_montecarlo(eq, _make_config(seed=42), n_simulations=500)
    assert result1.to_dict() == result2.to_dict()


def test_percentiles_shape():
    result = run_ilr_montecarlo(_make_equilibrium(0.55), _make_config(), n_simulations=100)
    for bloc, pcts in result.percentiles.items():
        assert len(pcts) == 5, f"{bloc}: expected 5 percentiles, got {len(pcts)}"
        for p in pcts:
            assert 0.0 <= p <= 1.0, f"{bloc}: percentile {p} out of [0, 1]"


# ── Tests retained from earlier spec ─────────────────────────────────────────


def test_montecarlo_degenerate():
    """Highly concentrated weights (w0=0.99) must still produce valid output."""
    races = RACES
    weights = {races[0]: 0.99, **{r: 0.0025 for r in races[1:]}}
    eq = EquilibriumData(
        method="test",
        party="democrat",
        shock="degenerate_test",
        weights=weights,
        mu_shifted={r: 0.50 for r in races},
        feasible=True,
        target_met=False,
        target=0.51,
    )
    result = run_ilr_montecarlo(eq, _make_config(), n_simulations=1000)
    assert 0.0 <= result.win_probability <= 1.0
    assert 0.0 <= result.win_probability_low <= result.win_probability_high <= 1.0
    for bloc, pcts in result.percentiles.items():
        for p in pcts:
            assert np.isfinite(p) and 0.0 <= p <= 1.0, f"{bloc}: bad percentile {p}"


def test_montecarlo_zero_weight_raises():
    """Exact zero weight must raise ValueError, not silently floor."""
    races = RACES
    eq = EquilibriumData(
        method="test",
        party="democrat",
        shock="zero_weight",
        weights={races[0]: 1.0, **{r: 0.0 for r in races[1:]}},
        mu_shifted={r: 0.50 for r in races},
        feasible=True,
        target_met=False,
        target=0.51,
    )
    with pytest.raises(ValueError, match="zero-weight blocs"):
        run_ilr_montecarlo(eq, _make_config(), n_simulations=100)
