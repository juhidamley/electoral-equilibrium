"""Tests for electoral/simulation/montecarlo.py ILR Monte Carlo — Week 5."""

from __future__ import annotations

import numpy as np
import pytest

from electoral.simulation.montecarlo import (
    helmert_matrix,
    ilr,
    ilr_inv,
    run_ilr_montecarlo,
)
from electoral.artifacts import EquilibriumData
from electoral.core.types import CANONICAL_RACES


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_equilibrium(
    weights: dict | None = None, mu_shifted: dict | None = None, target: float = 0.535
) -> EquilibriumData:
    blocs = list(CANONICAL_RACES)
    if weights is None:
        weights = {b: 1.0 / len(blocs) for b in blocs}
    if mu_shifted is None:
        mu_shifted = {b: 0.60 for b in blocs}
    return EquilibriumData(
        method="cvxpy_dqcp",
        party="democrat",
        shock="test_shock",
        weights=weights,
        mu_shifted=mu_shifted,
        feasible=True,
        target_met=True,
        target=target,
    )


class _FakeConfig:
    seed = 42
    races = list(CANONICAL_RACES)

    def derive_seed(self, stage_name: str) -> int:
        from electoral.core.rng import derive_seed

        return derive_seed(self.seed, stage_name)


# ── helmert_matrix ────────────────────────────────────────────────────────────


def test_helmert_matrix_shape():
    for k in range(2, 7):
        V = helmert_matrix(k)
        assert V.shape == (k, k - 1), f"Expected ({k}, {k-1}), got {V.shape}"


def test_helmert_matrix_is_orthonormal():
    for k in [2, 3, 5]:
        V = helmert_matrix(k)
        product = V.T @ V
        assert np.allclose(product, np.eye(k - 1), atol=1e-10), f"V^T V is not identity for k={k}"


def test_helmert_columns_sum_to_zero():
    for k in [2, 3, 5]:
        V = helmert_matrix(k)
        col_sums = V.sum(axis=0)
        assert np.allclose(col_sums, 0.0, atol=1e-10), f"Column sums not zero for k={k}: {col_sums}"


def test_helmert_matrix_k_lt_2_raises():
    with pytest.raises(ValueError, match="k >= 2"):
        helmert_matrix(1)


def test_helmert_matrix_k2_known_values():
    V = helmert_matrix(2)
    # For k=2: V[:, 0] = [1/sqrt(2), -1/sqrt(2)]
    expected = np.array([[1 / np.sqrt(2)], [-1 / np.sqrt(2)]])
    assert np.allclose(V, expected, atol=1e-10)


# ── ilr / ilr_inv round-trip ──────────────────────────────────────────────────


def test_ilr_back_transform_sums_to_one():
    k = 5
    V = helmert_matrix(k)
    rng = np.random.default_rng(42)
    z = rng.standard_normal(k - 1)
    w = ilr_inv(z, V)
    assert abs(w.sum() - 1.0) < 1e-12
    assert np.all(w > 0)


def test_ilr_round_trip():
    k = 5
    V = helmert_matrix(k)
    # Start from a simplex point
    w_orig = np.array([0.12, 0.11, 0.05, 0.62, 0.10])
    z = ilr(w_orig, V)
    w_recovered = ilr_inv(z, V)
    assert np.allclose(w_recovered, w_orig, atol=1e-10)


def test_ilr_zero_weight_raises():
    k = 3
    V = helmert_matrix(k)
    w = np.array([0.0, 0.5, 0.5])  # zero weight
    with pytest.raises(ValueError, match="strictly positive"):
        ilr(w, V)


def test_ilr_inv_output_is_simplex():
    V = helmert_matrix(5)
    rng = np.random.default_rng(0)
    for _ in range(20):
        z = rng.standard_normal(4)
        w = ilr_inv(z, V)
        assert abs(w.sum() - 1.0) < 1e-12
        assert np.all(w >= 0)


# ── run_ilr_montecarlo ────────────────────────────────────────────────────────


def test_win_probability_in_unit_interval():
    eq = _make_equilibrium()
    cfg = _FakeConfig()
    result = run_ilr_montecarlo(eq, cfg, n_simulations=500)
    assert 0.0 <= result.win_probability <= 1.0


def test_win_probability_converges_with_n_simulations():
    # Higher loyalty → higher win prob; both estimates must be finite
    eq_high = _make_equilibrium(mu_shifted={b: 0.80 for b in CANONICAL_RACES}, target=0.535)
    eq_low = _make_equilibrium(mu_shifted={b: 0.40 for b in CANONICAL_RACES}, target=0.535)
    cfg = _FakeConfig()
    p_high = run_ilr_montecarlo(eq_high, cfg, n_simulations=2000).win_probability
    p_low = run_ilr_montecarlo(eq_low, cfg, n_simulations=2000).win_probability
    assert p_high > p_low


def test_seed_produces_identical_draws():
    eq = _make_equilibrium()
    cfg = _FakeConfig()
    r1 = run_ilr_montecarlo(eq, cfg, n_simulations=200)
    r2 = run_ilr_montecarlo(eq, cfg, n_simulations=200)
    assert r1.win_probability == r2.win_probability
    for bloc in CANONICAL_RACES:
        assert r1.percentiles[bloc] == r2.percentiles[bloc]


def test_zero_weight_bloc_reported_as_infeasible_not_floored():
    blocs = list(CANONICAL_RACES)
    weights = {b: 0.25 for b in blocs}
    weights["asian"] = 0.0  # zero weight — should raise, not floor
    weights["white"] = 0.25  # rebalance: 0.25+0.25+0+0.25+0.25 = 1.0
    mu_shifted = {b: 0.60 for b in blocs}
    eq = EquilibriumData(
        method="cvxpy_dqcp",
        party="democrat",
        shock="test",
        weights=weights,
        mu_shifted=mu_shifted,
        feasible=False,
        target_met=False,
        target=0.535,
    )
    cfg = _FakeConfig()
    with pytest.raises(ValueError, match="zero-weight blocs"):
        run_ilr_montecarlo(eq, cfg, n_simulations=100)


def test_percentiles_have_five_values_per_bloc():
    eq = _make_equilibrium()
    cfg = _FakeConfig()
    result = run_ilr_montecarlo(eq, cfg, n_simulations=200)
    for bloc in CANONICAL_RACES:
        assert len(result.percentiles[bloc]) == 5, f"Expected 5 percentile values for {bloc}"


def test_percentiles_are_non_decreasing():
    eq = _make_equilibrium()
    cfg = _FakeConfig()
    result = run_ilr_montecarlo(eq, cfg, n_simulations=500)
    for bloc, pcts in result.percentiles.items():
        for i in range(len(pcts) - 1):
            assert (
                pcts[i] <= pcts[i + 1] + 1e-10
            ), f"Percentiles not non-decreasing for {bloc}: {pcts}"


def test_percentiles_in_unit_interval():
    eq = _make_equilibrium()
    cfg = _FakeConfig()
    result = run_ilr_montecarlo(eq, cfg, n_simulations=200)
    for bloc, pcts in result.percentiles.items():
        for p in pcts:
            assert 0.0 <= p <= 1.0, f"Percentile out of [0,1] for {bloc}: {p}"


def test_result_covers_all_canonical_race_blocs():
    eq = _make_equilibrium()
    cfg = _FakeConfig()
    result = run_ilr_montecarlo(eq, cfg, n_simulations=100)
    assert set(result.percentiles.keys()) == set(CANONICAL_RACES)


def test_result_validates():
    eq = _make_equilibrium()
    cfg = _FakeConfig()
    result = run_ilr_montecarlo(eq, cfg, n_simulations=200)
    result.validate()


def test_n_simulations_matches_request():
    eq = _make_equilibrium()
    cfg = _FakeConfig()
    result = run_ilr_montecarlo(eq, cfg, n_simulations=317)
    assert result.n_simulations == 317


def test_seed_matches_config_monte_carlo_stage():
    eq = _make_equilibrium()
    cfg = _FakeConfig()
    result = run_ilr_montecarlo(eq, cfg, n_simulations=100)
    assert result.seed == cfg.derive_seed("monte_carlo")


def test_high_loyalty_win_prob_near_one():
    # All blocs at 0.80 with lambda_1=0.5, fixed=0.5*0.5=0.25
    # mu_eff ≈ 0.5*0.80 + 0.25 = 0.65 >> 0.535 → near-certain win
    eq = _make_equilibrium(mu_shifted={b: 0.80 for b in CANONICAL_RACES}, target=0.535)
    cfg = _FakeConfig()
    result = run_ilr_montecarlo(eq, cfg, n_simulations=1000)
    assert result.win_probability > 0.90


def test_low_loyalty_win_prob_near_zero():
    # All blocs at 0.40 with lambda_1=0.5, fixed=0.25
    # mu_eff ≈ 0.5*0.40 + 0.25 = 0.45 << 0.535 → near-zero win
    eq = _make_equilibrium(mu_shifted={b: 0.40 for b in CANONICAL_RACES}, target=0.535)
    cfg = _FakeConfig()
    result = run_ilr_montecarlo(eq, cfg, n_simulations=1000)
    assert result.win_probability < 0.10
