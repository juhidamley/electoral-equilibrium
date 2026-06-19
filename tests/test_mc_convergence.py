"""Tests for electoral/simulation/montecarlo.py ILR Monte Carlo — Week 5."""

from __future__ import annotations

import logging
import numpy as np
import pytest
from unittest.mock import MagicMock

from electoral.simulation.montecarlo import (
    helmert_matrix,
    ilr,
    ilr_inv,
    run_ilr_montecarlo,
)
from electoral.artifacts import EquilibriumData
from electoral.core.types import CANONICAL_RACES

log = logging.getLogger(__name__)


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


def test_ci_width_shrinks_with_more_simulations():
    # Bootstrap CI width is O(1/sqrt(N)); more draws → tighter interval.
    # mu=0.57, target=0.535, sigma_default=0.05 → win_prob ≈ 0.19 (non-trivial).
    eq = _make_equilibrium(mu_shifted={b: 0.57 for b in CANONICAL_RACES}, target=0.535)
    cfg = _FakeConfig()
    r_small = run_ilr_montecarlo(eq, cfg, n_simulations=200, sigma_default=0.05)
    r_large = run_ilr_montecarlo(eq, cfg, n_simulations=5000, sigma_default=0.05)
    width_small = r_small.win_probability_high - r_small.win_probability_low
    width_large = r_large.win_probability_high - r_large.win_probability_low
    assert (
        width_large < width_small
    ), f"CI did not shrink: small={width_small:.4f}, large={width_large:.4f}"


def test_ci_bounds_are_not_trivially_zero_and_one():
    # Bootstrap CI must be strictly interior to [0, 1] at intermediate win prob.
    # mu=0.57, target=0.535, sigma_default=0.05 → win_prob ≈ 0.19.
    eq = _make_equilibrium(mu_shifted={b: 0.57 for b in CANONICAL_RACES}, target=0.535)
    cfg = _FakeConfig()
    result = run_ilr_montecarlo(eq, cfg, n_simulations=2000, sigma_default=0.05)
    assert result.win_probability_low > 0.0, "CI lower bound is trivially 0"
    assert result.win_probability_high < 1.0, "CI upper bound is trivially 1"


def test_mc_convergence():
    """Win probability estimates must converge as N increases.

    N=5 000 and N=10 000 must differ by < 0.005.
    N=1 000 must be within 0.02 of N=10 000 (looser bound).

    mu=0.535 sits right at target but lambda_1=1/3 scaling pulls
    mu_eff ≈ 0.511 below it, pinning wp near 0.  Using mu=0.57 with
    sigma_default=0.05 gives wp ≈ 0.19 — clearly non-degenerate.
    """
    races = list(CANONICAL_RACES)
    eq = EquilibriumData(
        method="test",
        party="democrat",
        shock="convergence_test",
        weights={r: 0.2 for r in races},
        mu_shifted={r: 0.57 for r in races},
        feasible=True,
        target_met=False,
        target=0.535,
    )

    config = MagicMock()
    config.derive_seed.return_value = 42

    wp_1k = run_ilr_montecarlo(eq, config, n_simulations=1_000, sigma_default=0.05).win_probability
    wp_5k = run_ilr_montecarlo(eq, config, n_simulations=5_000, sigma_default=0.05).win_probability
    wp_10k = run_ilr_montecarlo(
        eq, config, n_simulations=10_000, sigma_default=0.05
    ).win_probability

    log.info(
        "Convergence: N=1k → %.4f  N=5k → %.4f  N=10k → %.4f",
        wp_1k,
        wp_5k,
        wp_10k,
    )
    print(
        f"\nConvergence estimates:\n"
        f"  N= 1,000: {wp_1k:.4f}\n"
        f"  N= 5,000: {wp_5k:.4f}\n"
        f"  N=10,000: {wp_10k:.4f}\n"
        f"  |5k - 10k| = {abs(wp_5k - wp_10k):.4f}  (must be < 0.005)\n"
        f"  |1k - 10k| = {abs(wp_1k - wp_10k):.4f}  (must be < 0.02)"
    )

    assert abs(wp_5k - wp_10k) < 0.005, f"5k vs 10k diff {abs(wp_5k - wp_10k):.4f} exceeds 0.005"
    assert abs(wp_1k - wp_10k) < 0.02, f"1k vs 10k diff {abs(wp_1k - wp_10k):.4f} exceeds 0.02"
