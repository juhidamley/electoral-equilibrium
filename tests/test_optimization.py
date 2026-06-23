import logging

import numpy as np

from electoral.core.types import CANONICAL_GENDERS, CANONICAL_RACES, CANONICAL_RELIGIONS
from electoral.optimization.cvx import solve_rebalanced
from electoral.optimization.dqcp import compute_fixed_loyalty, solve_dqcp
from electoral.optimization.simplex import project_simplex
from electoral.portfolios.constraints import ConstraintSpec

RACES = list(CANONICAL_RACES)
_DIAG_COV = [[0.001 if i == j else 0.0 for j in range(5)] for i in range(5)]


def test_feasible_shock_target_met():
    """Small delta, weak covariance, mu above target → feasible & target_met."""
    mu_tilde = {r: 0.58 for r in RACES}
    result = solve_rebalanced(
        mu_tilde, _DIAG_COV, target=0.535, party="democrat", shock="feasible_test"
    )
    assert result.feasible is True
    assert result.target_met is True
    assert abs(sum(result.weights.values()) - 1.0) < 1e-6


def test_extreme_shock_target_not_met(caplog):
    """Mu all well below target → target_met=False; weights still form a valid simplex."""
    with caplog.at_level(logging.WARNING):
        mu_tilde = {r: 0.30 for r in RACES}
        result = solve_rebalanced(
            mu_tilde, _DIAG_COV, target=0.535, party="democrat", shock="extreme_test"
        )
    assert result.target_met is False
    assert abs(sum(result.weights.values()) - 1.0) < 1e-6


def test_project_simplex_valid():
    """project_simplex output is non-negative and sums to 1.0."""
    rng = np.random.default_rng(0)
    for _ in range(20):
        v = rng.uniform(-2, 2, size=5)
        result = project_simplex(v)
        assert np.all(result >= -1e-12)
        assert abs(result.sum() - 1.0) < 1e-9


def test_deterministic_output():
    """Same inputs produce identical EquilibriumData.to_dict() across two calls."""
    mu_tilde = {r: 0.55 for r in RACES}
    cov = [[0.002 if i == j else 0.0 for j in range(5)] for i in range(5)]
    r1 = solve_rebalanced(mu_tilde, cov, target=0.535, party="democrat", shock="determ")
    r2 = solve_rebalanced(mu_tilde, cov, target=0.535, party="democrat", shock="determ")
    assert r1.to_dict() == r2.to_dict()


# ── solve_dqcp per-bloc bound enforcement (Week-8 fix) ────────────────────────


def test_solve_dqcp_respects_spec_upper_bound():
    """A ConstraintSpec upper bound must hold EXACTLY — not be violated after renorm.

    Set african_american's loyalty far above the others so the unconstrained
    optimum would pile weight onto it; then cap it at 0.30 and assert the cap holds.
    """
    cov = np.eye(5) * 0.001
    # aa is the most attractive bloc → unconstrained optimum over-weights it.
    mu = {
        "african_american": 0.90,
        "latino": 0.50,
        "asian": 0.50,
        "white": 0.50,
        "other_race": 0.50,
    }

    # Baseline (no spec): aa should dominate, exceeding 0.30 — proves the cap binds.
    unconstrained = solve_dqcp(mu, cov, target=0.40, lambda_1=1.0, fixed_loyalty=0.0)
    assert unconstrained["african_american"] > 0.30

    spec = ConstraintSpec.from_bounds(RACES, upper={"african_american": 0.30})
    bounded = solve_dqcp(mu, cov, target=0.40, lambda_1=1.0, fixed_loyalty=0.0, spec=spec)

    assert bounded["african_american"] <= 0.30 + 1e-6  # cap respected (was violated by clip+renorm)
    assert abs(sum(bounded.values()) - 1.0) < 1e-6  # still a valid simplex
    assert all(w >= -1e-9 for w in bounded.values())


def test_solve_dqcp_respects_spec_lower_bound():
    """A ConstraintSpec lower bound must hold: an unattractive bloc still gets ≥ floor."""
    cov = np.eye(5) * 0.001
    # asian is the LEAST attractive → unconstrained optimum would starve it.
    mu = {
        "african_american": 0.70,
        "latino": 0.70,
        "asian": 0.30,
        "white": 0.70,
        "other_race": 0.70,
    }
    unconstrained = solve_dqcp(mu, cov, target=0.40, lambda_1=1.0, fixed_loyalty=0.0)
    assert unconstrained["asian"] < 0.15  # nearly starved without a floor

    spec = ConstraintSpec.from_bounds(RACES, lower={"asian": 0.15})
    bounded = solve_dqcp(mu, cov, target=0.40, lambda_1=1.0, fixed_loyalty=0.0, spec=spec)

    assert bounded["asian"] >= 0.15 - 1e-6  # floor respected
    assert abs(sum(bounded.values()) - 1.0) < 1e-6


# ── compute_fixed_loyalty (religion+gender contribution to μ_eff) ─────────────


def test_compute_fixed_loyalty_neutral_baseline_matches_old_placeholder():
    """All strata at 0.5 with no deltas → exactly the old neutral (λ₂+λ₃)·0.5."""
    rel = {r: 0.5 for r in CANONICAL_RELIGIONS}
    gen = {g: 0.5 for g in CANONICAL_GENDERS}
    assert abs(compute_fixed_loyalty(rel, gen, 0.3, 0.2) - (0.5 * 0.5)) < 1e-12


def test_compute_fixed_loyalty_uses_real_values_and_deltas():
    """Real loyalties differ from 0.5, and shock deltas shift the result."""
    rel = {r: 0.5 for r in CANONICAL_RELIGIONS}
    gen = {g: 0.5 for g in CANONICAL_GENDERS}
    base = compute_fixed_loyalty(rel, gen, 0.3, 0.2)
    # Push one religion bloc's loyalty up via a delta → fixed_loyalty rises.
    bumped = compute_fixed_loyalty(
        rel, gen, 0.3, 0.2, deltas_religion={CANONICAL_RELIGIONS[0]: 0.30}
    )
    assert bumped > base
    # Deltas are clipped so loyalties stay in [0, 1] → result stays a valid share.
    clipped = compute_fixed_loyalty(
        rel, gen, 0.3, 0.2, deltas_religion={r: 5.0 for r in CANONICAL_RELIGIONS}
    )
    assert 0.0 <= clipped <= 0.5  # ≤ λ₂·1 + λ₃·0.5 = 0.3 + 0.1 = 0.4
