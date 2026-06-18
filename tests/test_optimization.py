import logging

import numpy as np

from electoral.core.types import CANONICAL_RACES
from electoral.optimization.cvx import solve_rebalanced
from electoral.optimization.simplex import project_simplex

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
