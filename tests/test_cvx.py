"""Tests for electoral/portfolios/cvx.py — solve_baseline()."""

from __future__ import annotations

import numpy as np
import pytest

from electoral.core.types import CANONICAL_RACES
from electoral.portfolios.cvx import solve_baseline

# ── helpers ───────────────────────────────────────────────────────────────────


def _identity_mu(blocs: list[str], val: float) -> dict[str, float]:
    return {b: val for b in blocs}


def _diag_cov(blocs: list[str], variances: list[float]) -> np.ndarray:
    return np.diag(variances)


# ── simplex invariants (every feasible call must satisfy these) ───────────────


def test_weights_sum_to_one():
    blocs = ["a", "b", "c"]
    mu = {"a": 0.7, "b": 0.5, "c": 0.4}
    cov = np.eye(3)
    result = solve_baseline(mu, cov, target=0.55, blocs=blocs)
    assert sum(result.values()) == pytest.approx(1.0, abs=1e-6)


def test_weights_nonneg():
    blocs = ["a", "b", "c"]
    mu = {"a": 0.7, "b": 0.5, "c": 0.4}
    cov = np.eye(3)
    result = solve_baseline(mu, cov, target=0.55, blocs=blocs)
    assert all(v >= 0.0 for v in result.values())


def test_loyalty_constraint_satisfied():
    blocs = ["a", "b"]
    mu = {"a": 0.6, "b": 0.4}
    cov = np.eye(2)
    target = 0.55
    result = solve_baseline(mu, cov, target=target, blocs=blocs)
    achieved = sum(mu[b] * result[b] for b in blocs)
    assert achieved >= target - 1e-6


def test_result_keys_match_blocs():
    blocs = ["a", "b", "c"]
    mu = {"a": 0.6, "b": 0.5, "c": 0.4}
    cov = np.eye(3)
    result = solve_baseline(mu, cov, target=0.45, blocs=blocs)
    assert set(result.keys()) == set(blocs)


# ── analytical solutions ──────────────────────────────────────────────────────


def test_known_solution_constraint_binds():
    # 2-bloc QP with identity cov, target forces w_a >= 0.75.
    # min w_a^2 + w_b^2  s.t. w_a+w_b=1, 0.6*w_a+0.4*w_b >= 0.55
    # Unconstrained min at w_a=0.5; constraint pushes to w_a=0.75.
    blocs = ["a", "b"]
    mu = {"a": 0.6, "b": 0.4}
    cov = np.eye(2)
    result = solve_baseline(mu, cov, target=0.55, blocs=blocs)
    assert result["a"] == pytest.approx(0.75, abs=1e-4)
    assert result["b"] == pytest.approx(0.25, abs=1e-4)


def test_known_solution_shifts_from_high_variance_bloc():
    # 2-bloc QP: bloc "a" has 4× variance of "b", equal mu, non-binding target.
    # min 4*w_a^2 + w_b^2  s.t. w_a+w_b=1, 0.55*w_a+0.55*w_b >= 0.50
    # Loyalty always satisfied; unconstrained min at w_a=0.2, w_b=0.8.
    blocs = ["a", "b"]
    mu = {"a": 0.55, "b": 0.55}
    cov = _diag_cov(blocs, [4.0, 1.0])
    result = solve_baseline(mu, cov, target=0.50, blocs=blocs)
    assert result["a"] == pytest.approx(0.2, abs=1e-4)
    assert result["b"] == pytest.approx(0.8, abs=1e-4)


def test_known_solution_equal_weight_when_symmetric():
    # Equal mu, identity cov, non-binding target → equal weight is optimal.
    blocs = ["a", "b"]
    mu = {"a": 0.55, "b": 0.55}
    cov = np.eye(2)
    result = solve_baseline(mu, cov, target=0.50, blocs=blocs)
    assert result["a"] == pytest.approx(0.5, abs=1e-4)
    assert result["b"] == pytest.approx(0.5, abs=1e-4)


def test_all_weight_on_single_bloc_when_target_equals_max_mu():
    # target = max(mu) forces all weight onto the highest-loyalty bloc.
    blocs = ["a", "b", "c"]
    mu = {"a": 0.80, "b": 0.55, "c": 0.40}
    cov = np.eye(3)
    result = solve_baseline(mu, cov, target=0.80, blocs=blocs)
    assert result["a"] == pytest.approx(1.0, abs=1e-4)
    assert result["b"] == pytest.approx(0.0, abs=1e-4)
    assert result["c"] == pytest.approx(0.0, abs=1e-4)


# ── objective: result is minimum-variance ─────────────────────────────────────


def test_result_variance_le_equal_weight_variance():
    # Optimal solution must not have higher variance than equal-weight portfolio.
    blocs = ["a", "b", "c"]
    mu = {"a": 0.7, "b": 0.5, "c": 0.45}
    cov = np.array([[0.04, 0.01, 0.0], [0.01, 0.02, 0.005], [0.0, 0.005, 0.01]])
    target = 0.52
    result = solve_baseline(mu, cov, target=target, blocs=blocs)
    w_opt = np.array([result[b] for b in blocs])
    w_eq = np.ones(3) / 3.0
    var_opt = float(w_opt @ cov @ w_opt)
    var_eq = float(w_eq @ cov @ w_eq)
    assert var_opt <= var_eq + 1e-6


# ── default blocs = CANONICAL_RACES ───────────────────────────────────────────


def test_default_blocs_are_canonical_races():
    # When blocs is omitted, keys of result == CANONICAL_RACES.
    mu = {b: 0.55 for b in CANONICAL_RACES}
    cov = np.eye(len(CANONICAL_RACES))
    result = solve_baseline(mu, cov, target=0.50)
    assert set(result.keys()) == set(CANONICAL_RACES)


def test_five_canonical_race_blocs():
    mu = {
        "african_american": 0.87,
        "latino": 0.63,
        "asian": 0.72,
        "white": 0.41,
        "other_race": 0.55,
    }
    cov = np.diag([0.01, 0.011, 0.06, 0.005, 0.054])
    result = solve_baseline(mu, cov, target=0.60)
    assert set(result.keys()) == set(CANONICAL_RACES)
    assert sum(result.values()) == pytest.approx(1.0, abs=1e-6)
    achieved = sum(mu[b] * result[b] for b in CANONICAL_RACES)
    assert achieved >= 0.60 - 1e-6


# ── infeasibility ─────────────────────────────────────────────────────────────


def test_target_above_max_mu_raises_valueerror():
    blocs = ["a", "b"]
    mu = {"a": 0.6, "b": 0.4}
    cov = np.eye(2)
    with pytest.raises(ValueError, match="exceeds the maximum achievable"):
        solve_baseline(mu, cov, target=0.65, blocs=blocs)


def test_error_message_names_target():
    blocs = ["a", "b"]
    mu = {"a": 0.6, "b": 0.4}
    cov = np.eye(2)
    with pytest.raises(ValueError, match="0.650000"):
        solve_baseline(mu, cov, target=0.65, blocs=blocs)


def test_target_just_at_max_mu_is_feasible():
    # target == max(mu) must succeed (entire weight on best bloc).
    blocs = ["a", "b"]
    mu = {"a": 0.6, "b": 0.4}
    cov = np.eye(2)
    result = solve_baseline(mu, cov, target=0.6, blocs=blocs)
    assert sum(result.values()) == pytest.approx(1.0, abs=1e-6)


# ── input validation ──────────────────────────────────────────────────────────


def test_nan_mu_raises_valueerror():
    blocs = ["a", "b"]
    mu = {"a": float("nan"), "b": 0.4}
    cov = np.eye(2)
    with pytest.raises(ValueError, match="NaN"):
        solve_baseline(mu, cov, target=0.35, blocs=blocs)


def test_mismatched_cov_shape_raises_valueerror():
    blocs = ["a", "b", "c"]
    mu = {"a": 0.6, "b": 0.5, "c": 0.4}
    cov = np.eye(2)  # wrong: 2×2 for 3 blocs
    with pytest.raises(ValueError, match="shape"):
        solve_baseline(mu, cov, target=0.45, blocs=blocs)


# ── relaxation fallback ───────────────────────────────────────────────────────
#
# The retry loop fires when the CVXPY solver returns INFEASIBLE.  In practice
# this should not happen for a well-formed QP (pre-flight catches structural
# infeasibility), but the safeguard exists for numerical edge cases.
# Tests drive the retry path by monkeypatching cp.Problem.solve to return
# a controlled status without actually solving.


import cvxpy as cp  # noqa: E402  (needed for status constants in mocks)


def test_all_retries_exhausted_raises_valueerror(monkeypatch):
    # Patch solve to always report INFEASIBLE — exhaust all retries.
    def always_infeasible(self, **kwargs):
        self._status = cp.INFEASIBLE

    monkeypatch.setattr(cp.Problem, "solve", always_infeasible)

    blocs = ["a", "b"]
    mu = {"a": 0.6, "b": 0.4}
    cov = np.eye(2)

    with pytest.raises(ValueError, match="infeasible after"):
        solve_baseline(mu, cov, target=0.55, blocs=blocs, max_retries=3)


def test_error_message_contains_final_target(monkeypatch):
    def always_infeasible(self, **kwargs):
        self._status = cp.INFEASIBLE

    monkeypatch.setattr(cp.Problem, "solve", always_infeasible)

    blocs = ["a", "b"]
    mu = {"a": 0.6, "b": 0.4}
    cov = np.eye(2)
    # 2 retries of 0.10 each → final target = 0.55 - 2*0.10 = 0.35
    with pytest.raises(ValueError, match="0.350000"):
        solve_baseline(mu, cov, target=0.55, blocs=blocs, max_retries=2, relax_step=0.10)


def test_retry_count_is_exactly_max_retries_plus_one(monkeypatch):
    call_count = [0]

    def count_and_fail(self, **kwargs):
        call_count[0] += 1
        self._status = cp.INFEASIBLE

    monkeypatch.setattr(cp.Problem, "solve", count_and_fail)

    blocs = ["a", "b"]
    mu = {"a": 0.6, "b": 0.4}
    cov = np.eye(2)

    with pytest.raises(ValueError):
        solve_baseline(mu, cov, target=0.55, blocs=blocs, max_retries=3)

    assert call_count[0] == 4  # 1 original + 3 retries


def test_warning_logged_per_relaxation(monkeypatch, caplog):
    import logging

    def always_infeasible(self, **kwargs):
        self._status = cp.INFEASIBLE

    monkeypatch.setattr(cp.Problem, "solve", always_infeasible)

    blocs = ["a", "b"]
    mu = {"a": 0.6, "b": 0.4}
    cov = np.eye(2)

    with caplog.at_level(logging.WARNING, logger="electoral.portfolios.cvx"):
        with pytest.raises(ValueError):
            solve_baseline(mu, cov, target=0.55, blocs=blocs, max_retries=2)

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 2  # one per retry, not for the initial attempt


def test_relaxed_target_in_warning_message(monkeypatch, caplog):
    import logging

    def always_infeasible(self, **kwargs):
        self._status = cp.INFEASIBLE

    monkeypatch.setattr(cp.Problem, "solve", always_infeasible)

    blocs = ["a", "b"]
    mu = {"a": 0.6, "b": 0.4}
    cov = np.eye(2)

    with caplog.at_level(logging.WARNING, logger="electoral.portfolios.cvx"):
        with pytest.raises(ValueError):
            solve_baseline(mu, cov, target=0.55, blocs=blocs, max_retries=1, relax_step=0.05)

    # The single warning should mention the relaxed target 0.55 - 0.05 = 0.50
    assert any("0.500000" in msg for msg in caplog.messages)


def test_retry_succeeds_on_second_attempt(monkeypatch, caplog):
    import logging

    # First call: inject INFEASIBLE without solving.
    # Subsequent calls: delegate to the real solver.
    call_count = [0]
    original_solve = cp.Problem.solve

    def first_infeasible_then_real(self, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            self._status = cp.INFEASIBLE
        else:
            original_solve(self, **kwargs)

    monkeypatch.setattr(cp.Problem, "solve", first_infeasible_then_real)

    blocs = ["a", "b"]
    mu = {"a": 0.6, "b": 0.4}
    cov = np.eye(2)

    with caplog.at_level(logging.WARNING, logger="electoral.portfolios.cvx"):
        result = solve_baseline(mu, cov, target=0.55, blocs=blocs, max_retries=2)

    # Real solve used target - 1*relax_step = 0.54; result must be valid
    assert sum(result.values()) == pytest.approx(1.0, abs=1e-5)
    assert all(v >= 0.0 for v in result.values())
    # Exactly one WARNING logged (for the single retry)
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1


def test_noninfeasible_solver_error_raises_runtimeerror_immediately(monkeypatch):
    # SOLVER_ERROR should raise RuntimeError without any retries.
    call_count = [0]

    def solver_error(self, **kwargs):
        call_count[0] += 1
        self._status = cp.SOLVER_ERROR

    monkeypatch.setattr(cp.Problem, "solve", solver_error)

    blocs = ["a", "b"]
    mu = {"a": 0.6, "b": 0.4}
    cov = np.eye(2)

    with pytest.raises(RuntimeError, match="non-infeasibility failure"):
        solve_baseline(mu, cov, target=0.55, blocs=blocs, max_retries=5)

    assert call_count[0] == 1  # raised immediately, no retries


# ── Edge cases ────────────────────────────────────────────────────────────────


def test_single_bloc_returns_full_weight():
    """With one bloc the only feasible point is w=[1]; no solver call needed."""
    mu = {"african_american": 0.89}
    cov = np.array([[0.01]])
    result = solve_baseline(mu, cov, target=0.80, blocs=["african_american"])
    assert result == {"african_american": pytest.approx(1.0)}


def test_single_bloc_sum_is_one():
    """Single-bloc output sums to exactly 1.0."""
    result = solve_baseline({"x": 0.70}, np.array([[0.05]]), target=0.60, blocs=["x"])
    assert sum(result.values()) == pytest.approx(1.0)


def test_equal_weight_feasible_when_loyalty_meets_target():
    """Equal-weight allocation is a feasible solution when mu_avg >= target."""
    # mu_avg = (0.80 + 0.60 + 0.50) / 3 = 0.633 > 0.60
    blocs = ["a", "b", "c"]
    mu = {"a": 0.80, "b": 0.60, "c": 0.50}
    cov = np.eye(3)
    result = solve_baseline(mu, cov, target=0.60, blocs=blocs)
    # Solver may concentrate weight, but a feasible solution must exist
    assert sum(result.values()) == pytest.approx(1.0)
    achieved = sum(result[b] * mu[b] for b in blocs)
    assert achieved >= 0.60 - 1e-4


def test_target_above_one_raises_immediately():
    """target > 1.0 is impossible and raises before any solver or spec work."""
    mu = {"a": 0.90, "b": 0.80}
    cov = np.eye(2)
    with pytest.raises(ValueError, match="1.0"):
        solve_baseline(mu, cov, target=1.01, blocs=["a", "b"])


def test_target_exactly_one_raises():
    """target = 1.0 is only achievable if a bloc has mu = 1.0 exactly."""
    mu = {"a": 0.90, "b": 0.80}
    cov = np.eye(2)
    # max achievable = 0.90 < 1.0, so pre-flight catches this
    with pytest.raises(ValueError):
        solve_baseline(mu, cov, target=1.0, blocs=["a", "b"])


# ── solve_rebalanced (DQCP max P(win)) ───────────────────────────────────────


from electoral.core.types import CANONICAL_RACES as _RACES  # noqa: E402
from electoral.optimization.cvx import solve_rebalanced  # noqa: E402


def _uniform_mu(val: float) -> dict[str, float]:
    return {b: val for b in _RACES}


def _eye_cov() -> list[list[float]]:
    n = len(_RACES)
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def test_is_dqcp():
    # Charnes-Cooper SOCP is DCP; every DCP problem is also DQCP.
    n = len(_RACES)
    mu = np.array([0.55] * n, dtype=float)
    Sigma = np.eye(n)
    L = np.linalg.cholesky(Sigma)
    target = 0.50
    y = cp.Variable(n)
    tau = cp.Variable(nonneg=True)
    constraints = [
        cp.norm(L.T @ y, 2) <= 1,
        cp.sum(y) == tau,
        y >= 0.05 * tau,
        y <= 0.60 * tau,
        mu @ y >= target * tau,
    ]
    problem = cp.Problem(cp.Maximize(mu @ y - target * tau), constraints)
    assert problem.is_dqcp() is True


def test_feasible_solution():
    result = solve_rebalanced(_uniform_mu(0.70), _eye_cov(), target=0.52)
    assert result.feasible is True
    assert abs(sum(result.weights.values()) - 1.0) < 1e-6


def test_infeasible_returns_false():
    # All mu below target → mu @ w < target for any w → mu @ y >= target*tau infeasible
    result = solve_rebalanced(_uniform_mu(0.20), _eye_cov(), target=0.60)
    assert result.feasible is False


def test_weights_respect_bounds():
    floor, ceiling = 0.05, 0.60
    result = solve_rebalanced(_uniform_mu(0.65), _eye_cov(), target=0.52, floor=floor, ceiling=ceiling)
    assert result.feasible is True
    for w in result.weights.values():
        assert w >= floor - 1e-6
        assert w <= ceiling + 1e-6
    assert abs(sum(result.weights.values()) - 1.0) < 1e-6
