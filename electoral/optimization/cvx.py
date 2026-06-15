"""CVXPY optimizers for the electoral coalition rebalancing pipeline.

Re-exports solve_baseline (min-variance QP) from portfolios.cvx, and provides
solve_rebalanced (max P(win) Sharpe ratio) as the primary production optimizer.

solve_rebalanced uses the Charnes-Cooper SOCP transformation of the Sharpe ratio:

    max  (mu @ w - target) / sqrt(w^T Sigma w)
    s.t. sum(w) = 1,  floor <= w <= ceiling

is rewritten as the equivalent SOCP:

    max  mu @ y - target * tau
    s.t. ||L^T y||_2 <= 1          (y^T Sigma y <= 1, tight at optimum)
         sum(y) = tau               (normalization: w = y / tau)
         floor * tau <= y <= ceiling * tau
         mu @ y >= target * tau     (force positive Sharpe; infeasible if win unreachable)
         tau >= 0

where L = chol(Sigma) and the original weights are recovered as w* = y* / tau*.

The transformation preserves the quasiconvex objective: every DCP problem is also DQCP,
so problem.is_dqcp() == True is guaranteed for the SOCP.
"""

from __future__ import annotations

import logging

import cvxpy as cp
import numpy as np

from electoral.artifacts import EquilibriumData
from electoral.core.types import CANONICAL_RACES
from electoral.portfolios.cvx import solve_baseline as solve_baseline

__all__ = ["solve_baseline", "solve_rebalanced", "solve_equal_weight_rebalanced"]

logger = logging.getLogger(__name__)

_INFEASIBLE = frozenset(["infeasible", "infeasible_inaccurate"])
_MAX_RETRIES = 5
_RETRY_REG = 1e-6


def solve_rebalanced(
    mu_tilde: dict[str, float],
    cov_delta: list[list[float]],
    target: float,
    party: str = "democrat",
    shock: str = "",
    floor: float = 0.05,
    ceiling: float = 0.60,
) -> EquilibriumData:
    """Solve max P(win) via Charnes-Cooper SOCP (Sharpe-ratio formulation).

    Maximizes (mu_tilde @ w - target) / sqrt(w^T cov_delta w) subject to
    the coalition simplex constraints. The Charnes-Cooper transformation
    converts the quasiconvex ratio into an equivalent SOCP so standard
    solvers can find the global optimum.

    Retries up to 5 times with progressively relaxed covariance regularization
    before falling back to solve_equal_weight_rebalanced.
    """
    blocs = list(CANONICAL_RACES)
    n = len(blocs)

    mu = np.array([mu_tilde[b] for b in blocs], dtype=float)
    Sigma = np.array(cov_delta, dtype=float)

    # Regularize covariance to ensure positive definiteness for Cholesky
    min_eig = np.linalg.eigvalsh(Sigma).min()
    if min_eig < 1e-8:
        Sigma += np.eye(n) * (1e-8 - min_eig)

    try:
        L = np.linalg.cholesky(Sigma)
    except np.linalg.LinAlgError:
        logger.warning("solve_rebalanced: Cholesky failed — returning infeasible")
        return solve_equal_weight_rebalanced(blocs, mu_tilde, party, shock, target)

    # Charnes-Cooper variables: y = w * tau, tau = 1 / sqrt(w^T Sigma w)
    y = cp.Variable(n, name="y")
    tau = cp.Variable(nonneg=True, name="tau")

    def _build_problem(L_mat: np.ndarray) -> cp.Problem:
        constraints = [
            cp.norm(L_mat.T @ y, 2) <= 1,  # y^T Sigma y <= 1 (tight at optimum)
            cp.sum(y) == tau,  # sum(y/tau) = 1 → sum(w) = 1
            y >= floor * tau,  # w_i >= floor
            y <= ceiling * tau,  # w_i <= ceiling
            mu @ y >= target * tau,  # mu @ w >= target; infeasible if win unreachable
        ]
        return cp.Problem(cp.Maximize(mu @ y - target * tau), constraints)

    problem = _build_problem(L)

    # Every DCP problem is also DQCP; the SOCP IS DCP, so is_dqcp() == True.
    if not problem.is_dqcp():
        logger.error("Problem is not DQCP — cannot guarantee global optimum")
        return solve_equal_weight_rebalanced(blocs, mu_tilde, party, shock, target)

    for attempt in range(_MAX_RETRIES):
        try:
            problem.solve(qcp=True, solver=cp.SCS)
            if problem.status not in _INFEASIBLE:
                break
            # Relax regularization and rebuild with updated Cholesky factor
            Sigma += np.eye(n) * _RETRY_REG
            L = np.linalg.cholesky(Sigma)
            problem = _build_problem(L)
        except cp.SolverError:
            if attempt == _MAX_RETRIES - 1:
                return solve_equal_weight_rebalanced(blocs, mu_tilde, party, shock, target)
            Sigma += np.eye(n) * _RETRY_REG
            L = np.linalg.cholesky(Sigma)
            problem = _build_problem(L)
    else:
        return solve_equal_weight_rebalanced(blocs, mu_tilde, party, shock, target)

    if problem.status in _INFEASIBLE:
        return solve_equal_weight_rebalanced(blocs, mu_tilde, party, shock, target)

    tau_val = float(tau.value) if tau.value is not None else 0.0
    if tau_val < 1e-9 or y.value is None:
        logger.warning("solve_rebalanced: degenerate solution (tau≈0)")
        return solve_equal_weight_rebalanced(blocs, mu_tilde, party, shock, target)

    w_opt = y.value / tau_val
    mu_eff = float(mu @ w_opt)

    return EquilibriumData(
        method="cvxpy_dqcp",
        party=party,
        shock=shock,
        weights={b: float(w_opt[i]) for i, b in enumerate(blocs)},
        mu_shifted={b: float(mu_tilde[b]) for b in blocs},
        feasible=True,
        target_met=mu_eff >= target,
        target=target,
    )


def solve_equal_weight_rebalanced(
    blocs: list[str],
    mu_tilde: dict[str, float],
    party: str = "democrat",
    shock: str = "",
    target: float = 0.50,
) -> EquilibriumData:
    """Equal-weight fallback when QP is infeasible. Logs a warning."""
    logger.warning(
        "solve_equal_weight_rebalanced: returning equal-weight fallback "
        "for shock=%s (feasible=False)",
        shock,
    )
    return _infeasible_result(blocs, mu_tilde, party, shock, target)


def _infeasible_result(
    blocs: list[str],
    mu_tilde: dict[str, float],
    party: str,
    shock: str,
    target: float,
) -> EquilibriumData:
    n = len(blocs)
    return EquilibriumData(
        method="cvxpy_dqcp",
        party=party,
        shock=shock,
        weights={b: 1.0 / n for b in blocs},
        mu_shifted={b: float(mu_tilde[b]) for b in blocs},
        feasible=False,
        target_met=False,
        target=target,
    )
