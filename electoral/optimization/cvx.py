"""CVXPY optimizers for the electoral coalition rebalancing pipeline.

═══════════════════════════════════════════════════════════════════════════════
WHAT THIS FILE IS (and how it relates to dqcp.py)
═══════════════════════════════════════════════════════════════════════════════
This solves the SAME "maximize probability of winning" problem described in
optimization/dqcp.py (read that file's portfolio-analogy overview first — blocs
= stocks, μ = return, Σ = risk, Sharpe ratio → P(win)). The difference is purely
mechanical: this file uses a slightly different but mathematically equivalent
reformulation (Charnes–Cooper, explained below) and a different solver path.

CONSOLIDATION (was a Week-8 task, now DONE): there used to be TWO implementations
of this optimizer — a Charnes–Cooper SOCP here and dqcp.solve_dqcp — and they
solved subtly DIFFERENT problems (this one compared race-only μ·w to V_eq with no
λ weighting; dqcp used the full λ-weighted μ_eff). That divergence is resolved:
`solve_rebalanced` below is now a thin wrapper that delegates the actual solve to
the single canonical `optimization.dqcp.solve_dqcp`, then packages the result as
a validated EquilibriumData. There is now ONE optimizer; both the
build_shock_response kernel and the FastAPI route go through this function.

The effective-loyalty objective it maximizes (via dqcp) is the full three-stratum
form, with religion+gender held at neutral 0.5 for now:
    μ_eff(w) = λ₁·Σ(wᵢ·μ̃ᵢ) + (λ₂+λ₃)·0.50
(Using the REAL religion/gender loyalties instead of 0.5 is the separate Week-8
"μ_eff basis" decision.) Per-bloc weights are bounded to [floor, ceiling],
enforced exactly inside the SOCP.

This module also re-exports solve_baseline (a plain min-variance QP) from
portfolios.cvx for callers that want the conservative baseline portfolio.
"""

from __future__ import annotations

import logging

import numpy as np

from electoral.artifacts import EquilibriumData
from electoral.core.types import CANONICAL_RACES
from electoral.portfolios.cvx import solve_baseline as solve_baseline

__all__ = ["solve_baseline", "solve_rebalanced", "solve_equal_weight_rebalanced"]

logger = logging.getLogger(__name__)


def solve_rebalanced(
    mu_tilde: dict[str, float],
    cov_delta: list[list[float]],
    target: float,
    party: str = "democrat",
    shock: str = "",
    floor: float = 0.05,
    ceiling: float = 0.60,
    fixed_loyalty: float | None = None,
) -> EquilibriumData:
    """Solve max P(win) and return a validated EquilibriumData — the ONE optimizer.

    This is the canonical coalition-rebalancing entry point used by both the
    build_shock_response kernel and the FastAPI route. It delegates the SOCP solve
    to optimization.dqcp.solve_dqcp (the single implementation), then wraps the
    result as an EquilibriumData.

    It maximizes the full effective loyalty
        μ_eff(w) = λ₁·Σ(wᵢ·μ̃ᵢ) + fixed_loyalty
    where `fixed_loyalty` is the religion+gender contribution: pass the real value
    (dqcp.compute_fixed_loyalty) for the correct model, or omit it to fall back to
    the neutral placeholder (λ₂+λ₃)·0.50. Subject to per-bloc bounds
    w_i ∈ [floor, ceiling] enforced exactly inside the SOCP. If the target is
    unreachable or the solve fails, returns the equal-weight fallback (feasible=False).

    Args:
        mu_tilde:  race_id → post-shock within-race vote share (5 keys).
        cov_delta: 5×5 race covariance Σ_Δ.
        target:    V_eq win threshold.
        party/shock: passed through to the artifact.
        floor/ceiling: per-bloc weight bounds.
    """
    # Lazy imports: keep this module light and avoid any import-order surprises.
    from electoral.optimization.dqcp import compute_mu_eff, solve_dqcp
    from electoral.portfolios.constraints import ConstraintSpec
    from electoral.simulation.montecarlo import _load_layer_weights

    blocs = list(CANONICAL_RACES)

    # λ₁ scales the race contribution. `fixed_loyalty` is the religion+gender
    # contribution: callers that know the real religion/gender loyalties pass it in
    # (see dqcp.compute_fixed_loyalty); when omitted we fall back to the neutral
    # placeholder (λ₂+λ₃)·0.5 — preserving the prior behavior.
    lw = _load_layer_weights()
    lambda_1 = lw["lambda_1"]
    if fixed_loyalty is None:
        fixed_loyalty = (lw["lambda_2"] + lw["lambda_3"]) * 0.50

    try:
        # Translate the floor/ceiling into a ConstraintSpec; solve_dqcp enforces
        # these per-bloc bounds exactly inside the cone program.
        spec = ConstraintSpec.from_bounds(
            blocs,
            lower={b: floor for b in blocs},
            upper={b: ceiling for b in blocs},
        )
        weights = solve_dqcp(
            mu_tilde,
            np.asarray(cov_delta, dtype=float),
            target,
            lambda_1,
            fixed_loyalty,
            spec=spec,
        )
    except (ValueError, RuntimeError) as exc:
        # ValueError → target unreachable or bounds infeasible (solve_dqcp pre-flight
        # / ConstraintSpec validation); RuntimeError → solver non-optimal status.
        # Either way, degrade gracefully to the equal-weight fallback.
        logger.warning("solve_rebalanced: infeasible (%s) — equal-weight fallback", exc)
        return solve_equal_weight_rebalanced(blocs, mu_tilde, party, shock, target)

    # μ_eff on the SAME basis as the win check, so target_met is self-consistent.
    mu_eff = compute_mu_eff(weights, mu_tilde, lambda_1, fixed_loyalty)
    return EquilibriumData(
        method="cvxpy_dqcp",
        party=party,
        shock=shock,
        weights={b: float(weights[b]) for b in blocs},
        mu_shifted={b: float(mu_tilde[b]) for b in blocs},
        feasible=True,
        target_met=bool(mu_eff >= target),
        target=target,
        mu_eff_shifted=float(mu_eff),
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
        method="equal_weight_fallback",
        party=party,
        shock=shock,
        weights={b: 1.0 / n for b in blocs},
        mu_shifted={b: float(mu_tilde[b]) for b in blocs},
        feasible=False,
        target_met=False,
        target=target,
    )
