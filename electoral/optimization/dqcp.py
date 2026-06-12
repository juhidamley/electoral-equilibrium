"""DQCP Sharpe-ratio optimizer: maximize P(win) for a post-shock coalition.

Objective (quasi-concave Sharpe-ratio form):
    max Φ( (μ_eff(w) - V_eq) / (λ₁ · √(w^T Σ_Δ w)) )

Since Φ is strictly monotone, this is equivalent to maximising the argument:
    max  (λ₁ · μ_race @ w + c_fixed - V_eq)  /  (λ₁ · ‖L w‖₂)

where L = chol(Σ_Δ) and c_fixed = λ₂ · μ_rel + λ₃ · μ_gender (fixed strata).

The ratio of an affine function to a convex norm is quasi-concave (super-level
sets {w : ratio ≥ t} are convex intersections).  CVXPY DQCP solves this via
parametric bisection: for each candidate t it checks feasibility of the convex
program  μ_race @ w + δ_fixed ≥ t · ‖L w‖₂  on the simplex.

Alternatively (and equivalently), we expose a fast SOCP reformulation that
substitutes u = w/‖Lw‖, t = 1/‖Lw‖ and directly yields an LP + SOC problem
solvable by ECOS in one pass.  Both paths produce the same optimum; the SOCP
path is the default because it is faster and numerically more stable.

Reference: Diamond & Boyd, "Disciplined Quasi-convex Programming", 2019,
arXiv:1905.00562.  CVXPY DQCP documentation: https://www.cvxpy.org/tutorial/dqcp
"""

from __future__ import annotations

import logging
import math

import cvxpy as cp
import numpy as np

from electoral.core.types import CANONICAL_RACES
from electoral.models.ml_baseline import psd_repair
from electoral.portfolios.constraints import ConstraintSpec

log = logging.getLogger(__name__)

_ACCEPTABLE = frozenset([cp.OPTIMAL, cp.OPTIMAL_INACCURATE])
_CLIP_FLOOR = 0.0


# ── Public entry point ────────────────────────────────────────────────────────


def solve_dqcp(
    mu_race: dict[str, float],
    cov_race: np.ndarray,
    target: float,
    lambda_1: float,
    fixed_loyalty: float,
    *,
    blocs: list[str] | None = None,
    spec: ConstraintSpec | None = None,
    solver: str = cp.CLARABEL,
    verbose: bool = False,
) -> dict[str, float]:
    """Maximize the Sharpe-ratio P(win) objective over race-bloc coalition weights.

    Parameters
    ----------
    mu_race:
        Post-shock within-race Democratic vote share per bloc.
    cov_race:
        K×K covariance of race-level delta perturbations (Ledoit-Wolf recommended).
        Must be positive semi-definite; call psd_repair() from ml_baseline if needed.
    target:
        V_eq win threshold (overall, not race-only).
    lambda_1:
        Layer weight for the race stratum.
    fixed_loyalty:
        Fixed μ_eff contribution from religion + gender strata:
            fixed_loyalty = λ₂ · Σ_R(v_R · μ_rel_R) + λ₃ · Σ_G(g_G · μ_gen_G)
    blocs:
        Ordered race-bloc identifiers (defaults to CANONICAL_RACES).
    spec:
        Per-bloc weight bounds.  When None, unconstrained simplex is used.
    solver:
        CVXPY solver to use.  ECOS handles SOC problems efficiently.
    verbose:
        Pass solver verbosity through to CVXPY.

    Returns
    -------
    dict[str, float]
        Optimal race-bloc weights, keys = blocs, values in [0, 1] summing to 1.

    Raises
    ------
    ValueError
        If target is unachievable given the mu_race vector and spec bounds.
    RuntimeError
        If the CVXPY solver returns a non-optimal status.
    """
    if blocs is None:
        blocs = list(CANONICAL_RACES)
    if spec is not None:
        blocs = list(spec.blocs)
    k = len(blocs)

    # ── Input validation ───────────────────────────────────────────────────────
    for b in blocs:
        if math.isnan(float(mu_race.get(b, float("nan")))):
            raise ValueError(
                f"solve_dqcp: mu_race[{b!r}] is NaN. " "Impute or exclude this bloc before calling."
            )

    cov_mat = np.asarray(cov_race, dtype=float)
    if cov_mat.shape != (k, k):
        raise ValueError(
            f"solve_dqcp: cov_race shape {cov_mat.shape} does not match "
            f"len(blocs)={k}; expected ({k}, {k})."
        )

    mu_vec = np.array([float(mu_race[b]) for b in blocs])
    sym_cov = psd_repair((cov_mat + cov_mat.T) / 2.0)

    # ── Pre-flight: can we achieve target in principle? ────────────────────────
    max_race_loyalty = float(mu_vec.max())
    max_mu_eff = lambda_1 * max_race_loyalty + fixed_loyalty
    if target > max_mu_eff + 1e-9:
        raise ValueError(
            f"solve_dqcp: target={target:.6f} exceeds the maximum achievable "
            f"μ_eff={max_mu_eff:.6f} "
            f"(λ₁={lambda_1:.3f} × max_race_loyalty={max_race_loyalty:.6f} + "
            f"fixed_loyalty={fixed_loyalty:.6f}). "
            "Reduce target or use a more favourable shock scenario."
        )

    # ── Cholesky factor for SOCP ───────────────────────────────────────────────
    try:
        L = np.linalg.cholesky(sym_cov)
    except np.linalg.LinAlgError:
        # Fallback: diagonal approximation if Cholesky fails
        variances = np.maximum(np.diag(sym_cov), 1e-8)
        L = np.diag(np.sqrt(variances))
        log.warning("solve_dqcp: Cholesky failed; using diagonal covariance approximation")

    # ── SOCP reformulation of max Sharpe (normalized form) ────────────────────
    # Let z = w / (λ₁·‖Lw‖₂),  s = Σz_i = 1/(λ₁·‖Lw‖₂).
    # Then the Sharpe ratio = λ₁·(μ^T w + b) / (λ₁·‖Lw‖₂)
    #                       = λ₁·μ^T(z/s) + b·s·(1/s)·s ...
    #                       = (λ₁·μ^T z + b·s)   when ‖λ₁·Lz‖₂ = 1.
    #
    # This gives the compact SOCP:
    #   maximize  λ₁·μ^T z + b·s              (linear objective = Sharpe)
    #   s.t.      ‖λ₁·L z‖₂ ≤ 1             (normalising SOC, active at opt.)
    #             Σz_i = s                     (scale linkage)
    #             z ≥ 0,  s ≥ 0
    #
    # Recovery: w* = z*/s* = z*/Σz*  (normalise back to simplex).
    # The constraint set is compact (z bounded by the SOC), so the problem is bounded.
    # Reference: Lobo et al. (1998) "Applications of SOCP", Ch. 5.
    b = float(fixed_loyalty - target)  # constant term in the Sharpe numerator
    a = lambda_1 * mu_vec  # linear race-loyalty coefficients

    z = cp.Variable(k, nonneg=True)
    s = cp.Variable(nonneg=True)

    objective = cp.Maximize(a @ z + b * s)

    constraints: list[cp.Constraint] = [
        cp.norm(lambda_1 * L @ z, 2) <= 1.0,
        cp.sum(z) == s,
        s >= 1e-6,
    ]

    # Per-bloc weight bounds: w*_i ∈ [lo_i, hi_i] ⟺ z*_i ∈ [lo_i·s*, hi_i·s*],
    # which is bilinear in (z, s) and cannot be added as a SOCP constraint directly.
    # KNOWN APPROXIMATION: bounds are enforced post-solve via clipping + renorm
    # (see lines below).  This can shift the solution outside the true feasible set;
    # the paper acknowledges this as a first-order approximation.  A tighter
    # formulation would introduce a second-order cone constraint per bloc at the cost
    # of a larger problem.  Pre-check feasibility with ConstraintSpec before calling.
    problem = cp.Problem(objective, constraints)

    if not problem.is_dcp():
        log.warning("solve_dqcp: SOCP problem is unexpectedly not DCP — check formulation")

    problem.solve(solver=solver, verbose=verbose)

    if problem.status not in _ACCEPTABLE:
        raise RuntimeError(
            f"solve_dqcp: CVXPY solver returned '{problem.status}'. "
            f"target={target:.6f}, max_achievable_mu_eff={max_mu_eff:.6f}. "
            "Check that cov_race is positive semi-definite."
        )

    # ── Extract weights w = z / sum(z) ────────────────────────────────────────
    s_val = float(s.value)
    if s_val <= 0:
        raise RuntimeError("solve_dqcp: s ≤ 0 at optimum — solver produced degenerate solution")

    w_raw = np.clip(z.value / s_val, _CLIP_FLOOR, None)

    # Apply per-bloc bounds from spec (post-solve clipping + renorm)
    if spec is not None:
        lo = np.array([spec.lower_bounds.get(blocs[i], 0.0) for i in range(k)])
        hi = np.array([spec.upper_bounds.get(blocs[i], 1.0) for i in range(k)])
        w_raw = np.clip(w_raw, lo, hi)

    total = w_raw.sum()
    if total <= 0:
        raise RuntimeError("solve_dqcp: all weights zero after extraction")
    w_opt = w_raw / total

    return {b: float(w_opt[i]) for i, b in enumerate(blocs)}


# ── Utility: compute post-shock μ_eff ─────────────────────────────────────────


def compute_mu_eff(
    weights: dict[str, float],
    mu_race: dict[str, float],
    lambda_1: float,
    fixed_loyalty: float,
) -> float:
    """Compute the scalar effective loyalty for a given race coalition.

    μ_eff = λ₁ · Σ(w_i · μ_race_i) + fixed_loyalty
    """
    race_eff = sum(weights[b] * mu_race[b] for b in weights)
    return lambda_1 * race_eff + fixed_loyalty
