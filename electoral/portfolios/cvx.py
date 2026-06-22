"""Minimum-variance baseline portfolio optimizer.

WHY THIS EXISTS (and how it differs from the real optimizer): the production
optimizer maximizes P(win) — the Sharpe ratio (optimization/dqcp.py, cvx.py). THIS
one does something simpler and deliberately different: it finds the LOWEST-RISK
coalition that still clears the win threshold ("min variance subject to mu^T w ≥
target"). That is the textbook-conservative choice, and it's WRONG for the main
problem — in a losing scenario you must TAKE ON risk to have any shot at winning,
which min-variance refuses to do. So this is kept only as a BASELINE to compare
against in the paper ("our P(win) optimizer beats plain min-variance"), not as the
thing that drives the app. A QP ("quadratic program") is the optimization family
for a quadratic objective (variance = w^T Σ w) with linear constraints.

solve_baseline() is the QP comparison baseline for the DQCP optimizer.  It
finds the lowest-variance race-bloc coalition that still meets the win-
probability loyalty threshold V_eq:

    min   w^T Σ w
    s.t.  sum(w_i) = 1
          w_i >= 0  for all i
          mu^T w >= target

If the solver reports infeasibility the constraint is relaxed by *relax_step*
and the QP is re-submitted up to *max_retries* times.  Each relaxation is
logged at WARNING level.

The DQCP Sharpe-ratio optimizer (max P(win)) lives in the Week-5 kernel.
This baseline is used for:
  - Benchmarking: how much additional win probability does the DQCP gain over
    the minimum-variance allocation?
  - Feasibility pre-check: if this QP is infeasible, the DQCP will be too.
  - V_eq calibration: the minimum target that admits a feasible allocation.
"""

from __future__ import annotations

import logging
import math

import cvxpy as cp
import numpy as np

from electoral.core.types import CANONICAL_RACES
from electoral.portfolios.constraints import ConstraintSpec

log = logging.getLogger(__name__)

# Solver status values that constitute a usable solution.
_ACCEPTABLE = frozenset([cp.OPTIMAL, cp.OPTIMAL_INACCURATE])

# Statuses that indicate infeasibility and warrant a relaxation retry.
# Solver errors and unbounded results are NOT retried — they indicate a
# problem formulation issue unrelated to the tightness of the constraint.
_INFEASIBLE_STATUSES = frozenset([cp.INFEASIBLE, cp.INFEASIBLE_INACCURATE])

# Clip threshold: CVXPY can return weights of O(-1e-10) for zero-weighted
# blocs due to interior-point numerics.  Clip at this floor before renorm.
_CLIP_FLOOR = 0.0


def solve_baseline(
    mu: dict[str, float],
    cov: np.ndarray,
    target: float,
    *,
    blocs: list[str] | None = None,
    spec: ConstraintSpec | None = None,
    relax_step: float = 0.01,
    max_retries: int = 5,
) -> dict[str, float]:
    """Solve the minimum-variance portfolio subject to a loyalty floor.

    Parameters
    ----------
    mu:
        Mapping of bloc_id -> mean vote share for the party being optimized.
        Must contain a finite (non-NaN) value for every entry in *blocs*.
    cov:
        Square covariance matrix whose rows/cols correspond to *blocs* in
        order.  Must be positive semi-definite; call ``psd_repair()`` from
        ``ml_baseline`` first if the matrix may have negative eigenvalues.
        Floating-point asymmetry is corrected internally via symmetrization.
    target:
        V_eq win threshold.  The loyalty constraint is ``mu @ w >= target``.
    blocs:
        Ordered bloc identifiers for the rows/cols of *cov*.  Defaults to
        ``CANONICAL_RACES`` (the five race blocs used by the optimizer).
        Ignored when *spec* is provided — the spec's blocs take precedence.
    spec:
        ``ConstraintSpec`` with per-bloc lower/upper bounds.  When None, an
        unconstrained simplex is used (all weights in [0, 1]).
    relax_step:
        Amount to subtract from *target* on each retry (default 0.01 = 1 pp).
    max_retries:
        Maximum number of relaxation attempts before raising (default 5).

    Returns
    -------
    dict[str, float]
        Optimal race-bloc weights, one key per entry in *blocs*, values in
        [0, 1] summing to 1.0 ± 1e-9.

    Raises
    ------
    ValueError
        If any entry of *mu* for a requested bloc is NaN, if the shape of
        *cov* does not match ``len(blocs)``, if *target* provably exceeds the
        maximum achievable loyalty, or if the problem remains infeasible after
        *max_retries* relaxations.
    RuntimeError
        If the CVXPY solver returns a non-infeasibility failure (e.g.
        ``solver_error``, ``unbounded``) that relaxation cannot resolve.
    """
    # ── Trivial rejection: impossible target ──────────────────────────────────
    # Vote share is bounded by [0, 1]; no convex combination can exceed 1.
    # Checked before anything else so callers get a clear message immediately,
    # without building the spec or touching the covariance matrix.
    if target > 1.0:
        raise ValueError(
            f"solve_baseline: target={target:.6f} > 1.0 is impossible; "
            "loyalty is a vote share in [0, 1]."
        )

    if spec is not None:
        blocs = list(spec.blocs)
    elif blocs is None:
        blocs = list(CANONICAL_RACES)
    n = len(blocs)

    if spec is None:
        spec = ConstraintSpec.default(blocs)

    # ── Input validation ───────────────────────────────────────────────────────
    nan_blocs = [b for b in blocs if math.isnan(float(mu.get(b, float("nan"))))]
    if nan_blocs:
        raise ValueError(
            f"solve_baseline: mu contains NaN for bloc(s) {nan_blocs}. "
            "Impute or exclude these blocs before calling."
        )

    cov_mat = np.asarray(cov, dtype=float)
    if cov_mat.shape != (n, n):
        raise ValueError(
            f"solve_baseline: cov shape {cov_mat.shape} does not match "
            f"len(blocs)={n}; expected ({n}, {n})."
        )

    mu_vec = np.array([float(mu[b]) for b in blocs])

    # ── Pre-flight feasibility check ───────────────────────────────────────────
    # Upper bound on achievable loyalty given per-bloc weight caps in spec.
    max_achievable = spec.max_achievable_loyalty(mu)
    if target > max_achievable + 1e-9:
        raise ValueError(
            f"solve_baseline: target={target:.6f} exceeds the maximum achievable "
            f"loyalty {max_achievable:.6f} given the ConstraintSpec bounds. "
            "Reduce target, widen upper bounds, or lower the floor constraints."
        )

    # ── Single-bloc short-circuit ─────────────────────────────────────────────
    # With one bloc the simplex has a single point (w = [1]).  No solver call
    # is needed; the pre-flight check above already confirmed target is met.
    if n == 1:
        return {blocs[0]: 1.0}

    # ── Symmetrize covariance ──────────────────────────────────────────────────
    # Corrects floating-point asymmetry before quad_form (CVXPY requires a
    # symmetric matrix for DCP recognition as a convex objective).
    sym_cov = (cov_mat + cov_mat.T) / 2.0

    # ── Solve with relaxation fallback ────────────────────────────────────────
    # Attempt 0: original target.
    # Attempt k (k >= 1): relaxed by k * relax_step; log WARNING before each.
    # If the solver returns a non-infeasibility failure, raise immediately.
    # If all attempts are infeasible, raise ValueError with the final target.
    w = cp.Variable(n, nonneg=True)
    current_target = target
    last_status: str = ""

    for attempt in range(max_retries + 1):
        if attempt > 0:
            current_target = target - attempt * relax_step
            log.warning(
                "solve_baseline: solver returned '%s' for target=%.6f; "
                "relaxing to %.6f (retry %d/%d)",
                last_status,
                target - (attempt - 1) * relax_step,
                current_target,
                attempt,
                max_retries,
            )

        constraints = spec.cvxpy_constraints(w) + [mu_vec @ w >= current_target]
        problem = cp.Problem(cp.Minimize(cp.quad_form(w, sym_cov)), constraints)
        problem.solve()
        last_status = problem.status

        if problem.status in _ACCEPTABLE:
            if attempt > 0:
                log.info(
                    "solve_baseline: feasible solution found at relaxed target %.6f "
                    "(%d relaxation(s) applied, original target=%.6f)",
                    current_target,
                    attempt,
                    target,
                )
            break

        if problem.status not in _INFEASIBLE_STATUSES:
            raise RuntimeError(
                f"solve_baseline: CVXPY solver returned '{problem.status}' "
                "(non-infeasibility failure; relaxation cannot help). "
                f"target={target:.6f}, max_achievable={max_achievable:.6f}. "
                "Check the covariance matrix; call psd_repair() if eigenvalues "
                "are negative."
            )
    else:
        raise ValueError(
            f"solve_baseline: still infeasible after {max_retries} relaxation(s). "
            f"Original target={target:.6f}; "
            f"final constraint attempted={current_target:.6f}."
        )

    # ── Extract and clean weights ──────────────────────────────────────────────
    # Interior-point solvers can return tiny negative values (O(-1e-10)) for
    # zero-weighted blocs.  Clip at zero and renormalize to enforce the simplex
    # constraint exactly.
    raw = np.clip(w.value, _CLIP_FLOOR, None)
    total = raw.sum()
    if total <= 0:
        raise RuntimeError(
            "solve_baseline: all weights clipped to zero after solve. "
            f"Solver status was '{problem.status}'."
        )
    weights = raw / total

    return {b: float(weights[i]) for i, b in enumerate(blocs)}
