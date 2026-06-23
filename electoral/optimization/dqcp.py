"""DQCP Sharpe-ratio optimizer: maximize P(win) for a post-shock coalition.

═══════════════════════════════════════════════════════════════════════════════
PLAIN-ENGLISH OVERVIEW (read this before the math below)
═══════════════════════════════════════════════════════════════════════════════
THE QUESTION this file answers: after a shock has shifted each bloc's vote share,
how should a campaign split its reliance across the 5 race blocs (the "coalition
weights" w, which sum to 1) to MAXIMIZE THE PROBABILITY OF WINNING?

THE PORTFOLIO ANALOGY: this is borrowed from finance. An investor splits money
across stocks to maximize return while controlling risk. Here:
    • each race bloc is a "stock",
    • its vote share μ is the "expected return",
    • our uncertainty about the shock's effect (the covariance Σ_Δ) is the "risk",
    • the coalition weights w are the "portfolio allocation".
The best portfolio maximizes the **Sharpe ratio** = (reward − threshold) / risk.
Here reward is the coalition's effective loyalty μ_eff, the threshold is the win
bar V_eq, and risk is the spread of outcomes. A higher Sharpe ratio literally
means a higher probability of clearing V_eq (winning), because — assuming a
roughly normal spread of outcomes — P(win) = Φ(Sharpe ratio), where Φ is the
normal CDF (an increasing function). So maximizing the Sharpe ratio == maximizing
P(win). That is why we never use plain "minimize variance": in a losing scenario
you must take on risk to have any chance of clearing V_eq.

WHY THIS IS HARD: the Sharpe ratio is a FRACTION (reward on top, risk on bottom).
Fractions are not "convex", so ordinary convex optimizers can't solve it directly.
But it IS "quasi-concave" (a weaker, well-behaved property). Two standard tricks
turn it into something a solver can handle exactly:
    1. DQCP bisection — guess a target Sharpe value t, ask "is t achievable?"
       (a convex check), and binary-search for the best t. (Diamond & Boyd 2019.)
    2. An SOCP reformulation (Charnes–Cooper / Lobo et al.) — a clever change of
       variables that converts the fraction into a single-pass convex program.
We use trick #2 by default: it is faster and more numerically stable. The two
give the same answer. The detailed math for both is documented below.

If the math notation below is unfamiliar, you do not need it to USE this module:
call solve_dqcp(...) and read the parameter docs. The notation is here so the
result is auditable and reproducible for the paper.

═══════════════════════════════════════════════════════════════════════════════
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

from electoral.core.types import CANONICAL_GENDERS, CANONICAL_RACES, CANONICAL_RELIGIONS
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
        CVXPY solver to use.  Defaults to CLARABEL (a modern interior-point
        solver that handles second-order-cone problems efficiently and ships
        with CVXPY).  ECOS is a fine alternative if you pass it explicitly.
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

    # Per-bloc weight bounds, enforced EXACTLY inside the SOCP.
    # In the recovered weights w = z / s, the bound lo_i ≤ w_i ≤ hi_i is identical
    # to lo_i·s ≤ z_i ≤ hi_i·s. Because s is a *variable* and lo_i/hi_i are
    # constants, these are LINEAR constraints (not bilinear — z_i·s would be), so
    # they're perfectly valid in an SOCP. This replaces the old post-solve
    # clip+renorm, which could renormalize a clipped weight back outside its bound.
    if spec is not None:
        lo = np.array([spec.lower.get(blocs[i], 0.0) for i in range(k)])
        hi = np.array([spec.upper.get(blocs[i], 1.0) for i in range(k)])
        constraints.append(z >= cp.multiply(lo, s))
        constraints.append(z <= cp.multiply(hi, s))

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

    # Per-bloc bounds (if any) were enforced exactly inside the SOCP above, so no
    # post-solve clipping is needed. We only clip at 0 to wipe tiny negative
    # round-off from the solver before the final renormalization.
    w_raw = np.clip(z.value / s_val, _CLIP_FLOOR, None)

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
    """Compute the scalar "effective loyalty" μ_eff for a given race coalition.

    This collapses a whole coalition down to ONE number you can compare against
    the win threshold V_eq: if μ_eff ≥ V_eq the coalition wins (on the point
    estimate). It blends the three strata:

        μ_eff = λ₁ · Σ(w_i · μ_race_i)        ← the race part we optimize
                + fixed_loyalty               ← religion + gender, precomputed

    `fixed_loyalty` already bundles λ₂·(religion contribution) + λ₃·(gender
    contribution); those strata aren't optimized, so we pass them in as a single
    constant rather than recomputing them here.

    Args:
        weights:       race_id → coalition weight (the w being evaluated).
        mu_race:       race_id → post-shock vote share (the μ).
        lambda_1:      race-stratum layer weight.
        fixed_loyalty: precomputed religion+gender contribution to μ_eff.
    """
    # Weighted sum of per-bloc vote shares = the coalition's race-only loyalty.
    # (Iterating `weights` keys assumes mu_race has every bloc weights does.)
    race_eff = sum(weights[b] * mu_race[b] for b in weights)
    return lambda_1 * race_eff + fixed_loyalty


# ── Utility: the religion + gender contribution to μ_eff ──────────────────────


def compute_fixed_loyalty(
    mu_religion: dict[str, float],
    mu_gender: dict[str, float],
    lambda_2: float,
    lambda_3: float,
    deltas_religion: dict[str, float] | None = None,
    deltas_gender: dict[str, float] | None = None,
) -> float:
    """Compute the religion+gender ("fixed") contribution to μ_eff.

    Returns λ₂·Σ(v_R·μ_relR) + λ₃·Σ(g_G·μ_genG) with EQUAL within-stratum weights
    (v_R = 1/n_rel, g_G = 1/n_gen — a placeholder until raking.py supplies real
    panel marginals). This is the value passed as `fixed_loyalty` to the optimizer
    and Monte Carlo, replacing the old neutral (λ₂+λ₃)·0.5 placeholder.

    "Fixed" because race is the only stratum the optimizer rebalances; religion
    and gender loyalties still SHIFT with the shock, but they aren't decision
    variables. If `deltas_religion`/`deltas_gender` are given, they're added to the
    baseline loyalties (and clipped to [0, 1]) to get the POST-shock values — the
    correct, signal-using choice, consistent with how race uses μ̃ = μ + Δ.

    Args:
        mu_religion / mu_gender: baseline within-bloc loyalties per stratum.
        lambda_2 / lambda_3:     religion / gender layer weights.
        deltas_religion / deltas_gender: optional per-bloc shock deltas; when
            omitted, the baseline (pre-shock) loyalties are used as-is.
    """
    d_rel = deltas_religion or {}
    d_gen = deltas_gender or {}

    # Post-shock within-bloc loyalties, clipped to a valid [0, 1] share.
    rel = {r: min(1.0, max(0.0, mu_religion[r] + d_rel.get(r, 0.0))) for r in CANONICAL_RELIGIONS}
    gen = {g: min(1.0, max(0.0, mu_gender[g] + d_gen.get(g, 0.0))) for g in CANONICAL_GENDERS}

    n_rel = len(CANONICAL_RELIGIONS)
    n_gen = len(CANONICAL_GENDERS)
    religion_term = lambda_2 * sum(rel[r] / n_rel for r in CANONICAL_RELIGIONS)
    gender_term = lambda_3 * sum(gen[g] / n_gen for g in CANONICAL_GENDERS)
    return float(religion_term + gender_term)
