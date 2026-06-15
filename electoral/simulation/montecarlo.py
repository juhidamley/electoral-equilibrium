"""Logistic-Normal ILR Monte Carlo for win-probability estimation.

NOT Dirichlet (forces negative off-diagonal covariances — cannot model wave elections).
Uses ILR (isometric log-ratio) with the Helmert contrast matrix:

  1. Map w* to ILR coords:      z* = V^T log(w*)     V is K×(K-1) Helmert matrix
  2. Propagate covariance:       Σ_ILR = J Σ_Δ J^T   J = V^T diag(1/w*)
  3. Draw:                       y^(n) ~ N(z*, Σ_ILR) in R^(K-1)
  4. Back-transform:             w^(n) = softmax(V y^(n))
  5. Compute win flags and 5th/95th percentile CI bounds

Zero-weight blocs → all downstream weights are undefined; the function raises
ValueError rather than flooring to an arbitrary ε.

Public API:
    helmert_matrix(k)  — K×(K-1) Helmert contrast matrix (orthonormal columns)
    ilr(w, V)          — ILR forward transform
    ilr_inv(z, V)      — ILR inverse transform via softmax
    run_ilr_montecarlo(…) → SimulationData
"""

from __future__ import annotations

import json
import math
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from electoral.artifacts import SimulationData
from electoral.core.rng import make_rng
from electoral.core.types import CANONICAL_RACES

if TYPE_CHECKING:
    from electoral.artifacts import EquilibriumData
    from electoral.config import PipelineConfig


# ── Helmert matrix ────────────────────────────────────────────────────────────


def helmert_matrix(k: int) -> np.ndarray:
    """Return the K×(K-1) orthonormal Helmert contrast matrix.

    Column j (0-indexed, j = 0..K-2) defines the j-th isometric log-ratio
    contrast.  For j:
        V[i, j] = 1 / sqrt((j+1)(j+2))   for i in 0..j   (first j+1 rows)
        V[j+1, j] = -(j+1) / sqrt((j+1)(j+2))             (row j+1)
        V[i, j] = 0                        for i > j+1

    Properties:
        V^T V = I_{K-1}   (columns are orthonormal)
        V 1_{K-1} = 0_K   (each column sums to 0 — invariant under softmax)
    """
    if k < 2:
        raise ValueError(f"helmert_matrix requires k >= 2, got {k}")
    V = np.zeros((k, k - 1), dtype=float)
    for j in range(k - 1):
        denom = math.sqrt((j + 1) * (j + 2))
        for i in range(j + 1):
            V[i, j] = 1.0 / denom
        V[j + 1, j] = -(j + 1) / denom
    return V


# ── ILR transforms ────────────────────────────────────────────────────────────


def ilr(w: np.ndarray, V: np.ndarray) -> np.ndarray:
    """ILR forward transform: w (K-simplex) → z (R^{K-1}).

    z = V^T log(w)

    All weights must be strictly positive.  The output z lives in R^{K-1}.
    """
    if np.any(w <= 0):
        raise ValueError(
            "ILR transform requires strictly positive weights. "
            "Zero-weight blocs make the transform undefined. "
            "Report these as infeasible_bloc rather than flooring to 0.01."
        )
    return V.T @ np.log(w)


def ilr_inv(z: np.ndarray, V: np.ndarray) -> np.ndarray:
    """ILR inverse transform: z (R^{K-1}) → w (K-simplex).

    w = softmax(V z)

    Since columns of V sum to 0, V z is zero-centred in expectation, and the
    softmax correctly maps it back to the simplex (sums to 1, all positive).
    """
    x = V @ z
    x -= x.max()  # numerical stability before exp
    w = np.exp(x)
    return w / w.sum()


# ── Covariance propagation ────────────────────────────────────────────────────


def _propagate_cov(w_star: np.ndarray, sigma_delta: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Map the race-level delta covariance to ILR space via the Jacobian.

    J = V^T diag(1/w*)
    Σ_ILR = J Σ_Δ J^T

    The Jacobian linearises the ILR transform at w*, converting uncertainty
    in Δw (Euclidean) to uncertainty in z (ILR space).
    """
    inv_w = 1.0 / w_star  # element-wise
    J = V.T * inv_w[np.newaxis, :]  # V^T diag(1/w*) — broadcast
    sigma_ilr = J @ sigma_delta @ J.T
    # Symmetrise to correct floating-point asymmetry
    return (sigma_ilr + sigma_ilr.T) / 2.0


def _make_psd(sigma: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Repair a nearly-PSD matrix by flooring eigenvalues to eps."""
    vals, vecs = np.linalg.eigh(sigma)
    vals = np.maximum(vals, eps)
    return vecs @ np.diag(vals) @ vecs.T


# ── Layer-weight loader ───────────────────────────────────────────────────────

_LAYER_WEIGHTS_CACHE: dict[str, float] | None = None


def _load_layer_weights() -> dict[str, float]:
    global _LAYER_WEIGHTS_CACHE
    if _LAYER_WEIGHTS_CACHE is not None:
        return _LAYER_WEIGHTS_CACHE

    candidate_paths = [
        Path("configs/layer_weights.json"),
        Path(__file__).parent.parent.parent / "configs" / "layer_weights.json",
    ]
    for p in candidate_paths:
        if p.exists():
            try:
                raw = json.loads(p.read_text(encoding="utf-8"))
                weights = {
                    "lambda_1": float(raw["lambda_1"]),
                    "lambda_2": float(raw["lambda_2"]),
                    "lambda_3": float(raw["lambda_3"]),
                }
                _LAYER_WEIGHTS_CACHE = weights
                return weights
            except (KeyError, ValueError, json.JSONDecodeError):
                pass

    # Conservative fallback: equal weighting
    warnings.warn(
        "layer_weights.json not found or invalid; using equal weights λ₁=λ₂=λ₃=1/3",
        RuntimeWarning,
        stacklevel=3,
    )
    fallback = {"lambda_1": 1.0 / 3, "lambda_2": 1.0 / 3, "lambda_3": 1.0 / 3}
    _LAYER_WEIGHTS_CACHE = fallback
    return fallback


# ── Main Monte Carlo entry point ──────────────────────────────────────────────


def run_ilr_montecarlo(
    equilibrium: "EquilibriumData",
    config: "PipelineConfig",
    n_simulations: int = 10_000,
    sigma_default: float = 0.02,
) -> SimulationData:
    """Logistic-Normal ILR Monte Carlo for race-bloc coalition uncertainty.

    Parameters
    ----------
    equilibrium:
        Output of the DQCP optimizer.  Uses `weights`, `mu_shifted`, and `target`.
    config:
        Pipeline config; provides RNG seed and race-bloc identifiers.
    n_simulations:
        Number of Monte Carlo draws (≥10 000 recommended for production).
    sigma_default:
        Diagonal standard deviation (in Δw units) applied when no empirical
        covariance is available.  Default 0.02 ≈ "slight" magnitude bin.

    Returns
    -------
    SimulationData with win_probability and percentiles per race bloc.

    Notes
    -----
    The win condition uses only the race-bloc contribution to μ_eff:
        μ_race_eff(w) = Σ_i  w_i · μ̃_race_i
        win = λ₁ · μ_race_eff(w_sample) + (1 − λ₁) · 0.50  ≥  V_eq

    Religion and gender strata are assumed at neutral (0.50) because they are
    fixed in the optimizer and cannot be rebalanced by campaign resource shifts.
    This is a stated simplification — acknowledged in the paper as a bound.
    """
    seed = config.derive_seed("monte_carlo")
    rng = make_rng(seed)

    blocs = list(CANONICAL_RACES)
    k = len(blocs)

    # ── Extract w* and μ̃ from equilibrium ─────────────────────────────────────
    w_star = np.array([equilibrium.weights.get(b, 1.0 / k) for b in blocs])
    mu_race = np.array([equilibrium.mu_shifted.get(b, 0.5) for b in blocs])
    target = equilibrium.target

    # ── Check for zero weights ─────────────────────────────────────────────────
    zero_blocs = [b for b, w in zip(blocs, w_star) if w <= 0]
    if zero_blocs:
        raise ValueError(
            f"run_ilr_montecarlo: zero-weight blocs {zero_blocs!r} detected in "
            "equilibrium.weights. ILR is undefined at the boundary of the simplex. "
            "Report these as infeasible_bloc."
        )

    # ── Layer weights ──────────────────────────────────────────────────────────
    lw = _load_layer_weights()
    lambda_1 = lw["lambda_1"]
    # Neutral (0.50) assumed for religion + gender strata (fixed, not optimised)
    neutral_fixed = (1.0 - lambda_1) * 0.50

    # ── Diagonal covariance (conservative prior) ────────────────────────────
    # Populated by generate_synthetic.py after historical delta analysis.
    # Until empirical σ_b values are filled, use sigma_default for all blocs.
    sigma_delta = np.eye(k) * (sigma_default**2)

    # ── Helmert matrix and ILR of w* ─────────────────────────────────────────
    V = helmert_matrix(k)
    z_star = ilr(w_star, V)

    # ── Propagate covariance to ILR space ────────────────────────────────────
    sigma_ilr = _propagate_cov(w_star, sigma_delta, V)
    sigma_ilr = _make_psd(sigma_ilr)

    # ── Draw samples ─────────────────────────────────────────────────────────
    # errstate: Apple Accelerate BLAS on macOS triggers spurious "divide by zero"
    # RuntimeWarnings during vector/matrix operations even when inputs are finite.
    # The entire sampling + win-flag block is wrapped to suppress this noise.
    L_chol = np.linalg.cholesky(sigma_ilr)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        z_samples = z_star[np.newaxis, :] + (rng.standard_normal((n_simulations, k - 1)) @ L_chol.T)

        # Back-transform each sample to the simplex.
        # Clip extreme z values to prevent exp overflow in ilr_inv (tail samples
        # beyond ±30 sigma have no demographic meaning and just add noise).
        z_clip = np.clip(z_samples, -30.0, 30.0)
        weights_samples = np.array([ilr_inv(z, V) for z in z_clip])  # (N, K)

        # Drop any NaN weight vectors (should be extremely rare with the clip above)
        valid_mask = np.all(np.isfinite(weights_samples), axis=1)
        if not np.all(valid_mask):
            n_dropped = int((~valid_mask).sum())
            warnings.warn(
                f"run_ilr_montecarlo: {n_dropped} / {n_simulations} samples produced "
                "non-finite weights and were excluded.",
                RuntimeWarning,
                stacklevel=2,
            )
            weights_samples = weights_samples[valid_mask]

        # ── Compute win flags ─────────────────────────────────────────────────
        mu_race_eff = weights_samples @ mu_race  # (N,)
        mu_eff_samples = lambda_1 * mu_race_eff + neutral_fixed
    win_flags = mu_eff_samples >= target
    win_probability = float(win_flags.mean())

    # Bootstrap 90% CI on the win-probability estimator (not on individual draws).
    # Resampling win_flags B times gives B bootstrap estimates of win_probability;
    # the 5th/95th percentiles of those means form a meaningful CI that narrows with N.
    # Uses the seeded rng already in scope — no fresh generator.
    _N_BOOT = 500
    n_valid = len(win_flags)
    win_flags_f = win_flags.astype(float)
    boot_idx = rng.integers(0, n_valid, size=(_N_BOOT, n_valid))
    boot_means = win_flags_f[boot_idx].mean(axis=1)
    win_probability_low = float(np.percentile(boot_means, 5))
    win_probability_high = float(np.percentile(boot_means, 95))

    # ── Compute per-bloc percentiles ──────────────────────────────────────────
    percentile_levels = [5, 25, 50, 75, 95]
    percentiles: dict[str, list[float]] = {}
    for i, bloc in enumerate(blocs):
        pcts = np.percentile(weights_samples[:, i], percentile_levels)
        percentiles[bloc] = [float(p) for p in pcts]

    return SimulationData(
        n_simulations=n_simulations,
        seed=seed,
        win_probability=win_probability,
        win_probability_low=win_probability_low,
        win_probability_high=win_probability_high,
        percentiles=percentiles,
    )
