"""Logistic-Normal ILR Monte Carlo for win-probability estimation.

═══════════════════════════════════════════════════════════════════════════════
PLAIN-ENGLISH OVERVIEW (read this before the math)
═══════════════════════════════════════════════════════════════════════════════
WHAT THIS FILE DOES: the optimizer gives us ONE "best" coalition w* and a single
predicted win/lose. But every input was uncertain (the shock deltas are
estimates!), so a single yes/no is overconfident. This file answers the better
question: "given that uncertainty, what's the PROBABILITY of winning, with a
confidence interval?" It does that with **Monte Carlo simulation** — generate
thousands of plausible variations of the coalition, check how many win, and
report the fraction.

THE HARD PART — you can't just add random noise to the weights. The 5 race
weights live on the **simplex**: they must each stay ≥ 0 AND sum to exactly 1.
If you naively add Gaussian noise (w_i + ε), you'll get negative weights or sums
≠ 1 — nonsense coalitions. We need a way to "jiggle" the weights that always
lands back on the simplex.

THE TRICK (ILR — isometric log-ratio): transform the constrained weights into an
ordinary *unconstrained* space (plain R^{K-1}, where any real vector is valid),
add normal noise THERE, then transform back. The round-trip back is `softmax`,
which by construction always returns positive numbers that sum to 1 — a valid
coalition, guaranteed. This is the standard way to do statistics on
compositional data (parts of a whole); the underlying theory is called Aitchison
geometry. ILR is just a clean, distance-preserving ("isometric") choice of that
transform; the Helmert matrix (below) is the specific recipe for it.

WHY NOT SIMPLER ALTERNATIVES (these are deliberate design decisions):
  • NOT a Dirichlet distribution — its math forces every pair of blocs to be
    NEGATIVELY correlated, so it literally cannot represent a "wave election"
    where many blocs move the same direction together.
  • NOT the "delta method" with an ε floor — clamping w_i ≥ 0.01 bends the
    geometry non-linearly and biases the result.

THE 5-STEP RECIPE (each step is a function/section below):

  1. Map w* to ILR coords:      z* = V^T log(w*)     V is K×(K-1) Helmert matrix
  2. Propagate covariance:       Σ_ILR = J Σ_Δ J^T   J = V^T diag(1/w*)
  3. Draw:                       y^(n) ~ N(z*, Σ_ILR) in R^(K-1)
  4. Back-transform:             w^(n) = softmax(V y^(n))
  5. Compute win flags and 5th/95th percentile CI bounds

(In words: 1 = jump to the unconstrained space, 2 = carry the uncertainty over
into that space, 3 = draw thousands of noisy samples there, 4 = bring each one
safely back to a valid coalition, 5 = count wins and form the confidence band.)

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

    This is the "bring it safely back to a valid coalition" step. It's just a
    softmax: exponentiate, then divide by the total. Softmax ALWAYS outputs
    positive numbers that sum to 1, which is exactly the simplex constraint — so
    no matter how wild the noisy z is, the resulting w is a legal coalition.

        w = softmax(V z)

    Since the columns of V sum to 0, V z is zero-centred, and softmax maps it
    back onto the simplex cleanly.
    """
    x = V @ z
    # Subtract the max before exp(): exp() of a large number overflows to inf.
    # Shifting by a constant doesn't change the softmax result (it cancels in the
    # division) but keeps the exponentials in a safe numeric range. Standard trick.
    x -= x.max()
    w = np.exp(x)
    return w / w.sum()


# ── Covariance propagation ────────────────────────────────────────────────────


def _propagate_cov(w_star: np.ndarray, sigma_delta: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Map the race-level delta covariance to ILR space via the Jacobian.

    J = V^T diag(1/w*)
    Σ_ILR = J Σ_Δ J^T

    The Jacobian linearises the ILR transform at w*, converting uncertainty
    in Δw (Euclidean) to uncertainty in z (ILR space).

    INTUITION: our uncertainty (Σ_Δ) is expressed in the original weight space,
    but we add noise in the ILR space. A "Jacobian" J is the standard calculus
    tool for translating a small wiggle from one coordinate system to another;
    the formula Σ_ILR = J Σ_Δ Jᵀ is how a covariance (a spread) carries through
    that change of coordinates.
    """
    inv_w = 1.0 / w_star  # element-wise reciprocal of the weights
    J = V.T * inv_w[np.newaxis, :]  # J = V^T · diag(1/w*), built via broadcasting
    sigma_ilr = J @ sigma_delta @ J.T
    # Round-off can make the result very slightly non-symmetric; force exact
    # symmetry by averaging with its transpose (a covariance must be symmetric).
    return (sigma_ilr + sigma_ilr.T) / 2.0


def _make_psd(sigma: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Repair a nearly-PSD matrix by flooring its eigenvalues to a tiny positive eps.

    WHY THIS EXISTS: the next step draws samples using a Cholesky factorization,
    which only works on a "positive-definite" covariance matrix (all eigenvalues
    > 0 — intuitively, a valid notion of spread in every direction). Floating-
    point round-off can leave a covariance with a slightly-negative eigenvalue,
    which would make Cholesky fail. We decompose the matrix into its eigenvalues,
    clamp any that dipped at/below zero up to a tiny eps, and rebuild it. The
    result is the closest valid covariance, so sampling can proceed.
    """
    vals, vecs = np.linalg.eigh(sigma)  # eigenvalues + eigenvectors (symmetric)
    vals = np.maximum(vals, eps)  # floor any non-positive eigenvalue to eps
    return vecs @ np.diag(vals) @ vecs.T  # reassemble the repaired matrix


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
    cov_delta: list[list[float]] | None = None,
    fixed_loyalty: float | None = None,
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
        Diagonal standard deviation (in Δw units) used ONLY when `cov_delta` is
        not supplied.  Default 0.02 ≈ "slight" magnitude bin.
    cov_delta:
        The real 5×5 race-bloc covariance Σ_Δ (e.g. ShockResponseData.covariance,
        a Ledoit-Wolf estimate). When provided, the simulation propagates this
        actual cross-bloc correlation structure — the correct behavior. When None
        (the default), it falls back to the isotropic diagonal sigma_default²·I.
        Passing the shock's covariance here is how the headline win-probability CI
        comes to reflect genuine ("wave") correlation between blocs rather than an
        arbitrary placeholder.
    fixed_loyalty:
        The religion+gender contribution to μ_eff (see dqcp.compute_fixed_loyalty).
        MUST equal what the optimizer used, so the win condition here matches the
        optimization. When None, falls back to the neutral (1-λ₁)·0.5 placeholder.

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

    # ── Layer weights + religion/gender ("fixed") contribution ──────────────────
    lw = _load_layer_weights()
    lambda_1 = lw["lambda_1"]
    if fixed_loyalty is None:
        # DERIVE the religion+gender contribution from the equilibrium so the MC's
        # win condition uses EXACTLY the μ_eff basis the optimizer used. Since
        #     μ_eff_shifted = λ₁·Σ(w·μ̃_race) + fixed_loyalty,
        # we recover  fixed_loyalty = μ_eff_shifted − λ₁·Σ(w·μ̃_race).
        # This needs no extra plumbing (the equilibrium already carries everything)
        # and can't drift from the optimizer. Fall back to the neutral (1-λ₁)·0.5
        # placeholder when μ_eff_shifted is 0.0 (unset on old artifacts, or an
        # equal-weight fallback result) or the derivation goes negative.
        mu_eff_shifted = float(getattr(equilibrium, "mu_eff_shifted", 0.0) or 0.0)
        race_part = lambda_1 * float(w_star @ mu_race)
        derived = mu_eff_shifted - race_part
        if mu_eff_shifted > 1e-9 and derived >= 0.0:
            fixed_loyalty = derived
        else:
            fixed_loyalty = (1.0 - lambda_1) * 0.50

    # ── Covariance Σ_Δ ───────────────────────────────────────────────────────
    # Prefer the REAL covariance the caller passes (e.g. the shock's Ledoit-Wolf
    # estimate), which carries the cross-bloc correlation structure ("when Black
    # support moves, does Latino support move with it?"). Only when no covariance
    # is supplied do we fall back to an isotropic diagonal sigma_default²·I (every
    # bloc equal variance, zero correlation) — a conservative placeholder.
    #
    # ⚠️ Even with a real Σ_Δ, watch for DEGENERATE CIs: a Ledoit-Wolf estimate
    # from very few election cycles can be near-zero, collapsing the win-prob CI to
    # [1,1]/[0,0]. The Week-8 "production covariance path" task tracks ensuring the
    # supplied Σ_Δ is the genuine 14-cycle estimate (and adding a variance floor if
    # it proves too small) rather than itself being the diagonal fallback.
    if cov_delta is not None:
        sigma_delta = np.asarray(cov_delta, dtype=float)
        if sigma_delta.shape != (k, k):
            raise ValueError(
                f"run_ilr_montecarlo: cov_delta shape {sigma_delta.shape} != ({k}, {k})"
            )
        # Symmetrize + PSD-repair so the Cholesky draw below can't fail on a
        # slightly-non-PSD estimate (same safeguard _make_psd gives sigma_ilr).
        sigma_delta = _make_psd((sigma_delta + sigma_delta.T) / 2.0)
    else:
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
    # HOW WE DRAW CORRELATED SAMPLES: standard_normal() gives independent N(0,1)
    # noise. To turn that into noise with our desired covariance Σ_ILR, multiply
    # it by L, the Cholesky factor (the "matrix square root", Σ_ILR = L Lᵀ). Then
    # add the center z*. Result: each row is one draw y ~ N(z*, Σ_ILR). Doing all
    # N draws as one matrix multiply is far faster than a Python loop.
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
        mu_eff_samples = lambda_1 * mu_race_eff + fixed_loyalty
    win_flags = mu_eff_samples >= target
    win_probability = float(win_flags.mean())

    # Bootstrap 90% CI on the win-probability estimator (not on individual draws).
    # Resampling win_flags B times gives B bootstrap estimates of win_probability;
    # the 5th/95th percentiles of those means form a meaningful CI that narrows with N.
    # Uses the seeded rng already in scope — no fresh generator.
    _N_BOOT = 500
    n_valid = len(win_flags)
    # For Bernoulli data, bootstrapping the resampled mean is equivalent to
    # drawing counts from Binomial(n_valid, p_hat) where p_hat is the observed mean.
    boot_counts = rng.binomial(n_valid, win_probability, size=_N_BOOT)
    boot_means = boot_counts / n_valid
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


# ── CLI entry point (python -m electoral.simulation.montecarlo) ───────────────

if __name__ == "__main__":
    import argparse
    import sys

    from electoral.artifacts import EquilibriumData, StageArtifact
    from electoral.core.io import sanitize_floats
    from electoral.core.rng import derive_seed as _derive_seed

    parser = argparse.ArgumentParser(
        description="ILR Logistic-Normal Monte Carlo — win-probability estimation"
    )
    parser.add_argument(
        "--shock-artifact",
        required=True,
        metavar="PATH",
        help="Path to shock_response JSON artifact (loaded for metadata/logging)",
    )
    parser.add_argument(
        "--equilibrium-artifact",
        required=True,
        metavar="PATH",
        help="Path to equilibrium JSON artifact produced by solve_rebalanced",
    )
    parser.add_argument(
        "--output",
        required=True,
        metavar="PATH",
        help="Destination path for SimulationData JSON artifact",
    )
    parser.add_argument(
        "--n-simulations",
        type=int,
        default=50_000,
        metavar="N",
        help="Number of ILR Monte Carlo draws (default: 50 000)",
    )
    args = parser.parse_args()

    # Load equilibrium artifact (the only input run_ilr_montecarlo needs)
    eq_path = Path(args.equilibrium_artifact)
    if not eq_path.exists():
        print(f"ERROR: equilibrium artifact not found: {eq_path}", file=sys.stderr)
        sys.exit(1)
    eq_raw = json.loads(eq_path.read_text(encoding="utf-8"))
    equilibrium = EquilibriumData.from_dict(eq_raw.get("data", eq_raw))

    # Load shock artifact for metadata/log context AND its real Σ_Δ covariance.
    shock_path = Path(args.shock_artifact)
    shock_id = equilibrium.shock or shock_path.stem.removeprefix("shock_")
    cov_delta = None
    if shock_path.exists():
        shock_data = json.loads(shock_path.read_text(encoding="utf-8")).get("data", {})
        shock_id = shock_data.get("shock", shock_id)
        # Feed the shock's covariance to the MC so the CI reflects real bloc
        # correlation (falls back to the diagonal if the artifact has none).
        cov_delta = shock_data.get("covariance")

    # Minimal config: only derive_seed is consumed by run_ilr_montecarlo
    class _CliConfig:
        seed: int = 42

        def derive_seed(self, stage_name: str) -> int:
            return _derive_seed(self.seed, stage_name)

    config = _CliConfig()

    print(
        f"running {args.n_simulations:,} ILR Monte Carlo draws "
        f"for shock='{shock_id}' party={equilibrium.party}",
        flush=True,
    )
    result = run_ilr_montecarlo(
        equilibrium, config, n_simulations=args.n_simulations, cov_delta=cov_delta
    )
    result.validate()

    print(
        f"win_probability={result.win_probability:.4f} "
        f"90%CI=[{result.win_probability_low:.4f}, {result.win_probability_high:.4f}]",
        flush=True,
    )

    # Write SimulationData wrapped in a StageArtifact envelope
    envelope = StageArtifact(
        stage="simulation",
        run_key=f"slurm_{shock_id}",
        metadata={"n_simulations": args.n_simulations, "shock": shock_id},
        data=result.to_dict(),
    )
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # sanitize_floats: convert any inf/-inf/nan → null so the SLURM sim_{id}.json
    # is valid JSON (same guarantee write_json gives the other artifacts).
    out_path.write_text(json.dumps(sanitize_floats(envelope.to_dict()), indent=2), encoding="utf-8")
    print(f"wrote {out_path}", flush=True)
