"""Kernel: LLM shock-response estimation + covariance bootstrap + optimizer.

build_shock_response() is the entry point called from stages.py.
It runs three sub-steps in sequence:
  1. estimate_shock     — ShockEstimator.estimate() via constrained decoding
  2. bootstrap_cov      — Gaussian-noise bootstrap of the 5×5 race covariance
  3. solve_rebalanced   — CVXPY DQCP optimizer for new coalition weights
"""

from __future__ import annotations

import dataclasses
import logging

import numpy as np

from electoral.artifacts import EquilibriumData, ShockResponseData
from electoral.config import PipelineConfig
from electoral.core.io import write_artifact
from electoral.core.types import CANONICAL_RACES

log = logging.getLogger(__name__)

_DEFAULT_ADAPTER = "models/mistral-r16"
_DEFAULT_BASE_MODEL = "mistralai/Mistral-7B-v0.3"
_N_BOOTSTRAP = 100
_NOISE_STD = 0.01


# ── Public entry point ────────────────────────────────────────────────────────


def build_shock_response(
    config: PipelineConfig,
    event: str,
    intensity: float,
) -> tuple[ShockResponseData, EquilibriumData]:
    """Run the full shock pipeline and return both typed artifacts.

    Steps:
      1. estimate_shock  — LLM constrained decoding → ShockResponseData
      2. bootstrap_cov   — Gaussian-noise bootstrap → 5×5 race covariance
      3. solve_rebalanced — CVXPY DQCP → rebalanced coalition weights
      4. Write shock_response.json and equilibrium.json to config.output_dir
    """
    shock = _estimate_shock(config, event, intensity)
    shock = _with_bootstrapped_cov(shock)
    equilibrium = _solve_rebalanced(shock, config)

    _write_artifacts(config, shock, equilibrium)
    return shock, equilibrium


# ── Step 1: LLM estimation ────────────────────────────────────────────────────


def _estimate_shock(
    config: PipelineConfig,
    event: str,
    intensity: float,
) -> ShockResponseData:
    from electoral.llm.inference import ShockEstimator

    adapter_path: str = getattr(config, "adapter_path", None) or _DEFAULT_ADAPTER
    base_model: str = getattr(config, "base_model", None) or _DEFAULT_BASE_MODEL

    estimator = ShockEstimator(adapter_path=adapter_path, base_model=base_model)

    event_dict = {
        "description": event,
        "party": config.party,
        "shock_id": event[:30].lower().replace(" ", "_"),
        "year": 2024,
        "news_roberta_scores": {},
        "social_roberta_scores": {},
    }

    log.info("estimating shock for: %s (intensity=%.2f)", event[:60], intensity)
    return estimator.estimate(event_dict, intensity=intensity)


# ── Step 2: Covariance bootstrap ──────────────────────────────────────────────


def bootstrap_cov(
    shock_response: ShockResponseData,
    n_bootstrap: int = _N_BOOTSTRAP,
    noise_std: float = _NOISE_STD,
    seed: int = 42,
) -> list[list[float]]:
    """Bootstrap a 5×5 race covariance matrix from the race deltas.

    Adds Gaussian noise (std=noise_std) to each race delta n_bootstrap times
    and computes the sample covariance of the resulting (n_bootstrap, 5) matrix.
    """
    deltas = np.array(
        [shock_response.deltas_race[r] for r in CANONICAL_RACES],
        dtype=float,
    )
    from electoral.core.rng import make_rng
    rng = make_rng(seed)
    samples = deltas[None, :] + rng.normal(0.0, noise_std, size=(n_bootstrap, len(deltas)))
    cov: np.ndarray = np.cov(samples.T)
    return cov.tolist()


def _with_bootstrapped_cov(shock: ShockResponseData) -> ShockResponseData:
    """Return a new ShockResponseData with the bootstrapped covariance."""
    cov = bootstrap_cov(shock)
    return dataclasses.replace(shock, covariance=cov)


# ── Step 3: CVXPY optimizer ───────────────────────────────────────────────────


def _solve_rebalanced(
    shock: ShockResponseData,
    config: PipelineConfig,
) -> EquilibriumData:
    """Rebalance coalition weights to maximise P(win) after the shock.

    Uses the min-variance CVXPY DQCP formulation (solve_baseline).  If the
    solver fails (infeasible target, numerical issues), falls back to equal
    weights and marks feasible=False.
    """
    from electoral.portfolios.cvx import solve_baseline

    # Post-shock within-bloc vote shares: neutral baseline 0.5 + Δ
    mu_shifted = {
        r: float(np.clip(0.5 + shock.deltas_race[r], 0.01, 0.99)) for r in CANONICAL_RACES
    }
    cov_arr = np.array(shock.covariance, dtype=float)

    try:
        weights = solve_baseline(mu_shifted, cov_arr, config.target)
        mu_eff = sum(weights[r] * mu_shifted[r] for r in CANONICAL_RACES)
        feasible = True
        target_met = mu_eff >= config.target
        method = "cvxpy_dqcp"
        log.info(
            "solver: feasible=True  mu_eff=%.4f  target=%.4f  target_met=%s",
            mu_eff,
            config.target,
            target_met,
        )
    except Exception as exc:
        log.warning("solver failed (%s) — falling back to equal weights", exc)
        n = len(CANONICAL_RACES)
        weights = {r: 1.0 / n for r in CANONICAL_RACES}
        mu_shifted = {r: max(0.01, min(0.99, 0.5 + shock.deltas_race[r])) for r in CANONICAL_RACES}
        feasible = False
        target_met = False
        method = "equal_weight_fallback"

    equilibrium = EquilibriumData(
        method=method,
        party=config.party,
        shock=shock.shock,
        weights=weights,
        mu_shifted=mu_shifted,
        feasible=feasible,
        target_met=target_met,
        target=config.target,
    )
    equilibrium.validate()
    return equilibrium


# ── Artifact I/O ──────────────────────────────────────────────────────────────


def _write_artifacts(
    config: PipelineConfig,
    shock: ShockResponseData,
    equilibrium: EquilibriumData,
) -> None:
    from electoral.artifacts import StageArtifact

    shock_envelope = StageArtifact(
        stage="shock_response",
        run_key=config.run_key,
        metadata={"shock": shock.shock, "delta_eff": shock.delta_eff},
        data=shock.to_dict(),
    )
    write_artifact(f"{config.output_dir}/shock_response.json", shock_envelope.to_dict())

    eq_envelope = StageArtifact(
        stage="equilibrium",
        run_key=config.run_key,
        metadata={"feasible": equilibrium.feasible, "target_met": equilibrium.target_met},
        data=equilibrium.to_dict(),
    )
    write_artifact(f"{config.output_dir}/equilibrium.json", eq_envelope.to_dict())
    log.info("wrote shock_response.json and equilibrium.json to %s", config.output_dir)
