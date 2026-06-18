"""Kernel: LLM shock-response estimation + covariance bootstrap + optimizer.

build_shock_response() is the entry point called from stages.py.
Six-step chain:
  1. _estimate_shock    — ShockEstimator.estimate() via constrained decoding
  2. _load_baseline     — BaselinePortfolioData → mu_race, w0
  3. mu_tilde           — mu_baseline + deltas_race, clipped to [0.01, 0.99]
  4. _build_sigma_delta — Ledoit-Wolf Σ_Δ on historical panel first-differences
  5. solve_rebalanced   — CVXPY DQCP optimizer → EquilibriumData
  6. _write_artifacts   — persist shock_{id}.json and equilibrium_{id}.json
"""

from __future__ import annotations

import dataclasses
import json
import logging
from pathlib import Path

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
      1. _estimate_shock       — LLM constrained decoding → ShockResponseData
      2. _load_baseline        — load mu_race and w0 from baseline_portfolio.json
      3. mu_tilde              — mu_baseline + deltas_race, clipped to [0.01, 0.99]
      4. _build_sigma_delta     — Ledoit-Wolf Σ_Δ on panel first-differences + validate
      5. solve_rebalanced      — CVXPY DQCP → EquilibriumData + validate
      6. _write_artifacts      — persist shock_{id}.json and equilibrium_{id}.json
    """
    from electoral.optimization.cvx import solve_rebalanced

    # Step 1
    shock = _estimate_shock(config, event, intensity)

    # Step 2
    mu_baseline, _w0 = _load_baseline(config)
    # _w0 reserved for warm-start initialization in a future optimizer upgrade

    # Step 3
    mu_tilde = {
        r: float(np.clip(mu_baseline[r] + shock.deltas_race[r], 0.01, 0.99))
        for r in CANONICAL_RACES
    }

    # Step 4 — Ledoit-Wolf Σ_Δ from historical panel first-differences
    cov_list = _build_sigma_delta(config)
    shock = dataclasses.replace(shock, covariance=cov_list)
    shock.validate()

    # Step 5
    equilibrium = solve_rebalanced(
        mu_tilde=mu_tilde,
        cov_delta=shock.covariance,
        target=config.target,
        party=config.party,
        shock=shock.shock,
    )
    equilibrium.validate()

    # Step 6
    _write_artifacts(config, shock, equilibrium)
    return shock, equilibrium


# ── Step 2: Baseline loader ───────────────────────────────────────────────────


def _load_baseline(
    config: PipelineConfig,
) -> tuple[dict[str, float], dict[str, float]]:
    """Load mu_race and w0 from baseline_portfolio.json.

    Returns (mu, w0) dicts keyed by CANONICAL_RACES.
    Falls back to mu=0.5 and equal weights with a warning on missing/broken file.
    """
    n = len(CANONICAL_RACES)
    _fallback_mu: dict[str, float] = {r: 0.5 for r in CANONICAL_RACES}
    _fallback_w0: dict[str, float] = {r: 1.0 / n for r in CANONICAL_RACES}

    path = Path(config.output_dir) / "baseline_portfolio.json"
    if not path.exists():
        log.warning("_load_baseline: %s not found — using mu=0.5 and equal weights", path)
        return _fallback_mu, _fallback_w0

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        data = raw["data"]
        mu = {r: float(data["mu_race"][r]) for r in CANONICAL_RACES}
        w0 = {r: float(data["weights"][r]) for r in CANONICAL_RACES}
        return mu, w0
    except (KeyError, ValueError, json.JSONDecodeError) as exc:
        log.warning(
            "_load_baseline: failed to parse %s (%s) — using mu=0.5 and equal weights",
            path,
            exc,
        )
        return _fallback_mu, _fallback_w0


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


# ── Step 4: Sigma_Delta estimator ────────────────────────────────────────────


def _build_sigma_delta(config: PipelineConfig) -> list[list[float]]:
    """Build 5×5 race Σ_Δ via Ledoit-Wolf on historical per-cycle vote-share deltas.

    Loads panel_race.parquet (written by build_voter_panel), pivots to
    (n_cycles, 5) in CANONICAL_RACES order, computes first-differences, and
    applies Ledoit-Wolf shrinkage. Column order matches list(CANONICAL_RACES)
    so rows/cols align with the blocs order expected by solve_rebalanced.

    Falls back to σ=0.02·I (≈ "slight" magnitude bin) with a warning when:
    - panel_race.parquet doesn't exist (pipeline hasn't reached voter_panel stage)
    - fewer than 3 cycles available (need ≥2 deltas for Ledoit-Wolf)
    - any other parsing/IO error
    """
    import pandas as pd

    from electoral.models.bootstrap import ledoit_wolf_cov

    panel_path = Path(config.output_dir) / "panel" / "panel_race.parquet"
    blocs = list(CANONICAL_RACES)
    n = len(blocs)
    _fallback: list[list[float]] = (np.eye(n) * 0.02**2).tolist()

    if not panel_path.exists():
        log.warning(
            "_build_sigma_delta: %s not found — using diagonal fallback σ=0.02",
            panel_path,
        )
        return _fallback

    try:
        df = pd.read_parquet(panel_path)
        pivot = (
            df[df["bloc"].isin(CANONICAL_RACES)]
            .pivot_table(index="cycle", columns="bloc", values="vote_share", aggfunc="mean")
            .reindex(columns=blocs, fill_value=np.nan)
            .sort_index()
            .dropna()
        )

        if len(pivot) < 3:
            log.warning(
                "_build_sigma_delta: only %d complete cycle(s) in panel — "
                "need ≥3 for first-differences; using diagonal fallback",
                len(pivot),
            )
            return _fallback

        delta_matrix = np.diff(pivot.to_numpy(dtype=float), axis=0)  # (n_cycles-1, 5)
        cov = ledoit_wolf_cov(delta_matrix)
        log.info(
            "_build_sigma_delta: Ledoit-Wolf Σ_Δ from %d cycle deltas",
            len(pivot) - 1,
        )
        return cov.tolist()

    except Exception as exc:
        log.warning("_build_sigma_delta: failed (%s) — using diagonal fallback σ=0.02", exc)
        return _fallback


# ── Diagnostic covariance bootstrap (not in production chain) ─────────────────


def bootstrap_cov(
    shock_response: ShockResponseData,
    n_bootstrap: int = _N_BOOTSTRAP,
    noise_std: float = _NOISE_STD,
    seed: int = 42,
) -> list[list[float]]:
    """Diagnostic bootstrap covariance from a single delta vector.

    Adds i.i.d. Gaussian noise (std=noise_std) to a single delta vector
    n_bootstrap times and computes the sample covariance of the resulting
    (n_bootstrap, 5) matrix. The result is approximately noise_std²·I —
    diagonal, isotropic, and independent of any inter-bloc correlation.

    Not used in the production chain. Step 4 uses _build_sigma_delta →
    ledoit_wolf_cov on real historical deltas. Retained as a diagnostic
    tool and for tests that need a cheap synthetic covariance without panel
    parquets on disk.
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


# ── Artifact I/O ──────────────────────────────────────────────────────────────


def _write_artifacts(
    config: PipelineConfig,
    shock: ShockResponseData,
    equilibrium: EquilibriumData,
) -> None:
    from electoral.artifacts import StageArtifact

    shock_id = shock.shock or "unknown"

    shock_envelope = StageArtifact(
        stage="shock_response",
        run_key=config.run_key,
        metadata={"shock": shock_id, "delta_eff": shock.delta_eff},
        data=shock.to_dict(),
    )
    write_artifact(f"{config.output_dir}/shock_{shock_id}.json", shock_envelope.to_dict())

    eq_envelope = StageArtifact(
        stage="equilibrium",
        run_key=config.run_key,
        metadata={"feasible": equilibrium.feasible, "target_met": equilibrium.target_met},
        data=equilibrium.to_dict(),
    )
    write_artifact(f"{config.output_dir}/equilibrium_{shock_id}.json", eq_envelope.to_dict())
    log.info(
        "wrote shock_%s.json and equilibrium_%s.json to %s",
        shock_id,
        shock_id,
        config.output_dir,
    )
