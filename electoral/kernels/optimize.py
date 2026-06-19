"""Kernel: CVXPY DQCP optimizer for coalition rebalancing.

build_optimization() is the entry point called from stages.py.

Steps:
  1. Load mu_race from baseline_portfolio.json (fallback: 0.5 per race)
  2. Reconstruct mu_tilde = clip(mu_baseline + deltas_race, 0.01, 0.99)
  3. solve_rebalanced(mu_tilde, shock.covariance, target, party) → EquilibriumData
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from electoral.artifacts import EquilibriumData, ShockResponseData
from electoral.config import PipelineConfig
from electoral.core.types import CANONICAL_RACES
from electoral.optimization.cvx import solve_rebalanced

log = logging.getLogger(__name__)


def build_optimization(
    config: PipelineConfig,
    shock: ShockResponseData,
) -> EquilibriumData:
    """Load baseline mu_race, reconstruct mu_tilde, run CVXPY optimizer.

    Returns EquilibriumData with method='cvxpy_dqcp' (never 'placeholder').
    """
    mu_baseline = _load_mu_baseline(config)
    mu_tilde = {
        r: float(np.clip(mu_baseline[r] + shock.deltas_race[r], 0.01, 0.99))
        for r in CANONICAL_RACES
    }

    equilibrium = solve_rebalanced(
        mu_tilde=mu_tilde,
        cov_delta=shock.covariance,
        target=config.target,
        party=config.party,
        shock=shock.shock,
    )
    equilibrium.validate()
    log.info(
        "build_optimization: feasible=%s target_met=%s method=%s",
        equilibrium.feasible,
        equilibrium.target_met,
        equilibrium.method,
    )
    return equilibrium


def _load_mu_baseline(config: PipelineConfig) -> dict[str, float]:
    """Load mu_race from baseline_portfolio.json. Falls back to 0.5 with a warning."""
    _fallback: dict[str, float] = {r: 0.5 for r in CANONICAL_RACES}
    path = Path(config.output_dir) / "baseline_portfolio.json"
    if not path.exists():
        log.warning("_load_mu_baseline: %s not found — using mu=0.5 fallback", path)
        return _fallback
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return {r: float(raw["data"]["mu_race"][r]) for r in CANONICAL_RACES}
    except (KeyError, ValueError, json.JSONDecodeError) as exc:
        log.warning("_load_mu_baseline: parse error (%s) — using mu=0.5 fallback", exc)
        return _fallback
