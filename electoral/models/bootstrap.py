"""Covariance estimators for the historical delta matrix.

Primary: ledoit_wolf_cov — Ledoit-Wolf shrinkage, designed for the p > n regime
(6-8 election cycles, up to 10 blocs). Always returns a full-rank PSD matrix
without requiring psd_repair.

Alternative: bootstrap_cov_matrix — raw bootstrap covariance, kept for comparison.
Rank-deficient when n_cycles < n_blocs; not suitable as the optimizer Sigma_Delta.

See DECISIONS.md §[Ledoit-Wolf] for rationale.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from electoral.core.rng import make_rng
from electoral.core.types import CANONICAL_RACES

logger = logging.getLogger(__name__)


def ledoit_wolf_cov(delta_matrix: np.ndarray) -> np.ndarray:
    """Estimate a well-conditioned covariance via Ledoit-Wolf shrinkage.

    Designed for the p > n regime (few election cycles, many blocs).
    Applies analytically optimal shrinkage toward a diagonal target,
    producing a full-rank PSD estimate without psd_repair.

    Args:
        delta_matrix: shape (n_cycles, n_blocs) — historical per-cycle
            deltas. For the optimizer Sigma_Delta this is (n_cycles, 5)
            race-only per the 5x5 covariance rule in DECISIONS.md.

    Returns:
        (n_blocs, n_blocs) covariance, always full-rank and PSD.
    """
    if delta_matrix.ndim != 2:
        raise ValueError(f"delta_matrix must be 2D, got shape {delta_matrix.shape}")
    if delta_matrix.shape[0] < 2:
        raise ValueError(f"Need >= 2 cycles, got {delta_matrix.shape[0]}")

    lw = LedoitWolf(assume_centered=False)
    lw.fit(delta_matrix)
    return lw.covariance_


def bootstrap_cov_matrix(
    delta_matrix: np.ndarray,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> np.ndarray:
    """Raw bootstrap covariance — kept for comparison only.

    Rank-deficient when n_cycles < n_blocs. Ledoit-Wolf is preferred
    for all production use; this function exists to quantify the
    shrinkage benefit in the paper's covariance comparison table.
    """
    rng = make_rng(seed)
    n_cycles, _ = delta_matrix.shape
    boot_covs = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n_cycles, size=n_cycles)
        boot_covs.append(np.cov(delta_matrix[idx].T))
    return np.mean(boot_covs, axis=0)


def bootstrap_cov_weighted(
    panel_df: pd.DataFrame,
    social_elasticities: dict[str, float],
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> np.ndarray:
    """Bootstrap covariance weighted by social media elasticity per cycle.

    Cycles with higher mean absolute elasticity get a larger resampling
    probability, so the estimated covariance reflects high-signal electoral
    shocks more than low-signal ones.

    Falls back to np.eye(n_blocs) * 1e-4 when no valid bootstrap samples
    can be produced (e.g., fewer than 2 cycles in the panel).

    Args:
        panel_df: DataFrame with columns 'cycle', 'bloc', 'delta'.
            Must contain only race blocs (CANONICAL_RACES); non-race blocs
            raise ValueError. Output dimension equals the number of distinct
            blocs present in the DataFrame — callers must filter to all five
            CANONICAL_RACES before calling to guarantee a 5×5 result.
            Use _build_sigma_delta (shock.py) for the production 5×5 Σ_Δ.
        social_elasticities: Maps cycle → mean absolute elasticity from
            social media regression. Cycles absent from the dict receive
            weight 0 and are excluded from resampling.
        n_bootstrap: Number of bootstrap replicates.
        seed: RNG seed (passed to make_rng for reproducibility contract).

    Returns:
        (n_blocs, n_blocs) covariance matrix. May be rank-deficient when
        high weights concentrate resampling on a small number of cycles.
        Use ledoit_wolf_cov for the optimizer Sigma_Delta.
    """
    rng = make_rng(seed)

    cycles = sorted(panel_df["cycle"].unique())
    blocs = sorted(panel_df["bloc"].unique())
    n_cycles = len(cycles)
    n_blocs = len(blocs)

    unexpected = set(blocs) - set(CANONICAL_RACES)
    if unexpected:
        raise ValueError(
            f"bootstrap_cov_weighted: panel_df contains non-race blocs "
            f"{sorted(unexpected)}. Pass race-only rows (filter to "
            "CANONICAL_RACES before calling). See DECISIONS.md §[Covariance] "
            "for the 5×5 Σ_Δ constraint."
        )

    # (n_cycles, n_blocs) delta matrix — one averaged row per cycle
    pivot = panel_df.pivot_table(index="cycle", columns="bloc", values="delta", aggfunc="mean")
    pivot = pivot.reindex(index=cycles, columns=blocs, fill_value=0.0)
    delta_matrix = pivot.to_numpy(dtype=float)

    # Sampling probability proportional to mean absolute social elasticity
    raw_weights = np.array([abs(float(social_elasticities.get(c, 0.0))) for c in cycles])
    total = raw_weights.sum()
    probs = raw_weights / total if total > 0 else np.ones(n_cycles) / n_cycles

    boot_covs = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n_cycles, size=n_cycles, replace=True, p=probs)
        sample = delta_matrix[idx]
        if sample.shape[0] >= 2:
            boot_covs.append(np.cov(sample.T))

    if not boot_covs:
        logger.warning(
            "bootstrap_cov_weighted: no valid bootstrap samples — returning identity fallback"
        )
        return np.eye(n_blocs) * 1e-4

    return np.mean(boot_covs, axis=0)
