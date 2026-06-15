from __future__ import annotations

import numpy as np


def project_simplex(v: np.ndarray) -> np.ndarray:
    """Project vector v onto the probability simplex using
    Duchi et al. (2008) O(n log n) algorithm.

    Algorithm:
    1. Sort v descending
    2. Find largest rho such that v_(rho) - (1/rho)(sum_{j<=rho} v_(j) - 1) > 0
    3. Set theta = (1/rho)(sum_{j<=rho} v_(j) - 1)
    4. Return max(v - theta, 0)
    """
    n = len(v)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, n + 1) > (cssv - 1))[0][-1]
    theta = (cssv[rho] - 1.0) / (rho + 1.0)
    return np.maximum(v - theta, 0.0)


def project_simplex_batch(W: np.ndarray) -> np.ndarray:
    """Project each row of W onto the probability simplex.

    Args:
        W: (N, d) matrix of unconstrained vectors.

    Returns:
        (N, d) matrix where each row sums to 1 and all entries are >= 0.

    Uses Duchi et al. (2008) O(d log d) per row. A Python loop over rows
    is acceptable: at N=10k, d=10 the sort dominates and per-row overhead
    is negligible. This is a fallback utility for when the CVXPY optimizer
    returns infeasible — not the primary Monte Carlo path.
    """
    if W.ndim != 2:
        raise ValueError(f"project_simplex_batch requires 2D array, got shape {W.shape}")

    N, d = W.shape
    out = np.empty_like(W, dtype=float)
    for i in range(N):
        out[i] = project_simplex(W[i])
    return out
