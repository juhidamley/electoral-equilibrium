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
