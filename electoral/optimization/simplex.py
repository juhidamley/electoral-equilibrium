"""Euclidean projection onto the probability simplex.

WHAT "PROJECT ONTO THE SIMPLEX" MEANS: the "probability simplex" is the set of
vectors whose entries are all ≥ 0 and sum to exactly 1 — i.e. valid coalition
weights. "Projecting" a vector v onto it means finding the *closest* such valid
vector to v (closest in ordinary straight-line / Euclidean distance).

WHEN IT'S USED HERE: it's a lightweight repair tool. If some step produces a
vector that is almost-but-not-quite a valid set of weights (e.g. a fallback path,
or numerical drift), project_simplex snaps it back to the nearest legal weights.
It is NOT the main Monte Carlo path (that uses the ILR/softmax transform in
simulation/montecarlo.py, which serves a different purpose).

The algorithm is from Duchi et al. (2008), "Efficient Projections onto the
ℓ1-Ball for Learning in High Dimensions" — an O(n log n) closed-form method
(no iterative solver needed). The math is just: subtract the right constant
`theta` from every entry, then clamp negatives to 0; the trick is computing the
correct `theta` so the survivors sum to exactly 1.
"""

from __future__ import annotations

import numpy as np


def project_simplex(v: np.ndarray) -> np.ndarray:
    """Project a single vector v onto the probability simplex (≥0, sums to 1).

    Returns the closest vector to v whose entries are non-negative and sum to 1.

    Algorithm (Duchi et al. 2008, O(n log n)):
    1. Sort v descending.
    2. Find the largest prefix length rho where the entries stay positive after
       the shift (the rho line below).
    3. Compute the shift theta from that prefix.
    4. Return max(v - theta, 0): subtract theta everywhere, clamp negatives to 0.
    """
    n = len(v)

    # Step 1: sort entries from largest to smallest. The biggest entries are the
    # ones that will survive (stay positive); the smallest get clamped to 0.
    u = np.sort(v)[::-1]

    # cssv[k] = sum of the top (k+1) entries (cumulative sum of the sorted list).
    cssv = np.cumsum(u)

    # Step 2: find rho = how many of the largest entries remain positive after
    # the shift. np.arange(1, n+1) is [1,2,...,n] (the prefix lengths). The
    # condition u_k * k > (cssv_k - 1) marks which prefixes are still "active";
    # we take the last (largest) such index. Index 0 always satisfies it, so
    # nonzero(...) is never empty — [0][-1] is safe.
    rho = np.nonzero(u * np.arange(1, n + 1) > (cssv - 1))[0][-1]

    # Step 3: theta is the single constant we subtract from every entry so the
    # positive survivors end up summing to exactly 1. (rho is 0-indexed, so the
    # prefix length is rho+1.)
    theta = (cssv[rho] - 1.0) / (rho + 1.0)

    # Step 4: shift down by theta and clamp negatives to 0 → valid weights.
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
