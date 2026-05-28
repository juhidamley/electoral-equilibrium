"""Deterministic random number generation for the Electoral Equilibrium pipeline.

Seed contract (non-negotiable):
  - make_rng(seed) returns a seeded np.random.Generator
  - derive_seed(base_seed, stage_name) returns a deterministic per-stage sub-seed
  - Never call np.random directly anywhere in the pipeline
  - Never call random.seed() globally
  - Every stochastic operation takes a seeded generator as a parameter
  - Dirichlet sampling: rng.dirichlet(alpha, size=N) from seeded generator only
"""
from __future__ import annotations

import hashlib
import struct

import numpy as np


def derive_seed(base_seed: int, stage_name: str) -> int:
    """Derive a deterministic per-stage sub-seed.

    Args:
        base_seed: Global pipeline seed from PipelineConfig.seed
        stage_name: Unique identifier for the pipeline stage or component
                    (e.g. "voter_panel", "monte_carlo", "setfit")

    Returns:
        Deterministic non-negative integer seed in [0, 2**31)
    """
    h = hashlib.sha256(f"{base_seed}:{stage_name}".encode("utf-8")).digest()
    # Take first 8 bytes as little-endian uint64, then mod into numpy's range
    raw = struct.unpack("<Q", h[:8])[0]
    return int(raw % (2**31))


def make_rng(seed: int) -> np.random.Generator:
    """Create a seeded NumPy random number generator.

    Args:
        seed: Integer seed. Use derive_seed(config.seed, stage_name) for
              per-stage generators.

    Returns:
        Seeded np.random.Generator instance (PCG64 algorithm)
    """
    return np.random.default_rng(seed)