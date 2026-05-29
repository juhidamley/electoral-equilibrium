"""Deterministic random number generation for the Electoral Equilibrium pipeline.

Seed contract (non-negotiable):
  - make_rng(seed) returns a seeded np.random.Generator
  - derive_seed_tokens(tokens) hashes a list of strings into a reproducible seed
  - derive_seed(base_seed, stage_name) is the canonical per-stage convenience wrapper
  - Never call np.random directly anywhere in the pipeline
  - Never call random.seed() globally
  - Every stochastic operation takes a seeded generator as a parameter
  - Dirichlet sampling: rng.dirichlet(alpha, size=N) from seeded generator only
"""

from __future__ import annotations

import hashlib
import struct

import numpy as np


def derive_seed_tokens(tokens: list[str]) -> int:
    """Hash a list of string tokens into a deterministic, reproducible integer seed.

    The token list is joined with ":" as a separator before hashing, so:
      - Order matters:  ["a", "b"] != ["b", "a"]
      - All tokens contribute: ["42", "x"] != ["42", "y"]
      - Empty list is valid but produces a constant seed

    Args:
        tokens: Ordered list of string tokens that uniquely identify this seed
                context. Typical usage: [str(config.seed), stage_name].

    Returns:
        Deterministic non-negative integer in [0, 2**31).
    """
    joined = ":".join(tokens).encode("utf-8")
    h = hashlib.sha256(joined).digest()
    # First 8 bytes as little-endian uint64, modded into NumPy's accepted range.
    raw = struct.unpack("<Q", h[:8])[0]
    return int(raw % (2**31))


def derive_seed(base_seed: int, stage_name: str) -> int:
    """Derive a deterministic per-stage sub-seed from a global seed and stage name.

    Canonical convenience wrapper around derive_seed_tokens. The output is
    identical to derive_seed_tokens([str(base_seed), stage_name]).

    Args:
        base_seed:  Global pipeline seed from PipelineConfig.seed.
        stage_name: Unique lowercase snake_case identifier for the stage
                    (e.g. "voter_panel", "monte_carlo", "setfit").

    Returns:
        Deterministic non-negative integer in [0, 2**31).
    """
    return derive_seed_tokens([str(base_seed), stage_name])


def make_rng(seed: int) -> np.random.Generator:
    """Create a seeded NumPy random number generator (PCG64).

    Args:
        seed: Integer seed. Always use derive_seed(config.seed, stage_name)
              to produce this value — never pass a raw literal.

    Returns:
        Seeded np.random.Generator instance.
    """
    return np.random.default_rng(seed)
