"""Deterministic random number generation for the Electoral Equilibrium pipeline.

═══════════════════════════════════════════════════════════════════════════════
WHY THIS FILE EXISTS (read this first)
═══════════════════════════════════════════════════════════════════════════════
This whole project is a *research* pipeline. The single most important property
of a research result is that someone else (your supervisor, a reviewer, or you
six months from now) can re-run it and get the EXACT same numbers. That property
is called **reproducibility**.

The problem: almost every stage here uses randomness — the Monte Carlo
simulation draws thousands of random samples, the bootstrap resamples data, the
ML models shuffle batches. Randomness and reproducibility sound like opposites,
but they aren't. A computer can't make "true" randomness; it runs a
**pseudo-random number generator (PRNG)** — a deterministic formula that, given
a starting number called a **seed**, spits out a stream of numbers that *look*
random but are 100% determined by that seed. Same seed in → same stream out,
every time, on every machine.

So the rule for the entire codebase is: never use "ambient" randomness (where
the seed is secretly the current time or OS entropy). Instead, *every* random
operation receives a generator that was created from a known seed. That makes
the run replayable.

═══════════════════════════════════════════════════════════════════════════════
THE SEED CONTRACT (non-negotiable rules every module must follow)
═══════════════════════════════════════════════════════════════════════════════
  - make_rng(seed) → a seeded np.random.Generator (the object you draw from).
  - derive_seed_tokens(tokens) → hashes a list of strings into a stable seed.
  - derive_seed(base_seed, stage_name) → the everyday convenience wrapper;
        turns (global seed, stage name) into a per-stage sub-seed.
  - NEVER call np.random.* directly (e.g. np.random.normal()). Those use a
        hidden global generator whose state you can't control or replay.
  - NEVER call random.seed() globally. Global state is shared and order-
        dependent, which silently breaks reproducibility.
  - Every stochastic function takes a seeded generator as a parameter, so the
        caller — not the function — controls the randomness.

WHY PER-STAGE SUB-SEEDS? If every stage used the same single global seed, two
stages might accidentally draw the *same* "random" numbers (correlated noise),
and adding/removing one stage would shift the random stream for all the others.
Instead we derive a *separate* seed for each stage from the global seed plus the
stage's name. The stages are then independent of each other, yet the whole run
is still fully determined by the one global seed in configs/base.json.
"""

from __future__ import annotations

import hashlib  # SHA-256: turns arbitrary text into a fixed-size "fingerprint"
import struct  # reads raw bytes as a number (used to convert the hash → int)

import numpy as np


def derive_seed_tokens(tokens: list[str]) -> int:
    """Hash a list of string tokens into a deterministic, reproducible integer seed.

    The idea: we want to map a *human-readable context* (like the global seed
    plus a stage name) onto a *number* a PRNG can use — and we want that mapping
    to be stable forever (same input → same output on any computer, any Python
    version). A cryptographic hash function (SHA-256) is the perfect tool: it
    deterministically crunches any text into a fixed 256-bit fingerprint, and
    tiny changes in the input produce completely different fingerprints.

    The token list is joined with ":" as a separator before hashing, so:
      - Order matters:        ["a", "b"] != ["b", "a"]
      - All tokens contribute: ["42", "x"] != ["42", "y"]
      - Empty list is valid but produces a constant seed.
    The ":" separator is why tokens themselves may not contain ":" — otherwise
    ["a:b"] and ["a", "b"] would collide (hash to the same seed).

    Args:
        tokens: Ordered list of string tokens that uniquely identify this seed
                context. Typical usage: [str(config.seed), stage_name].

    Returns:
        Deterministic non-negative integer in [0, 2**31).
    """
    # Guard the separator's uniqueness: if a token contained ":", the joined
    # string would be ambiguous and two different token lists could collide.
    if any(":" in token for token in tokens):
        raise ValueError("derive_seed_tokens tokens must not contain ':'")

    # Join into one string and encode to raw bytes (hashing works on bytes).
    joined = ":".join(tokens).encode("utf-8")

    # SHA-256 → a 32-byte (256-bit) digest. Deterministic and identical on every
    # platform, which is exactly what reproducibility needs.
    h = hashlib.sha256(joined).digest()

    # We only need a single integer seed, not all 256 bits. Take the first 8
    # bytes and interpret them as one unsigned 64-bit integer.
    #   "<Q"  →  "<" little-endian byte order, "Q" = unsigned 64-bit int.
    # Fixing the byte order ("<") is important: it makes the result identical on
    # both little- and big-endian CPUs, so the seed never depends on hardware.
    raw = struct.unpack("<Q", h[:8])[0]

    # Squeeze the big 64-bit number into [0, 2**31). 2**31 is a safe, generous
    # range that every NumPy version accepts as a seed, and staying non-negative
    # avoids any signed-integer surprises.
    return int(raw % (2**31))


def derive_seed(base_seed: int, stage_name: str) -> int:
    """Derive a deterministic per-stage sub-seed from a global seed and stage name.

    This is the function you will call 99% of the time. It is just a friendly
    wrapper: the output is exactly derive_seed_tokens([str(base_seed), stage_name]).

    Example: with a global seed of 42, the Monte Carlo stage always gets the
    same sub-seed `derive_seed(42, "monte_carlo")`, while the bootstrap stage
    gets a *different but equally fixed* sub-seed `derive_seed(42, "bootstrap")`.
    Independent streams, one reproducible run.

    Args:
        base_seed:  Global pipeline seed from PipelineConfig.seed (configs/base.json).
        stage_name: Unique lowercase snake_case identifier for the stage
                    (e.g. "voter_panel", "monte_carlo", "setfit").

    Returns:
        Deterministic non-negative integer in [0, 2**31).
    """
    # str(base_seed) so the integer global seed becomes a token alongside the
    # stage name; both feed the hash so each (seed, stage) pair maps to its own
    # sub-seed.
    return derive_seed_tokens([str(base_seed), stage_name])


def make_rng(seed: int) -> np.random.Generator:
    """Create a seeded NumPy random number generator (the object you draw from).

    np.random.Generator is NumPy's modern randomness API. Under the hood it uses
    the PCG64 algorithm — a high-quality PRNG. The key point for us: a Generator
    built from a given seed will always produce the identical sequence of draws
    (.normal(), .integers(), .choice(), etc.). Pass that Generator into any
    stochastic function instead of touching np.random.* directly.

    Args:
        seed: Integer seed. Always produce this via derive_seed(config.seed,
              stage_name) — never hand-write a raw literal, or you break the
              per-stage independence described at the top of this file.

    Returns:
        A seeded np.random.Generator instance.
    """
    # default_rng(seed) is the recommended constructor; it wraps the seed in a
    # SeedSequence and initializes a PCG64 bit generator for us.
    return np.random.default_rng(seed)
