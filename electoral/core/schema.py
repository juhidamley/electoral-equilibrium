"""Schema validation helpers for Electoral Equilibrium pipeline artifacts.

These are the small reusable "assert_*" checks that every artifact's validate()
method (in artifacts.py) builds on. Centralizing them here means two things:
  1. DON'T REPEAT YOURSELF — the "shares must sum to 1" rule is written once, not
     re-implemented in ten classes (where the versions could subtly disagree).
  2. CONSISTENT ERRORS — every message names its `context` (which class + field),
     so when validation fails you immediately know WHERE to look.

Each helper RAISES ValueError on a violation and returns None on success — the
calling validate() chains several of them, and the first failure stops the show.
All helpers are pure functions: no file I/O, no global state, no surprises.
"""

from __future__ import annotations

from typing import Any


def assert_required_keys(
    payload: dict[str, Any],
    keys: list[str],
    *,
    context: str = "payload",
) -> None:
    """Raise ValueError if any required key is missing from `payload`.

    Used to confirm a dict (e.g. layer_weights) contains every key it must have
    before code tries to read those keys and trips over a KeyError later.
    """
    # Collect ALL missing keys at once (not just the first) so the error message
    # is maximally useful in a single run.
    missing = [k for k in keys if k not in payload]
    if missing:
        raise ValueError(
            f"{context}: missing required keys {missing}. " f"Got keys: {sorted(payload.keys())}"
        )


def assert_unique(
    items: list[Any],
    *,
    name: str = "items",
    context: str = "payload",
) -> None:
    """Raise ValueError if `items` contains any duplicates.

    Used for bloc lists (you can't list "latino" twice) and similar.
    """
    # Walk once, remembering what we've seen; anything seen twice is a duplicate.
    # A set gives O(1) membership tests, so this is O(n) overall.
    seen: set = set()
    dupes = []
    for item in items:
        if item in seen:
            dupes.append(item)
        seen.add(item)
    if dupes:
        raise ValueError(f"{context}.{name}: duplicate values found: {dupes}")


def assert_sorted_unique(
    items: list[Any],
    *,
    name: str = "items",
    context: str = "payload",
) -> None:
    """Raise ValueError unless `items` is strictly increasing (sorted, no dups).

    Used for election cycles: they must be in chronological order with no
    repeated year. "Strictly increasing" means each value is larger than the
    one before — which automatically rules out duplicates too.
    """
    # First reuse the duplicate check (composition keeps the rules in one place),
    # then verify the ordering pairwise.
    assert_unique(items, name=name, context=context)
    for i in range(len(items) - 1):
        if items[i] >= items[i + 1]:
            raise ValueError(
                f"{context}.{name}: must be strictly increasing, "
                f"but {items[i]} >= {items[i + 1]} at index {i}"
            )


def assert_valid_share(
    value: float,
    *,
    name: str,
    context: str,
) -> None:
    """Raise ValueError unless 0.0 <= value <= 1.0.

    A "share" is a probability/fraction (a vote share, a coalition weight), so
    by definition it can't be negative or exceed 1. Catching an out-of-range
    value here stops nonsense (e.g. a 1.3 vote share) from reaching the math.
    """
    if not (0.0 <= value <= 1.0):
        raise ValueError(f"{context}.{name}: must be in [0.0, 1.0], got {value}")


def assert_shares_sum_to_one(
    d: dict[str, float],
    *,
    context: str,
    tol: float = 1e-6,
) -> None:
    """Raise ValueError if the dict's values do not sum to 1.0 (within tol).

    This enforces the "parts of a whole" / simplex constraint: a set of coalition
    weights, or the three λ layer weights, must add up to exactly 1. We allow a
    tiny tolerance `tol` because floating-point addition is not exact — summing
    five numbers that "should" total 1.0 can land at 0.9999999998. 1e-6 is a
    generous serialization-level tolerance (looser than the optimizer's own
    internal 1e-9 precision target noted in CLAUDE.md); it accepts honest round-
    off while still catching real bugs like weights that sum to 0.95 or 1.2.
    """
    # Empty dict can't sum to 1 and almost always signals a construction bug.
    if not d:
        raise ValueError(f"{context}: share dict must not be empty")
    total = sum(d.values())
    # abs(total - 1.0): distance from 1, regardless of which side we're off on.
    if abs(total - 1.0) > tol:
        raise ValueError(
            f"{context}: shares must sum to 1.0 ± {tol}, got {total:.10f}. "
            f"Keys: {sorted(d.keys())}"
        )
