"""Schema validation helpers for Electoral Equilibrium pipeline artifacts.

All helpers are pure functions with no I/O or global state.
Every error message names the context (class + field) for fast diagnosis.
"""

from __future__ import annotations

from typing import Any


def assert_required_keys(
    payload: dict[str, Any],
    keys: list[str],
    *,
    context: str = "payload",
) -> None:
    """Raise ValueError if any required key is missing from payload."""
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
    """Raise ValueError if items contains any duplicates."""
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
    """Raise ValueError unless items is strictly increasing (sorted with no duplicates)."""
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
    """Raise ValueError unless 0.0 <= value <= 1.0."""
    if not (0.0 <= value <= 1.0):
        raise ValueError(f"{context}.{name}: must be in [0.0, 1.0], got {value}")


def assert_shares_sum_to_one(
    d: dict[str, float],
    *,
    context: str,
    tol: float = 1e-6,
) -> None:
    """Raise ValueError if dict values do not sum to 1.0 within tol."""
    if not d:
        raise ValueError(f"{context}: share dict must not be empty")
    total = sum(d.values())
    if abs(total - 1.0) > tol:
        raise ValueError(
            f"{context}: shares must sum to 1.0 ± {tol}, got {total:.10f}. "
            f"Keys: {sorted(d.keys())}"
        )
