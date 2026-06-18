from __future__ import annotations

from typing import Any


def win_probability(sim_data: Any) -> dict[str, float]:
    """Extract win probability and its CI from a SimulationData artifact.

    Accepts either a SimulationData instance or its dict form.
    Returns JSON-native dict.
    """
    if hasattr(sim_data, "win_probability"):
        wp = sim_data.win_probability
        lo = getattr(sim_data, "win_probability_low", 0.0)
        hi = getattr(sim_data, "win_probability_high", 1.0)
    else:
        wp = float(sim_data["win_probability"])
        lo = float(sim_data.get("win_probability_low", 0.0))
        hi = float(sim_data.get("win_probability_high", 1.0))
    return {
        "win_probability": float(wp),
        "win_probability_low": float(lo),
        "win_probability_high": float(hi),
    }


def equilibrium_gap(
    weights: dict[str, float],
    mu: dict[str, float],
    target: float,
) -> float:
    """Return w^T mu - target. Positive means the coalition clears the
    win threshold; negative means it falls short.

    Only blocs present in BOTH weights and mu contribute (intersection),
    so callers can pass race-only dicts without religion/gender leaking in.
    """
    blocs = set(weights) & set(mu)
    mu_eff = sum(weights[b] * mu[b] for b in blocs)
    return float(mu_eff - target)


def bloc_delta_summary(
    baseline_weights: dict[str, float],
    rebalanced_weights: dict[str, float],
) -> dict[str, float]:
    """Signed per-bloc weight change (rebalanced - baseline), returned as
    a dict ordered by descending absolute magnitude.

    Union of bloc keys; a bloc missing from one side is treated as 0.0.
    """
    blocs = set(baseline_weights) | set(rebalanced_weights)
    deltas = {
        b: float(rebalanced_weights.get(b, 0.0) - baseline_weights.get(b, 0.0))
        for b in blocs
    }
    ordered = dict(sorted(deltas.items(), key=lambda kv: abs(kv[1]), reverse=True))
    return ordered
