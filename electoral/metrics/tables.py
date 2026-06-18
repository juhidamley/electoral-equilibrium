from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from electoral.metrics.performance import (
    win_probability,
    equilibrium_gap,
    bloc_delta_summary,
)

log = logging.getLogger(__name__)


def _load_artifact(path: Path) -> dict[str, Any] | None:
    """Load an artifact JSON, returning the inner data payload.
    Handles both StageArtifact-wrapped {data: {...}} and bare dicts.
    Returns None if the file is missing or unparseable."""
    if not path.exists():
        log.warning("artifact not found: %s", path)
        return None
    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError:
        log.warning("artifact unparseable: %s", path)
        return None
    return raw.get("data", raw)


def build_shock_summary_table(
    shock_ids: list[str],
    config: Any,
    artifact_dir: str | None = None,
) -> list[dict[str, Any]]:
    """Assemble a summary row per shock from its EquilibriumData and
    SimulationData artifacts.

    Columns: shock_id, category, win_probability, equilibrium_gap,
    top_moving_bloc, top_moving_delta.

    Missing artifacts produce a row with nulls and a 'status' note rather
    than being skipped, so the table length matches shock_ids.
    """
    base = Path(artifact_dir or getattr(config, "output_dir", "artifacts"))
    taxonomy = getattr(config, "shock_taxonomy", {}) or {}

    rows: list[dict[str, Any]] = []
    for sid in shock_ids:
        key = sid[:30]
        eq = _load_artifact(base / f"equilibrium_{key}.json")
        sim = _load_artifact(base / f"sim_{sid}.json")
        if sim is None:
            sim = _load_artifact(base / f"simulation.json")

        row: dict[str, Any] = {
            "shock_id": sid,
            "category": taxonomy.get(sid, {}).get("category") if isinstance(taxonomy.get(sid), dict) else taxonomy.get(sid),
            "win_probability": None,
            "equilibrium_gap": None,
            "top_moving_bloc": None,
            "top_moving_delta": None,
            "status": "ok",
        }

        if eq is None:
            row["status"] = "missing_equilibrium"
            rows.append(row)
            continue

        if sim is not None:
            wp = win_probability(sim)
            row["win_probability"] = wp["win_probability"]
        else:
            row["status"] = "missing_simulation"

        weights = eq.get("weights", {})
        mu = eq.get("mu_shifted", {})
        target = eq.get("target", getattr(config, "target", 0.5066))
        if weights and mu:
            row["equilibrium_gap"] = equilibrium_gap(weights, mu, float(target))

        if weights:
            n = len(weights)
            equal = {b: 1.0 / n for b in weights}
            summary = bloc_delta_summary(equal, weights)
            if summary:
                top_bloc = next(iter(summary))
                row["top_moving_bloc"] = top_bloc
                row["top_moving_delta"] = summary[top_bloc]

        rows.append(row)

    return rows
