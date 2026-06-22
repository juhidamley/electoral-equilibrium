"""Sensitivity analysis: how do the results change as a shock's INTENSITY varies?

A reporting SCRIPT (run by hand). It re-runs the full shock→optimizer→Monte-Carlo
chain for one event at a sweep of intensity values (e.g. 0.5 → 2.0) and plots how
the win probability (with its CI band) and the per-bloc coalition weights move as
the shock gets stronger. The output figure is the "sensitivity plot" the Week-6 PR
description includes — it shows the model responds smoothly/sensibly to intensity
rather than flipping erratically. Uses matplotlib's headless "Agg" backend so it
runs on a server with no display.
"""

import argparse
import logging
import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")  # headless for HPC
import matplotlib.pyplot as plt

from electoral.config import PipelineConfig
from electoral.core.types import CANONICAL_RACES
from electoral.kernels.shock import build_shock_response
from electoral.simulation.montecarlo import run_ilr_montecarlo

log = logging.getLogger(__name__)


def run_sensitivity(config, event, shock_id, intensities=None):
    """Vary intensity from 0.5 to 2.0 and record win prob + per-bloc weights."""
    if intensities is None:
        intensities = np.arange(0.5, 2.01, 0.25)

    win_probs = []
    win_lows = []
    win_highs = []
    bloc_weights = {r: [] for r in CANONICAL_RACES}

    for intensity in intensities:
        shock, equilibrium = build_shock_response(config, event, float(intensity))
        sim = run_ilr_montecarlo(equilibrium, config, n_simulations=10000)

        win_probs.append(sim.win_probability)
        win_lows.append(sim.win_probability_low)
        win_highs.append(sim.win_probability_high)
        for r in CANONICAL_RACES:
            bloc_weights[r].append(equilibrium.weights.get(r, 0.0))

        log.info("intensity=%.2f → win_prob=%.4f", intensity, sim.win_probability)

    return intensities, win_probs, win_lows, win_highs, bloc_weights


def make_plot(intensities, win_probs, win_lows, win_highs, bloc_weights, shock_id, out_path):
    """Two-panel figure: win prob vs intensity, and per-bloc weights vs intensity."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: win probability with CI band
    ax1.plot(intensities, win_probs, "o-", color="#2c3e50", linewidth=2, label="Win probability")
    ax1.fill_between(intensities, win_lows, win_highs, alpha=0.2, color="#3498db", label="90% CI")
    ax1.axhline(0.5, ls="--", color="gray", alpha=0.5)
    ax1.set_xlabel("Shock intensity")
    ax1.set_ylabel("Win probability")
    ax1.set_title(f"Win probability vs intensity — {shock_id}")
    ax1.set_ylim(-0.02, 1.02)
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Panel 2: per-bloc weights
    for r in CANONICAL_RACES:
        ax2.plot(intensities, bloc_weights[r], "o-", label=r, linewidth=1.5)
    ax2.set_xlabel("Shock intensity")
    ax2.set_ylabel("Coalition weight")
    ax2.set_title(f"Per-bloc weights vs intensity — {shock_id}")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    log.info("saved plot to %s", out_path)


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/smoke.json")
    parser.add_argument("--event", required=True)
    parser.add_argument("--shock-id", required=True)
    parser.add_argument("--output-dir", default="artifacts")
    args = parser.parse_args()

    config = PipelineConfig.from_json(args.config)

    intensities, wp, wl, wh, bw = run_sensitivity(config, args.event, args.shock_id)

    out_path = Path(args.output_dir) / f"sensitivity_{args.shock_id}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    make_plot(intensities, wp, wl, wh, bw, args.shock_id, out_path)

    # Also save the raw data as JSON
    data = {
        "shock_id": args.shock_id,
        "intensities": [float(i) for i in intensities],
        "win_probabilities": wp,
        "win_prob_low": wl,
        "win_prob_high": wh,
        "bloc_weights": bw,
    }
    json_path = Path(args.output_dir) / f"sensitivity_{args.shock_id}.json"
    json_path.write_text(json.dumps(data, indent=2))
    log.info("saved data to %s", json_path)


if __name__ == "__main__":
    main()
