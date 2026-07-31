#!/usr/bin/env python3
"""CLI for electoral/metrics/ground_truth_accuracy.py.

Scores a model's shock-response predictions against the three ground-truth
layers built in Phase 1 (data/ground_truth/{panel,ces,exit_poll}_deltas.json).

USAGE (real run, once predictions exist):
    python scripts/score_against_ground_truth.py --predictions path/to/preds.json

    predictions JSON shape: {shock_id: <ShockResponseData dict, optionally
    {"data": {...}}-wrapped>}, i.e. exactly what you get by dumping
    {s.shock: s.to_dict() for s in shock_response_results}.

USAGE (self-test, no model needed -- sanity-checks the scorer's arithmetic):
    python scripts/score_against_ground_truth.py --self-test

    Builds a synthetic "predictions" file FROM the ground truth itself (a
    perfect predictor) and confirms direction_accuracy=1.0, mean_abs_error=0.0
    on every eligible cell. This validates the SCORING CODE, not any model --
    it is clearly labeled as such in the output and must never be reported as
    a real accuracy result.

As of this writing, NO real predictions file exists for the 56 configured
shocks (electoral/stages.py's build_shock_response has not been run across
them -- see the codebase note this script's docstring inherits from). Run the
pipeline first, save its ShockResponseData outputs in the shape above, then
invoke this script for real.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))
from electoral.metrics.ground_truth_accuracy import (
    normalize_ces,
    normalize_exit_poll,
    normalize_panel,
    score_all_sources,
)

REPO_ROOT = Path(__file__).parents[1]
GT_DIR = REPO_ROOT / "data" / "ground_truth"


def load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _predictions_from_cells(cells) -> dict:
    """A perfect predictor for exactly one source's cells: predict the
    measured_delta for every cell, wrapped as party='democrat'
    ShockResponseData-shaped entries so normalize_prediction() exercises the
    real code path (including the sign-flip logic, tested here with the
    identity case since party='democrat' is a no-op).
    """
    from electoral.metrics.ground_truth_accuracy import BLOC_TO_STRATUM, STRATUM_TO_PRED_KEY

    preds: dict[str, dict] = {}
    for cell in cells:
        entry = preds.setdefault(cell.shock_id, {"party": "democrat", "deltas_race": {}, "deltas_religion": {}, "deltas_gender": {}})
        key = STRATUM_TO_PRED_KEY[BLOC_TO_STRATUM[cell.bloc]]
        entry[key][cell.bloc] = cell.measured_delta
    return preds


def build_self_test_predictions() -> dict[str, dict]:
    """Returns {"panel": preds, "ces": preds, "exit_poll": preds} -- one
    perfect-predictor dict PER SOURCE, built only from that source's own
    cells. Deliberately NOT merged into one shared dict: CES and exit-poll
    ground truth overlap on trump_election_upset_2016 (CES has a 2016<->2017
    year-pair; exit polls have a 2004->2016 cycle delta for the same shock)
    with genuinely different measured_delta values for the same blocs -- a
    merged dict would let one source silently overwrite the other and the
    self-test would then show spurious nonzero error for the overwritten
    source, even though the scorer itself is correct. Each source must be
    self-tested against its own ground truth, matching the "cross-check,
    don't merge" principle the ground-truth layer itself follows.
    """
    panel = load_json(GT_DIR / "panel_deltas.json")
    ces = load_json(GT_DIR / "ces_deltas.json")
    exit_poll = load_json(GT_DIR / "exit_poll_deltas.json")
    return {
        "panel": _predictions_from_cells(normalize_panel(panel)),
        "ces": _predictions_from_cells(normalize_ces(ces)),
        "exit_poll": _predictions_from_cells(normalize_exit_poll(exit_poll)),
    }


def print_report(name: str, report) -> None:
    print(f"\n{'=' * 70}\n{name.upper()}  (fidelity_rank={report.fidelity_rank})\n{'=' * 70}")
    print(f"  ground-truth cells:  {report.n_ground_truth_cells}")
    print(f"  scored (any dimension): {report.n_scored_total}")
    print(f"  REFUSED: {report.n_refused_total}  "
          f"(no_prediction={report.n_refused_no_prediction}, ineligible={report.n_refused_ineligible})")

    print("\n  --- DIRECTION SCORE ---")
    d = report.direction
    if d is None:
        print("  n/a -- no cells eligible for direction scoring in this source "
              "(expected for CES/exit_poll, which are never direction-eligible)")
    else:
        print(f"  accuracy: {d.accuracy:.3f}  ({d.n_correct}/{d.n_scored} correct)")
        print(f"  n_shocks represented: {d.n_shocks}   n_independent_windows: {d.n_windows}")
        print(f"  bloc distribution: {d.bloc_distribution}")
        print(f"  *** {d.caveat}")

    print("\n  --- MAGNITUDE / CALIBRATION SCORE ---")
    m = report.magnitude
    if m is None:
        print("  n/a -- no cells eligible for magnitude scoring")
    else:
        print(f"  n_scored: {m.n_scored}")
        print(f"  mean_abs_error: {m.mean_abs_error:.5f}")
        if m.calibration_ratio is not None:
            print(f"  calibration_ratio (mean|pred|/mean|true|): {m.calibration_ratio:.2f}x")
        else:
            print("  calibration_ratio: n/a (mean measured magnitude is 0)")
        if m.median_cellwise_ratio is not None:
            print(f"  median cell-wise ratio: {m.median_cellwise_ratio:.2f}x "
                  f"({m.n_cellwise_ratio_excluded_near_zero} cell(s) excluded as near-zero measured)")
        if m.global_pearson_correlation is not None:
            print(f"  global Pearson correlation (pred vs. measured): {m.global_pearson_correlation:.3f}")
        if m.per_stratum_mae:
            for k, v in m.per_stratum_mae.items():
                print(f"    {k} MAE: {v:.5f}")

    print("\n  --- RANK CORRELATION (per shock, >=5 magnitude-usable blocs) ---")
    rc = report.rank_correlation
    print(f"  shocks with any magnitude cells: {rc.n_shocks_total_with_any_magnitude_cells}  "
          f"| eligible (>=5 blocs): {rc.n_shocks_eligible}")
    if rc.mean_spearman_rho is not None:
        print(f"  mean Spearman rho across eligible shocks: {rc.mean_spearman_rho:.3f}")
    for s in rc.per_shock:
        rho_str = f"{s.spearman_rho:.3f}" if s.spearman_rho is not None else "n/a"
        print(f"    {s.shock_id:38s} n_blocs={s.n_blocs:2d}  rho={rho_str}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", type=Path, help="Path to a predictions JSON (see docstring for shape)")
    ap.add_argument("--self-test", action="store_true", help="Score the ground truth against itself (perfect predictor) to sanity-check the scoring code")
    ap.add_argument("--out", type=Path, default=None, help="Optional path to write the full JSON report")
    args = ap.parse_args()

    if not args.predictions and not args.self_test:
        ap.error("pass --predictions <file> for a real run, or --self-test to sanity-check the scorer")

    panel = load_json(GT_DIR / "panel_deltas.json")
    ces = load_json(GT_DIR / "ces_deltas.json")
    exit_poll = load_json(GT_DIR / "exit_poll_deltas.json")

    if args.self_test:
        print("=" * 70)
        print("SELF-TEST MODE: each source is scored against a perfect predictor")
        print("built ONLY from its own ground truth (never merged across sources --")
        print("see build_self_test_predictions() docstring for why). This validates")
        print("the SCORING CODE, not any model. A real model run should NOT")
        print("reproduce these numbers.")
        print("=" * 70)
        preds_by_source = build_self_test_predictions()
        reports = {
            "panel": score_all_sources(preds_by_source["panel"], panel_json=panel)["panel"],
            "ces": score_all_sources(preds_by_source["ces"], ces_json=ces)["ces"],
            "exit_poll": score_all_sources(preds_by_source["exit_poll"], exit_poll_json=exit_poll)["exit_poll"],
        }
    else:
        predictions = load_json(args.predictions)
        print(f"Loaded {len(predictions)} shock predictions from {args.predictions}")
        reports = score_all_sources(predictions, panel_json=panel, ces_json=ces, exit_poll_json=exit_poll)

    for name, report in reports.items():
        print_report(name, report)

    if args.out:
        out_data = {name: r.to_dict() for name, r in reports.items()}
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(out_data, f, indent=2, default=str)
        print(f"\nWrote full report to {args.out}")


if __name__ == "__main__":
    main()
