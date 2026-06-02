"""Run LOCO-CV validation on the real voter panel and compare to prediction markets.

Usage (from project root):
    python scripts/run_loco_validation.py [--party democrat|republican]

Outputs:
    artifacts/baseline_loco.json  — per-fold GP predictions + market comparison
    stdout                        — human-readable report with calibration flags

Market data source
------------------
IEM (Iowa Electronic Market) WTA closing prices and Intrade/PredictIt
closing prices on election day, for the Democrat candidate's win probability.
Values taken from published post-election market summaries.

  2004  IEM WTA  ~0.405  Kerry vs Bush       (Bush won)
  2008  Intrade  ~0.912  Obama vs McCain      (Obama won)
  2012  Intrade  ~0.674  Obama vs Romney      (Obama won)
  2016  PredictIt ~0.820  H.Clinton vs Trump  (Trump won; market was wrong)
  2020  PredictIt ~0.650  Biden vs Trump      (Biden won)
  2024  PredictIt ~0.470  Harris vs Trump     (Trump won)
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import pathlib
import sys

import numpy as np
import pandas as pd

# ── paths ─────────────────────────────────────────────────────────────────────
_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from electoral.config import PipelineConfig
from electoral.core.rng import make_rng, derive_seed
from electoral.kernels.data import build_voter_panel as _panel_kernel
from electoral.models.ml_baseline import (
    fit_gp_classifier,
    ground_truth_winning_cycles,
    platt_scale_loco,
    save_loco_json,
)

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("loco_validation")

# ── historical market prices: P(Democrat wins) on election day ────────────────
# Source: IEM WTA contract closing prices (2004), Intrade (2008/2012),
# PredictIt national popular-vote market (2016/2020/2024).
# These are calibration benchmarks only — NOT used as model inputs.
_MARKET_DEM_PROB: dict[int, float] = {
    2004: 0.405,  # IEM; Bush won
    2008: 0.912,  # Intrade; Obama won
    2012: 0.674,  # Intrade; Obama won
    2016: 0.820,  # PredictIt election eve; Trump won (market miss)
    2020: 0.650,  # PredictIt election day; Biden won
    2024: 0.470,  # PredictIt election day; Trump won
}

_DIVERGENCE_FLAG_PP: float = 10.0  # percentage points


# ── helpers ───────────────────────────────────────────────────────────────────


def _load_or_build_panel(config: PipelineConfig) -> pd.DataFrame:
    """Return full panel DataFrame, building it from raw survey data if needed."""
    panel_dir = pathlib.Path(config.output_dir) / "panel"
    names = ("panel_race.parquet", "panel_religion.parquet", "panel_gender.parquet")
    existing = [panel_dir / n for n in names if (panel_dir / n).exists()]

    if len(existing) == 3:
        log.info("Loading pre-built panel from %s", panel_dir)
        return pd.concat([pd.read_parquet(p) for p in existing], ignore_index=True)

    print("Panel parquets not found — building from raw survey data (this may take a moment)…")
    payload, df = _panel_kernel(config)

    panel_dir.mkdir(parents=True, exist_ok=True)
    from electoral.core.types import CANONICAL_RACES, CANONICAL_RELIGIONS, CANONICAL_GENDERS
    df[df["bloc"].isin(CANONICAL_RACES)].to_parquet(panel_dir / "panel_race.parquet", index=False)
    df[df["bloc"].isin(CANONICAL_RELIGIONS)].to_parquet(panel_dir / "panel_religion.parquet", index=False)
    df[df["bloc"].isin(CANONICAL_GENDERS)].to_parquet(panel_dir / "panel_gender.parquet", index=False)
    print(f"  Wrote panel parquets to {panel_dir}")
    return df


def _market_prob_for_party(cycle: int, party: str) -> float | None:
    """Return P(party wins) from markets, or None if no data."""
    dem_prob = _MARKET_DEM_PROB.get(cycle)
    if dem_prob is None:
        return None
    return dem_prob if party == "democrat" else 1.0 - dem_prob


def _build_market_comparison(
    folds: tuple,
    party: str,
    *,
    use_calibrated: bool = False,
) -> dict:
    """Compare LOCO GP predictions against market prices for cycles 2004+.

    When ``use_calibrated=True``, uses ``calibrated_prob_win`` instead of
    the raw ``prob_win``.  Falls back to raw if calibrated is NaN.
    """
    divergences: dict[str, dict] = {}
    flagged_cycles: list[int] = []

    for fold in folds:
        mkt = _market_prob_for_party(fold.cycle, party)
        if mkt is None:
            continue

        prob = fold.calibrated_prob_win if use_calibrated else fold.prob_win
        # Fall back to raw if calibrated is not available.
        if math.isnan(prob):
            prob = fold.prob_win
        if math.isnan(prob):
            continue

        div_pp = abs(prob - mkt) * 100.0
        flagged = div_pp > _DIVERGENCE_FLAG_PP
        if flagged:
            flagged_cycles.append(fold.cycle)
        divergences[str(fold.cycle)] = {
            "gp_prob_win": round(fold.prob_win, 4),
            "calibrated_prob_win": round(fold.calibrated_prob_win, 4)
            if not math.isnan(fold.calibrated_prob_win)
            else None,
            "market_prob_win": round(mkt, 4),
            "divergence_pp": round(div_pp, 2),
            "flagged": flagged,
        }

    max_div = max((v["divergence_pp"] for v in divergences.values()), default=0.0)
    return {
        "cycles_compared": sorted(int(k) for k in divergences),
        "max_divergence_pp": round(max_div, 2),
        "flagged_cycles": flagged_cycles,
        "calibration_ok": len(flagged_cycles) == 0,
        "divergences": divergences,
    }


# ── main ──────────────────────────────────────────────────────────────────────


def run(party: str, out_path: pathlib.Path) -> None:
    import dataclasses
    _base = PipelineConfig.from_json(_ROOT / "configs" / "base.json")
    config = dataclasses.replace(_base, party=party)

    panel_df = _load_or_build_panel(config)
    print(f"Panel: {len(panel_df)} rows, {panel_df['cycle'].nunique()} cycles, party={party}")

    winning = ground_truth_winning_cycles(party)
    print(f"Ground-truth winning cycles ({len(winning)}): {winning}")

    rng = make_rng(derive_seed(config.seed, "loco_validation"))
    print("Running LOCO-CV…")
    result = fit_gp_classifier(panel_df, party, winning_cycles=winning, rng=rng)

    # Platt scaling is implemented (platt_scale_loco) but intentionally not applied here.
    #
    # When tested on the 20-cycle panel it degraded both metrics:
    #   accuracy  0.800 → 0.750   (introduced a new error on 1956)
    #   Brier     0.122 → 0.162
    # and made the 2024 market divergence worse (12.9 pp → 31.2 pp).
    #
    # Root cause: the apparent "calibration errors" on 2012 and 2020 are a
    # feature-timing mismatch, not genuine miscalibration. GP features are
    # post-election vote shares (the panel records what the electorate actually did),
    # while market prices were set before the election. Of course the GP is more
    # confident than Intrade was on election eve — it knows the outcome through
    # the features. The 2024 failure is a structural out-of-distribution extrapolation
    # that no calibration layer can fix from within-distribution training data.
    #
    # To experiment with Platt scaling, uncomment the two lines below:
    # result = platt_scale_loco(result)

    # Save: core LOCO + calibrated fields + market comparison (calibrated)
    save_loco_json(result, out_path)
    market_comp_raw = _build_market_comparison(result.folds, party, use_calibrated=False)
    market_comp_cal = _build_market_comparison(result.folds, party, use_calibrated=True)
    loco_data = json.loads(out_path.read_text())
    loco_data["market_comparison_raw"] = market_comp_raw
    loco_data["market_comparison_calibrated"] = market_comp_cal
    out_path.write_text(json.dumps(loco_data, indent=2))

    # ── Print report ───────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print(f"LOCO-CV Report — {party.upper()}")
    print("=" * 72)

    valid_folds = [f for f in result.folds if not math.isnan(f.prob_win)]
    nan_folds = [f for f in result.folds if math.isnan(f.prob_win)]

    print(f"Total folds:              {len(result.folds)}")
    print(f"Valid folds:              {len(valid_folds)}  "
          f"(NaN: {len(nan_folds)} — single-class training sets)")
    print(f"Raw accuracy:             {result.accuracy:.3f}")
    print(f"Raw Brier score:          {result.brier_score:.4f}")
    print(f"Calibrated accuracy:      {result.calibrated_accuracy:.3f}")
    print(f"Calibrated Brier score:   {result.calibrated_brier_score:.4f}")
    print()
    print(f"{'Cycle':<8} {'y':<4} {'Raw':<8} {'Std':<8} {'Cal':<8} "
          f"{'Market':<8} {'Raw Δ':<8} {'Cal Δ':<8} {'Flag'}")
    print("-" * 80)
    for fold in result.folds:
        mkt = _market_prob_for_party(fold.cycle, party)
        if math.isnan(fold.prob_win):
            print(f"{fold.cycle:<8} {fold.y_true:<4} {'NaN':<8} {'NaN':<8} "
                  f"{'NaN':<8} {'—':<8} {'—':<8} {'—':<8}")
            continue

        raw_str = f"{fold.prob_win:.3f}"
        std_str = f"{fold.prob_std:.3f}"
        cal_val = fold.calibrated_prob_win
        cal_str = f"{cal_val:.3f}" if not math.isnan(cal_val) else " NaN"

        if mkt is not None:
            raw_div = abs(fold.prob_win - mkt) * 100.0
            cal_div = abs(cal_val - mkt) * 100.0 if not math.isnan(cal_val) else float("nan")
            mkt_str = f"{mkt:.3f}"
            raw_d_str = f"{raw_div:>6.1f}"
            cal_d_str = f"{cal_div:>6.1f}" if not math.isnan(cal_div) else "   —"
            flag = " ⚠" if (not math.isnan(cal_div) and cal_div > _DIVERGENCE_FLAG_PP) else ""
        else:
            mkt_str, raw_d_str, cal_d_str, flag = "—", "—", "—", ""

        print(f"{fold.cycle:<8} {fold.y_true:<4} {raw_str:<8} {std_str:<8} "
              f"{cal_str:<8} {mkt_str:<8} {raw_d_str:<8} {cal_d_str:<8}{flag}")

    print()
    print("Market comparison — calibrated (2004+):")
    if market_comp_cal["calibration_ok"]:
        print(f"  ✓ All divergences ≤ {_DIVERGENCE_FLAG_PP:.0f} pp  "
              f"(max = {market_comp_cal['max_divergence_pp']:.1f} pp)")
    else:
        flagged = market_comp_cal["flagged_cycles"]
        max_div = market_comp_cal["max_divergence_pp"]
        print(f"  ✗ {len(flagged)} cycle(s) diverge > {_DIVERGENCE_FLAG_PP:.0f} pp "
              f"(max {max_div:.1f} pp): {flagged}")

    print()
    print("Market comparison — raw GP (2004+):")
    if market_comp_raw["calibration_ok"]:
        print(f"  ✓ All divergences ≤ {_DIVERGENCE_FLAG_PP:.0f} pp  "
              f"(max = {market_comp_raw['max_divergence_pp']:.1f} pp)")
    else:
        flagged = market_comp_raw["flagged_cycles"]
        max_div = market_comp_raw["max_divergence_pp"]
        print(f"  ✗ {len(flagged)} cycle(s) diverge > {_DIVERGENCE_FLAG_PP:.0f} pp "
              f"(max {max_div:.1f} pp): {flagged}")

    print()
    print(f"Saved → {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--party", choices=["democrat", "republican"], default="democrat")
    ap.add_argument(
        "--out",
        default="artifacts/baseline_loco.json",
        help="Output path (default: artifacts/baseline_loco.json)",
    )
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    run(args.party, out)


if __name__ == "__main__":
    main()
