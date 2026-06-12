"""Three-way baseline comparison on the eval set.

Methods:
  A  news_only    — RoBERTa news scores → score_to_bin → MAE vs ground truth
  B  social_only  — RoBERTa social scores → score_to_bin → MAE vs ground truth
  C  unified_llm  — ShockEstimator.estimate() → MAE vs ground truth

Results written to:
  artifacts/baseline_comparison.json
  docs/baseline_comparison.md
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from electoral.llm.eval import mae_in_delta_units
from electoral.nlp.elasticity import score_to_bin

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

_EVAL_PATH = ROOT / "data" / "finetune" / "eval.jsonl"
_OUT_JSON = ROOT / "artifacts" / "baseline_comparison.json"
_OUT_MD = ROOT / "docs" / "baseline_comparison.md"

_DEFAULT_ADAPTER = os.environ.get(
    "ADAPTER_PATH",
    "/Volumes/JUHIDRIVE/electoralData/models/mistral-r16",
)


# ── Data loading ──────────────────────────────────────────────────────────────


def load_eval(path: Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    log.info("Loaded %d eval records from %s", len(records), path)
    return records


# ── Per-method scoring ────────────────────────────────────────────────────────


def _scores_are_empty(scores: dict) -> bool:
    return not scores or all(v == 0.0 for v in scores.values())


def run_news_only(records: list[dict]) -> tuple[list[float], dict[str, list[float]]]:
    """Method A: news_roberta_scores → bin → MAE per record."""
    maes: list[float] = []
    bloc_errors: dict[str, list[float]] = {}
    skipped = 0

    for rec in records:
        scores: dict = rec.get("news_roberta_scores") or {}
        if _scores_are_empty(scores):
            skipped += 1
            continue
        true_bins: dict[str, str] = rec.get("delta_bins") or {}
        pred_bins = {bloc: score_to_bin(float(v)) for bloc, v in scores.items()}
        common = sorted(set(pred_bins) & set(true_bins))
        if not common:
            skipped += 1
            continue
        pred_sub = {b: pred_bins[b] for b in common}
        true_sub = {b: true_bins[b] for b in common}
        mae = mae_in_delta_units(pred_sub, true_sub)
        maes.append(mae)
        for b in common:
            from electoral.core.types import BIN_MIDPOINTS
            err = abs(BIN_MIDPOINTS[pred_sub[b]] - BIN_MIDPOINTS[true_sub[b]])
            bloc_errors.setdefault(b, []).append(err)

    log.info("news_only: %d processed, %d skipped (no RoBERTa scores)", len(maes), skipped)
    return maes, bloc_errors


def run_social_only(records: list[dict]) -> tuple[list[float], dict[str, list[float]]]:
    """Method B: social_roberta_scores → bin → MAE per record."""
    maes: list[float] = []
    bloc_errors: dict[str, list[float]] = {}
    skipped = 0

    for rec in records:
        scores: dict = rec.get("social_roberta_scores") or {}
        if _scores_are_empty(scores):
            skipped += 1
            continue
        true_bins: dict[str, str] = rec.get("delta_bins") or {}
        pred_bins = {bloc: score_to_bin(float(v)) for bloc, v in scores.items()}
        common = sorted(set(pred_bins) & set(true_bins))
        if not common:
            skipped += 1
            continue
        pred_sub = {b: pred_bins[b] for b in common}
        true_sub = {b: true_bins[b] for b in common}
        mae = mae_in_delta_units(pred_sub, true_sub)
        maes.append(mae)
        for b in common:
            from electoral.core.types import BIN_MIDPOINTS
            err = abs(BIN_MIDPOINTS[pred_sub[b]] - BIN_MIDPOINTS[true_sub[b]])
            bloc_errors.setdefault(b, []).append(err)

    log.info("social_only: %d processed, %d skipped (no social scores)", len(maes), skipped)
    return maes, bloc_errors


def run_unified_llm(
    records: list[dict],
    adapter_path: str,
) -> tuple[list[float], dict[str, list[float]]]:
    """Method C: ShockEstimator.estimate() → MAE per record."""
    from electoral.llm.inference import ShockEstimator
    from electoral.core.types import BIN_MIDPOINTS

    log.info("Loading ShockEstimator from %s", adapter_path)
    estimator = ShockEstimator(adapter_path=adapter_path)

    maes: list[float] = []
    bloc_errors: dict[str, list[float]] = {}
    failed = 0

    for i, rec in enumerate(records):
        event = {
            "shock_id": rec.get("id", f"eval_{i}"),
            "cycle": int(rec.get("collected_at", "2024")[:4]),
            "party": rec.get("party", "democrat"),
            "description": rec.get("description", ""),
            "news_roberta_scores": rec.get("news_roberta_scores") or {},
            "social_roberta_scores": rec.get("social_roberta_scores") or {},
        }
        true_bins: dict[str, str] = rec.get("delta_bins") or {}

        try:
            result = estimator.estimate(event, intensity=1.0)
        except Exception as exc:
            log.warning("estimate() failed for record %d (%s): %s", i, event["shock_id"], exc)
            failed += 1
            continue

        pred_flat = {
            **result.delta_bins_race,
            **result.delta_bins_religion,
            **result.delta_bins_gender,
        }
        common = sorted(set(pred_flat) & set(true_bins))
        if not common:
            failed += 1
            continue

        pred_sub = {b: pred_flat[b] for b in common}
        true_sub = {b: true_bins[b] for b in common}
        mae = mae_in_delta_units(pred_sub, true_sub)
        maes.append(mae)
        for b in common:
            err = abs(BIN_MIDPOINTS[pred_sub[b]] - BIN_MIDPOINTS[true_sub[b]])
            bloc_errors.setdefault(b, []).append(err)

        if (i + 1) % 10 == 0:
            log.info("  unified_llm: %d/%d records done", i + 1, len(records))

    log.info("unified_llm: %d processed, %d failed", len(maes), failed)
    return maes, bloc_errors


# ── Aggregation ───────────────────────────────────────────────────────────────


_NO_DATA_NOTE = (
    "N/A — no real RoBERTa scores in eval set (all synthetic). "
    "Run scripts/score_shock.py on eval examples first, then re-run this script."
)


def aggregate(maes: list[float], bloc_errors: dict[str, list[float]]) -> dict:
    if not maes:
        return {"overall_mae": None, "n_records": 0, "per_bloc_mae": {}, "note": _NO_DATA_NOTE}
    overall = sum(maes) / len(maes)
    per_bloc = {
        b: sum(errs) / len(errs)
        for b, errs in sorted(bloc_errors.items())
    }
    return {"overall_mae": overall, "n_records": len(maes), "per_bloc_mae": per_bloc}


# ── Output formatting ─────────────────────────────────────────────────────────


def _fmt(v: float) -> str:
    return f"{v:.4f}" if math.isfinite(v) else "n/a"


def write_markdown(results: dict, path: Path) -> None:
    lines = [
        "# Baseline Comparison — ShockEstimator Eval Set",
        "",
        "## Overall MAE",
        "",
        "> **Note:** Methods A (news_only) and B (social_only) require real RoBERTa scores",
        "> in the eval records. Run `scripts/score_shock.py` on the eval examples first,",
        "> then re-run this script to populate those columns.",
        "",
        "| Method | Overall MAE | N records |",
        "|--------|-------------|-----------|",
    ]
    for method, data in results.items():
        if data.get("note"):
            lines.append(f"| {method} | N/A (see note above) | 0 |")
        else:
            overall = data["overall_mae"]
            lines.append(
                f"| {method} | {_fmt(overall) if overall is not None else 'n/a'} | {data['n_records']} |"
            )

    lines += ["", "## Top-3 Worst Blocs per Method", ""]
    for method, data in results.items():
        if data.get("note"):
            lines.append(f"**{method}**: {data['note']}")
            lines.append("")
            continue
        per_bloc: dict[str, float] = data["per_bloc_mae"]
        if not per_bloc:
            lines.append(f"**{method}**: no data")
            lines.append("")
            continue
        worst = sorted(per_bloc.items(), key=lambda x: x[1], reverse=True)[:3]
        lines.append(f"**{method}**")
        lines.append("")
        lines.append("| Bloc | MAE |")
        lines.append("|------|-----|")
        for bloc, mae in worst:
            lines.append(f"| {bloc} | {_fmt(mae)} |")
        lines.append("")

    lines += ["*Generated by scripts/run_baseline_comparison.py*", ""]
    path.write_text("\n".join(lines), encoding="utf-8")
    log.info("Wrote markdown to %s", path)


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Three-way baseline comparison")
    parser.add_argument("--eval", default=str(_EVAL_PATH), help="Path to eval.jsonl")
    parser.add_argument(
        "--adapter", default=_DEFAULT_ADAPTER, help="Path to LoRA adapter"
    )
    parser.add_argument(
        "--skip-llm", action="store_true",
        help="Skip unified_llm method (no adapter needed)",
    )
    args = parser.parse_args()

    records = load_eval(Path(args.eval))

    results: dict[str, dict] = {}

    log.info("── Method A: news_only ──────────────────────────────────")
    maes_a, errs_a = run_news_only(records)
    results["news_only"] = aggregate(maes_a, errs_a)

    log.info("── Method B: social_only ────────────────────────────────")
    maes_b, errs_b = run_social_only(records)
    results["social_only"] = aggregate(maes_b, errs_b)

    if not args.skip_llm:
        adapter = Path(args.adapter)
        if not adapter.exists():
            log.warning(
                "Adapter not found at %s — skipping unified_llm. "
                "Set ADAPTER_PATH or pass --adapter.",
                adapter,
            )
        else:
            log.info("── Method C: unified_llm ────────────────────────────────")
            maes_c, errs_c = run_unified_llm(records, str(adapter))
            results["unified_llm"] = aggregate(maes_c, errs_c)

    # Save JSON
    _OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    _OUT_JSON.write_text(json.dumps(results, indent=2), encoding="utf-8")
    log.info("Wrote JSON to %s", _OUT_JSON)

    # Save markdown
    _OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    write_markdown(results, _OUT_MD)

    # Print summary table to stdout
    print(f"\n{'Method':<14}  {'Overall MAE':>11}  {'N':>5}")
    print("─" * 36)
    for method, data in results.items():
        mae_str = "N/A" if data.get("note") else _fmt(data["overall_mae"])
        print(f"{method:<14}  {mae_str:>11}  {data['n_records']:>5}")


if __name__ == "__main__":
    main()
