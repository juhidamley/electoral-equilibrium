"""Evaluation metrics for LLM delta-bin predictions.

Three metrics:
  mae_in_delta_units      — Mean Absolute Error in numeric Δμ units via BIN_MIDPOINTS
  direction_accuracy      — Fraction of blocs where sign(pred) == sign(true)
  compute_eval_report     — Aggregate both over a list of (pred_bins, true_bins) pairs

No model loading — pure arithmetic on bin token strings.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

from electoral.core.types import BIN_MIDPOINTS, DELTA_BINS


def _sign(token: str) -> int:
    """Map a bin token to -1, 0, or +1."""
    v = BIN_MIDPOINTS[token]
    if v > 0.005:
        return 1
    if v < -0.005:
        return -1
    return 0


def mae_in_delta_units(
    pred_bins: dict[str, str],
    true_bins: dict[str, str],
) -> float:
    """Mean Absolute Error over shared blocs in numeric delta units.

    Each bin token is mapped to its BIN_MIDPOINTS midpoint before differencing.
    Only blocs present in both dicts contribute to the mean.
    """
    blocs = sorted(set(pred_bins) & set(true_bins))
    if not blocs:
        raise ValueError("pred_bins and true_bins share no common keys")
    for b in blocs:
        if pred_bins[b] not in BIN_MIDPOINTS:
            raise ValueError(f"pred_bins[{b!r}] = {pred_bins[b]!r} is not a valid delta bin")
        if true_bins[b] not in BIN_MIDPOINTS:
            raise ValueError(f"true_bins[{b!r}] = {true_bins[b]!r} is not a valid delta bin")
    errors = [abs(BIN_MIDPOINTS[pred_bins[b]] - BIN_MIDPOINTS[true_bins[b]]) for b in blocs]
    return sum(errors) / len(errors)


def direction_accuracy(
    pred_bins: dict[str, str],
    true_bins: dict[str, str],
) -> float:
    """Fraction of blocs where predicted direction matches true direction.

    Direction is mapped to {-1, 0, +1} using DELTA_BINS boundary at ±0.005.
    Neutral-neutral, positive-positive, and negative-negative are all counted correct.
    """
    blocs = sorted(set(pred_bins) & set(true_bins))
    if not blocs:
        raise ValueError("pred_bins and true_bins share no common keys")
    correct = sum(1 for b in blocs if _sign(pred_bins[b]) == _sign(true_bins[b]))
    return correct / len(blocs)


def per_stratum_mae(
    pred_bins: dict[str, str],
    true_bins: dict[str, str],
    races: list[str],
    religions: list[str],
    genders: list[str],
) -> dict[str, float]:
    """MAE broken down by stratum: race, religion, gender."""
    result: dict[str, float] = {}
    for stratum_name, stratum_blocs in [
        ("race", races),
        ("religion", religions),
        ("gender", genders),
    ]:
        sub_pred = {b: pred_bins[b] for b in stratum_blocs if b in pred_bins}
        sub_true = {b: true_bins[b] for b in stratum_blocs if b in true_bins}
        if sub_pred and sub_true:
            result[stratum_name] = mae_in_delta_units(sub_pred, sub_true)
    return result


def compute_eval_report(
    examples: list[tuple[dict[str, str], dict[str, str]]],
) -> dict[str, object]:
    """Aggregate MAE and direction accuracy over a list of (pred_bins, true_bins) pairs.

    Returns a dict with keys: mae, direction_accuracy, n_examples.
    """
    if not examples:
        raise ValueError("examples list is empty")

    maes = [mae_in_delta_units(pred, true) for pred, true in examples]
    dir_accs = [direction_accuracy(pred, true) for pred, true in examples]

    return {
        "mae": sum(maes) / len(maes),
        "direction_accuracy": sum(dir_accs) / len(dir_accs),
        "n_examples": len(examples),
    }


# ── Model evaluation ─────────────────────────────────────────────────────────


@dataclass
class EvalResult:
    shock_id: str
    mae_per_bloc: dict[str, float]  # bloc → absolute error in delta units
    mean_mae: float  # mean of mae_per_bloc; nan if output was invalid JSON
    output_json: str  # raw model output string


def _flatten_pred(pred: dict) -> dict[str, str]:
    """Flatten nested stratum output or pass through flat dict."""
    if "delta_bins_race" in pred:
        return {
            **pred.get("delta_bins_race", {}),
            **pred.get("delta_bins_religion", {}),
            **pred.get("delta_bins_gender", {}),
        }
    return {k: v for k, v in pred.items() if isinstance(v, str)}


def _flatten_true(record: dict) -> dict[str, str]:
    """Extract flat {bloc: bin} ground truth from a finetune record."""
    if "delta_bins_race" in record:
        return {
            **record.get("delta_bins_race", {}),
            **record.get("delta_bins_religion", {}),
            **record.get("delta_bins_gender", {}),
        }
    return record.get("delta_bins", {})


def _per_bloc_mae(pred_flat: dict[str, str], true_flat: dict[str, str]) -> dict[str, float]:
    """Absolute error per bloc in numeric delta units."""
    result: dict[str, float] = {}
    for b in sorted(set(pred_flat) & set(true_flat)):
        p, t = pred_flat[b], true_flat[b]
        if p in BIN_MIDPOINTS and t in BIN_MIDPOINTS:
            result[b] = abs(BIN_MIDPOINTS[p] - BIN_MIDPOINTS[t])
    return result


def evaluate_model(
    adapter_path: str | Path,
    eval_path: str | Path,
) -> list[EvalResult]:
    """Load a LoRA adapter and evaluate it on an eval JSONL file.

    Uses mlx_lm (M5 MacBook backend). For HPC evaluation, use _eval_mae()
    in trainer.py which runs the HuggingFace transformers pipeline instead.

    Args:
        adapter_path: Path to the saved LoRA adapter directory (output of train_mlx).
        eval_path:    Path to the eval JSONL file.

    Returns:
        One EvalResult per eval record. Records where the model output is not
        valid JSON have mae_per_bloc={} and mean_mae=nan.
    """
    try:
        import mlx_lm
    except ImportError as exc:
        raise ImportError(
            "mlx_lm is required for evaluate_model on M5. " "Install with: pip install mlx-lm"
        ) from exc

    # Lazy import to avoid circular dependency (trainer imports mae_in_delta_units from here).
    from electoral.llm.trainer import format_prompt, load_jsonl  # noqa: PLC0415

    adapter_path = Path(adapter_path)
    model, tokenizer = mlx_lm.load(str(adapter_path))
    records = load_jsonl(Path(eval_path))

    results: list[EvalResult] = []
    for rec in records:
        shock_id = str(rec.get("shock_id", rec.get("shock", "unknown")))
        prompt = format_prompt(rec)
        output_str: str = mlx_lm.generate(
            model, tokenizer, prompt=prompt, max_tokens=256, verbose=False
        )

        try:
            pred = json.loads(output_str.strip())
            if not isinstance(pred, dict):
                raise ValueError("model output is not a JSON object")
            pred_flat = _flatten_pred(pred)
            pred_flat = {k: v for k, v in pred_flat.items() if v in BIN_MIDPOINTS}
            true_flat = _flatten_true(rec)
            mae_blocs = _per_bloc_mae(pred_flat, true_flat)
            mean_mae = sum(mae_blocs.values()) / len(mae_blocs) if mae_blocs else float("nan")
        except (json.JSONDecodeError, ValueError):
            mae_blocs = {}
            mean_mae = float("nan")

        results.append(
            EvalResult(
                shock_id=shock_id,
                mae_per_bloc=mae_blocs,
                mean_mae=mean_mae,
                output_json=output_str,
            )
        )

    return results


def print_eval_summary(results: list[EvalResult]) -> None:
    """Print a table of mean MAE per bloc sorted descending, plus summary stats."""
    if not results:
        print("No eval results.")
        return

    # Aggregate per-bloc errors across all results
    bloc_errors: dict[str, list[float]] = {}
    valid_count = 0
    finite_maes: list[float] = []

    for r in results:
        if r.mae_per_bloc:
            valid_count += 1
        if not math.isnan(r.mean_mae):
            finite_maes.append(r.mean_mae)
        for bloc, err in r.mae_per_bloc.items():
            bloc_errors.setdefault(bloc, []).append(err)

    bloc_means = {b: sum(errs) / len(errs) for b, errs in bloc_errors.items()}
    sorted_blocs = sorted(bloc_means.items(), key=lambda x: x[1], reverse=True)

    print(f"\n{'Bloc':<22}  {'Mean MAE':>9}  {'N':>4}")
    print("-" * 40)
    for bloc, mean in sorted_blocs:
        n = len(bloc_errors[bloc])
        print(f"{bloc:<22}  {mean:>9.4f}  {n:>4}")
    print("-" * 40)

    overall = sum(finite_maes) / len(finite_maes) if finite_maes else float("nan")
    frac = valid_count / len(results)
    print(f"\nOverall mean MAE :  {overall:.4f}  (across {len(finite_maes)} valid outputs)")
    print(f"Valid JSON outputs: {valid_count}/{len(results)} ({100 * frac:.1f}%)")


def _validate_token(token: str) -> str:
    if token not in DELTA_BINS:
        raise ValueError(f"{token!r} is not in DELTA_BINS: {DELTA_BINS}")
    return token


def main(argv: list[str] | None = None) -> int:
    """CLI: score predictions against ground truth from two JSONL files.

    Each line in --predictions and --targets is a JSON object with a 'delta_bins' dict.
    """
    parser = argparse.ArgumentParser(description="Evaluate LLM delta-bin predictions.")
    parser.add_argument("--predictions", required=True, help="JSONL of predicted records")
    parser.add_argument("--targets", required=True, help="JSONL of target records")
    parser.add_argument("--output", help="Write JSON report to this path (default: stdout)")
    args = parser.parse_args(argv)

    def _load(path: str) -> list[dict]:
        records = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    preds = _load(args.predictions)
    targets = _load(args.targets)

    if len(preds) != len(targets):
        print(
            f"ERROR: predictions ({len(preds)}) and targets ({len(targets)}) "
            "must have the same length",
            file=sys.stderr,
        )
        return 1

    examples = []
    for p, t in zip(preds, targets):
        pred_bins = p.get("delta_bins", p)
        true_bins = t.get("delta_bins", t)
        examples.append((pred_bins, true_bins))

    report = compute_eval_report(examples)
    out_str = json.dumps(report, indent=2)

    if args.output:
        Path(args.output).write_text(out_str, encoding="utf-8")
        print(f"Report written to {args.output}")
    else:
        print(out_str)

    return 0


if __name__ == "__main__":
    sys.exit(main())
