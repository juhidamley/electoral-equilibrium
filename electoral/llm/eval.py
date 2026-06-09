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
import sys
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
