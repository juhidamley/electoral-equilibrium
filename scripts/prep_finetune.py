"""Build a reproducible, guarded stratified fine-tuning split.

Evaluation probes and configured held-out shocks are hard failures.  This module never
silently filters them because that would conceal a data-leakage mistake.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from electoral.core.rng import derive_seed, make_rng
from electoral.data.held_out import assert_none_held_out


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def stratified_split(records: list[dict], eval_fraction: float, seed: int) -> tuple[list[dict], list[dict]]:
    """Return reproducibly shuffled train/eval splits stratified by modeled party."""
    rng = make_rng(derive_seed(seed, "prep_finetune"))
    by_party: dict[str, list[dict]] = {}
    for rec in records:
        by_party.setdefault(rec.get("party", "unknown"), []).append(rec)
    train_all: list[dict] = []
    eval_all: list[dict] = []
    for _, group in sorted(by_party.items()):
        rng.shuffle(group)
        n_eval = max(1, round(len(group) * eval_fraction))
        eval_all.extend(group[:n_eval])
        train_all.extend(group[n_eval:])
    rng.shuffle(train_all)
    rng.shuffle(eval_all)
    return train_all, eval_all


def _guard_records(records: list[dict]) -> None:
    probes = [
        rec.get("_probe", {}).get("probe_id", rec.get("shock_id", "<unknown>"))
        for rec in records
        if rec.get("_provenance", {}).get("training_eligible") is False
        or rec.get("_provenance", {}).get("dataset_role") == "evaluation_probe"
        or rec.get("_probe")
    ]
    if probes:
        raise ValueError(
            "evaluation-probe record(s) supplied to training builder; these are permanently "
            f"held out and cannot enter train/eval: {probes}"
        )
    shock_refs: list[str] = []
    for rec in records:
        if isinstance(rec.get("shock_id"), str):
            shock_refs.append(rec["shock_id"])
        for key in ("_grounding", "_grounding_v2"):
            grounding = rec.get(key) or {}
            source = grounding.get("grounded_from") or grounding.get("source_shock")
            if isinstance(source, str):
                shock_refs.append(source)
    assert_none_held_out(shock_refs, context="prep_finetune.py train/eval split input")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare guarded fine-tuning train/eval split.")
    parser.add_argument("--data-dir", default="data/finetune", help="Legacy input/output directory.")
    parser.add_argument("--input", type=Path, default=None, help="Single explicit JSONL corpus input.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Fresh directory for train.jsonl/eval.jsonl.")
    parser.add_argument("--eval-fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if not 0 < args.eval_fraction < 1:
        parser.error("--eval-fraction must be strictly between 0 and 1")

    legacy_dir = Path(args.data_dir)
    if args.input:
        all_records = _load_jsonl(args.input)
        source_desc = str(args.input)
    else:
        synthetic = _load_jsonl(legacy_dir / "synthetic.jsonl")
        new_events = _load_jsonl(legacy_dir / "new_events.jsonl")
        all_records = synthetic + new_events
        source_desc = f"{legacy_dir}/synthetic.jsonl + new_events.jsonl"
    if not all_records:
        print(f"ERROR: no records found in {source_desc}", file=sys.stderr)
        return 1
    _guard_records(all_records)

    train, eval_ = stratified_split(all_records, args.eval_fraction, args.seed)
    def record_key(rec: dict) -> str:
        return json.dumps(rec, sort_keys=True, separators=(",", ":"))

    if {record_key(r) for r in train} & {record_key(r) for r in eval_}:
        raise ValueError("train/eval duplicate-record overlap detected")
    output_dir = args.output_dir or legacy_dir
    _write_jsonl(output_dir / "train.jsonl", train)
    _write_jsonl(output_dir / "eval.jsonl", eval_)

    def party_counts(rows: list[dict]) -> dict[str, int]:
        return {party: sum(r.get("party") == party for r in rows) for party in ("democrat", "republican")}
    print(f"GUARDS=PASS probe_exclusion=PASS held_out_shocks=PASS source={source_desc}")
    print(f"Total: {len(all_records)} Train: {len(train)} {party_counts(train)} Eval: {len(eval_)} {party_counts(eval_)} overlap=0")
    print(f"Written to {output_dir}/train.jsonl and {output_dir}/eval.jsonl")
    return 0


if __name__ == "__main__":
    sys.exit(main())
