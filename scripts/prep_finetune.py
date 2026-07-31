"""Stratified 80/20 train/eval split of the fine-tuning dataset.

Reads data/finetune/synthetic.jsonl (+ data/finetune/new_events.jsonl if present)
and writes data/finetune/train.jsonl + data/finetune/eval.jsonl.

Stratification preserves the democrat/republican ratio in both splits.
Reproducible via --seed (default 42).
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
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def stratified_split(
    records: list[dict],
    eval_fraction: float,
    seed: int,
) -> tuple[list[dict], list[dict]]:
    """Return (train, eval) with stratification by 'party' field."""
    rng = make_rng(derive_seed(seed, "prep_finetune"))
    by_party: dict[str, list[dict]] = {}
    for rec in records:
        party = rec.get("party", "unknown")
        by_party.setdefault(party, []).append(rec)

    train_all: list[dict] = []
    eval_all: list[dict] = []
    for party, group in sorted(by_party.items()):
        rng.shuffle(group)
        n_eval = max(1, round(len(group) * eval_fraction))
        eval_all.extend(group[:n_eval])
        train_all.extend(group[n_eval:])

    rng.shuffle(train_all)
    rng.shuffle(eval_all)
    return train_all, eval_all


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare fine-tuning train/eval split.")
    parser.add_argument(
        "--data-dir",
        default="data/finetune",
        help="Directory containing synthetic.jsonl and new_events.jsonl",
    )
    parser.add_argument(
        "--eval-fraction",
        type=float,
        default=0.20,
        help="Fraction of examples to reserve for evaluation (default 0.20)",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    data_dir = Path(args.data_dir)
    synthetic = _load_jsonl(data_dir / "synthetic.jsonl")
    new_events = _load_jsonl(data_dir / "new_events.jsonl")

    all_records = synthetic + new_events
    if not all_records:
        print(f"ERROR: no records found in {data_dir}", file=sys.stderr)
        return 1

    # Step 1.6: fail loudly if any input record's shock_id -- or a
    # grounding-injection source_shock, direct or indirect -- is a held-out
    # shock, rather than silently baking it into train.jsonl/eval.jsonl.
    # new_events.jsonl in particular could plausibly carry a hand-authored
    # REAL shock_id (unlike synthetic.jsonl's slugs), which is exactly the
    # direct-leak shape found in data/finetune/grounded_aligned.jsonl by
    # scripts/check_held_out_leakage.py.
    shock_refs: list[str] = []
    for rec in all_records:
        if isinstance(rec.get("shock_id"), str):
            shock_refs.append(rec["shock_id"])
        g = rec.get("_grounding") or {}
        if isinstance(g.get("grounded_from"), str):
            shock_refs.append(g["grounded_from"])
        g2 = rec.get("_grounding_v2") or {}
        if isinstance(g2.get("source_shock"), str):
            shock_refs.append(g2["source_shock"])
    assert_none_held_out(shock_refs, context="prep_finetune.py train/eval split input (synthetic.jsonl + new_events.jsonl)")

    train, eval_ = stratified_split(all_records, args.eval_fraction, args.seed)

    _write_jsonl(data_dir / "train.jsonl", train)
    _write_jsonl(data_dir / "eval.jsonl", eval_)

    by_party_train = {}
    by_party_eval = {}
    for rec in train:
        by_party_train[rec.get("party", "?")] = by_party_train.get(rec.get("party", "?"), 0) + 1
    for rec in eval_:
        by_party_eval[rec.get("party", "?")] = by_party_eval.get(rec.get("party", "?"), 0) + 1

    print(f"Total: {len(all_records)} ({len(synthetic)} synthetic + {len(new_events)} new_events)")
    print(f"Train: {len(train)} — {by_party_train}")
    print(f"Eval:  {len(eval_)} — {by_party_eval}")
    print(f"Written to {data_dir}/train.jsonl and {data_dir}/eval.jsonl")
    return 0


if __name__ == "__main__":
    sys.exit(main())
