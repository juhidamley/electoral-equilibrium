#!/usr/bin/env python3
"""Generate adapter predictions for Phase-1 ground-truth scoring.

This is inference only. It predicts each requested shock once from configs/shocks.json,
always on the Democrat-loyalty scale expected by ground_truth_accuracy.py. Ground-truth
values and trust flags select the evaluation scope but are never inserted into prompts.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from electoral.core.types import BIN_MIDPOINTS, CANONICAL_GENDERS, CANONICAL_RACES, CANONICAL_RELIGIONS
from electoral.llm.inference import load_model, predict_delta_bins
from electoral.metrics.ground_truth_accuracy import normalize_ces, normalize_exit_poll, normalize_panel

GT_FILES = {
    "panel": (Path("data/ground_truth/panel_deltas.json"), normalize_panel),
    "ces": (Path("data/ground_truth/ces_deltas.json"), normalize_ces),
    "exit_poll": (Path("data/ground_truth/exit_poll_deltas.json"), normalize_exit_poll),
}


def _shock_ids(source: tuple[Path, object]) -> set[str]:
    path, normalizer = source
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {cell.shock_id for cell in normalizer(raw)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source", choices=["all", *GT_FILES], default="all")
    parser.add_argument("--base-model", default="mistralai/Mistral-7B-v0.3")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quantized", action="store_true", help="Load base model in 4-bit on CUDA.")
    args = parser.parse_args()

    sources = GT_FILES.values() if args.source == "all" else [GT_FILES[args.source]]
    wanted = set().union(*(_shock_ids(source) for source in sources))
    shocks = {
        str(row["id"]): row
        for row in json.loads(Path("configs/shocks.json").read_text(encoding="utf-8"))
    }
    missing = sorted(wanted - shocks.keys())
    if missing:
        raise ValueError(f"ground-truth shock(s) missing from configs/shocks.json: {missing}")

    model, tokenizer = load_model(
        adapter_path=args.adapter,
        base_model=args.base_model,
        use_quantization=args.quantized,
    )
    predictions: dict[str, dict] = {}
    for shock_id in sorted(wanted):
        event = shocks[shock_id]
        bins = predict_delta_bins(
            shock_text=str(event["description"]),
            party="democrat",
            model=model,
            tokenizer=tokenizer,
            seed=args.seed,
            base_model=args.base_model,
        )
        predictions[shock_id] = {
            "party": "democrat",
            "deltas_race": {bloc: BIN_MIDPOINTS[bins[bloc]] for bloc in CANONICAL_RACES},
            "deltas_religion": {bloc: BIN_MIDPOINTS[bins[bloc]] for bloc in CANONICAL_RELIGIONS},
            "deltas_gender": {bloc: BIN_MIDPOINTS[bins[bloc]] for bloc in CANONICAL_GENDERS},
            "_provenance": {
                "adapter": args.adapter,
                "base_model": args.base_model,
                "seed": args.seed,
                "ground_truth_not_in_prompt": True,
            },
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(predictions, indent=2) + "\n", encoding="utf-8")
    print(f"predictions={len(predictions)} source={args.source} adapter={args.adapter} out={args.out}")


if __name__ == "__main__":
    main()
