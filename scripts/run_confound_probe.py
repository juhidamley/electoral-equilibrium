#!/usr/bin/env python3
"""Run the permanently held-out mobilization-confound probe against one adapter.

The probe is evaluation-only.  This script reads the probe JSONL and never reads
training data or configs/shocks.json.  Because the inference helper returns
per-bloc bins rather than the completion's scalar ``delta_eff``, it defines the
model net-direction proxy as the arithmetic mean of the 15 canonical bin
midpoints.  The proxy is reported alongside every vector and is used only for
this deliberately balanced sign test.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from electoral.core.types import BIN_MIDPOINTS, CANONICAL_GENDERS, CANONICAL_RACES, CANONICAL_RELIGIONS, DELTA_BINS
from electoral.llm.inference import load_model, predict_delta_bins

PARTIES = frozenset({"democrat", "republican"})
BLOCS = tuple(CANONICAL_RACES + CANONICAL_RELIGIONS + CANONICAL_GENDERS)
CORRECT = "MOBILIZATION-CORRECT"
FALLBACK = "SENTIMENT-FALLBACK"
OTHER = "OTHER/AMBIGUOUS"


def read_probe(path: Path) -> list[dict[str, Any]]:
    """Read and fail closed on the probe's explicit two-answer contract."""
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not records:
        raise ValueError(f"probe is empty: {path}")
    for record in records:
        probe = record.get("_probe") or {}
        meta = record.get("_seed_meta") or {}
        provenance = record.get("_provenance") or {}
        probe_id = probe.get("probe_id", record.get("shock_id", "<unknown>"))
        beneficiary = probe.get("benefiting_party")
        sentiment_party = probe.get("sentiment_only_predicted_party")
        correct_sign = probe.get("known_correct_delta_eff_sign")
        if beneficiary not in PARTIES or meta.get("benefiting_party") != beneficiary:
            raise ValueError(f"{probe_id}: beneficiary is missing or inconsistent")
        if record.get("party") != beneficiary:
            raise ValueError(f"{probe_id}: modeled party must be the beneficiary")
        if sentiment_party not in PARTIES - {beneficiary}:
            raise ValueError(f"{probe_id}: sentiment answer must name the opposing party")
        if correct_sign not in {"positive", "negative"}:
            raise ValueError(f"{probe_id}: known correct modeled-party sign is missing")
        if provenance.get("dataset_role") != "evaluation_probe" or provenance.get("training_eligible") is not False:
            raise ValueError(f"{probe_id}: not permanently evaluation-only")
    return records


def net_direction(bins: dict[str, str]) -> tuple[str, float]:
    """Return modeled-party sign from the unweighted mean of all 15 bin midpoints."""
    missing = [bloc for bloc in BLOCS if bloc not in bins]
    invalid = {bloc: bins[bloc] for bloc in BLOCS if bloc in bins and bins[bloc] not in DELTA_BINS}
    if missing or invalid:
        raise ValueError(f"prediction must contain valid bins for all blocs; missing={missing}, invalid={invalid}")
    mean_delta = sum(BIN_MIDPOINTS[bins[bloc]] for bloc in BLOCS) / len(BLOCS)
    if mean_delta > 0:
        return "positive", mean_delta
    if mean_delta < 0:
        return "negative", mean_delta
    return "neutral", mean_delta


def verdict_for_prediction(record: dict[str, Any], bins: dict[str, str]) -> dict[str, Any]:
    """Classify a parsed bin vector without model I/O; unit-testable by fixtures."""
    predicted_sign, net_delta_proxy = net_direction(bins)
    probe = record["_probe"]
    correct_sign = probe["known_correct_delta_eff_sign"]
    # Since probe records model their declared beneficiary, sentiment's opposing
    # party means the opposite sign on this modeled-party scale.
    sentiment_sign = "negative" if correct_sign == "positive" else "positive"
    verdict = CORRECT if predicted_sign == correct_sign else FALLBACK if predicted_sign == sentiment_sign else OTHER
    return {
        "probe_id": probe["probe_id"],
        "shock_id": record["shock_id"],
        "event": record["description"],
        "benefiting_party": probe["benefiting_party"],
        "correct_sign": correct_sign,
        "sentiment_implied_party": probe["sentiment_only_predicted_party"],
        "sentiment_sign_on_modeled_party_scale": sentiment_sign,
        "model_predicted_sign": predicted_sign,
        "model_net_delta_proxy": net_delta_proxy,
        "verdict": verdict,
        "predicted_bins": {bloc: bins[bloc] for bloc in BLOCS},
    }


def score_probe(records: list[dict[str, Any]], predict: Callable[[dict[str, Any]], dict[str, str]]) -> list[dict[str, Any]]:
    return [verdict_for_prediction(record, predict(record)) for record in records]


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    overall = Counter(row["verdict"] for row in rows)
    by_party = {
        party: dict(Counter(row["verdict"] for row in rows if row["benefiting_party"] == party))
        for party in sorted(PARTIES)
    }
    return {"n": len(rows), "overall": dict(overall), "by_beneficiary": by_party}


def run_self_test() -> None:
    fixture = {
        "shock_id": "fixture",
        "description": "Negative tone, but backlash helps the modeled party.",
        "party": "democrat",
        "_seed_meta": {"benefiting_party": "democrat"},
        "_provenance": {"dataset_role": "evaluation_probe", "training_eligible": False},
        "_probe": {
            "probe_id": "fixture_probe",
            "benefiting_party": "democrat",
            "sentiment_only_predicted_party": "republican",
            "known_correct_delta_eff_sign": "positive",
        },
    }
    correct = {bloc: "slight_pos" for bloc in BLOCS}
    fallback = {bloc: "slight_neg" for bloc in BLOCS}
    correct_verdict = verdict_for_prediction(fixture, correct)["verdict"]
    fallback_verdict = verdict_for_prediction(fixture, fallback)["verdict"]
    if correct_verdict != CORRECT or fallback_verdict != FALLBACK:
        raise AssertionError(f"self-test failed: correct={correct_verdict}, fallback={fallback_verdict}")
    print(f"SELF_TEST=PASS correct={correct_verdict} fallback={fallback_verdict}")


def markdown(rows: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    lines = ["# Mobilization confound probe", "", "## Summary", "", "```json", json.dumps(summary, indent=2), "```", "", "## Per-record results", ""]
    lines.append("| Probe | Beneficiary | Correct sign | Sentiment sign | Model sign | Net proxy | Verdict |")
    lines.append("|---|---|---|---|---|---:|---|")
    for row in rows:
        lines.append(
            f"| `{row['probe_id']}` | {row['benefiting_party']} | {row['correct_sign']} | "
            f"{row['sentiment_sign_on_modeled_party_scale']} | {row['model_predicted_sign']} | "
            f"{row['model_net_delta_proxy']:+.5f} | {row['verdict']} |"
        )
        lines.extend(["", f"**Event:** {row['event']}", "", "**Predicted bins:** `" + "; ".join(f"{k}={v}" for k, v in row["predicted_bins"].items()) + "`", ""])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", required=False, help="PEFT adapter directory")
    parser.add_argument("--probe", type=Path, default=Path("data/eval/mobilization_confound_probe.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/mobilization_confound_probe_results.json"))
    parser.add_argument("--base-model", default="mistralai/Mistral-7B-v0.3")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        run_self_test()
        return
    if not args.adapter:
        parser.error("--adapter is required unless --self-test is supplied")
    adapter_path = Path(args.adapter)
    if not (adapter_path.is_dir() and (adapter_path / "adapter_config.json").is_file()):
        raise FileNotFoundError(
            f"adapter is not a local PEFT directory with adapter_config.json: {adapter_path}"
        )
    records = read_probe(args.probe)
    model, tokenizer = load_model(str(adapter_path), args.base_model, use_quantization=False)

    def predict(record: dict[str, Any]) -> dict[str, str]:
        return predict_delta_bins(
            shock_text=str(record["description"]), party=str(record["party"]), model=model,
            tokenizer=tokenizer, seed=args.seed, base_model=args.base_model,
        )

    rows = score_probe(records, predict)
    summary = summarize(rows)
    payload = {
        "probe": str(args.probe), "adapter": args.adapter, "base_model": args.base_model,
        "seed": args.seed, "net_direction_definition": "unweighted mean of 15 canonical bin midpoints", 
        "summary": summary, "records": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    args.out.with_suffix(".md").write_text(markdown(rows, summary), encoding="utf-8")
    print(f"probe_records={summary['n']} overall={summary['overall']} by_beneficiary={summary['by_beneficiary']}")
    print(f"out={args.out} markdown={args.out.with_suffix('.md')}")


if __name__ == "__main__":
    main()
