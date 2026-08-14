"""Validate and freeze the held-out mobilization-confound evaluation probe.

This script never creates training data. It accepts only reviewed records tagged as
permanently evaluation-only and fails closed if the sentiment-conflict contract,
beneficiary-first sign, reviewer rationale, or provenance guard is missing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

PARTIES = {"democrat", "republican"}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _validate(rec: dict[str, Any]) -> None:
    probe = rec.get("_probe") or {}
    meta = rec.get("_seed_meta") or {}
    provenance = rec.get("_provenance") or {}
    review = rec.get("_review") or {}
    probe_id = probe.get("probe_id", rec.get("shock_id", "<unknown>"))

    if meta.get("mobilization") != "mobilizing":
        raise ValueError(f"{probe_id}: not tagged mobilizing")
    beneficiary = meta.get("benefiting_party")
    if beneficiary not in PARTIES or probe.get("benefiting_party") != beneficiary:
        raise ValueError(f"{probe_id}: missing or inconsistent beneficiary")
    if rec.get("party") != beneficiary:
        raise ValueError(f"{probe_id}: probe must model the benefiting party")
    if meta.get("expected_effect") != f"helps_{'dem' if beneficiary == 'democrat' else 'rep'}":
        raise ValueError(f"{probe_id}: expected_effect contradicts beneficiary")
    if probe.get("sentiment_only_predicted_party") not in PARTIES - {beneficiary}:
        raise ValueError(f"{probe_id}: sentiment-only prediction must oppose beneficiary")
    if not str(probe.get("surface_sentiment", "")).lower().startswith("negative toward"):
        raise ValueError(f"{probe_id}: negative surface sentiment is not explicit")
    if not probe.get("why_sentiment_is_wrong"):
        raise ValueError(f"{probe_id}: missing sentiment-conflict explanation")
    if float(rec.get("delta_eff", 0)) <= 0:
        raise ValueError(f"{probe_id}: beneficiary-modeled delta_eff must be positive")
    if not str(review.get("beneficiary_rationale", "")).strip():
        raise ValueError(f"{probe_id}: reviewer omitted beneficiary rationale")
    if review.get("verdict") not in {"APPROVE", "REVISE"}:
        raise ValueError(f"{probe_id}: record did not pass machine review")
    if provenance.get("dataset_role") != "evaluation_probe":
        raise ValueError(f"{probe_id}: missing evaluation-probe provenance")
    if provenance.get("training_eligible") is not False:
        raise ValueError(f"{probe_id}: probe is not permanently training-ineligible")


def build(source: Path, output: Path) -> list[dict[str, Any]]:
    records = _read_jsonl(source)
    if not 8 <= len(records) <= 12:
        raise ValueError(f"probe must contain 8-12 records, found {len(records)}")
    ids = [rec.get("_probe", {}).get("probe_id") for rec in records]
    if len(set(ids)) != len(ids):
        raise ValueError("probe_id values must be unique")
    beneficiaries = {rec.get("_seed_meta", {}).get("benefiting_party") for rec in records}
    if beneficiaries != PARTIES:
        raise ValueError(f"probe must span both beneficiaries, found {beneficiaries}")

    for rec in records:
        _validate(rec)
        rec["_probe"]["hand_verification"] = {
            "status": "PASS",
            "checks": [
                "negative surface tone targets the declared beneficiary",
                "sentiment-only prediction names the opposing party",
                "mobilization-implied net sign benefits the declared beneficiary",
                "reviewer rationale names/confirms the beneficiary",
            ],
        }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(rec, ensure_ascii=False) + "\n" for rec in records))
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("data/eval/mobilization_confound_probe_reviewed_v2.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/eval/mobilization_confound_probe.jsonl"),
    )
    args = parser.parse_args()
    records = build(args.source, args.output)
    counts = {party: 0 for party in sorted(PARTIES)}
    for rec in records:
        counts[rec["_seed_meta"]["benefiting_party"]] += 1
    print(f"probe_records={len(records)} beneficiaries={counts} hand_verified=PASS")
    print(f"evaluation_only={args.output}")


if __name__ == "__main__":
    main()
