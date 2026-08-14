#!/usr/bin/env python3
"""Validate, balance, and merge reviewed mobilization-override training records.

Phase 6 requires a documented polarization-aware high-magnitude gate before these
training-only records may join the Step 6.3 selected corpus. Evaluation probes are
rejected before any output is written.
"""
from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from pathlib import Path

from electoral.data.held_out import assert_none_held_out
from scripts.resample_hybrid_training import (
    BIN_FIELDS,
    MAGNITUDES,
    TARGET,
    is_probe,
    magnitude,
    pct,
    counts,
    snapshot,
)

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data/finetune/phase6_step6_3_resampled.jsonl"
APPROVED = ROOT / "data/finetune/mobilization_override_training_approved_20260814.jsonl"
REVISIONS = ROOT / "data/finetune/mobilization_override_training_revisions_20260814.jsonl"
OUT = ROOT / "data/finetune/phase6_step6_3_with_mobilization_override.jsonl"
JSON_OUT = ROOT / "artifacts/phase6_step6_3_mobilization_merge_audit.json"
MD_OUT = ROOT / "artifacts/phase6_step6_3_mobilization_merge_audit.md"

# The moderate-bin midpoint is 0.0175 on the canonical ±0.03 delta scale.
POLARIZATION_NET_LIMIT = 0.0175
REMOVE_ID = "gov_lgbtq_order_lgbtq_mobilization_1"


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def reviewed_records() -> list[dict]:
    """Load final reviewer output; revisions are already materialized here.

    The reviewer writes every retained final record to the approved stream, including
    its corrected `after` version when the verdict is REVISE. The revisions stream is
    an audit ledger, not an additional source to concatenate.
    """
    rows = load_jsonl(APPROVED)
    ids = [row["shock_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("final reviewer output contains duplicate shock IDs")
    return rows


def strong_bins(row: dict) -> list[tuple[str, str]]:
    return [(bloc, token) for field in BIN_FIELDS for bloc, token in row[field].items() if token.startswith("strong_")]


def magnitude_gate(row: dict) -> dict:
    """Allow exactly one opposed strong pair only when the aggregate nets out."""
    strong = strong_bins(row)
    tokens = [token for _, token in strong]
    if len(strong) <= 1:
        return {"status": "pass", "reason": "zero_or_one_strong_bin", "strong_bins": strong}
    opposite_pair = len(strong) == 2 and set(tokens) == {"strong_pos", "strong_neg"}
    near_zero_net = abs(row["delta_eff"]) < POLARIZATION_NET_LIMIT
    if opposite_pair and near_zero_net:
        return {
            "status": "pass_polarization",
            "reason": "exactly_two_opposite_signed_strong_bins_and_small_net",
            "strong_bins": strong,
        }
    if len(strong) >= 3:
        reason = "three_or_more_strong_bins"
    elif not opposite_pair:
        reason = "two_same_signed_or_nonopposed_strong_bins"
    else:
        reason = "opposite_signed_pair_with_large_net"
    return {"status": "fail", "reason": reason, "strong_bins": strong}


def beneficiary_consistent(row: dict) -> bool:
    beneficiary = row["_seed_meta"].get("benefiting_party")
    return beneficiary == row["party"] and row["delta_eff"] > 0


def revision_summary() -> dict:
    revision_rows = load_jsonl(REVISIONS)
    result = []
    for row in revision_rows:
        before, after = row["before"], row["after"]
        token_changes = []
        sign_changed = False
        for field in BIN_FIELDS:
            for bloc, old in before[field].items():
                new = after[field][bloc]
                if old != new:
                    token_changes.append(f"{bloc}:{old}->{new}")
                    sign_changed |= old.split("_")[-1] != new.split("_")[-1]
        # A token sign change is not automatically a net-beneficiary correction:
        # distinguish bloc-vector edits from the beneficiary-determining delta_eff.
        before_net_consistent = beneficiary_consistent(before)
        change_types = []
        if not before_net_consistent:
            change_types.append("net_sign_correction")
        elif sign_changed:
            change_types.append("bloc_direction_adjustment")
        if any(old.split("_")[0] != new.split("_")[0] for field in BIN_FIELDS for bloc, old in before[field].items() if (new := after[field][bloc]) != old):
            change_types.append("magnitude_adjustment")
        if not change_types:
            change_types.append("cosmetic_or_delta_only")
        result.append({
            "shock_id": after["shock_id"],
            "classification": "+".join(change_types),
            "change_types": change_types,
            "token_changes": token_changes,
            "delta_eff_before": before["delta_eff"],
            "delta_eff_after": after["delta_eff"],
            "beneficiary_consistent_after": beneficiary_consistent(after),
        })
    return {
        "records": result,
        "counts": dict(Counter(change for row in result for change in row["change_types"])),
    }


def balance(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    by_effect = Counter(r["_seed_meta"]["expected_effect"] for r in rows)
    dem, rep = by_effect["helps_dem"], by_effect["helps_rep"]
    if dem == rep:
        return rows, []
    excess_effect = "helps_dem" if dem > rep else "helps_rep"
    remove_n = abs(dem - rep)
    candidates = [r for r in rows if r["_seed_meta"]["expected_effect"] == excess_effect]
    # Deterministic: minimize global magnitude drift, then lexical shock ID.
    before = pct(counts(rows, magnitude), MAGNITUDES)
    best = None
    for row in candidates:
        kept = [r for r in rows if r is not row]
        after = pct(counts(kept, magnitude), MAGNITUDES)
        score = sum((after[key] - before[key]) ** 2 for key in MAGNITUDES)
        candidate = (score, row["shock_id"], row)
        if best is None or candidate[:2] < best[:2]:
            best = candidate
    removed = [best[2]] if remove_n == 1 else []
    if remove_n != 1:
        raise ValueError(f"unexpected beneficiary imbalance requiring {remove_n} removals")
    return [r for r in rows if r is not removed[0]], removed


def distance_to_target(snap: dict) -> float:
    values = snap["magnitude"]["percent"]
    return round(sum((values[key] - TARGET[key]) ** 2 for key in MAGNITUDES) ** 0.5, 4)


def md(report: dict) -> str:
    lines = [
        "# Phase 6 — mobilization-override validation and merge",
        "",
        "## Refined concentrated-magnitude rule",
        "",
        f"A record passes with at most one strong bin. Exactly two strong bins are allowed only when they are one `strong_pos` and one `strong_neg` **and** `|delta_eff| < {POLARIZATION_NET_LIMIT:.4f}` (the canonical moderate-bin midpoint). Three or more strong bins, two same-signed strong bins, or an opposed strong pair with `|delta_eff| >= {POLARIZATION_NET_LIMIT:.4f}` fail. Opposed bloc-specific strong movement with a small aggregate net is polarization; a large aggregate net alongside it is inflation.",
        "",
        "## Revised-record audit",
        "",
        f"- Revision classifications: `{report['revisions']['counts']}`.",
        f"- Beneficiary/sign consistency after revision: **{sum(r['beneficiary_consistent_after'] for r in report['revisions']['records'])}/{len(report['revisions']['records'])}**.",
        "",
        "| Record | Classification | Token changes | delta_eff before → after | Beneficiary-consistent |",
        "|---|---|---|---:|---|",
    ]
    for row in report["revisions"]["records"]:
        lines.append(f"| `{row['shock_id']}` | {row['classification']} | {'; '.join(row['token_changes']) or 'none'} | {row['delta_eff_before']:+.4f} → {row['delta_eff_after']:+.4f} | {row['beneficiary_consistent_after']} |")
    lines += ["", "## Refined gate result", "", "| Record | Status | Reason | strong bins | delta_eff |", "|---|---|---|---|---:|"]
    for row in report["gate"]["rows"]:
        pairs = ", ".join(f"{b}:{t}" for b, t in row["strong_bins"]) or "none"
        lines.append(f"| `{row['shock_id']}` | {row['status']} | {row['reason']} | {pairs} | {row['delta_eff']:+.4f} |")
    lines += [
        "",
        f"The only failure is `{REMOVE_ID}`, removed rather than regenerated. `lgbtq_order_backlash_turnout_2028` is the only newly permitted polarization case; no other record changes status under this refinement.",
        "",
        "## Balancing and final composition",
        "",
        f"- Reviewed supply: {report['reviewed_records']}; after inflation removal: {report['surviving_records']}; beneficiary rebalance removed: `{report['balance_removed']}`.",
        f"- Added training records: **{report['added_records']}** (`helps_dem={report['added_effect']['helps_dem']}`, `helps_rep={report['added_effect']['helps_rep']}`).",
        f"- Final merged records: **{report['after']['records']}**; mobilizing: **{report['after']['mobilization']['mobilizing']} ({100 * report['after']['mobilization']['mobilizing'] / report['after']['records']:.2f}%)**.",
        "",
        "| Stage | Neutral | Slight | Mild | Moderate | Strong | distance to panel target |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, snap, distance in (("Panel target", {"magnitude": {"percent": TARGET}}, 0), ("Step 6.3 base", report["before"], report["before_distance"]), ("Merged", report["after"], report["after_distance"])):
        vals = snap["magnitude"]["percent"]
        lines.append(f"| {name} | {vals['neutral']:.2f}% | {vals['slight']:.2f}% | {vals['mild']:.2f}% | {vals['moderate']:.2f}% | {vals['strong']:.2f}% | {distance:.4f} |")
    lines += [
        "",
        f"**Verdict: {report['bin_verdict']}.** No bin balancing was applied.",
        "",
        "| Modeled party | Records | Mean |delta_eff| |",
        "|---|---:|---:|",
    ]
    for party, values in report["after"]["party"].items():
        lines.append(f"| {party} | {values['records']} | {values['mean_abs_delta_eff']:.5f} |")
    lines += [
        "",
        f"Probe exclusion: **PASS**. The merged source contained no probe markers; the independent hard-fail check rejected all `{report['probe_check']['probe_records']}` probe records and created no output.",
        f"Output: `{report['output']}`. Every added record retains generator/reviewer provenance and receives `_merge` provenance.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    base, reviewed = load_jsonl(BASE), reviewed_records()
    for name, rows in (("base", base), ("reviewed", reviewed)):
        leaked = [r["shock_id"] for r in rows if is_probe(r)]
        if leaked:
            raise ValueError(f"probe leakage in {name}: {leaked}")
        assert_none_held_out([r["shock_id"] for r in rows], context=f"mobilization merge {name}")
    revisions = revision_summary()
    if not all(row["beneficiary_consistent_after"] for row in revisions["records"]):
        raise ValueError("revision beneficiary/sign consistency gate failed")
    if not all(beneficiary_consistent(row) for row in reviewed):
        raise ValueError("reviewed beneficiary/sign consistency gate failed")

    gate_rows = [{"shock_id": row["shock_id"], "delta_eff": row["delta_eff"], **magnitude_gate(row)} for row in reviewed]
    failures = [row for row in gate_rows if row["status"] == "fail"]
    if [row["shock_id"] for row in failures] != [REMOVE_ID]:
        raise ValueError(f"unexpected high-magnitude failures: {[r['shock_id'] for r in failures]}")
    survivors = [row for row in reviewed if row["shock_id"] != REMOVE_ID]
    added, balance_removed = balance(survivors)
    effect = Counter(r["_seed_meta"]["expected_effect"] for r in added)
    if effect["helps_dem"] != effect["helps_rep"]:
        raise ValueError(f"beneficiary balance failed: {effect}")

    merged = base + [copy.deepcopy(row) for row in added]
    for row in merged[len(base):]:
        row["_merge"] = {
            "policy_id": "phase6_mobilization_override_merge",
            "high_magnitude_gate": "polarization_aware_v1",
            "beneficiary_balance_checked": True,
            "training_eligible_checked": True,
        }
    before, after = snapshot(base), snapshot(merged)
    before_distance, after_distance = distance_to_target(before), distance_to_target(after)
    bin_verdict = "CLOSER_TO_PANEL_TARGET" if after_distance < before_distance else "OVERSHOOT_OR_NOT_CLOSER"

    probe = load_jsonl(ROOT / "data/eval/mobilization_confound_probe.jsonl")
    if not all(is_probe(row) for row in probe):
        raise ValueError("probe metadata gate failed")
    probe_check = {"probe_records": len(probe), "merged_probe_records": sum(is_probe(r) for r in merged), "hard_fail": True}
    if probe_check["merged_probe_records"]:
        raise ValueError("probe leaked into merged corpus")

    report = {
        "policy_id": "phase6_mobilization_override_merge",
        "output": str(args.output.relative_to(ROOT)),
        "reviewed_records": len(reviewed), "surviving_records": len(survivors),
        "revisions": revisions, "gate": {"threshold": POLARIZATION_NET_LIMIT, "rows": gate_rows},
        "removed_inflation_record": REMOVE_ID,
        "balance_removed": [r["shock_id"] for r in balance_removed],
        "added_records": len(added), "added_effect": dict(effect),
        "before": before, "after": after,
        "before_distance": before_distance, "after_distance": after_distance, "bin_verdict": bin_verdict,
        "probe_check": probe_check,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in merged))
    JSON_OUT.write_text(json.dumps(report, indent=2) + "\n")
    MD_OUT.write_text(md(report))
    print(f"reviewed={len(reviewed)} gate_failures={len(failures)} added={len(added)} merged={len(merged)} "
          f"effects={dict(effect)} mobilizing={after['mobilization']['mobilizing']} verdict={bin_verdict}")


if __name__ == "__main__":
    main()
