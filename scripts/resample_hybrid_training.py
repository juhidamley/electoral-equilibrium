#!/usr/bin/env python3
"""Phase 6 Step 6.3: audit and minimally resample the approved training records.

This does not construct panel/social records and never reads evaluation probes as input.
The only record-level correction is a constrained partisan downsample. The measured
magnitude distribution is an invariant, not a balancing target.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable

from electoral.core.rng import derive_seed, make_rng
from electoral.data.held_out import assert_none_held_out

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "data/finetune/synthetic_step5_5_full_20260809_approved.jsonl"
DEFAULT_OUTPUT = ROOT / "data/finetune/phase6_step6_3_resampled.jsonl"
DEFAULT_JSON = ROOT / "artifacts/phase6_step6_3_balance_audit.json"
DEFAULT_MD = ROOT / "artifacts/phase6_step6_3_balance_audit.md"
COVERAGE = ROOT / "artifacts/hybrid_source_coverage.json"
PANEL = ROOT / "data/ground_truth/panel_deltas.json"
SEEDS = ROOT / "configs/synthetic_events.json"

BIN_FIELDS = ("delta_bins_race", "delta_bins_religion", "delta_bins_gender")
SIGNED_BINS = (
    "strong_neg", "mod_neg", "mild_neg", "slight_neg", "neutral",
    "slight_pos", "mild_pos", "mod_pos", "strong_pos",
)
MAGNITUDES = ("neutral", "slight", "mild", "moderate", "strong")
TARGET = {"neutral": 34.0, "slight": 31.0, "mild": 28.0, "moderate": 6.0, "strong": 1.0}


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def is_probe(record: dict) -> bool:
    provenance = record.get("_provenance", {})
    return bool(
        record.get("_probe")
        or provenance.get("training_eligible") is False
        or provenance.get("dataset_role") == "evaluation_probe"
    )


def bins(record: dict) -> list[str]:
    return [token for field in BIN_FIELDS for token in record[field].values()]


def magnitude(token: str) -> str:
    prefix = token.split("_", 1)[0]
    return "moderate" if prefix == "mod" else prefix


def counts(records: list[dict], transform: Callable[[str], str] = lambda x: x) -> Counter:
    return Counter(transform(token) for record in records for token in bins(record))


def pct(counter: Counter, order: tuple[str, ...]) -> dict[str, float]:
    total = sum(counter.values())
    return {key: round(100.0 * counter[key] / total, 4) if total else 0.0 for key in order}


def grouped(records: list[dict], key: Callable[[dict], str]) -> dict[str, list[dict]]:
    result: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        result[key(record)].append(record)
    return dict(result)


def snapshot(records: list[dict]) -> dict:
    by_party = grouped(records, lambda r: r["party"])
    by_effect = grouped(records, lambda r: r["_seed_meta"]["expected_effect"])
    by_mob = grouped(records, lambda r: r["_seed_meta"]["mobilization"])
    shock_counts = Counter(record["shock_id"] for record in records)
    bloc_counts = Counter(bloc for record in records for field in BIN_FIELDS for bloc in record[field])

    def group_stats(groups: dict[str, list[dict]]) -> dict:
        return {
            key: {
                "records": len(rows),
                "magnitude_percent": pct(counts(rows, magnitude), MAGNITUDES),
                "signed_bin_percent": pct(counts(rows), SIGNED_BINS),
                "mean_delta_eff": round(sum(r["delta_eff"] for r in rows) / len(rows), 7),
                "mean_abs_delta_eff": round(sum(abs(r["delta_eff"]) for r in rows) / len(rows), 7),
                "mobilization": dict(sorted(Counter(r["_seed_meta"]["mobilization"] for r in rows).items())),
                "valence": dict(sorted(Counter(r["_seed_meta"]["valence"] for r in rows).items())),
            }
            for key, rows in sorted(groups.items())
        }

    return {
        "records": len(records),
        "label_tokens": len(records) * 15,
        "shock_ids": {
            "unique": len(shock_counts),
            "minimum": min(shock_counts.values()),
            "median": sorted(shock_counts.values())[len(shock_counts) // 2],
            "maximum": max(shock_counts.values()),
            "volume_frequency": dict(sorted(Counter(shock_counts.values()).items())),
            "repeated": dict(sorted((k, v) for k, v in shock_counts.items() if v > 1)),
        },
        "bloc_records": dict(sorted(bloc_counts.items())),
        "magnitude": {
            "counts": dict(counts(records, magnitude)),
            "percent": pct(counts(records, magnitude), MAGNITUDES),
            "target_percent": TARGET,
        },
        "party": group_stats(by_party),
        "expected_effect": group_stats(by_effect),
        "mobilization": {key: len(rows) for key, rows in sorted(by_mob.items())},
    }


def selection_score(kept: list[dict], before_mag: dict[str, float]) -> float:
    """Prefer party symmetry while heavily penalizing corpus-level magnitude drift."""
    parties = grouped(kept, lambda r: r["party"])
    party_mag = {p: pct(counts(rows, magnitude), MAGNITUDES) for p, rows in parties.items()}
    party_signed = {p: pct(counts(rows), SIGNED_BINS) for p, rows in parties.items()}
    global_mag = pct(counts(kept, magnitude), MAGNITUDES)
    drift = sum((global_mag[k] - before_mag[k]) ** 2 for k in MAGNITUDES)
    mag_gap = sum((party_mag["democrat"][k] - party_mag["republican"][k]) ** 2 for k in MAGNITUDES)
    signed_gap = sum((party_signed["democrat"][k] - party_signed["republican"][k]) ** 2 for k in SIGNED_BINS)
    # Lexicographic in practice: preservation of the deliberately empirical
    # corpus-wide magnitude mix comes first; party-specific shape is only a
    # tie-break among candidates that preserve it equally well.  Otherwise a
    # superficially symmetric party sample could distort the protected scale.
    return 10_000.0 * drift + mag_gap + 0.20 * signed_gap


def balance_parties(records: list[dict], seed: int, trials: int) -> tuple[list[dict], list[dict]]:
    """Mirror count and mobilization strata without touching mobilizing records.

    Supply is 5 records high for helps_rep/consolidating and 13 high for
    helps_rep/depressing. Removing exactly those excesses makes modeled party,
    expected effect, and each party's mobilization strata symmetric at once.
    """
    remove_pools = {
        "consolidating": [
            r for r in records
            if r["_seed_meta"]["expected_effect"] == "helps_rep"
            and r["_seed_meta"]["mobilization"] == "consolidating"
        ],
        "depressing": [
            r for r in records
            if r["_seed_meta"]["expected_effect"] == "helps_rep"
            and r["_seed_meta"]["mobilization"] == "depressing"
        ],
    }
    remove_n = {"consolidating": 5, "depressing": 13}
    rng = make_rng(derive_seed(seed, "phase6_step6_3_party_symmetry"))
    before_mag = pct(counts(records, magnitude), MAGNITUDES)
    best_score = math.inf
    best_removed: list[dict] = []
    all_ids = {id(r): r for r in records}
    for _ in range(trials):
        removed = []
        for stratum in ("consolidating", "depressing"):
            indexes = rng.choice(len(remove_pools[stratum]), size=remove_n[stratum], replace=False)
            removed.extend(remove_pools[stratum][int(i)] for i in indexes)
        removed_ids = {id(r) for r in removed}
        kept = [r for rid, r in all_ids.items() if rid not in removed_ids]
        score = selection_score(kept, before_mag)
        tie_key = tuple(sorted(r["shock_id"] for r in removed))
        best_tie = tuple(sorted(r["shock_id"] for r in best_removed))
        if score < best_score or (math.isclose(score, best_score) and tie_key < best_tie):
            best_score, best_removed = score, removed
    removed_ids = {id(r) for r in best_removed}
    return [r for r in records if id(r) not in removed_ids], best_removed


def panel_weight_policy() -> dict:
    coverage = json.loads(COVERAGE.read_text())
    panel = json.loads(PANEL.read_text())
    cells = [row for row in coverage["rows"] if row["selected_tier"] == "panel"]
    by_window: dict[str, list[dict]] = defaultdict(list)
    for row in cells:
        shock = panel[row["shock_id"]]
        bracket = shock["bracket"]
        window = f'{bracket["before_wave"]}->{bracket["after_wave"]}'
        by_window[window].append(row)
    synthetic_cells = sum(row["selected_tier"] == "synthetic" for row in coverage["rows"])
    target_share = 0.03
    panel_total_weight = target_share * synthetic_cells / (1.0 - target_share)
    cluster_weight = panel_total_weight / len(by_window)
    weights = {}
    for window, rows in sorted(by_window.items()):
        for row in rows:
            cell_id = f'panel:{row["shock_id"]}:{row["bloc"]}'
            weights[cell_id] = round(cluster_weight / len(rows), 6)
    return {
        "status": "specified_not_materialized",
        "reason": "The repository has eight selected panel labels but no valid full-record panel adapter: modeled_party and required mobilization annotations are absent. Applying weights to nonexistent model rows would be fictitious.",
        "natural_cell_share_percent": round(100 * len(cells) / (synthetic_cells + len(cells)), 4),
        "target_effective_share_percent": 3.0,
        "rationale": "A modest doubling from 1.53% to 3%; cluster-normalization prevents five correlated travel-ban cells from counting as five independent windows.",
        "synthetic_coverage_cells": synthetic_cells,
        "panel_cells": len(cells),
        "independent_windows": len(by_window),
        "future_per_cell_loss_weights": weights,
    }


def render_md(report: dict) -> str:
    before, after = report["before"], report["after"]
    lines = [
        "# Phase 6, Step 6.3 — targeted corpus balance audit",
        "",
        "The approved Step 5.5 source is audited before correction. Bin magnitudes are protected, never balanced toward uniform. The output contains training-eligible synthetic records only; the held-out mobilization probe is rejected on sight.",
        "",
        "## Material corpus reality",
        "",
        f"- Input: **{before['records']} approved synthetic records / {before['label_tokens']} bloc labels**.",
        "- Social: **0 labels**, as established in Step 6.2.",
        "- Panel: **8 eligible label cells**, but they are not full training records and cannot honestly be materialized until modeled-party and mobilization annotations exist. They are not silently represented as synthetic records.",
        f"- Selected record output: **{after['records']} synthetic records**. This artifact is therefore not falsely described as a completed three-tier merge.",
        "",
        "## 1. Per-shock / seed volume before",
        "",
        f"- {before['shock_ids']['unique']} distinct generated shock IDs; min/median/max records = {before['shock_ids']['minimum']}/{before['shock_ids']['median']}/{before['shock_ids']['maximum']}.",
        f"- Volume frequency (`records per shock`: `number of shocks`): `{before['shock_ids']['volume_frequency']}`.",
        f"- Repeated IDs: `{before['shock_ids']['repeated']}`.",
        "- No shock exceeds 3/338 (0.89%), so no shock cap is justified.",
        "- **Seed-level volume is not auditable in the frozen run:** its provenance omitted a stable seed ID/index. `_seed_meta` contains only broad taxonomy fields shared by multiple seeds. The 64-seed registry is 32 Democratic / 32 Republican, not 21/21; generated record balance must be checked independently. Future generation should persist `seed_index` and a seed hash.",
        "",
        "## 2. Per-bloc coverage before",
        "",
        "Every canonical bloc occurs once in every full synthetic record:",
        "",
        "| Bloc | Records |",
        "|---|---:|",
    ]
    lines.extend(f"| `{bloc}` | {n} |" for bloc, n in before["bloc_records"].items())
    lines += [
        "",
        "No bloc is starved; no bloc resampling was applied.",
        "",
        "## 3. Party symmetry before and correction",
        "",
        "| Axis | Group | Records | Mean |delta_eff| | Neutral | Slight | Mild | Moderate | Strong |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for axis in ("party", "expected_effect"):
        for key, value in before[axis].items():
            m = value["magnitude_percent"]
            lines.append(f"| Before {axis} | `{key}` | {value['records']} | {value['mean_abs_delta_eff']:.5f} | {m['neutral']:.2f}% | {m['slight']:.2f}% | {m['mild']:.2f}% | {m['moderate']:.2f}% | {m['strong']:.2f}% |")
    for axis in ("party", "expected_effect"):
        for key, value in after[axis].items():
            m = value["magnitude_percent"]
            lines.append(f"| After {axis} | `{key}` | {value['records']} | {value['mean_abs_delta_eff']:.5f} | {m['neutral']:.2f}% | {m['slight']:.2f}% | {m['mild']:.2f}% | {m['moderate']:.2f}% | {m['strong']:.2f}% |")
    lines += [
        "",
        "Correction: remove only the excess 5 `helps_rep/consolidating` and 13 `helps_rep/depressing` records. This makes modeled-party and beneficiary counts 160/160 and mirrors mobilization strata (109 consolidating, 33 depressing, 18 mobilizing per modeled party). A deterministic constrained search preserves the corpus-level magnitude mix first, using party magnitude/sign discrepancy only as a tie-break. No mobilizing record is removed.",
        "",
        "Count symmetry is exact, but shape symmetry is not: after selection, modeled-Democratic versus modeled-Republican labels differ by 6.79pp in neutral share and their mean |delta_eff| differs by 0.00109. The `helps_dem` / `helps_rep` magnitude table above likewise retains a 2.42pp moderate-bin difference. These are flagged rather than erased: further selective trimming to homogenize them would make the empirical overall magnitude mix a secondary target.",
        "",
        "### Valence composition (record counts)",
        "",
        "| Stage / axis | Group | Valence counts |",
        "|---|---|---|",
    ]
    for stage, snap in (("Before", before), ("After", after)):
        for axis in ("party", "expected_effect"):
            for key, value in snap[axis].items():
                lines.append(f"| {stage} {axis} | `{key}` | `{value['valence']}` |")
    lines += [
        "",
        "## 4. Mobilization representation",
        "",
        f"- Before: `{before['mobilization']}`.",
        f"- After: `{after['mobilization']}`.",
        f"- Mobilizing share after: **{100 * after['mobilization']['mobilizing'] / after['records']:.2f}%** (36 records).",
        "- Count alone is not the decisive defect: all 36 are same-sign impeachment cases. They do not teach the sentiment-conflicting override. Before retraining, generate and independently review **64 new training-eligible records** (32 per beneficiary) over at least 8 non-probe archetypes through the fixed prompts. This would yield 100/384 = 26.0% mobilizing records without using or paraphrasing the held-out probe. Re-audit bins afterward.",
        "",
        "## 5. Panel placement",
        "",
        f"- Natural coverage-cell share: **{report['panel_weight_policy']['natural_cell_share_percent']:.2f}%** (8/522). Leaving it natural is nearly decorative; naive duplication risks memorizing three windows.",
        "- Defensible middle: target **3% effective label-loss mass**, allocated equally across the three independent panel windows and then equally among cells within each window.",
        "- This policy is **specified but not materialized** because no valid panel full-record adapter currently exists. The future weights are in the JSON audit; training must not pretend isolated cells are complete 15-label records.",
        "",
        "## Protected magnitude distribution",
        "",
        "| Stage | Neutral | Slight | Mild | Moderate | Strong |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, snap in (("Measured-panel target", {"magnitude": {"percent": TARGET}}), ("Before", before), ("After party correction", after)):
        m = snap["magnitude"]["percent"]
        lines.append(f"| {name} | {m['neutral']:.2f}% | {m['slight']:.2f}% | {m['mild']:.2f}% | {m['moderate']:.2f}% | {m['strong']:.2f}% |")
    lines += [
        "",
        f"Maximum correction drift is **{report['bin_invariant']['max_drift_pp']:.3f} percentage points**. The distribution remains small-shift dominated and very far from uniform 20%/category. Note that the frozen approved source was already more neutral and less mild/moderate/strong than the 34/31/28/6/1 target; Step 6.3 does not rewrite labels to force the target.",
        "",
        "## Final composition",
        "",
        f"- Records selected now: **{after['records']} synthetic, 0 panel records, 0 social records**.",
        "- Eligible measured labels awaiting a valid adapter: **8 panel cells across 3 windows**.",
        f"- Output: `{report['output']}`.",
        f"- Removed-record IDs and complete before/after distributions are queryable in `{DEFAULT_JSON.relative_to(ROOT)}`.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--md-out", type=Path, default=DEFAULT_MD)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trials", type=int, default=20_000)
    args = parser.parse_args()

    records = load_jsonl(args.source)
    leaked = [r.get("shock_id", "<unknown>") for r in records if is_probe(r)]
    if leaked:
        raise ValueError(f"evaluation probe(s) supplied to Step 6.3 training resampler: {leaked}")
    assert_none_held_out([r["shock_id"] for r in records], context="Phase 6 Step 6.3 input")
    if any(len(bins(r)) != 15 for r in records):
        raise ValueError("every synthetic training record must carry exactly 15 bloc labels")

    before = snapshot(records)
    selected, removed = balance_parties(records, args.seed, args.trials)
    after = snapshot(selected)
    assert after["party"]["democrat"]["records"] == after["party"]["republican"]["records"] == 160
    assert after["expected_effect"]["helps_dem"]["records"] == after["expected_effect"]["helps_rep"]["records"] == 160
    assert after["mobilization"]["mobilizing"] == before["mobilization"]["mobilizing"] == 36

    before_pct = before["magnitude"]["percent"]
    after_pct = after["magnitude"]["percent"]
    drift = {key: round(after_pct[key] - before_pct[key], 4) for key in MAGNITUDES}
    max_drift = max(abs(value) for value in drift.values())
    if max_drift > 0.5:
        raise ValueError(f"protected magnitude distribution drifted by {max_drift:.3f}pp")

    output_records = []
    for record in selected:
        copied = copy.deepcopy(record)
        copied["_resampling"] = {
            "policy_id": "phase6_step6_3_targeted_balance",
            "policy_version": "1.0.0",
            "source": str(args.source.relative_to(ROOT)),
            "party_symmetry_selected": True,
            "bin_distribution_protected": True,
            "probe_exclusion_checked": True,
        }
        output_records.append(copied)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        for record in output_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    seed_config = json.loads(SEEDS.read_text())["seeds"]
    report = {
        "policy_id": "phase6_step6_3_targeted_balance",
        "source": str(args.source.relative_to(ROOT)),
        "output": str(args.output.relative_to(ROOT)),
        "seed_registry": {
            "seeds": len(seed_config),
            "party": dict(Counter(s["party"] for s in seed_config)),
            "expected_effect": dict(Counter(s["expected_effect"] for s in seed_config)),
            "frozen_record_seed_id_available": False,
        },
        "before": before,
        "correction": {
            "axis": "party_symmetry",
            "removed_records": len(removed),
            "removed_by_stratum": {
                f"{effect}:{mobilization}": count
                for (effect, mobilization), count in sorted(
                    Counter(
                        (r["_seed_meta"]["expected_effect"], r["_seed_meta"]["mobilization"])
                        for r in removed
                    ).items()
                )
            },
            "removed_shock_ids": sorted(r["shock_id"] for r in removed),
            "no_shock_cap_applied": True,
            "no_bloc_resampling_applied": True,
            "no_bin_resampling_applied": True,
            "no_mobilizing_records_removed": True,
        },
        "after": after,
        "bin_invariant": {"drift_pp": drift, "max_drift_pp": round(max_drift, 4), "uniform_is_regression": True},
        "panel_weight_policy": panel_weight_policy(),
        "probe_exclusion": {"input_probe_records": 0, "hard_fail": True},
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2) + "\n")
    args.md_out.write_text(render_md(report))
    print(
        f"before={len(records)} after={len(selected)} removed={len(removed)} "
        f"party=160/160 mobilizing={after['mobilization']['mobilizing']} "
        f"max_bin_drift_pp={max_drift:.4f} output={args.output.relative_to(ROOT)}"
    )


if __name__ == "__main__":
    main()
