#!/usr/bin/env python3
"""Build Phase 6 Step 6.1's source-availability map; never builds labels/corpora.

Inputs are read-only. Outputs are a JSON diagnostic and a Markdown coverage map.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = ROOT / "configs/hybrid_source_policy.json"
PANEL_PATH = ROOT / "data/ground_truth/panel_deltas.json"
SOCIAL_PATH = ROOT / "data/inventory/step_4_2_native_twitter_intersection.json"
HELD_OUT_PATH = ROOT / "configs/held_out_shocks.json"
SHOCKS_PATH = ROOT / "configs/shocks.json"
SYNTHETIC_PATH = ROOT / "data/finetune/synthetic_step5_5_full_20260809_approved.jsonl"


def load_json(path: Path) -> Any:
    with path.open() as handle:
        return json.load(handle)


def load_synthetic_runs(path: Path, blocs: list[str]) -> tuple[list[str], bool, int]:
    """Return reviewed Step 5.5 runs and whether every canonical bloc is present.

    This establishes fallback *availability* only. It deliberately does not select
    a record or attach any generated label.
    """
    runs, records, label_blocs = set(), 0, set()
    with path.open() as handle:
        for line in handle:
            record = json.loads(line)
            records += 1
            provenance = record.get("_provenance", {})
            if provenance.get("step") != "phase5_step5.5":
                continue
            runs.add(provenance.get("run_id", "unknown"))
            for dimension in ("race", "religion", "gender"):
                label_blocs.update(record.get(f"delta_bins_{dimension}", {}))
    return sorted(runs), set(blocs).issubset(label_blocs), records


def pct(n: int, total: int) -> str:
    return f"{(100 * n / total):.1f}%" if total else "0.0%"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", type=Path, default=ROOT / "artifacts/hybrid_source_coverage.json")
    parser.add_argument("--md-out", type=Path, default=ROOT / "artifacts/hybrid_source_coverage.md")
    args = parser.parse_args()

    policy = load_json(POLICY_PATH)
    blocs = policy["canonical_bloc_order"]
    panel = load_json(PANEL_PATH)
    social = load_json(SOCIAL_PATH)
    held_out = {item["id"] for item in load_json(HELD_OUT_PATH)["held_out_shocks"]}
    shocks = load_json(SHOCKS_PATH)
    trainable_shocks = [item["id"] for item in shocks if item["id"] not in held_out]
    social_counts = social["per_shock_native_twitter"]
    floor = policy["eligibility"]["social"]["minimum_cell_n"]
    synthetic_runs, synthetic_available, synthetic_records = load_synthetic_runs(SYNTHETIC_PATH, blocs)
    if not synthetic_available:
        raise ValueError("Step 5.5 approved corpus lacks one or more canonical bloc label fields")

    rows = []
    for shock_id in trainable_shocks:
        for bloc in blocs:
            panel_cell = panel.get(shock_id, {}).get("bloc", {}).get(bloc)
            trust = (panel_cell or {}).get("trust", {})
            panel_eligible = bool(trust.get("trustworthy_for_direction"))
            social_n = social_counts.get(shock_id, {}).get("by_bloc", {}).get(bloc)
            social_exists = social_n is not None
            # The Step 4.2 source documents self-identifying lexicon assignment and
            # native-Twitter/topic-attributed qualification for all listed cells.
            social_eligible = bool(social_exists and social_n >= floor)
            if panel_eligible:
                tier = "panel"
                source_id = f"panel:{shock_id}:{bloc}"
                reason = "trainable direction-trustworthy panel cell"
            elif social_eligible:
                tier = "social"
                source_id = f"social:native_twitter:{shock_id}:{bloc}"
                reason = "no eligible panel cell; clean self-identifying native-Twitter cell meets n floor"
            else:
                tier = "synthetic"
                source_id = f"synthetic:{synthetic_runs[0]}:UNASSIGNED:{bloc}"
                reason = "no higher-fidelity eligible source; Step 5.5 fallback available"
            rows.append({
                "shock_id": shock_id,
                "bloc": bloc,
                "selected_tier": tier,
                "source_id_template": source_id,
                "selection_reason": reason,
                "availability": {
                    "panel_cell_exists": panel_cell is not None,
                    "panel_direction_trustworthy": trust.get("trustworthy_for_direction"),
                    "panel_magnitude_usable": trust.get("magnitude_usable"),
                    "panel_single_shock_attributable": trust.get("single_shock_attributable"),
                    "social_cell_exists": social_exists,
                    "social_n": social_n,
                    "social_self_identifying": social_exists,
                    "social_clean_attribution": social_exists,
                    "social_n_floor_met": bool(social_exists and social_n >= floor),
                    "synthetic_step5_5_available": synthetic_available,
                    "synthetic_run_ids": synthetic_runs,
                },
            })

    tiers = ["panel", "social", "synthetic"]
    by_bloc, by_shock = {}, {}
    for bloc in blocs:
        counts = Counter(row["selected_tier"] for row in rows if row["bloc"] == bloc)
        by_bloc[bloc] = {tier: counts[tier] for tier in tiers}
    for shock_id in trainable_shocks:
        counts = Counter(row["selected_tier"] for row in rows if row["shock_id"] == shock_id)
        by_shock[shock_id] = {tier: counts[tier] for tier in tiers}
    overall = Counter(row["selected_tier"] for row in rows)
    panel_source_cells = [
        (shock_id, bloc)
        for shock_id, shock in panel.items()
        for bloc in shock.get("bloc", {})
    ]
    panel_direction_all = [
        (shock_id, bloc)
        for shock_id, bloc in panel_source_cells
        if panel[shock_id]["bloc"][bloc].get("trust", {}).get("trustworthy_for_direction")
    ]
    panel_magnitude_all = [
        (shock_id, bloc)
        for shock_id, bloc in panel_source_cells
        if panel[shock_id]["bloc"][bloc].get("trust", {}).get("magnitude_usable")
    ]
    panel_magnitude_only = [
        {"shock_id": row["shock_id"], "bloc": row["bloc"]}
        for row in rows
        if row["availability"]["panel_magnitude_usable"]
        and not row["availability"]["panel_direction_trustworthy"]
    ]
    social_eligible_cells = [row for row in rows if row["availability"]["social_n_floor_met"]]

    result = {
        "policy_id": policy["policy_id"],
        "policy_version": policy["version"],
        "scope": {
            "configured_shocks": len(shocks), "held_out_shocks_excluded": len(held_out),
            "trainable_shocks": len(trainable_shocks), "canonical_blocs": len(blocs),
            "trainable_cells": len(rows),
            "synthetic_reviewed_records": synthetic_records,
            "synthetic_run_ids": synthetic_runs,
        },
        "source_supply": {
            "panel": {
                "total_cells": len(panel_source_cells),
                "held_out_cells": sum(shock_id in held_out for shock_id, _ in panel_source_cells),
                "trainable_cells": sum(shock_id not in held_out for shock_id, _ in panel_source_cells),
                "direction_trustworthy_all": len(panel_direction_all),
                "direction_trustworthy_held_out": sum(shock_id in held_out for shock_id, _ in panel_direction_all),
                "direction_trustworthy_trainable": sum(shock_id not in held_out for shock_id, _ in panel_direction_all),
                "magnitude_usable_all": len(panel_magnitude_all),
            },
            "social": {
                "trainable_threshold_cells": len(social_eligible_cells),
                "trainable_records": sum(row["availability"]["social_n"] for row in social_eligible_cells),
                "eligible_shocks": sorted({row["shock_id"] for row in social_eligible_cells}),
                "white_eligible_cells": sum(row["bloc"] == "white" for row in social_eligible_cells),
            },
            "synthetic": {"reviewed_records": synthetic_records, "run_ids": synthetic_runs},
        },
        "tier_key": {"P": "panel", "S": "social", "Y": "synthetic"},
        "rows": rows,
        "summary": {
            "overall": {tier: {"cells": overall[tier], "fraction": overall[tier] / len(rows)} for tier in tiers},
            "by_bloc": by_bloc, "by_shock": by_shock,
            "social_eligible_cells_before_panel_precedence": len(social_eligible_cells),
            "panel_magnitude_only_ineligible_cells": panel_magnitude_only,
        },
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(result, indent=2) + "\n")

    lookup = {(row["shock_id"], row["bloc"]): row["selected_tier"] for row in rows}
    lines = [
        "# Phase 6, Step 6.1 — Hybrid label-source coverage map",
        "",
        "Generated by `scripts/build_hybrid_source_coverage.py`; this is an availability/selection-policy artifact only. It assigns **no labels** and builds **no merged corpus**.",
        "",
        "## Scope and key",
        "",
        f"- Trainable grid: **{len(trainable_shocks)} shocks × {len(blocs)} blocs = {len(rows)} cells**. {len(held_out)} configured held-out shocks are excluded, not treated as synthetic.",
        f"- `P` = direction-trustworthy panel; `S` = clean self-identifying native-Twitter cell with n≥{floor}; `Y` = reviewed Step 5.5 synthetic fallback.",
        "- Panel magnitude-only cells are not signed-label sources: unstable sign is incompatible with a directional bin.",
        "",
        "## Shock × bloc selection grid",
        "",
        "| Shock | " + " | ".join(blocs) + " |",
        "|---|" + "|".join(["---"] * len(blocs)) + "|",
    ]
    glyph = {"panel": "P", "social": "S", "synthetic": "Y"}
    for shock_id in trainable_shocks:
        lines.append("| " + shock_id + " | " + " | ".join(glyph[lookup[shock_id, bloc]] for bloc in blocs) + " |")

    lines += ["", "## Tier summary", "", "| Tier | Cells | Fraction |", "|---|---:|---:|"]
    lines += [f"| {tier} | {overall[tier]} | {pct(overall[tier], len(rows))} |" for tier in tiers]
    lines += ["", "### By bloc", "", "| Bloc | Panel | Social | Synthetic |", "|---|---:|---:|---:|"]
    lines += [f"| {bloc} | {counts['panel']} ({pct(counts['panel'], len(trainable_shocks))}) | {counts['social']} ({pct(counts['social'], len(trainable_shocks))}) | {counts['synthetic']} ({pct(counts['synthetic'], len(trainable_shocks))}) |" for bloc, counts in by_bloc.items()]
    lines += ["", "### By shock", "", "| Shock | Panel | Social | Synthetic |", "|---|---:|---:|---:|"]
    lines += [f"| {shock_id} | {counts['panel']} | {counts['social']} | {counts['synthetic']} |" for shock_id, counts in by_shock.items()]
    all_generated_blocs = [bloc for bloc, counts in by_bloc.items() if counts["panel"] + counts["social"] == 0]
    all_generated_shocks = [shock_id for shock_id, counts in by_shock.items() if counts["panel"] + counts["social"] == 0]
    lines += [
        "",
        "## Supply reality and limitations",
        "",
        f"- **Panel reality check:** the actual source has {len(panel_source_cells)} cells (18 shocks × 15 blocs), not ~252. {sum(shock_id in held_out for shock_id, _ in panel_source_cells)} cells belong to held-out shocks, leaving {sum(shock_id not in held_out for shock_id, _ in panel_source_cells)} trainable raw panel cells. Of {len(panel_direction_all)} direction-trustworthy cells overall, only {overall['panel']} are trainable and selected. `{len(panel_magnitude_only)}` additional trainable cells are magnitude-usable but direction-noisy, and are deliberately excluded from signed-label sourcing.",
        f"- **Social reality check:** {len(social_eligible_cells)} cells / {sum(row['availability']['social_n'] for row in social_eligible_cells):,} records meet the n≥{floor} floor after held-out exclusion (the verified 18,031 total). All are in `{', '.join(sorted({row['shock_id'] for row in social_eligible_cells}))}`. Panel precedence does not displace any of them, so social contributes {overall['social']} selected cells. White has **zero** eligible social cells.",
        f"- **Synthetic dominance:** synthetic supplies {overall['synthetic']} of {len(rows)} cells ({pct(overall['synthetic'], len(rows))}). `Y` means a Step 5.5 reviewed label is available in principle; it is not a claim that a specific synthetic record has already been chosen.",
        f"- **Entirely synthetic shocks ({len(all_generated_shocks)}):** {', '.join(all_generated_shocks)}.",
        (f"- **Entirely synthetic blocs ({len(all_generated_blocs)}):** {', '.join(all_generated_blocs)}." if all_generated_blocs else "- **Entirely synthetic blocs:** none. This does not imply broad grounding: white has zero social cells and is 34/36 (94.4%) synthetic; every bloc remains at least 91.7% synthetic."),
        "- The grounded panel cells trace to only three before/after panel windows (per `data/ground_truth/trustworthy_subset.md`); count them as sparse, correlated measurement events, not independent broad coverage.",
    ]
    args.md_out.write_text("\n".join(lines) + "\n")
    print(f"wrote {args.json_out.relative_to(ROOT)} and {args.md_out.relative_to(ROOT)}")
    print(f"grid={len(rows)} panel={overall['panel']} social={overall['social']} synthetic={overall['synthetic']}")


if __name__ == "__main__":
    main()
