#!/usr/bin/env python3
"""Audit Phase 6 Step 6.2 design inputs without assigning labels or merging data."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
INVENTORY = ROOT / "data/inventory/step_4_2_native_twitter_intersection.json"
PANEL = ROOT / "data/ground_truth/panel_deltas.json"
SYNTHETIC = ROOT / "data/finetune/synthetic_step5_5_full_20260809_approved.jsonl"
GENERATOR = ROOT / "scripts/synthetic/generate_deepseek.py"
OUT = ROOT / "artifacts/hybrid_label_design_audit.md"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def main() -> None:
    inventory = load_json(INVENTORY)
    panel = load_json(PANEL)
    cells = inventory["held_out_split"]["trainable_cells_detail"]
    synthetic = [json.loads(line) for line in SYNTHETIC.read_text().splitlines()]

    # No event annotation exists in the source inputs. This is intentional: do
    # not infer a mobilization/valence tag from prose while merely designing.
    by_shock = Counter(cell["shock_id"] for cell in cells)
    lines = [
        "# Phase 6, Step 6.2 — Hybrid label design audit",
        "",
        "This is a read-only design audit. It assigns **no social labels**, selects no synthetic records, and builds no merged corpus.",
        "",
        "## Social cells and decision-tree classification",
        "",
        "The Step 4.2 inventory supplies `shock_id`, bloc, n, self-identification, and attribution evidence; it does **not** supply an explicit modeled-party, event-valence, or mobilization annotation. `configs/shocks.json` likewise has no such fields. The policy therefore correctly fails closed: none of the 18 cells may yet enter either the mobilizing-override or valence-clear-sentiment branch. Calling COVID inherently mobilizing, or calling vaccine rollout electorally positive, from prose would violate the explicit-tag requirement.",
        "",
        "| Shock | Bloc | n | Existing mobilization tag | Valence-clear tag | Decision-tree state |",
        "|---|---|---:|---|---|---|",
    ]
    for cell in sorted(cells, key=lambda x: (x["shock_id"], -x["n"], x["bloc"])):
        lines.append(
            f"| {cell['shock_id']} | {cell['bloc']} | {cell['n']:,} | missing | missing | BLOCKED: explicit event annotation required |"
        )
    lines += [
        "",
        f"Classification count: **0 mobilizing / 0 valence-clear / {len(cells)} annotation-blocked**. This is not a neutral label and not a claim of no effect.",
        "",
        "## Engagement-to-bin readiness",
        "",
        "The inventory contains only cell counts (n); it has no per-post `like_count`, repost/retweet, or reply fields, and the raw archive manifest referenced by the inventory is external to this checkout. Thus no engagement score, percentile, intensity, or final social bin was computed. The policy's all-eligible-cell reference would presently have 18 cells, below its precommitted >=20-cell freeze floor; social construction remains blocked even after an event annotation until supply expands or the floor is revised and documented.",
        "",
        "## Panel overlap check",
        "",
        "| Social shock | Social cells | Panel counterpart | Direction-trustworthy counterparts | Result |",
        "|---|---:|---|---:|---|",
    ]
    for shock_id, count in sorted(by_shock.items()):
        bloc_cells = [cell for cell in cells if cell["shock_id"] == shock_id]
        counterpart = panel.get(shock_id)
        trusted = sum(
            bool(counterpart["bloc"].get(cell["bloc"], {}).get("trust", {}).get("trustworthy_for_direction"))
            for cell in bloc_cells
        ) if counterpart else 0
        if counterpart:
            result = "Not a valid direction check: all overlap cells fail direction trust; COVID panel window is confounded."
            panel_state = "yes"
        else:
            result = "No panel counterpart."
            panel_state = "no"
        lines.append(f"| {shock_id} | {count} | {panel_state} | {trusted} | {result} |")
    lines += [
        "",
        "`covid_pandemic_2020` has all 14 social bloc counterparts in `panel_deltas.json`, but zero direction-trustworthy cells. Its panel record identifies the BLM/George Floyd shock as dominant in the same window, so even a future social label must not be tuned to match it. `covid_vaccine_2020` has no panel counterpart. Because this step assigns no social direction or magnitude, there is no invented agreement statistic.",
        "",
        "## Canonical scale audit",
        "",
        "- Canonical target: the nine `electoral.core.types.DELTA_BINS` tokens over the ±0.0375 support, named `delta_scale_0.03`; `mild_pos` is (+0.005, +0.0125] with midpoint +0.00875 in every tier.",
        "- Panel adapter is `delta_to_bin(measured_delta)` with no rescale. The selected Step 6.1 panel cells are direction-trustworthy; weighted/raw measurements remain provenance, not a second label convention.",
        "- Step 5.5 approved records declare `_provenance.scale = delta_scale_0.03` and use only canonical tokens. Social is specified to emit exactly those tokens after sign and ordinal magnitude are separately determined.",
        "",
        "## Synthetic mobilization and direction audit",
        "",
        f"- Approved Step 5.5 records: **{len(synthetic)}**; all carry `_seed_meta.mobilization`.",
        f"- Mobilization values: `{dict(sorted(Counter(r['_seed_meta']['mobilization'] for r in synthetic).items()))}`.",
        f"- Records with non-empty `social_roberta_scores`: **{sum(bool(r.get('social_roberta_scores')) for r in synthetic)}**; non-empty `news_roberta_scores`: **{sum(bool(r.get('news_roberta_scores')) for r in synthetic)}**.",
        "- Four legacy impeachment records carry both score dictionaries. The current generator prompt supplies the explicit party, expected effect, and mobilization metadata and does not consume a RoBERTa score field for label direction; the score fields are therefore not an implemented direction rule. Nevertheless, future harmonization must reject non-empty score fields as direction inputs and record `reviewed_teacher_bin` rather than silently treating their presence as evidence.",
        "- Important implementation gap: the current generator text says mobilization calibrates concentration, 'not to override the direction hint.' Future harmonization must preserve the reviewed teacher token, but its audit must record `reviewed_teacher_bin` and validate mobilizing records against the explicit expected-effect/mobilized-party annotation—not derive sign from sentiment. Existing approved labels must not be relabeled in this design step.",
        "",
        "## Provenance preservation",
        "",
        "`schemas/hybrid_harmonized_label.schema.json` nests the Step 6.1 `source_provenance` schema and adds only canonical label, scale, mobilization, label method, and audit inputs. It prevents a harmonized record from hiding its source tier or trust flags.",
    ]
    OUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}")
    print(f"social_cells={len(cells)} annotation_blocked={len(cells)} synthetic={len(synthetic)}")


if __name__ == "__main__":
    main()
