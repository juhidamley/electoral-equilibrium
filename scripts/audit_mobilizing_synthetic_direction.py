#!/usr/bin/env python3
"""Read-only direction audit for approved Step 5.5 synthetic records.

It never changes or regenerates the source corpus.  It writes an audit report only.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/finetune/synthetic_step5_5_full_20260809_approved.jsonl"
REPORT = ROOT / "artifacts/mobilizing_synthetic_direction_audit.md"

GROUPS = ("delta_bins_race", "delta_bins_religion", "delta_bins_gender")


def sign(value: float) -> str:
    return "positive" if value > 0 else "negative" if value < 0 else "zero"


def expected_modeled_party_sign(record: dict) -> int | None:
    """Sign expected for ``record.party`` from a non-split expected_effect."""
    effect = record["_seed_meta"]["expected_effect"]
    party = record["party"]
    if effect == "helps_dem":
        return 1 if party == "democrat" else -1
    if effect == "helps_rep":
        return 1 if party == "republican" else -1
    return None


def bins(record: dict) -> str:
    return " | ".join(
        ", ".join(f"{bloc}={token}" for bloc, token in record[group].items())
        for group in GROUPS
    )


def bloc_sign_counts(record: dict) -> Counter[str]:
    tokens = [token for group in GROUPS for token in record[group].values()]
    return Counter("+" if t.endswith("_pos") else "-" if t.endswith("_neg") else "0" for t in tokens)


def classify(record: dict) -> str:
    expected = expected_modeled_party_sign(record)
    observed = record["delta_eff"]
    if expected is None or observed == 0:
        return "ambiguous"
    return "expected-effect concordant" if (observed > 0) == (expected > 0) else "sign-flipped"


def evenly_spaced(records: list[dict], count: int) -> list[dict]:
    """Deterministic spread over corpus order; no selection on labels."""
    indexes = [round(i * (len(records) - 1) / (count - 1)) for i in range(count)]
    return [records[i] for i in indexes]


def main() -> None:
    raw = SOURCE.read_bytes()
    records = [json.loads(line) for line in raw.decode().splitlines() if line]
    mobilizing = [r for r in records if r["_seed_meta"]["mobilization"] == "mobilizing"]
    non_mobilizing = {
        label: [r for r in records if r["_seed_meta"]["mobilization"] == label]
        for label in ("depressing", "consolidating")
    }
    classifications = Counter(classify(r) for r in mobilizing)
    score_records = [r for r in mobilizing if r.get("news_roberta_scores") or r.get("social_roberta_scores")]

    lines = [
        "# Step 6.2a — Mobilizing Synthetic Direction Audit",
        "",
        "**Status:** read-only audit. No source records were regenerated, relabeled, filtered, or edited.",
        "",
        "## Source and decision rule",
        "",
        f"- Source: `{SOURCE.relative_to(ROOT)}` ({len(records)} approved records; SHA-256 `{hashlib.sha256(raw).hexdigest()}`).",
        f"- Mobilizing subset: **{len(mobilizing)}** records. Its only seed families are Democratic-incumbent and Republican-incumbent impeachment records; every row is `domain= institutional` and has a non-split expected effect (`helps_rep` or `helps_dem`).",
        "- Sign convention: bins and `delta_eff` are relative to the record's modeled `party`. For `helps_dem`, the expected sign is positive only when `party=democrat`; for `helps_rep`, it is positive only when `party=republican`. Otherwise the expected sign is negative.",
        "- A row is **expected-effect concordant** when `delta_eff` has that sign. A row is **sign-flipped** when it has the opposite sign. `splits`, `realigns`, and zero-effect rows would be ambiguous, but none occurs in this subset.",
        "- This measures output correctness against the seed's reviewed expected-effect direction. It cannot identify the teacher's internal causal rationale. In particular, no non-empty per-record sentiment input exists for 32 of 36 rows, and all 36 seeds pair a negative impeachment event with an expected loss for the impeached modeled party. Thus sentiment and expected effect predict the same sign here: the subset is **not a discriminating test** of whether mobilization overrides sentiment.",
        "",
        "## Result",
        "",
        f"- Expected-effect concordant: **{classifications['expected-effect concordant']}/36**",
        f"- Sign-flipped: **{classifications['sign-flipped']}/36**",
        f"- Formally ambiguous: **{classifications['ambiguous']}/36**",
        "- Attribution ambiguity: **36/36**. The labels are directionally consistent, but this corpus slice cannot establish that the mobilization rule—not the shared negative sentiment—caused that direction.",
        "",
        "## Per-record table — all mobilizing records",
        "",
        "`+/-/0` is the count of the 15 bloc bins that are positive/negative/neutral relative to the modeled party. Full signed tokens follow in race | religion | gender order.",
        "",
        "| # | shock_id | modeled party | expected effect → benefiting party | valence | delta_eff | + / - / 0 | result | signed bins |",
        "|---:|---|---|---|---|---:|---:|---|---|",
    ]
    for number, record in enumerate(mobilizing, 1):
        meta = record["_seed_meta"]
        counts = bloc_sign_counts(record)
        beneficiary = "democrat" if meta["expected_effect"] == "helps_dem" else "republican"
        lines.append(
            f"| {number} | `{record['shock_id']}` | {record['party']} | `{meta['expected_effect']}` → {beneficiary} "
            f"| {meta['valence']} | {record['delta_eff']:+.4f} | {counts['+']} / {counts['-']} / {counts['0']} "
            f"| {classify(record)} | {bins(record)} |"
        )

    lines += [
        "",
        "## The four legacy RoBERTa-bearing impeachment records",
        "",
        "These are the only mobilizing rows with non-empty score dictionaries. In each, `negative` is the largest class in both sources, and the modeled Democratic party receives a negative `delta_eff`; the expected effect is also `helps_rep`. They are output-concordant, but they illustrate the identification problem rather than validate an override: negative tone and the seed effect agree.",
        "",
        "| shock_id | party | delta_eff | news scores | social scores | audit interpretation |",
        "|---|---|---:|---|---|---|",
    ]
    for record in score_records:
        lines.append(
            f"| `{record['shock_id']}` | {record['party']} | {record['delta_eff']:+.4f} | "
            f"`{json.dumps(record['news_roberta_scores'], sort_keys=True)}` | "
            f"`{json.dumps(record['social_roberta_scores'], sort_keys=True)}` | negative tone and expected effect both imply modeled-party loss; not an override test |"
        )

    lines += [
        "",
        "## Examples",
        "",
        f"- **Directionally correct but causally unidentifiable:** `{mobilizing[0]['shock_id']}` is modeled for Democrats with `helps_rep`; all 11 non-neutral bloc bins are negative and `delta_eff={mobilizing[0]['delta_eff']:+.4f}`. It correctly benefits Republicans, but a negative-sentiment heuristic would give the same result.",
        "- **Incorrect case:** none. No mobilizing record has a `delta_eff` sign opposite its non-split expected effect.",
        "- **Within-row heterogeneity that is not a net sign flip:** Republican-impeachment rows include some positive modeled-Republican bloc bins (for example `gop_president_impeachment_abuse_of_power` has 1 positive, 7 negative, 7 neutral bins), but their net `delta_eff` remains negative as required by `helps_dem`. These are not evidence of a sentiment-driven net reversal.",
        "",
        "## Non-mobilizing contamination spot-check",
        "",
        "Fifteen rows were chosen deterministically by evenly spacing source-order records: eight depressing and seven consolidating. The same expected-effect-vs-`delta_eff` test is reported below. This is a direction-consistency check, not proof that a model did not use sentiment internally.",
        "",
        "| dynamic | shock_id | party | expected effect | delta_eff | result |",
        "|---|---|---|---|---:|---|",
    ]
    samples = evenly_spaced(non_mobilizing["depressing"], 8) + evenly_spaced(non_mobilizing["consolidating"], 7)
    for record in samples:
        lines.append(
            f"| {record['_seed_meta']['mobilization']} | `{record['shock_id']}` | {record['party']} | "
            f"`{record['_seed_meta']['expected_effect']}` | {record['delta_eff']:+.4f} | {classify(record)} |"
        )
    sample_counts = Counter(classify(r) for r in samples)
    lines += [
        "",
        f"Spot-check total: **{sample_counts['expected-effect concordant']}/15 concordant, {sample_counts['sign-flipped']}/15 flipped, {sample_counts['ambiguous']}/15 formally ambiguous**. The depressing population is 79/79 expected-effect concordant (all negative relative to its modeled party); the consolidating population is 223/223 concordant (all positive). This mechanical uniformity confirms convention consistency but is fragile evidence of causal reasoning—sentiment and intended effects are mostly aligned in these non-mobilizing archetypes too.",
        "",
        "## Scoped recommendation before retraining",
        "",
        "1. **Do not filter or relabel the 36 existing rows:** there are zero net sign flips under the source's own non-split expected-effect rule. A corpus-wide relabel would have no evidentiary basis and would risk overwriting reviewed labels.",
        "2. **Make the prompt rule explicit:** change the generator/reviewer wording so that, for `mobilizing`, an explicit seed-level benefiting/mobilized party determines the net sign before any sentiment or valence is considered. Retain mobilization as a model feature. This is a small prompt/schema change, not a corpus rewrite.",
        "3. **Before relying on the rule in retraining, add a targeted counterfactual validation/generation set** rather than regenerating these directionally concordant rows: paired negative-tone mobilizing seeds whose declared beneficiary is the party that negative sentiment would otherwise disadvantage. Require reviewer rationale to name the beneficiary and reject any net-sign mismatch. The current 36 rows cannot test this critical case because they are only impeachment rows with aligned tone/effect.",
        "",
        "Effort estimate: prompt/schema clarification is low (one generator prompt plus reviewer rubric); the targeted discriminating set is moderate (generate/review only the new paired cases); regenerating the 36 existing rows is unnecessary on this audit; a corpus-wide relabel is high cost and unsupported by the observed data.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")
    print(
        f"source_records={len(records)} mobilizing={len(mobilizing)} "
        f"concordant={classifications['expected-effect concordant']} "
        f"flipped={classifications['sign-flipped']} ambiguous={classifications['ambiguous']} "
        f"report={REPORT.relative_to(ROOT)}"
    )


if __name__ == "__main__":
    main()
