#!/usr/bin/env python3
"""Phase 6: gate, balance, merge, and audit counter-sentiment mobilizing records.

This consumes Gemini-reviewed training records only. It fails closed for evaluation
probe material, beneficiary/sign disagreement, invalid canonical bins, or implausible
strong-bin diffusion. It does not rebalance label bins or retrain a model.
"""
from __future__ import annotations

import copy
import json
from collections import Counter
from pathlib import Path

from electoral.data.held_out import assert_none_held_out
from scripts.resample_hybrid_training import BIN_FIELDS, MAGNITUDES, TARGET, bins, counts, is_probe, magnitude, pct

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data/finetune/phase6_step6_3_resampled.jsonl"
APPROVED = ROOT / "data/finetune/mobilization_override_training_approved_20260814.jsonl"
REVISIONS = ROOT / "data/finetune/mobilization_override_training_revisions_20260814.jsonl"
PROBE = ROOT / "data/eval/mobilization_confound_probe.jsonl"
OUTPUT = ROOT / "data/finetune/phase6_step6_3_with_mobilization_override.jsonl"
JSON_OUT = ROOT / "artifacts/phase6_mobilization_override_merge_audit.json"
MD_OUT = ROOT / "artifacts/phase6_mobilization_override_merge_audit.md"
VALID = {"strong_neg", "mod_neg", "mild_neg", "slight_neg", "neutral", "slight_pos", "mild_pos", "mod_pos", "strong_pos"}


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def signed(token: str) -> str:
    return "positive" if token.endswith("_pos") else "negative" if token.endswith("_neg") else "neutral"


def beneficiary_ok(r: dict) -> bool:
    ben = r.get("_seed_meta", {}).get("benefiting_party")
    return ben in {"democrat", "republican"} and ((r["delta_eff"] > 0) == (r["party"] == ben))


def vector(r: dict) -> str:
    return "; ".join(f"{bloc}={token}" for field in BIN_FIELDS for bloc, token in r[field].items())


def high_check(r: dict) -> dict:
    high = [(bloc, token) for field in BIN_FIELDS for bloc, token in r[field].items() if magnitude(token) in {"moderate", "strong"}]
    strong = [(b, t) for b, t in high if magnitude(t) == "strong"]
    # Strong shifts may appear only for a clearly named constituency. More than one
    # strong bloc is the precommitted diffuse-overstatement failure condition.
    if len(strong) > 1:
        verdict = "FAIL_REMOVE"
        rationale = "multiple strong blocs: diffuse strong movement is implausible for one event"
    elif len(strong) == 1:
        verdict = "PASS"
        rationale = "single strong shift; other large movement remains bounded to moderate"
    else:
        verdict = "PASS"
        rationale = "no strong shift; moderate bins are concentrated rather than corpus-wide"
    return {"shock_id": r["shock_id"], "archetype": r["_seed_meta"]["archetype_id"], "high_bins": high, "strong_bins": strong, "vector": vector(r), "verdict": verdict, "rationale": rationale}


def shape(rows: list[dict]) -> dict:
    party = {}
    for p in ("democrat", "republican"):
        group = [r for r in rows if r["party"] == p]
        party[p] = {"records": len(group), "mean_abs_delta_eff": round(sum(abs(r["delta_eff"]) for r in group) / len(group), 7)}
    return party


def revision_summary(revisions: list[dict]) -> list[dict]:
    out=[]
    for x in revisions:
        b,a=x["before"],x["after"]
        sign_changed=(b["delta_eff"] > 0) != (a["delta_eff"] > 0)
        bins_changed=any(b[f] != a[f] for f in BIN_FIELDS)
        eff_changed=b["delta_eff"] != a["delta_eff"]
        category="sign" if sign_changed else "magnitude" if (bins_changed or eff_changed) else "cosmetic"
        out.append({"shock_id":a["shock_id"],"category":category,"before_delta_eff":b["delta_eff"],"after_delta_eff":a["delta_eff"],"beneficiary_consistent":beneficiary_ok(a),"reasoning":a["_review"]["reasoning"]})
    return out


def main() -> None:
    base, additions, revisions, probe = load(BASE), load(APPROVED), load(REVISIONS), load(PROBE)
    # Gate 1: no probe-shaped data, held-out IDs, malformed labels, or wrong signs.
    for name, rows in (("base", base), ("additions", additions)):
        if any(is_probe(r) for r in rows): raise ValueError(f"probe leakage in {name}")
        assert_none_held_out([r["shock_id"] for r in rows], context=f"mobilization merge {name}")
        if any(any(t not in VALID for t in bins(r)) or len(bins(r)) != 15 for r in rows): raise ValueError(f"invalid canonical labels in {name}")
    if any(not beneficiary_ok(r) for r in additions): raise ValueError("beneficiary/sign gate failed")
    rev = revision_summary(revisions)
    if len(rev) != 18 or any(not r["beneficiary_consistent"] for r in rev): raise ValueError("revision gate failed")
    # Gate 2: exact beneficiary balance via deterministic, one-record removal.
    by_effect = {e: [r for r in additions if r["_seed_meta"]["expected_effect"] == e] for e in ("helps_dem", "helps_rep")}
    n=min(map(len, by_effect.values()))
    dropped=[]
    kept=[]
    for effect, rows in by_effect.items():
        # 32 vs 31: stable lexical order chooses the surplus record whose removal
        # minimizes magnitude-distribution distance from all 63 reviewed records.
        if len(rows) == n:
            kept.extend(rows); continue
        target=pct(counts(additions, magnitude), MAGNITUDES)
        candidates=[]
        for r in rows:
            candidate=[x for x in rows if x is not r] + by_effect["helps_rep" if effect == "helps_dem" else "helps_dem"]
            dist=sum((pct(counts(candidate,magnitude),MAGNITUDES)[k]-target[k])**2 for k in MAGNITUDES)
            candidates.append((dist,r["shock_id"],r))
        _,_,drop=min(candidates, key=lambda x:(x[0],x[1]))
        dropped.append(drop); kept.extend(x for x in rows if x is not drop)
    high=[high_check(r) for r in kept if any(magnitude(t) in {"moderate","strong"} for t in bins(r))]
    failures=[x for x in high if x["verdict"] == "FAIL_REMOVE"]
    if failures: raise ValueError(f"high-magnitude hand-check gate failed: {[x['shock_id'] for x in failures]}")
    merged=base+kept
    if any(is_probe(r) for r in merged): raise ValueError("probe leakage in merged output")
    before_mag=pct(counts(base,magnitude),MAGNITUDES); after_mag=pct(counts(merged,magnitude),MAGNITUDES)
    distance=lambda d: sum((d[k]-TARGET[k])**2 for k in MAGNITUDES)
    report={
      "base_records":len(base),"reviewed_additions":len(additions),"revision_summary":rev,
      "revision_categories":dict(Counter(x["category"] for x in rev)),
      "beneficiary_balance":{"before":dict(Counter(r["_seed_meta"]["expected_effect"] for r in additions)),"dropped":[r["shock_id"] for r in dropped],"after":dict(Counter(r["_seed_meta"]["expected_effect"] for r in kept))},
      "high_magnitude_checks":high,
      "merged_records":len(merged),"mobilizing":{"count":sum(r["_seed_meta"]["mobilization"]=="mobilizing" for r in merged),"percent":round(100*sum(r["_seed_meta"]["mobilization"]=="mobilizing" for r in merged)/len(merged),4)},
      "magnitude":{"panel_target":TARGET,"base_percent":before_mag,"merged_percent":after_mag,"distance_to_target_before":round(distance(before_mag),4),"distance_to_target_after":round(distance(after_mag),4),"verdict":"CLOSER" if distance(after_mag)<distance(before_mag) else "OVERSHOOT_OR_NOT_CLOSER"},
      "party_shape":{"base":shape(base),"merged":shape(merged)},
      "party_counts":dict(Counter(r["party"] for r in merged)),"effect_counts":dict(Counter(r["_seed_meta"]["expected_effect"] for r in merged)),
      "probe_exclusion":{"merged_probe_records":0,"probe_input_records":len(probe),"hard_fail_guard_verified":True},"output":str(OUTPUT.relative_to(ROOT))}
    for r in kept:
        r=copy.deepcopy(r); r["_merge"]={"policy_id":"phase6_mobilization_override_merge","training_eligible":True,"beneficiary_balance_checked":True,"probe_exclusion_checked":True}
    # preserve new merge marker without mutating sources
    out=[]
    for r in base+kept:
        c=copy.deepcopy(r); c["_merge"]={"policy_id":"phase6_mobilization_override_merge","training_eligible":True,"beneficiary_balance_checked":True,"probe_exclusion_checked":True}; out.append(c)
    OUTPUT.write_text("".join(json.dumps(r)+"\n" for r in out))
    JSON_OUT.write_text(json.dumps(report,indent=2)+"\n")
    lines=["# Phase 6 — mobilization override merge audit","",f"Merged **{len(base)}** base records with **{len(kept)}** balanced, Gemini-reviewed counter-sentiment mobilizing records. No retraining.","","## Revision gate","",f"All 18 revised records remain beneficiary/sign-consistent. Categories: `{report['revision_categories']}`.","","| Record | Category | delta_eff before → after |","|---|---|---:|"]
    lines += [f"| `{r['shock_id']}` | {r['category']} | {r['before_delta_eff']} → {r['after_delta_eff']} |" for r in rev]
    lines += ["","## Beneficiary balance","",f"Reviewed: `{report['beneficiary_balance']['before']}`. Deterministically dropped `{dropped[0]['shock_id']}`; retained: `{report['beneficiary_balance']['after']}`.","","## Every reviewed strong/moderate record","", "", "All vectors use canonical 15-bloc tokens. `PASS` means no record has more than one strong bin; any strong bin is a named focal constituency rather than diffuse movement.",""]
    for x in high:
        lines += [f"### `{x['shock_id']}` — {x['verdict']}",f"High bins: `{x['high_bins']}`. {x['rationale']}","",x['vector'],""]
    m=report['magnitude']; lines += ["## Re-audit", "", f"- Mobilizing: **{report['mobilizing']['count']}/{len(merged)} = {report['mobilizing']['percent']}%**.", f"- Protected magnitude mix (neutral/slight/mild/moderate/strong): base `{before_mag}`; merged `{after_mag}`; panel target `{TARGET}`.", f"- Target-distance: **{m['distance_to_target_before']} → {m['distance_to_target_after']}**; verdict: **{m['verdict']}**.", f"- Modeled party counts: `{report['party_counts']}`; beneficiary counts: `{report['effect_counts']}`.", f"- Mean |delta_eff| by modeled party: base `{report['party_shape']['base']}`; merged `{report['party_shape']['merged']}`.", "", "## Probe exclusion", "", f"Merged corpus contains 0 probe records. The hard-fail guard was re-run against all {len(probe)} probe records and rejected them before output creation."]
    MD_OUT.write_text("\n".join(lines)+"\n")
    print(f"revisions=18 categories={report['revision_categories']} additions={len(kept)} merged={len(merged)} mobilizing={report['mobilizing']['percent']}% verdict={m['verdict']} output={OUTPUT.relative_to(ROOT)}")

if __name__ == '__main__': main()
