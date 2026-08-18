#!/usr/bin/env python3
"""Mismatch test: does the model reason from event CONTENT, or follow the stated LABEL?

The coalition-structure replication showed that beneficiary framing shifts coalition
structure toward the stated beneficiary regardless of which party is modeled.  Two
explanations fit that data equally well:

  (A) the model reasons about the beneficiary from the EVENT's content, and the framing
      merely makes explicit what the event already implies; or
  (B) the model blindly follows the stated beneficiary LABEL, which dominates whatever
      the event says.

They are indistinguishable when the label always agrees with the event.  This test breaks
the tie by making them DISAGREE.

DESIGN -- a minimal pair.  For each held-out probe event (which carries a declared true
beneficiary), three conditions:

  BARE        bare event, no beneficiary framing                      (control)
  CONGRUENT   bare event + TEMPLATE(true beneficiary)                 (baseline)
  MISMATCHED  bare event + TEMPLATE(opposite party)                   (the test)

CONGRUENT and MISMATCHED use the SAME templated clause, differing only in the party named.
That matters: if CONGRUENT used the probe's original prose and MISMATCHED used a rewrite,
the two would differ in wording as well as in label, and any difference would be
uninterpretable.  Using one template for both isolates the label as the single variable.

MEASUREMENT -- coalition contrast, not net sign (net sign hid the previous result):

    rep_minus_dem = mean(REP_CORE bins) - mean(DEM_CORE bins)

On a MISMATCHED run the true beneficiary and the stated label are opposite parties, so
they predict opposite signs.  Whichever sign appears tells us which one the model followed:

    follows EVENT  -> sign(rep_minus_dem) matches the TRUE beneficiary   -> explanation A
    follows LABEL  -> sign(rep_minus_dem) matches the STATED party       -> explanation B

Read-only inference.  The probe is evaluation-only and is never used for training.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from electoral.core.types import (
    BIN_MIDPOINTS,
    CANONICAL_GENDERS,
    CANONICAL_RACES,
    CANONICAL_RELIGIONS,
)

PROMPT_PATH = "ShockEstimator.estimate -> format_prompt(event_dict) -> [INST]...[/INST]"

# Identical to the coalition-structure harness, kept fixed for comparability.
REP_CORE = ("white", "evangelical", "protestant", "men")
DEM_CORE = ("african_american", "latino", "secular", "jewish", "women")

BLOCS = tuple(CANONICAL_RACES) + tuple(CANONICAL_RELIGIONS) + tuple(CANONICAL_GENDERS)
PARTIES = ("democrat", "republican")
CONDITIONS = ("BARE", "CONGRUENT", "MISMATCHED")

_PIVOTS = (", yet ", ", but ", "; the ", ". ")

# One template, parameterised by party. Wording is symmetric across parties so that the
# ONLY difference between CONGRUENT and MISMATCHED is which party is named.
_CLAUSE = {
    "republican": (
        " The controversy energizes the Republican base and boosts conservative turnout "
        "in defense of the measure."
    ),
    "democrat": (
        " The controversy energizes the Democratic base and boosts progressive turnout "
        "in defense of the measure."
    ),
}


def other(party: str) -> str:
    return "democrat" if party == "republican" else "republican"


def bare_of(description: str) -> str:
    """Strip the mobilization clause, leaving the bare event (see coalition harness)."""
    d = description.strip()
    hits = [(d.find(p), p) for p in _PIVOTS if d.find(p) != -1]
    if not hits:
        return d
    cut, _ = min(hits)
    stem = d[:cut].strip()
    return stem if stem.endswith(".") else stem + "."


def read_probe(path: Path) -> list[dict[str, Any]]:
    records = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    if not records:
        raise ValueError(f"probe is empty: {path}")
    for rec in records:
        probe = rec.get("_probe") or {}
        prov = rec.get("_provenance") or {}
        pid = probe.get("probe_id", "<unknown>")
        if probe.get("benefiting_party") not in PARTIES:
            raise ValueError(f"{pid}: missing/invalid benefiting_party")
        if (
            prov.get("dataset_role") != "evaluation_probe"
            or prov.get("training_eligible") is not False
        ):
            raise ValueError(f"{pid}: not permanently evaluation-only")
    return records


def text_for(rec: dict[str, Any], condition: str) -> tuple[str, str | None]:
    """Return (description, stated_party). stated_party is None for BARE."""
    bare = bare_of(rec["description"])
    true_b = rec["_probe"]["benefiting_party"]
    if condition == "BARE":
        return bare, None
    stated = true_b if condition == "CONGRUENT" else other(true_b)
    return bare + _CLAUSE[stated], stated


def core_mean(bins: dict[str, str], core: tuple[str, ...]) -> float:
    return sum(BIN_MIDPOINTS[bins[b]] for b in core if b in bins) / len(core)


def rep_minus_dem(bins: dict[str, str]) -> float:
    return core_mean(bins, REP_CORE) - core_mean(bins, DEM_CORE)


def followed(rmd: float, true_b: str, stated: str | None) -> str:
    """Which reference does the coalition movement match?"""
    if rmd == 0:
        return "flat"
    leans = "republican" if rmd > 0 else "democrat"
    if stated is None or stated == true_b:
        return "event" if leans == true_b else "opposite"
    return "event" if leans == true_b else ("label" if leans == stated else "flat")


def flatten(resp: Any) -> dict[str, str]:
    bins = {
        **dict(resp.delta_bins_race),
        **dict(resp.delta_bins_religion),
        **dict(resp.delta_bins_gender),
    }
    missing = [b for b in BLOCS if b not in bins]
    if missing:
        raise ValueError(f"prediction missing blocs: {missing}")
    return {b: bins[b] for b in BLOCS}


def run_all(adapter: Path, base_model: str, probe: Path, seed: int) -> list[dict[str, Any]]:
    from electoral.llm.inference import ShockEstimator
    import torch

    records = read_probe(probe)
    est = ShockEstimator(adapter_path=str(adapter), base_model=base_model)
    rows: list[dict[str, Any]] = []
    for rec in records:
        p = rec["_probe"]
        true_b = p["benefiting_party"]
        for cond in CONDITIONS:
            desc, stated = text_for(rec, cond)
            for modeled in PARTIES:
                event = {
                    "shock_id": f"{p['probe_id']}::{cond}::{modeled}",
                    "cycle": int(rec.get("cycle") or 2024),
                    "party": modeled,
                    "description": desc,
                    "news_roberta_scores": {},
                    "social_roberta_scores": {},
                }
                torch.manual_seed(seed)
                bins = flatten(est.estimate(event, intensity=1.0))
                rmd = rep_minus_dem(bins)
                rows.append(
                    {
                        "probe_id": p["probe_id"],
                        "true_beneficiary": true_b,
                        "condition": cond,
                        "stated_party": stated,
                        "modeled_party": modeled,
                        "description": desc,
                        "bins": bins,
                        "rep_minus_dem": rmd,
                        "follows": followed(rmd, true_b, stated),
                    }
                )
                print(
                    f"  {p['probe_id']:<34} {cond:<11} modeled={modeled:<11} "
                    f"rep-dem={rmd:+.5f} follows={rows[-1]['follows']}",
                    flush=True,
                )
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_event: dict[str, dict[str, Any]] = {}
    for r in rows:
        e = by_event.setdefault(
            r["probe_id"], {"true_beneficiary": r["true_beneficiary"], "vals": {}}
        )
        e["vals"].setdefault(r["condition"], []).append(r["rep_minus_dem"])

    events = []
    for pid, e in by_event.items():
        true_b = e["true_beneficiary"]
        m = {c: sum(v) / len(v) for c, v in e["vals"].items()}
        # On MISMATCHED, event-following and label-following predict opposite signs.
        mism_leans = "republican" if m["MISMATCHED"] > 0 else "democrat"
        events.append(
            {
                "probe_id": pid,
                "true_beneficiary": true_b,
                "bare": m["BARE"],
                "congruent": m["CONGRUENT"],
                "mismatched": m["MISMATCHED"],
                "mismatched_follows": ("event" if mism_leans == true_b else "label"),
                "congruent_correct": (
                    ("republican" if m["CONGRUENT"] > 0 else "democrat") == true_b
                ),
            }
        )
    events.sort(key=lambda x: (x["true_beneficiary"], x["probe_id"]))

    n = len(events)
    follows_event = sum(1 for e in events if e["mismatched_follows"] == "event")
    return {
        "n_events": n,
        "n_generations": len(rows),
        "mismatched_follows_event": follows_event,
        "mismatched_follows_label": n - follows_event,
        "congruent_correct": sum(1 for e in events if e["congruent_correct"]),
        "verdict": (
            "A: reasons from event content"
            if follows_event > n / 2
            else "B: framing/label dominates"
        ),
        "events": events,
    }


def markdown(s: dict[str, Any]) -> str:
    L = [
        "# Beneficiary mismatch test",
        "",
        f"**Prompt path:** `{PROMPT_PATH}`",
        "",
        "`rep_minus_dem = mean(REP-core) - mean(DEM-core)`, averaged over both modeled "
        "parties. Positive = coalition moved Republican-favourable.",
        "",
        "CONGRUENT and MISMATCHED share one templated clause; only the named party differs.",
        "",
        "| Event | True benef. | BARE | CONGRUENT | MISMATCHED | mismatched follows |",
        "|---|---|---:|---:|---:|---|",
    ]
    for e in s["events"]:
        L.append(
            f"| `{e['probe_id']}` | {e['true_beneficiary']} | {e['bare']:+.5f} | "
            f"{e['congruent']:+.5f} | {e['mismatched']:+.5f} | "
            f"**{e['mismatched_follows']}** |"
        )
    L += [
        "",
        "## Aggregate",
        "",
        f"- events: {s['n_events']} ({s['n_generations']} generations)",
        f"- CONGRUENT recovers the true beneficiary: {s['congruent_correct']}/{s['n_events']}",
        f"- MISMATCHED follows the EVENT: **{s['mismatched_follows_event']}/{s['n_events']}**",
        f"- MISMATCHED follows the LABEL: **{s['mismatched_follows_label']}/{s['n_events']}**",
        "",
        f"**Verdict: {s['verdict']}**",
        "",
        "Descriptive only — small n, no significance test.",
    ]
    return "\n".join(L) + "\n"


def run_self_test() -> None:
    probe = Path("data/eval/mobilization_confound_probe.jsonl")
    recs = read_probe(probe)
    assert len(recs) == 10, f"expected 10 probe records, got {len(recs)}"
    for r in recs:
        true_b = r["_probe"]["benefiting_party"]
        bare, s0 = text_for(r, "BARE")
        con, s1 = text_for(r, "CONGRUENT")
        mis, s2 = text_for(r, "MISMATCHED")
        assert s0 is None and s1 == true_b and s2 == other(true_b)
        # minimal pair: congruent and mismatched differ ONLY in the clause's party terms
        assert con.startswith(bare) and mis.startswith(bare), "both must extend the same bare event"
        assert len(con) > len(bare) and len(mis) > len(bare)
        assert con != mis
    # follows() sign convention
    assert followed(+0.01, "republican", "democrat") == "event"
    assert followed(-0.01, "republican", "democrat") == "label"
    assert followed(+0.01, "democrat", "republican") == "label"
    assert followed(-0.01, "democrat", "republican") == "event"
    print(f"SELF_TEST=PASS  ({len(recs)} events, minimal pair verified)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--adapter")
    ap.add_argument("--base-model", default="mistralai/Mistral-7B-v0.3")
    ap.add_argument(
        "--probe", type=Path, default=Path("data/eval/mobilization_confound_probe.jsonl")
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=Path("artifacts/beneficiary_mismatch.json"))
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()

    if a.self_test:
        run_self_test()
        return
    if not a.adapter:
        ap.error("--adapter is required unless --self-test is supplied")
    adapter = Path(a.adapter)
    if not (adapter.is_dir() and (adapter / "adapter_config.json").is_file()):
        raise FileNotFoundError(f"adapter is not a local PEFT directory: {adapter}")

    rows = run_all(adapter, a.base_model, a.probe, a.seed)
    s = summarize(rows)
    payload = {
        "prompt_path": PROMPT_PATH,
        "adapter": a.adapter,
        "seed": a.seed,
        "rep_core": list(REP_CORE),
        "dem_core": list(DEM_CORE),
        "clause_template": _CLAUSE,
        "runs": rows,
        "summary": s,
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    a.out.with_suffix(".md").write_text(markdown(s), encoding="utf-8")
    print(markdown(s), end="")
    print(f"out={a.out} markdown={a.out.with_suffix('.md')}")


if __name__ == "__main__":
    main()
