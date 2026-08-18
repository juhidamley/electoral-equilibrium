#!/usr/bin/env python3
"""Multi-event replication of the coalition-structure finding, across the confound probe.

Runs each of the 10 held-out mobilization-confound probe events FOUR ways --
BARE/democrat, BARE/republican, STATED/democrat, STATED/republican -- through the
production serving path (ShockEstimator.estimate -> format_prompt(event_dict)), 40
generations from a single model load.

It scores COALITION CONTRAST rather than net sign.  The single-event pilot showed the
net-sign rule can reject a working intervention: STATED framing flipped the Republican
core positive while the aggregate stayed negative because other blocs outweighed it.

  contrast = mean(beneficiary-core bins) - mean(opponent-core bins)

Positive contrast = the declared beneficiary's coalition moved in its favour relative to
the opponent's, regardless of the aggregate.  The probe set is balanced 5 democrat- /
5 republican-beneficiary, so a genuine framing effect must show up in BOTH directions;
an effect in only one direction is a party lean the framing cannot overcome.

BARE derivation is mechanical and auditable -- see bare_of().  The probe is
evaluation-only and is never used for training.
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

# Core coalitions -- identical to the single-event re-scoring, kept fixed for continuity.
REP_CORE = ("white", "evangelical", "protestant", "men")
DEM_CORE = ("african_american", "latino", "secular", "jewish", "women")

BLOCS = tuple(CANONICAL_RACES) + tuple(CANONICAL_RELIGIONS) + tuple(CANONICAL_GENDERS)
PARTIES = ("democrat", "republican")
FRAMINGS = ("BARE", "STATED")

# Pivot markers separating the event clause from the mobilization clause.  Every probe
# description is "<event + hostile reaction> <pivot> <who it actually mobilises>";
# 7 of 10 join the two halves inside a single sentence, so first-sentence-only is NOT
# sufficient.  We cut at the EARLIEST pivot.
_PIVOTS = (", yet ", ", but ", "; the ", ". ")


def bare_of(description: str) -> str:
    """Strip the mobilization/beneficiary clause, leaving the bare event statement.

    Deterministic and auditable: cut at the earliest pivot marker, keep the prefix.
    The resulting text still names the actor's party (that is part of the event) but
    never states who BENEFITS -- which is the variable under test.
    """
    d = description.strip()
    hits = [(d.find(p), p) for p in _PIVOTS if d.find(p) != -1]
    if not hits:
        return d
    cut, _ = min(hits)
    stem = d[:cut].strip()
    return stem if stem.endswith(".") else stem + "."


def read_probe(path: Path) -> list[dict[str, Any]]:
    """Read the probe, failing closed on its evaluation-only contract."""
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


def core_mean(bins: dict[str, str], core: tuple[str, ...]) -> float:
    vals = [BIN_MIDPOINTS[bins[b]] for b in core if b in bins]
    if not vals:
        raise ValueError(f"no bins present for core {core}")
    return sum(vals) / len(vals)


def contrast(bins: dict[str, str], beneficiary: str) -> float:
    """(beneficiary-core movement) - (opponent-core movement). Positive = appropriate."""
    rep, dem = core_mean(bins, REP_CORE), core_mean(bins, DEM_CORE)
    return (rep - dem) if beneficiary == "republican" else (dem - rep)


def build_event(rec: dict[str, Any], framing: str, party: str) -> dict[str, Any]:
    desc = rec["description"] if framing == "STATED" else bare_of(rec["description"])
    return {
        "shock_id": f"{rec['_probe']['probe_id']}::{framing}::{party}",
        "cycle": int(rec.get("cycle") or 2024),
        "party": party,
        "description": desc,
        "news_roberta_scores": {},
        "social_roberta_scores": {},
    }


def flatten_bins(response: Any) -> dict[str, str]:
    bins = {
        **dict(response.delta_bins_race),
        **dict(response.delta_bins_religion),
        **dict(response.delta_bins_gender),
    }
    missing = [b for b in BLOCS if b not in bins]
    if missing:
        raise ValueError(f"prediction missing blocs: {missing}")
    return {b: bins[b] for b in BLOCS}


def run_all(adapter: Path, base_model: str, probe: Path, seed: int) -> list[dict[str, Any]]:
    """One model load, 40 generations."""
    from electoral.llm.inference import ShockEstimator
    import torch

    records = read_probe(probe)
    estimator = ShockEstimator(adapter_path=str(adapter), base_model=base_model)

    rows: list[dict[str, Any]] = []
    for rec in records:
        p = rec["_probe"]
        for framing in FRAMINGS:
            for party in PARTIES:
                event = build_event(rec, framing, party)
                torch.manual_seed(seed)
                resp = estimator.estimate(event, intensity=1.0)
                bins = flatten_bins(resp)
                rows.append(
                    {
                        "probe_id": p["probe_id"],
                        "beneficiary": p["benefiting_party"],
                        "framing": framing,
                        "modeled_party": party,
                        "description": event["description"],
                        "bins": bins,
                        "rep_core": core_mean(bins, REP_CORE),
                        "dem_core": core_mean(bins, DEM_CORE),
                        "contrast": contrast(bins, p["benefiting_party"]),
                        "delta_eff_raw": float(resp.delta_eff),
                    }
                )
                print(
                    f"  {p['probe_id']:<34} {framing:<7} {party:<11} "
                    f"contrast={rows[-1]['contrast']:+.5f}",
                    flush=True,
                )
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-event and per-beneficiary-class aggregates. Averages over both modeled parties."""
    by_event: dict[str, dict[str, Any]] = {}
    for r in rows:
        e = by_event.setdefault(
            r["probe_id"], {"beneficiary": r["beneficiary"], "BARE": [], "STATED": []}
        )
        e[r["framing"]].append(r["contrast"])

    events = []
    for pid, e in by_event.items():
        bare = sum(e["BARE"]) / len(e["BARE"])
        stated = sum(e["STATED"]) / len(e["STATED"])
        events.append(
            {
                "probe_id": pid,
                "beneficiary": e["beneficiary"],
                "bare_contrast": bare,
                "stated_contrast": stated,
                "delta": stated - bare,
                # "tracks" = STATED moved the beneficiary's own coalition favourably
                # AND did so more than BARE did.
                "stated_tracks_beneficiary": bool(stated > 0 and stated > bare),
            }
        )

    classes = {}
    for cls in PARTIES:
        sub = [e for e in events if e["beneficiary"] == cls]
        classes[cls] = {
            "n": len(sub),
            "n_tracking": sum(1 for e in sub if e["stated_tracks_beneficiary"]),
            "mean_bare_contrast": sum(e["bare_contrast"] for e in sub) / len(sub),
            "mean_stated_contrast": sum(e["stated_contrast"] for e in sub) / len(sub),
        }

    both = all(c["n_tracking"] >= (c["n"] + 1) // 2 for c in classes.values())
    return {
        "events": sorted(events, key=lambda e: (e["beneficiary"], e["probe_id"])),
        "by_beneficiary": classes,
        "holds_for_both_parties": both,
    }


def markdown(rows: list[dict[str, Any]], s: dict[str, Any]) -> str:
    L = [
        "# Coalition structure across the confound probe (40 generations)",
        "",
        f"**Prompt path:** `{PROMPT_PATH}`",
        "",
        "`contrast = mean(beneficiary-core bins) - mean(opponent-core bins)`; "
        "positive = beneficiary-appropriate. Each value averages the two modeled parties.",
        "",
        f"REP-core: `{', '.join(REP_CORE)}` · DEM-core: `{', '.join(DEM_CORE)}`",
        "",
        "## Per event",
        "",
        "| Event | Beneficiary | BARE contrast | STATED contrast | Δ | STATED tracks? |",
        "|---|---|---:|---:|---:|---|",
    ]
    for e in s["events"]:
        L.append(
            f"| `{e['probe_id']}` | {e['beneficiary']} | {e['bare_contrast']:+.5f} | "
            f"{e['stated_contrast']:+.5f} | {e['delta']:+.5f} | "
            f"{'yes' if e['stated_tracks_beneficiary'] else 'NO'} |"
        )
    L += [
        "",
        "## By beneficiary class",
        "",
        "| Beneficiary | n | tracking | mean BARE | mean STATED |",
        "|---|---:|---:|---:|---:|",
    ]
    for cls, c in s["by_beneficiary"].items():
        L.append(
            f"| {cls} | {c['n']} | {c['n_tracking']}/{c['n']} | "
            f"{c['mean_bare_contrast']:+.5f} | {c['mean_stated_contrast']:+.5f} |"
        )
    L += [
        "",
        "## Verdict",
        "",
        f"**Holds for both parties: {s['holds_for_both_parties']}**",
        "",
        "Descriptive only — n=10 events, no significance test.",
    ]
    return "\n".join(L) + "\n"


def run_self_test() -> None:
    """Validate BARE derivation and contrast maths without a model."""
    probe = Path("data/eval/mobilization_confound_probe.jsonl")
    recs = read_probe(probe)
    assert len(recs) == 10, f"expected 10 probe records, got {len(recs)}"
    leak = (
        "energiz",
        "galvaniz",
        "rally",
        "rallies",
        "turnout",
        "boost",
        "solidif",
        "activat",
        "backfire",
        "convert",
    )
    for r in recs:
        b = bare_of(r["description"])
        assert b, f"empty bare for {r['_probe']['probe_id']}"
        assert len(b) < len(r["description"]), f"bare not shorter for {r['_probe']['probe_id']}"
        low = b.lower()
        hit = [w for w in leak if w in low]
        assert not hit, f"beneficiary leak {hit} in bare of {r['_probe']['probe_id']}: {b}"
    # contrast sign convention
    pos = {b: "neutral" for b in BLOCS}
    for b in REP_CORE:
        pos[b] = "mod_pos"
    assert contrast(pos, "republican") > 0, "rep-favourable bins must give positive rep contrast"
    assert contrast(pos, "democrat") < 0, "rep-favourable bins must give negative dem contrast"
    print(f"SELF_TEST=PASS  ({len(recs)} probe records, bare derivation clean)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--adapter")
    ap.add_argument("--base-model", default="mistralai/Mistral-7B-v0.3")
    ap.add_argument(
        "--probe", type=Path, default=Path("data/eval/mobilization_confound_probe.jsonl")
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=Path("artifacts/coalition_structure_probe.json"))
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
        "base_model": a.base_model,
        "seed": a.seed,
        "rep_core": list(REP_CORE),
        "dem_core": list(DEM_CORE),
        "bare_derivation": "cut at earliest of " + repr(_PIVOTS),
        "n_generations": len(rows),
        "runs": rows,
        "summary": s,
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    a.out.with_suffix(".md").write_text(markdown(rows, s), encoding="utf-8")
    print(markdown(rows, s), end="")
    print(f"out={a.out} markdown={a.out.with_suffix('.md')}")


if __name__ == "__main__":
    main()
