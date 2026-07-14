#!/usr/bin/env python3
"""build_grounded_v2.py — grounded-v2 fine-tuning corpus.

WHY: the v1 grounded model over-weighted injected sentiment — it parroted social
polarity (which is ~uniformly negative on political social media) and FLIPPED correct
predictions on held-out events (Dobbs). v2 makes sentiment *noisy, partial evidence*
rather than an answer key, and adds explicit counterexamples where sentiment polarity
opposes the (correct) partisan label.

DESIGN (all seed 42, deterministic):
  Base = data/finetune/synthetic.jsonl (1,228 records). LABELS ARE NEVER CHANGED —
  the expert-reasoned synthetic delta bins stay exactly as-is. We only ever populate
  the INPUT field ``social_roberta_scores`` (news stays empty — real news carries no
  per-bloc bio signal). Real per-bloc social sentiment comes from
  data/finetune/shock_sentiment_aggregates.json (blocs with n >= min_bloc_n).

  1. PARTIAL INJECTION: a synthetic record is "mapped" if its description matches a
     real-shock thematic family (below). Only ~INJECT_FRAC of mapped (non-divergent)
     records get sentiment injected; the rest keep ``social_roberta_scores = {}`` so
     the model cannot learn to depend on the field and must keep reasoning from text.
  2. NOISE: injected values get Gaussian jitter (sigma JITTER_SIGMA) and ~ZERO_FRAC of
     the injected blocs are randomly dropped — sentiment as noisy evidence.
  3. DIVERGENT COUNTEREXAMPLES: designated valence-divergent real shocks (negative
     sentiment yet historically party-*mobilizing*) are mapped to theme records that
     have >=1 positive label bloc. Their real NEGATIVE sentiment is injected while the
     POSITIVE synthetic label is kept → the model sees "negative sentiment + positive
     delta" and learns polarity != partisan effect. These are GUARANTEED injected.
  4. HELD-OUT real shocks are never injection sources (fair testing). If BLM/Kavanaugh
     are used as divergent TRAINING sources, other divergent shocks (MeToo, Chauvin)
     are held out instead.

Outputs (no retrain):
  data/finetune/synthetic_grounded_v2.jsonl   (full corpus + _grounding_v2 provenance)
  data/finetune/train_grounded_v2.jsonl       (party-stratified 80%)
  data/finetune/eval_grounded_v2.jsonl        (party-stratified 20%)
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from electoral.core.rng import derive_seed, make_rng
from electoral.core.types import (
    CANONICAL_GENDERS,
    CANONICAL_RACES,
    CANONICAL_RELIGIONS,
)

_ALL_BLOCS = [*CANONICAL_RACES, *CANONICAL_RELIGIONS, *CANONICAL_GENDERS]
_POS_BINS = {"slight_pos", "mild_pos", "mod_pos", "strong_pos"}

INJECT_FRAC = 0.40        # fraction of NON-divergent mapped records to inject
JITTER_SIGMA = 0.10       # Gaussian noise added to each injected roberta value
ZERO_FRAC = 0.20          # fraction of injected blocs randomly dropped to {}
NEG_THRESHOLD = -0.05     # a bloc's injected sentiment is "negative" below this
EVAL_FRACTION = 0.20

# ── Held-out real shocks: NEVER an injection source (fair testing) ────────────────
# BLM + Kavanaugh are wanted as divergent TRAINING sources, so per the task the
# other divergent shocks (MeToo, Chauvin) are held out in their place.
HELD_OUT_SOURCES = {
    "dobbs_2022",
    "afghanistan_withdrawal_2021",
    "affirmative_action_scotus_2023",
    "metoo_2017",
    "chauvin_conviction_2021",
}

# ── Designated divergent counterexample sources (negative sentiment, mobilizing) ──
DIVERGENT_SOURCES = {
    "blm_george_floyd_2020",
    "kavanaugh_2018",
    "ruth_bader_ginsburg_2020",
}

# ── Thematic families: real shocks → synthetic records (regex on description) ─────
# Divergent families listed first so their records win assignment (priority order).
FAMILIES: list[dict[str, Any]] = [
    {
        "theme": "race_policing",
        "sources": ["blm_george_floyd_2020", "ferguson_michael_brown_2014",
                    "eric_garner_2014", "trayvon_martin_2012"],
        "regex": r"police|policing|black lives|racial justice|protest|brutality|"
                 r"floyd|unarmed|civil rights|defund|racial",
    },
    {
        "theme": "gender_misconduct",
        "sources": ["kavanaugh_2018", "access_hollywood_2016"],
        "regex": r"sexual|harass|assault|misconduct|me\s?too|women'?s rights|"
                 r"transgender|gender",
    },
    {
        "theme": "abortion",
        "sources": ["ruth_bader_ginsburg_2020"],
        "regex": r"abortion|roe\b|reproductive|pro-life|pro-choice|planned parenthood",
    },
    {
        "theme": "immigration",
        "sources": ["daca_rescission_2017", "family_separation_2018", "travel_ban_2017"],
        "regex": r"immigr|migrant|border|daca|deport|asylum|amnesty|refugee|sanctuary",
    },
    {
        "theme": "guns",
        "sources": ["sandy_hook_2012", "parkland_shooting_2018",
                    "pulse_nightclub_2016", "las_vegas_shooting_2017"],
        "regex": r"\bgun|firearm|mass shooting|\bnra\b|assault weapon|second amendment",
    },
    {
        "theme": "foreign_war",
        "sources": ["russia_ukraine_invasion_2022", "israel_hamas_war_2023",
                    "bin_laden_killing_2011", "boston_marathon_2013"],
        "regex": r"\bwar\b|invasion|troops|military|\bnato\b|terror|missile|"
                 r"nuclear|foreign policy|airstrike",
    },
    {
        "theme": "econ",
        "sources": ["inflation_cpi_peak_2022", "obamacare_passage_2010"],
        "regex": r"recession|inflation|econom|jobs report|\btax\b|market crash|"
                 r"\bwage|trade war|tariff|healthcare|obamacare|cost of living",
    },
    {
        "theme": "scandal",
        "sources": ["trump_indictment_2023", "trump_conviction_2024",
                    "fbi_letter_2016", "ukraine_impeachment_2019"],
        "regex": r"scandal|corrupt|indict|bribe|pardon|\bfraud|leak|impeach|emails?\b",
    },
    {
        "theme": "climate",
        "sources": ["paris_climate_withdrawal_2017"],
        "regex": r"climate|paris agreement|carbon|emissions|green new|environment",
    },
]


def _flat_bins(rec: dict) -> dict[str, str]:
    return {
        **rec.get("delta_bins_race", {}),
        **rec.get("delta_bins_religion", {}),
        **rec.get("delta_bins_gender", {}),
    }


def load_real_social(aggregates_path: Path, min_bloc_n: int) -> dict[str, dict[str, float]]:
    """real_shock_id -> {bloc: roberta} for blocs with n >= min_bloc_n."""
    agg = json.loads(aggregates_path.read_text())["aggregates"]
    out: dict[str, dict[str, float]] = {}
    for sid, a in agg.items():
        soc = {
            b: float(v["roberta"])
            for b, v in a.get("social", {}).items()
            if v.get("n", 0) >= min_bloc_n and v.get("roberta") is not None
        }
        if soc:
            out[sid] = soc
    return out


def assign_family(rec: dict, real_social: dict) -> tuple[str, str] | None:
    """Return (theme, source_shock) for a synthetic record, or None if unmapped.

    Families are tried in priority order (divergent themes first). Within a family,
    only sources that (a) are not held out and (b) have social data are eligible.
    For divergent families we prefer the divergent source when the record has >=1
    positive label bloc (so injecting negative sentiment yields a counterexample).
    """
    desc = rec.get("description", "")
    flat = _flat_bins(rec)
    has_pos_bloc = any(v in _POS_BINS for v in flat.values())
    for fam in FAMILIES:
        if not re.search(fam["regex"], desc, re.I):
            continue
        eligible = [s for s in fam["sources"]
                    if s not in HELD_OUT_SOURCES and s in real_social]
        if not eligible:
            continue
        # Prefer a divergent source for records that carry a positive mobilized bloc.
        div = [s for s in eligible if s in DIVERGENT_SOURCES]
        if div and has_pos_bloc:
            return fam["theme"], div[0]
        return fam["theme"], eligible[0]
    return None


def inject(scores: dict[str, float], rng, sigma: float, zero_frac: float
           ) -> tuple[dict[str, float], list[str]]:
    """Jitter each bloc's roberta and randomly drop ~zero_frac of blocs.

    Guard: an injected record must keep >=1 real bloc — otherwise "injected" would
    be indistinguishable from "empty". If zeroing removes every bloc (common for
    thin single-bloc sources), the first-zeroed bloc is restored (jittered).
    """
    out: dict[str, float] = {}
    zeroed: list[str] = []
    for bloc in sorted(scores):  # deterministic order
        if rng.random() < zero_frac:
            zeroed.append(bloc)
            continue
        val = scores[bloc] + float(rng.normal(0.0, sigma))
        out[bloc] = round(max(-1.0, min(1.0, val)), 4)
    if not out and zeroed:  # never leave an injected record fully empty
        restore = zeroed.pop(0)
        val = scores[restore] + float(rng.normal(0.0, sigma))
        out[restore] = round(max(-1.0, min(1.0, val)), 4)
    return out, zeroed


def is_divergent(injected: dict[str, float], rec: dict) -> bool:
    """True if any surviving bloc pairs negative sentiment with a positive label."""
    flat = _flat_bins(rec)
    return any(v < NEG_THRESHOLD and flat.get(b) in _POS_BINS
               for b, v in injected.items())


def build(synthetic_path: Path, aggregates_path: Path, out_dir: Path,
          seed: int, min_bloc_n: int) -> dict:
    rng = make_rng(derive_seed(seed, "grounded_v2"))
    real_social = load_real_social(aggregates_path, min_bloc_n)

    records = [json.loads(l) for l in synthetic_path.read_text().splitlines() if l.strip()]
    records.sort(key=lambda r: r.get("shock_id", ""))  # deterministic consumption order

    # ── Pass 1: assign families ───────────────────────────────────────────────────
    mapped: list[tuple[dict, str, str]] = []   # (rec, theme, source)
    for rec in records:
        a = assign_family(rec, real_social)
        if a:
            mapped.append((rec, a[0], a[1]))
    mapped_ids = {id(r) for r, _, _ in mapped}

    # Divergent-eligible = mapped from a divergent source AND has a positive bloc.
    def is_div_eligible(rec, source):
        return source in DIVERGENT_SOURCES and any(
            v in _POS_BINS for v in _flat_bins(rec).values()
        )

    # ── Pass 2: choose which mapped records to inject ─────────────────────────────
    # Divergent-eligible → GUARANTEED inject. Others → INJECT_FRAC random (seed 42).
    inject_flag: dict[int, bool] = {}
    for rec, theme, source in mapped:
        if is_div_eligible(rec, source):
            inject_flag[id(rec)] = True
        else:
            inject_flag[id(rec)] = rng.random() < INJECT_FRAC

    # ── Pass 3: build output records (labels untouched) ───────────────────────────
    fam_lookup = {id(r): (t, s) for r, t, s in mapped}
    out_records: list[dict] = []
    stats = {
        "total": len(records), "mapped": len(mapped), "injected": 0,
        "divergent": 0, "empty_mapped": 0, "unmapped": 0,
        "by_theme": Counter(), "by_source": Counter(), "divergent_by_source": Counter(),
        "zeroed_blocs": 0, "injected_blocs": 0,
    }
    for rec in records:
        new = dict(rec)
        new["news_roberta_scores"] = {}          # news never carries per-bloc signal
        grounding = {"regime": "grounded_v2", "seed": seed, "min_bloc_n": min_bloc_n}

        if id(rec) not in mapped_ids:
            new["social_roberta_scores"] = {}
            grounding.update(mapped=False, injected=False, divergent=False)
            stats["unmapped"] += 1
        else:
            theme, source = fam_lookup[id(rec)]
            grounding.update(mapped=True, theme=theme, source_shock=source)
            if inject_flag[id(rec)]:
                injected, zeroed = inject(real_social[source], rng, JITTER_SIGMA, ZERO_FRAC)
                div = is_divergent(injected, rec)
                new["social_roberta_scores"] = injected
                grounding.update(
                    injected=True, divergent=div, jitter_sigma=JITTER_SIGMA,
                    zeroed_blocs=zeroed, n_injected_blocs=len(injected),
                )
                stats["injected"] += 1
                stats["by_theme"][theme] += 1
                stats["by_source"][source] += 1
                stats["injected_blocs"] += len(injected)
                stats["zeroed_blocs"] += len(zeroed)
                if div:
                    stats["divergent"] += 1
                    stats["divergent_by_source"][source] += 1
            else:
                new["social_roberta_scores"] = {}
                grounding.update(injected=False, divergent=False)
                stats["empty_mapped"] += 1

        new["_grounding_v2"] = grounding
        out_records.append(new)

    # ── Write full corpus ─────────────────────────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)
    full = out_dir / "synthetic_grounded_v2.jsonl"
    with full.open("w", encoding="utf-8") as f:
        for r in out_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ── Party-stratified 80/20 split (seed 42) ────────────────────────────────────
    split_rng = make_rng(derive_seed(seed, "grounded_v2_split"))
    by_party: dict[str, list[dict]] = defaultdict(list)
    for r in out_records:
        by_party[r.get("party", "unknown")].append(r)
    train, eval_ = [], []
    for party, group in sorted(by_party.items()):
        group = list(group)
        split_rng.shuffle(group)
        n_eval = max(1, round(len(group) * EVAL_FRACTION))
        eval_.extend(group[:n_eval])
        train.extend(group[n_eval:])
    split_rng.shuffle(train)
    split_rng.shuffle(eval_)

    for name, recs in (("train_grounded_v2.jsonl", train), ("eval_grounded_v2.jsonl", eval_)):
        with (out_dir / name).open("w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    stats["n_train"] = len(train)
    stats["n_eval"] = len(eval_)
    stats["train_injected"] = sum(1 for r in train if r["_grounding_v2"].get("injected"))
    stats["eval_injected"] = sum(1 for r in eval_ if r["_grounding_v2"].get("injected"))
    stats["real_shocks_available"] = len(real_social)
    # Split counterexamples: from a DESIGNATED divergent shock vs emergent
    # (regular source whose near-uniform-negative sentiment meets a positive label).
    stats["divergent_designated"] = sum(
        v for k, v in stats["divergent_by_source"].items() if k in DIVERGENT_SOURCES
    )
    stats["divergent_emergent"] = stats["divergent"] - stats["divergent_designated"]
    return {"stats": stats, "records": out_records, "out_dir": str(out_dir)}


def _example(records, pred, label):
    for r in records:
        if pred(r):
            g = r["_grounding_v2"]
            return {
                "label": label,
                "shock_id": r["shock_id"],
                "party": r["party"],
                "description": r["description"][:110],
                "_grounding_v2": g,
                "social_roberta_scores": r["social_roberta_scores"],
                "delta_bins_race": r.get("delta_bins_race", {}),
                "delta_bins_gender": r.get("delta_bins_gender", {}),
            }
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--synthetic", type=Path, default=Path("data/finetune/synthetic.jsonl"))
    ap.add_argument("--aggregates", type=Path,
                    default=Path("data/finetune/shock_sentiment_aggregates.json"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/finetune"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-bloc-n", type=int, default=30)
    args = ap.parse_args()

    res = build(args.synthetic, args.aggregates, args.out_dir, args.seed, args.min_bloc_n)
    s = res["stats"]
    recs = res["records"]

    line = "=" * 74
    print(f"\n{line}\nGROUNDED-V2 CORPUS REPORT\n{line}")
    print(f"Base synthetic records         : {s['total']}")
    print(f"Real shocks with social data   : {s['real_shocks_available']}")
    print(f"Mapped (theme-matched)         : {s['mapped']}")
    print(f"  → injected                   : {s['injected']}  "
          f"({s['injected']/max(s['mapped'],1)*100:.0f}% of mapped)")
    print(f"  → mapped-but-left-empty      : {s['empty_mapped']}")
    print(f"Unmapped (stay empty)          : {s['unmapped']}")
    print(f"Divergent counterexamples      : {s['divergent']}  "
          f"(neg sentiment + positive label bloc)")
    print(f"  from designated div. sources : {s['divergent_designated']}  "
          f"(BLM / Kavanaugh / RBG)")
    print(f"  emergent (regular sources)   : {s['divergent_emergent']}  "
          f"(negative social sentiment happens to meet a positive label)")
    print(f"  by source                    : {dict(s['divergent_by_source'])}")
    print(f"Injected blocs (total)         : {s['injected_blocs']}  "
          f"(zeroed by noise: {s['zeroed_blocs']})")
    print(f"\nInjections by theme            : {dict(s['by_theme'])}")
    print(f"Injections by source shock     : {dict(s['by_source'])}")
    print(f"\nHELD-OUT sources (never injected): {sorted(HELD_OUT_SOURCES)}")
    print(f"DIVERGENT training sources       : {sorted(DIVERGENT_SOURCES)}")
    print(f"\nSplit (party-stratified, seed {args.seed}): "
          f"train={s['n_train']} (inj {s['train_injected']}) / "
          f"eval={s['n_eval']} (inj {s['eval_injected']})")

    print(f"\nInjection math: {INJECT_FRAC:.0%} random of NON-divergent mapped records "
          f"+ all divergent-eligible (guaranteed) = {s['injected']}/{s['mapped']} "
          f"({s['injected']/max(s['mapped'],1):.0%}) overall.")

    print(f"\n{line}\n3 EXAMPLE RECORDS (distinct)\n{line}")
    used: set[str] = set()

    def pick(pred, label):
        ex = _example([r for r in recs if r["shock_id"] not in used], pred, label)
        if ex:
            used.add(ex["shock_id"])
        return ex

    g = lambda r: r["_grounding_v2"]
    # 1) clean: injected, not divergent, no blocs zeroed, >=2 injected blocs
    ex_clean = pick(
        lambda r: g(r).get("injected") and not g(r).get("divergent")
        and not g(r).get("zeroed_blocs") and g(r).get("n_injected_blocs", 0) >= 2,
        "CLEAN INJECT (aligned sentiment, no blocs zeroed)",
    )
    # 2) noisy: injected, not divergent, HAS zeroed blocs (noise visible)
    ex_noisy = pick(
        lambda r: g(r).get("injected") and not g(r).get("divergent")
        and g(r).get("zeroed_blocs"),
        "NOISY INJECT (jittered + some blocs zeroed)",
    )
    # 3) divergent: from a designated divergent source (BLM/Kavanaugh/RBG)
    ex_div = pick(
        lambda r: g(r).get("divergent") and g(r).get("source_shock") in DIVERGENT_SOURCES,
        "DIVERGENT COUNTEREXAMPLE (negative sentiment + positive label)",
    )
    for ex in (ex_clean, ex_noisy, ex_div):
        print("\n" + json.dumps(ex, ensure_ascii=False, indent=2))

    print(f"\n{line}\nWrote: {res['out_dir']}/synthetic_grounded_v2.jsonl, "
          f"train_grounded_v2.jsonl, eval_grounded_v2.jsonl\n{line}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
