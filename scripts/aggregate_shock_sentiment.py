#!/usr/bin/env python3
"""aggregate_shock_sentiment.py — build the per-shock sentiment grounding table.

BIG PICTURE
-----------
The fine-tune training prompts carry two fields — ``news_roberta_scores`` and
``social_roberta_scores`` — that were designed to hold *real* per-bloc reactions
scored from archive posts, but have always been empty ``{}``. The scored archive
data exists at::

    /Volumes/JUHIDRIVE/electoralData/scored/{shock_id}/scored.jsonl

where each record is::

    {
      "roberta_score": <float in [-1, 1]>,
      "payload": {
        "bio_bloc": <demographic bloc or "unknown">,
        "sentiment_score": <float>,
        "platform": <str>, "archive_id": <str>,
        "shock_id": <str>, "day_offset": <int>, ...
      }
    }

This script reads every ``scored.jsonl`` (READ-ONLY), groups records per shock by
``bio_bloc``, separates NEWS from SOCIAL by archive/platform, and writes a
per-shock aggregate::

    {shock_id: {"news": {bloc: {...}}, "social": {bloc: {...}}, "counts": {...}}}

to ``data/finetune/shock_sentiment_aggregates.json``. It ALSO writes a coverage
report (``..._coverage.json``) and prints a human summary. It does NOT touch the
training corpus and does NOT retrain — the point is to see how much real grounding
is actually available before deciding to inject.

NEWS vs SOCIAL
--------------
News archives carry no author bio (see electoral/nlp/news_loader.py), so their
``bio_bloc`` is essentially always ``"unknown"`` — news therefore yields an
aggregate reaction, not a per-bloc breakdown. Social archives (Reddit, Discord,
Twitter/CSV, Truth Social, Telegram, LIWC) carry SetFit-inferred ``bio_bloc`` and
DO break down per bloc. The classifier below reflects that: a record is NEWS iff
its ``archive_id``/``platform`` is a known news source, else SOCIAL.

Usage
-----
    python scripts/aggregate_shock_sentiment.py                 # defaults
    python scripts/aggregate_shock_sentiment.py --min-bloc-n 30 --seed 42

Reproducible via --seed (default 42); aggregation is deterministic (means over all
records), so the seed is recorded for provenance rather than used for sampling.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

from electoral.core.rng import derive_seed, make_rng
from electoral.core.types import (
    CANONICAL_GENDERS,
    CANONICAL_RACES,
    CANONICAL_RELIGIONS,
)

log = logging.getLogger("aggregate_shock_sentiment")

# --- Canonical demographic blocs (race + religion + gender). "unknown" is NOT a
#     bloc: it is the SetFit fallback and is reported separately, never injected. ---
CANONICAL_BLOCS: tuple[str, ...] = tuple(
    [*CANONICAL_RACES, *CANONICAL_RELIGIONS, *CANONICAL_GENDERS]
)

# --- NEWS vs SOCIAL classification -------------------------------------------------
# Verified against the JUHIDRIVE scored inventory (2026-07) and configs/archives.json:
#   NEWS   : 3dlnews (platform "local_news"); webhose / scraped "news" if present.
#   SOCIAL : reddit_pushshift, reddit_monthly_filtered, discord, and every Twitter/
#            CSV archive (election_20XX, covid_*, truth_social_2024, telegram_2024,
#            metoo_liwc, daca_scotus_2020 — all platform_proxy / bio_classifier tweets).
# The distinction is by SOURCE (archive_id / platform), not by shock.
NEWS_ARCHIVE_IDS: frozenset[str] = frozenset({"3dlnews", "webhose"})
NEWS_PLATFORMS: frozenset[str] = frozenset({"local_news", "news"})


def classify_channel(payload: dict) -> str:
    """Return 'news' or 'social' for a scored record's payload."""
    if payload.get("archive_id") in NEWS_ARCHIVE_IDS:
        return "news"
    if payload.get("platform") in NEWS_PLATFORMS:
        return "news"
    return "social"


def _round(x: float | None, n: int = 4) -> float | None:
    return None if x is None else round(float(x), n)


def aggregate_one_shock(scored_path: Path) -> dict:
    """Aggregate a single shock's scored.jsonl into news/social per-bloc summaries.

    Returns a dict with keys ``news``, ``social``, ``counts``. Score dicts are keyed
    by CANONICAL bloc only (``unknown`` excluded); ``unknown`` volume is preserved in
    ``counts`` so the caller can see how much signal is unusable.
    """
    # channel -> bloc -> list of (roberta, sentiment)
    roberta: dict[str, dict[str, list[float]]] = {
        "news": defaultdict(list),
        "social": defaultdict(list),
    }
    sentiment: dict[str, dict[str, list[float]]] = {
        "news": defaultdict(list),
        "social": defaultdict(list),
    }
    n_records = 0
    n_bad = 0

    with scored_path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                payload = rec["payload"]
            except (json.JSONDecodeError, KeyError, TypeError):
                n_bad += 1
                continue
            n_records += 1

            channel = classify_channel(payload)
            bloc = payload.get("bio_bloc") or "unknown"
            if bloc != "unknown" and bloc not in CANONICAL_BLOCS:
                # non-canonical, non-unknown label — bucket as unknown for visibility
                bloc = "unknown"

            rob = rec.get("roberta_score")
            sen = payload.get("sentiment_score")
            if isinstance(rob, (int, float)):
                roberta[channel][bloc].append(float(rob))
            if isinstance(sen, (int, float)):
                sentiment[channel][bloc].append(float(sen))

    def summarize(channel: str) -> dict[str, dict]:
        """Per-canonical-bloc {roberta, sentiment, n} for one channel (drop unknown)."""
        out: dict[str, dict] = {}
        blocs = set(roberta[channel]) | set(sentiment[channel])
        for bloc in blocs:
            if bloc == "unknown":
                continue
            r_vals = roberta[channel].get(bloc, [])
            s_vals = sentiment[channel].get(bloc, [])
            n = max(len(r_vals), len(s_vals))
            if n == 0:
                continue
            out[bloc] = {
                "roberta": _round(mean(r_vals)) if r_vals else None,
                "sentiment": _round(mean(s_vals)) if s_vals else None,
                "n": n,
            }
        # deterministic, canonical ordering
        return {b: out[b] for b in CANONICAL_BLOCS if b in out}

    def counts(channel: str) -> dict:
        by_bloc = {
            b: len(roberta[channel].get(b, [])) or len(sentiment[channel].get(b, []))
            for b in (set(roberta[channel]) | set(sentiment[channel]))
        }
        unknown = by_bloc.pop("unknown", 0)
        canonical_total = sum(by_bloc.values())
        return {
            "total": canonical_total + unknown,
            "canonical": canonical_total,
            "unknown": unknown,
            "by_bloc": {b: by_bloc[b] for b in CANONICAL_BLOCS if b in by_bloc},
        }

    return {
        "news": summarize("news"),
        "social": summarize("social"),
        "counts": {
            "records": n_records,
            "malformed": n_bad,
            "news": counts("news"),
            "social": counts("social"),
        },
    }


def build_coverage(
    aggregates: dict[str, dict],
    corpus_shock_ids: dict[str, int] | None,
    min_bloc_n: int,
) -> dict:
    """Summarize how much real grounding exists and where it is thin/absent."""
    per_shock = {}
    for sid, agg in aggregates.items():
        meaningful = {"news": [], "social": [], "thin": {"news": [], "social": []}}
        for channel in ("news", "social"):
            for bloc, s in agg[channel].items():
                if s["n"] >= min_bloc_n:
                    meaningful[channel].append(bloc)
                else:
                    meaningful["thin"][channel].append(bloc)
        per_shock[sid] = {
            "news_records": agg["counts"]["news"]["total"],
            "social_records": agg["counts"]["social"]["total"],
            "news_blocs_meaningful": sorted(meaningful["news"]),
            "social_blocs_meaningful": sorted(meaningful["social"]),
            "thin_blocs": {
                "news": sorted(meaningful["thin"]["news"]),
                "social": sorted(meaningful["thin"]["social"]),
            },
        }

    scored_ids = set(aggregates)
    coverage = {
        "min_bloc_n": min_bloc_n,
        "n_shocks_with_scored_data": len(scored_ids),
        "per_shock": per_shock,
    }

    if corpus_shock_ids is not None:
        corpus_set = set(corpus_shock_ids)
        matched = sorted(corpus_set & scored_ids)
        unmatched = sorted(corpus_set - scored_ids)
        coverage["corpus"] = {
            "n_unique_shock_ids": len(corpus_set),
            "n_records": sum(corpus_shock_ids.values()),
            "n_matched_exact": len(matched),
            "n_unmatched": len(unmatched),
            "matched_shock_ids": matched,
            "note": (
                "Exact shock_id join. Corpus shock_ids are synthetic descriptive "
                "slugs; scored shock_ids are real historical events. Any grounding "
                "requires a thematic real->synthetic mapping, not an exact join. "
                "Shocks with no match keep empty sentiment."
            ),
        }
    return coverage


def load_corpus_shock_ids(path: Path) -> dict[str, int]:
    """shock_id -> record count for a finetune corpus jsonl (read-only)."""
    ids: dict[str, int] = defaultdict(int)
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                sid = json.loads(line).get("shock_id")
            except json.JSONDecodeError:
                continue
            if sid is not None:
                ids[sid] += 1
    return dict(ids)


def print_report(aggregates: dict, coverage: dict, min_bloc_n: int) -> None:
    """Human-readable coverage summary to stdout."""
    line = "=" * 78
    print(f"\n{line}\nSHOCK SENTIMENT GROUNDING — COVERAGE REPORT\n{line}")
    n = coverage["n_shocks_with_scored_data"]
    print(f"Shocks with scored archive data : {n}")
    print(f"Meaningful-bloc threshold (N)    : {min_bloc_n} records/bloc\n")

    # channel totals
    news_tot = sum(a["counts"]["news"]["total"] for a in aggregates.values())
    news_canon = sum(a["counts"]["news"]["canonical"] for a in aggregates.values())
    soc_tot = sum(a["counts"]["social"]["total"] for a in aggregates.values())
    soc_canon = sum(a["counts"]["social"]["canonical"] for a in aggregates.values())
    print("Channel volume (all shocks):")
    print(
        f"  NEWS   total={news_tot:>7,}  with-usable-bloc={news_canon:>7,}  "
        f"unknown={news_tot - news_canon:>7,}"
    )
    print(
        f"  SOCIAL total={soc_tot:>7,}  with-usable-bloc={soc_canon:>7,}  "
        f"unknown={soc_tot - soc_canon:>7,}"
    )
    if news_canon == 0:
        print(
            "  ! NEWS has NO per-bloc signal (news posts have no author bio -> "
            "bio_bloc=unknown). News can only provide an aggregate reaction."
        )

    # per-shock table
    print(f"\n{'shock_id':32} {'news':>6} {'social':>7}  social meaningful blocs (>=N)")
    print("-" * 78)
    for sid in sorted(aggregates):
        c = coverage["per_shock"][sid]
        blocs = ",".join(c["social_blocs_meaningful"]) or "(none)"
        print(
            f"{sid:32} {c['news_records']:>6} {c['social_records']:>7}  "
            f"{len(c['social_blocs_meaningful']):>2}: {blocs[:60]}"
        )

    # corpus overlap
    if "corpus" in coverage:
        cc = coverage["corpus"]
        print(f"\n{line}\nTRAINING-CORPUS COVERAGE\n{line}")
        print(f"Corpus unique shock_ids : {cc['n_unique_shock_ids']} "
              f"({cc['n_records']} records)")
        print(f"Exact-match to scored   : {cc['n_matched_exact']}")
        print(f"No scored coverage      : {cc['n_unmatched']} "
              f"(these keep empty sentiment under an exact join)")
        if cc["n_matched_exact"] == 0:
            print(
                "  ! ZERO exact matches. Corpus uses synthetic slugs "
                "(e.g. 'affirmative_action_overturned_1'); scored data uses real\n"
                "    events (e.g. 'affirmative_action_scotus_2023'). Grounding needs "
                "a thematic real->synthetic mapping, which is a separate decision."
            )
    print(line + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--scored-root",
        type=Path,
        default=Path("/Volumes/JUHIDRIVE/electoralData/scored"),
        help="Directory of {shock_id}/scored.jsonl (read-only).",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("data/finetune/shock_sentiment_aggregates.json"),
    )
    ap.add_argument(
        "--coverage-out",
        type=Path,
        default=Path("data/finetune/shock_sentiment_coverage.json"),
    )
    ap.add_argument(
        "--corpus",
        type=Path,
        default=Path("data/finetune/synthetic.jsonl"),
        help="Finetune corpus to check shock_id overlap against (read-only).",
    )
    ap.add_argument(
        "--min-bloc-n",
        type=int,
        default=30,
        help="Min records for a per-shock per-bloc mean to be 'meaningful'.",
    )
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    # Seed for provenance/convention (aggregation itself is deterministic).
    make_rng(derive_seed(args.seed, "aggregate_shock_sentiment"))

    if not args.scored_root.is_dir():
        log.error("scored root not found (drive not mounted?): %s", args.scored_root)
        return 2

    scored_files = sorted(args.scored_root.glob("*/scored.jsonl"))
    if not scored_files:
        log.error("no {shock_id}/scored.jsonl under %s", args.scored_root)
        return 2
    log.info("found %d scored shock file(s)", len(scored_files))

    aggregates: dict[str, dict] = {}
    for f in scored_files:
        shock_id = f.parent.name
        aggregates[shock_id] = aggregate_one_shock(f)
        log.info(
            "  %-32s news=%d social=%d",
            shock_id,
            aggregates[shock_id]["counts"]["news"]["total"],
            aggregates[shock_id]["counts"]["social"]["total"],
        )
    aggregates = {k: aggregates[k] for k in sorted(aggregates)}

    corpus_ids = (
        load_corpus_shock_ids(args.corpus) if args.corpus.exists() else None
    )
    coverage = build_coverage(aggregates, corpus_ids, args.min_bloc_n)

    output = {
        "_meta": {
            "seed": args.seed,
            "scored_root": str(args.scored_root),
            "min_bloc_n": args.min_bloc_n,
            "n_shocks": len(aggregates),
            "score_fields": ["roberta", "sentiment"],
            "news_archive_ids": sorted(NEWS_ARCHIVE_IDS),
            "news_platforms": sorted(NEWS_PLATFORMS),
            "note": (
                "Per-bloc means over ALL scored records per shock. 'unknown' bloc "
                "excluded from score dicts (reported in counts). News has no author "
                "bio, so its per-bloc breakdown is empty by construction."
            ),
        },
        "aggregates": aggregates,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, sort_keys=False))
    args.coverage_out.write_text(json.dumps(coverage, indent=2, sort_keys=False))
    log.info("wrote aggregates -> %s", args.out)
    log.info("wrote coverage   -> %s", args.coverage_out)

    print_report(aggregates, coverage, args.min_bloc_n)
    return 0


if __name__ == "__main__":
    sys.exit(main())
