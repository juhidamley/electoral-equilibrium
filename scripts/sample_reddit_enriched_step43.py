#!/usr/bin/env python3
"""Phase 4, Step 4.3 — enriched Reddit resampling with content-relevance narrowing.

Problem (Step 0.2): Reddit posts get their shock_id from DATE WINDOW ALONE.
electoral/nlp/collectors/schema.py's match_shocks()/build_keyword_index() keyword
router exists and is used by archive.py/news_loader.py/bluesky_firehose.py, but was
never wired into scripts/sample_archives.py's Reddit path -- so no Reddit post has
ever been content-checked against the shock it was stamped with.

This script closes that gap WITHOUT reinventing the narrowing that already exists:
  - Date window: reused verbatim from each shock's `date_window` field (same
    precedence order as sample_archives.py:build_task_list -- date_window if
    present, else PRE_SHOCK_DAYS/shock_window_days). Every candidate shock here
    already has date_window set, so this is just read, not recomputed.
  - Subreddit targeting: reused, not rebuilt. reddit_pushshift's 84 subreddits
    were already hand-curated for political/demographic relevance at archive-
    download time (data/archives/README.md); reddit_monthly_filtered was already
    narrowed by scripts/filter_reddit_monthly.py's KEYWORD_FILTERS regex. Both are
    upstream of this script.
  - Content relevance: THE missing step. electoral.nlp.collectors.schema's
    build_keyword_index()/match_shocks() are imported and applied verbatim against
    each candidate post's body text.

Two candidate populations, handled differently for cost reasons:
  1. Shocks with an existing cleaned/{shock_id}/{archive_id}.jsonl cache (already
     date-window-narrowed by a prior sample_archives.py run) -- load directly, no
     need to re-touch the raw archive. Covers all but 3 target shocks.
  2. Shocks with NO cache (obergefell_2015, iran_nuclear_deal_2015,
     paris_attacks_2015 -- 2015 dates that were never run) -- single fresh
     streaming pass over reddit_pushshift's 168 .zst files, read-only, via the
     `zstd` CLI (piped, not the zstandard python package, which isn't installed
     in .venv). A `zstd -dc | grep -iE` coarse pre-filter (superset of each
     shock's keywords) keeps this pass affordable; match_shocks() is still the
     sole authority on what counts as a match -- grep only skips lines that
     provably cannot match.

Output: NEW location, never overwrites cleaned/ or the raw archive.
  /Volumes/JUHIDRIVE/electoralData/reddit_enriched_step43/{shock_id}/{archive_id}.jsonl

Read-only w.r.t. /Volumes/JUHIDRIVE/electoralData/archives/. This script performs
STEP 1 (enriched sampling) only. It does not call any LLM / DeepSeek API.
"""

from __future__ import annotations

import json
import logging
import random
import re
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from electoral.core.rng import derive_seed  # noqa: E402
from electoral.data.excluded_sources import excluded_archive_ids, filter_excluded  # noqa: E402
from electoral.nlp.collectors.schema import build_keyword_index, match_shocks  # noqa: E402

import sample_archives as SA  # noqa: E402  -- reuse, don't reinvent

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
)
logger = logging.getLogger("step43")

# ── Paths ────────────────────────────────────────────────────────────────────
CLEANED_ROOT = Path("/Volumes/JUHIDRIVE/electoralData/cleaned")
RAW_PUSHSHIFT_DIR = Path("/Volumes/JUHIDRIVE/electoralData/archives/reddit/reddit_pushshift")
OUTPUT_ROOT = Path("/Volumes/JUHIDRIVE/electoralData/reddit_enriched_step43")
REPORT_PATH = REPO_ROOT.parent / "step43_report.json"  # overwritten by caller with real scratch path

SEED = 42
REDDIT_ARCHIVES = {"reddit_pushshift", "reddit_monthly", "reddit_monthly_filtered"}
TARGET_PRIORITY = 5_000
TARGET_SECONDARY = 1_500
GLOBAL_CAP = 500_000
USABLE_FLOOR = 100  # project's established "clean" usability tier

# The 3 shocks with no prior sampling run against Reddit at all (2015, no cache).
FRESH_SCAN_SHOCKS = {"obergefell_2015", "iran_nuclear_deal_2015", "paris_attacks_2015"}


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def compute_window(shock: dict) -> tuple[datetime, datetime, datetime]:
    """shock_date, window_start, window_end -- same precedence as build_task_list()."""
    dw = shock.get("date_window", {})
    window_days = shock.get("shock_window_days", 14)
    shock_date_str = dw.get("shock_date") or shock["date"]
    shock_dt = datetime.fromisoformat(shock_date_str).replace(tzinfo=timezone.utc)
    start_str = dw.get("start") or (shock_dt - timedelta(days=SA.PRE_SHOCK_DAYS)).strftime("%Y-%m-%d")
    end_str = dw.get("end") or (shock_dt + timedelta(days=window_days)).strftime("%Y-%m-%d")
    start_dt = datetime.fromisoformat(start_str).replace(tzinfo=timezone.utc)
    end_dt = datetime.fromisoformat(end_str).replace(tzinfo=timezone.utc)
    return shock_dt, start_dt, end_dt


def classify_shocks(shocks: list[dict], held_out_ids: set[str]) -> dict[str, dict]:
    """Determine, per trainable shock: priority tier + which reddit archive_ids apply.

    'usable non-reddit coverage' is measured directly off cleaned/{shock_id}/*.jsonl
    (excluding reddit_* and excluded_sources.json archive_ids), counting records
    with a bio_bloc other than unknown/None -- the same usability criterion used
    throughout this project (a bloc-blind record can't produce a per-bloc label).
    """
    excl = excluded_archive_ids()
    info: dict[str, dict] = {}
    for shock in shocks:
        sid = shock["id"]
        if sid in held_out_ids:
            continue
        if not shock.get("active", True):
            info[sid] = {"status": "unaddressable_inactive", "shock": shock}
            continue
        reddit_ids = [a for a in shock.get("archive_ids", []) if a in REDDIT_ARCHIVES]
        if not reddit_ids:
            info[sid] = {"status": "unaddressable_no_archive", "shock": shock}
            continue

        shock_dir = CLEANED_ROOT / sid
        non_reddit_usable = 0
        non_reddit_raw = 0
        if shock_dir.is_dir():
            for fn in shock_dir.iterdir():
                if not fn.name.endswith(".jsonl") or fn.name.startswith("."):
                    continue
                archive_id = fn.name[:-6]
                if archive_id in REDDIT_ARCHIVES or archive_id in excl:
                    continue
                with open(fn, encoding="utf-8", errors="replace") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        non_reddit_raw += 1
                        try:
                            r = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        payload = r.get("payload", r)
                        bb = payload.get("bio_bloc")
                        if bb and bb != "unknown":
                            non_reddit_usable += 1

        tier = "secondary" if non_reddit_usable >= USABLE_FLOOR else "priority"
        info[sid] = {
            "status": "addressable",
            "shock": shock,
            "tier": tier,
            "reddit_archive_ids": reddit_ids,
            "non_reddit_usable": non_reddit_usable,
            "non_reddit_raw": non_reddit_raw,
            "had_cache": {a: (CLEANED_ROOT / sid / f"{a}.jsonl").exists() for a in reddit_ids},
        }
    return info


# ── Cached-file loading (reuses on-disk bio_bloc/sentiment/day_offset) ────────


def load_cached_records(shock_id: str, archive_id: str) -> list[dict]:
    path = CLEANED_ROOT / shock_id / f"{archive_id}.jsonl"
    out = []
    if not path.exists():
        return out
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            payload = r.get("payload", r)
            out.append(payload)
    return out


# ── Fresh scan of raw reddit_pushshift for uncached shocks ────────────────────


def build_grep_pattern(shocks_by_id: dict, shock_ids: list[str]) -> str:
    kws = []
    for sid in shock_ids:
        kws.extend(shocks_by_id[sid].get("keywords", []))
    escaped = sorted({re.escape(k) for k in kws}, key=len, reverse=True)
    return "|".join(escaped)


def fresh_scan_pushshift(
    shock_ids: list[str],
    shocks_by_id: dict,
    windows: dict[str, tuple[datetime, datetime, datetime]],
    keyword_index: dict,
) -> dict[str, list[dict]]:
    """Single pass over all reddit_pushshift .zst files for shocks with no cache.

    grep -iE pre-filter (superset of the target keywords) then authoritative
    match_shocks() + date-window check per surviving line. Read-only.
    """
    pattern = build_grep_pattern(shocks_by_id, shock_ids)
    files = sorted(RAW_PUSHSHIFT_DIR.glob("*.zst"))
    logger.info("fresh_scan: %d pushshift files, grep pattern length=%d", len(files), len(pattern))

    out: dict[str, list[dict]] = defaultdict(list)
    earliest = min(w[1] for w in windows.values()) - timedelta(days=30)
    latest = max(w[2] for w in windows.values()) + timedelta(days=30)

    for i, path in enumerate(files, 1):
        zstd = subprocess.Popen(
            ["zstd", "-dc", "--long=31", str(path)], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
        )
        grep = subprocess.Popen(
            ["grep", "-iE", pattern],
            stdin=zstd.stdout,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        zstd.stdout.close()
        n_grep_hits = 0
        for raw_line in grep.stdout:
            n_grep_hits += 1
            try:
                row = json.loads(raw_line)
            except json.JSONDecodeError:
                continue
            body = str(row.get("body") or "").strip()
            if not body or body in ("[deleted]", "[removed]"):
                continue
            created_utc = row.get("created_utc")
            created_at = SA.parse_timestamp(created_utc) if created_utc is not None else None
            if created_at is None or created_at < earliest or created_at > latest:
                continue
            matched = match_shocks(body, keyword_index)
            for sid in matched & set(shock_ids):
                _, wstart, wend = windows[sid]
                if wstart <= created_at <= wend:
                    out[sid].append(
                        {
                            "text": body,
                            "created_at": created_at,
                            "author_description": str(row.get("author_flair_text") or "").strip()
                            or None,
                            "post_id": str(row.get("id") or ""),
                            "platform": str(row.get("subreddit") or "reddit"),
                            "likes": row.get("score"),
                        }
                    )
        grep.wait()
        zstd.wait()
        if i % 20 == 0 or i == len(files):
            logger.info(
                "fresh_scan: %d/%d files done, running totals: %s",
                i,
                len(files),
                {sid: len(v) for sid, v in out.items()},
            )
    return out


# ── Stratified resample (mirrors sample_archives.sample_archive's day/bloc/sentiment logic) ──


def stratified_resample(
    posts: list[dict],
    shock_dt: datetime,
    target: int,
    lexicons: dict,
    scorer,
    rng: random.Random,
) -> list[dict]:
    bucket_posts: dict[int, list[dict]] = defaultdict(list)
    for p in posts:
        created_at = p.get("created_at")
        if isinstance(created_at, str):
            try:
                created_at = datetime.fromisoformat(created_at)
            except ValueError:
                created_at = None
        day = SA.temporal_day_key(created_at, shock_dt) if created_at else p.get("day_offset")
        if day is None:
            continue
        bio_bloc = p.get("bio_bloc") or SA.keyword_bio_bloc(p.get("author_description"), lexicons)
        sentiment = p.get("sentiment_score")
        if sentiment is None:
            sentiment = scorer(p["text"])
        p = dict(p)
        p["day_offset"] = day
        p["bio_bloc"] = bio_bloc
        p["sentiment_score"] = sentiment
        p["sampled_weight"] = SA.EXTREMITY_WEIGHT if abs(sentiment) >= SA.EXTREMITY_THRESHOLD else 1.0
        p["created_at"] = created_at
        bucket_posts[day].append(p)

    if not bucket_posts:
        return []

    n_days = len(bucket_posts)
    target_per_day = max(1, target // n_days)
    sampled: list[dict] = []
    for day_offset, day_posts in sorted(bucket_posts.items()):
        by_bloc: dict[str, list[dict]] = defaultdict(list)
        for p in day_posts:
            by_bloc[p["bio_bloc"]].append(p)
        present_blocs = list(by_bloc.keys())
        raw_shares = {b: SA.BLOC_SHARES.get(b, 1.0 / len(present_blocs)) for b in present_blocs}
        total_share = sum(raw_shares.values())
        norm_shares = {b: s / total_share for b, s in raw_shares.items()}
        day_sample: list[dict] = []
        for bloc, posts_in_bloc in by_bloc.items():
            bloc_target = max(1, round(target_per_day * norm_shares[bloc]))
            if len(posts_in_bloc) <= bloc_target:
                day_sample.extend(posts_in_bloc)
                continue
            weights = [p["sampled_weight"] for p in posts_in_bloc]
            chosen = rng.choices(posts_in_bloc, weights=weights, k=bloc_target)
            day_sample.extend(chosen)
        sampled.extend(day_sample)

    if len(sampled) > target:
        rng.shuffle(sampled)
        sampled = sampled[:target]
    return sampled


def main() -> None:
    shocks = load_json(REPO_ROOT / "configs" / "shocks.json")
    shocks_by_id = {s["id"]: s for s in shocks}
    held_out_ids = {s["id"] for s in load_json(REPO_ROOT / "configs" / "held_out_shocks.json")["held_out_shocks"]}
    keyword_index = build_keyword_index(shocks)
    lexicons = SA.load_lexicons()
    scorer = SA.make_sentiment_scorer()

    info = classify_shocks(shocks, held_out_ids)
    addressable = {sid: v for sid, v in info.items() if v["status"] == "addressable"}
    priority_ids = sorted(sid for sid, v in addressable.items() if v["tier"] == "priority")
    secondary_ids = sorted(sid for sid, v in addressable.items() if v["tier"] == "secondary")
    unaddressable = {sid: v["status"] for sid, v in info.items() if v["status"] != "addressable"}

    logger.info("=" * 70)
    logger.info("CLASSIFICATION")
    logger.info("priority (n=%d): %s", len(priority_ids), priority_ids)
    logger.info("secondary (n=%d): %s", len(secondary_ids), secondary_ids)
    logger.info("unaddressable: %s", unaddressable)
    logger.info("=" * 70)

    windows = {sid: compute_window(shocks_by_id[sid]) for sid in list(addressable)}

    report: dict[str, dict] = {}
    total_written = 0

    # ── Phase B: cached shocks ─────────────────────────────────────────────
    fresh_needed = []
    for sid, v in addressable.items():
        shock_dt, wstart, wend = windows[sid]
        target = TARGET_PRIORITY if v["tier"] == "priority" else TARGET_SECONDARY
        shock_report = {"tier": v["tier"], "archives": {}}
        for archive_id in v["reddit_archive_ids"]:
            if not v["had_cache"].get(archive_id):
                if sid in FRESH_SCAN_SHOCKS and archive_id == "reddit_pushshift":
                    fresh_needed.append(sid)
                continue
            raw = load_cached_records(sid, archive_id)
            n_raw = len(raw)
            keyword_matched = [
                p for p in raw if sid in match_shocks(p.get("text", ""), keyword_index)
            ]
            n_matched = len(keyword_matched)
            excl_filtered = filter_excluded(
                [{"payload": {**p, "archive_id": archive_id}} for p in keyword_matched],
                context=f"step43/{sid}/{archive_id}",
            )
            excl_filtered = [r["payload"] for r in excl_filtered]
            rng = random.Random(derive_seed(SEED, f"step43_{sid}_{archive_id}"))
            sampled = stratified_resample(excl_filtered, shock_dt, target, lexicons, scorer, rng)
            n_bloc_usable = sum(1 for p in sampled if p.get("bio_bloc", "unknown") != "unknown")

            out_path = OUTPUT_ROOT / sid / f"{archive_id}.jsonl"
            SA.write_output(sampled, out_path, sid, archive_id, SEED)
            total_written += len(sampled)

            shock_report["archives"][archive_id] = {
                "source": "cached",
                "raw_window_narrowed": n_raw,
                "after_keyword_match": n_matched,
                "sampled": len(sampled),
                "bloc_usable_in_sample": n_bloc_usable,
            }
            logger.info(
                "%s/%s: raw=%d -> keyword_match=%d -> sampled=%d (bloc_usable=%d)",
                sid,
                archive_id,
                n_raw,
                n_matched,
                len(sampled),
                n_bloc_usable,
            )
        report[sid] = shock_report

    # ── Phase C: fresh scan for uncached 2015 shocks ────────────────────────
    fresh_needed = sorted(set(fresh_needed))
    if fresh_needed:
        logger.info("=" * 70)
        logger.info("PHASE C: fresh pushshift scan for %s", fresh_needed)
        logger.info("=" * 70)
        fresh_windows = {sid: windows[sid] for sid in fresh_needed}
        fresh_candidates = fresh_scan_pushshift(fresh_needed, shocks_by_id, fresh_windows, keyword_index)
        for sid in fresh_needed:
            shock_dt, wstart, wend = windows[sid]
            target = TARGET_PRIORITY if addressable[sid]["tier"] == "priority" else TARGET_SECONDARY
            raw = fresh_candidates.get(sid, [])
            # already keyword-matched + date-windowed inside fresh_scan_pushshift
            excl_filtered = filter_excluded(
                [{"payload": {**p, "archive_id": "reddit_pushshift"}} for p in raw],
                context=f"step43/{sid}/reddit_pushshift(fresh)",
            )
            excl_filtered = [r["payload"] for r in excl_filtered]
            rng = random.Random(derive_seed(SEED, f"step43_{sid}_reddit_pushshift_fresh"))
            sampled = stratified_resample(excl_filtered, shock_dt, target, lexicons, scorer, rng)
            n_bloc_usable = sum(1 for p in sampled if p.get("bio_bloc", "unknown") != "unknown")

            out_path = OUTPUT_ROOT / sid / "reddit_pushshift.jsonl"
            SA.write_output(sampled, out_path, sid, "reddit_pushshift", SEED)
            total_written += len(sampled)

            report.setdefault(sid, {"tier": addressable[sid]["tier"], "archives": {}})
            report[sid]["archives"]["reddit_pushshift"] = {
                "source": "fresh_scan",
                "raw_window_narrowed": len(raw),
                "after_keyword_match": len(raw),  # fresh_scan already applied match_shocks
                "sampled": len(sampled),
                "bloc_usable_in_sample": n_bloc_usable,
            }
            logger.info(
                "%s/reddit_pushshift (fresh): matched=%d -> sampled=%d (bloc_usable=%d)",
                sid,
                len(raw),
                len(sampled),
                n_bloc_usable,
            )

    # ── Final report dump ───────────────────────────────────────────────────
    summary = {
        "priority_ids": priority_ids,
        "secondary_ids": secondary_ids,
        "unaddressable": unaddressable,
        "held_out_excluded_count": len(held_out_ids),
        "total_written": total_written,
        "per_shock": report,
        "coverage_before": {sid: addressable[sid]["non_reddit_usable"] for sid in addressable},
    }
    report_out = Path(sys.argv[1]) if len(sys.argv) > 1 else REPORT_PATH
    report_out.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    logger.info("Report written to %s", report_out)
    logger.info("TOTAL WRITTEN: %d", total_written)


if __name__ == "__main__":
    main()
