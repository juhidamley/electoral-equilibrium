#!/usr/bin/env python3
"""sample_archives.py — three-strata stratified sampler for electoral archives.

Three strata per (archive_id, shock_id) pair:
  1. Temporal   — equal budget per day across the shock window so the full arc of
                  public reaction is captured, not just the peak day.
  2. Demographic — within each day, sample proportionally by keyword-inferred bio
                  bloc so low-volume blocs (Jewish, Muslim) aren't drowned out.
  3. Sentiment  — oversample posts at sentiment extremes (VADER |compound| > 0.5)
                  by 2× weight; they carry more signal for fine-tuning than neutral ones.

Targets: ~200k posts per major archive dataset, ~5k per shock event per platform.

Usage
-----
List tasks (prints count, one line per task):
    python scripts/sample_archives.py --list-tasks

Single task (local dev):
    python scripts/sample_archives.py --task-id 0 --archive-root /Volumes/JUHIDRIVE/...

SLURM array (via scripts/hpc/sample_archives.slurm):
    sbatch --array=0-$N scripts/hpc/sample_archives.slurm
"""

from __future__ import annotations

import argparse
import bz2
import csv
import io
import json
import logging
import os
import random
import re
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator

try:
    import zstandard as _zstd_mod
except ImportError:
    _zstd_mod = None

try:
    import pyreadstat as _pyreadstat_mod
except ImportError:
    _pyreadstat_mod = None

# ── Paths ─────────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "base.json"
SHOCKS_PATH = REPO_ROOT / "configs" / "shocks.json"
RACE_LEXICON_PATH = REPO_ROOT / "configs" / "race_lexicon.json"
RELIGION_LEXICON_PATH = REPO_ROOT / "configs" / "religion_lexicon.json"
GENDER_LEXICON_PATH = REPO_ROOT / "configs" / "gender_lexicon.json"

# ── Sampling constants ────────────────────────────────────────────────────────

TARGET_PER_ARCHIVE = 200_000  # posts per standalone major archive dataset
TARGET_PER_SHOCK = 5_000  # posts per (shock, archive) pair
PRE_SHOCK_DAYS = 3  # days before shock date to include
EXTREMITY_THRESHOLD = 0.5  # VADER |compound| > this = extreme
EXTREMITY_WEIGHT = 2.0  # oversample multiplier for extreme posts
UNKNOWN_BLOC = "unknown"

# Relative sampling weights per bloc — CNN 2024 NEP (religion/gender) and
# Pew/Census (race). These shares OVERLAP: a person is simultaneously in a
# race bloc, a religion bloc, and a gender bloc. keyword_bio_bloc() assigns
# each post to exactly one primary bloc (highest-confidence lexicon hit), so
# these weights are used as proportional targets within whatever stratum the
# bio classifier lands in, not as mutually exclusive population fractions.
BLOC_SHARES: dict[str, float] = {
    # Religion (CNN 2024 NEP)
    "evangelical": 0.26,
    "protestant": 0.17,
    "catholic": 0.21,
    "secular": 0.24,
    "jewish": 0.02,
    "muslim": 0.01,
    "other_rel": 0.09,
    # Race (Pew / Census)
    "african_american": 0.11,
    "latino": 0.11,
    "asian": 0.03,
    "white": 0.71,
    "other_race": 0.04,
    # Gender (CNN 2024 NEP)
    "women": 0.53,
    "men": 0.47,
    "other_gender": 0.01,
}

# CSV field name candidates (tried in order, first match wins)
_TEXT_COLS = [
    "text",
    "tweet",
    "full_text",
    "status",
    "content",
    "body",
    "message",
    "post_text",
    "Text",
]
_DATE_COLS = [
    "created_at",
    "date",
    "timestamp",
    "time",
    "created",
    "posted_at",
    "Date",
    "time_created",
    "date_utc",
]
_BIO_COLS = [
    "user_description",
    "description",
    "bio",
    "user.description",
    "author_description",
    "user_bio",
    "profile_description",
]
_ID_COLS = ["id", "tweet_id", "post_id", "message_id", "_id", "ID", "tweetid"]

logger = logging.getLogger(__name__)


# ── Task list ─────────────────────────────────────────────────────────────────


def build_task_list(shocks_path: Path) -> list[dict]:
    """Return list of {shock_id, archive_id, shock_date, window_start, window_end, ...} dicts.

    One task per (shock_id, archive_id) pair for active shocks with archive_ids.
    Shocks with no archive_ids are skipped — nothing to sample yet.

    date_window.shock_date takes precedence over the top-level date field so that
    archives with partial coverage (e.g. truth_social_2024 ending before election day)
    sample against the correct anchor date. window_start and window_end are read from
    date_window if present, otherwise calculated from shock_date and shock_window_days.
    """
    shocks = json.loads(shocks_path.read_text())
    tasks = []
    for shock in shocks:
        if not shock.get("active", True):
            continue
        dw = shock.get("date_window", {})
        window_days = shock.get("shock_window_days", 14)
        shock_date_str = dw.get("shock_date") or shock["date"]
        shock_dt = datetime.fromisoformat(shock_date_str)
        window_start = dw.get("start") or (shock_dt - timedelta(days=PRE_SHOCK_DAYS)).strftime(
            "%Y-%m-%d"
        )
        window_end = dw.get("end") or (shock_dt + timedelta(days=window_days)).strftime("%Y-%m-%d")
        for archive_id in shock.get("archive_ids", []):
            tasks.append(
                {
                    "shock_id": shock["id"],
                    "archive_id": archive_id,
                    "shock_date": shock_date_str,
                    "window_start": window_start,
                    "window_end": window_end,
                    "window_days": window_days,
                    "target_blocs": shock.get("target_blocs", []),
                }
            )
    return tasks


# ── Lexicon loading and keyword bio pass ──────────────────────────────────────


def load_lexicons() -> dict[str, dict[str, dict[str, float]]]:
    """Load race/religion/gender lexicons. Returns {stratum: {keyword: {bloc: weight}}}."""
    result: dict[str, dict[str, dict[str, float]]] = {}
    for stratum, path in [
        ("race", RACE_LEXICON_PATH),
        ("religion", RELIGION_LEXICON_PATH),
        ("gender", GENDER_LEXICON_PATH),
    ]:
        if path.exists():
            data = json.loads(path.read_text())
            result[stratum] = {k.lower(): v for k, v in data.get("keywords", {}).items()}
        else:
            result[stratum] = {}
    return result


def keyword_bio_bloc(bio: str | None, lexicons: dict) -> str:
    """Priority-ordered keyword scan of author bio → best-matching canonical bloc.

    Checks lexicons in order: religion → race → gender. Returns the best match
    from the highest-priority lexicon that has any match at all; only falls through
    to the next lexicon if the current one has zero keyword hits. Within a lexicon,
    selects the bloc with the highest weight among all matching keywords.
    Falls back to UNKNOWN_BLOC if no lexicon matches anything.
    """
    if not bio:
        return UNKNOWN_BLOC
    bio_lower = bio.lower()
    for stratum in ("religion", "race", "gender"):
        stratum_kws = lexicons.get(stratum, {})
        best_bloc = UNKNOWN_BLOC
        best_weight = 0.0
        for keyword, weights in stratum_kws.items():
            if keyword in bio_lower:
                top_bloc = max(weights, key=lambda b: weights[b])
                top_weight = weights[top_bloc]
                if top_weight > best_weight:
                    best_weight = top_weight
                    best_bloc = top_bloc
        if best_bloc != UNKNOWN_BLOC:
            logger.debug("bio_bloc '%s' matched via stratum '%s'", best_bloc, stratum)
            return best_bloc
    return UNKNOWN_BLOC


# ── Sentiment scorer ──────────────────────────────────────────────────────────


def make_sentiment_scorer():
    """Return a text → VADER compound score function.

    Falls back to a zero-returning lambda if vaderSentiment is not installed
    (VADER is optional; the sampler still runs, just without sentiment extremity).
    """
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

        _analyzer = SentimentIntensityAnalyzer()
        return lambda text: _analyzer.polarity_scores(text[:512])["compound"]
    except ImportError:
        logger.warning(
            "vaderSentiment not installed — sentiment extremity stratum disabled. "
            "Install with: pip install vaderSentiment"
        )
        return lambda text: 0.0


# ── Timestamp parsing ─────────────────────────────────────────────────────────


def parse_timestamp(value: Any) -> datetime | None:
    """Parse a timestamp from int/float epoch, ISO string, or common date strings."""
    if value is None:
        return None
    try:
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(float(value), tz=timezone.utc)
        s = str(value).strip()
        if not s:
            return None
        # Fix malformed ISO timestamps where the seconds field has 3 digits
        # (e.g. "2020-06-01T00:00:030Z" → "2020-06-01T00:00:30Z").
        s = re.sub(r"(\d{2}):(\d{2}):0(\d{2})", r"\1:\2:\3", s)
        # Snowflake decode for Truth Social / Mastodon IDs embedded as _id
        # Caller responsibility to pass snowflake via explicit decode before calling here.
        for fmt in (
            "%Y-%m-%dT%H:%M:%S.%fZ",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%dT%H:%M:%S%z",
            "%Y-%m-%dT%H:%M:%S.%f%z",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d",
            "%a %b %d %H:%M:%S %z %Y",  # Twitter API v1: "Mon Nov 05 01:23:45 +0000 2012"
            "%d-%m-%Y %H:%M",  # covidvaccine.csv: "18-08-2020 12:55"
            "%d-%m-%Y",
        ):
            try:
                dt = datetime.strptime(s, fmt)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt
            except ValueError:
                continue
    except (OSError, OverflowError, ValueError):
        pass
    return None


def snowflake_to_dt(snowflake_id: Any) -> datetime | None:
    """Decode a Mastodon snowflake ID to UTC datetime."""
    try:
        return datetime.fromtimestamp((int(str(snowflake_id)) >> 16) / 1000.0, tz=timezone.utc)
    except (ValueError, TypeError, OSError, OverflowError):
        return None


def _months_in_window(
    shock_dt: datetime, pre_days: int, window_days: int, buffer_days: int = 30
) -> list[tuple[int, int]]:
    """Return (year, month) pairs for every calendar month that overlaps the window.

    Window: [shock_dt - pre_days - buffer_days, shock_dt + window_days + buffer_days]
    """
    start = shock_dt - timedelta(days=pre_days + buffer_days)
    end = shock_dt + timedelta(days=window_days + buffer_days)
    months: list[tuple[int, int]] = []
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        months.append((y, m))
        m += 1
        if m > 12:
            m, y = 1, y + 1
    return months


# ── Archive readers ───────────────────────────────────────────────────────────


def _pick_col(row: dict, candidates: list[str]) -> str | None:
    for c in candidates:
        if row.get(c):
            return str(row[c]).strip() or None
    return None


def _iter_jsonl(path: Path) -> Iterator[dict]:
    with open(path, encoding="utf-8", errors="replace") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                # Unwrap envelope if present
                if "payload" in row:
                    row = row["payload"]
                yield row
            except json.JSONDecodeError:
                logger.debug("JSON decode error at %s:%d", path, lineno)


def _iter_csv(path: Path, use_snowflake_date: bool = False) -> Iterator[dict]:
    """Iterate over a CSV/TSV, auto-detecting delimiter and yielding normalized dicts."""
    csv.field_size_limit(10_000_000)  # Zenodo Truth Social truths.tsv has large text fields
    delimiter = "\t" if path.suffix == ".tsv" else ","
    try:
        with open(path, encoding="utf-8", errors="replace", newline="") as f:
            content = f.read().replace("\x00", "")
        reader = csv.DictReader(content.splitlines(keepends=True), delimiter=delimiter)
        for row in reader:
            text = _pick_col(row, _TEXT_COLS)
            if not text:
                continue
            if use_snowflake_date:
                # Try _id / id as snowflake; fall through to _DATE_COLS if missing
                raw_id = _pick_col(row, ["_id", "id", "ID"])
                created_at = snowflake_to_dt(raw_id) if raw_id else None
                if created_at is None:
                    created_at = parse_timestamp(_pick_col(row, _DATE_COLS))
            else:
                created_at = parse_timestamp(_pick_col(row, _DATE_COLS))
            yield {
                "text": text,
                "created_at": created_at,
                "author_description": _pick_col(row, _BIO_COLS),
                "post_id": _pick_col(row, _ID_COLS) or "",
                "platform": "csv",
            }
    except (csv.Error, OSError) as exc:
        logger.warning("CSV read error on %s: %s", path, exc)


def _iter_zst(path: Path) -> Iterator[dict]:
    """Stream-decompress a .zst file and yield one parsed JSON dict per line.

    Skips lines that fail UTF-8 decode or JSON parse, logging a warning for each.
    Returns immediately (with a warning) if zstandard is not installed.
    """
    if _zstd_mod is None:
        logger.warning(
            "zstandard not installed — skipping %s. Install with: pip install zstandard",
            path,
        )
        return
    dctx = _zstd_mod.ZstdDecompressor()
    try:
        with open(path, "rb") as fh:
            with dctx.stream_reader(fh) as reader:
                text_stream = io.TextIOWrapper(reader, encoding="utf-8", errors="replace")
                for lineno, line in enumerate(text_stream, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError as exc:
                        logger.warning("%s line %d: JSON error (%s)", path.name, lineno, exc)
    except (OSError, _zstd_mod.ZstdError) as exc:
        logger.warning("zst read error on %s: %s", path, exc)


def _iter_bz2(path: Path) -> Iterator[dict]:
    """Open a .bz2 file in text mode and yield one parsed JSON dict per line."""
    try:
        with bz2.open(path, mode="rt", encoding="utf-8", errors="replace") as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    logger.warning("%s line %d: JSON error (%s)", path.name, lineno, exc)
    except (OSError, EOFError) as exc:
        logger.warning("bz2 read error on %s: %s", path, exc)


def _iter_pushshift_rows(
    raw_rows: Iterator[dict],
    path_name: str,
    cutoff_date: datetime | None,
) -> Iterator[dict]:
    """Apply Reddit Pushshift field mapping, filtering, early exit, and progress logging.

    Shared by both .zst and .bz2 readers. Assumes raw_rows yields one parsed
    JSON dict per Reddit post with Pushshift schema.
    """
    posts_found = 0
    for lineno, row in enumerate(raw_rows, 1):
        if lineno % 100_000 == 0:
            _ts = row.get("created_utc")
            _date_str = (
                datetime.utcfromtimestamp(float(_ts)).strftime("%Y-%m-%d")
                if _ts is not None
                else "unknown"
            )
            logger.debug(
                "%s: scanned %s lines, %s posts in window, current date %s",
                path_name,
                f"{lineno:,}",
                f"{posts_found:,}",
                _date_str,
            )
        body = str(row.get("body") or "").strip()
        if not body or body in ("[deleted]", "[removed]"):
            continue
        created_utc = row.get("created_utc")
        created_at = parse_timestamp(created_utc) if created_utc is not None else None
        if cutoff_date is not None and created_at is not None and created_at > cutoff_date:
            logger.info(
                "%s: post date %s exceeds cutoff %s — stopping early at line %s",
                path_name,
                created_at.strftime("%Y-%m-%d"),
                cutoff_date.strftime("%Y-%m-%d"),
                f"{lineno:,}",
            )
            break
        posts_found += 1
        yield {
            "text": body,
            "created_at": created_at,
            "author_description": str(row.get("author_flair_text") or "").strip() or None,
            "post_id": str(row.get("id") or ""),
            "platform": str(row.get("subreddit") or "reddit"),
            "likes": row.get("score"),
        }


def iter_archive(
    archive_dir: Path,
    archive_id: str,
    cutoff_date: datetime | None = None,
    shock_date: datetime | None = None,
    window_days: int = 14,
) -> Iterator[dict]:
    """Yield raw post dicts from an archive directory, handling JSONL, CSV, ZST, and BZ2 formats.

    cutoff_date: if set, .zst/.bz2 reading stops when a post's created_utc exceeds this date.
                 Use shock_window_end + 30 days. Has no effect on JSONL/CSV paths.
    shock_date / window_days: required for reddit_monthly smart file selection.
    Snowflake date decode is applied automatically for Truth Social archives.
    """
    if not archive_dir.exists():
        logger.warning("Archive directory not found: %s", archive_dir)
        return

    # ── reddit_monthly: open only the bz2 month-files that overlap the shock window ──
    if archive_id == "reddit_monthly":
        if shock_date is None:
            logger.warning(
                "reddit_monthly: shock_date not supplied — falling through to rglob (slow). "
                "Pass shock_date to iter_archive for targeted month selection."
            )
        else:
            months = _months_in_window(shock_date, PRE_SHOCK_DAYS, window_days, buffer_days=30)
            logger.info(
                "reddit_monthly: shock_date=%s, scanning %d month(s): %s",
                shock_date.strftime("%Y-%m-%d"),
                len(months),
                ", ".join(f"RC_{y}-{m:02d}.bz2" for y, m in months),
            )
            for year, month in months:
                bz2_path = archive_dir / str(year) / f"RC_{year}-{month:02d}.bz2"
                if not bz2_path.exists():
                    logger.debug("reddit_monthly: %s not found, skipping", bz2_path.name)
                    continue
                logger.info("reddit_monthly: reading %s", bz2_path)
                yield from _iter_pushshift_rows(_iter_bz2(bz2_path), bz2_path.name, cutoff_date)
            return

    # ── reddit_monthly_filtered: read pre-filtered JSONL by overlapping months ──
    # Layout: {archive_dir}/{year}/{subreddit}/{MM}.jsonl — canonical schema, no bz2.
    if archive_id == "reddit_monthly_filtered":
        if shock_date is None:
            logger.warning(
                "reddit_monthly_filtered: shock_date not supplied — falling through to rglob. "
                "Pass shock_date for targeted month selection."
            )
        else:
            months = _months_in_window(shock_date, PRE_SHOCK_DAYS, window_days, buffer_days=30)
            logger.info(
                "reddit_monthly_filtered: shock_date=%s, scanning %d month(s)",
                shock_date.strftime("%Y-%m-%d"),
                len(months),
            )
            for year, month in months:
                year_dir = archive_dir / str(year)
                if not year_dir.exists():
                    continue
                for sub_dir in sorted(year_dir.iterdir()):
                    if not sub_dir.is_dir():
                        continue
                    jsonl_path = sub_dir / f"{month:02d}.jsonl"
                    if jsonl_path.exists():
                        yield from _iter_jsonl(jsonl_path)
            return

    use_snowflake = archive_id.startswith("truth_social")

    for path in sorted(archive_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix in (".jsonl", ".json"):
            for row in _iter_jsonl(path):
                text = ""
                for _tf in ("text", "tweet", "full_text", "content", "body", "status"):
                    _v = str(row.get(_tf) or "").strip()
                    if _v:
                        text = _v
                        logger.debug("%s: text from field '%s'", path.name, _tf)
                        break
                if not text:
                    continue
                created_at = None
                for _df in ("created_at", "date", "timestamp"):
                    _v = row.get(_df)
                    if _v:
                        created_at = parse_timestamp(_v)
                        logger.debug("%s: created_at from field '%s'", path.name, _df)
                        break
                yield {
                    "text": text,
                    "created_at": created_at,
                    "author_description": row.get("author_description")
                    or row.get("user_description")
                    or row.get("description")
                    or row.get("author"),
                    "post_id": str(row.get("id") or row.get("post_id") or ""),
                    "platform": row.get("platform", "unknown"),
                }
        elif path.suffix in (".csv", ".tsv"):
            yield from _iter_csv(path, use_snowflake_date=use_snowflake)
        elif path.suffix == ".zst":
            yield from _iter_pushshift_rows(_iter_zst(path), path.name, cutoff_date)
        elif path.suffix == ".bz2":
            yield from _iter_pushshift_rows(_iter_bz2(path), path.name, cutoff_date)
        elif path.suffix == ".sav":
            if _pyreadstat_mod is None:
                logger.warning(
                    "pyreadstat not installed — skipping %s. Install with: pip install pyreadstat",
                    path,
                )
                continue
            # Event code → (shock_id, known_date) for the LIWC MeToo/Kavanaugh dataset
            _SAV_EVENT_MAP: dict[int, tuple[str, str]] = {
                1: ("metoo_2017", "2017-10-15"),
                2: ("kavanaugh_ford", "2018-09-27"),
                3: ("kavanaugh_confirmed", "2018-10-06"),
                4: ("weinstein_convicted", "2020-02-24"),
            }
            try:
                df, _ = _pyreadstat_mod.read_sav(str(path))
            except Exception as exc:
                logger.warning("pyreadstat read error on %s: %s", path, exc)
                continue
            logger.debug("%s: %d rows, columns: %s", path.name, len(df), list(df.columns))
            for row_idx, row in df.iterrows():
                text = str(row.get("Tweet") or "").strip()
                if not text:
                    continue
                event_code = int(row["Event"]) if "Event" in df.columns else None
                shock_id, date_str = _SAV_EVENT_MAP.get(event_code, (None, None))
                created_at = (
                    datetime.fromisoformat(date_str).replace(tzinfo=timezone.utc)
                    if date_str
                    else None
                )
                yield {
                    "text": text,
                    "created_at": created_at,
                    "author_description": None,
                    "post_id": str(row_idx),
                    "platform": "twitter_liwc",
                    "shock_id": shock_id,
                }


# ── Three-strata sampler ──────────────────────────────────────────────────────


def temporal_day_key(created_at: datetime | str | None, shock_date: datetime) -> int | None:
    """Return integer day offset from shock_date, or None if outside window or missing."""
    if created_at is None:
        return None
    if isinstance(created_at, str):
        try:
            created_at = datetime.fromisoformat(created_at)
        except ValueError:
            return None
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=timezone.utc)
    return (created_at.date() - shock_date.date()).days


def sample_archive(
    archive_dir: Path,
    archive_id: str,
    shock_date: datetime,
    window_days: int,
    target: int,
    lexicons: dict,
    scorer,
    rng: random.Random,
    window_start_dt: datetime | None = None,
    window_end_dt: datetime | None = None,
) -> list[dict]:
    """Draw a stratified sample from an archive directory.

    window_start_dt / window_end_dt: explicit window boundaries from date_window.
    When provided, these override the PRE_SHOCK_DAYS / window_days defaults so that
    archives with non-standard coverage (e.g. election_2024 Truth Social) sample
    the correct date range relative to their archive-specific shock_date anchor.

    Returns a list of post dicts with added fields:
        day_offset, bio_bloc, sentiment_score, sampled_weight
    """
    # ── Load and annotate all posts within the temporal window ────────────────
    pre = (
        (window_start_dt.date() - shock_date.date()).days
        if window_start_dt is not None
        else -PRE_SHOCK_DAYS
    )
    post = (
        (window_end_dt.date() - shock_date.date()).days
        if window_end_dt is not None
        else window_days
    )

    bucket_posts: dict[int, list[dict]] = defaultdict(list)
    total_loaded = 0

    cutoff = (
        (window_end_dt + timedelta(days=30))
        if window_end_dt is not None
        else (shock_date + timedelta(days=window_days + 30))
    )
    for raw in iter_archive(
        archive_dir,
        archive_id,
        cutoff_date=cutoff,
        shock_date=shock_date,
        window_days=window_days,
    ):
        day = temporal_day_key(raw["created_at"], shock_date)
        if day is None or day < pre or day > post:
            continue
        bio_bloc = keyword_bio_bloc(raw.get("author_description"), lexicons)
        sentiment = scorer(raw["text"])
        raw["day_offset"] = day
        raw["bio_bloc"] = bio_bloc
        raw["sentiment_score"] = sentiment
        raw["sampled_weight"] = EXTREMITY_WEIGHT if abs(sentiment) >= EXTREMITY_THRESHOLD else 1.0
        bucket_posts[day].append(raw)
        total_loaded += 1

    if total_loaded == 0:
        logger.warning("No posts found in shock window for archive %s", archive_id)
        return []

    n_days = len(bucket_posts)
    target_per_day = max(1, target // n_days)

    sampled: list[dict] = []

    for day_offset, posts in sorted(bucket_posts.items()):
        # ── Stratum 2: group by bio_bloc, compute proportional targets ────────
        by_bloc: dict[str, list[dict]] = defaultdict(list)
        for p in posts:
            by_bloc[p["bio_bloc"]].append(p)

        # Blocs present in this day's posts; unknown gets equal share
        present_blocs = list(by_bloc.keys())
        # Compute unnormalized share for each present bloc
        raw_shares = {b: BLOC_SHARES.get(b, 1.0 / len(present_blocs)) for b in present_blocs}
        total_share = sum(raw_shares.values())
        norm_shares = {b: s / total_share for b, s in raw_shares.items()}

        day_sample: list[dict] = []

        for bloc, posts_in_bloc in by_bloc.items():
            bloc_target = max(1, round(target_per_day * norm_shares[bloc]))
            if len(posts_in_bloc) <= bloc_target:
                day_sample.extend(posts_in_bloc)
                continue
            # ── Stratum 3: weighted sampling toward sentiment extremes ────────
            weights = [p["sampled_weight"] for p in posts_in_bloc]
            chosen = rng.choices(posts_in_bloc, weights=weights, k=bloc_target)
            day_sample.extend(chosen)

        sampled.extend(day_sample)

    # Trim to target if we over-sampled (possible when many blocs all below quota)
    if len(sampled) > target:
        rng.shuffle(sampled)
        sampled = sampled[:target]

    return sampled


# ── Output writer ─────────────────────────────────────────────────────────────


def write_output(
    posts: list[dict],
    output_path: Path,
    shock_id: str,
    archive_id: str,
    seed: int,
) -> None:
    """Write sampled posts as JSONL with canonical envelope schema."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    created_at_str = datetime.now(tz=timezone.utc).isoformat()
    with open(output_path, "w", encoding="utf-8") as f:
        for post in posts:
            envelope = {
                "schema_version": "1.0",
                "created_at": created_at_str,
                "stage": "sample",
                "seed": seed,
                "payload": {
                    "text": post["text"],
                    "created_at": (
                        post["created_at"].isoformat()
                        if isinstance(post.get("created_at"), datetime)
                        else None
                    ),
                    "lang": post.get("lang", "en"),
                    "platform": post.get("platform", "unknown"),
                    "archive_id": archive_id,
                    "shock_id": shock_id,
                    "bio_bloc": post.get("bio_bloc", UNKNOWN_BLOC),
                    "sentiment_score": post.get("sentiment_score", 0.0),
                    "day_offset": post.get("day_offset"),
                    "author_description": post.get("author_description"),
                    "post_id": post.get("post_id", ""),
                },
            }
            f.write(json.dumps(envelope, ensure_ascii=False) + "\n")
    logger.info("Wrote %d posts → %s", len(posts), output_path)


# ── CLI ───────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stratified archive sampler")
    p.add_argument(
        "--task-id",
        type=int,
        default=int(os.environ.get("SLURM_ARRAY_TASK_ID", -1)),
        help="Array task index (default: $SLURM_ARRAY_TASK_ID)",
    )
    p.add_argument(
        "--list-tasks",
        action="store_true",
        help="Print task count (used by Justfile to set array bounds)",
    )
    p.add_argument(
        "--archive-root",
        type=Path,
        default=None,
        help="Root of archive directories (default: base.json data_root)",
    )
    p.add_argument(
        "--output-dir", type=Path, default=None, help="Output directory for sampled JSONL files"
    )
    p.add_argument("--target-per-shock", type=int, default=TARGET_PER_SHOCK)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument(
        "--archive-filter",
        type=str,
        default=None,
        metavar="ARCHIVE_ID",
        help="Only process tasks whose archive_id matches this value (e.g. reddit_monthly_filtered)",
    )
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Load config
    config = json.loads(CONFIG_PATH.read_text()) if CONFIG_PATH.exists() else {}
    base_seed = args.seed if args.seed is not None else config.get("seed", 42)

    # Archive root: CLI arg > base.json data_root > fallback
    if args.archive_root is not None:
        archive_root = args.archive_root
    else:
        root_str = config.get("data_root", "data/archives/")
        archive_root = Path(root_str) if Path(root_str).is_absolute() else REPO_ROOT / root_str

    # Output dir: CLI arg > {archive_root}/../sampled/
    if args.output_dir is not None:
        output_dir = args.output_dir
    else:
        output_dir = archive_root.parent / "sampled"

    # Build task list
    if not SHOCKS_PATH.exists():
        logger.error("shocks.json not found at %s", SHOCKS_PATH)
        sys.exit(1)
    tasks = build_task_list(SHOCKS_PATH)

    if args.archive_filter:
        tasks = [t for t in tasks if t["archive_id"] == args.archive_filter]
        logger.info("archive-filter=%s: %d tasks after filtering", args.archive_filter, len(tasks))

    if args.list_tasks:
        # Print count for Justfile array calculation
        for i, t in enumerate(tasks):
            logger.info("  [%d] shock=%s archive=%s", i, t["shock_id"], t["archive_id"])
        print(len(tasks))
        return

    if not tasks:
        logger.error("No active shocks with archive_ids found in shocks.json")
        sys.exit(1)

    # Select task
    task_id = args.task_id
    if task_id < 0 or task_id >= len(tasks):
        logger.error(
            "task-id %d out of range [0, %d). Use --list-tasks to see available tasks.",
            task_id,
            len(tasks),
        )
        sys.exit(1)

    task = tasks[task_id]
    shock_id = task["shock_id"]
    archive_id = task["archive_id"]
    shock_date = datetime.fromisoformat(task["shock_date"]).replace(tzinfo=timezone.utc)
    window_days = task["window_days"]
    window_start_dt = datetime.fromisoformat(task["window_start"]).replace(tzinfo=timezone.utc)
    window_end_dt = datetime.fromisoformat(task["window_end"]).replace(tzinfo=timezone.utc)

    logger.info(
        "Task %d/%d: shock=%s archive=%s shock_date=%s window=[%s, %s]",
        task_id,
        len(tasks),
        shock_id,
        archive_id,
        task["shock_date"],
        task["window_start"],
        task["window_end"],
    )

    # Determine archive directory — search platform subdirectories
    # Special cases: logical archive_ids that map to a fixed folder path.
    _ARCHIVE_ID_OVERRIDES: dict[str, Path] = {
        "reddit_pushshift": archive_root / "reddit" / "reddit_pushshift",
        # reddit_monthly: Pushshift monthly dumps organised by date rather than subreddit.
        # Contains posts from ALL subreddits — filter post['platform'] (subreddit name)
        # after sampling to isolate target communities for each shock.
        "reddit_monthly": archive_root / "reddit" / "reddit_monthly",
        # Pre-filtered output from filter_reddit_monthly.py.
        # Layout: {archive_dir}/{year}/{subreddit}/{MM}.jsonl — already in canonical schema.
        "reddit_monthly_filtered": archive_root / "reddit" / "reddit_monthly_filtered",
        "truth_social_2024": archive_root / "truthsocial" / "kashish_usc",
        "truth_social_2022": archive_root / "truthsocial" / "zenodo_notredame",
        "truth_social_2025": archive_root / "truthsocial" / "notmooodoo9",
        # 3dlnews: pre-parsed JSONL output from scripts/parse_3dlnews_html.py.
        # Source is 3DLNews-2.0-HTML.zip; state used as geographic demographic proxy.
        "3dlnews": archive_root / "news" / "3dlnews_parsed",
        "discord": archive_root / "discord" / "sampled",
    }

    archive_dir: Path | None = None
    if archive_id in _ARCHIVE_ID_OVERRIDES:
        override = _ARCHIVE_ID_OVERRIDES[archive_id]
        if override.exists():
            archive_dir = override
            logger.debug("archive_id '%s' mapped to override path: %s", archive_id, override)
            if archive_id == "reddit_monthly":
                logger.info(
                    "reddit_monthly: archive contains posts from all subreddits — "
                    "filter post['platform'] (subreddit name) after sampling to "
                    "isolate target communities for this shock."
                )
        else:
            logger.warning(
                "Override path for archive_id=%s does not exist: %s — skipping task.",
                archive_id,
                override,
            )
            return
    else:
        for platform_subdir in archive_root.glob("*/"):
            candidate = platform_subdir / archive_id
            if candidate.exists():
                archive_dir = candidate
                break
        if archive_dir is None:
            # Try flat layout (archive_id directly under archive_root)
            flat = archive_root / archive_id
            if flat.exists():
                archive_dir = flat
            else:
                logger.error(
                    "Archive directory not found for archive_id=%s under %s",
                    archive_id,
                    archive_root,
                )
                sys.exit(1)

    logger.info("Archive directory: %s", archive_dir)

    # Deterministic seed per task (seed contract from CLAUDE.md)
    task_seed = base_seed ^ (task_id * 2654435761 & 0xFFFFFFFF)
    rng = random.Random(task_seed)

    lexicons = load_lexicons()
    scorer = make_sentiment_scorer()

    posts = sample_archive(
        archive_dir=archive_dir,
        archive_id=archive_id,
        shock_date=shock_date,
        window_days=window_days,
        target=args.target_per_shock,
        lexicons=lexicons,
        scorer=scorer,
        rng=rng,
        window_start_dt=window_start_dt,
        window_end_dt=window_end_dt,
    )

    logger.info("Sampled %d posts (target=%d)", len(posts), args.target_per_shock)

    if not posts:
        logger.warning("No posts sampled — output file not written.")
        return

    output_path = output_dir / shock_id / f"{archive_id}.jsonl"
    write_output(posts, output_path, shock_id=shock_id, archive_id=archive_id, seed=task_seed)


if __name__ == "__main__":
    main()
