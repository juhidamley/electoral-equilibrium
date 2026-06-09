#!/usr/bin/env python3
"""filter_reddit_monthly.py — filter Pushshift monthly Reddit bz2 dumps to politically
and demographically relevant posts.

Input:  --input-dir (default /Volumes/JUHIDRIVE/electoralData/archives/reddit_monthly/)
        OR --input-file <single.bz2>  (SLURM array mode — pass one file directly)
Output: --output-dir / {year} / {subreddit} / {MM}.jsonl
        Default: /scratch/JDamley28@cmc.edu/electoralData/archives/reddit/reddit_monthly_filtered/

A post is kept when its subreddit name matches any keyword in KEYWORD_FILTERS
(case-insensitive substring) OR appears in configs/reddit_target_subreddits.json.
Posts with body [deleted]/[removed] or under 10 characters are always skipped.
Empty output files are never written.

Usage
-----
List processable files (for SLURM array sizing):
    python scripts/filter_reddit_monthly.py --list-tasks

Process a single file (SLURM array mode — preferred):
    python scripts/filter_reddit_monthly.py --input-file RC_2020-11.bz2 [--output-dir ...] [--workers 8]

Process via task ID (legacy SLURM array mode):
    python scripts/filter_reddit_monthly.py --task-id 0 [--input-dir ...] [--output-dir ...]

Process all files in a year range:
    python scripts/filter_reddit_monthly.py --year-start 2017 --year-end 2020

Dry-run (discover subreddits without writing output):
    python scripts/filter_reddit_monthly.py --dry-run --year-start 2015 --year-end 2018
"""

from __future__ import annotations

import argparse
import bz2
import json
import logging
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Iterator

# ── Paths ─────────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parents[1]
TARGET_SUBREDDITS_PATH = REPO_ROOT / "configs" / "reddit_target_subreddits.json"

DEFAULT_INPUT_DIR = Path("/Volumes/JUHIDRIVE/electoralData/archives/reddit_monthly/")
DEFAULT_OUTPUT_DIR = Path(
    "/scratch/JDamley28@cmc.edu/electoralData/archives/reddit/reddit_monthly_filtered"
)

# ── Tuning constants ───────────────────────────────────────────────────────────

CHUNK_BYTES = 50 * 1024 * 1024       # 50 MB decompressed read chunks
WRITE_BUFFER_BYTES = 64 * 1024 * 1024  # 64 MB write buffer per output file
DEFAULT_WORKERS = 8
PROGRESS_INTERVAL = 500_000

# ── Filtering constants ────────────────────────────────────────────────────────

KEYWORD_FILTERS: list[str] = [
    "politics",
    "political",
    "democrat",
    "republican",
    "conservative",
    "liberal",
    "progressive",
    "catholic",
    "christian",
    "protestant",
    "evangelical",
    "baptist",
    "jewish",
    "jew",
    "islam",
    "muslim",
    "black",
    "african",
    "latino",
    "hispanic",
    "chicano",
    "asian",
    "atheist",
    "atheism",
    "secular",
    "feminist",
    "feminism",
    "religion",
    "religious",
    "race",
    "racial",
    "immigration",
    "immigrant",
    "civil",
    "rights",
    "vote",
    "voting",
    "election",
    "congress",
    "senate",
    "president",
    "supreme",
    "court",
    "gun",
    "abortion",
    "lgbtq",
    "gay",
]

_KW_RE = re.compile(
    "|".join(re.escape(k) for k in KEYWORD_FILTERS),
    re.IGNORECASE,
)

SKIP_BODIES: frozenset[str] = frozenset({"[deleted]", "[removed]", "[removed by reddit]"})
MIN_BODY_LEN = 10

logger = logging.getLogger(__name__)


# ── Subreddit filtering ────────────────────────────────────────────────────────


def load_target_subreddits(path: Path) -> set[str]:
    """Load optional list of known high-value subreddits from JSON config.

    Expected format: {"subreddits": ["subreddit1", "subreddit2", ...]}
    Returns an empty set if the file does not exist.
    """
    if not path.exists():
        return set()
    try:
        data = json.loads(path.read_text())
        subs = data.get("subreddits", [])
        logger.info("Loaded %d target subreddits from %s", len(subs), path)
        return set(subs)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not load %s: %s — proceeding without it", path, exc)
        return set()


def is_relevant_subreddit(subreddit: str, target_set: set[str]) -> bool:
    """Return True if subreddit matches any keyword or appears in target_set."""
    return bool(_KW_RE.search(subreddit)) or subreddit in target_set


# ── File discovery ─────────────────────────────────────────────────────────────


def _year_month_from_path(path: Path) -> str | None:
    """Extract YYYY-MM from a Pushshift filename like RC_2017-01.bz2."""
    m = re.search(r"(\d{4}-\d{2})", path.name)
    return m.group(1) if m else None


def list_input_files(
    input_dir: Path,
    year_start: int | None = None,
    year_end: int | None = None,
) -> list[Path]:
    """Return sorted list of .bz2 files in input_dir, optionally filtered by year range."""
    files = sorted(f for f in input_dir.glob("*.bz2") if f.is_file())
    if year_start is not None or year_end is not None:
        filtered = []
        for f in files:
            ym = _year_month_from_path(f)
            if ym is None:
                continue
            year = int(ym[:4])
            if year_start is not None and year < year_start:
                continue
            if year_end is not None and year > year_end:
                continue
            filtered.append(f)
        return filtered
    return files


# ── Chunk-based bz2 reader ─────────────────────────────────────────────────────


def _iter_bz2_lines_chunked(
    path: Path, chunk_bytes: int = CHUNK_BYTES
) -> Iterator[list[str]]:
    """Read a bz2 file in ~chunk_bytes decompressed byte chunks.

    Yields batches of complete line strings. Partial lines at chunk boundaries
    are carried over and prepended to the next chunk so no line is split.
    """
    leftover = b""
    try:
        with bz2.open(path, mode="rb") as fh:
            while True:
                raw = fh.read(chunk_bytes)
                if not raw:
                    break
                block = leftover + raw
                last_nl = block.rfind(b"\n")
                if last_nl == -1:
                    leftover = block
                    continue
                leftover = block[last_nl + 1 :]
                yield block[:last_nl].decode("utf-8", errors="replace").splitlines()
        if leftover:
            lines = leftover.decode("utf-8", errors="replace").splitlines()
            if lines:
                yield lines
    except (OSError, EOFError) as exc:
        logger.warning("Read error on %s: %s", path, exc)


# ── Parallel batch filter ──────────────────────────────────────────────────────


def _filter_batch(args: tuple[list[str], frozenset[str]]) -> list[dict]:
    """Worker: parse and filter a batch of raw JSON line strings.

    Must be a module-level function for multiprocessing pickling.
    Uses module-level _KW_RE, SKIP_BODIES, MIN_BODY_LEN via fork inheritance.
    """
    lines, target_set = args
    results: list[dict] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue

        subreddit = str(row.get("subreddit") or "").strip()
        if not subreddit or not is_relevant_subreddit(subreddit, target_set):
            continue

        body = str(row.get("body") or "").strip()
        if not body or body in SKIP_BODIES or len(body) < MIN_BODY_LEN:
            continue

        created_utc = row.get("created_utc")
        try:
            created_at = datetime.fromtimestamp(float(created_utc), tz=timezone.utc).isoformat()
        except (TypeError, ValueError, OSError):
            created_at = None

        results.append(
            {
                "text": body,
                "created_at": created_at,
                "username": str(row.get("author") or ""),
                "platform": subreddit,
                "post_id": str(row.get("id") or ""),
                "likes": row.get("score"),
                "author_description": str(row.get("author_flair_text") or "").strip() or None,
                "archive_id": "reddit_monthly_filtered",
            }
        )
    return results


# ── Per-file processing ────────────────────────────────────────────────────────


def process_file(
    path: Path,
    output_dir: Path,
    target_set: set[str],
    dry_run: bool = False,
    workers: int = DEFAULT_WORKERS,
) -> Counter:
    """Process one monthly bz2 file using a multiprocessing pool.

    Reads bz2 in 50 MB chunks, distributes line batches to workers for parallel
    filtering, then writes per-subreddit output to
    {output_dir}/{year}/{subreddit}/{MM}.jsonl with a 64 MB write buffer.
    Empty output files are never written.

    Returns Counter of {subreddit: posts_kept}.
    """
    year_month = _year_month_from_path(path)
    if year_month is None:
        logger.warning("Cannot extract year-month from %s — skipping", path.name)
        return Counter()

    year = year_month[:4]
    month = year_month[5:]  # MM

    logger.info(
        "Processing %s (year_month=%s, dry_run=%s, workers=%d)",
        path.name,
        year_month,
        dry_run,
        workers,
    )

    subreddit_counts: Counter = Counter()
    buffer: dict[str, list[str]] = defaultdict(list)
    total_lines = 0
    t_start = time.monotonic()
    frozen_target = frozenset(target_set)

    with Pool(workers) as pool:
        for chunk_lines in _iter_bz2_lines_chunked(path):
            n = len(chunk_lines)
            prev_total = total_lines
            total_lines += n

            # Log throughput every PROGRESS_INTERVAL lines
            if (total_lines // PROGRESS_INTERVAL) > (prev_total // PROGRESS_INTERVAL):
                elapsed = time.monotonic() - t_start
                rate = total_lines / elapsed if elapsed > 0 else 0
                logger.info(
                    "%s: %s lines scanned, %s posts kept, %.0f lines/s",
                    path.name,
                    f"{total_lines:,}",
                    f"{sum(subreddit_counts.values()):,}",
                    rate,
                )

            # Split chunk into one batch per worker for parallel filtering
            batch_size = max(1, len(chunk_lines) // workers)
            batches = [
                (chunk_lines[i : i + batch_size], frozen_target)
                for i in range(0, len(chunk_lines), batch_size)
            ]

            for records in pool.map(_filter_batch, batches):
                for rec in records:
                    sub = rec["platform"]
                    subreddit_counts[sub] += 1
                    if not dry_run:
                        buffer[sub].append(json.dumps(rec, ensure_ascii=False) + "\n")

    # Write buffered output with 64 MB write buffer; skip empty subreddits
    if not dry_run:
        for subreddit, lines in buffer.items():
            if not lines:
                continue
            out_path = output_dir / year / subreddit / f"{month}.jsonl"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "a", encoding="utf-8", buffering=WRITE_BUFFER_BYTES) as f:
                f.writelines(lines)

    total_kept = sum(subreddit_counts.values())
    elapsed_total = time.monotonic() - t_start
    logger.info(
        "%s: done — %s lines, %s posts kept across %d subreddits in %.1fs",
        path.name,
        f"{total_lines:,}",
        f"{total_kept:,}",
        len(subreddit_counts),
        elapsed_total,
    )
    top20 = subreddit_counts.most_common(20)
    logger.info("%s: top 20 subreddits by posts kept:", path.name)
    for rank, (sub, count) in enumerate(top20, 1):
        logger.info("  %2d. r/%-40s %s", rank, sub, f"{count:,}")

    return subreddit_counts


# ── CLI ────────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Filter Reddit monthly bz2 dumps")
    p.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument(
        "--input-file",
        type=Path,
        default=None,
        help="Process a single bz2 file directly (SLURM array mode; overrides --task-id)",
    )
    p.add_argument("--year-start", type=int, default=None)
    p.add_argument("--year-end", type=int, default=None)
    p.add_argument("--dry-run", action="store_true", help="Scan and report without writing output")
    p.add_argument(
        "--task-id",
        type=int,
        default=int(__import__("os").environ.get("SLURM_ARRAY_TASK_ID", -1)),
        help="Process only the Nth sorted bz2 file (legacy SLURM array mode)",
    )
    p.add_argument(
        "--list-tasks", action="store_true", help="Print file count for SLURM array sizing and exit"
    )
    p.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help="Multiprocessing pool size (default: 8)",
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

    # --input-file: single bz2 passed directly by SLURM array script
    if args.input_file is not None:
        if not args.input_file.exists():
            logger.error("Input file not found: %s", args.input_file)
            sys.exit(1)
        target_set = load_target_subreddits(TARGET_SUBREDDITS_PATH)
        process_file(args.input_file, args.output_dir, target_set, args.dry_run, args.workers)
        return

    input_dir = args.input_dir
    if not input_dir.exists():
        logger.error("Input directory not found: %s", input_dir)
        sys.exit(1)

    files = list_input_files(input_dir, args.year_start, args.year_end)

    if args.list_tasks:
        for i, f in enumerate(files):
            logger.info("  [%d] %s", i, f.name)
        print(len(files))
        return

    if not files:
        logger.error("No .bz2 files found in %s for the given year range", input_dir)
        sys.exit(1)

    target_set = load_target_subreddits(TARGET_SUBREDDITS_PATH)

    if args.task_id >= 0:
        # Legacy SLURM array mode: process one file by index
        if args.task_id >= len(files):
            logger.error("task-id %d out of range [0, %d)", args.task_id, len(files))
            sys.exit(1)
        process_file(files[args.task_id], args.output_dir, target_set, args.dry_run, args.workers)
    else:
        # Local mode: process all files in range
        grand_total: Counter = Counter()
        for path in files:
            counts = process_file(path, args.output_dir, target_set, args.dry_run, args.workers)
            grand_total.update(counts)

        logger.info(
            "All files done — %s total posts kept across %d subreddits",
            f"{sum(grand_total.values()):,}",
            len(grand_total),
        )


if __name__ == "__main__":
    main()
