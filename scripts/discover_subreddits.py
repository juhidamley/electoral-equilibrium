#!/usr/bin/env python3
"""discover_subreddits.py — scan Reddit monthly bz2 dumps in dry-run mode and produce
a ranked list of all subreddits that match the keyword filter, with post counts by year.

Useful for identifying which communities were active in early years before the
known-subreddit list was established.

Usage
-----
Scan 2008–2012 and show what was there:
    python scripts/discover_subreddits.py --year-start 2008 --year-end 2012

Scan everything and write a CSV report:
    python scripts/discover_subreddits.py --output-csv /tmp/subreddit_discovery.csv
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path

from filter_reddit_monthly import (
    DEFAULT_INPUT_DIR,
    TARGET_SUBREDDITS_PATH,
    load_target_subreddits,
    list_input_files,
    _year_month_from_path,
    _iter_bz2_lines,
    is_relevant_subreddit,
    SKIP_BODIES,
    MIN_BODY_LEN,
    PROGRESS_INTERVAL,
)

logger = logging.getLogger(__name__)


def discover(
    input_dir: Path,
    year_start: int | None,
    year_end: int | None,
    target_set: set[str],
) -> dict[str, Counter]:
    """Scan all bz2 files and return {subreddit: Counter({year: count})}."""
    files = list_input_files(input_dir, year_start, year_end)
    if not files:
        logger.error("No .bz2 files found in %s for the given year range", input_dir)
        return {}

    # subreddit → {year_str → count}
    by_subreddit: dict[str, Counter] = defaultdict(Counter)

    for path in files:
        year_month = _year_month_from_path(path)
        if not year_month:
            continue
        year = year_month[:4]
        logger.info("Scanning %s ...", path.name)
        total = 0
        kept = 0
        for row in _iter_bz2_lines(path):
            total += 1
            if total % PROGRESS_INTERVAL == 0:
                logger.info("  %s lines, %s matched so far", f"{total:,}", f"{kept:,}")
            subreddit = str(row.get("subreddit") or "").strip()
            if not subreddit or not is_relevant_subreddit(subreddit, target_set):
                continue
            body = str(row.get("body") or "").strip()
            if not body or body in SKIP_BODIES or len(body) < MIN_BODY_LEN:
                continue
            by_subreddit[subreddit][year] += 1
            kept += 1
        logger.info("  done — %s lines, %s matched", f"{total:,}", f"{kept:,}")

    return dict(by_subreddit)


def print_report(by_subreddit: dict[str, Counter], top_n: int = 50) -> None:
    """Print a ranked table: subreddit, total posts, breakdown by year."""
    totals = {sub: sum(years.values()) for sub, years in by_subreddit.items()}
    ranked = sorted(totals, key=lambda s: -totals[s])

    all_years = sorted({y for c in by_subreddit.values() for y in c})
    header = f"{'rank':>4}  {'subreddit':<40}  {'total':>8}  " + "  ".join(
        f"{y:>6}" for y in all_years
    )
    print(header)
    print("-" * len(header))
    for rank, sub in enumerate(ranked[:top_n], 1):
        yearly = by_subreddit[sub]
        yearly_str = "  ".join(f"{yearly.get(y, 0):>6}" for y in all_years)
        print(f"{rank:>4}  {sub:<40}  {totals[sub]:>8,}  {yearly_str}")


def write_csv(by_subreddit: dict[str, Counter], output_path: Path) -> None:
    all_years = sorted({y for c in by_subreddit.values() for y in c})
    totals = {sub: sum(years.values()) for sub, years in by_subreddit.items()}
    ranked = sorted(totals, key=lambda s: -totals[s])

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["subreddit", "total"] + all_years)
        for sub in ranked:
            yearly = by_subreddit[sub]
            writer.writerow([sub, totals[sub]] + [yearly.get(y, 0) for y in all_years])

    logger.info("CSV written to %s", output_path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Discover relevant subreddits in monthly dumps")
    p.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    p.add_argument("--year-start", type=int, default=None)
    p.add_argument("--year-end", type=int, default=None)
    p.add_argument("--top-n", type=int, default=50, help="Subreddits to show in ranked table")
    p.add_argument("--output-csv", type=Path, default=None,
                   help="Write full ranked table to this CSV path")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if not args.input_dir.exists():
        logger.error("Input directory not found: %s", args.input_dir)
        sys.exit(1)

    target_set = load_target_subreddits(TARGET_SUBREDDITS_PATH)
    by_subreddit = discover(args.input_dir, args.year_start, args.year_end, target_set)

    if not by_subreddit:
        logger.warning("No matching posts found.")
        return

    print_report(by_subreddit, args.top_n)

    if args.output_csv:
        write_csv(by_subreddit, args.output_csv)


if __name__ == "__main__":
    main()
