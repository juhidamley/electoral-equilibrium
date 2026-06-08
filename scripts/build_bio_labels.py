#!/usr/bin/env python3
"""build_bio_labels.py — auto-label user bios from downloaded archives for SetFit training.

Sources (all have confirmed bio fields):
  - election_2020 Kaggle (hashtag_joebiden.csv + hashtag_donaldtrump.csv) → user_description
  - covid_vaccine_2020 (covidvaccine.csv) → user_description
  - covid_pandemic_2020 (covid19_tweets.csv) → user_description
  - election_2016 (election_day_tweets.csv) → user.description
  - election_2012 (tweets.json) → user.description (nested)

Label method: keyword_auto (BioClassifier Stage 1 only).
Only bios with a clear majority bloc (>0.6 weight) in at least one stratum are kept.
Bios matching multiple strata contribute one record per matched stratum.

Output: data/bio_labels/labeled_bios.jsonl
Schema per line:
  {
    "bio": str,
    "race_bloc": str | null,       primary race bloc if confident
    "religion_bloc": str | null,   primary religion bloc if confident
    "gender_signal": "F"|"M"|null, derived from gender_weights
    "label_method": "keyword_auto",
    "source": str,                 archive slug
    "confidence": float,           max weight across matched strata
    "collected_at": ISO8601
  }

Usage:
    python scripts/build_bio_labels.py
    python scripts/build_bio_labels.py --target-per-bloc 200 --verbose
    python scripts/build_bio_labels.py --dry-run   # just print counts
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from electoral.nlp.bio_classifier import BioClassifier

logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────

ARCHIVE_ROOT = Path("/Volumes/JUHIDRIVE/electoralData/archives")
OUTPUT_PATH  = REPO_ROOT / "data" / "bio_labels" / "labeled_bios.jsonl"

# ── Thresholds ────────────────────────────────────────────────────────────────

CONFIDENCE_THRESHOLD = 0.55   # top-bloc weight must exceed this to be kept
TARGET_PER_BLOC      = 150    # aim for this many per bloc; script stops early

# ── Source definitions ────────────────────────────────────────────────────────

def _iter_csv_bios(path: Path, bio_col: str, source: str) -> Iterator[tuple[str, str]]:
    """Yield (bio, source) from a CSV file."""
    csv.field_size_limit(10_000_000)
    try:
        with open(path, encoding="utf-8", errors="replace", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                bio = (row.get(bio_col) or "").strip()
                if bio and len(bio) > 5:
                    yield bio, source
    except OSError as exc:
        logger.warning("Could not open %s: %s", path, exc)


def _iter_json_bios(path: Path, source: str) -> Iterator[tuple[str, str]]:
    """Yield (bio, source) from line-delimited Twitter JSON (user.description)."""
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    bio = (obj.get("user", {}) or {}).get("description", "") or ""
                    bio = bio.strip()
                    if bio and len(bio) > 5:
                        yield bio, source
                except (json.JSONDecodeError, AttributeError):
                    continue
    except OSError as exc:
        logger.warning("Could not open %s: %s", path, exc)


def _all_bio_sources() -> Iterator[tuple[str, str]]:
    """Yield (bio_text, source_label) from all available archives."""
    sources = [
        # (iterator, priority) — higher-volume sources first
        _iter_csv_bios(
            ARCHIVE_ROOT / "twitter" / "election_2020" / "hashtag_joebiden.csv",
            "user_description", "election_2020_joebiden",
        ),
        _iter_csv_bios(
            ARCHIVE_ROOT / "twitter" / "election_2020" / "hashtag_donaldtrump.csv",
            "user_description", "election_2020_trump",
        ),
        _iter_csv_bios(
            ARCHIVE_ROOT / "twitter" / "covid_vaccine_2020" / "covidvaccine.csv",
            "user_description", "covid_vaccine_2020",
        ),
        _iter_csv_bios(
            ARCHIVE_ROOT / "twitter" / "covid_pandemic_2020" / "covid19_tweets.csv",
            "user_description", "covid_pandemic_2020",
        ),
        _iter_csv_bios(
            ARCHIVE_ROOT / "twitter" / "election_2016" / "election_day_tweets.csv",
            "user.description", "election_2016",
        ),
        _iter_json_bios(
            ARCHIVE_ROOT / "twitter" / "election_2012" / "tweets.json",
            "election_2012",
        ),
    ]
    for src in sources:
        yield from src


# ── Label derivation ──────────────────────────────────────────────────────────

def _top_bloc(weights: dict[str, float], threshold: float) -> tuple[str | None, float]:
    """Return (top_bloc, weight) if above threshold, else (None, 0)."""
    if not weights:
        return None, 0.0
    top = max(weights, key=lambda k: weights[k])
    w = weights[top]
    return (top, w) if w >= threshold else (None, 0.0)


def _gender_signal(weights: dict[str, float]) -> str | None:
    """Map gender_weights to F / M / None."""
    if not weights:
        return None
    top, w = _top_bloc(weights, 0.5)
    if top == "women":
        return "F"
    if top == "men":
        return "M"
    return None   # other_gender or split


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Auto-label bios from archives for SetFit training")
    p.add_argument("--target-per-bloc", type=int, default=TARGET_PER_BLOC)
    p.add_argument("--confidence",      type=float, default=CONFIDENCE_THRESHOLD)
    p.add_argument("--output",          type=Path,  default=OUTPUT_PATH)
    p.add_argument("--dry-run",         action="store_true",
                   help="Print final counts only; do not write file")
    p.add_argument("--verbose",         action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    classifier = BioClassifier.from_config(
        config_path=REPO_ROOT / "configs" / "base.json"
    )

    # Track counts per bloc
    race_counts:     defaultdict[str, int] = defaultdict(int)
    religion_counts: defaultdict[str, int] = defaultdict(int)
    gender_counts:   defaultdict[str, int] = defaultdict(int)

    seen_hashes: set[str] = set()
    records: list[dict] = []
    total_scanned = 0

    def _all_satisfied() -> bool:
        """True when every non-null bloc has reached the target."""
        from electoral.core.types import CANONICAL_RACES, CANONICAL_RELIGIONS, CANONICAL_GENDERS
        race_ok    = all(race_counts[b]     >= args.target_per_bloc for b in CANONICAL_RACES)
        rel_ok     = all(religion_counts[b] >= args.target_per_bloc for b in CANONICAL_RELIGIONS)
        gender_ok  = all(gender_counts[b]   >= args.target_per_bloc for b in CANONICAL_GENDERS)
        return race_ok and rel_ok and gender_ok

    now_iso = datetime.now(tz=timezone.utc).isoformat()

    for bio, source in _all_bio_sources():
        if _all_satisfied():
            logger.info("All bloc targets reached — stopping early")
            break

        total_scanned += 1
        if total_scanned % 100_000 == 0:
            logger.info(
                "Scanned %s bios, collected %d records",
                f"{total_scanned:,}", len(records),
            )

        # Dedup by SHA-1 of lowercased bio
        bio_hash = hashlib.sha1(bio.lower().encode()).hexdigest()
        if bio_hash in seen_hashes:
            continue
        seen_hashes.add(bio_hash)

        result = classifier.classify(bio)

        race_bloc,     race_conf     = _top_bloc(result.race_weights,     args.confidence)
        religion_bloc, religion_conf = _top_bloc(result.religion_weights, args.confidence)
        gender_sig = _gender_signal(result.gender_weights)

        # Skip if nothing was detected above threshold
        if race_bloc is None and religion_bloc is None and gender_sig is None:
            continue

        # Respect per-bloc caps
        if race_bloc is not None and race_counts[race_bloc] >= args.target_per_bloc:
            race_bloc = None
        if religion_bloc is not None and religion_counts[religion_bloc] >= args.target_per_bloc:
            religion_bloc = None
        if gender_sig is not None:
            gen_bloc = "women" if gender_sig == "F" else ("men" if gender_sig == "M" else "other_gender")
            if gender_counts[gen_bloc] >= args.target_per_bloc:
                gender_sig = None
                gen_bloc = None
        else:
            gen_bloc = None

        if race_bloc is None and religion_bloc is None and gender_sig is None:
            continue

        if race_bloc:     race_counts[race_bloc]         += 1
        if religion_bloc: religion_counts[religion_bloc] += 1
        if gen_bloc:      gender_counts[gen_bloc]        += 1

        confidence = max(
            race_conf,
            religion_conf,
            max(result.gender_weights.values()) if result.gender_weights else 0.0,
        )

        records.append({
            "bio":            bio,
            "race_bloc":      race_bloc,
            "religion_bloc":  religion_bloc,
            "gender_signal":  gender_sig,
            "label_method":   "keyword_auto",
            "source":         source,
            "confidence":     round(confidence, 4),
            "collected_at":   now_iso,
        })

    logger.info("Scan complete — %s bios scanned, %d records collected", f"{total_scanned:,}", len(records))

    # Print summary
    from electoral.core.types import CANONICAL_RACES, CANONICAL_RELIGIONS, CANONICAL_GENDERS
    logger.info("── Race bloc counts ──────────────────────")
    for b in CANONICAL_RACES:
        logger.info("  %-22s %d", b, race_counts[b])
    logger.info("── Religion bloc counts ──────────────────")
    for b in CANONICAL_RELIGIONS:
        logger.info("  %-22s %d", b, religion_counts[b])
    logger.info("── Gender bloc counts ────────────────────")
    for b in CANONICAL_GENDERS:
        logger.info("  %-22s %d", b, gender_counts[b])

    if args.dry_run:
        logger.info("--dry-run: no file written")
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    logger.info("Wrote %d records → %s", len(records), args.output)


if __name__ == "__main__":
    main()
