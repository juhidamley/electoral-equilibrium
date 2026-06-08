#!/usr/bin/env python3
"""parse_3dlnews_html.py — extract articles from 3DLNews-2.0-HTML.zip.

Reads HTML.gz files directly from the zip archive without full extraction,
decompresses each in memory, parses with BeautifulSoup, and writes canonical
posts.jsonl files (one per state per year) to --output-dir.

Usage
-----
    python scripts/parse_3dlnews_html.py \\
        --zip-path /Volumes/JUHIDRIVE/electoralData/archives/news/3DLNews-2.0-HTML.zip \\
        --states CA,TX,FL,NY,PA,OH,MI,GA,AZ,NV \\
        --years 2016,2017,2018,2019,2020,2021,2022,2023,2024 \\
        --output-dir /Volumes/JUHIDRIVE/electoralData/archives/news/3dlnews_parsed

Zip structure assumed
---------------------
    <source_type>/<STATE>/<YYYY>/<filename>.html.gz
e.g.
    3-TV/TX/2020/abc7news_2020-11-05_article.html.gz
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import logging
import zipfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

try:
    from bs4 import BeautifulSoup
except ImportError as exc:
    raise SystemExit("beautifulsoup4 is required: pip install beautifulsoup4") from exc

logger = logging.getLogger(__name__)

# ── Date parsing ───────────────────────────────────────────────────────────────

_DATE_FMTS = (
    "%Y-%m-%dT%H:%M:%S%z",
    "%Y-%m-%dT%H:%M:%SZ",
    "%Y-%m-%dT%H:%M:%S.%f%z",
    "%Y-%m-%dT%H:%M:%S.%fZ",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d",
    "%B %d, %Y",
    "%b %d, %Y",
    "%m/%d/%Y",
)


def _parse_date(value: str | None) -> str | None:
    """Return ISO 8601 UTC string or None."""
    if not value:
        return None
    value = value.strip()
    for fmt in _DATE_FMTS:
        try:
            dt = datetime.strptime(value, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.isoformat()
        except ValueError:
            continue
    return None


# ── HTML extraction ────────────────────────────────────────────────────────────


def _extract_article(html_bytes: bytes) -> dict:
    """Parse article HTML and return {title, body, date}."""
    soup = BeautifulSoup(html_bytes, "html.parser")

    # ── Title ──────────────────────────────────────────────────────────────────
    title = ""
    # Try Open Graph first, then <title>, then first <h1>
    og_title = soup.find("meta", property="og:title")
    if og_title and og_title.get("content"):
        title = og_title["content"].strip()
    elif soup.title and soup.title.string:
        title = soup.title.string.strip()
    else:
        h1 = soup.find("h1")
        if h1:
            title = h1.get_text(" ", strip=True)

    # ── Publication date ───────────────────────────────────────────────────────
    raw_date: str | None = None

    # 1. <meta property="article:published_time"> / <meta name="publishdate">
    for attr, key in (
        ("property", "article:published_time"),
        ("name", "publishdate"),
        ("name", "date"),
        ("name", "DC.date"),
        ("itemprop", "datePublished"),
    ):
        tag = soup.find("meta", attrs={attr: key})
        if tag and tag.get("content"):
            raw_date = tag["content"]
            break

    # 2. <time datetime="...">
    if not raw_date:
        time_tag = soup.find("time", attrs={"datetime": True})
        if time_tag:
            raw_date = time_tag["datetime"]

    # 3. JSON-LD
    if not raw_date:
        for script in soup.find_all("script", type="application/ld+json"):
            try:
                data = json.loads(script.string or "")
                if isinstance(data, dict):
                    raw_date = data.get("datePublished") or data.get("dateModified")
                    if raw_date:
                        break
            except (json.JSONDecodeError, AttributeError):
                continue

    date_iso = _parse_date(raw_date)

    # ── Body text ──────────────────────────────────────────────────────────────
    # Remove nav/header/footer/script/style/aside noise before extraction.
    for tag in soup(
        ["script", "style", "nav", "header", "footer", "aside", "form", "noscript", "iframe"]
    ):
        tag.decompose()

    body = ""
    # Prefer <article>, then main content containers, then <body>
    article_tag = soup.find("article") or soup.find(
        attrs={
            "class": lambda c: c
            and any(
                kw in " ".join(c).lower()
                for kw in (
                    "article-body",
                    "story-body",
                    "article__body",
                    "post-body",
                    "entry-content",
                    "content-body",
                )
            )
        }
    )
    if article_tag:
        body = article_tag.get_text(" ", strip=True)
    elif soup.body:
        body = soup.body.get_text(" ", strip=True)

    # Collapse runs of whitespace
    body = " ".join(body.split())

    return {"title": title, "body": body, "date_iso": date_iso}


# ── Zip walker ─────────────────────────────────────────────────────────────────


def _post_id(zip_member_name: str) -> str:
    """Stable 12-char hex ID from filename."""
    return hashlib.sha1(zip_member_name.encode()).hexdigest()[:12]


def _parse_zip_path(name: str) -> dict | None:
    """Parse state, year, and source_type from a zip member path.

    Actual zip structure:
        3DLNews-2.0-HTML/{source_type...}/HTML/{STATE}/{YEAR}/{hash}.html.gz

    The literal 'HTML' directory is the structural anchor. STATE is the
    component immediately after it; YEAR is the component after STATE.
    source_type is the first component between the zip root and 'HTML'.
    """
    parts = name.replace("\\", "/").split("/")
    try:
        html_idx = parts.index("HTML")
    except ValueError:
        return None

    # Need at least STATE and YEAR after 'HTML'
    if html_idx + 2 >= len(parts):
        return None

    state = parts[html_idx + 1]
    year = parts[html_idx + 2]

    if len(state) != 2 or not state.isupper():
        return None
    if len(year) != 4 or not year.isdigit():
        return None

    # source_type: first component between zip root (index 0) and HTML
    between = parts[1:html_idx]
    source_type = between[-1] if between else None

    return {"state": state, "year": year, "source_type": source_type}


def iter_articles(
    zip_path: Path,
    states: set[str],
    years: set[str],
    source_type: str | None,
) -> dict:
    """Yield parsed article dicts from matching zip members."""
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.namelist()
        logger.info("Zip contains %d members total", len(members))

        matched = 0
        for name in members:
            if not name.endswith(".html.gz"):
                continue

            parsed = _parse_zip_path(name)
            if parsed is None:
                continue

            state = parsed["state"]
            year = parsed["year"]
            stype = parsed["source_type"]

            if state not in states:
                continue
            if year not in years:
                continue
            if source_type is not None and stype != source_type:
                continue

            matched += 1
            try:
                compressed = zf.read(name)
                html_bytes = gzip.decompress(compressed)
            except Exception as exc:
                logger.warning("Failed to decompress %s: %s", name, exc)
                continue

            try:
                article = _extract_article(html_bytes)
            except Exception as exc:
                logger.warning("Failed to parse %s: %s", name, exc)
                continue

            title = article["title"]
            body = article["body"]
            text = f"{title} {body}".strip() if title else body
            if not text:
                logger.debug("No text extracted from %s — skipping", name)
                continue

            yield {
                "name": name,
                "state": state,
                "year": year,
                "text": text,
                "created_at": article["date_iso"],
                "post_id": _post_id(name),
            }

        logger.info("Matched %d .html.gz members for selected states/years", matched)


# ── Writer ─────────────────────────────────────────────────────────────────────


def write_outputs(
    articles,
    output_dir: Path,
) -> None:
    """Group articles by (state, year) and write one .jsonl per group."""
    output_dir.mkdir(parents=True, exist_ok=True)
    buckets: dict[tuple[str, str], list[dict]] = defaultdict(list)

    for a in articles:
        buckets[(a["state"], a["year"])].append(a)

    created_at_now = datetime.now(tz=timezone.utc).isoformat()
    total = 0

    for (state, year), group in sorted(buckets.items()):
        out_path = output_dir / state / f"{year}.jsonl"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            for a in group:
                envelope = {
                    "schema_version": "1.0",
                    "created_at": created_at_now,
                    "stage": "collect",
                    "seed": None,
                    "payload": {
                        "text": a["text"],
                        "created_at": a["created_at"],
                        "lang": "en",
                        "platform": "local_news",
                        "archive_id": "3dlnews",
                        "state": a["state"],
                        "source": "local_tv",
                        "author_description": None,
                        "post_id": a["post_id"],
                    },
                }
                f.write(json.dumps(envelope, ensure_ascii=False) + "\n")
        logger.info("Wrote %d articles → %s", len(group), out_path)
        total += len(group)

    logger.info("Total articles written: %d", total)


# ── CLI ────────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Parse 3DLNews HTML.gz files from zip into posts.jsonl")
    p.add_argument("--zip-path", required=True, type=Path, help="Path to 3DLNews-2.0-HTML.zip")
    p.add_argument("--states", required=True, help="Comma-separated state codes, e.g. CA,TX,FL")
    p.add_argument("--years", required=True, help="Comma-separated years, e.g. 2016,2017,2020")
    p.add_argument(
        "--output-dir", required=True, type=Path, help="Directory to write state/year .jsonl files"
    )
    p.add_argument(
        "--source-type",
        default=None,
        help="Filter to zip subdirectory e.g. '3-TV' or '1-Google' (default: all)",
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

    if not args.zip_path.exists():
        raise SystemExit(f"Zip not found: {args.zip_path}")

    states = {s.strip().upper() for s in args.states.split(",") if s.strip()}
    years = {y.strip() for y in args.years.split(",") if y.strip()}

    logger.info(
        "Parsing 3DLNews: %d states, %d years, source_type=%s",
        len(states),
        len(years),
        args.source_type or "all",
    )

    articles = iter_articles(
        zip_path=args.zip_path,
        states=states,
        years=years,
        source_type=args.source_type,
    )
    write_outputs(articles, args.output_dir)


if __name__ == "__main__":
    main()
