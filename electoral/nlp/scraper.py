"""Nightly news scraper — runs on the Intel Mac via launchd (2 am).

Targets: Christianity Today, CBN, Univision, NYT, WaPo, Fox News.

Output root:
    ~/juhidamley/electoral-sync/rawdata/articles/{outlet}/{YYYY-MM-DD}.jsonl

Each record is a canonical JSONL envelope (schema_version 1.0) whose
payload matches the article schema below. shock_id and inference_method
are null at collection time; the scorer and bio classifier fill them later.

Dedup: URL SHA-256 hash (first 16 hex chars) is the record id. Before
writing to a daily file, existing ids are loaded — so re-running the
scraper on the same day never duplicates a record or overwrites a file.

Usage:
    python -m electoral.nlp.scraper              # normal nightly run
    python -m electoral.nlp.scraper --dry-run    # log but do not write
    python -m electoral.nlp.scraper --max 5      # cap articles per outlet

launchd stdout/stderr both land in the configured log file. Every line is
prefixed with an ISO-8601 timestamp so the audit trail is unambiguous.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from bs4 import BeautifulSoup

# ── Paths ─────────────────────────────────────────────────────────────────────

# Override via env var ELECTORAL_DATA_ROOT for testing or alternate deployments.
_env_root = os.environ.get("ELECTORAL_DATA_ROOT")
DATA_ROOT: Path = (
    Path(_env_root).expanduser().resolve()
    if _env_root
    else (Path.home() / "juhidamley" / "electoral-sync" / "rawdata" / "articles")
)

# ── Tuning constants ──────────────────────────────────────────────────────────

SCHEMA_VERSION = "1.0"
MIN_WORD_COUNT = 100  # articles below this are discarded
REQUEST_TIMEOUT = 20  # seconds; applies to both feed and article fetches
INTER_OUTLET_DELAY = 2.0  # seconds between outlets (politeness)
INTER_ARTICLE_DELAY = 0.5  # seconds between article fetches within an outlet

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
    force=True,
)
log = logging.getLogger("electoral.scraper")

# ── HTTP headers ──────────────────────────────────────────────────────────────

_HEADERS: dict[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; ElectoralEquilibriumBot/1.0; " "research project, non-commercial)"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

# ── Outlet registry ───────────────────────────────────────────────────────────

# content_selectors: tried in order; first match with ≥30 words wins.
# A generic <p>-tag fallback is always attempted last.
OUTLETS: list[dict[str, Any]] = [
    {
        "slug": "christianity_today",
        "display": "Christianity Today",
        "feed_url": "https://www.christianitytoday.com/ct/sections/news.rss",
        "content_selectors": [
            "div.article-body",
            "div.entry-content",
            "main article",
        ],
    },
    {
        "slug": "cbn",
        "display": "CBN News",
        "feed_url": "https://www.cbn.com/cbnnews/politics/feed/",
        "content_selectors": [
            "div.article-body",
            "div.field-item",
            "main article",
        ],
    },
    {
        "slug": "univision",
        "display": "Univision",
        "feed_url": "https://www.univision.com/rss/feed.xml",
        "content_selectors": [
            "div[class*='article-body']",
            "div[class*='body-content']",
            "article",
        ],
    },
    {
        "slug": "nyt",
        "display": "New York Times",
        "feed_url": "https://rss.nytimes.com/services/xml/rss/nyt/Politics.xml",
        "content_selectors": [
            "section[name='articleBody']",
            "div.StoryBodyCompanionColumn",
            "article",
        ],
    },
    {
        "slug": "wapo",
        "display": "Washington Post",
        "feed_url": "https://feeds.washingtonpost.com/rss/politics",
        "content_selectors": [
            "div.article-body",
            "div[data-pb-type='art']",
            "article",
        ],
    },
    {
        "slug": "fox",
        "display": "Fox News",
        "feed_url": "https://feeds.foxnews.com/foxnews/politics",
        "content_selectors": [
            "div.article-body",
            "div.page-content",
            "article",
        ],
    },
]

# ── HTTP ──────────────────────────────────────────────────────────────────────


def _fetch(url: str) -> requests.Response | None:
    """GET a URL; return the Response or None on any failure."""
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        return resp
    except requests.RequestException as exc:
        log.warning("fetch failed (%s): %s", type(exc).__name__, url)
        return None


# ── Feed parsing ──────────────────────────────────────────────────────────────

_ATOM_NS = "http://www.w3.org/2005/Atom"


def _parse_feed(xml_bytes: bytes) -> list[dict[str, str]]:
    """Parse RSS 2.0 or Atom feed bytes into [{url, title, published_at}]."""
    items: list[dict[str, str]] = []
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError as exc:
        log.warning("feed XML parse error: %s", exc)
        return items

    # RSS 2.0 — <item> elements
    for item in root.iter("item"):
        url = (item.findtext("link") or "").strip()
        if not url:
            continue
        items.append(
            {
                "url": url,
                "title": (item.findtext("title") or "").strip(),
                "published_at": (item.findtext("pubDate") or "").strip(),
            }
        )

    # Atom fallback — <entry> elements
    if not items:
        for entry in root.iter(f"{{{_ATOM_NS}}}entry"):
            link_el = entry.find(f"{{{_ATOM_NS}}}link[@rel='alternate']")
            if link_el is None:
                link_el = entry.find(f"{{{_ATOM_NS}}}link")
            url = (link_el.get("href", "") if link_el is not None else "").strip()
            if not url:
                continue
            title_el = entry.find(f"{{{_ATOM_NS}}}title")
            pub_el = entry.find(f"{{{_ATOM_NS}}}published") or entry.find(f"{{{_ATOM_NS}}}updated")
            items.append(
                {
                    "url": url,
                    "title": (title_el.text or "").strip() if title_el is not None else "",
                    "published_at": (pub_el.text or "").strip() if pub_el is not None else "",
                }
            )

    return items


# ── Text extraction ───────────────────────────────────────────────────────────

_NOISE_TAGS = frozenset(
    [
        "script",
        "style",
        "nav",
        "header",
        "footer",
        "aside",
        "figure",
        "figcaption",
        "iframe",
        "noscript",
        "form",
    ]
)


def _extract_text(html: str, selectors: list[str]) -> str:
    """Extract article body text from raw HTML.

    Tries each CSS selector in order; the first match that yields ≥30 words
    is returned. Falls back to all <p> tags in the page.
    """
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(_NOISE_TAGS):
        tag.decompose()

    for sel in selectors:
        el = soup.select_one(sel)
        if el is not None:
            candidate = el.get_text(separator=" ", strip=True)
            if len(candidate.split()) >= 30:
                return candidate

    # Generic fallback: stitch <p> elements
    paras = [p.get_text(separator=" ", strip=True) for p in soup.find_all("p")]
    return " ".join(p for p in paras if p)


def _word_count(text: str) -> int:
    return len(text.split())


# ── URL hashing ───────────────────────────────────────────────────────────────


def _url_hash(url: str) -> str:
    """16-char hex SHA-256 prefix — stable record ID and dedup key."""
    return hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]


# ── JSONL I/O ─────────────────────────────────────────────────────────────────


def _load_seen_ids(path: Path) -> set[str]:
    """Return IDs of records already present in a JSONL file.

    Tolerates partial/corrupt lines without crashing.
    """
    seen: set[str] = set()
    if not path.exists():
        return seen
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    record_id = record.get("payload", {}).get("id")
                    if record_id:
                        seen.add(str(record_id))
                except json.JSONDecodeError:
                    continue
    except OSError as exc:
        log.warning("could not read existing JSONL %s: %s", path, exc)
    return seen


def _build_record(
    *,
    url: str,
    title: str,
    text: str,
    published_at: str,
    outlet_slug: str,
    word_count: int,
) -> dict[str, Any]:
    """Build the canonical JSONL envelope for one news article."""
    return {
        "schema_version": SCHEMA_VERSION,
        "stage": "collect",
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "seed": None,  # applied downstream by the pipeline
        "payload": {
            "id": _url_hash(url),
            "url": url,
            "title": title,
            "text": text,
            "published_at": published_at,
            "outlet": outlet_slug,
            "platform": "news",
            "source": "scrape",
            "word_count": word_count,
            "lang": "en",
            "shock_id": None,  # filled by scorer
            "inference_method": None,  # filled by bio classifier
        },
    }


def _append_record(path: Path, record: dict[str, Any], dry_run: bool) -> None:
    if dry_run:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False))
        fh.write("\n")


# ── Per-outlet scrape ─────────────────────────────────────────────────────────


def scrape_outlet(
    outlet: dict[str, Any],
    date_str: str,
    *,
    dry_run: bool = False,
    max_articles: int | None = None,
) -> dict[str, int]:
    """Scrape one outlet.

    Returns:
        {"attempted": int, "discarded": int, "written": int}
    """
    slug = outlet["slug"]
    display = outlet["display"]
    selectors: list[str] = outlet["content_selectors"]
    out_path = DATA_ROOT / slug / f"{date_str}.jsonl"

    counts: dict[str, int] = {"attempted": 0, "discarded": 0, "written": 0}

    log.info("[%s] fetching feed: %s", display, outlet["feed_url"])
    feed_resp = _fetch(outlet["feed_url"])
    if feed_resp is None:
        log.error("[%s] feed unreachable — skipping outlet", display)
        return counts

    items = _parse_feed(feed_resp.content)
    if not items:
        log.warning("[%s] feed returned 0 items", display)
        return counts

    log.info("[%s] feed: %d items found", display, len(items))
    seen_ids = _load_seen_ids(out_path)

    for i, item in enumerate(items):
        if max_articles is not None and counts["written"] >= max_articles:
            log.info("[%s] --max %d reached, stopping", display, max_articles)
            break

        url = item["url"]
        article_id = _url_hash(url)
        counts["attempted"] += 1

        if article_id in seen_ids:
            log.debug("[%s] skip already-written: %s", display, url)
            continue

        if i > 0:
            time.sleep(INTER_ARTICLE_DELAY)

        page_resp = _fetch(url)
        if page_resp is None:
            counts["discarded"] += 1
            continue

        text = _extract_text(page_resp.text, selectors)
        wc = _word_count(text)

        if wc < MIN_WORD_COUNT:
            log.debug("[%s] discard word_count=%d < %d: %s", display, wc, MIN_WORD_COUNT, url)
            counts["discarded"] += 1
            continue

        record = _build_record(
            url=url,
            title=item["title"],
            text=text,
            published_at=item["published_at"],
            outlet_slug=slug,
            word_count=wc,
        )
        _append_record(out_path, record, dry_run=dry_run)
        seen_ids.add(article_id)
        counts["written"] += 1
        log.debug("[%s] wrote word_count=%d: %s", display, wc, url)

    return counts


# ── CLI entry point ───────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Electoral Equilibrium nightly news scraper.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch and parse but do not write any files.",
    )
    parser.add_argument(
        "--max",
        dest="max_articles",
        type=int,
        default=None,
        metavar="N",
        help="Maximum articles to write per outlet (useful for smoke tests).",
    )
    parser.add_argument(
        "--outlet",
        dest="outlet_filter",
        default=None,
        metavar="SLUG",
        help="Run a single outlet by slug (e.g. fox, nyt).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    log.info("=" * 60)
    log.info("electoral scraper — run start %s", date_str)
    log.info("data root : %s", DATA_ROOT)
    log.info("dry run   : %s", args.dry_run)
    if args.max_articles is not None:
        log.info("max/outlet: %d", args.max_articles)
    log.info("=" * 60)

    outlets = OUTLETS
    if args.outlet_filter:
        outlets = [o for o in OUTLETS if o["slug"] == args.outlet_filter]
        if not outlets:
            log.error("unknown outlet slug: %r", args.outlet_filter)
            sys.exit(1)

    total_attempted = 0
    total_discarded = 0
    total_written = 0

    for idx, outlet in enumerate(outlets):
        if idx > 0:
            time.sleep(INTER_OUTLET_DELAY)

        try:
            counts = scrape_outlet(
                outlet,
                date_str,
                dry_run=args.dry_run,
                max_articles=args.max_articles,
            )
        except Exception:
            log.exception("[%s] unhandled error — continuing to next outlet", outlet["display"])
            continue

        total_attempted += counts["attempted"]
        total_discarded += counts["discarded"]
        total_written += counts["written"]

        log.info(
            "[%s] done — attempted=%d  discarded=%d  written=%d",
            outlet["display"],
            counts["attempted"],
            counts["discarded"],
            counts["written"],
        )

    log.info("=" * 60)
    log.info(
        "run complete — total attempted=%d  discarded=%d  written=%d",
        total_attempted,
        total_discarded,
        total_written,
    )
    log.info("=" * 60)


if __name__ == "__main__":
    main()
