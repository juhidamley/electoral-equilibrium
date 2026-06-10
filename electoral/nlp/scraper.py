"""News article scraper for shock events — runs on the Intel Mac.

Uses Google News RSS to discover relevant articles within a date window for
each source, then fetches and parses full article text with BeautifulSoup.
Results saved to JUHIDRIVE and synced to the M5 via Syncthing.

Output path per article batch:
    /Volumes/JUHIDRIVE/electoralData/rawdata/articles/{shock_id}/{source}.jsonl

Article schema (payload fields):
    shock_id, source, url, headline, text, published_date, word_count, scraped_at

Usage
-----
All sources, full date range:
    python -m electoral.nlp.scraper --shock-id ayatollah_assassination \\
        --date-range 2026-02-25 2026-03-15

Specific sources:
    python -m electoral.nlp.scraper --shock-id ayatollah_assassination \\
        --sources nyt wapo reuters --date-range 2026-02-25 2026-03-15

Quick validation — 5 articles per source:
    python -m electoral.nlp.scraper --shock-id ayatollah_assassination \\
        --test --date-range 2026-02-25 2026-03-15

Test target: ayatollah_assassination, all sources, 2026-02-25 → 2026-03-15
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from bs4 import BeautifulSoup

# ── Paths ─────────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parents[2]
SHOCKS_PATH = REPO_ROOT / "configs" / "shocks.json"

_env_root = os.environ.get("ELECTORAL_DATA_ROOT")
DATA_ROOT: Path = (
    Path(_env_root).expanduser().resolve()
    if _env_root
    else Path("/Volumes/JUHIDRIVE/electoralData/rawdata/articles")
)

# ── Constants ─────────────────────────────────────────────────────────────────

SCHEMA_VERSION = "1.0"
MIN_WORD_COUNT = 100
REQUEST_TIMEOUT = 20
REQUEST_SLEEP = 2.0    # seconds between every article fetch (avoids rate-limiting)
TEST_MAX = 5           # articles per source in --test mode

GNEWS_RSS_BASE = "https://news.google.com/rss/search"

# ── Source registry ───────────────────────────────────────────────────────────

# content_selectors: tried in order; first match yielding ≥30 words wins.
# A generic <p>-tag fallback is always attempted last.
SOURCES: dict[str, dict[str, Any]] = {
    "christianity_today": {
        "display": "Christianity Today",
        "domain": "christianitytoday.com",
        "content_selectors": ["div.article-body", "div.entry-content", "main article"],
    },
    "cbn": {
        "display": "CBN News",
        "domain": "cbn.com",
        "content_selectors": ["div.article-body", "div.field-item", "main article"],
    },
    "univision": {
        "display": "Univision",
        "domain": "univision.com",
        "content_selectors": [
            "div[class*='article-body']",
            "div[class*='body-content']",
            "article",
        ],
    },
    "fox_news": {
        "display": "Fox News",
        "domain": "foxnews.com",
        "content_selectors": ["div.article-body", "div.page-content", "article"],
    },
    "nyt": {
        "display": "New York Times",
        "domain": "nytimes.com",
        "content_selectors": [
            "section[name='articleBody']",
            "div.StoryBodyCompanionColumn",
            "article",
        ],
    },
    "wapo": {
        "display": "Washington Post",
        "domain": "washingtonpost.com",
        "content_selectors": ["div.article-body", "div[data-pb-type='art']", "article"],
    },
    "reuters": {
        "display": "Reuters",
        "domain": "reuters.com",
        "content_selectors": [
            "div[class*='article-body']",
            "div.StandardArticleBody_body",
            "article",
        ],
    },
    "ap": {
        "display": "Associated Press",
        "domain": "apnews.com",
        "content_selectors": ["div.RichTextStoryBody", "div.Article", "article"],
    },
}

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
    force=True,
)
log = logging.getLogger("electoral.scraper")

# ── HTTP ──────────────────────────────────────────────────────────────────────

_HEADERS: dict[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; ElectoralEquilibriumBot/1.0; "
        "research project, non-commercial)"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


def _fetch(url: str) -> requests.Response | None:
    """GET a URL; return Response or None on any failure including 403 paywalls."""
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=REQUEST_TIMEOUT)
        if resp.status_code == 403:
            log.warning("403 paywall — skipping: %s", url)
            return None
        resp.raise_for_status()
        return resp
    except requests.RequestException as exc:
        log.warning("fetch failed (%s): %s", type(exc).__name__, url)
        return None


# ── Google News RSS ───────────────────────────────────────────────────────────


def _gnews_url(keywords: list[str], domain: str, start: str, end: str) -> str:
    """Build a Google News RSS URL for a site-scoped keyword search.

    Uses the first 5 keywords to keep the query short. Date operators:
      after:YYYY-MM-DD  before:YYYY-MM-DD
    """
    kw_str = " ".join(keywords[:5])
    query = f"{kw_str} site:{domain} after:{start} before:{end}"
    params = urllib.parse.urlencode({
        "q": query,
        "hl": "en-US",
        "gl": "US",
        "ceid": "US:en",
    })
    return f"{GNEWS_RSS_BASE}?{params}"


# ── Feed parsing ──────────────────────────────────────────────────────────────

_ATOM_NS = "http://www.w3.org/2005/Atom"


def _parse_feed(xml_bytes: bytes) -> list[dict[str, str]]:
    """Parse RSS 2.0 or Atom bytes → list of {url, headline, published_date}."""
    items: list[dict[str, str]] = []
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError as exc:
        log.warning("feed parse error: %s", exc)
        return items

    for item in root.iter("item"):
        url = (item.findtext("link") or "").strip()
        if url:
            items.append({
                "url": url,
                "headline": (item.findtext("title") or "").strip(),
                "published_date": (item.findtext("pubDate") or "").strip(),
            })

    if not items:
        for entry in root.iter(f"{{{_ATOM_NS}}}entry"):
            link_el = (
                entry.find(f"{{{_ATOM_NS}}}link[@rel='alternate']")
                or entry.find(f"{{{_ATOM_NS}}}link")
            )
            url = (link_el.get("href", "") if link_el is not None else "").strip()
            if not url:
                continue
            title_el = entry.find(f"{{{_ATOM_NS}}}title")
            pub_el = (
                entry.find(f"{{{_ATOM_NS}}}published")
                or entry.find(f"{{{_ATOM_NS}}}updated")
            )
            items.append({
                "url": url,
                "headline": (title_el.text or "").strip() if title_el is not None else "",
                "published_date": (pub_el.text or "").strip() if pub_el is not None else "",
            })

    return items


# ── Text extraction ───────────────────────────────────────────────────────────

_NOISE_TAGS = frozenset([
    "script", "style", "nav", "header", "footer", "aside",
    "figure", "figcaption", "iframe", "noscript", "form",
])


def _extract_text(html: str, selectors: list[str]) -> str:
    """Extract article body text. Tries each selector; falls back to <p> tags."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(_NOISE_TAGS):
        tag.decompose()
    for sel in selectors:
        el = soup.select_one(sel)
        if el is not None:
            candidate = el.get_text(separator=" ", strip=True)
            if len(candidate.split()) >= 30:
                return candidate
    paras = [p.get_text(separator=" ", strip=True) for p in soup.find_all("p")]
    return " ".join(p for p in paras if p)


def _word_count(text: str) -> int:
    return len(text.split())


# ── Dedup helpers ─────────────────────────────────────────────────────────────


def _url_hash(url: str) -> str:
    return hashlib.sha256(url.encode()).hexdigest()[:16]


def _load_seen_ids(path: Path) -> set[str]:
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
                    rid = json.loads(line).get("payload", {}).get("id")
                    if rid:
                        seen.add(str(rid))
                except json.JSONDecodeError:
                    pass
    except OSError:
        pass
    return seen


# ── Shock keyword loader ──────────────────────────────────────────────────────


def _load_keywords(shock_id: str) -> list[str]:
    """Return keyword list for a shock from shocks.json; fall back to [shock_id]."""
    if SHOCKS_PATH.exists():
        try:
            for shock in json.loads(SHOCKS_PATH.read_text()):
                if shock.get("id") == shock_id:
                    kws = shock.get("keywords", [])
                    if kws:
                        return kws
        except (json.JSONDecodeError, KeyError):
            pass
    return [shock_id]


# ── Public API ────────────────────────────────────────────────────────────────


def scrape_articles(
    shock_id: str,
    sources: list[str] | None = None,
    date_range: tuple[str, str] | None = None,
    *,
    max_articles: int | None = None,
) -> dict[str, Any]:
    """Scrape news articles about a shock event via Google News RSS.

    Parameters
    ----------
    shock_id:
        Shock slug (e.g. ``ayatollah_assassination``). Keys the output
        subdirectory and used to look up keywords from shocks.json.
    sources:
        Subset of SOURCES slugs to scrape. None = all eight sources.
    date_range:
        ``(start, end)`` ISO date strings YYYY-MM-DD. Passed as
        Google News ``after:``/``before:`` operators. If None, no date
        filter is applied.
    max_articles:
        Cap per source (e.g. TEST_MAX=5 for quick validation).

    Returns
    -------
    dict with ``shock_id``, ``sources`` (per-source stat dicts),
    and ``total`` aggregate stats.
    """
    active = {s: SOURCES[s] for s in (sources or list(SOURCES)) if s in SOURCES}
    if not active:
        raise ValueError(
            f"No matching sources for {sources!r}. Valid slugs: {list(SOURCES)}"
        )

    if date_range is not None:
        start, end = date_range
    else:
        start = "2020-01-01"
        end = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    keywords = _load_keywords(shock_id)
    scraped_at = datetime.now(timezone.utc).isoformat()

    log.info(
        "shock=%s  sources=%s  window=%s→%s  keywords=%s",
        shock_id, list(active), start, end, keywords[:3],
    )

    result: dict[str, Any] = {
        "shock_id": shock_id,
        "sources": {},
        "total": {"attempted": 0, "discarded": 0, "written": 0},
    }

    for idx, (slug, cfg) in enumerate(active.items()):
        if idx > 0:
            time.sleep(REQUEST_SLEEP)

        display = cfg["display"]
        out_path = DATA_ROOT / shock_id / f"{slug}.jsonl"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        seen_ids = _load_seen_ids(out_path)
        counts: dict[str, int] = {"attempted": 0, "discarded": 0, "written": 0}

        rss_url = _gnews_url(keywords, cfg["domain"], start, end)
        log.info("[%s] Google News RSS query", display)
        log.debug("[%s] %s", display, rss_url)

        feed_resp = _fetch(rss_url)
        if feed_resp is None:
            log.error("[%s] Google News RSS unreachable — skipping source", display)
            result["sources"][slug] = counts
            continue

        items = _parse_feed(feed_resp.content)
        log.info("[%s] %d articles in RSS", display, len(items))

        for item in items:
            if max_articles is not None and counts["written"] >= max_articles:
                log.info("[%s] max=%d reached — stopping", display, max_articles)
                break

            url = item["url"]
            article_id = _url_hash(url)
            counts["attempted"] += 1

            if article_id in seen_ids:
                log.debug("[%s] skip dedup: %s", display, url)
                counts["discarded"] += 1
                continue

            time.sleep(REQUEST_SLEEP)
            page_resp = _fetch(url)
            if page_resp is None:
                counts["discarded"] += 1
                continue

            text = _extract_text(page_resp.text, cfg["content_selectors"])
            wc = _word_count(text)

            if wc < MIN_WORD_COUNT:
                log.debug("[%s] discard wc=%d (<100): %s", display, wc, url)
                counts["discarded"] += 1
                continue

            record = {
                "schema_version": SCHEMA_VERSION,
                "stage": "collect",
                "seed": None,
                "payload": {
                    "id": article_id,
                    "shock_id": shock_id,
                    "source": slug,
                    "url": url,
                    "headline": item["headline"],
                    "text": text,
                    "published_date": item["published_date"],
                    "word_count": wc,
                    "scraped_at": scraped_at,
                    "lang": "en",
                    "platform": "news",
                },
            }
            with open(out_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            seen_ids.add(article_id)
            counts["written"] += 1
            log.info("[%s] wrote wc=%d: %s", display, wc, url)

        discard_pct = (
            counts["discarded"] / counts["attempted"] * 100
            if counts["attempted"] else 0.0
        )
        log.info(
            "[%s] done — attempted=%d  discarded=%d (%.0f%%)  written=%d",
            display, counts["attempted"], counts["discarded"], discard_pct, counts["written"],
        )
        result["sources"][slug] = counts
        for k in result["total"]:
            result["total"][k] += counts[k]

    return result


# ── CLI ───────────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Scrape news articles for a shock event via Google News RSS."
    )
    p.add_argument(
        "--shock-id",
        required=True,
        metavar="SLUG",
        help="Shock event slug (e.g. ayatollah_assassination)",
    )
    p.add_argument(
        "--sources",
        nargs="+",
        metavar="SOURCE",
        choices=list(SOURCES),
        default=None,
        help=f"Sources to scrape (default: all). Choices: {list(SOURCES)}",
    )
    p.add_argument(
        "--date-range",
        nargs=2,
        metavar=("START", "END"),
        help="ISO date window: YYYY-MM-DD YYYY-MM-DD",
    )
    p.add_argument(
        "--test",
        action="store_true",
        help=f"Quick validation: only {TEST_MAX} articles per source",
    )
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    date_range: tuple[str, str] | None = tuple(args.date_range) if args.date_range else None  # type: ignore[assignment]
    max_articles = TEST_MAX if args.test else None

    log.info("shock=%s  date_range=%s  test=%s", args.shock_id, date_range, args.test)

    result = scrape_articles(
        shock_id=args.shock_id,
        sources=args.sources,
        date_range=date_range,
        max_articles=max_articles,
    )

    t = result["total"]
    log.info(
        "DONE — shock=%s  attempted=%d  discarded=%d  written=%d",
        args.shock_id, t["attempted"], t["discarded"], t["written"],
    )


if __name__ == "__main__":
    main()
