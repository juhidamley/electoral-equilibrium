"""news_loader: Loaders for scraped and archival news articles.

Two loaders:
  ScrapedNewsLoader   — reads rawdata/news/{outlet}/{YYYY-MM-DD}.jsonl
  load_huggingface_news() — downloads 3DLNews2 or other HF news datasets

Outlet-level demographic proxies are stored in the record's author_description
field as JSON so the scorer can retrieve them without a bio classifier call.
These are separate from individual-author bios — they reflect the publication's
readership composition, not a specific author's identity.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

try:
    import requests as _requests
    from bs4 import BeautifulSoup as _BeautifulSoup

    _FETCH_AVAILABLE = True
except ImportError:
    _FETCH_AVAILABLE = False

from electoral.nlp.collectors.schema import (
    build_keyword_index,
    build_post_payload,
    load_shocks,
    match_shocks,
    normalize_timestamp,
)

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_SHOCKS_PATH = _REPO_ROOT / "configs" / "shocks.json"

# Outlet slug → stratum demographic weights for the scoring aggregation.
# These represent the readership composition of each outlet, not individual authors.
# Stored as {"race": {...}, "religion": {...}, "gender": {...}} — strata may be absent
# if the outlet has no meaningful signal for that stratum.
OUTLET_DEMO_PROXY: dict[str, dict[str, dict[str, float]]] = {
    "christianity_today": {
        "religion": {"evangelical": 0.85, "protestant": 0.15},
    },
    "cbn": {
        "religion": {"evangelical": 1.0},
    },
    "ewtn": {
        "religion": {"catholic": 1.0},
    },
    "univision": {
        "race": {"latino": 1.0},
        "religion": {"catholic": 0.65, "evangelical": 0.20, "other_rel": 0.15},
    },
    "nyt": {
        "religion": {"secular": 0.70, "jewish": 0.20, "other_rel": 0.10},
        "race": {"white": 0.65, "asian": 0.10, "african_american": 0.10, "other_race": 0.15},
    },
    "wapo": {
        "religion": {"secular": 0.75, "jewish": 0.15, "other_rel": 0.10},
        "race": {"white": 0.65, "african_american": 0.12, "asian": 0.10, "other_race": 0.13},
    },
    "fox": {
        "religion": {"evangelical": 0.40, "protestant": 0.20, "catholic": 0.20, "secular": 0.20},
        "race": {"white": 0.75, "other_race": 0.25},
    },
    "npr": {
        "religion": {"secular": 0.60, "other_rel": 0.25, "jewish": 0.10, "other_rel2": 0.05},
        "race": {"white": 0.60, "african_american": 0.15, "asian": 0.10, "other_race": 0.15},
    },
    "breitbart": {
        "religion": {"evangelical": 0.50, "protestant": 0.20, "catholic": 0.15, "secular": 0.15},
        "race": {"white": 0.85, "other_race": 0.15},
    },
    "daily_wire": {
        "religion": {"evangelical": 0.55, "protestant": 0.20, "catholic": 0.10, "secular": 0.15},
        "race": {"white": 0.80, "other_race": 0.20},
    },
}

# Known outlet name variants → canonical slug
_OUTLET_ALIASES: dict[str, str] = {
    "christianitytoday": "christianity_today",
    "christianity-today": "christianity_today",
    "christian_today": "christianity_today",
    "fox_news": "fox",
    "foxnews": "fox",
    "washington_post": "wapo",
    "washingtonpost": "wapo",
    "new_york_times": "nyt",
    "newyorktimes": "nyt",
    "national_public_radio": "npr",
    "thedailywire": "daily_wire",
    "daily-wire": "daily_wire",
}

_DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _canonical_outlet(slug: str) -> str:
    """Return the canonical outlet slug, applying known aliases."""
    slug = slug.lower().replace("-", "_").replace(" ", "_")
    return _OUTLET_ALIASES.get(slug, slug)


def _outlet_proxy_description(outlet: str) -> str:
    """Return JSON-encoded outlet proxy for storage in author_description."""
    proxy = OUTLET_DEMO_PROXY.get(outlet, {})
    return json.dumps({"outlet": outlet, "proxy": proxy})


class ScrapedNewsLoader:
    """Load scraped news articles from rawdata/news/{outlet}/{YYYY-MM-DD}.jsonl.

    Each JSONL line is a raw scraped article. Field names vary by outlet scraper.
    All records are normalized to the canonical post payload schema.

    Outlet-level demographic proxies are embedded in author_description as JSON
    so the scorer can aggregate sentiment without calling the bio classifier.
    """

    def __init__(
        self,
        news_root: str | Path,
        shocks_path: str | Path | None = None,
    ) -> None:
        self._root = Path(news_root)
        self._shocks_path = Path(shocks_path) if shocks_path else _DEFAULT_SHOCKS_PATH
        self._keyword_index: dict[str, list[str]] | None = None

    def _get_keyword_index(self) -> dict[str, list[str]]:
        if self._keyword_index is None:
            if self._shocks_path.exists():
                shocks = load_shocks(self._shocks_path)
                self._keyword_index = build_keyword_index(shocks)
            else:
                logger.warning("shocks.json not found at %s", self._shocks_path)
                self._keyword_index = {}
        return self._keyword_index

    def _route_shock(self, text: str) -> str | None:
        index = self._get_keyword_index()
        matched = match_shocks(text, index)
        if len(matched) == 1:
            return next(iter(matched))
        if len(matched) > 1:
            return sorted(matched)[0]
        return None

    def load(
        self,
        outlet: str | None = None,
        since: str | date | None = None,
        until: str | date | None = None,
    ) -> list[dict]:
        """Load articles, optionally filtered by outlet and date range.

        Args:
            outlet: Outlet directory name. None loads all outlets.
            since: Start date inclusive (YYYY-MM-DD or date).
            until: End date inclusive (YYYY-MM-DD or date).
        """
        if not self._root.exists():
            logger.debug("News root not found: %s", self._root)
            return []

        since_d = _parse_date(since)
        until_d = _parse_date(until)

        if outlet is not None:
            outlet_dirs = [self._root / outlet]
        else:
            outlet_dirs = [p for p in sorted(self._root.iterdir()) if p.is_dir()]

        posts: list[dict] = []
        for outlet_dir in outlet_dirs:
            if not outlet_dir.is_dir():
                continue
            outlet_slug = _canonical_outlet(outlet_dir.name)

            for path in sorted(outlet_dir.glob("*.jsonl")):
                # Filter by date using the filename
                if _DATE_PATTERN.match(path.stem):
                    file_date = _parse_date(path.stem)
                    if since_d and file_date and file_date < since_d:
                        continue
                    if until_d and file_date and file_date > until_d:
                        continue

                n_before = len(posts)
                posts.extend(self._load_file(path, outlet_slug))
                loaded = len(posts) - n_before
                if loaded:
                    logger.debug("%s/%s: %d articles", outlet_slug, path.name, loaded)

        logger.info(
            "ScrapedNewsLoader: %d articles from %s (outlet=%s, since=%s, until=%s)",
            len(posts),
            self._root,
            outlet,
            since,
            until,
        )
        return posts

    def load_all(self) -> list[dict]:
        return self.load()

    def _load_file(self, path: Path, outlet_slug: str) -> list[dict]:
        records: list[dict] = []
        with open(path, encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError as exc:
                    logger.warning("%s line %d: %s", path, lineno, exc)
                    continue
                if "payload" in raw:
                    raw = raw["payload"]
                record = self._normalize(raw, outlet_slug)
                if record is not None:
                    records.append(record)
        return records

    def _normalize(self, raw: dict[str, Any], outlet_slug: str) -> dict | None:
        """Normalize a raw scraped article to canonical post schema."""
        text = (
            raw.get("article_text")
            or raw.get("body")
            or raw.get("content")
            or raw.get("text")
            or raw.get("description")
            or raw.get("summary")
            or ""
        )
        if not isinstance(text, str):
            text = ""
        text = text.strip()
        if not text:
            return None

        post_id = str(raw.get("id") or raw.get("url") or raw.get("link") or "")
        if not post_id:
            post_id = f"news_{outlet_slug}_{abs(hash(text)):016x}"

        created_at = normalize_timestamp(
            raw.get("published_at")
            or raw.get("published")
            or raw.get("date")
            or raw.get("created_at")
        )
        shock_id = self._route_shock(text)

        return build_post_payload(
            post_id=f"news:{post_id}",
            text=text,
            created_at=created_at,
            lang=raw.get("language") or raw.get("lang") or "en",
            source="archive",
            archive_id=outlet_slug,
            platform="news",
            shock_id=shock_id,
            author_did=None,
            author_handle=outlet_slug,
            author_description=_outlet_proxy_description(outlet_slug),
            inference_method=None,
        )


def load_huggingface_news(
    dataset_name: str,
    shocks_path: str | Path | None = None,
    max_records: int | None = None,
    split: str = "train",
) -> list[dict]:
    """Load a HuggingFace news dataset and normalize to canonical post schema.

    Designed for 3DLNews2 (multilingual news dataset on HuggingFace Hub).
    Falls back gracefully if the `datasets` library is not installed.

    Args:
        dataset_name: HuggingFace dataset identifier e.g. "ucsbnlp/3dlnews".
        shocks_path: Path to configs/shocks.json for shock routing.
        max_records: Optional cap on records loaded (None = all).
        split: Dataset split to load (default "train").

    Returns:
        List of canonical post payload dicts.
    """
    try:
        from datasets import load_dataset  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "The 'datasets' library is required to load HuggingFace news datasets.\n"
            "Install with: pip install datasets"
        ) from exc

    shocks_path = Path(shocks_path) if shocks_path else _DEFAULT_SHOCKS_PATH
    keyword_index: dict[str, list[str]] = {}
    if shocks_path.exists():
        keyword_index = build_keyword_index(load_shocks(shocks_path))

    logger.info("Loading HuggingFace dataset '%s' (split=%s)...", dataset_name, split)
    ds = load_dataset(dataset_name, split=split)

    posts: list[dict] = []
    for i, row in enumerate(ds):
        if max_records is not None and i >= max_records:
            break

        text = (
            row.get("text")
            or row.get("content")
            or row.get("article")
            or row.get("body")
            or row.get("title")
            or ""
        )
        if not isinstance(text, str) or not text.strip():
            continue
        text = text.strip()

        post_id = str(row.get("id") or row.get("url") or i)
        outlet = str(row.get("outlet") or row.get("source") or row.get("domain") or "unknown")
        outlet_slug = _canonical_outlet(outlet)
        lang = str(row.get("language") or row.get("lang") or "en")

        created_at = normalize_timestamp(
            row.get("date") or row.get("published_at") or row.get("created_at")
        )

        matched = match_shocks(text, keyword_index)
        shock_id = sorted(matched)[0] if matched else None

        posts.append(
            build_post_payload(
                post_id=f"hf:{dataset_name.replace('/', '_')}:{post_id}",
                text=text,
                created_at=created_at,
                lang=lang,
                source="archive",
                archive_id=dataset_name,
                platform="news",
                shock_id=shock_id,
                author_did=None,
                author_handle=outlet_slug,
                author_description=_outlet_proxy_description(outlet_slug),
                inference_method=None,
            )
        )

    logger.info("Loaded %d records from HuggingFace dataset '%s'", len(posts), dataset_name)
    return posts


def _parse_date(d: str | date | None) -> date | None:
    """Parse a YYYY-MM-DD string or date object; returns None on failure."""
    if d is None:
        return None
    if isinstance(d, date):
        return d
    try:
        return datetime.strptime(str(d), "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


# ── Fetch helpers (used by ThreeDLNewsLoader) ─────────────────────────────────

_FETCH_HEADERS: dict[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; ElectoralEquilibriumBot/1.0; " "research project, non-commercial)"
    ),
    "Accept": "text/html,application/xhtml+xml,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

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

_CONTENT_SELECTORS = [
    "article",
    "div.article-body",
    "div.entry-content",
    "div[class*='article-body']",
    "div[class*='story-body']",
    "div[class*='article__body']",
    "div[class*='post-body']",
    "div[class*='entry-content']",
    "main",
]

MIN_ARTICLE_WORDS = 100


def _fetch_url(url: str, timeout: int = 20) -> str | None:
    if not _FETCH_AVAILABLE:
        raise ImportError(
            "requests and beautifulsoup4 are required for URL fetching — "
            "run: pip install requests beautifulsoup4"
        )
    try:
        resp = _requests.get(url, headers=_FETCH_HEADERS, timeout=timeout)
        resp.raise_for_status()
        return resp.text
    except Exception as exc:
        logger.warning("fetch failed (%s): %s", type(exc).__name__, url)
        return None


def _extract_article_text(html: str) -> str:
    """Extract main article text from HTML. Same logic as scraper._extract_text."""
    soup = _BeautifulSoup(html, "html.parser")
    for tag in soup(_NOISE_TAGS):
        tag.decompose()
    for sel in _CONTENT_SELECTORS:
        el = soup.select_one(sel)
        if el is not None:
            candidate = el.get_text(separator=" ", strip=True)
            if len(candidate.split()) >= 30:
                return " ".join(candidate.split())
    paras = [p.get_text(separator=" ", strip=True) for p in soup.find_all("p")]
    return " ".join(p for p in paras if p)


def _url_id(url: str) -> str:
    return hashlib.sha256(url.encode()).hexdigest()[:16]


def _parse_article_date(raw: dict[str, Any]) -> date | None:
    """Try common date field names and return a date object."""
    for field in ("publication_date", "published", "date", "published_at", "created_at", "crawled"):
        val = raw.get(field)
        if not val:
            continue
        if isinstance(val, (int, float)):
            try:
                return datetime.fromtimestamp(val, tz=timezone.utc).date()
            except (OSError, OverflowError):
                continue
        s = str(val).strip()
        # ISO prefix is enough for date comparison
        m = re.match(r"(\d{4}-\d{2}-\d{2})", s)
        if m:
            try:
                return datetime.strptime(m.group(1), "%Y-%m-%d").date()
            except ValueError:
                continue
    return None


# ── ThreeDLNewsLoader ─────────────────────────────────────────────────────────


class ThreeDLNewsLoader:
    """Load 3DLNews archive URL entries, fetch HTML, and return canonical payloads.

    The 3dlnews archive at archive_root contains JSONL metadata files with
    one article entry per line. Each entry has at minimum a URL and a date;
    some may already carry pre-extracted text (used directly to skip the fetch).

    State filtering uses the ``state`` metadata field when present. Outlet
    filtering is a case-insensitive substring match against the ``outlet``,
    ``source``, or ``domain`` fields and the URL hostname.

    Records without 100+ words after extraction are discarded.
    """

    def __init__(
        self,
        archive_root: str | Path,
        shocks_path: str | Path | None = None,
        request_timeout: int = 20,
        inter_article_delay: float = 0.3,
    ) -> None:
        self._root = Path(archive_root)
        self._shocks_path = Path(shocks_path) if shocks_path else _DEFAULT_SHOCKS_PATH
        self._timeout = request_timeout
        self._delay = inter_article_delay

    def load(
        self,
        since: str | date | None = None,
        until: str | date | None = None,
        states: set[str] | None = None,
        outlet: str | None = None,
        shock_id: str | None = None,
        max_articles: int | None = None,
    ) -> list[dict]:
        """Load articles matching the given filters.

        Args:
            since: Inclusive start date (YYYY-MM-DD).
            until: Inclusive end date (YYYY-MM-DD).
            states: Set of 2-letter state codes to keep (e.g. {"PA", "GA", "TX"}).
            outlet: Substring to match against outlet name or URL host.
            shock_id: If given, set as shock_id in all returned payloads.
            max_articles: Cap on articles returned (None = unlimited).
        """
        if not self._root.exists():
            logger.warning("3DLNews archive root not found: %s", self._root)
            return []

        since_d = _parse_date(since)
        until_d = _parse_date(until)
        outlet_lower = outlet.lower() if outlet else None
        states_upper = {s.upper() for s in states} if states else None

        records: list[dict] = []
        n_skipped_date = n_skipped_state = n_skipped_outlet = n_short = n_fetch_fail = 0

        for path in sorted(self._root.rglob("*.jsonl")):
            with open(path, encoding="utf-8", errors="replace") as f:
                for lineno, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        raw = json.loads(line)
                    except json.JSONDecodeError as exc:
                        logger.debug("%s line %d: %s", path, lineno, exc)
                        continue
                    if "payload" in raw:
                        raw = raw["payload"]

                    # ── Date filter ───────────────────────────────────────────
                    art_date = _parse_article_date(raw)
                    if since_d and art_date and art_date < since_d:
                        n_skipped_date += 1
                        continue
                    if until_d and art_date and art_date > until_d:
                        n_skipped_date += 1
                        continue

                    # ── State filter ──────────────────────────────────────────
                    if states_upper:
                        rec_state = str(raw.get("state") or "").upper()
                        if rec_state not in states_upper:
                            n_skipped_state += 1
                            continue

                    # ── Outlet filter ─────────────────────────────────────────
                    url = str(raw.get("url") or raw.get("link") or "")
                    if outlet_lower:
                        outlet_field = str(
                            raw.get("outlet") or raw.get("source") or raw.get("domain") or ""
                        ).lower()
                        url_host = re.sub(r"https?://([^/]+).*", r"\1", url).lower()
                        if outlet_lower not in outlet_field and outlet_lower not in url_host:
                            n_skipped_outlet += 1
                            continue

                    # ── Text: use cached or fetch ─────────────────────────────
                    text = str(
                        raw.get("content")
                        or raw.get("article_text")
                        or raw.get("text")
                        or raw.get("body")
                        or ""
                    ).strip()

                    if len(text.split()) < MIN_ARTICLE_WORDS:
                        if not url:
                            n_short += 1
                            continue
                        if not _FETCH_AVAILABLE:
                            logger.warning("requests/bs4 not installed; cannot fetch %s", url)
                            n_fetch_fail += 1
                            continue
                        html = _fetch_url(url, timeout=self._timeout)
                        if html is None:
                            n_fetch_fail += 1
                            continue
                        text = _extract_article_text(html)
                        if self._delay > 0:
                            time.sleep(self._delay)

                    if len(text.split()) < MIN_ARTICLE_WORDS:
                        n_short += 1
                        continue

                    # ── Build payload ─────────────────────────────────────────
                    rec_outlet = str(
                        raw.get("outlet") or raw.get("source") or raw.get("domain") or "unknown"
                    )
                    created_at = normalize_timestamp(
                        raw.get("publication_date")
                        or raw.get("published")
                        or raw.get("date")
                        or raw.get("published_at")
                        or raw.get("created_at")
                    )
                    post_id = str(raw.get("id") or raw.get("url") or "")
                    if not post_id:
                        post_id = _url_id(text[:200])

                    payload = build_post_payload(
                        post_id=f"3dlnews:{post_id}",
                        text=text,
                        created_at=created_at,
                        lang=raw.get("language") or raw.get("lang") or "en",
                        source="3dlnews",
                        archive_id="3dlnews",
                        platform="local_news",
                        shock_id=shock_id,
                        author_did=None,
                        author_handle=rec_outlet,
                        author_description=json.dumps(
                            {"outlet": rec_outlet, "state": raw.get("state")}
                        ),
                        inference_method="platform_proxy",
                    )
                    payload["url"] = url
                    payload["published"] = created_at

                    records.append(payload)

                    if max_articles and len(records) >= max_articles:
                        break

            if max_articles and len(records) >= max_articles:
                break

        logger.info(
            "3DLNewsLoader: %d articles loaded (skipped: date=%d state=%d outlet=%d "
            "short=%d fetch_fail=%d)",
            len(records),
            n_skipped_date,
            n_skipped_state,
            n_skipped_outlet,
            n_short,
            n_fetch_fail,
        )
        return records


# ── WebhoseLoader ─────────────────────────────────────────────────────────────


class WebhoseLoader:
    """Load pre-extracted Webhose news articles from archive JSONL files.

    Reads JSONL from archive_root, filters by date and optionally by URL
    domain substring, enforces 100-word minimum. No HTTP fetch is performed.
    """

    def __init__(
        self,
        archive_root: str | Path,
        shocks_path: str | Path | None = None,
    ) -> None:
        self._root = Path(archive_root)
        self._shocks_path = Path(shocks_path) if shocks_path else _DEFAULT_SHOCKS_PATH
        self._keyword_index: dict[str, list[str]] | None = None

    def _get_keyword_index(self) -> dict[str, list[str]]:
        if self._keyword_index is None:
            if self._shocks_path.exists():
                self._keyword_index = build_keyword_index(load_shocks(self._shocks_path))
            else:
                logger.warning("shocks.json not found at %s", self._shocks_path)
                self._keyword_index = {}
        return self._keyword_index

    def load(
        self,
        since: str | date | None = None,
        until: str | date | None = None,
        domain: str | None = None,
        shock_id: str | None = None,
        max_articles: int | None = None,
    ) -> list[dict]:
        """Load Webhose articles matching the given filters.

        Args:
            since: Inclusive start date (YYYY-MM-DD).
            until: Inclusive end date (YYYY-MM-DD).
            domain: Substring to match against article URL or site field.
            shock_id: If given, set as shock_id in all returned payloads.
                      If None, keyword routing from shocks.json is used.
            max_articles: Cap on articles returned (None = unlimited).
        """
        if not self._root.exists():
            logger.warning("Webhose archive root not found: %s", self._root)
            return []

        since_d = _parse_date(since)
        until_d = _parse_date(until)
        domain_lower = domain.lower() if domain else None

        records: list[dict] = []
        n_skipped_date = n_skipped_domain = n_short = 0

        for path in sorted(self._root.rglob("*.jsonl")):
            with open(path, encoding="utf-8", errors="replace") as f:
                for lineno, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        raw = json.loads(line)
                    except json.JSONDecodeError as exc:
                        logger.debug("%s line %d: %s", path, lineno, exc)
                        continue
                    if "payload" in raw:
                        raw = raw["payload"]

                    # ── Date filter ───────────────────────────────────────────
                    art_date = _parse_article_date(raw)
                    if since_d and art_date and art_date < since_d:
                        n_skipped_date += 1
                        continue
                    if until_d and art_date and art_date > until_d:
                        n_skipped_date += 1
                        continue

                    # ── Domain filter ─────────────────────────────────────────
                    url = str(raw.get("url") or "")
                    if domain_lower:
                        thread = raw.get("thread") or {}
                        site = str(
                            thread.get("site") or raw.get("site_full") or raw.get("site") or url
                        ).lower()
                        if domain_lower not in site:
                            n_skipped_domain += 1
                            continue

                    # ── Text ──────────────────────────────────────────────────
                    text = str(
                        raw.get("text")
                        or raw.get("body")
                        or raw.get("content")
                        or raw.get("title")
                        or ""
                    ).strip()

                    if len(text.split()) < MIN_ARTICLE_WORDS:
                        n_short += 1
                        continue

                    # ── Build payload ─────────────────────────────────────────
                    thread = raw.get("thread") or {}
                    site = thread.get("site") or raw.get("site_full") or raw.get("site") or ""
                    outlet = site.replace("www.", "").split(".")[0] if site else "unknown"

                    post_id = str(raw.get("uuid") or raw.get("id") or "")
                    if not post_id:
                        post_id = _url_id(url or text[:200])

                    created_at = normalize_timestamp(
                        raw.get("published") or raw.get("crawled") or raw.get("created_at")
                    )

                    # Use keyword routing only when no shock_id supplied
                    resolved_shock = shock_id
                    if resolved_shock is None:
                        index = self._get_keyword_index()
                        matched = match_shocks(text, index)
                        resolved_shock = sorted(matched)[0] if matched else None

                    payload = build_post_payload(
                        post_id=f"webhose:{post_id}",
                        text=text,
                        created_at=created_at,
                        lang=raw.get("language") or raw.get("lang") or "en",
                        source="webhose",
                        archive_id="webhose",
                        platform="news",
                        shock_id=resolved_shock,
                        author_did=None,
                        author_handle=outlet,
                        author_description=json.dumps({"outlet": outlet}),
                        inference_method=None,
                    )
                    payload["url"] = url
                    payload["published"] = created_at

                    records.append(payload)

                    if max_articles and len(records) >= max_articles:
                        break

            if max_articles and len(records) >= max_articles:
                break

        logger.info(
            "WebhoseLoader: %d articles loaded (skipped: date=%d domain=%d short=%d)",
            len(records),
            n_skipped_date,
            n_skipped_domain,
            n_short,
        )
        return records


# ── Output writer ─────────────────────────────────────────────────────────────


def save_articles(
    records: list[dict],
    output_root: str | Path,
    shock_id: str,
    source: str,
    seed: int | None = None,
) -> Path:
    """Write article payloads to rawdata/articles/{shock_id}/{source}.jsonl.

    Each line is a canonical JSONL envelope (schema_version 1.0).
    Returns the path written.
    """
    out_dir = Path(output_root) / shock_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{source}.jsonl"
    now = datetime.now(tz=timezone.utc).isoformat()
    with open(out_path, "w", encoding="utf-8") as f:
        for payload in records:
            envelope = {
                "schema_version": "1.0",
                "created_at": now,
                "stage": "collect",
                "seed": seed,
                "payload": payload,
            }
            f.write(json.dumps(envelope, ensure_ascii=False) + "\n")
    logger.info("Wrote %d articles → %s", len(records), out_path)
    return out_path


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import sys

    _DEFAULT_DATA_ROOT = "/Volumes/JUHIDRIVE/electoralData"
    _DEFAULT_OUTPUT_DIR = "/Volumes/JUHIDRIVE/electoralData/rawdata/articles"

    p = argparse.ArgumentParser(
        description=(
            "Test news_loader.py loaders from the command line.\n\n"
            "Examples:\n"
            "  python -m electoral.nlp.news_loader --loader 3dlnews "
            "--shock-id metoo_2017 --since 2017-10-01 --until 2017-11-30 "
            "--states PA GA TX\n"
            "  python -m electoral.nlp.news_loader --loader webhose "
            "--shock-id roe_v_wade_2022 --since 2022-06-24 --until 2022-07-15 "
            "--max-articles 500"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--loader",
        required=True,
        choices=["scraped", "3dlnews", "webhose"],
        help="Which loader to instantiate.",
    )
    p.add_argument(
        "--shock-id",
        required=True,
        metavar="SLUG",
        help="Shock event slug (e.g. metoo_2017). Used for output path and payload tagging.",
    )
    p.add_argument("--since", metavar="YYYY-MM-DD", help="Start date inclusive.")
    p.add_argument("--until", metavar="YYYY-MM-DD", help="End date inclusive.")
    p.add_argument(
        "--states",
        nargs="+",
        metavar="STATE",
        help="2-letter state codes for 3dlnews state filter (e.g. --states PA GA TX).",
    )
    p.add_argument(
        "--output-dir",
        metavar="DIR",
        default=None,
        help=(
            f"Save output via save_articles() to this root directory. "
            f"Default: {_DEFAULT_OUTPUT_DIR}"
        ),
    )
    p.add_argument(
        "--max-articles",
        type=int,
        metavar="N",
        help="Cap on articles loaded (for quick testing).",
    )
    p.add_argument("--verbose", action="store_true", help="Enable DEBUG logging.")
    args = p.parse_args()

    logging.basicConfig(
        stream=sys.stdout,
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
        force=True,
    )

    data_root = Path(_DEFAULT_DATA_ROOT)

    if args.loader == "scraped":
        loader_obj: ScrapedNewsLoader | ThreeDLNewsLoader | WebhoseLoader = ScrapedNewsLoader(
            data_root / "rawdata" / "news"
        )
        records = loader_obj.load(since=args.since, until=args.until)  # type: ignore[call-arg]
        if args.max_articles is not None:
            records = records[: args.max_articles]

    elif args.loader == "3dlnews":
        loader_obj = ThreeDLNewsLoader(data_root / "archives" / "news" / "3dlnews_parsed")
        records = loader_obj.load(  # type: ignore[call-arg]
            since=args.since,
            until=args.until,
            states=set(args.states) if args.states else None,
            shock_id=args.shock_id,
            max_articles=args.max_articles,
        )

    else:  # webhose
        loader_obj = WebhoseLoader(data_root / "archives" / "news" / "webhose")
        records = loader_obj.load(  # type: ignore[call-arg]
            since=args.since,
            until=args.until,
            shock_id=args.shock_id,
            max_articles=args.max_articles,
        )

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\nloader:    {args.loader}")
    print(f"shock_id:  {args.shock_id}")
    print(f"since:     {args.since or '(none)'}")
    print(f"until:     {args.until or '(none)'}")
    if args.states:
        print(f"states:    {' '.join(args.states)}")
    print(f"loaded:    {len(records)} articles")

    if records:
        sample = records[0]
        snippet = sample.get("text", "")[:80].replace("\n", " ")
        print(f"sample:    shock={sample.get('shock_id')} | {snippet!r}")

    # ── Save ──────────────────────────────────────────────────────────────────
    output_dir = args.output_dir or _DEFAULT_OUTPUT_DIR
    if records:
        out_path = save_articles(
            records,
            output_root=output_dir,
            shock_id=args.shock_id,
            source=args.loader,
        )
        print(f"saved →    {out_path}")
    else:
        print("(no articles to save)")
