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

import json
import logging
import re
from datetime import date, datetime
from pathlib import Path
from typing import Any

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
            len(posts), self._root, outlet, since, until,
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

    logger.info(
        "Loaded %d records from HuggingFace dataset '%s'", len(posts), dataset_name
    )
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
