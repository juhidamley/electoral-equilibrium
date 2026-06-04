"""archive: Historical archive loader for pre-existing post collections.

Loads posts from data/archives/{reddit,news,discord}/ and normalizes them
to the canonical post payload schema (same fields as build_post_payload()).

Reddit subreddits map to demographic proxies (subreddit_proxy); this is
validated but NOT run through the bio classifier — exclude from Σ_Δ
estimation per the subreddit_proxy inference_method rule.

Archive structure (from data/archives/README.md):
    data/archives/news/3dlnews/   — 3DLNews2 URL+metadata JSONL
    data/archives/news/webhose/   — Webhose news dumps
    data/archives/reddit/         — sampled Reddit posts per subreddit
    data/archives/discord/        — Discord exports (may be empty)
"""

from __future__ import annotations

import json
import logging
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

# Reddit subreddit → demographic bloc proxy (from CLAUDE.md)
SUBREDDIT_PROXY: dict[str, str] = {
    "Catholicism": "catholic",
    "Christianity": "protestant",
    "exchristian": "secular",
    "Conservative": "white",
    "progressive": "secular",
    "BlackPeopleTwitter": "african_american",
    "LatinoPeopleTwitter": "latino",
    "Jewish": "jewish",
    "islam": "muslim",
}


class HistoricalArchiveLoader:
    """Load and normalize historical post archives to canonical payload schema.

    Does not require network access. Reads files from data/archives/.
    Missing subdirectories are silently skipped (returns empty list).
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
                shocks = load_shocks(self._shocks_path)
                self._keyword_index = build_keyword_index(shocks)
            else:
                logger.warning(
                    "shocks.json not found at %s; shock routing disabled.",
                    self._shocks_path,
                )
                self._keyword_index = {}
        return self._keyword_index

    def _route_shock(self, text: str) -> str | None:
        """Return a single shock_id if the text matches exactly one shock, else None."""
        index = self._get_keyword_index()
        matched = match_shocks(text, index)
        if len(matched) == 1:
            return next(iter(matched))
        if len(matched) > 1:
            # Pick the first alphabetically for determinism
            return sorted(matched)[0]
        return None

    # ── Reddit ────────────────────────────────────────────────────────────────

    def load_reddit(self) -> list[dict]:
        """Load Reddit posts from data/archives/reddit/*.jsonl.

        Each subreddit file is named {subreddit}.jsonl or lives in a
        subdirectory. All posts get inference_method="subreddit_proxy".
        """
        reddit_root = self._root / "reddit"
        if not reddit_root.exists():
            logger.debug("Reddit archive directory not found: %s", reddit_root)
            return []

        posts: list[dict] = []
        for path in sorted(reddit_root.rglob("*.jsonl")):
            # Derive subreddit from parent directory name or file stem
            subreddit = path.stem if path.parent == reddit_root else path.parent.name
            n_before = len(posts)
            posts.extend(self._load_reddit_file(path, subreddit))
            logger.debug(
                "Reddit archive %s: loaded %d posts", path, len(posts) - n_before
            )

        logger.info("Reddit archive: %d posts total from %s", len(posts), reddit_root)
        return posts

    def _load_reddit_file(self, path: Path, subreddit: str) -> list[dict]:
        records: list[dict] = []
        with open(path, encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError as exc:
                    logger.warning("%s line %d: JSON error (%s)", path, lineno, exc)
                    continue
                # Support both enveloped and raw records
                if "payload" in raw:
                    raw = raw["payload"]
                record = self._normalize_reddit(raw, subreddit)
                if record is not None:
                    records.append(record)
        return records

    def _normalize_reddit(self, raw: dict[str, Any], subreddit: str) -> dict | None:
        """Normalize a raw Reddit post dict to canonical payload schema."""
        # Extract text from various field names used by different Reddit API versions
        text = (
            raw.get("selftext")
            or raw.get("body")
            or raw.get("text")
            or raw.get("title")
            or ""
        )
        if not isinstance(text, str):
            text = ""
        text = text.strip()
        if not text or text == "[deleted]" or text == "[removed]":
            return None

        post_id = str(
            raw.get("id") or raw.get("post_id") or raw.get("name") or ""
        )
        if not post_id:
            post_id = f"reddit_{abs(hash(text)):016x}"

        created_at = normalize_timestamp(
            raw.get("created_utc") or raw.get("created_at") or raw.get("timestamp")
        )

        shock_id = self._route_shock(text)

        return build_post_payload(
            post_id=f"reddit:{post_id}",
            text=text,
            created_at=created_at,
            lang="en",
            source="archive",
            archive_id=f"reddit_{subreddit}",
            platform="reddit",
            shock_id=shock_id,
            author_did=f"reddit:{raw.get('author', '')}",
            author_handle=raw.get("author"),
            author_description=None,
            inference_method="subreddit_proxy",
        )

    # ── News: Webhose ─────────────────────────────────────────────────────────

    def load_news(self, source: str = "all") -> list[dict]:
        """Load news archive posts. source: '3dlnews' | 'webhose' | 'all'."""
        news_root = self._root / "news"
        if not news_root.exists():
            logger.debug("News archive directory not found: %s", news_root)
            return []

        posts: list[dict] = []
        if source in ("webhose", "all"):
            webhose_root = news_root / "webhose"
            if webhose_root.exists():
                for path in sorted(webhose_root.rglob("*.jsonl")):
                    n_before = len(posts)
                    posts.extend(self._load_webhose_file(path))
                    logger.debug(
                        "Webhose archive %s: loaded %d posts",
                        path, len(posts) - n_before,
                    )

        if source in ("3dlnews", "all"):
            dlnews_root = news_root / "3dlnews"
            if dlnews_root.exists():
                for path in sorted(dlnews_root.rglob("*.jsonl")):
                    n_before = len(posts)
                    posts.extend(self._load_3dlnews_file(path))
                    logger.debug(
                        "3DLNews archive %s: loaded %d posts",
                        path, len(posts) - n_before,
                    )

        logger.info("News archive (%s): %d posts total", source, len(posts))
        return posts

    def _load_webhose_file(self, path: Path) -> list[dict]:
        records: list[dict] = []
        with open(path, encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError as exc:
                    logger.warning("%s line %d: JSON error (%s)", path, lineno, exc)
                    continue
                if "payload" in raw:
                    raw = raw["payload"]
                record = self._normalize_webhose(raw)
                if record is not None:
                    records.append(record)
        return records

    def _normalize_webhose(self, raw: dict[str, Any]) -> dict | None:
        """Normalize a Webhose API article to canonical post schema."""
        text = (
            raw.get("text")
            or raw.get("body")
            or raw.get("title")
            or raw.get("summary")
            or ""
        )
        if not isinstance(text, str):
            text = ""
        text = text.strip()
        if not text:
            return None

        thread = raw.get("thread") or {}
        post_id = str(raw.get("uuid") or raw.get("id") or "")
        if not post_id:
            post_id = f"webhose_{abs(hash(text)):016x}"

        site = thread.get("site") or raw.get("site_full") or raw.get("site") or ""
        outlet = site.replace("www.", "").split(".")[0] if site else "unknown"
        created_at = normalize_timestamp(
            raw.get("published") or raw.get("crawled") or raw.get("created_at")
        )

        shock_id = self._route_shock(text)

        return build_post_payload(
            post_id=f"webhose:{post_id}",
            text=text,
            created_at=created_at,
            lang=raw.get("language") or "en",
            source="archive",
            archive_id="webhose",
            platform="news",
            shock_id=shock_id,
            author_did=None,
            author_handle=outlet,
            author_description=json.dumps({"outlet": outlet}),
            inference_method=None,
        )

    def _load_3dlnews_file(self, path: Path) -> list[dict]:
        records: list[dict] = []
        with open(path, encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError as exc:
                    logger.warning("%s line %d: JSON error (%s)", path, lineno, exc)
                    continue
                if "payload" in raw:
                    raw = raw["payload"]
                record = self._normalize_3dlnews(raw)
                if record is not None:
                    records.append(record)
        return records

    def _normalize_3dlnews(self, raw: dict[str, Any]) -> dict | None:
        """Normalize a 3DLNews2 metadata+article record to canonical post schema."""
        text = (
            raw.get("content")
            or raw.get("article_text")
            or raw.get("text")
            or raw.get("title")
            or ""
        )
        if not isinstance(text, str):
            text = ""
        text = text.strip()
        if not text:
            return None

        post_id = str(raw.get("id") or raw.get("url") or "")
        if not post_id:
            post_id = f"3dlnews_{abs(hash(text)):016x}"

        outlet = raw.get("outlet") or raw.get("source") or raw.get("domain") or "unknown"
        created_at = normalize_timestamp(
            raw.get("date") or raw.get("published_at") or raw.get("created_at")
        )

        shock_id = self._route_shock(text)

        return build_post_payload(
            post_id=f"3dlnews:{post_id}",
            text=text,
            created_at=created_at,
            lang=raw.get("language") or "en",
            source="archive",
            archive_id="3dlnews",
            platform="news",
            shock_id=shock_id,
            author_did=None,
            author_handle=outlet,
            author_description=json.dumps({"outlet": outlet}),
            inference_method=None,
        )

    # ── Discord ───────────────────────────────────────────────────────────────

    def load_discord(self) -> list[dict]:
        """Load Discord export archives from data/archives/discord/*.jsonl."""
        discord_root = self._root / "discord"
        if not discord_root.exists():
            logger.debug("Discord archive directory not found: %s", discord_root)
            return []

        posts: list[dict] = []
        for path in sorted(discord_root.rglob("*.jsonl")):
            with open(path, encoding="utf-8") as f:
                for lineno, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        raw = json.loads(line)
                    except json.JSONDecodeError as exc:
                        logger.warning(
                            "%s line %d: JSON error (%s)", path, lineno, exc
                        )
                        continue
                    if "payload" in raw:
                        raw = raw["payload"]
                    record = self._normalize_discord(raw)
                    if record is not None:
                        posts.append(record)

        logger.info("Discord archive: %d posts total from %s", len(posts), discord_root)
        return posts

    def _normalize_discord(self, raw: dict[str, Any]) -> dict | None:
        """Normalize a Discord message export to canonical post schema."""
        text = raw.get("content") or raw.get("text") or ""
        if not isinstance(text, str):
            text = ""
        text = text.strip()
        if not text:
            return None

        post_id = str(raw.get("id") or raw.get("message_id") or "")
        if not post_id:
            post_id = f"discord_{abs(hash(text)):016x}"

        author = raw.get("author") or {}
        author_id = author.get("id") if isinstance(author, dict) else None
        author_handle = author.get("username") if isinstance(author, dict) else None

        created_at = normalize_timestamp(raw.get("timestamp") or raw.get("created_at"))
        shock_id = self._route_shock(text)

        return build_post_payload(
            post_id=f"discord:{post_id}",
            text=text,
            created_at=created_at,
            lang="en",
            source="archive",
            archive_id="discord",
            platform="discord",
            shock_id=shock_id,
            author_did=f"discord:{author_id}" if author_id else None,
            author_handle=author_handle,
            author_description=None,
            inference_method=None,
        )

    # ── Convenience ───────────────────────────────────────────────────────────

    def load_all(self) -> list[dict]:
        """Load all available archive sources."""
        return self.load_reddit() + self.load_news(source="all") + self.load_discord()