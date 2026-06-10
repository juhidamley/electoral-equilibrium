"""archive: Historical archive loader for pre-existing post collections.

Loads posts from JUHIDRIVE archives and normalizes them to the canonical
post payload schema. Three loader classes are provided:

  HistoricalArchiveLoader  — JSONL / CSV archives (Twitter, Reddit, Discord, News)
  TelegramLoader           — Telegram JSON message exports
  TikTokLoader             — TikTok video metadata JSONL

All loaders share the same `load()` interface:
    load(shock_id, archive_id, window_hours=72, bot_blocklist=None)

Platform proxies applied when no user.description bio is available:
  reddit*   → subreddit_proxy (per SUBREDDIT_PROXY dict)
  telegram  → platform_proxy (evangelical / conservative Protestant)
  tiktok    → platform_proxy (secular / younger / diverse)
  truthsocial → platform_proxy (evangelical / MAGA)
"""

from __future__ import annotations

import csv
import json
import logging
import os
from datetime import datetime, timedelta, timezone
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
            logger.debug("Reddit archive %s: loaded %d posts", path, len(posts) - n_before)

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
        text = raw.get("selftext") or raw.get("body") or raw.get("text") or raw.get("title") or ""
        if not isinstance(text, str):
            text = ""
        text = text.strip()
        if not text or text == "[deleted]" or text == "[removed]":
            return None

        post_id = str(raw.get("id") or raw.get("post_id") or raw.get("name") or "")
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
                        path,
                        len(posts) - n_before,
                    )

        if source in ("3dlnews", "all"):
            dlnews_root = news_root / "3dlnews"
            if dlnews_root.exists():
                for path in sorted(dlnews_root.rglob("*.jsonl")):
                    n_before = len(posts)
                    posts.extend(self._load_3dlnews_file(path))
                    logger.debug(
                        "3DLNews archive %s: loaded %d posts",
                        path,
                        len(posts) - n_before,
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
        text = raw.get("text") or raw.get("body") or raw.get("title") or raw.get("summary") or ""
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
                        logger.warning("%s line %d: JSON error (%s)", path, lineno, exc)
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

    # ── Orchestrated pipeline load ─────────────────────────────────────────────

    def load(
        self,
        shock_id: str,
        archive_id: str,
        window_hours: int = 72,
        bot_blocklist: set[str] | None = None,
        output_root: str | Path | None = None,
        seed: int | None = None,
    ) -> list[dict]:
        """Full pipeline load for one (shock_id, archive_id) pair.

        Steps:
          1. Resolve archive directory (JUHIDRIVE data_root from base.json)
          2. Detect file format and load all posts in the directory
          3. Filter to shock window ± window_hours
          4. Drop posts whose author_did is in bot_blocklist
          5. Bio-classify posts that have author_description (Pi server)
          6. Apply platform proxy for posts without a bio
          7. Save to rawdata/social/archive/{shock_id}/{archive_id}.jsonl

        Returns the list of normalized payload dicts (envelope stripped).
        """
        # ── 1. Resolve archive directory ─────────────────────────────────────
        # Resolution order: JUHIDRIVE_ROOT env var > base.json data_root > self._root
        config_path = _REPO_ROOT / "configs" / "base.json"
        try:
            _cfg_root = json.loads(config_path.read_text()).get("data_root", str(self._root))
        except (OSError, json.JSONDecodeError):
            _cfg_root = str(self._root)
        data_root = Path(os.environ.get("JUHIDRIVE_ROOT") or _cfg_root)

        archive_dir = _resolve_archive_dir(data_root, archive_id)
        if archive_dir is None:
            logger.error(
                "Archive directory not found for archive_id=%s under %s", archive_id, data_root
            )
            return []

        # ── 2. Load posts ─────────────────────────────────────────────────────
        posts = _load_from_dir(archive_dir, archive_id, self._route_shock)

        # ── 3. Filter to shock window ─────────────────────────────────────────
        shock_dt = _shock_date(shock_id, self._shocks_path)
        if shock_dt is not None:
            posts = _filter_window(posts, shock_dt, window_hours)
            logger.info(
                "load(%s, %s): %d posts within ±%dh of %s",
                shock_id,
                archive_id,
                len(posts),
                window_hours,
                shock_dt.strftime("%Y-%m-%d"),
            )
        else:
            logger.warning(
                "load: shock_id=%s not found in shocks.json — skipping window filter", shock_id
            )

        # ── 4. Bot blocklist ──────────────────────────────────────────────────
        if bot_blocklist:
            before = len(posts)
            posts = [p for p in posts if p.get("author_did") not in bot_blocklist]
            dropped = before - len(posts)
            if dropped:
                logger.info("load: dropped %d bot-listed posts", dropped)

        # ── 5. Bio classification (posts with author_description) ─────────────
        posts = _apply_bio_classification(posts, config_path)

        # ── 6. Platform proxy fallback ────────────────────────────────────────
        for post in posts:
            if post.get("inference_method") is None:
                proxy = _platform_proxy(post.get("platform", ""))
                if proxy:
                    post["inference_method"] = "platform_proxy"

        # ── 7. Save to rawdata/social/archive ────────────────────────────────
        if output_root is None:
            output_root = _REPO_ROOT / "rawdata" / "social" / "archive"
        out_path = Path(output_root) / shock_id / f"{archive_id}.jsonl"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        from electoral.nlp.collectors.schema import append_post_record

        for post in posts:
            append_post_record(out_path, post, seed=seed)
        logger.info("load: wrote %d posts → %s", len(posts), out_path)

        return posts

    # ── Twitter / Kaggle CSV ──────────────────────────────────────────────────

    def load_twitter_csv(self, archive_id: str | None = None) -> list[dict]:
        """Load Twitter-format CSV/JSONL archives (Kaggle election datasets).

        Handles multiple field name conventions across different Kaggle dumps:
          - text / tweet / full_text / status
          - created_at / date / timestamp (Twitter v1 format supported)
          - user_description / description / user.description
        """
        twitter_root = self._root / "twitter"
        if not twitter_root.exists():
            logger.debug("Twitter archive directory not found: %s", twitter_root)
            return []

        search_root = (twitter_root / archive_id) if archive_id else twitter_root
        posts: list[dict] = []
        for path in sorted(search_root.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix in (".json", ".jsonl"):
                posts.extend(self._load_twitter_jsonl(path, archive_id or path.stem))
            elif path.suffix in (".csv", ".tsv"):
                posts.extend(self._load_twitter_csv_file(path, archive_id or path.stem))
        logger.info("Twitter archive (%s): %d posts", archive_id or "all", len(posts))
        return posts

    _TEXT_CANDIDATES = ["text", "tweet", "full_text", "status", "content", "body"]
    _DATE_CANDIDATES = ["created_at", "date", "timestamp", "time", "created", "posted_at"]
    _BIO_CANDIDATES = [
        "user_description",
        "description",
        "user.description",
        "author_description",
        "user_bio",
    ]
    _ID_CANDIDATES = ["id", "tweet_id", "post_id", "id_str", "tweetid"]

    @classmethod
    def _pick(cls, row: dict, candidates: list[str]) -> str | None:
        for key in candidates:
            val = row.get(key)
            if val and str(val).strip():
                return str(val).strip()
        return None

    def _load_twitter_jsonl(self, path: Path, archive_id: str) -> list[dict]:
        records: list[dict] = []
        with open(path, encoding="utf-8", errors="replace") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "payload" in raw:
                    raw = raw["payload"]
                rec = self._normalize_twitter_row(raw, archive_id)
                if rec:
                    records.append(rec)
        return records

    def _load_twitter_csv_file(self, path: Path, archive_id: str) -> list[dict]:
        records: list[dict] = []
        try:
            csv.field_size_limit(10_000_000)
            delimiter = "\t" if path.suffix == ".tsv" else ","
            with open(path, encoding="utf-8", errors="replace", newline="") as f:
                content = f.read().replace("\x00", "")
            reader = csv.DictReader(content.splitlines(keepends=True), delimiter=delimiter)
            for row in reader:
                rec = self._normalize_twitter_row(dict(row), archive_id)
                if rec:
                    records.append(rec)
        except (csv.Error, OSError) as exc:
            logger.warning("Twitter CSV read error %s: %s", path, exc)
        return records

    # Event code → shock_id for LIWC SAV archives (metoo / kavanaugh dataset)
    _SAV_EVENT_TO_SHOCK: dict[int, str] = {
        1: "metoo_2017",
        2: "kavanaugh_2018",
        3: "kavanaugh_2018",
        4: "kavanaugh_2018",
    }

    def _load_sav_file(self, path: Path, archive_id: str) -> list[dict]:
        """Load a SPSS SAV file using pyreadstat and return canonical payloads.

        Expects a ``Tweet`` column (text) and an ``Event`` column (integer code).
        No timestamp is available; created_at is set to the shock's known date
        from shocks.json. author_description is always None (no bio field).
        """
        try:
            import pyreadstat
        except ImportError:
            logger.warning("pyreadstat not installed; skipping SAV file %s", path)
            return []

        try:
            df, _meta = pyreadstat.read_sav(str(path))
        except Exception as exc:
            logger.warning("SAV read error %s: %s", path, exc)
            return []

        # Build shock_id → date string map from shocks.json
        shock_dates: dict[str, str] = {}
        if self._shocks_path.exists():
            try:
                for shock in load_shocks(self._shocks_path):
                    if shock.get("date"):
                        shock_dates[shock["id"]] = shock["date"]
            except Exception:
                pass

        records: list[dict] = []
        for _, row in df.iterrows():
            text = str(row.get("Tweet") or "").strip()
            if not text:
                continue

            try:
                event_code = int(row.get("Event") or 0)
            except (ValueError, TypeError):
                event_code = 0
            shock_id = self._SAV_EVENT_TO_SHOCK.get(event_code)

            created_at = normalize_timestamp(shock_dates.get(shock_id or ""))

            post_id = f"liwc_{archive_id}_{abs(hash(text)):016x}"
            records.append(
                build_post_payload(
                    post_id=post_id,
                    text=text,
                    created_at=created_at,
                    lang="en",
                    source="archive",
                    archive_id=archive_id,
                    platform="twitter_liwc",
                    shock_id=shock_id,
                    author_did=None,
                    author_handle=None,
                    author_description=None,
                    inference_method=None,
                )
            )

        logger.info("SAV %s: %d records loaded", path.name, len(records))
        return records

    def _normalize_twitter_row(self, raw: dict[str, Any], archive_id: str) -> dict | None:
        text = self._pick(raw, self._TEXT_CANDIDATES)
        if not text:
            return None
        post_id = self._pick(raw, self._ID_CANDIDATES) or f"tw_{abs(hash(text)):016x}"
        ts_raw = self._pick(raw, self._DATE_CANDIDATES)
        created_at = normalize_timestamp(ts_raw)
        user = raw.get("user") or {}
        bio = self._pick(raw, self._BIO_CANDIDATES) or (
            user.get("description") if isinstance(user, dict) else None
        )
        author_id = str(
            raw.get("user_id")
            or raw.get("user_id_str")
            or (user.get("id_str") if isinstance(user, dict) else None)
            or ""
        )
        author_handle = (
            raw.get("username")
            or raw.get("screen_name")
            or (user.get("screen_name") if isinstance(user, dict) else None)
        )
        shock_id = self._route_shock(text)
        return build_post_payload(
            post_id=f"twitter:{post_id}",
            text=text,
            created_at=created_at,
            lang=raw.get("lang") or raw.get("language") or "en",
            source="archive",
            archive_id=archive_id,
            platform="twitter",
            shock_id=shock_id,
            author_did=f"twitter:{author_id}" if author_id else None,
            author_handle=author_handle,
            author_description=bio,
            inference_method=None,
        )


# ── Module-level helpers ───────────────────────────────────────────────────────

# Platform → default bloc proxy for posts with no bio
_PLATFORM_PROXY_MAP: dict[str, str] = {
    "telegram": "evangelical",
    "truthsocial": "evangelical",
    "tiktok": "secular",
    "discord": "secular",
}


def _platform_proxy(platform: str) -> str | None:
    return _PLATFORM_PROXY_MAP.get(platform.lower().split("_")[0])


def _resolve_archive_dir(data_root: Path, archive_id: str) -> Path | None:
    """Search for archive_id under any platform subdirectory of data_root."""
    for platform_dir in sorted(data_root.glob("*/")):
        candidate = platform_dir / archive_id
        if candidate.exists():
            return candidate
    flat = data_root / archive_id
    return flat if flat.exists() else None


def _shock_date(shock_id: str, shocks_path: Path) -> datetime | None:
    """Return the shock anchor datetime from shocks.json date_window or date field."""
    try:
        shocks = load_shocks(shocks_path)
    except (OSError, json.JSONDecodeError):
        return None
    for shock in shocks:
        if shock["id"] == shock_id:
            dw = shock.get("date_window", {})
            date_str = dw.get("shock_date") or shock.get("date")
            if date_str:
                return datetime.fromisoformat(date_str).replace(tzinfo=timezone.utc)
    return None


def _filter_window(posts: list[dict], shock_dt: datetime, window_hours: int) -> list[dict]:
    """Keep posts within shock_dt ± window_hours."""
    lo = shock_dt - timedelta(hours=window_hours)
    hi = shock_dt + timedelta(hours=window_hours)
    kept = []
    for post in posts:
        raw_ts = post.get("created_at")
        if not raw_ts:
            kept.append(post)  # no timestamp → don't discard
            continue
        try:
            ts = datetime.fromisoformat(str(raw_ts))
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            if lo <= ts <= hi:
                kept.append(post)
        except (ValueError, TypeError):
            kept.append(post)
    return kept


def _load_from_dir(
    archive_dir: Path,
    archive_id: str,
    route_shock,
) -> list[dict]:
    """Dispatch to the right per-file loader based on extension."""
    loader = HistoricalArchiveLoader(archive_root=archive_dir.parent)
    loader._keyword_index = None  # will be rebuilt lazily
    posts: list[dict] = []
    for path in sorted(archive_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix in (".jsonl", ".json"):
            posts.extend(loader._load_twitter_jsonl(path, archive_id))
        elif path.suffix in (".csv", ".tsv"):
            posts.extend(loader._load_twitter_csv_file(path, archive_id))
        elif path.suffix == ".sav":
            posts.extend(loader._load_sav_file(path, archive_id))
    return posts


def _apply_bio_classification(posts: list[dict], config_path: Path) -> list[dict]:
    """Run bio classifier on posts that have author_description.

    Falls back gracefully if the Pi server is unreachable or BioClassifier
    raises an import error.
    """
    with_bio = [p for p in posts if p.get("author_description")]
    if not with_bio:
        return posts
    try:
        from electoral.nlp.bio_classifier import BioClassifier

        classifier = BioClassifier.from_config(config_path)
        results = classifier.classify_batch(with_bio)
        for post, result in zip(with_bio, results):
            if result.is_estimable():
                post["inference_method"] = result.inference_method
    except Exception as exc:
        logger.warning("Bio classification skipped: %s", exc)
    return posts


# ── TelegramLoader ─────────────────────────────────────────────────────────────


class TelegramLoader:
    """Normalize Telegram JSON message exports to canonical post payload schema.

    Telegram field names differ from Twitter:
      message     → text
      date        → created_at (Unix epoch or ISO string)
      from.id     → author_did
      from.username → author_handle
      id          → post_id

    No user.description field exists in Telegram exports. All posts are tagged
    with inference_method="platform_proxy" and a conservative Protestant proxy
    (Evangelical), consistent with the political Telegram channel demographic
    profile.
    """

    PLATFORM = "telegram"
    PROXY_BLOC = "evangelical"

    def __init__(
        self,
        archive_root: str | Path,
        shocks_path: str | Path | None = None,
    ) -> None:
        self._root = Path(archive_root)
        self._shocks_path = Path(shocks_path) if shocks_path else _DEFAULT_SHOCKS_PATH
        self._keyword_index: dict[str, list[str]] | None = None

    def _route_shock(self, text: str) -> str | None:
        if self._keyword_index is None:
            try:
                shocks = load_shocks(self._shocks_path)
                self._keyword_index = build_keyword_index(shocks)
            except Exception:
                self._keyword_index = {}
        matched = match_shocks(text, self._keyword_index)
        if len(matched) == 1:
            return next(iter(matched))
        return sorted(matched)[0] if matched else None

    def load(
        self,
        shock_id: str,
        archive_id: str,
        window_hours: int = 72,
        bot_blocklist: set[str] | None = None,
        output_root: str | Path | None = None,
        seed: int | None = None,
    ) -> list[dict]:
        """Load Telegram archive for one shock, filter, and save."""
        archive_dir = self._root / archive_id
        if not archive_dir.exists():
            # Try platform subdirectory layout
            archive_dir = self._root / "telegram" / archive_id
        if not archive_dir.exists():
            logger.error("TelegramLoader: archive not found: %s", archive_dir)
            return []

        posts: list[dict] = []
        for path in sorted(archive_dir.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix == ".json":
                posts.extend(self._load_file(path, archive_id))
            elif path.suffix == ".csv":
                posts.extend(self._load_csv_file(path, archive_id))

        shock_dt = _shock_date(shock_id, self._shocks_path)
        if shock_dt:
            posts = _filter_window(posts, shock_dt, window_hours)

        if bot_blocklist:
            before = len(posts)
            posts = [p for p in posts if p.get("author_did") not in bot_blocklist]
            logger.info("TelegramLoader: dropped %d bot-listed posts", before - len(posts))

        if output_root is None:
            output_root = _REPO_ROOT / "rawdata" / "social" / "archive"
        out_path = Path(output_root) / shock_id / f"{archive_id}.jsonl"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        from electoral.nlp.collectors.schema import append_post_record

        for post in posts:
            append_post_record(out_path, post, seed=seed)
        logger.info("TelegramLoader: wrote %d posts → %s", len(posts), out_path)
        return posts

    def _load_file(self, path: Path, archive_id: str) -> list[dict]:
        """Load one Telegram JSON export (may be a dict with 'messages' list
        or a JSONL file with one message per line)."""
        records: list[dict] = []
        try:
            with open(path, encoding="utf-8", errors="replace") as f:
                raw = json.load(f)
            # Telegram Desktop export: {"name": ..., "messages": [...]}
            messages = raw.get("messages") if isinstance(raw, dict) else raw
            if isinstance(messages, list):
                for msg in messages:
                    rec = self._normalize(msg, archive_id)
                    if rec:
                        records.append(rec)
        except json.JSONDecodeError:
            # Try JSONL
            with open(path, encoding="utf-8", errors="replace") as f:
                for lineno, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        msg = json.loads(line)
                        rec = self._normalize(msg, archive_id)
                        if rec:
                            records.append(rec)
                    except json.JSONDecodeError:
                        logger.debug("%s line %d: JSON error", path.name, lineno)
        except OSError as exc:
            logger.warning("TelegramLoader read error %s: %s", path, exc)
        return records

    _TOXICITY_FIELDS = ("toxicity", "severe_toxicity", "identity_attack")

    def _load_csv_file(self, path: Path, archive_id: str) -> list[dict]:
        """Load a Telegram CSV export (telegram_2024 schema) using pandas.

        Column mapping:
          content       → text  (rows where content is NaN or empty are skipped)
          message_id    → post_id  (telegram: prefix)
          date          → created_at  (ISO 8601)
          language      → lang
          from_id       → author_did  (telegram: prefix)
          post_author   → author_handle
        No bio field; inference_method = "platform_proxy".
        toxicity, severe_toxicity, identity_attack stored in metadata if present.
        """
        try:
            import pandas as pd
        except ImportError:
            logger.warning("pandas not installed; skipping CSV %s", path)
            return []

        csv.field_size_limit(10_000_000)
        try:
            df = pd.read_csv(path, dtype=str, keep_default_na=True)
        except Exception as exc:
            logger.warning("TelegramLoader CSV read error %s: %s", path, exc)
            return []

        records: list[dict] = []
        for _, row in df.iterrows():
            text = row.get("content")
            if pd.isna(text) or not str(text).strip():
                continue
            text = str(text).strip()

            raw_id = row.get("message_id")
            post_id = (
                f"telegram:{str(raw_id).strip()}"
                if not pd.isna(raw_id) and str(raw_id).strip()
                else f"telegram:tg_{abs(hash(text)):016x}"
            )

            raw_date = row.get("date")
            created_at = normalize_timestamp(
                str(raw_date).strip() if not pd.isna(raw_date) else None
            )

            raw_lang = row.get("language")
            lang = str(raw_lang).strip() if not pd.isna(raw_lang) and str(raw_lang).strip() else "en"

            raw_from = row.get("from_id")
            author_did = (
                f"telegram:{str(raw_from).strip()}"
                if not pd.isna(raw_from) and str(raw_from).strip()
                else None
            )

            raw_handle = row.get("post_author")
            author_handle = (
                str(raw_handle).strip()
                if not pd.isna(raw_handle) and str(raw_handle).strip()
                else None
            )

            shock_id = self._route_shock(text)

            rec = build_post_payload(
                post_id=post_id,
                text=text,
                created_at=created_at,
                lang=lang,
                source="archive",
                archive_id=archive_id,
                platform=self.PLATFORM,
                shock_id=shock_id,
                author_did=author_did,
                author_handle=author_handle,
                author_description=None,
                inference_method="platform_proxy",
            )

            metadata: dict[str, float] = {}
            for field in self._TOXICITY_FIELDS:
                val = row.get(field)
                if val is not None and not pd.isna(val):
                    try:
                        metadata[field] = float(val)
                    except (ValueError, TypeError):
                        pass
            if metadata:
                rec["metadata"] = metadata

            records.append(rec)

        logger.info("TelegramLoader CSV %s: %d records loaded", path.name, len(records))
        return records

    def _normalize(self, msg: dict[str, Any], archive_id: str) -> dict | None:
        # Telegram message text can be a string or a list of text entities
        raw_text = msg.get("text") or msg.get("message") or ""
        if isinstance(raw_text, list):
            raw_text = " ".join(
                e.get("text", "") if isinstance(e, dict) else str(e) for e in raw_text
            )
        text = str(raw_text).strip()
        if not text:
            return None

        post_id = str(msg.get("id") or f"tg_{abs(hash(text)):016x}")
        created_at = normalize_timestamp(msg.get("date") or msg.get("date_unixtime"))

        sender = msg.get("from") or {}
        if isinstance(sender, dict):
            author_id = str(sender.get("id") or "")
            author_handle = sender.get("username") or sender.get("first_name")
        else:
            author_id = str(sender)
            author_handle = None

        # from_id is the flat field in some export formats
        if not author_id:
            author_id = str(msg.get("from_id") or "")

        shock_id = self._route_shock(text)

        return build_post_payload(
            post_id=f"telegram:{post_id}",
            text=text,
            created_at=created_at,
            lang=msg.get("lang") or "en",
            source="archive",
            archive_id=archive_id,
            platform=self.PLATFORM,
            shock_id=shock_id,
            author_did=f"telegram:{author_id}" if author_id else None,
            author_handle=author_handle,
            author_description=None,
            inference_method="platform_proxy",
        )


# ── TikTokLoader ───────────────────────────────────────────────────────────────


class TikTokLoader:
    """Normalize TikTok video metadata JSONL to canonical post payload schema.

    TikTok field names:
      desc / text / title  → text (video caption)
      createTime           → created_at (Unix epoch)
      author.uniqueId      → author_handle
      author.id            → author_did
      id                   → post_id

    No bio field is available in standard TikTok scrape output. All posts are
    tagged with inference_method="platform_proxy" and a secular/diverse/younger
    proxy, reflecting TikTok's demographic profile relative to the other
    platforms in the pipeline.
    """

    PLATFORM = "tiktok"
    PROXY_BLOC = "secular"

    def __init__(
        self,
        archive_root: str | Path,
        shocks_path: str | Path | None = None,
    ) -> None:
        self._root = Path(archive_root)
        self._shocks_path = Path(shocks_path) if shocks_path else _DEFAULT_SHOCKS_PATH
        self._keyword_index: dict[str, list[str]] | None = None

    def _route_shock(self, text: str) -> str | None:
        if self._keyword_index is None:
            try:
                shocks = load_shocks(self._shocks_path)
                self._keyword_index = build_keyword_index(shocks)
            except Exception:
                self._keyword_index = {}
        matched = match_shocks(text, self._keyword_index)
        if len(matched) == 1:
            return next(iter(matched))
        return sorted(matched)[0] if matched else None

    def load(
        self,
        shock_id: str,
        archive_id: str,
        window_hours: int = 72,
        bot_blocklist: set[str] | None = None,
        output_root: str | Path | None = None,
        seed: int | None = None,
    ) -> list[dict]:
        """Load TikTok archive for one shock, filter, and save."""
        archive_dir = self._root / archive_id
        if not archive_dir.exists():
            archive_dir = self._root / "tiktok" / archive_id
        if not archive_dir.exists():
            logger.error("TikTokLoader: archive not found: %s", archive_dir)
            return []

        posts: list[dict] = []
        for path in sorted(archive_dir.rglob("*.jsonl")):
            posts.extend(self._load_file(path, archive_id))

        shock_dt = _shock_date(shock_id, self._shocks_path)
        if shock_dt:
            posts = _filter_window(posts, shock_dt, window_hours)

        if bot_blocklist:
            before = len(posts)
            posts = [p for p in posts if p.get("author_did") not in bot_blocklist]
            logger.info("TikTokLoader: dropped %d bot-listed posts", before - len(posts))

        if output_root is None:
            output_root = _REPO_ROOT / "rawdata" / "social" / "archive"
        out_path = Path(output_root) / shock_id / f"{archive_id}.jsonl"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        from electoral.nlp.collectors.schema import append_post_record

        for post in posts:
            append_post_record(out_path, post, seed=seed)
        logger.info("TikTokLoader: wrote %d posts → %s", len(posts), out_path)
        return posts

    def _load_file(self, path: Path, archive_id: str) -> list[dict]:
        records: list[dict] = []
        with open(path, encoding="utf-8", errors="replace") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                    if "payload" in raw:
                        raw = raw["payload"]
                    rec = self._normalize(raw, archive_id)
                    if rec:
                        records.append(rec)
                except json.JSONDecodeError:
                    logger.debug("%s line %d: JSON error", path.name, lineno)
        return records

    def _normalize(self, raw: dict[str, Any], archive_id: str) -> dict | None:
        text = str(raw.get("desc") or raw.get("text") or raw.get("title") or "").strip()
        if not text:
            return None

        post_id = str(raw.get("id") or f"tt_{abs(hash(text)):016x}")
        created_at = normalize_timestamp(
            raw.get("createTime") or raw.get("create_time") or raw.get("created_at")
        )

        author = raw.get("author") or {}
        if isinstance(author, dict):
            author_id = str(author.get("id") or author.get("uid") or "")
            author_handle = author.get("uniqueId") or author.get("nickname")
        else:
            author_id = str(author)
            author_handle = None

        shock_id = self._route_shock(text)

        return build_post_payload(
            post_id=f"tiktok:{post_id}",
            text=text,
            created_at=created_at,
            lang=raw.get("textLanguage") or raw.get("lang") or "en",
            source="archive",
            archive_id=archive_id,
            platform=self.PLATFORM,
            shock_id=shock_id,
            author_did=f"tiktok:{author_id}" if author_id else None,
            author_handle=author_handle,
            author_description=None,
            inference_method="platform_proxy",
        )
