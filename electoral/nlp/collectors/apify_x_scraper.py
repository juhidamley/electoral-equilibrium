"""Apify X (Twitter) scraper — per-shock event collection.

Runs once per shock event (not continuously). Stays within the 500-result
free tier limit. Writes append-only JSONL to:
    rawdata/social/apify/{shock_id}/intel_mac_posts.jsonl

Write-once ownership: only the Intel Mac runs this script.

Usage:
    python -m electoral.nlp.collectors.apify_x_scraper \\
        --shock-id ayatollah_assassination \\
        --shocks configs/shocks.json \\
        --output rawdata/social \\
        --seed 42

    # Or with explicit keyword override:
    python -m electoral.nlp.collectors.apify_x_scraper \\
        --shock-id kavanaugh_2018 \\
        --keywords '["Kavanaugh", "SCOTUS", "Christine Ford"]' \\
        --max-items 500

Environment variables:
    APIFY_TOKEN     Apify API token from console.apify.com/account/integrations
    APIFY_ACTOR_ID  Actor to run (default: apidojo/tweet-scraper)

Free tier:
    500 results per actor run. The free Apify account includes $5/month credit.
    At ~$0.001 per result, 500 results ≈ $0.50 per shock run.

Actor options (configure via APIFY_ACTOR_ID):
    apidojo/tweet-scraper    — most reliable, handles pagination
    quacker/twitter-scraper  — alternative, may have different field names
    Set via env or --actor-id flag.

Dependencies:
    pip install apify-client>=1.6
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

from electoral.nlp.collectors.schema import (
    append_post_record,
    build_post_payload,
    load_shocks,
    normalize_timestamp,
)

logger = logging.getLogger(__name__)

# ── Default actor configuration ───────────────────────────────────────────────

DEFAULT_ACTOR_ID = "apidojo/tweet-scraper"
MAX_ITEMS_FREE_TIER = 500  # Hard limit to stay within free tier

# Apify actor input field names vary by actor version.
# We support two common schemas and fall back gracefully.
_ACTOR_SCHEMAS: dict[str, dict[str, str]] = {
    "apidojo/tweet-scraper": {
        "searchTerms_field": "searchTerms",
        "maxItems_field": "maxItems",
        "lang_field": "tweetLanguage",
        "since_field": "since",
        "until_field": "until",
        "mode": "Latest",
    },
    "quacker/twitter-scraper": {
        "searchTerms_field": "searchTerms",
        "maxItems_field": "maxItems",
        "lang_field": "lang",
        "since_field": "since",
        "until_field": "until",
        "mode": "Latest",
    },
}


# ── Apify import with guard ───────────────────────────────────────────────────


def _import_apify():
    try:
        from apify_client import ApifyClient

        return ApifyClient
    except ImportError as exc:
        raise ImportError(
            "apify-client is required for the Apify X scraper.\n"
            "Install with: pip install apify-client>=1.6"
        ) from exc


# ── Field normalizers for different actor output schemas ─────────────────────


def _extract_text(item: dict[str, Any]) -> str:
    """Extract tweet text, handling actor-specific field names."""
    for field in ("full_text", "text", "tweet_text", "tweetText", "content"):
        val = item.get(field)
        if val and isinstance(val, str):
            # Strip Twitter's truncation marker
            return val.replace("… https://t.co/", "").replace("…", "").strip()
    return ""


def _extract_id(item: dict[str, Any]) -> str:
    """Extract the tweet ID, falling back to URL construction."""
    for field in ("id", "id_str", "tweet_id", "tweetId", "statusId"):
        val = item.get(field)
        if val:
            return str(val)
    # Construct from URL if present
    url = item.get("url") or item.get("tweet_url") or ""
    if "twitter.com" in url or "x.com" in url:
        parts = url.rstrip("/").rsplit("/", 1)
        if parts:
            return parts[-1]
    return ""


def _extract_created_at(item: dict[str, Any]) -> str:
    """Extract and normalize the tweet creation timestamp."""
    for field in (
        "created_at",
        "createdAt",
        "timestamp",
        "date",
        "tweet_created_at",
        "publishedAt",
    ):
        val = item.get(field)
        if val:
            return normalize_timestamp(str(val))
    return normalize_timestamp(None)


def _extract_lang(item: dict[str, Any]) -> str:
    """Extract language code."""
    for field in ("lang", "language", "tweet_lang", "tweetLanguage"):
        val = item.get(field)
        if val and isinstance(val, str):
            return val.lower().strip()
    return ""


def _extract_author(item: dict[str, Any]) -> tuple[str | None, str | None, str | None]:
    """Extract (author_id, author_handle, author_description) from tweet item."""
    # Author info may be nested under user/author/authorMeta
    author_obj = item.get("user") or item.get("author") or item.get("authorMeta") or {}

    # Author ID
    author_id = None
    for field in ("id", "id_str", "user_id", "userId"):
        val = author_obj.get(field) or item.get(f"user_{field}")
        if val:
            author_id = f"twitter:{val}"
            break

    # Handle / screen name
    handle = None
    for field in ("screen_name", "screenName", "handle", "username", "name"):
        val = author_obj.get(field) or item.get(f"user_{field}")
        if val and isinstance(val, str):
            handle = val.lstrip("@")
            break

    # Bio description
    description = None
    for field in ("description", "bio", "profile_description", "user_description"):
        val = author_obj.get(field) or item.get(f"user_{field}")
        if val and isinstance(val, str):
            description = val.strip()
            break

    return author_id, handle, description


def normalize_apify_tweet(
    item: dict[str, Any],
    shock_id: str,
    actor_id: str = DEFAULT_ACTOR_ID,
) -> dict[str, Any] | None:
    """Normalize a raw Apify actor output item to the canonical post payload.

    Returns None if the item is missing required fields.
    """
    text = _extract_text(item)
    if not text:
        return None

    post_id = _extract_id(item)
    if not post_id:
        # Build a synthetic ID from text hash
        post_id = f"apify_{abs(hash(text)):016x}"

    author_id, author_handle, author_description = _extract_author(item)
    created_at = _extract_created_at(item)
    lang = _extract_lang(item)

    return build_post_payload(
        post_id=f"twitter:{post_id}",
        text=text,
        created_at=created_at,
        lang=lang,
        source="live_scrape",
        archive_id="apify_x",
        platform="apify_x",
        shock_id=shock_id,
        author_did=author_id,
        author_handle=author_handle,
        author_description=author_description,
        inference_method=None,
    )


# ── Scraper class ─────────────────────────────────────────────────────────────


class ApifyXScraper:
    """Runs the Apify X (Twitter) scraper actor for a specific shock event.

    Stays strictly within the 500-result free tier cap. Normalizes all output
    to the canonical post schema so the downstream merge_posts() task is blind
    to platform differences.
    """

    def __init__(
        self,
        shock_id: str,
        keywords: list[str],
        output_root: str | Path,
        seed: int | None = None,
        apify_token: str | None = None,
        actor_id: str = DEFAULT_ACTOR_ID,
        max_items: int = MAX_ITEMS_FREE_TIER,
        since: str | None = None,
        until: str | None = None,
        lang: str = "en",
    ) -> None:
        if max_items > MAX_ITEMS_FREE_TIER:
            raise ValueError(
                f"max_items={max_items} exceeds free-tier limit of {MAX_ITEMS_FREE_TIER}. "
                f"Set max_items <= {MAX_ITEMS_FREE_TIER} to avoid charges."
            )

        self._shock_id = shock_id
        self._keywords = keywords
        self._output_root = Path(output_root)
        self._seed = seed
        self._apify_token = apify_token or os.environ.get("APIFY_TOKEN")
        self._actor_id = actor_id
        self._max_items = max_items
        self._since = since
        self._until = until
        self._lang = lang

        if not self._apify_token:
            raise ValueError(
                "Apify token is required. Set APIFY_TOKEN environment variable "
                "or pass apify_token= argument."
            )

    @property
    def output_path(self) -> Path:
        return self._output_root / "apify" / self._shock_id / "intel_mac_posts.jsonl"

    def _build_actor_input(self) -> dict[str, Any]:
        """Build the actor run input dict.

        searchTerms is a list of search queries; Apify combines them with OR logic.
        We run all shock keywords as a single multi-term search to maximize
        the 500-result budget across the most relevant content.
        """
        schema = _ACTOR_SCHEMAS.get(self._actor_id, _ACTOR_SCHEMAS[DEFAULT_ACTOR_ID])

        # Join all keywords into a single OR query to use the 500 results efficiently
        # Each keyword gets roughly equal allocation
        search_queries: list[str] = self._keywords[:20]  # cap at 20 to avoid rate limits

        run_input: dict[str, Any] = {
            schema["searchTerms_field"]: search_queries,
            schema["maxItems_field"]: self._max_items,
            "sort": schema.get("mode", "Latest"),
        }

        if self._lang:
            run_input[schema["lang_field"]] = self._lang

        if self._since:
            run_input[schema["since_field"]] = self._since
        if self._until:
            run_input[schema["until_field"]] = self._until

        return run_input

    def run(self) -> int:
        """Execute the Apify actor run and write results to JSONL.

        Returns the number of records successfully written.
        """
        ApifyClient = _import_apify()
        client = ApifyClient(self._apify_token)

        run_input = self._build_actor_input()
        logger.info(
            "Starting Apify actor '%s' for shock '%s': %d keywords, max=%d",
            self._actor_id,
            self._shock_id,
            len(self._keywords),
            self._max_items,
        )
        logger.debug("Actor input: %s", json.dumps(run_input, indent=2))

        try:
            actor_run = client.actor(self._actor_id).call(
                run_input=run_input,
                timeout_secs=300,  # 5 min max; free tier runs are fast
                memory_mbytes=256,
            )
        except Exception as exc:
            logger.error("Apify actor run failed: %s", exc)
            raise

        dataset_id = actor_run.get("defaultDatasetId")
        if not dataset_id:
            logger.error("Actor run returned no dataset: %s", actor_run)
            return 0

        logger.info(
            "Actor run completed: runId=%s datasetId=%s",
            actor_run.get("id"),
            dataset_id,
        )

        written = 0
        skipped = 0
        dataset = client.dataset(dataset_id)

        for item in dataset.iterate_items():
            payload = normalize_apify_tweet(item, self._shock_id, self._actor_id)
            if payload is None:
                skipped += 1
                continue
            try:
                append_post_record(self.output_path, payload, seed=self._seed)
                written += 1
            except OSError as exc:
                logger.error("Write failed: %s", exc)

        logger.info(
            "Apify collection complete: shock=%s written=%d skipped=%d → %s",
            self._shock_id,
            written,
            skipped,
            self.output_path,
        )
        return written


# ── Entry point ───────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Apify X scraper — per-shock event collection (Intel Mac)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--shock-id",
        required=True,
        help="Shock event ID from configs/shocks.json (e.g. ayatollah_assassination)",
    )
    p.add_argument(
        "--shocks",
        default="configs/shocks.json",
        help="Path to shock event registry JSON (used to load keywords if --keywords not set)",
    )
    p.add_argument(
        "--keywords",
        default=None,
        help="JSON array of keywords to search. If omitted, loaded from shocks.json.",
    )
    p.add_argument(
        "--output",
        default="rawdata/social",
        help="Root directory for JSONL output (rawdata/social/apify/...)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Global pipeline seed to embed in each record",
    )
    p.add_argument(
        "--max-items",
        type=int,
        default=MAX_ITEMS_FREE_TIER,
        help=f"Maximum results per run. Free tier cap: {MAX_ITEMS_FREE_TIER}",
    )
    p.add_argument(
        "--actor-id",
        default=os.environ.get("APIFY_ACTOR_ID", DEFAULT_ACTOR_ID),
        help="Apify actor ID. Common options: apidojo/tweet-scraper, quacker/twitter-scraper",
    )
    p.add_argument(
        "--since",
        default=None,
        help="Start date for tweet collection (YYYY-MM-DD)",
    )
    p.add_argument(
        "--until",
        default=None,
        help="End date for tweet collection (YYYY-MM-DD)",
    )
    p.add_argument(
        "--lang",
        default="en",
        help="Language filter for tweets",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    # Resolve keywords: explicit --keywords flag OR load from shocks.json
    if args.keywords:
        try:
            keywords = json.loads(args.keywords)
            if not isinstance(keywords, list):
                raise ValueError("--keywords must be a JSON array of strings")
        except (json.JSONDecodeError, ValueError) as exc:
            logger.error("Invalid --keywords: %s", exc)
            sys.exit(1)
    else:
        if not Path(args.shocks).exists():
            logger.error("shocks.json not found: %s", args.shocks)
            sys.exit(1)
        shocks = load_shocks(args.shocks)
        shock_entry = next((s for s in shocks if s["id"] == args.shock_id), None)
        if shock_entry is None:
            logger.error(
                "Shock ID '%s' not found in %s. Available: %s",
                args.shock_id,
                args.shocks,
                [s["id"] for s in shocks],
            )
            sys.exit(1)
        keywords = shock_entry.get("keywords", [])
        if not keywords:
            logger.error("No keywords defined for shock '%s'", args.shock_id)
            sys.exit(1)

    if args.max_items > MAX_ITEMS_FREE_TIER:
        logger.error(
            "--max-items=%d exceeds free-tier cap of %d. "
            "This will incur Apify charges. Aborting.",
            args.max_items,
            MAX_ITEMS_FREE_TIER,
        )
        sys.exit(1)

    apify_token = os.environ.get("APIFY_TOKEN")
    if not apify_token:
        logger.error(
            "APIFY_TOKEN environment variable is not set. "
            "Get your token from console.apify.com/account/integrations"
        )
        sys.exit(1)

    scraper = ApifyXScraper(
        shock_id=args.shock_id,
        keywords=keywords,
        output_root=args.output,
        seed=args.seed,
        apify_token=apify_token,
        actor_id=args.actor_id,
        max_items=args.max_items,
        since=args.since,
        until=args.until,
        lang=args.lang,
    )

    logger.info(
        "Apify X scraper: shock=%s keywords=%s output=%s",
        args.shock_id,
        keywords,
        scraper.output_path,
    )

    written = scraper.run()
    sys.exit(0 if written > 0 else 1)


if __name__ == "__main__":
    main()
