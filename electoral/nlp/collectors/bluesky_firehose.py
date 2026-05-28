"""Bluesky AT Protocol firehose collector.

Subscribes to the com.atproto.sync.subscribeRepos firehose, filters for
app.bsky.feed.post records matching configured shock keywords, and writes
append-only JSONL to rawdata/social/bluesky/{shock_id}/intel_mac_posts.jsonl.

Write-once ownership: only the Intel Mac runs this script.
Every other machine reads via Syncthing; nothing else writes to rawdata/social/bluesky/.

Usage:
    python -m electoral.nlp.collectors.bluesky_firehose \\
        --shocks configs/shocks.json \\
        --output rawdata/social \\
        --seed 42

Environment variables:
    BSKY_HANDLE    Bluesky handle (e.g. user.bsky.social) — optional, used for auth
    BSKY_PASSWORD  App password from bsky.app/settings/app-passwords — optional

The firehose is publicly accessible without auth. Credentials are optional but
recommended for higher rate limits and to avoid IP-based throttling.

Dependencies:
    pip install atproto>=0.0.54

Architecture note:
    rawdata/social/bluesky/{shock_id}/intel_mac_posts.jsonl
    ↑ written by this script (Intel Mac only)
    ↓ read by merge_posts() Prefect task and HistoricalArchiveLoader
"""
from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import time
from pathlib import Path
from threading import Event
from typing import Any

from electoral.nlp.collectors.schema import (
    append_post_record,
    build_keyword_index,
    build_post_payload,
    extract_primary_lang,
    is_english,
    load_shocks,
    match_shocks,
)

logger = logging.getLogger(__name__)


# ── Atproto imports with version guard ───────────────────────────────────────

def _import_atproto():
    """Import atproto components, raising a clear error if not installed."""
    try:
        from atproto import FirehoseSubscribeReposClient, parse_subscribe_repos_message
        from atproto import CAR
        return FirehoseSubscribeReposClient, parse_subscribe_repos_message, CAR
    except ImportError as exc:
        raise ImportError(
            "atproto is required for the Bluesky firehose collector.\n"
            "Install with: pip install atproto>=0.0.54"
        ) from exc


# ── Collector class ───────────────────────────────────────────────────────────

class BlueskyFirehoseCollector:
    """Subscribes to the Bluesky firehose and routes matching posts to shock files.

    The firehose delivers every public post on Bluesky in real time. This collector:
    1. Builds a keyword → shock_id index from configs/shocks.json
    2. For each incoming post, checks if text contains any keyword
    3. If matched, writes one JSONL record per matching shock file
    4. Optionally sends bios to the Pi bio server for inline classification

    Keyword matching is case-insensitive substring match. For 200–300 keywords
    this is fast enough at firehose volume (~1k events/sec); upgrade to
    pyahocorasick if profiling shows CPU saturation.
    """

    # Broad political pre-filter to reduce noise before per-shock matching.
    # Any post not containing at least one of these is discarded without further
    # processing. Keeps CPU load low on high-volume firehose.
    POLITICAL_PREFILTER: frozenset[str] = frozenset([
        "president", "election", "vote", "biden", "trump", "congress",
        "senate", "democrat", "republican", "gop", "liberal", "conservative",
        "scotus", "supreme court", "maga", "abortion", "immigration", "iran",
        "russia", "ukraine", "nato", "economy", "inflation", "protest",
        "police", "gun", "vaccine", "covid", "scandal", "indictment",
    ])

    def __init__(
        self,
        shocks_path: str | Path,
        output_root: str | Path,
        seed: int | None = None,
        pi_bio_server: str | None = None,
        bsky_handle: str | None = None,
        bsky_password: str | None = None,
        english_only: bool = True,
        use_political_prefilter: bool = True,
    ) -> None:
        self._shocks = load_shocks(shocks_path)
        self._keyword_index = build_keyword_index(self._shocks)
        self._output_root = Path(output_root)
        self._seed = seed
        self._pi_bio_server = pi_bio_server
        self._bsky_handle = bsky_handle
        self._bsky_password = bsky_password
        self._english_only = english_only
        self._use_prefilter = use_political_prefilter

        self._stop_event = Event()

        # Counters for periodic logging
        self._total_seen = 0
        self._total_matched = 0
        self._total_written = 0
        self._last_log_time = time.monotonic()

        # Bio batch buffer: (did, shock_id, post_id) tuples pending Pi classification
        self._bio_queue: list[tuple[str, str, str]] = []
        self._bio_batch_size = 50  # as per devplan Option J

        active_shocks = [s for s in self._shocks if s.get("active", True)]
        logger.info(
            "BlueskyFirehoseCollector ready: %d shocks loaded (%d active), "
            "%d keywords indexed, output=%s",
            len(self._shocks),
            len(active_shocks),
            len(self._keyword_index),
            self._output_root,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the firehose subscription. Blocks until stop() is called or Ctrl-C."""
        FirehoseSubscribeReposClient, parse_subscribe_repos_message, CAR = _import_atproto()

        # Stash references so the callback can use them without passing args
        self._parse = parse_subscribe_repos_message
        self._CAR = CAR

        client = FirehoseSubscribeReposClient()
        self._client = client

        # Reconnect loop with exponential backoff
        backoff = 1.0
        max_backoff = 60.0

        while not self._stop_event.is_set():
            try:
                logger.info("Connecting to Bluesky firehose...")
                client.start(self._on_message)
                # start() blocks until connection closes
                backoff = 1.0  # Reset on clean exit
            except KeyboardInterrupt:
                logger.info("Keyboard interrupt — stopping firehose")
                self._stop_event.set()
                break
            except Exception as exc:
                if self._stop_event.is_set():
                    break
                logger.warning(
                    "Firehose connection error: %s — reconnecting in %.0fs",
                    exc, backoff,
                )
                time.sleep(backoff)
                backoff = min(backoff * 2, max_backoff)

        logger.info(
            "Firehose stopped. seen=%d matched=%d written=%d",
            self._total_seen, self._total_matched, self._total_written,
        )

    def stop(self) -> None:
        """Signal the collector to stop after the current message."""
        self._stop_event.set()
        if hasattr(self, "_client"):
            try:
                self._client.stop()
            except Exception:
                pass

    # ── Internal message handler ──────────────────────────────────────────────

    def _on_message(self, message: Any) -> None:
        """Firehose message callback — called for every event on the network."""
        if self._stop_event.is_set():
            raise KeyboardInterrupt  # Causes client.start() to exit cleanly

        # Parse the raw firehose message into a typed commit
        try:
            commit = self._parse(message)
        except Exception:
            return  # Malformed messages are common; silently skip

        # We only care about repository commits (creates/updates)
        if not hasattr(commit, "ops") or not commit.ops:
            return

        # Parse CAR blocks containing the actual record data
        if not commit.blocks:
            return
        try:
            car = self._CAR.from_bytes(commit.blocks)
        except Exception:
            return

        for op in commit.ops:
            self._process_op(commit, op, car)

        # Periodic stats log every 60 seconds
        now = time.monotonic()
        if now - self._last_log_time >= 60.0:
            rate = self._total_seen / max(now - self._last_log_time, 1)
            logger.info(
                "Firehose stats: seen=%d (+%.0f/s) matched=%d written=%d",
                self._total_seen, rate,
                self._total_matched, self._total_written,
            )
            self._last_log_time = now
            self._total_seen = 0  # Reset window counter

    def _process_op(self, commit: Any, op: Any, car: Any) -> None:
        """Process a single repository operation from a commit."""
        # Only handle new post creations
        if op.action != "create":
            return
        if not op.path.startswith("app.bsky.feed.post/"):
            return

        self._total_seen += 1

        # Fetch the record from the CAR block store
        try:
            block = car.blocks.get(op.cid)
        except Exception:
            return
        if not block:
            return

        # Verify record type (should always be app.bsky.feed.post here)
        record_type = block.get("$type", "")
        if record_type and record_type != "app.bsky.feed.post":
            return

        text: str = block.get("text", "") or ""
        langs: list[str] = block.get("langs") or []
        created_at: str = block.get("createdAt", "") or ""

        # Language filter: English-only or all languages
        if self._english_only and not is_english(langs):
            return

        # Broad political pre-filter (optional, reduces CPU on high-volume days)
        if self._use_prefilter:
            text_lower = text.lower()
            if not any(kw in text_lower for kw in self.POLITICAL_PREFILTER):
                return

        # Per-shock keyword matching
        matched = match_shocks(text, self._keyword_index)
        if not matched:
            return

        self._total_matched += 1

        # Build the canonical post URI from repo DID + record key
        rkey = op.path.split("/", 1)[1] if "/" in op.path else op.path
        post_uri = f"at://{commit.repo}/app.bsky.feed.post/{rkey}"

        # Write one record per matching shock file
        for shock_id in matched:
            output_path = (
                self._output_root / "bluesky" / shock_id / "intel_mac_posts.jsonl"
            )
            payload = build_post_payload(
                post_id=post_uri,
                text=text,
                created_at=created_at,
                lang=extract_primary_lang(langs),
                source="live_stream",
                archive_id="bluesky",
                platform="bluesky",
                shock_id=shock_id,
                author_did=commit.repo,
                author_handle=None,        # resolved later in batch pass
                author_description=None,   # resolved by bio classifier
                inference_method=None,
            )
            try:
                append_post_record(output_path, payload, seed=self._seed)
                self._total_written += 1
            except OSError as exc:
                logger.error("Write failed for %s: %s", output_path, exc)

        # Queue DID for optional inline bio classification via Pi server
        if self._pi_bio_server and commit.repo:
            self._bio_queue.append((commit.repo, next(iter(matched)), post_uri))
            if len(self._bio_queue) >= self._bio_batch_size:
                self._flush_bio_batch()

    # ── Optional: inline batch bio classification ─────────────────────────────

    def _flush_bio_batch(self) -> None:
        """POST a batch of DIDs to the Pi bio server for classification.

        This is Option J from the devplan: batch 50 items per HTTP call to the
        Pi NPU. At 1,000 posts × 5ms serial = 5s overhead; batching drops to 0.1s.

        The response assigns bloc_weights to each DID. Posts are then re-written
        with updated bloc_weights fields. In practice, bio classification is done
        as a post-processing pass (Week 4); this is the real-time path for the
        continuous pipeline.
        """
        if not self._bio_queue or not self._pi_bio_server:
            self._bio_queue.clear()
            return
        try:
            import urllib.request
            import json as _json

            batch = self._bio_queue[: self._bio_batch_size]
            dids = [item[0] for item in batch]

            req_data = _json.dumps({"dids": dids}).encode()
            req = urllib.request.Request(
                f"{self._pi_bio_server}/classify_batch",
                data=req_data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                _ = _json.loads(resp.read())
                # bio results are stored server-side keyed by DID;
                # apply in post-processing pass (not in-line here to avoid
                # slowing the firehose callback loop)
        except Exception as exc:
            logger.debug("Bio batch classification failed: %s", exc)
        finally:
            self._bio_queue.clear()


# ── Entry point ───────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Bluesky firehose collector — Intel Mac daemon",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--shocks",
        default="configs/shocks.json",
        help="Path to shock event registry JSON",
    )
    p.add_argument(
        "--output",
        default="rawdata/social",
        help="Root directory for JSONL output (rawdata/social/bluesky/...)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Global pipeline seed to embed in each record (from configs/base.json)",
    )
    p.add_argument(
        "--pi-bio-server",
        default=None,
        help="Pi bio server URL (e.g. http://100.x.x.x:9000). Omit to skip inline bio.",
    )
    p.add_argument(
        "--all-langs",
        action="store_true",
        help="Collect posts in all languages (default: English only)",
    )
    p.add_argument(
        "--no-prefilter",
        action="store_true",
        help="Disable political keyword pre-filter (collects more noise)",
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

    bsky_handle = os.environ.get("BSKY_HANDLE")
    bsky_password = os.environ.get("BSKY_PASSWORD")
    pi_bio_server = args.pi_bio_server or os.environ.get("PI_TAILSCALE_IP") and (
        f"http://{os.environ['PI_TAILSCALE_IP']}:9000"
    )

    if not Path(args.shocks).exists():
        logger.error("shocks.json not found: %s", args.shocks)
        sys.exit(1)

    collector = BlueskyFirehoseCollector(
        shocks_path=args.shocks,
        output_root=args.output,
        seed=args.seed,
        pi_bio_server=pi_bio_server,
        bsky_handle=bsky_handle,
        bsky_password=bsky_password,
        english_only=not args.all_langs,
        use_political_prefilter=not args.no_prefilter,
    )

    # Graceful SIGTERM / SIGINT shutdown
    def _handle_signal(sig, frame):
        logger.info("Signal %s received — stopping collector", sig)
        collector.stop()

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    collector.start()


if __name__ == "__main__":
    main()
