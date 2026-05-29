"""Platform-agnostic social media collector interface.

SocialCollector is the abstract base class. All concrete subclasses write
identically-formatted posts.jsonl so the scorer and merge_posts() are
blind to platform. Add new platforms by subclassing SocialCollector and
implementing collect().

Concrete subclasses (all run on Intel Mac):
  BlueSkyCollector    — 24/7 AT Protocol firehose (delegates to bluesky_firehose)
  ApifyCollector      — per-shock X/Twitter scraper (delegates to apify_x_scraper)

Note: the collector scripts themselves live in electoral/nlp/collectors/ as
standalone daemons. These classes provide the pipeline-facing interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class SocialCollector(ABC):
    """Abstract base for all social media collectors.

    Subclasses must write identically-formatted posts.jsonl records via
    the canonical envelope/schema defined in collectors/schema.py.

    Write-once ownership enforced by convention: only the Intel Mac
    runs collector scripts. Other machines read via Syncthing.
    """

    @abstractmethod
    def collect(self, shock_id: str, **kwargs: Any) -> int:
        """Collect posts for a shock event.

        Args:
            shock_id: Shock identifier from configs/shocks.json
            **kwargs: Platform-specific options

        Returns:
            Number of records written to JSONL
        """
        ...

    @property
    @abstractmethod
    def output_path(self) -> Path:
        """Path to the output JSONL file for the current shock."""
        ...


class BlueSkyCollector(SocialCollector):
    """Bluesky AT Protocol firehose — primary live collection path.

    Runs 24/7. No rate limits. Covers secular, diverse, left-leaning
    demographics (note: Bluesky skews far-left post-Twitter migration;
    NOT a broadly representative demographic sample — see DECISIONS.md).

    Wraps BlueskyFirehoseCollector from collectors/bluesky_firehose.py.
    """

    def __init__(
        self,
        shocks_path: str | Path,
        output_root: str | Path,
        seed: int | None = None,
        pi_bio_server: str | None = None,
    ) -> None:
        from electoral.nlp.collectors.bluesky_firehose import BlueskyFirehoseCollector

        self._impl = BlueskyFirehoseCollector(
            shocks_path=shocks_path,
            output_root=output_root,
            seed=seed,
            pi_bio_server=pi_bio_server,
        )
        self._shock_id: str | None = None
        self._output_root = Path(output_root)

    def collect(self, shock_id: str, **kwargs: Any) -> int:
        """Start the firehose for a specific shock (returns only after stop()).

        For the 24/7 continuous path, call start() on the underlying
        BlueskyFirehoseCollector directly and let it run as a daemon.
        This method is for one-shot collection in the Prefect DAG.
        """
        self._shock_id = shock_id
        self._impl.start()  # blocks until stopped
        return self._impl._total_written

    @property
    def output_path(self) -> Path:
        shock_id = self._shock_id or "unknown"
        return self._output_root / "bluesky" / shock_id / "intel_mac_posts.jsonl"


class ApifyCollector(SocialCollector):
    """Apify X (Twitter) scraper — secondary live collection path.

    Runs per-shock event (not continuously). Free tier: 500 results/run.
    Covers Evangelical, Catholic, and high-education urban demographics.

    Wraps ApifyXScraper from collectors/apify_x_scraper.py.
    """

    def __init__(
        self,
        output_root: str | Path,
        seed: int | None = None,
        apify_token: str | None = None,
        actor_id: str | None = None,
        max_items: int = 500,
    ) -> None:
        self._output_root = Path(output_root)
        self._seed = seed
        self._apify_token = apify_token
        self._actor_id = actor_id
        self._max_items = max_items
        self._shock_id: str | None = None

    def collect(
        self,
        shock_id: str,
        keywords: list[str] | None = None,
        since: str | None = None,
        until: str | None = None,
        **kwargs: Any,
    ) -> int:
        """Run the Apify X scraper for a shock event.

        Args:
            shock_id: Shock identifier from configs/shocks.json
            keywords: Override keywords. If None, loaded from shocks.json.
            since: Start date (YYYY-MM-DD)
            until: End date (YYYY-MM-DD)
        """
        from electoral.nlp.collectors.apify_x_scraper import ApifyXScraper

        if keywords is None:
            import json as _json

            shocks = _json.load(open("configs/shocks.json"))
            shock = next((s for s in shocks if s["id"] == shock_id), None)
            if shock is None:
                raise ValueError(f"Shock '{shock_id}' not found in configs/shocks.json")
            keywords = shock.get("keywords", [])

        self._shock_id = shock_id
        scraper = ApifyXScraper(
            shock_id=shock_id,
            keywords=keywords,
            output_root=self._output_root,
            seed=self._seed,
            apify_token=self._apify_token,
            actor_id=self._actor_id or "apidojo/tweet-scraper",
            max_items=self._max_items,
            since=since,
            until=until,
        )
        return scraper.run()

    @property
    def output_path(self) -> Path:
        shock_id = self._shock_id or "unknown"
        return self._output_root / "apify" / shock_id / "intel_mac_posts.jsonl"
