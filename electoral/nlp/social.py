"""Platform-agnostic social media collector interface.

SocialCollector is the abstract base class. All concrete subclasses write
identically-formatted posts.jsonl so the scorer and merge_posts() are
blind to platform. Add new platforms by subclassing SocialCollector and
implementing collect().

Concrete subclasses (all run on Intel Mac):
  BlueSkyCollector      — 24/7 AT Protocol firehose (delegates to bluesky_firehose)
  ApifyCollector        — per-shock X/Twitter scraper (delegates to apify_x_scraper)
  TruthSocialCollector  — Mastodon-compatible public timeline, no auth required
  FacebookCollector     — Meta Content Library (if approved) or Graph API
                          public-page reaction-count fallback (auto-selected)

Hard fallback rules (from devplan):
  - Apify: if free-tier credit exhausted, retry with max_items=100; if still
    fails, skip shock and rely on Bluesky alone. Pipeline must not crash.
  - Facebook: if Content Library not approved by Week 4 Day 1, switch to
    reaction_fallback mode automatically. Both modes write identical schema.

Note: the collector scripts themselves live in electoral/nlp/collectors/ as
standalone daemons. These classes provide the pipeline-facing interface.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


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


# ── TruthSocialCollector ──────────────────────────────────────────────────────


class TruthSocialCollector(SocialCollector):
    """Truth Social Mastodon-compatible public search — no auth required.

    Uses /api/v2/search?type=statuses endpoint. Small user base; weight
    results accordingly in downstream merge. Platform-level proxy applied
    to all posts (Evangelical / conservative Protestant demographic).

    Free tier, no credentials needed. Rate-limited to ~40 results/search.
    """

    _API_BASE = "https://truthsocial.com"
    _SEARCH_PATH = "/api/v2/search"
    _PLATFORM_PROXY = "evangelical"

    def __init__(
        self,
        output_root: str | Path,
        seed: int | None = None,
        max_results: int = 40,
        timeout_secs: int = 30,
    ) -> None:
        self._output_root = Path(output_root)
        self._seed = seed
        self._max_results = max_results
        self._timeout = timeout_secs
        self._shock_id: str | None = None

    def collect(
        self,
        shock_id: str,
        keywords: list[str] | None = None,
        since: str | None = None,
        until: str | None = None,
        **kwargs: Any,
    ) -> int:
        """Search Truth Social for posts matching shock keywords.

        Returns number of records written. Returns 0 (not an error) if the
        platform is unreachable — Truth Social has intermittent availability.
        """
        import json as _json
        import urllib.error
        import urllib.parse
        import urllib.request

        from electoral.nlp.collectors.schema import append_post_record, build_post_payload, normalize_timestamp

        if keywords is None:
            if not (self._output_root.parent / "configs" / "shocks.json").exists():
                logger.warning("TruthSocialCollector: no keywords and shocks.json not found; skipping.")
                return 0
            shocks_path = self._output_root.parent / "configs" / "shocks.json"
            shocks = _json.load(open(shocks_path))
            shock = next((s for s in shocks if s["id"] == shock_id), None)
            keywords = shock.get("keywords", []) if shock else []

        if not keywords:
            logger.warning("TruthSocialCollector: no keywords for shock '%s'; skipping.", shock_id)
            return 0

        self._shock_id = shock_id
        query = " OR ".join(keywords[:5])  # Truth Social search supports limited boolean
        params = urllib.parse.urlencode({
            "q": query,
            "type": "statuses",
            "limit": min(self._max_results, 40),
        })
        url = f"{self._API_BASE}{self._SEARCH_PATH}?{params}"

        try:
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "electoral-equilibrium-research/1.0",
                    "Accept": "application/json",
                },
            )
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                data = _json.loads(resp.read().decode())
        except urllib.error.URLError as exc:
            logger.warning("TruthSocialCollector: network error (%s); skipping shock '%s'.", exc, shock_id)
            return 0
        except Exception as exc:
            logger.warning("TruthSocialCollector: unexpected error (%s); skipping.", exc)
            return 0

        statuses = data.get("statuses") or []
        written = 0
        for status in statuses:
            text = status.get("content", "") or ""
            # Strip HTML tags that Truth Social includes in content
            import re
            text = re.sub(r"<[^>]+>", " ", text).strip()
            if not text:
                continue

            status_id = str(status.get("id", ""))
            created_at = normalize_timestamp(status.get("created_at"))
            account = status.get("account") or {}
            author_handle = account.get("username")

            payload = build_post_payload(
                post_id=f"truthsocial:{status_id}",
                text=text,
                created_at=created_at,
                lang=status.get("language") or "en",
                source="live_scrape",
                archive_id="truthsocial",
                platform="truthsocial",
                shock_id=shock_id,
                author_did=f"truthsocial:{account.get('id', '')}",
                author_handle=author_handle,
                author_description=None,
                inference_method="platform_proxy",
            )
            try:
                append_post_record(self.output_path, payload, seed=self._seed)
                written += 1
            except OSError as exc:
                logger.error("TruthSocialCollector write failed: %s", exc)

        logger.info(
            "TruthSocialCollector: shock=%s written=%d/%d → %s",
            shock_id, written, len(statuses), self.output_path,
        )
        return written

    @property
    def output_path(self) -> Path:
        shock_id = self._shock_id or "unknown"
        return self._output_root / "truthsocial" / shock_id / "intel_mac_posts.jsonl"


# ── FacebookCollector ─────────────────────────────────────────────────────────


class FacebookCollector(SocialCollector):
    """Meta Content Library (if approved) or Graph API reaction-count fallback.

    Dual-mode design
    ----------------
    MODE_CONTENT_LIBRARY ("content_library"):
        Full text search via the Meta Content Library API. Requires institutional
        approval (apply at developers.facebook.com/docs/content-library). Uses
        meta_content_library_api_client Python package. Set FACEBOOK_CL_TOKEN
        in .env.

    MODE_REACTION_FALLBACK ("reaction_fallback"):
        Scrapes public Facebook page reaction counts (LIKE, LOVE, WOW, HAHA,
        ANGRY, SAD) from major news pages using the Graph API. Returns a
        valence-only signal per page per day — no post text. Posts are synthetic
        records with text="" and a computed valence_score field. Requires a
        standard Graph API user/app token (FACEBOOK_GRAPH_TOKEN in .env).

    Mode selection
    --------------
    mode="auto" (default): uses content_library if FACEBOOK_CL_TOKEN is set in
        the environment, else reaction_fallback.
    mode="content_library" or mode="reaction_fallback": forces the mode.

    Schema note
    -----------
    Both modes write to the same posts.jsonl schema. In reaction_fallback mode,
    text is empty and the reaction valence score is stored in a custom field
    "reaction_valence" on the payload. The scorer ignores empty-text records;
    the valence field is consumed by the elasticity regression as a separate
    signal path. Tag with platform="facebook_cl" or "facebook_reactions".
    """

    MODE_CONTENT_LIBRARY = "content_library"
    MODE_REACTION_FALLBACK = "reaction_fallback"

    # Public page IDs for major news/political sources.
    # Covers the full demographic spectrum (secular/liberal to Evangelical/conservative).
    # Stable numerical IDs — not slugs (slugs can be renamed).
    NEWS_PAGE_IDS: dict[str, str] = {
        "nyt": "5281959998",
        "fox_news": "15704546335",
        "cnn": "5550296508",
        "wapo": "5765369431",
        "breitbart": "95475020353",
        "npr": "10643211755",
        "cbn_news": "126895670687699",
        "daily_wire": "388913168093890",
        "univision": "215589471797",
        "ewtn": "51209571737",
    }

    # Valence weights per reaction type: positive → +1 side, negative → -1 side
    REACTION_VALENCE: dict[str, float] = {
        "LIKE": 1.0,
        "LOVE": 1.5,
        "WOW": 0.5,
        "HAHA": 0.5,
        "ANGRY": -1.5,
        "SAD": -1.0,
    }

    GRAPH_API_VERSION = "v19.0"
    GRAPH_API_BASE = "https://graph.facebook.com"

    def __init__(
        self,
        output_root: str | Path,
        mode: str = "auto",
        cl_token: str | None = None,
        graph_token: str | None = None,
        seed: int | None = None,
        page_ids: dict[str, str] | None = None,
        timeout_secs: int = 30,
        max_posts_per_page: int = 25,
    ) -> None:
        import os

        self._output_root = Path(output_root)
        self._seed = seed
        self._timeout = timeout_secs
        self._max_posts_per_page = max_posts_per_page
        self._page_ids = page_ids or self.NEWS_PAGE_IDS
        self._shock_id: str | None = None

        # Resolve tokens from args or environment
        self._cl_token = cl_token or os.environ.get("FACEBOOK_CL_TOKEN")
        self._graph_token = graph_token or os.environ.get("FACEBOOK_GRAPH_TOKEN")

        # Determine operational mode
        if mode == "auto":
            if self._cl_token:
                self._mode = self.MODE_CONTENT_LIBRARY
            else:
                self._mode = self.MODE_REACTION_FALLBACK
                logger.info(
                    "FacebookCollector: FACEBOOK_CL_TOKEN not set — "
                    "activating reaction_fallback mode (valence-only signal from public page reactions)."
                )
        elif mode in (self.MODE_CONTENT_LIBRARY, self.MODE_REACTION_FALLBACK):
            self._mode = mode
        else:
            raise ValueError(
                f"FacebookCollector mode must be 'auto', 'content_library', or 'reaction_fallback'; "
                f"got {mode!r}"
            )

        if self._mode == self.MODE_CONTENT_LIBRARY and not self._cl_token:
            raise ValueError(
                "FacebookCollector mode='content_library' requires FACEBOOK_CL_TOKEN. "
                "Set the env var or pass cl_token=. If approval is pending, use mode='reaction_fallback'."
            )
        if self._mode == self.MODE_REACTION_FALLBACK and not self._graph_token:
            raise ValueError(
                "FacebookCollector mode='reaction_fallback' requires FACEBOOK_GRAPH_TOKEN. "
                "Set the env var or pass graph_token=. "
                "A standard app token (App ID + App Secret) works for public page data."
            )

    @property
    def mode(self) -> str:
        return self._mode

    def collect(
        self,
        shock_id: str,
        keywords: list[str] | None = None,
        since: str | None = None,
        until: str | None = None,
        **kwargs: Any,
    ) -> int:
        """Collect Facebook signals for a shock event.

        Dispatches to the active mode. Returns number of records written.
        """
        self._shock_id = shock_id
        if self._mode == self.MODE_CONTENT_LIBRARY:
            return self._collect_content_library(shock_id, keywords, since, until)
        else:
            return self._collect_reaction_fallback(shock_id, since, until)

    # ── Content Library mode ──────────────────────────────────────────────────

    def _collect_content_library(
        self,
        shock_id: str,
        keywords: list[str] | None,
        since: str | None,
        until: str | None,
    ) -> int:
        """Full text search via Meta Content Library API."""
        try:
            from meta_content_library_api_client import ContentLibraryClient  # type: ignore[import]
        except ImportError as exc:
            raise RuntimeError(
                "meta_content_library_api_client is not installed.\n"
                "Install via the Meta Content Library SDK or switch to reaction_fallback mode."
            ) from exc

        from electoral.nlp.collectors.schema import append_post_record, build_post_payload, normalize_timestamp

        client = ContentLibraryClient(access_token=self._cl_token)
        query = " OR ".join(keywords or [])

        search_params: dict[str, Any] = {"q": query, "limit": 100}
        if since:
            search_params["since"] = since
        if until:
            search_params["until"] = until

        written = 0
        try:
            results = client.search_posts(**search_params)
            for post in results:
                text = post.get("message") or post.get("story") or ""
                if not text:
                    continue
                payload = build_post_payload(
                    post_id=f"facebook:{post['id']}",
                    text=text,
                    created_at=normalize_timestamp(post.get("created_time")),
                    lang="en",
                    source="content_library",
                    archive_id="facebook_cl",
                    platform="facebook_cl",
                    shock_id=shock_id,
                    author_did=f"facebook:{post.get('from', {}).get('id', '')}",
                    author_handle=post.get("from", {}).get("name"),
                    author_description=None,
                    inference_method="platform_proxy",
                )
                append_post_record(self.output_path, payload, seed=self._seed)
                written += 1
        except Exception as exc:
            logger.error("FacebookCollector content_library error: %s", exc)
            raise

        logger.info(
            "FacebookCollector (content_library): shock=%s written=%d → %s",
            shock_id, written, self.output_path,
        )
        return written

    # ── Reaction fallback mode ────────────────────────────────────────────────

    def _collect_reaction_fallback(
        self,
        shock_id: str,
        since: str | None,
        until: str | None,
    ) -> int:
        """Scrape public page reaction counts as a valence-only signal.

        For each news page, fetches posts in the shock date window and computes
        a valence score from reaction type counts. Writes one synthetic record
        per page-day with the valence score as metadata.

        Valence = (LIKE + LOVE + WOW + HAHA) − (ANGRY + SAD)  normalized to [-1, 1]
        based on total reactions. This is a directional aggregate, not per-user.
        """
        import json as _json
        import urllib.error
        import urllib.parse
        import urllib.request
        from datetime import datetime, timezone

        from electoral.nlp.collectors.schema import append_post_record, build_post_payload, normalize_timestamp

        written = 0

        # Build reaction fields string for the Graph API
        reaction_fields = ",".join(
            f"reactions.type({rt}).limit(0).summary(true).as(reactions_{rt.lower()})"
            for rt in self.REACTION_VALENCE
        )
        fields = f"id,created_time,{reaction_fields}"

        for page_name, page_id in self._page_ids.items():
            params: dict[str, str] = {
                "fields": fields,
                "limit": str(self._max_posts_per_page),
                "access_token": self._graph_token or "",
            }
            if since:
                params["since"] = since
            if until:
                params["until"] = until

            url = (
                f"{self.GRAPH_API_BASE}/{self.GRAPH_API_VERSION}/{page_id}/posts"
                f"?{urllib.parse.urlencode(params)}"
            )

            try:
                req = urllib.request.Request(
                    url,
                    headers={
                        "User-Agent": "electoral-equilibrium-research/1.0",
                        "Accept": "application/json",
                    },
                )
                with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                    data = _json.loads(resp.read().decode())
            except urllib.error.HTTPError as exc:
                logger.warning(
                    "FacebookCollector: HTTP %d for page '%s' (%s) — skipping.",
                    exc.code, page_name, page_id,
                )
                continue
            except urllib.error.URLError as exc:
                logger.warning(
                    "FacebookCollector: network error for page '%s': %s — skipping.",
                    page_name, exc,
                )
                continue

            posts = data.get("data") or []
            for post in posts:
                valence = self._compute_valence(post)
                created_at = normalize_timestamp(post.get("created_time"))

                # Synthetic record: text is empty, valence stored in a metadata field.
                # The scorer skips empty-text records; elasticity.py reads valence directly.
                payload = build_post_payload(
                    post_id=f"facebook:{post['id']}",
                    text="",
                    created_at=created_at,
                    lang="en",
                    source="reaction_fallback",
                    archive_id="facebook_reactions",
                    platform="facebook_reactions",
                    shock_id=shock_id,
                    author_did=f"facebook:page:{page_id}",
                    author_handle=page_name,
                    author_description=None,
                    inference_method="platform_proxy",
                )
                # Attach valence metadata (non-standard field; consumed by elasticity.py)
                payload["reaction_valence"] = valence
                payload["page_name"] = page_name

                try:
                    append_post_record(self.output_path, payload, seed=self._seed)
                    written += 1
                except OSError as exc:
                    logger.error("FacebookCollector write failed: %s", exc)

        logger.info(
            "FacebookCollector (reaction_fallback): shock=%s written=%d across %d pages → %s",
            shock_id, written, len(self._page_ids), self.output_path,
        )
        return written

    def _compute_valence(self, post: dict[str, Any]) -> float:
        """Compute normalized valence score from reaction counts in a Graph API post."""
        positive = 0.0
        negative = 0.0
        for reaction_type, weight in self.REACTION_VALENCE.items():
            key = f"reactions_{reaction_type.lower()}"
            count = (post.get(key) or {}).get("summary", {}).get("total_count", 0)
            if weight > 0:
                positive += weight * count
            else:
                negative += abs(weight) * count

        total = positive + negative
        if total == 0:
            return 0.0
        return (positive - negative) / total  # in [-1, 1]

    @property
    def output_path(self) -> Path:
        shock_id = self._shock_id or "unknown"
        subdir = "facebook_cl" if self._mode == self.MODE_CONTENT_LIBRARY else "facebook_reactions"
        return self._output_root / subdir / shock_id / "intel_mac_posts.jsonl"
