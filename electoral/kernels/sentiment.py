"""sentiment kernel: orchestrates bio classification → RoBERTa scoring → aggregation.

Wires together:
  1. BioClassifier (from bio_classifier.py) — demographic inference per author
  2. RoBERTaScorer (from scorer.py) — sentiment scoring per text
  3. HistoricalArchiveLoader (from archive.py) — pre-existing archive data
  4. ScrapedNewsLoader (from news_loader.py) — live-scraped news articles
  5. score_news_for_shocks() — SentimentData from news corpus
  6. score_social_for_shock() — SocialMediaSentimentData per shock

Entry point: run_sentiment_pipeline()

Output artifacts:
  - SentimentData (news-derived, all blocs × all shocks)
  - list[SocialMediaSentimentData] (social-derived, per shock)

Both are written to output_dir if provided.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from electoral.artifacts import SentimentData, SocialMediaSentimentData
from electoral.config import PipelineConfig
from electoral.core.io import write_artifact
from electoral.nlp.archive import HistoricalArchiveLoader
from electoral.nlp.bio_classifier import BioClassifier
from electoral.nlp.news_loader import ScrapedNewsLoader
from electoral.nlp.scorer import (
    DEFAULT_MODEL,
    RoBERTaScorer,
    score_news_for_shocks,
    score_social_for_shock,
)

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_SHOCKS_PATH = _REPO_ROOT / "configs" / "shocks.json"


def _load_social_posts(
    posts_root: Path,
    shock_ids: list[str],
) -> dict[str, dict[str, list[dict]]]:
    """Load social posts for each shock from rawdata/social/{platform}/{shock_id}/.

    Returns nested dict: shock_id → platform → list[post payload dicts].
    Missing directories are silently skipped.
    """
    import json as _json

    result: dict[str, dict[str, list[dict]]] = {sid: {} for sid in shock_ids}

    if not posts_root.exists():
        logger.debug("Social posts root not found: %s", posts_root)
        return result

    for platform_dir in sorted(posts_root.iterdir()):
        if not platform_dir.is_dir():
            continue
        platform = platform_dir.name

        for shock_id in shock_ids:
            shock_dir = platform_dir / shock_id
            if not shock_dir.is_dir():
                continue

            posts: list[dict] = []
            for jsonl_path in sorted(shock_dir.glob("*.jsonl")):
                with open(jsonl_path, encoding="utf-8") as f:
                    for lineno, line in enumerate(f, 1):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            record = _json.loads(line)
                        except _json.JSONDecodeError as exc:
                            logger.warning(
                                "%s line %d: JSON error (%s)", jsonl_path, lineno, exc
                            )
                            continue
                        # Unwrap envelope if present
                        payload = record.get("payload", record)
                        if payload.get("text"):
                            posts.append(payload)

            if posts:
                result[shock_id][platform] = posts
                logger.debug(
                    "Loaded %d posts: shock='%s' platform='%s'",
                    len(posts), shock_id, platform,
                )

    return result


def run_sentiment_pipeline(
    config_path: str | Path,
    posts_root: str | Path | None = None,
    archive_root: str | Path | None = None,
    news_root: str | Path | None = None,
    shock_ids: list[str] | None = None,
    model_name: str = DEFAULT_MODEL,
    window_hours: int = 72,
    output_dir: str | Path | None = None,
    shocks_path: str | Path | None = None,
) -> tuple[SentimentData, list[SocialMediaSentimentData]]:
    """Run the full sentiment pipeline for a set of shock events.

    Steps:
      1. Load BioClassifier from config lexicons
      2. Load RoBERTaScorer (requires transformers+torch)
      3. Load archive news + scraped news → score → SentimentData
      4. Load social posts per shock → score → list[SocialMediaSentimentData]
      5. Optionally write artifacts to output_dir

    Args:
        config_path: Path to configs/base.json.
        posts_root: rawdata/social/ root (default: {repo}/rawdata/social).
        archive_root: data/archives/ root (default: {repo}/data/archives).
        news_root: rawdata/news/ root (default: {repo}/rawdata/news).
        shock_ids: List of shock event IDs to score. If None, loads all from shocks.json.
        model_name: HuggingFace model name for RoBERTa.
        window_hours: Social collection window stored in SocialMediaSentimentData.
        output_dir: If provided, write SentimentData and SocialMediaSentimentData JSON.
        shocks_path: Path to configs/shocks.json. Defaults to standard location.

    Returns:
        (SentimentData, list[SocialMediaSentimentData])
    """
    config_path = Path(config_path)
    shocks_path = Path(shocks_path) if shocks_path else _DEFAULT_SHOCKS_PATH

    # Resolve directory defaults relative to repo root
    posts_root = Path(posts_root) if posts_root else _REPO_ROOT / "rawdata" / "social"
    archive_root = Path(archive_root) if archive_root else _REPO_ROOT / "data" / "archives"
    news_root = Path(news_root) if news_root else _REPO_ROOT / "rawdata" / "news"

    # Load shock registry
    if shock_ids is None:
        if shocks_path.exists():
            with open(shocks_path, encoding="utf-8") as f:
                shocks_cfg = json.load(f)
            shock_ids = [s["id"] for s in shocks_cfg]
        else:
            logger.warning("shocks.json not found; using empty shock list.")
            shock_ids = []
    else:
        shocks_cfg = []
        if shocks_path.exists():
            with open(shocks_path, encoding="utf-8") as f:
                shocks_cfg = json.load(f)

    logger.info(
        "run_sentiment_pipeline: %d shocks, model='%s'", len(shock_ids), model_name
    )

    # ── Step 1: Bio classifier ────────────────────────────────────────────────
    bio_clf = BioClassifier.from_config(
        config_path=config_path,
        pi_server_url=None,  # loaded from config inside from_config()
    )

    # ── Step 2: RoBERTa scorer ────────────────────────────────────────────────
    scorer = RoBERTaScorer(model_name=model_name)

    # ── Step 3: News corpus → SentimentData ───────────────────────────────────
    all_articles: list[dict] = []

    archive_loader = HistoricalArchiveLoader(
        archive_root=archive_root,
        shocks_path=shocks_path,
    )
    archive_articles = archive_loader.load_news(source="all")
    all_articles.extend(archive_articles)
    logger.info("Archive news: %d articles", len(archive_articles))

    scraped_loader = ScrapedNewsLoader(
        news_root=news_root,
        shocks_path=shocks_path,
    )
    scraped_articles = scraped_loader.load_all()
    all_articles.extend(scraped_articles)
    logger.info("Scraped news: %d articles", len(scraped_articles))

    sentiment_data = score_news_for_shocks(
        articles=all_articles,
        scorer=scorer,
        shocks=shock_ids,
        model_name=model_name,
    )

    # ── Step 4: Social posts → SocialMediaSentimentData ──────────────────────
    social_posts = _load_social_posts(posts_root, shock_ids)
    social_results: list[SocialMediaSentimentData] = []

    for shock_id in shock_ids:
        posts_by_platform = social_posts.get(shock_id, {})
        if not any(posts_by_platform.values()):
            logger.debug("No social posts for shock '%s'; skipping.", shock_id)
            continue

        social_sentiment = score_social_for_shock(
            posts_by_platform=posts_by_platform,
            scorer=scorer,
            bio_classifier=bio_clf,
            shock_id=shock_id,
            window_hours=window_hours,
        )
        social_results.append(social_sentiment)

    logger.info(
        "run_sentiment_pipeline complete: "
        "SentimentData(shocks=%d blocs=%d) social=%d",
        len(sentiment_data.shocks),
        len(sentiment_data.scores),
        len(social_results),
    )

    # ── Step 5: Write artifacts ───────────────────────────────────────────────
    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        _write_sentiment_artifact(sentiment_data, out / "sentiment_news.json")
        for sms in social_results:
            fname = f"sentiment_social_{sms.shock}.json"
            _write_social_artifact(sms, out / fname)

    return sentiment_data, social_results


def _write_sentiment_artifact(data: SentimentData, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data.to_dict(), f, indent=2, ensure_ascii=False)
    logger.info("Wrote SentimentData → %s", path)


def _write_social_artifact(data: SocialMediaSentimentData, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data.to_dict(), f, indent=2, ensure_ascii=False)
    logger.info("Wrote SocialMediaSentimentData[%s] → %s", data.shock, path)


def merge_posts(
    shock_id: str,
    posts_root: str | Path | None = None,
    merged_root: str | Path | None = None,
) -> int:
    """Concatenate all per-platform JSONL files for a shock into rawdata/merged/.

    Scans rawdata/social/{platform}/{shock_id}/*.jsonl across every platform
    directory, plus rawdata/social/archive/{shock_id}/*.jsonl. Writes a single
    rawdata/merged/{shock_id}/posts.jsonl. The scorer reads only from merged/
    and is blind to how many platforms or files contributed.

    Running twice overwrites deterministically — output is identical.

    Returns total number of post records written.
    """
    import json as _json

    posts_root = Path(posts_root) if posts_root else _REPO_ROOT / "rawdata" / "social"
    merged_root = Path(merged_root) if merged_root else _REPO_ROOT / "rawdata" / "merged"

    out_path = merged_root / shock_id / "posts.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    platform_counts: dict[str, int] = {}
    total = 0

    with open(out_path, "w", encoding="utf-8") as out_f:
        # Walk every platform directory under rawdata/social/
        if posts_root.exists():
            for platform_dir in sorted(posts_root.iterdir()):
                if not platform_dir.is_dir():
                    continue
                shock_dir = platform_dir / shock_id
                if not shock_dir.is_dir():
                    continue
                platform = platform_dir.name
                count = 0
                for jsonl_path in sorted(shock_dir.glob("*.jsonl")):
                    with open(jsonl_path, encoding="utf-8", errors="replace") as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                record = _json.loads(line)
                            except _json.JSONDecodeError:
                                continue
                            # Unwrap envelope; keep payload only
                            payload = record.get("payload", record)
                            if not payload.get("text"):
                                continue
                            out_f.write(_json.dumps(payload, ensure_ascii=False))
                            out_f.write("\n")
                            count += 1
                            total += 1
                if count:
                    platform_counts[platform] = platform_counts.get(platform, 0) + count

    logger.info(
        "merge_posts(%s): %d total posts → %s  breakdown=%s",
        shock_id, total, out_path,
        {k: v for k, v in sorted(platform_counts.items())},
    )
    return total


def merge_all_posts(
    shock_ids: list[str] | None = None,
    posts_root: str | Path | None = None,
    merged_root: str | Path | None = None,
    shocks_path: str | Path | None = None,
) -> dict[str, int]:
    """Run merge_posts() for every shock event. Returns {shock_id: post_count}."""
    shocks_path = Path(shocks_path) if shocks_path else _DEFAULT_SHOCKS_PATH
    if shock_ids is None:
        if shocks_path.exists():
            with open(shocks_path, encoding="utf-8") as f:
                shock_ids = [s["id"] for s in json.load(f)]
        else:
            shock_ids = []

    results: dict[str, int] = {}
    for shock_id in shock_ids:
        results[shock_id] = merge_posts(shock_id, posts_root=posts_root, merged_root=merged_root)
    logger.info(
        "merge_all_posts: %d shocks merged, %d total posts",
        len(results), sum(results.values()),
    )
    return results
