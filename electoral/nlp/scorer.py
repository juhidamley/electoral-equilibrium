"""scorer: RoBERTa sentiment scoring with bloc-weighted aggregation.

Model: cardiffnlp/twitter-roberta-base-sentiment (3-class)
  LABEL_0 = negative, LABEL_1 = neutral, LABEL_2 = positive
  score = P(positive) - P(negative) ∈ [-1, 1]

Bloc-weighted aggregation:
  Each post p contributes score s_p to bloc b with weight w_p[b]:
    score[bloc] = Σ(s_p × w_p[bloc]) / Σ(w_p[bloc])
  Blocs with zero total weight get score 0.0 (neutral).

Language-prior posts (inference_method="language_prior") are scored but
their weights are excluded from estimation per CLAUDE.md.

News articles use outlet-level demographic proxies stored in author_description
as JSON {"outlet": ..., "proxy": {"race": {...}, "religion": {...}, ...}}.
These proxies are converted to synthetic BioClassification objects by the scorer.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from electoral.artifacts import SentimentData, SocialMediaSentimentData
from electoral.core.types import CANONICAL_GENDERS, CANONICAL_RACES, CANONICAL_RELIGIONS
from electoral.nlp.bio_classifier import BioClassification, BioClassifier

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
_MAX_WORDS = 128  # Truncation limit before RoBERTa tokenization

# All 15 canonical bloc IDs across all three strata
_ALL_BLOCS: list[str] = list(CANONICAL_RACES) + list(CANONICAL_RELIGIONS) + list(CANONICAL_GENDERS)


def _truncate_text(text: str, max_words: int = _MAX_WORDS) -> str:
    """Truncate text to max_words words to stay within RoBERTa token budget."""
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words])


def _proxy_to_bio(proxy: dict[str, dict[str, float]]) -> BioClassification:
    """Convert an outlet/platform proxy dict to a BioClassification."""
    return BioClassification(
        inference_method="platform_proxy",
        race_weights=dict(proxy.get("race", {})),
        religion_weights=dict(proxy.get("religion", {})),
        gender_weights=dict(proxy.get("gender", {})),
    )


def _extract_outlet_proxy(post: dict) -> BioClassification | None:
    """Extract outlet-level demographic proxy from author_description JSON.

    News loaders store {"outlet": ..., "proxy": {...}} in author_description.
    Returns None if the field is absent or malformed.
    """
    desc = post.get("author_description")
    if not desc:
        return None
    try:
        data = json.loads(desc)
        proxy = data.get("proxy") or {}
        if proxy:
            return _proxy_to_bio(proxy)
    except (json.JSONDecodeError, AttributeError):
        pass
    return None


def _zero_scores() -> dict[str, float]:
    return {bloc: 0.0 for bloc in _ALL_BLOCS}


def _aggregate_scores(
    texts_scores: list[float],
    bio_results: list[BioClassification],
    exclude_language_prior: bool = True,
) -> dict[str, float]:
    """Weighted aggregation of RoBERTa scores across all 15 canonical blocs.

    Returns {bloc_id: score} where score ∈ [-1, 1].
    Blocs with no contributing posts get 0.0 (neutral).
    """
    numerator: dict[str, float] = {b: 0.0 for b in _ALL_BLOCS}
    denominator: dict[str, float] = {b: 0.0 for b in _ALL_BLOCS}

    for score, bio in zip(texts_scores, bio_results):
        if bio.inference_method is None:
            continue
        if exclude_language_prior and bio.inference_method == "language_prior":
            continue

        for bloc, w in bio.race_weights.items():
            if bloc in numerator:
                numerator[bloc] += score * w
                denominator[bloc] += w
        for bloc, w in bio.religion_weights.items():
            if bloc in numerator:
                numerator[bloc] += score * w
                denominator[bloc] += w
        for bloc, w in bio.gender_weights.items():
            if bloc in numerator:
                numerator[bloc] += score * w
                denominator[bloc] += w

    return {
        bloc: (numerator[bloc] / denominator[bloc] if denominator[bloc] > 0 else 0.0)
        for bloc in _ALL_BLOCS
    }


class RoBERTaScorer:
    """RoBERTa-based sentiment scorer with lazy model loading.

    Raises ImportError (with install instructions) when transformers/torch
    are not installed — only at instantiation, not import time.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str | None = None,
        batch_size: int = 32,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self._pipeline = self._load_pipeline(model_name, device)

    @staticmethod
    def _load_pipeline(model_name: str, device: str | None):
        try:
            from transformers import pipeline as hf_pipeline  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "transformers and torch are required for the RoBERTa scorer.\n"
                "Install with: pip install transformers torch"
            ) from exc

        # Auto-select device: MPS on Apple Silicon, CUDA if available, else CPU
        if device is None:
            try:
                import torch
                if torch.backends.mps.is_available():
                    device = "mps"
                elif torch.cuda.is_available():
                    device = "cuda"
                else:
                    device = "cpu"
            except Exception:
                device = "cpu"

        logger.info("Loading RoBERTa model '%s' on device '%s'...", model_name, device)
        pipe = hf_pipeline(
            "text-classification",
            model=model_name,
            tokenizer=model_name,
            device=device,
            return_all_scores=True,
            truncation=True,
            max_length=512,
        )
        logger.info("RoBERTa model ready.")
        return pipe

    def score_texts(self, texts: list[str]) -> list[float]:
        """Score a list of texts. Returns per-text score ∈ [-1, 1].

        Uses batched inference. Empty texts return 0.0 (neutral).
        """
        if not texts:
            return []

        # Split into non-empty vs empty to avoid model errors on empty strings
        indices_nonempty: list[int] = []
        truncated: list[str] = []
        for i, t in enumerate(texts):
            t_strip = t.strip()
            if t_strip:
                indices_nonempty.append(i)
                truncated.append(_truncate_text(t_strip))

        results = [0.0] * len(texts)
        if not indices_nonempty:
            return results

        # Batch inference
        all_scores: list[float] = []
        for batch_start in range(0, len(truncated), self.batch_size):
            batch = truncated[batch_start : batch_start + self.batch_size]
            raw = self._pipeline(batch)
            for item in raw:
                # item is a list of {"label": ..., "score": ...} dicts
                score_map = {d["label"]: d["score"] for d in item}
                # cardiffnlp labels: LABEL_0=neg, LABEL_1=neu, LABEL_2=pos
                pos = score_map.get("LABEL_2", score_map.get("positive", 0.0))
                neg = score_map.get("LABEL_0", score_map.get("negative", 0.0))
                all_scores.append(float(pos - neg))

        for idx, score in zip(indices_nonempty, all_scores):
            results[idx] = score

        return results

    def score_posts_for_shock(
        self,
        posts: list[dict],
        bio_classifier: BioClassifier,
        shock_id: str,
        exclude_language_prior: bool = True,
    ) -> dict[str, float]:
        """Score a list of posts for a single shock event.

        Args:
            posts: Canonical post payload dicts (text, lang, author_description, ...).
            bio_classifier: Used to assign demographic weights.
            shock_id: Shock identifier (used only for logging).
            exclude_language_prior: Whether to exclude language-prior posts from weights.

        Returns:
            Dict mapping all 15 canonical bloc IDs → score ∈ [-1, 1].
        """
        if not posts:
            logger.debug("score_posts_for_shock: no posts for shock '%s'", shock_id)
            return _zero_scores()

        texts = [p.get("text", "") for p in posts]
        scores = self.score_texts(texts)

        # Assign bio weights: prefer outlet proxy for news, bio classifier for social
        bio_results: list[BioClassification] = []
        for post in posts:
            outlet_proxy = _extract_outlet_proxy(post)
            if outlet_proxy is not None:
                bio_results.append(outlet_proxy)
            else:
                bio_results.append(
                    bio_classifier.classify(
                        bio=post.get("author_description"),
                        lang=post.get("lang", ""),
                        inference_method=post.get("inference_method"),
                    )
                )

        aggregated = _aggregate_scores(scores, bio_results, exclude_language_prior)
        logger.info(
            "scorer: shock='%s' n_posts=%d scores=%s",
            shock_id, len(posts),
            {k: f"{v:+.3f}" for k, v in list(aggregated.items())[:5]},
        )
        return aggregated


def score_news_for_shocks(
    articles: list[dict],
    scorer: RoBERTaScorer,
    shocks: list[str],
    model_name: str | None = None,
) -> SentimentData:
    """Score news articles and aggregate into SentimentData.

    Articles that lack a shock_id are skipped. Articles with outlet proxies
    use those proxies for bloc attribution rather than bio classification.

    Args:
        articles: Canonical payload dicts from ScrapedNewsLoader or HistoricalArchiveLoader.
        scorer: Initialized RoBERTaScorer.
        shocks: Shock IDs to include in the output (determines SentimentData.shocks).
        model_name: Override model name in the artifact (defaults to scorer.model_name).

    Returns:
        SentimentData with scores for all 15 blocs across all requested shocks.
    """
    # Group articles by shock_id
    by_shock: dict[str, list[dict]] = {s: [] for s in shocks}
    for article in articles:
        sid = article.get("shock_id")
        if sid and sid in by_shock:
            by_shock[sid].append(article)

    # Dummy bio classifier — news articles use outlet proxies, so classify() won't be called
    # for any article that has an outlet proxy. For articles without proxies, use keyword fallback.
    class _NoOpBioClassifier:
        def classify(self, bio, lang="", inference_method=None) -> BioClassification:
            return BioClassification(
                inference_method=inference_method,
                race_weights={},
                religion_weights={},
                gender_weights={},
            )

    dummy_clf = _NoOpBioClassifier()  # type: ignore[assignment]

    scores_by_bloc_shock: dict[str, dict[str, float]] = {
        bloc: {} for bloc in _ALL_BLOCS
    }

    for shock_id in shocks:
        articles_for_shock = by_shock[shock_id]
        texts = [a.get("text", "") for a in articles_for_shock]
        text_scores = scorer.score_texts(texts)

        bio_results: list[BioClassification] = []
        for article in articles_for_shock:
            proxy = _extract_outlet_proxy(article)
            if proxy is not None:
                bio_results.append(proxy)
            else:
                bio_results.append(
                    BioClassification(
                        inference_method=None,
                        race_weights={},
                        religion_weights={},
                        gender_weights={},
                    )
                )

        agg = _aggregate_scores(text_scores, bio_results, exclude_language_prior=True)
        for bloc, score in agg.items():
            scores_by_bloc_shock[bloc][shock_id] = score

        logger.info(
            "score_news_for_shocks: shock='%s' n_articles=%d",
            shock_id, len(articles_for_shock),
        )

    return SentimentData(
        model=model_name or scorer.model_name,
        shocks=list(shocks),
        scores=scores_by_bloc_shock,
    )


def score_social_for_shock(
    posts_by_platform: dict[str, list[dict]],
    scorer: RoBERTaScorer,
    bio_classifier: BioClassifier,
    shock_id: str,
    window_hours: int = 72,
) -> SocialMediaSentimentData:
    """Score social media posts per-platform and produce SocialMediaSentimentData.

    For each platform, aggregates sentiment using bio classifier weights.
    Platform-proxy posts (TruthSocial, Facebook reactions) use their upstream
    proxy directly; subreddit-proxy posts use subreddit demographics.

    Args:
        posts_by_platform: Dict of platform_name → list[canonical post dicts].
        scorer: Initialized RoBERTaScorer.
        bio_classifier: BioClassifier for individual post demographic attribution.
        shock_id: Shock event identifier.
        window_hours: Collection window in hours (stored in artifact).

    Returns:
        SocialMediaSentimentData with per-platform bloc-weighted scores.
    """
    platforms = sorted(posts_by_platform.keys())
    scores: dict[str, dict[str, float]] = {}
    n_posts: dict[str, int] = {}

    for platform in platforms:
        posts = posts_by_platform[platform]
        n_posts[platform] = len(posts)

        if not posts:
            scores[platform] = {}
            continue

        texts = [p.get("text", "") for p in posts]
        text_scores = scorer.score_texts(texts)

        bio_results: list[BioClassification] = []
        for post in posts:
            proxy = _extract_outlet_proxy(post)
            if proxy is not None:
                bio_results.append(proxy)
            else:
                bio_results.append(
                    bio_classifier.classify(
                        bio=post.get("author_description"),
                        lang=post.get("lang", ""),
                        inference_method=post.get("inference_method"),
                    )
                )

        agg = _aggregate_scores(text_scores, bio_results, exclude_language_prior=True)
        # For SocialMediaSentimentData, scores[platform] is a bloc_proxy → score map.
        # We include all blocs that have non-zero contributions.
        scores[platform] = {k: v for k, v in agg.items() if v != 0.0} or {
            b: 0.0 for b in _ALL_BLOCS
        }

        logger.info(
            "score_social_for_shock: shock='%s' platform='%s' n_posts=%d",
            shock_id, platform, len(posts),
        )

    return SocialMediaSentimentData(
        shock=shock_id,
        platforms=platforms,
        window_hours=window_hours,
        scores=scores,
        n_posts=n_posts,
        lagged_delta=None,
    )
