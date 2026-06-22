"""scorer: RoBERTa sentiment scoring with bloc-weighted aggregation.

═══════════════════════════════════════════════════════════════════════════════
BEGINNER ORIENTATION: what "RoBERTa sentiment scoring" means
═══════════════════════════════════════════════════════════════════════════════
SENTIMENT SCORING = reading a piece of text and rating how positive or negative
it is. RoBERTa is a pre-trained language model good at this; the specific model
here (cardiffnlp/twitter-roberta-base-sentiment) outputs three probabilities —
P(negative), P(neutral), P(positive) — for any text. We collapse those into one
number, score = P(positive) − P(negative), which lands in [−1, +1] (−1 = very
negative, +1 = very positive).

But one post's sentiment isn't what we want — we want sentiment PER DEMOGRAPHIC
BLOC. Each post comes with a guess (from bio_classifier.py) of which blocs its
author likely belongs to, expressed as weights. So we compute a WEIGHTED AVERAGE:
a post by a probably-Latino author counts mostly toward the Latino bloc's score.
That per-bloc sentiment becomes a feature for the LLM fine-tuning stage.

(Two rules from CLAUDE.md are enforced below: "language_prior" posts are scored
but excluded from estimation; news articles use outlet-level demographic proxies
instead of a per-author bio.)

─── original technical summary ───

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
from pathlib import Path

import numpy as np

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


class EmbeddingCache:
    """Parquet-backed cache mapping post_id → 3-class probability vector.

    Each cached entry stores [P(neg), P(neu), P(pos)] as float32, allowing
    any downstream score formula (e.g. pos-neg, pos only) to be recomputed
    without re-running the model.

    All entries live in a single file:
        data/embeddings/cache.parquet
    with columns (post_id STRING, prob_neg FLOAT32, prob_neu FLOAT32, prob_pos FLOAT32).

    The file is rewritten only on flush() — call it after each batch or at
    the end of a scoring run. Auto-flush triggers every FLUSH_EVERY new entries.
    """

    FLUSH_EVERY = 500

    def __init__(self, cache_dir: str | Path) -> None:
        self._path = Path(cache_dir) / "cache.parquet"
        self._store: dict[str, np.ndarray] = {}  # post_id → float32[3]
        self._dirty = 0
        self._hits = 0
        self._misses = 0
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            import pyarrow.parquet as pq

            table = pq.read_table(
                self._path, columns=["post_id", "prob_neg", "prob_neu", "prob_pos"]
            )
            ids = table.column("post_id").to_pylist()
            negs = table.column("prob_neg").to_pylist()
            neus = table.column("prob_neu").to_pylist()
            poss = table.column("prob_pos").to_pylist()
            for pid, neg, neu, pos in zip(ids, negs, neus, poss):
                self._store[pid] = np.array([neg, neu, pos], dtype=np.float32)
            logger.info("EmbeddingCache: loaded %d entries from %s", len(self._store), self._path)
        except Exception as exc:
            logger.warning(
                "EmbeddingCache: failed to load %s (%s) — starting empty", self._path, exc
            )

    def get(self, post_id: str) -> np.ndarray | None:
        vec = self._store.get(post_id)
        if vec is not None:
            self._hits += 1
        else:
            self._misses += 1
        return vec

    def put(self, post_id: str, probs: np.ndarray) -> None:
        self._store[post_id] = probs.astype(np.float32)
        self._dirty += 1
        if self._dirty >= self.FLUSH_EVERY:
            self.flush()

    def flush(self) -> None:
        if self._dirty == 0 or not self._store:
            return
        import pyarrow as pa
        import pyarrow.parquet as pq

        self._path.parent.mkdir(parents=True, exist_ok=True)
        ids = list(self._store.keys())
        vecs = np.stack(list(self._store.values()))
        table = pa.table(
            {
                "post_id": pa.array(ids, type=pa.string()),
                "prob_neg": pa.array(vecs[:, 0].tolist(), type=pa.float32()),
                "prob_neu": pa.array(vecs[:, 1].tolist(), type=pa.float32()),
                "prob_pos": pa.array(vecs[:, 2].tolist(), type=pa.float32()),
            }
        )
        pq.write_table(table, self._path, compression="snappy")
        logger.debug("EmbeddingCache: flushed %d entries → %s", len(ids), self._path)
        self._dirty = 0

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total else 0.0

    @property
    def size(self) -> int:
        return len(self._store)

    def log_stats(self) -> None:
        logger.info(
            "EmbeddingCache stats: size=%d hits=%d misses=%d hit_rate=%.1f%%",
            self.size,
            self._hits,
            self._misses,
            100 * self.hit_rate,
        )


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
        cache_dir: str | Path | None = None,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self._pipeline = self._load_pipeline(model_name, device)
        if cache_dir is None:
            cache_dir = _REPO_ROOT / "data" / "embeddings"
        self._cache = EmbeddingCache(cache_dir)

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
            top_k=None,
            truncation=True,
            max_length=512,
        )
        logger.info("RoBERTa model ready.")
        return pipe

    def score_text(self, text: str, post_id: str | None = None) -> float:
        """Score a single text. Checks embedding cache first if post_id is given.

        Args:
            text: Post text to score.
            post_id: Canonical post identifier used as cache key. When provided,
                     a cache hit skips the model forward pass entirely.

        Returns:
            Sentiment score ∈ [-1, 1] (P(pos) - P(neg)).
        """
        scores = self.score_texts([text], post_ids=[post_id] if post_id else None)
        return scores[0]

    def score_texts(
        self,
        texts: list[str],
        post_ids: list[str | None] | None = None,
    ) -> list[float]:
        """Score a list of texts. Returns per-text score ∈ [-1, 1].

        Args:
            texts: Post texts to score.
            post_ids: Optional list of cache keys (same length as texts).
                      Cache hits skip the model forward pass. Misses are
                      run through the model and stored in the cache.

        Uses batched inference. Empty texts return 0.0 (neutral).
        """
        if not texts:
            return []

        n = len(texts)
        results = [0.0] * n
        ids = post_ids if post_ids and len(post_ids) == n else [None] * n

        # Partition into cache hits and model-needed indices
        needs_model: list[int] = []  # indices into texts/results
        needs_model_ids: list[str | None] = []
        truncated: list[str] = []  # texts for the model

        for i, (t, pid) in enumerate(zip(texts, ids)):
            t_strip = t.strip()
            if not t_strip:
                continue  # empty → 0.0 (default)
            if pid is not None:
                vec = self._cache.get(pid)
                if vec is not None:
                    results[i] = float(vec[2] - vec[0])  # pos - neg
                    continue
            needs_model.append(i)
            needs_model_ids.append(pid)
            truncated.append(_truncate_text(t_strip))

        n_cached = n - len(needs_model) - sum(1 for t in texts if not t.strip())
        if n_cached > 0:
            logger.debug("score_texts: %d/%d cache hits", n_cached, n)

        if not truncated:
            self._cache.log_stats()
            return results

        # Batch model inference on cache misses
        raw_probs: list[np.ndarray] = []
        for batch_start in range(0, len(truncated), self.batch_size):
            batch = truncated[batch_start : batch_start + self.batch_size]
            raw = self._pipeline(batch)
            for item in raw:
                if isinstance(item, list):
                    score_map = {d["label"]: d["score"] for d in item}
                elif isinstance(item, dict):
                    score_map = {item["label"]: item["score"]}
                else:
                    score_map = {}
                # cardiffnlp labels: LABEL_0=neg, LABEL_1=neu, LABEL_2=pos
                neg = float(score_map.get("LABEL_0", score_map.get("negative", 0.0)))
                neu = float(score_map.get("LABEL_1", score_map.get("neutral", 0.0)))
                pos = float(score_map.get("LABEL_2", score_map.get("positive", 0.0)))
                raw_probs.append(np.array([neg, neu, pos], dtype=np.float32))

        for idx, pid, probs in zip(needs_model, needs_model_ids, raw_probs):
            results[idx] = float(probs[2] - probs[0])
            if pid is not None:
                self._cache.put(pid, probs)

        self._cache.flush()
        self._cache.log_stats()
        return results

    def score_posts_weighted(
        self,
        posts: list[dict],
        bio_classifier: BioClassifier,
        shock_id: str | None = None,
        exclude_language_prior: bool = True,
    ) -> dict[str, float]:
        """Score posts and aggregate by bio-inferred bloc weights.

        Uses the embedding cache (keyed by post["id"]) to skip the model
        forward pass for previously seen posts. The first call is slow
        (embeds everything); subsequent calls with different bio weights
        or aggregation are near-instant.

        Args:
            posts: Canonical post payload dicts with "id", "text",
                   "author_description", "lang", "inference_method" fields.
            bio_classifier: BioClassifier for demographic weight assignment.
            shock_id: Used only for logging.
            exclude_language_prior: Exclude language-prior posts from weights.

        Returns:
            Dict mapping all 15 canonical bloc IDs → score ∈ [-1, 1].
        """
        if not posts:
            logger.debug("score_posts_weighted: no posts (shock=%s)", shock_id)
            return _zero_scores()

        texts = [p.get("text", "") for p in posts]
        post_ids = [str(p.get("id") or p.get("post_id") or "") or None for p in posts]

        scores = self.score_texts(texts, post_ids=post_ids)

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
            "score_posts_weighted: shock=%s n_posts=%d cache_hit_rate=%.1f%% scores=%s",
            shock_id,
            len(posts),
            100 * self._cache.hit_rate,
            {k: f"{v:+.3f}" for k, v in list(aggregated.items())[:5]},
        )
        return aggregated

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
            shock_id,
            len(posts),
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

    scores_by_bloc_shock: dict[str, dict[str, float]] = {bloc: {} for bloc in _ALL_BLOCS}

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
            shock_id,
            len(articles_for_shock),
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
            shock_id,
            platform,
            len(posts),
        )

    return SocialMediaSentimentData(
        shock=shock_id,
        platforms=platforms,
        window_hours=window_hours,
        scores=scores,
        n_posts=n_posts,
        lagged_delta=None,
    )
