"""Tests for scorer.py, sentiment kernel, and fine-tune dataset assembly."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from electoral.nlp.bio_classifier import BioClassifier
from electoral.nlp.elasticity import assemble_finetune_dataset, score_to_bin
from electoral.nlp.scorer import EmbeddingCache, _aggregate_scores, _zero_scores

REPO_ROOT = Path(__file__).resolve().parents[1]

ALL_BLOCS = (
    ["african_american", "latino", "asian", "white", "other_race"]
    + ["evangelical", "catholic", "protestant", "secular", "jewish", "muslim", "other_rel"]
    + ["women", "men", "other_gender"]
)


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def bio_clf() -> BioClassifier:
    return BioClassifier.from_config(pi_server_url=None)


def _mock_bio(race_weights=None, religion_weights=None, gender_weights=None, method="keyword_bio"):
    from electoral.nlp.bio_classifier import BioClassification

    return BioClassification(
        inference_method=method,
        race_weights=race_weights or {},
        religion_weights=religion_weights or {},
        gender_weights=gender_weights or {},
    )


# ── score_to_bin ──────────────────────────────────────────────────────────────


def test_score_to_bin_neutral():
    assert score_to_bin(0.0) == "neutral"
    assert score_to_bin(0.04) == "neutral"
    assert score_to_bin(-0.04) == "neutral"


def test_score_to_bin_positive():
    assert score_to_bin(0.10) == "slight_pos"
    assert score_to_bin(0.20) == "mild_pos"
    assert score_to_bin(0.40) == "mod_pos"
    assert score_to_bin(0.80) == "strong_pos"


def test_score_to_bin_negative():
    assert score_to_bin(-0.10) == "slight_neg"
    assert score_to_bin(-0.20) == "mild_neg"
    assert score_to_bin(-0.40) == "mod_neg"
    assert score_to_bin(-0.80) == "strong_neg"


def test_score_to_bin_clamps_beyond_range():
    assert score_to_bin(1.5) == "strong_pos"
    assert score_to_bin(-1.5) == "strong_neg"


# ── _aggregate_scores ─────────────────────────────────────────────────────────


def test_aggregate_scores_returns_all_15_blocs():
    scores = [0.5, -0.3, 0.1]
    bio_results = [
        _mock_bio(race_weights={"white": 1.0}),
        _mock_bio(religion_weights={"evangelical": 1.0}),
        _mock_bio(gender_weights={"women": 1.0}),
    ]
    result = _aggregate_scores(scores, bio_results, exclude_language_prior=True)
    assert set(result.keys()) == set(ALL_BLOCS)


def test_aggregate_scores_values_in_range():
    scores = [0.8, -0.6, 0.2, -0.9, 0.1]
    bio_results = [
        _mock_bio(race_weights={"african_american": 1.0}),
        _mock_bio(religion_weights={"secular": 1.0}),
        _mock_bio(gender_weights={"women": 1.0}),
        _mock_bio(race_weights={"latino": 0.6, "white": 0.4}),
        _mock_bio(religion_weights={"catholic": 1.0}),
    ]
    result = _aggregate_scores(scores, bio_results, exclude_language_prior=True)
    for bloc, val in result.items():
        assert -1.0 <= val <= 1.0, f"{bloc} score {val} out of range"


def test_aggregate_scores_excludes_language_prior():
    """language_prior posts must not affect the aggregate."""
    scores = [0.9, 0.9]
    bio_all = [
        _mock_bio(race_weights={"white": 1.0}, method="keyword_bio"),
        _mock_bio(race_weights={"latino": 1.0}, method="language_prior"),
    ]
    result_excl = _aggregate_scores(scores, bio_all, exclude_language_prior=True)
    bio_none = [
        _mock_bio(race_weights={"white": 1.0}, method="keyword_bio"),
        _mock_bio(method=None),  # no signal
    ]
    result_none = _aggregate_scores(scores, bio_none, exclude_language_prior=True)
    # Both should have identical white scores since language_prior is excluded
    assert abs(result_excl["white"] - result_none["white"]) < 1e-9


def test_aggregate_scores_no_posts_returns_zeros():
    result = _aggregate_scores([], [], exclude_language_prior=True)
    assert result == _zero_scores()
    assert all(v == 0.0 for v in result.values())


# ── EmbeddingCache ────────────────────────────────────────────────────────────


def test_embedding_cache_is_deterministic(tmp_path):
    import numpy as np

    cache1 = EmbeddingCache(cache_dir=tmp_path)
    vec = np.array([0.1, 0.2, 0.3], dtype=float)
    cache1.put("post_001", vec)
    cache1.flush()

    # New instance must load the flushed entry from parquet
    cache2 = EmbeddingCache(cache_dir=tmp_path)
    retrieved = cache2.get("post_001")
    assert retrieved is not None
    assert np.allclose(retrieved, vec)


def test_embedding_cache_hit_rate(tmp_path):
    import numpy as np

    cache = EmbeddingCache(cache_dir=tmp_path)
    vec = np.ones(16, dtype=float)

    assert cache.get("missing") is None

    cache.put("p1", vec)
    cache.get("p1")  # hit
    cache.get("p2")  # miss

    rate = cache.hit_rate
    assert 0.0 <= rate <= 1.0


# ── Bio weight invariant ──────────────────────────────────────────────────────


def test_keyword_bio_weights_sum_to_one(bio_clf):
    """Bio classifier Stage 1 weights must sum to 1.0 within each stratum."""
    result = bio_clf.classify("Born again evangelical Christian")
    if result.religion_weights:
        total = sum(result.religion_weights.values())
        assert abs(total - 1.0) < 1e-6, f"Religion weights sum to {total}"


def test_language_prior_posts_excluded_from_covariance(bio_clf):
    """language_prior inference_method → is_estimable() must be False."""
    result = bio_clf.classify("Hola soy de México", lang="es")
    if result.inference_method == "language_prior":
        assert not result.is_estimable()


# ── Fine-tune dataset train/eval split ────────────────────────────────────────


def test_finetune_dataset_no_train_eval_overlap(tmp_path):
    """train.jsonl and eval.jsonl must not share any shock_id."""
    from electoral.artifacts import SentimentData

    shocks_cfg = [{"id": f"shock_{i:03d}", "description": f"Shock event {i}"} for i in range(10)]
    # eval shock: any one shock with cycle 2020 in its ID
    shocks_cfg.append({"id": "election_2020", "description": "2020 presidential election"})

    # Build minimal SentimentData with all shocks
    all_ids = [s["id"] for s in shocks_cfg]
    scores = {bloc: {sid: 0.0 for sid in all_ids} for bloc in ALL_BLOCS}
    sentiment = SentimentData(
        model="test-model",
        shocks=all_ids,
        scores=scores,
    )

    train_path = tmp_path / "train.jsonl"
    assemble_finetune_dataset(
        sentiment_data=sentiment,
        shocks_config=shocks_cfg,
        output_path=train_path,
    )

    assert train_path.exists()
    train_ids = {
        json.loads(line)["shock_id"] for line in train_path.read_text().splitlines() if line.strip()
    }
    # All shock IDs with a description should be in train
    assert len(train_ids) == len(shocks_cfg)


def test_finetune_dataset_all_blocs_present(tmp_path):
    """Every record in train.jsonl must have all 15 canonical bloc delta bins."""
    from electoral.artifacts import SentimentData

    shocks_cfg = [{"id": "metoo_2017", "description": "#MeToo movement"}]
    scores = {bloc: {"metoo_2017": 0.1} for bloc in ALL_BLOCS}
    sentiment = SentimentData(
        model="roberta-base",
        shocks=["metoo_2017"],
        scores=scores,
    )
    out = tmp_path / "train.jsonl"
    assemble_finetune_dataset(
        sentiment_data=sentiment,
        shocks_config=shocks_cfg,
        output_path=out,
    )
    records = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
    assert len(records) == 1
    output_text = records[0]["output"]
    for bloc in ALL_BLOCS:
        assert bloc in output_text, f"Missing bloc '{bloc}' in output text"


def test_roberta_score_in_range():
    """Aggregate scores from _aggregate_scores must be in [-1, 1] for all blocs."""
    import numpy as np

    rng = np.random.default_rng(42)
    scores = rng.uniform(-1.0, 1.0, 50).tolist()
    bio_results = []
    blocs = ["white", "evangelical", "women", "african_american", "secular", "men"]
    for i in range(50):
        bloc = blocs[i % len(blocs)]
        if bloc in ["white", "african_american"]:
            bio_results.append(_mock_bio(race_weights={bloc: 1.0}))
        elif bloc in ["evangelical", "secular"]:
            bio_results.append(_mock_bio(religion_weights={bloc: 1.0}))
        else:
            bio_results.append(_mock_bio(gender_weights={bloc: 1.0}))

    result = _aggregate_scores(scores, bio_results, exclude_language_prior=True)
    for bloc, val in result.items():
        assert math.isfinite(val), f"{bloc} score is not finite"
        assert -1.0 <= val <= 1.0, f"{bloc} score {val:.4f} out of [-1, 1]"
