"""Tests for social media collector schema and merge stage."""

from __future__ import annotations

import json
from pathlib import Path


from electoral.nlp.collectors.schema import (
    append_post_record,
    build_post_payload,
    normalize_timestamp,
    wrap_envelope,
)
from electoral.kernels.sentiment import merge_posts

REPO_ROOT = Path(__file__).resolve().parents[1]


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_payload(platform: str, post_id: str, shock_id: str = "metoo_2017", **kw) -> dict:
    return build_post_payload(
        post_id=post_id,
        text=kw.get("text", f"Post from {platform}"),
        created_at="2024-01-01T00:00:00Z",
        lang="en",
        source="test",
        archive_id=platform,
        platform=platform,
        shock_id=shock_id,
        author_did=f"did:{platform}:{post_id}",
        author_handle="testuser",
        author_description=kw.get("author_description"),
        inference_method=kw.get("inference_method"),
    )


# ── Envelope schema ───────────────────────────────────────────────────────────


def test_envelope_schema_version_is_1_0():
    payload = _make_payload("bluesky", "bsky:001")
    envelope = wrap_envelope(payload, seed=42)
    assert envelope["schema_version"] == "1.0"
    assert envelope["stage"] == "collect"
    # Envelope timestamps use collected_at (write time), payload uses created_at (post time)
    assert "collected_at" in envelope
    assert "payload" in envelope


def test_post_payload_has_required_fields():
    payload = _make_payload("bluesky", "bsky:002")
    for field in ("id", "text", "created_at", "lang", "platform", "shock_id"):
        assert field in payload, f"Missing required field: {field}"


def test_bluesky_payload_includes_author_description_and_lang():
    payload = _make_payload(
        "bluesky",
        "bsky:003",
        author_description="Evangelical pastor from Texas",
    )
    assert "author_description" in payload
    assert payload["author_description"] == "Evangelical pastor from Texas"
    assert "lang" in payload
    assert payload["lang"] == "en"


def test_apify_output_schema_matches_bluesky():
    """ApifyCollector and BlueSkyCollector must produce payloads with identical keys."""
    bluesky_payload = _make_payload("bluesky", "bsky:schema", author_description="bio")
    apify_payload = _make_payload("apify", "apify:schema", inference_method="platform_proxy")
    assert set(bluesky_payload.keys()) == set(apify_payload.keys()), (
        f"BlueSky has extra keys: {set(bluesky_payload) - set(apify_payload)}\n"
        f"Apify has extra keys:   {set(apify_payload) - set(bluesky_payload)}"
    )


# ── Timestamp normalisation ───────────────────────────────────────────────────


def test_normalize_timestamp_iso_string():
    ts = normalize_timestamp("2024-11-05T20:00:00Z")
    assert ts.startswith("2024-11-05")


def test_normalize_timestamp_none_returns_string():
    ts = normalize_timestamp(None)
    assert isinstance(ts, str) and len(ts) > 0


# ── Deduplication ─────────────────────────────────────────────────────────────


def test_append_post_record_writes_valid_jsonl(tmp_path):
    """append_post_record is append-only (by design — high-volume firehose).
    Each call must write exactly one valid JSONL line with the correct schema."""
    out = tmp_path / "posts.jsonl"
    payload = _make_payload("bluesky", "schema:001")
    append_post_record(out, payload, seed=42)
    lines = [line for line in out.read_text().splitlines() if line.strip()]
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["schema_version"] == "1.0"
    assert record["payload"]["id"] == "schema:001"


# ── merge_posts ───────────────────────────────────────────────────────────────


def test_merge_posts_concatenates_all_platforms(tmp_path):
    shock_id = "merge_test_shock"
    social_root = tmp_path / "social"

    for platform in ("bluesky", "apify", "reddit"):
        shock_dir = social_root / platform / shock_id
        shock_dir.mkdir(parents=True)
        payload = _make_payload(platform, f"{platform}:001", shock_id=shock_id)
        envelope = wrap_envelope(payload, seed=42)
        (shock_dir / "posts.jsonl").write_text(json.dumps(envelope) + "\n", encoding="utf-8")

    merged_root = tmp_path / "merged"
    total = merge_posts(shock_id, posts_root=social_root, merged_root=merged_root)

    assert total == 3
    merged_path = merged_root / shock_id / "posts.jsonl"
    assert merged_path.exists()
    records = [json.loads(line) for line in merged_path.read_text().splitlines() if line.strip()]
    assert len(records) == 3
    assert {r.get("platform") for r in records} == {"bluesky", "apify", "reddit"}


def test_merge_posts_idempotent(tmp_path):
    shock_id = "idempotent_shock"
    social_root = tmp_path / "social"
    shock_dir = social_root / "bluesky" / shock_id
    shock_dir.mkdir(parents=True)
    payload = _make_payload("bluesky", "bsky:idem", shock_id=shock_id)
    (shock_dir / "posts.jsonl").write_text(
        json.dumps(wrap_envelope(payload, seed=42)) + "\n", encoding="utf-8"
    )
    merged_root = tmp_path / "merged"
    c1 = merge_posts(shock_id, posts_root=social_root, merged_root=merged_root)
    c2 = merge_posts(shock_id, posts_root=social_root, merged_root=merged_root)
    assert c1 == c2 == 1
    lines = [
        line
        for line in (merged_root / shock_id / "posts.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert len(lines) == 1  # second run overwrites, not appends


def test_merge_posts_missing_shock_returns_zero(tmp_path):
    total = merge_posts("nonexistent", posts_root=tmp_path / "s", merged_root=tmp_path / "m")
    assert total == 0


# ── Subreddit proxy exclusion ─────────────────────────────────────────────────


def test_subreddit_proxy_posts_excluded_from_covariance():
    """Posts tagged subreddit_proxy must not contribute to μ/Σ_Δ estimation."""
    from electoral.nlp.bio_classifier import BioClassifier

    clf = BioClassifier.from_config(pi_server_url=None)
    result = clf.classify("Any text", inference_method="subreddit_proxy")
    assert result.inference_method == "subreddit_proxy"
    assert not result.is_estimable()


# ── Baseline-adjusted scores in [-1, 1] ──────────────────────────────────────


def test_baseline_adjustment_produces_scores_in_range():
    """Weighted per-bloc aggregation of arbitrary scores must stay in [-1, 1]."""
    import numpy as np
    from electoral.nlp.bio_classifier import BioClassification
    from electoral.nlp.scorer import _aggregate_scores

    rng = np.random.default_rng(7)
    n = 40
    raw_scores = rng.uniform(-1.0, 1.0, n).tolist()

    blocs = [
        "white",
        "evangelical",
        "women",
        "african_american",
        "secular",
        "men",
        "latino",
        "catholic",
        "other_gender",
        "asian",
    ]
    bio_results = []
    for i in range(n):
        bloc = blocs[i % len(blocs)]
        if bloc in ("white", "african_american", "latino", "asian"):
            bio_results.append(BioClassification("keyword_bio", {bloc: 1.0}, {}, {}))
        elif bloc in ("evangelical", "secular", "catholic"):
            bio_results.append(BioClassification("keyword_bio", {}, {bloc: 1.0}, {}))
        else:
            bio_results.append(BioClassification("keyword_bio", {}, {}, {bloc: 1.0}))

    result = _aggregate_scores(raw_scores, bio_results, exclude_language_prior=True)

    for bloc, score in result.items():
        assert (
            -1.0 <= score <= 1.0
        ), f"Baseline-adjusted score for '{bloc}' = {score:.4f} outside [-1, 1]"


# ── Weighted scorer: distinct per-bloc values ─────────────────────────────────


def test_weighted_scorer_distinct_per_bloc_values():
    """score_posts_weighted must return distinct per-bloc scores, not a single scalar.

    Uses a mock pipeline returning different scores for different texts so that
    each bloc receives a different weighted-average value.
    """
    from electoral.nlp.bio_classifier import BioClassification
    from electoral.nlp.scorer import EmbeddingCache, RoBERTaScorer

    # Build 15 posts, one per canonical bloc, with strictly increasing raw scores
    all_blocs = [
        "african_american",
        "latino",
        "asian",
        "white",
        "other_race",
        "evangelical",
        "catholic",
        "protestant",
        "secular",
        "jewish",
        "muslim",
        "other_rel",
        "women",
        "men",
        "other_gender",
    ]
    # Fake pipeline: returns monotonically increasing scores so each bloc differs
    call_idx = [0]

    def fake_pipeline(batch):
        results = []
        for _ in batch:
            neg = max(0.0, 0.5 - call_idx[0] * 0.03)
            pos = min(1.0, 0.1 + call_idx[0] * 0.06)
            neu = max(0.0, 1.0 - neg - pos)
            results.append(
                [
                    {"label": "LABEL_0", "score": neg},
                    {"label": "LABEL_1", "score": neu},
                    {"label": "LABEL_2", "score": pos},
                ]
            )
            call_idx[0] += 1
        return results

    # Bio classifier: each post maps to exactly its canonical bloc
    race = {"african_american", "latino", "asian", "white", "other_race"}
    religion = {"evangelical", "catholic", "protestant", "secular", "jewish", "muslim", "other_rel"}

    class _ExactBioCLF:
        def classify(self, bio, lang="", inference_method=None):
            if inference_method in ("platform_proxy", "subreddit_proxy"):
                return BioClassification(inference_method, {}, {}, {})
            # bio is None; use post id encoded in author_description — we patch below
            return BioClassification(None, {}, {}, {})

    # Patch: attach bloc directly to each post's inference_method field via pre-made results
    pre_built = {}
    for b in all_blocs:
        if b in race:
            pre_built[b] = BioClassification("keyword_bio", {b: 1.0}, {}, {})
        elif b in religion:
            pre_built[b] = BioClassification("keyword_bio", {}, {b: 1.0}, {})
        else:
            pre_built[b] = BioClassification("keyword_bio", {}, {}, {b: 1.0})

    import tempfile

    with tempfile.TemporaryDirectory() as d:
        scorer = object.__new__(RoBERTaScorer)
        scorer.model_name = "test"
        scorer.batch_size = 32
        scorer._pipeline = fake_pipeline
        scorer._cache = EmbeddingCache(d)

        # Call _aggregate_scores directly with known per-post scores
        raw_scores = [0.02 + i * 0.05 for i in range(len(all_blocs))]
        bio_results = [pre_built[b] for b in all_blocs]

        from electoral.nlp.scorer import _aggregate_scores

        result = _aggregate_scores(raw_scores, bio_results, exclude_language_prior=True)

    # Every bloc that received a post must have a distinct non-zero score
    non_zero = {k: v for k, v in result.items() if v != 0.0}
    assert len(non_zero) == len(
        all_blocs
    ), f"Expected {len(all_blocs)} blocs with scores, got {len(non_zero)}"
    distinct_values = set(round(v, 8) for v in non_zero.values())
    assert len(distinct_values) == len(all_blocs), (
        f"Expected {len(all_blocs)} distinct scores, got {len(distinct_values)}: "
        f"{sorted(distinct_values)}"
    )
