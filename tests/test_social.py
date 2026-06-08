"""Tests for social media collector schema and merge stage."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

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
        "bluesky", "bsky:003",
        author_description="Evangelical pastor from Texas",
    )
    assert "author_description" in payload
    assert payload["author_description"] == "Evangelical pastor from Texas"
    assert "lang" in payload
    assert payload["lang"] == "en"


def test_apify_output_schema_matches_bluesky():
    """ApifyCollector and BlueSkyCollector must produce payloads with identical keys."""
    bluesky_payload = _make_payload("bluesky", "bsky:schema", author_description="bio")
    apify_payload   = _make_payload("apify",   "apify:schema", inference_method="platform_proxy")
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
    lines = [l for l in out.read_text().splitlines() if l.strip()]
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
        (shock_dir / "posts.jsonl").write_text(
            json.dumps(envelope) + "\n", encoding="utf-8"
        )

    merged_root = tmp_path / "merged"
    total = merge_posts(shock_id, posts_root=social_root, merged_root=merged_root)

    assert total == 3
    merged_path = merged_root / shock_id / "posts.jsonl"
    assert merged_path.exists()
    records = [json.loads(l) for l in merged_path.read_text().splitlines() if l.strip()]
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
    lines = [l for l in (merged_root / shock_id / "posts.jsonl").read_text().splitlines() if l.strip()]
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
