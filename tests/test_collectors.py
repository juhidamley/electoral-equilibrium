"""Tests for social media collector schema, normalization, and routing logic.

These tests do NOT require network access or API credentials.
They test the normalization, keyword routing, and schema correctness
that downstream merge_posts() depends on.
"""

from __future__ import annotations

import json

import pytest

from electoral.nlp.collectors.schema import (
    append_post_record,
    build_keyword_index,
    build_post_payload,
    extract_primary_lang,
    is_english,
    load_shocks,
    match_shocks,
    normalize_timestamp,
    wrap_envelope,
    SCHEMA_VERSION,
    STAGE,
)
from electoral.nlp.collectors.apify_x_scraper import (
    normalize_apify_tweet,
    _extract_text,
)


# ── Timestamp normalization ───────────────────────────────────────────────────


class TestNormalizeTimestamp:
    def test_iso_with_z_suffix(self):
        result = normalize_timestamp("2020-01-03T05:11:00.000Z")
        assert "2020-01-03" in result
        assert "T" in result

    def test_iso_with_offset(self):
        result = normalize_timestamp("2020-01-03T05:11:00+00:00")
        assert "2020-01-03" in result

    def test_unix_epoch_seconds(self):
        result = normalize_timestamp("1578028260")
        assert "2020" in result  # 2020-01-03

    def test_unix_epoch_milliseconds(self):
        result = normalize_timestamp("1578028260000")
        assert "2020" in result

    def test_none_input(self):
        result = normalize_timestamp(None)
        assert result  # Returns current UTC time
        assert "T" in result

    def test_empty_string(self):
        result = normalize_timestamp("")
        assert result

    def test_mysql_style(self):
        result = normalize_timestamp("2020-01-03 05:11:00")
        assert "2020-01-03" in result

    def test_already_valid_iso(self):
        ts = "2020-01-03T05:11:00+00:00"
        result = normalize_timestamp(ts)
        assert "2020-01-03" in result


# ── Language detection ────────────────────────────────────────────────────────


class TestLanguageDetection:
    def test_english_list(self):
        assert is_english(["en"])
        assert is_english(["en-US"])
        assert is_english(["en-GB"])

    def test_empty_list(self):
        assert is_english([])
        assert is_english(None)
        assert is_english("")

    def test_non_english(self):
        assert not is_english(["fr"])
        assert not is_english(["es"])
        assert not is_english(["ar"])

    def test_mixed_langs(self):
        assert is_english(["en", "es"])

    def test_extract_primary_lang(self):
        assert extract_primary_lang(["EN-US", "fr"]) == "en-us"
        assert extract_primary_lang([]) == ""
        assert extract_primary_lang(None) == ""
        assert extract_primary_lang("ES") == "es"


# ── Keyword index and matching ────────────────────────────────────────────────

SAMPLE_SHOCKS = [
    {
        "id": "ayatollah_assassination",
        "keywords": ["Soleimani", "Iran", "IRGC", "drone strike"],
        "active": True,
    },
    {
        "id": "kavanaugh_2018",
        "keywords": ["Kavanaugh", "SCOTUS", "Christine Ford"],
        "active": False,
    },
]


class TestKeywordIndex:
    def test_build_index_lowercase(self):
        idx = build_keyword_index(SAMPLE_SHOCKS)
        assert "soleimani" in idx
        assert "kavanaugh" in idx
        assert "irgc" in idx

    def test_index_maps_to_shock_ids(self):
        idx = build_keyword_index(SAMPLE_SHOCKS)
        assert "ayatollah_assassination" in idx["soleimani"]
        assert "kavanaugh_2018" in idx["kavanaugh"]

    def test_match_single_shock(self):
        idx = build_keyword_index(SAMPLE_SHOCKS)
        matched = match_shocks("Breaking: Soleimani killed in drone strike", idx)
        assert "ayatollah_assassination" in matched
        assert "kavanaugh_2018" not in matched

    def test_match_multiple_shocks(self):
        # A post mentioning keywords from two different shocks
        idx = build_keyword_index(SAMPLE_SHOCKS)
        matched = match_shocks("After Kavanaugh and Iran IRGC actions...", idx)
        assert "ayatollah_assassination" in matched
        assert "kavanaugh_2018" in matched

    def test_no_match(self):
        idx = build_keyword_index(SAMPLE_SHOCKS)
        matched = match_shocks("Happy birthday to my dog", idx)
        assert len(matched) == 0

    def test_case_insensitive(self):
        idx = build_keyword_index(SAMPLE_SHOCKS)
        assert len(match_shocks("SOLEIMANI was killed", idx)) > 0
        assert len(match_shocks("soleimani was killed", idx)) > 0

    def test_substring_match(self):
        idx = build_keyword_index(SAMPLE_SHOCKS)
        assert len(match_shocks("The Iranian government reacts", idx)) > 0


# ── Payload building ──────────────────────────────────────────────────────────


class TestBuildPostPayload:
    def _make(self, **overrides) -> dict:
        defaults = dict(
            post_id="at://did:plc:test/app.bsky.feed.post/abc123",
            text="Soleimani killed in drone strike near Baghdad",
            created_at="2020-01-03T05:11:00.000Z",
            lang="en",
            source="live_stream",
            archive_id="bluesky",
            platform="bluesky",
            shock_id="ayatollah_assassination",
            author_did="did:plc:test123",
            author_handle=None,
            author_description=None,
            inference_method=None,
        )
        defaults.update(overrides)
        return build_post_payload(**defaults)

    def test_required_fields_present(self):
        p = self._make()
        required = ["id", "text", "created_at", "lang", "source", "archive_id", "platform"]
        for field in required:
            assert field in p, f"Missing field: {field}"

    def test_user_schema_alignment(self):
        """Ensure the schema matches what the user specified."""
        p = self._make()
        assert p["source"] == "live_stream"
        assert p["archive_id"] == "bluesky"
        assert p["id"] == "at://did:plc:test/app.bsky.feed.post/abc123"

    def test_timestamp_normalized(self):
        p = self._make(created_at="2020-01-03T05:11:00.000Z")
        assert "2020-01-03" in p["created_at"]
        assert "T" in p["created_at"]

    def test_null_fields(self):
        p = self._make(author_handle=None, author_description=None)
        assert p["author_handle"] is None
        assert p["author_description"] is None
        assert p["inference_method"] is None

    def test_text_stripped(self):
        p = self._make(text="  hello world  ")
        assert p["text"] == "hello world"

    def test_apify_source_tag(self):
        p = self._make(source="live_scrape", archive_id="apify_x", platform="apify_x")
        assert p["source"] == "live_scrape"
        assert p["archive_id"] == "apify_x"


# ── Envelope wrapping ─────────────────────────────────────────────────────────


class TestWrapEnvelope:
    def test_schema_version(self):
        env = wrap_envelope({"id": "test"})
        assert env["schema_version"] == SCHEMA_VERSION
        assert env["stage"] == STAGE

    def test_seed_included(self):
        env = wrap_envelope({"id": "test"}, seed=42)
        assert env["seed"] == 42

    def test_seed_none(self):
        env = wrap_envelope({"id": "test"}, seed=None)
        assert env["seed"] is None

    def test_collected_at_present(self):
        env = wrap_envelope({"id": "test"})
        assert "collected_at" in env
        assert "T" in env["collected_at"]

    def test_payload_preserved(self):
        payload = {"id": "x", "text": "hello", "custom": 123}
        env = wrap_envelope(payload)
        assert env["payload"] == payload

    def test_json_serializable(self):
        payload = build_post_payload(
            post_id="test",
            text="test",
            created_at="2020-01-01T00:00:00Z",
            lang="en",
            source="live_stream",
            archive_id="bluesky",
            platform="bluesky",
        )
        env = wrap_envelope(payload, seed=42)
        # Must be JSON serializable (no non-serializable types)
        serialized = json.dumps(env)
        assert serialized


# ── Append to file ────────────────────────────────────────────────────────────


class TestAppendPostRecord:
    def test_append_creates_file(self, tmp_path):
        path = tmp_path / "bluesky" / "test_shock" / "intel_mac_posts.jsonl"
        payload = build_post_payload(
            post_id="test1",
            text="Soleimani killed",
            created_at="2020-01-03T05:11:00Z",
            lang="en",
            source="live_stream",
            archive_id="bluesky",
            platform="bluesky",
            shock_id="ayatollah_assassination",
        )
        append_post_record(path, payload, seed=42)
        assert path.exists()

    def test_append_is_valid_jsonl(self, tmp_path):
        path = tmp_path / "intel_mac_posts.jsonl"
        for i in range(3):
            payload = build_post_payload(
                post_id=f"post_{i}",
                text=f"Test post {i}",
                created_at="2020-01-03T05:11:00Z",
                lang="en",
                source="live_stream",
                archive_id="bluesky",
                platform="bluesky",
            )
            append_post_record(path, payload)

        lines = path.read_text().strip().splitlines()
        assert len(lines) == 3
        for line in lines:
            record = json.loads(line)
            assert record["schema_version"] == "1.0"
            assert record["stage"] == "collect"
            assert "payload" in record
            assert "id" in record["payload"]

    def test_append_does_not_overwrite(self, tmp_path):
        path = tmp_path / "posts.jsonl"

        for i in range(5):
            payload = build_post_payload(
                post_id=f"p{i}",
                text="test",
                created_at="2020-01-01T00:00:00Z",
                lang="en",
                source="live_stream",
                archive_id="bluesky",
                platform="bluesky",
            )
            append_post_record(path, payload)

        lines = path.read_text().strip().splitlines()
        assert len(lines) == 5  # All 5 records preserved

    def test_output_directory_created_automatically(self, tmp_path):
        deep_path = tmp_path / "a" / "b" / "c" / "posts.jsonl"
        payload = build_post_payload(
            post_id="p1",
            text="test",
            created_at="2020-01-01T00:00:00Z",
            lang="en",
            source="live_stream",
            archive_id="bluesky",
            platform="bluesky",
        )
        append_post_record(deep_path, payload)
        assert deep_path.exists()


# ── Apify tweet normalization ─────────────────────────────────────────────────


class TestApifyNormalization:
    def _tweet(self, **overrides) -> dict:
        base = {
            "id": "1213024797693587456",
            "full_text": "Soleimani killed in drone strike near Baghdad airport",
            "createdAt": "2020-01-03T05:11:00.000Z",
            "lang": "en",
            "user": {
                "id": "1234567890",
                "screen_name": "testuser",
                "description": "Political commentator | Evangelical Christian",
            },
        }
        base.update(overrides)
        return base

    def test_normalize_basic(self):
        result = normalize_apify_tweet(self._tweet(), "ayatollah_assassination")
        assert result is not None
        assert result["text"] == "Soleimani killed in drone strike near Baghdad airport"
        assert result["source"] == "live_scrape"
        assert result["archive_id"] == "apify_x"
        assert result["platform"] == "apify_x"
        assert result["shock_id"] == "ayatollah_assassination"

    def test_normalize_author_fields(self):
        result = normalize_apify_tweet(self._tweet(), "ayatollah_assassination")
        assert result["author_handle"] == "testuser"
        assert result["author_description"] == "Political commentator | Evangelical Christian"

    def test_normalize_timestamp(self):
        result = normalize_apify_tweet(self._tweet(), "test_shock")
        assert "2020-01-03" in result["created_at"]

    def test_normalize_id_prefix(self):
        result = normalize_apify_tweet(self._tweet(), "test")
        assert result["id"].startswith("twitter:")

    def test_returns_none_for_empty_text(self):
        result = normalize_apify_tweet(
            {"id": "123", "full_text": "", "createdAt": "2020-01-01T00:00:00Z"},
            "test",
        )
        assert result is None

    def test_returns_none_for_missing_text(self):
        result = normalize_apify_tweet({"id": "123"}, "test")
        assert result is None

    def test_alternative_field_names(self):
        # Test apidojo actor format (text instead of full_text)
        item = {
            "id": "456",
            "text": "Iran IRGC Soleimani",
            "timestamp": "1578028260000",  # Unix ms
            "lang": "en",
            "author": {"screen_name": "user2"},
        }
        result = normalize_apify_tweet(item, "ayatollah_assassination")
        assert result is not None
        assert "Iran" in result["text"]
        assert "2020" in result["created_at"]

    def test_extract_text_strips_truncation(self):
        text = _extract_text({"full_text": "Breaking news… https://t.co/abc123"})
        assert "…" not in text or "https://t.co" not in text

    def test_max_items_free_tier(self):
        from electoral.nlp.collectors.apify_x_scraper import ApifyXScraper

        with pytest.raises(ValueError, match="free-tier"):
            ApifyXScraper(
                shock_id="test",
                keywords=["test"],
                output_root="/tmp",
                apify_token="fake_token",
                max_items=501,  # Exceeds 500 limit
            )


# ── Schema consistency between platforms ─────────────────────────────────────


class TestCrossPlatformConsistency:
    """Verify Bluesky and Apify records share the same schema so merge_posts works."""

    def _make_bluesky_record(self, shock_id="test") -> dict:
        return build_post_payload(
            post_id="at://did:plc:test/app.bsky.feed.post/rkey123",
            text="Iran IRGC news",
            created_at="2020-01-03T05:11:00Z",
            lang="en",
            source="live_stream",
            archive_id="bluesky",
            platform="bluesky",
            shock_id=shock_id,
            author_did="did:plc:test123",
        )

    def _make_apify_record(self, shock_id="test") -> dict:
        return normalize_apify_tweet(
            {
                "id": "1213024797693587456",
                "full_text": "Iran IRGC news",
                "createdAt": "2020-01-03T05:11:00Z",
                "lang": "en",
                "user": {"screen_name": "testuser"},
            },
            shock_id,
        )

    def test_same_required_keys(self):
        bsky = self._make_bluesky_record()
        apify = self._make_apify_record()

        required = ["id", "text", "created_at", "lang", "source", "archive_id", "platform"]
        for key in required:
            assert key in bsky, f"Bluesky missing: {key}"
            assert key in apify, f"Apify missing: {key}"

    def test_different_source_tags(self):
        bsky = self._make_bluesky_record()
        apify = self._make_apify_record()
        assert bsky["source"] == "live_stream"
        assert apify["source"] == "live_scrape"

    def test_different_archive_ids(self):
        bsky = self._make_bluesky_record()
        apify = self._make_apify_record()
        assert bsky["archive_id"] == "bluesky"
        assert apify["archive_id"] == "apify_x"

    def test_null_fields_consistent(self):
        bsky = self._make_bluesky_record()
        apify = self._make_apify_record()
        # Both should have inference_method as null
        assert bsky["inference_method"] is None
        assert apify["inference_method"] is None

    def test_envelope_schema_version_consistent(self):
        bsky_payload = self._make_bluesky_record()
        apify_payload = self._make_apify_record()
        env_bsky = wrap_envelope(bsky_payload, seed=42)
        env_apify = wrap_envelope(apify_payload, seed=42)
        assert env_bsky["schema_version"] == env_apify["schema_version"] == "1.0"
        assert env_bsky["stage"] == env_apify["stage"] == "collect"


# ── Shocks.json loading ───────────────────────────────────────────────────────


class TestLoadShocks:
    def test_loads_real_shocks_json(self):
        shocks = load_shocks("configs/shocks.json")
        assert len(shocks) >= 1
        first = shocks[0]
        assert "id" in first
        assert "keywords" in first
        assert len(first["keywords"]) > 0

    def test_ayatollah_shock_present(self):
        shocks = load_shocks("configs/shocks.json")
        ids = [s["id"] for s in shocks]
        assert "ayatollah_assassination" in ids

    def test_ayatollah_keywords_nonempty(self):
        shocks = load_shocks("configs/shocks.json")
        ayatollah = next(s for s in shocks if s["id"] == "ayatollah_assassination")
        assert len(ayatollah["keywords"]) > 0
        # Pilot test case must include the key identifying terms
        kw_lower = [k.lower() for k in ayatollah["keywords"]]
        assert any("soleimani" in k for k in kw_lower)
        assert any("iran" in k for k in kw_lower)
