"""Tests for electoral/nlp/bio_classifier.py."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from electoral.nlp.bio_classifier import BioClassifier, BioClassification

REPO_ROOT = Path(__file__).resolve().parents[1]


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def classifier() -> BioClassifier:
    """Real BioClassifier loaded from configs — no Pi server."""
    return BioClassifier.from_config(
        config_path=REPO_ROOT / "configs" / "base.json",
        pi_server_url=None,  # disable SetFit stage so tests are hermetic
    )


@pytest.fixture(scope="module")
def classifier_with_pi() -> BioClassifier:
    """BioClassifier pointing at a fake Pi URL (Stage 3 mocked in tests)."""
    return BioClassifier.from_config(
        config_path=REPO_ROOT / "configs" / "base.json",
        pi_server_url="http://fake-pi:9000",
    )


def _make_pi_response(bloc: str | None, embedding: list[float] | None = None) -> MagicMock:
    """Build a mock urllib response with the given bloc."""
    payload = json.dumps({"bloc": bloc, "embedding": embedding or []}).encode()
    mock_resp = MagicMock()
    mock_resp.read.return_value = payload
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    return mock_resp


# ── Stage 1: Keyword lexicon ──────────────────────────────────────────────────


def test_keyword_religion_evangelical(classifier):
    result = classifier.classify("I am an evangelical Christian pastor")
    assert result.inference_method == "keyword_bio"
    assert "evangelical" in result.religion_weights
    assert result.religion_weights["evangelical"] > 0.8


def test_keyword_religion_weights_sum_to_one(classifier):
    result = classifier.classify("Catholic mom of three, pro-life")
    assert result.inference_method == "keyword_bio"
    total = sum(result.religion_weights.values())
    assert abs(total - 1.0) < 1e-6


def test_keyword_race_african_american(classifier):
    result = classifier.classify("African American activist and community organizer")
    assert result.inference_method == "keyword_bio"
    assert "african_american" in result.race_weights
    assert result.race_weights["african_american"] >= 0.5


def test_keyword_race_weights_sum_to_one(classifier):
    result = classifier.classify("Proud latina woman, educator")
    if result.inference_method == "keyword_bio":
        total = sum(result.race_weights.values())
        if result.race_weights:
            assert abs(total - 1.0) < 1e-6


def test_keyword_multi_stratum(classifier):
    # "born again" → evangelical; "african american" → race
    result = classifier.classify("Born again Christian and African American writer")
    assert result.inference_method == "keyword_bio"
    assert result.religion_weights  # evangelical from "born again"
    assert result.race_weights  # african_american from "african american"


def test_keyword_gender_weights_sum_to_one(classifier):
    result = classifier.classify("she/her, feminist organizer")
    if result.inference_method == "keyword_bio" and result.gender_weights:
        total = sum(result.gender_weights.values())
        assert abs(total - 1.0) < 1e-6


def test_keyword_no_match_returns_none_method(classifier):
    result = classifier.classify("Software engineer at a startup in Austin")
    # No demographic keywords → Stage 1 misses, no Pi → inference_method None
    assert result.inference_method is None
    assert not result.race_weights
    assert not result.religion_weights
    assert not result.gender_weights


def test_empty_bio_returns_no_signal(classifier):
    result = classifier.classify("")
    assert not result.has_signal()
    assert result.inference_method is None


def test_none_bio_returns_no_signal(classifier):
    result = classifier.classify(None)
    assert not result.has_signal()


def test_whitespace_only_bio_returns_no_signal(classifier):
    result = classifier.classify("   ")
    assert not result.has_signal()


# ── Stage 2: Language prior ───────────────────────────────────────────────────


def test_language_prior_spanish(classifier):
    result = classifier.classify("Hola soy de México", lang="es")
    # Bio has no English keywords, Spanish → language_prior
    assert result.inference_method in ("language_prior", None)


def test_language_prior_spanish_boosts_latino(classifier):
    """Spanish lang with no bio keywords must assign non-zero latino race weight."""
    result = classifier.classify("", lang="es")
    assert result.inference_method == "language_prior"
    assert "latino" in result.race_weights
    assert result.race_weights["latino"] > 0


def test_language_prior_inference_method(classifier):
    result = classifier.classify("안녕하세요", lang="ko")
    if result.inference_method == "language_prior":
        # language_prior posts must NOT contribute to mean/covariance
        assert not result.is_estimable()


def test_english_lang_skips_language_prior(classifier):
    result = classifier.classify("I love coffee", lang="en")
    # English bio, no keywords → no signal (language prior not applied for English)
    assert result.inference_method in (None, "keyword_bio")
    assert result.inference_method != "language_prior"


# ── Stage 3: SetFit (mocked Pi server) ───────────────────────────────────────


def test_setfit_religion_bloc_parsed(classifier_with_pi):
    mock_resp = _make_pi_response("religion:evangelical")
    with patch("urllib.request.urlopen", return_value=mock_resp):
        result = classifier_with_pi.classify("Some bio with no keywords")
    assert result.inference_method == "setfit_bio"
    assert result.religion_weights == {"evangelical": 1.0}
    assert not result.race_weights
    assert not result.gender_weights


def test_setfit_race_bloc_parsed(classifier_with_pi):
    mock_resp = _make_pi_response("race:latino")
    with patch("urllib.request.urlopen", return_value=mock_resp):
        result = classifier_with_pi.classify("Bio that triggers SetFit stage")
    assert result.inference_method == "setfit_bio"
    assert result.race_weights == {"latino": 1.0}


def test_setfit_gender_bloc_parsed(classifier_with_pi):
    mock_resp = _make_pi_response("gender:women")
    with patch("urllib.request.urlopen", return_value=mock_resp):
        result = classifier_with_pi.classify("She cares about climate policy")
    assert result.inference_method == "setfit_bio"
    assert result.gender_weights == {"women": 1.0}


def test_setfit_null_bloc_returns_none(classifier_with_pi):
    mock_resp = _make_pi_response(None)
    with patch("urllib.request.urlopen", return_value=mock_resp):
        result = classifier_with_pi.classify("Ambiguous bio text")
    # Null bloc → Stage 3 returns None → overall no signal
    assert result.inference_method is None


def test_setfit_pi_unavailable_falls_through(classifier_with_pi):
    with patch("urllib.request.urlopen", side_effect=OSError("Connection refused")):
        result = classifier_with_pi.classify("Unreachable server bio")
    assert result.inference_method is None


def test_setfit_malformed_bloc_ignored(classifier_with_pi):
    mock_resp = _make_pi_response("badformat")  # no colon
    with patch("urllib.request.urlopen", return_value=mock_resp):
        result = classifier_with_pi.classify("Some bio text")
    assert result.inference_method is None


def test_setfit_unknown_stratum_ignored(classifier_with_pi):
    mock_resp = _make_pi_response("ideology:progressive")  # not a valid stratum
    with patch("urllib.request.urlopen", return_value=mock_resp):
        result = classifier_with_pi.classify("Progressive activist")
    assert result.inference_method is None


def test_setfit_unknown_bloc_id_ignored(classifier_with_pi):
    mock_resp = _make_pi_response("race:martian")  # not in CANONICAL_RACES
    with patch("urllib.request.urlopen", return_value=mock_resp):
        result = classifier_with_pi.classify("Martian voter")
    assert result.inference_method is None


# ── Upstream proxy passthrough ────────────────────────────────────────────────


def test_platform_proxy_passthrough(classifier):
    result = classifier.classify("Any bio", inference_method="platform_proxy")
    assert result.inference_method == "platform_proxy"
    assert not result.has_signal()


def test_subreddit_proxy_passthrough(classifier):
    result = classifier.classify("Any bio", inference_method="subreddit_proxy")
    assert result.inference_method == "subreddit_proxy"
    assert not result.has_signal()


# ── Batch classify ────────────────────────────────────────────────────────────


def test_classify_batch_length_matches(classifier):
    posts = [
        {"author_description": "Evangelical pastor", "lang": "en"},
        {"author_description": None, "lang": "en"},
        {"author_description": "latina teacher", "lang": "en"},
    ]
    results = classifier.classify_batch(posts)
    assert len(results) == 3


def test_classify_batch_types(classifier):
    posts = [
        {"author_description": "born again Christian", "lang": "en"},
        {"author_description": "", "lang": "en"},
    ]
    results = classifier.classify_batch(posts)
    for r in results:
        assert isinstance(r, BioClassification)


# ── Estimability invariant ────────────────────────────────────────────────────


def test_keyword_bio_is_estimable(classifier):
    result = classifier.classify("Evangelical voter")
    if result.inference_method == "keyword_bio":
        assert result.is_estimable()


def test_language_prior_not_estimable(classifier):
    result = classifier.classify("Hola desde España", lang="es")
    if result.inference_method == "language_prior":
        assert not result.is_estimable()


def test_no_signal_not_estimable(classifier):
    result = classifier.classify("Software engineer")
    assert not result.is_estimable()


# ── combine_signals: per-stratum weights sum to 1.0 ──────────────────────────


def test_combine_signals_religion_sums_to_one(classifier):
    """_normalize_weights must produce religion weights that sum to 1.0."""
    result = classifier.classify("Catholic evangelical born again Christian")
    assert result.inference_method == "keyword_bio"
    if result.religion_weights:
        assert abs(sum(result.religion_weights.values()) - 1.0) < 1e-6


def test_combine_signals_race_sums_to_one(classifier):
    result = classifier.classify("Proud latina and Black woman activist")
    if result.inference_method == "keyword_bio" and result.race_weights:
        assert abs(sum(result.race_weights.values()) - 1.0) < 1e-6


def test_combine_signals_gender_sums_to_one(classifier):
    result = classifier.classify("She/her, feminist and proud")
    if result.inference_method == "keyword_bio" and result.gender_weights:
        assert abs(sum(result.gender_weights.values()) - 1.0) < 1e-6


def test_combine_signals_multi_stratum_each_sums_to_one(classifier):
    """When multiple strata fire, each stratum's weights independently sum to 1.0."""
    result = classifier.classify("Evangelical African American woman pastor")
    if result.inference_method == "keyword_bio":
        for stratum_name, weights in [
            ("race", result.race_weights),
            ("religion", result.religion_weights),
            ("gender", result.gender_weights),
        ]:
            if weights:
                total = sum(weights.values())
                assert abs(total - 1.0) < 1e-6, f"{stratum_name} weights sum to {total}, not 1.0"


# ── Unclassifiable bio → upstream proxy passthrough ──────────────────────────


def test_unclassifiable_bio_platform_proxy_passthrough(classifier):
    """Bio that cannot be classified returns the upstream platform_proxy label unchanged."""
    result = classifier.classify(
        "xkcd enthusiast, coffee addict",
        inference_method="platform_proxy",
    )
    assert result.inference_method == "platform_proxy"
    assert not result.race_weights
    assert not result.religion_weights
    assert not result.gender_weights
    assert not result.is_estimable()
