"""Tests for bio_classifier, scorer, and elasticity modules.

These tests do NOT require transformers/torch, the Pi server, or any model.
They test normalization, keyword matching, aggregation logic, and bin mapping.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from electoral.nlp.bio_classifier import (
    BioClassification,
    BioClassifier,
    _normalize_weights,
)
from electoral.nlp.elasticity import score_to_bin
from electoral.nlp.scorer import (
    _aggregate_scores,
    _extract_outlet_proxy,
    _truncate_text,
    _zero_scores,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_CONFIGS = _REPO_ROOT / "configs"


# ── score_to_bin ──────────────────────────────────────────────────────────────


class TestScoreToBin:
    def test_strong_neg(self):
        assert score_to_bin(-1.0) == "strong_neg"
        assert score_to_bin(-0.75) == "strong_neg"
        assert score_to_bin(-0.51) == "strong_neg"

    def test_mod_neg(self):
        # Boundary at -0.50 inclusive falls into mod_neg (lo <= score < hi)
        assert score_to_bin(-0.50) == "mod_neg"
        assert score_to_bin(-0.40) == "mod_neg"
        assert score_to_bin(-0.31) == "mod_neg"

    def test_mild_neg(self):
        # Boundary at -0.30 inclusive falls into mild_neg
        assert score_to_bin(-0.30) == "mild_neg"
        assert score_to_bin(-0.20) == "mild_neg"
        assert score_to_bin(-0.16) == "mild_neg"

    def test_slight_neg(self):
        # Boundary at -0.15 inclusive falls into slight_neg
        assert score_to_bin(-0.15) == "slight_neg"
        assert score_to_bin(-0.10) == "slight_neg"
        assert score_to_bin(-0.06) == "slight_neg"

    def test_neutral(self):
        assert score_to_bin(0.0) == "neutral"
        # Boundary at -0.05 inclusive falls into neutral
        assert score_to_bin(-0.05) == "neutral"
        assert score_to_bin(0.04) == "neutral"

    def test_slight_pos(self):
        assert score_to_bin(0.06) == "slight_pos"
        assert score_to_bin(0.10) == "slight_pos"

    def test_mild_pos(self):
        # Boundary at 0.15 exclusive — 0.15 is end of slight_pos (lo=0.05, hi=0.15)
        # so 0.15 falls into mild_pos (lo=0.15, hi=0.30)
        assert score_to_bin(0.15) == "mild_pos"
        assert score_to_bin(0.20) == "mild_pos"

    def test_mod_pos(self):
        # 0.30 is in mild_pos (lo=0.15, hi=0.30) — no wait: 0.30 < 0.30 is False
        # so 0.30 falls into mod_pos (lo=0.30, hi=0.50)
        assert score_to_bin(0.30) == "mod_pos"
        assert score_to_bin(0.40) == "mod_pos"

    def test_strong_pos(self):
        # 0.50 falls into strong_pos (lo=0.50, hi=1.001)
        assert score_to_bin(0.50) == "strong_pos"
        assert score_to_bin(0.75) == "strong_pos"
        assert score_to_bin(1.0) == "strong_pos"

    def test_all_bins_covered(self):
        bins = {score_to_bin(v) for v in [-0.9, -0.4, -0.2, -0.1, 0.0, 0.1, 0.2, 0.4, 0.9]}
        expected = {
            "strong_neg",
            "mod_neg",
            "mild_neg",
            "slight_neg",
            "neutral",
            "slight_pos",
            "mild_pos",
            "mod_pos",
            "strong_pos",
        }
        assert bins == expected


# ── _normalize_weights ────────────────────────────────────────────────────────


class TestNormalizeWeights:
    def test_basic_normalization(self):
        result = _normalize_weights({"a": 2.0, "b": 2.0})
        assert abs(sum(result.values()) - 1.0) < 1e-9
        assert abs(result["a"] - 0.5) < 1e-9

    def test_single_key(self):
        result = _normalize_weights({"evangelical": 1.0})
        assert result == {"evangelical": 1.0}

    def test_zero_total_returns_empty(self):
        result = _normalize_weights({"a": 0.0, "b": 0.0})
        assert result == {}

    def test_sums_to_one_multi(self):
        result = _normalize_weights({"a": 1.0, "b": 2.0, "c": 3.0})
        assert abs(sum(result.values()) - 1.0) < 1e-9

    def test_zero_weight_entries_removed(self):
        result = _normalize_weights({"a": 1.0, "b": 0.0})
        assert "b" not in result


# ── BioClassifier — keyword stage ─────────────────────────────────────────────


@pytest.fixture
def bio_clf():
    return BioClassifier.from_config(pi_server_url=None)


class TestBioClassifierKeyword:
    def test_evangelical_keyword(self, bio_clf):
        result = bio_clf.classify("I'm an evangelical pastor", lang="en")
        assert result.inference_method == "keyword_bio"
        assert result.religion_weights.get("evangelical", 0) > 0

    def test_catholic_keyword(self, bio_clf):
        result = bio_clf.classify("Proud Roman Catholic", lang="en")
        assert result.inference_method == "keyword_bio"
        assert result.religion_weights.get("catholic", 0) > 0

    def test_latino_keyword(self, bio_clf):
        result = bio_clf.classify("Latino community organizer", lang="en")
        assert result.inference_method == "keyword_bio"
        assert result.race_weights.get("latino", 0) > 0

    def test_she_her_pronouns(self, bio_clf):
        result = bio_clf.classify("she/her | teacher", lang="en")
        assert result.inference_method == "keyword_bio"
        assert result.gender_weights.get("women", 0) > 0

    def test_mixed_signals(self, bio_clf):
        result = bio_clf.classify("Black evangelical woman. She/her.", lang="en")
        assert result.inference_method == "keyword_bio"
        assert result.race_weights.get("african_american", 0) > 0
        assert result.religion_weights.get("evangelical", 0) > 0
        assert result.gender_weights.get("women", 0) > 0

    def test_weights_sum_to_one_per_stratum(self, bio_clf):
        result = bio_clf.classify("Latino catholic he/him", lang="en")
        if result.race_weights:
            assert abs(sum(result.race_weights.values()) - 1.0) < 1e-6
        if result.religion_weights:
            assert abs(sum(result.religion_weights.values()) - 1.0) < 1e-6
        if result.gender_weights:
            assert abs(sum(result.gender_weights.values()) - 1.0) < 1e-6

    def test_empty_bio_no_keyword_result(self, bio_clf):
        result = bio_clf.classify("", lang="en")
        # No keyword → no keyword stage result; falls through to Stage 3 or null
        assert result.inference_method != "keyword_bio"

    def test_case_insensitive(self, bio_clf):
        result = bio_clf.classify("EVANGELICAL", lang="en")
        assert result.inference_method == "keyword_bio"

    def test_no_match_returns_none_method(self, bio_clf):
        result = bio_clf.classify("Software engineer in NYC", lang="en")
        # No lexicon keywords → not keyword_bio; Pi unavailable → None
        assert result.inference_method != "keyword_bio"


class TestBioClassifierLanguagePrior:
    def test_spanish_lang_prior(self, bio_clf):
        result = bio_clf.classify(bio=None, lang="es")
        assert result.inference_method == "language_prior"
        assert result.race_weights.get("latino", 0) > 0

    def test_korean_lang_prior(self, bio_clf):
        result = bio_clf.classify(bio=None, lang="ko")
        assert result.inference_method == "language_prior"
        assert result.race_weights.get("asian", 0) > 0

    def test_english_does_not_trigger_language_prior(self, bio_clf):
        result = bio_clf.classify(bio=None, lang="en")
        assert result.inference_method != "language_prior"

    def test_language_prior_not_estimable(self, bio_clf):
        result = bio_clf.classify(bio=None, lang="es")
        assert not result.is_estimable()


class TestBioClassifierPlatformProxy:
    def test_platform_proxy_short_circuits(self, bio_clf):
        result = bio_clf.classify(
            bio="I'm a Catholic evangelical atheist",  # would match keywords
            lang="en",
            inference_method="platform_proxy",
        )
        assert result.inference_method == "platform_proxy"
        assert result.race_weights == {}
        assert result.religion_weights == {}

    def test_subreddit_proxy_short_circuits(self, bio_clf):
        result = bio_clf.classify(
            bio="I'm latino evangelical",
            lang="en",
            inference_method="subreddit_proxy",
        )
        assert result.inference_method == "subreddit_proxy"
        assert result.race_weights == {}


class TestBioClassification:
    def test_has_signal_true(self):
        bio = BioClassification(
            inference_method="keyword_bio",
            race_weights={"latino": 1.0},
            religion_weights={},
            gender_weights={},
        )
        assert bio.has_signal()

    def test_has_signal_false(self):
        bio = BioClassification(
            inference_method=None,
            race_weights={},
            religion_weights={},
            gender_weights={},
        )
        assert not bio.has_signal()

    def test_is_estimable_language_prior(self):
        bio = BioClassification(
            inference_method="language_prior",
            race_weights={"latino": 1.0},
            religion_weights={},
            gender_weights={},
        )
        assert not bio.is_estimable()

    def test_is_estimable_keyword_bio(self):
        bio = BioClassification(
            inference_method="keyword_bio",
            race_weights={"white": 1.0},
            religion_weights={},
            gender_weights={},
        )
        assert bio.is_estimable()


# ── Aggregation logic (no model needed) ──────────────────────────────────────


class TestAggregateScores:
    def _make_bio(self, race=None, religion=None, gender=None, method="keyword_bio"):
        return BioClassification(
            inference_method=method,
            race_weights=race or {},
            religion_weights=religion or {},
            gender_weights=gender or {},
        )

    def test_single_post_full_weight(self):
        bio = self._make_bio(race={"african_american": 1.0})
        result = _aggregate_scores([0.5], [bio])
        assert abs(result["african_american"] - 0.5) < 1e-6

    def test_two_posts_equal_weight_averaging(self):
        bio1 = self._make_bio(race={"latino": 1.0})
        bio2 = self._make_bio(race={"latino": 1.0})
        result = _aggregate_scores([0.8, 0.2], [bio1, bio2])
        assert abs(result["latino"] - 0.5) < 1e-6

    def test_zero_weight_bloc_returns_neutral(self):
        bio = self._make_bio(race={"asian": 1.0})
        result = _aggregate_scores([0.9], [bio])
        # No weight assigned to african_american → should be 0.0
        assert result["african_american"] == 0.0

    def test_language_prior_excluded_from_weights(self):
        bio_estimable = self._make_bio(race={"white": 1.0}, method="keyword_bio")
        bio_lp = self._make_bio(race={"latino": 1.0}, method="language_prior")
        result = _aggregate_scores(
            [1.0, -1.0], [bio_estimable, bio_lp], exclude_language_prior=True
        )
        # Only white gets scored; latino (language_prior) is excluded
        assert abs(result["white"] - 1.0) < 1e-6
        assert result["latino"] == 0.0

    def test_language_prior_included_when_flag_false(self):
        bio_lp = self._make_bio(race={"latino": 1.0}, method="language_prior")
        result = _aggregate_scores([-0.5], [bio_lp], exclude_language_prior=False)
        assert abs(result["latino"] - (-0.5)) < 1e-6

    def test_none_method_posts_excluded(self):
        bio = self._make_bio(method=None)
        bio.race_weights["white"] = 1.0  # would be picked up if not excluded
        # BioClassification is frozen? No — only artifacts are frozen. OK.
        result = _aggregate_scores([1.0], [bio])
        assert result["white"] == 0.0

    def test_all_blocs_present_in_result(self):
        result = _zero_scores()
        from electoral.core.types import CANONICAL_GENDERS, CANONICAL_RACES, CANONICAL_RELIGIONS

        for bloc in list(CANONICAL_RACES) + list(CANONICAL_RELIGIONS) + list(CANONICAL_GENDERS):
            assert bloc in result


# ── _extract_outlet_proxy ─────────────────────────────────────────────────────


class TestExtractOutletProxy:
    def test_valid_proxy(self):
        post = {
            "author_description": json.dumps(
                {
                    "outlet": "cbn",
                    "proxy": {"religion": {"evangelical": 1.0}},
                }
            )
        }
        result = _extract_outlet_proxy(post)
        assert result is not None
        assert result.religion_weights.get("evangelical", 0) == 1.0

    def test_no_author_description(self):
        post = {"text": "hello"}
        assert _extract_outlet_proxy(post) is None

    def test_malformed_json(self):
        post = {"author_description": "not valid json{{{"}
        assert _extract_outlet_proxy(post) is None

    def test_empty_proxy(self):
        post = {"author_description": json.dumps({"outlet": "unknown", "proxy": {}})}
        assert _extract_outlet_proxy(post) is None


# ── _truncate_text ────────────────────────────────────────────────────────────


class TestTruncateText:
    def test_short_text_unchanged(self):
        text = "hello world"
        assert _truncate_text(text, max_words=128) == text

    def test_long_text_truncated(self):
        words = ["word"] * 200
        text = " ".join(words)
        result = _truncate_text(text, max_words=128)
        assert len(result.split()) == 128

    def test_exact_limit_unchanged(self):
        words = ["w"] * 128
        text = " ".join(words)
        assert _truncate_text(text, max_words=128) == text


# ── RoBERTaScorer import guard ────────────────────────────────────────────────


class TestRoBERTaScorerImportGuard:
    def test_raises_import_error_without_transformers(self):
        import sys

        # Temporarily hide transformers from import
        with patch.dict(sys.modules, {"transformers": None}):
            from electoral.nlp import scorer as scorer_mod

            with pytest.raises((ImportError, TypeError)):
                scorer_mod.RoBERTaScorer(model_name="some/model")
