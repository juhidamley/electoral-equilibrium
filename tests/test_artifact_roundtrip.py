"""Round-trip tests for all stage payload dataclasses.

Each test:
  1. Instantiates a minimal valid payload
  2. Calls validate() — must not raise
  3. Serializes to dict via to_dict()
  4. Confirms JSON-serializability via json.dumps()
  5. Reconstructs via from_dict()
  6. Asserts obj2.to_dict() == original dict (no data loss)

Negative tests confirm that invariant violations raise informative ValueErrors.
"""

from __future__ import annotations

import json
import math
import pytest

from electoral.artifacts import (
    BaselinePortfolioData,
    EquilibriumData,
    LLMFineTuneData,
    MetricsTablesData,
    PredictionMarketData,
    SentimentData,
    ShockResponseData,
    SimulationData,
    SocialMediaSentimentData,
    StageArtifact,
    VoterPanelData,
)


# ── Helper ───────────────────────────────────────────────────────────────────


def assert_roundtrip(obj) -> dict:
    """Full round-trip check. Returns the serialized dict for further assertions."""
    obj.validate()
    d = obj.to_dict()
    # Must be JSON-serializable
    json.dumps(d)
    # Must reconstruct to identical state
    obj2 = type(obj).from_dict(d)
    obj2.validate()
    assert obj2.to_dict() == d, (
        f"Round-trip failed for {type(obj).__name__}: "
        f"to_dict() output changed after from_dict()"
    )
    return d


# ── Minimal valid instances ───────────────────────────────────────────────────

LAYER_WEIGHTS = {"lambda_1": 0.50, "lambda_2": 0.30, "lambda_3": 0.20}
RACE_IDS = ["african_american", "latino", "asian", "white", "other_race"]
RELIGION_IDS = ["evangelical", "catholic", "protestant", "secular", "jewish", "muslim", "other_rel"]
GENDER_IDS = ["women", "men", "other_gender"]

# Weights summing to exactly 1.0
RACE_WEIGHTS = {
    "african_american": 0.15,
    "latino": 0.12,
    "asian": 0.06,
    "white": 0.57,
    "other_race": 0.10,
}
assert abs(sum(RACE_WEIGHTS.values()) - 1.0) < 1e-9

# 5×5 diagonal covariance
COV_5X5 = [[0.01 if i == j else 0.0 for j in range(5)] for i in range(5)]


class TestStageArtifact:
    def test_roundtrip(self):
        obj = StageArtifact(
            stage="voter_panel",
            run_key="test_run",
            metadata={"seed": 42},
            data={"cycles": [2020], "n_rows_race": 100},
        )
        assert_roundtrip(obj)

    def test_empty_stage_raises(self):
        with pytest.raises(ValueError, match="stage"):
            StageArtifact(stage="", run_key="k", metadata={}, data={"x": 1}).validate()


class TestVoterPanelData:
    def test_roundtrip(self):
        obj = VoterPanelData(
            cycles=[2016, 2020],
            races=RACE_IDS,
            religions=RELIGION_IDS,
            genders=GENDER_IDS,
            n_rows_race=500,
            n_rows_religion=500,
            n_rows_gender=500,
            layer_weights=LAYER_WEIGHTS,
            source="ARDA+GSS+NEP",
        )
        assert_roundtrip(obj)

    def test_roundtrip_source_none(self):
        obj = VoterPanelData(
            cycles=[2020],
            races=RACE_IDS,
            religions=RELIGION_IDS,
            genders=GENDER_IDS,
            n_rows_race=0,
            n_rows_religion=0,
            n_rows_gender=0,
            layer_weights=LAYER_WEIGHTS,
            source=None,
        )
        d = assert_roundtrip(obj)
        assert d["source"] is None

    def test_duplicate_cycles_raises(self):
        with pytest.raises(ValueError, match="duplicate"):
            VoterPanelData(
                cycles=[2020, 2020],  # duplicate!
                races=RACE_IDS,
                religions=RELIGION_IDS,
                genders=GENDER_IDS,
                n_rows_race=0,
                n_rows_religion=0,
                n_rows_gender=0,
                layer_weights=LAYER_WEIGHTS,
                source=None,
            ).validate()

    def test_unsorted_cycles_raises(self):
        with pytest.raises(ValueError, match="strictly increasing"):
            VoterPanelData(
                cycles=[2020, 2016],  # reversed!
                races=RACE_IDS,
                religions=RELIGION_IDS,
                genders=GENDER_IDS,
                n_rows_race=0,
                n_rows_religion=0,
                n_rows_gender=0,
                layer_weights=LAYER_WEIGHTS,
                source=None,
            ).validate()

    def test_negative_n_rows_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            VoterPanelData(
                cycles=[2020],
                races=RACE_IDS,
                religions=RELIGION_IDS,
                genders=GENDER_IDS,
                n_rows_race=-1,
                n_rows_religion=0,
                n_rows_gender=0,
                layer_weights=LAYER_WEIGHTS,
                source=None,
            ).validate()

    def test_layer_weights_not_summing_raises(self):
        bad_weights = {"lambda_1": 0.5, "lambda_2": 0.5, "lambda_3": 0.5}  # sums to 1.5
        with pytest.raises(ValueError, match="sum"):
            VoterPanelData(
                cycles=[2020],
                races=RACE_IDS,
                religions=RELIGION_IDS,
                genders=GENDER_IDS,
                n_rows_race=0,
                n_rows_religion=0,
                n_rows_gender=0,
                layer_weights=bad_weights,
                source=None,
            ).validate()

    def test_non_canonical_race_raises(self):
        with pytest.raises(ValueError, match="canonical race"):
            VoterPanelData(
                cycles=[2020],
                races=["hispanic"],  # not in CANONICAL_RACES
                religions=RELIGION_IDS,
                genders=GENDER_IDS,
                n_rows_race=0,
                n_rows_religion=0,
                n_rows_gender=0,
                layer_weights=LAYER_WEIGHTS,
                source=None,
            ).validate()

    def test_non_canonical_religion_raises(self):
        with pytest.raises(ValueError, match="canonical religion"):
            VoterPanelData(
                cycles=[2020],
                races=RACE_IDS,
                religions=["buddhist"],  # not in CANONICAL_RELIGIONS
                genders=GENDER_IDS,
                n_rows_race=0,
                n_rows_religion=0,
                n_rows_gender=0,
                layer_weights=LAYER_WEIGHTS,
                source=None,
            ).validate()

    def test_non_canonical_gender_raises(self):
        with pytest.raises(ValueError, match="canonical gender"):
            VoterPanelData(
                cycles=[2020],
                races=RACE_IDS,
                religions=RELIGION_IDS,
                genders=["nonbinary"],  # not in CANONICAL_GENDERS
                n_rows_race=0,
                n_rows_religion=0,
                n_rows_gender=0,
                layer_weights=LAYER_WEIGHTS,
                source=None,
            ).validate()


class TestBaselinePortfolioData:
    def _make(self, **overrides):
        defaults = dict(
            method="cvxpy_dqcp",
            party="democrat",
            weights=RACE_WEIGHTS,
            mu_race={r: 0.50 for r in RACE_IDS},
            mu_religion={r: 0.50 for r in RELIGION_IDS},
            mu_gender={r: 0.50 for r in GENDER_IDS},
            mu_eff=0.535,
            layer_weights=LAYER_WEIGHTS,
            target=0.535,
        )
        defaults.update(overrides)
        return BaselinePortfolioData(**defaults)

    def test_roundtrip(self):
        assert_roundtrip(self._make())

    def test_weights_not_summing_raises(self):
        bad = {**RACE_WEIGHTS, "white": 0.99}  # sums > 1
        with pytest.raises(ValueError, match="sum"):
            self._make(weights=bad).validate()

    def test_invalid_party_raises(self):
        with pytest.raises(ValueError, match="party"):
            self._make(party="independent").validate()

    def test_mu_out_of_range_raises(self):
        bad_mu = {r: 0.50 for r in RACE_IDS}
        bad_mu["white"] = 1.5  # out of [0, 1]
        with pytest.raises(ValueError, match="mu_race"):
            self._make(mu_race=bad_mu).validate()

    def test_non_canonical_weight_key_raises(self):
        bad = {"hispanic": 1.0}  # not a canonical race
        with pytest.raises(ValueError, match="canonical race"):
            self._make(weights=bad).validate()

    def test_non_canonical_mu_religion_key_raises(self):
        bad = {"buddhist": 0.50}  # not a canonical religion
        with pytest.raises(ValueError, match="canonical religion"):
            self._make(mu_religion=bad).validate()

    def test_non_canonical_mu_gender_key_raises(self):
        bad = {"nonbinary": 0.50}  # not a canonical gender
        with pytest.raises(ValueError, match="canonical gender"):
            self._make(mu_gender=bad).validate()


class TestSentimentData:
    def test_roundtrip(self):
        obj = SentimentData(
            model="cardiffnlp/twitter-roberta-base-sentiment",
            shocks=["kavanaugh_2018", "metoo_2017"],
            scores={
                "evangelical": {"kavanaugh_2018": 0.25, "metoo_2017": -0.30},
                "secular": {"kavanaugh_2018": -0.40, "metoo_2017": 0.15},
            },
        )
        assert_roundtrip(obj)

    def test_score_out_of_range_raises(self):
        with pytest.raises(ValueError, match=r"\[-1"):
            SentimentData(
                model="m",
                shocks=["s1"],
                scores={"evangelical": {"s1": 1.5}},  # > 1.0
            ).validate()

    def test_missing_shock_in_bloc_raises(self):
        # scores["evangelical"] is missing "metoo_2017" — must be caught.
        with pytest.raises(ValueError, match="Missing") as exc_info:
            SentimentData(
                model="m",
                shocks=["kavanaugh_2018", "metoo_2017"],
                scores={"evangelical": {"kavanaugh_2018": 0.25}},  # metoo_2017 absent
            ).validate()
        assert "metoo_2017" in str(exc_info.value)
        assert "evangelical" in str(exc_info.value)

    def test_extra_shock_in_bloc_raises(self):
        # scores["secular"] has a shock ID not in self.shocks — must be caught.
        with pytest.raises(ValueError, match="Extra") as exc_info:
            SentimentData(
                model="m",
                shocks=["kavanaugh_2018"],
                scores={"secular": {"kavanaugh_2018": 0.10, "typo_shock": -0.05}},
            ).validate()
        assert "typo_shock" in str(exc_info.value)

    def test_duplicate_shocks_raises(self):
        with pytest.raises(ValueError, match="duplicate"):
            SentimentData(
                model="m",
                shocks=["kavanaugh_2018", "kavanaugh_2018"],
                scores={"evangelical": {"kavanaugh_2018": 0.25}},
            ).validate()

    def test_non_canonical_bloc_in_scores_raises(self):
        with pytest.raises(ValueError, match="canonical bloc"):
            SentimentData(
                model="m",
                shocks=["s1"],
                scores={"hispanic": {"s1": 0.10}},  # not a canonical bloc ID
            ).validate()


class TestLLMFineTuneData:
    def test_roundtrip(self):
        obj = LLMFineTuneData(
            base_model="mistralai/Mistral-7B-v0.3",
            lora_rank=16,
            n_examples=400,
            cycles_used=[2016, 2020],
            adapter_path=None,
        )
        assert_roundtrip(obj)

    def test_with_adapter_path(self):
        obj = LLMFineTuneData(
            base_model="mistralai/Mistral-7B-v0.3",
            lora_rank=32,
            n_examples=500,
            cycles_used=[2020],
            adapter_path="adapters/mistral-7b-electoral/",
        )
        d = assert_roundtrip(obj)
        assert d["adapter_path"] == "adapters/mistral-7b-electoral/"

    def test_zero_lora_rank_raises(self):
        with pytest.raises(ValueError, match="lora_rank"):
            LLMFineTuneData(
                base_model="m",
                lora_rank=0,
                n_examples=1,
                cycles_used=[2020],
                adapter_path=None,
            ).validate()

    def test_negative_lora_rank_raises(self):
        with pytest.raises(ValueError, match="lora_rank"):
            LLMFineTuneData(
                base_model="m",
                lora_rank=-8,
                n_examples=1,
                cycles_used=[2020],
                adapter_path=None,
            ).validate()

    def test_zero_n_examples_raises(self):
        with pytest.raises(ValueError, match="n_examples"):
            LLMFineTuneData(
                base_model="m",
                lora_rank=16,
                n_examples=0,
                cycles_used=[2020],
                adapter_path=None,
            ).validate()

    def test_negative_n_examples_raises(self):
        with pytest.raises(ValueError, match="n_examples"):
            LLMFineTuneData(
                base_model="m",
                lora_rank=16,
                n_examples=-1,
                cycles_used=[2020],
                adapter_path=None,
            ).validate()

    def test_duplicate_cycles_raises(self):
        with pytest.raises(ValueError, match="duplicate"):
            LLMFineTuneData(
                base_model="m",
                lora_rank=16,
                n_examples=100,
                cycles_used=[2020, 2020],
                adapter_path=None,
            ).validate()

    def test_unsorted_cycles_raises(self):
        with pytest.raises(ValueError, match="strictly increasing"):
            LLMFineTuneData(
                base_model="m",
                lora_rank=16,
                n_examples=100,
                cycles_used=[2020, 2016],
                adapter_path=None,
            ).validate()


class TestSocialMediaSentimentData:
    def test_roundtrip(self):
        obj = SocialMediaSentimentData(
            shock="kavanaugh_2018",
            platforms=["bluesky", "apify"],
            window_hours=72,
            scores={
                "bluesky": {"secular": -0.12, "evangelical": 0.08},
                "apify": {"evangelical": 0.35, "catholic": 0.10},
            },
            n_posts={"bluesky": 5000, "apify": 500},
            lagged_delta=None,
        )
        assert_roundtrip(obj)

    def test_with_lagged_delta(self):
        obj = SocialMediaSentimentData(
            shock="kavanaugh_2018",
            platforms=["bluesky"],
            window_hours=72,
            scores={"bluesky": {"secular": -0.10}},
            n_posts={"bluesky": 1000},
            lagged_delta={"evangelical": -0.04, "secular": 0.02},
        )
        d = assert_roundtrip(obj)
        assert d["lagged_delta"]["evangelical"] == pytest.approx(-0.04)

    def test_negative_n_posts_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            SocialMediaSentimentData(
                shock="s",
                platforms=["p"],
                window_hours=24,
                scores={"p": {"b": 0.0}},
                n_posts={"p": -1},
                lagged_delta=None,
            ).validate()

    def test_scores_extra_platform_raises(self):
        with pytest.raises(ValueError, match="scores keys must match platforms"):
            SocialMediaSentimentData(
                shock="s",
                platforms=["bluesky"],
                window_hours=24,
                scores={"bluesky": {"secular": 0.1}, "apify": {"secular": 0.2}},  # extra
                n_posts={"bluesky": 100},
                lagged_delta=None,
            ).validate()

    def test_n_posts_missing_platform_raises(self):
        with pytest.raises(ValueError, match="n_posts keys must match platforms"):
            SocialMediaSentimentData(
                shock="s",
                platforms=["bluesky", "apify"],
                window_hours=24,
                scores={"bluesky": {"secular": 0.1}, "apify": {"secular": 0.2}},
                n_posts={"bluesky": 100},  # missing "apify"
                lagged_delta=None,
            ).validate()


class TestPredictionMarketData:
    def _make(self, **overrides):
        defaults = dict(
            shock="kavanaugh_2018",
            party="democrat",
            pre_shock_prob=0.54,
            post_shock_1h=0.52,
            post_shock_24h=0.50,
            post_shock_72h=0.49,
            delta_prob=-0.04,
            sources=["polymarket", "predictit"],
            contract_ids={"polymarket": "0xabc", "predictit": "6274"},
            volume={"polymarket": 125000.0, "predictit": 50000.0},
        )
        defaults.update(overrides)
        return PredictionMarketData(**defaults)

    def test_roundtrip(self):
        assert_roundtrip(self._make())

    def test_roundtrip_all_none(self):
        obj = self._make(
            post_shock_1h=None,
            post_shock_24h=None,
            post_shock_72h=None,
            volume=None,
        )
        d = assert_roundtrip(obj)
        assert d["post_shock_1h"] is None

    def test_invalid_party_raises(self):
        with pytest.raises(ValueError, match="party"):
            self._make(party="libertarian").validate()

    def test_pre_shock_out_of_range_raises(self):
        with pytest.raises(ValueError, match="pre_shock_prob"):
            self._make(pre_shock_prob=1.5).validate()

    def test_contract_ids_missing_source_raises(self):
        with pytest.raises(ValueError, match="contract_ids keys must match sources"):
            self._make(
                sources=["polymarket", "predictit"],
                contract_ids={"polymarket": "0xabc"},  # missing "predictit"
            ).validate()

    def test_contract_ids_extra_source_raises(self):
        with pytest.raises(ValueError, match="contract_ids keys must match sources"):
            self._make(
                sources=["polymarket"],
                contract_ids={"polymarket": "0xabc", "kalshi": "XYZ"},  # extra "kalshi"
            ).validate()


class TestShockResponseData:
    def _make(self, **overrides):
        defaults = dict(
            shock="kavanaugh_2018",
            cycle=2018,
            party="democrat",
            delta_bins_race={r: "neutral" for r in RACE_IDS},
            delta_bins_religion={r: "neutral" for r in RELIGION_IDS},
            delta_bins_gender={g: "neutral" for g in GENDER_IDS},
            deltas_race={r: 0.0 for r in RACE_IDS},
            deltas_religion={r: 0.0 for r in RELIGION_IDS},
            deltas_gender={g: 0.0 for g in GENDER_IDS},
            delta_eff=0.0,
            covariance=COV_5X5,
            source="llm_unified",
        )
        defaults.update(overrides)
        return ShockResponseData(**defaults)

    def test_roundtrip(self):
        assert_roundtrip(self._make())

    def test_non_zero_deltas(self):
        assert_roundtrip(
            self._make(
                deltas_race={
                    "african_american": -0.012,
                    "latino": 0.0,
                    "asian": 0.0,
                    "white": 0.0,
                    "other_race": 0.0,
                },
                delta_bins_race={
                    "african_american": "slight_neg",
                    "latino": "neutral",
                    "asian": "neutral",
                    "white": "neutral",
                    "other_race": "neutral",
                },
                deltas_religion={r: 0.035 if r == "evangelical" else 0.0 for r in RELIGION_IDS},
                delta_bins_religion={
                    r: "mild_pos" if r == "evangelical" else "neutral" for r in RELIGION_IDS
                },
                deltas_gender={g: -0.070 if g == "women" else 0.0 for g in GENDER_IDS},
                delta_bins_gender={g: "mod_neg" if g == "women" else "neutral" for g in GENDER_IDS},
            )
        )

    def test_all_sources_valid(self):
        for src in ("llm_unified", "roberta_news_only", "roberta_social_only"):
            assert_roundtrip(self._make(source=src))

    def test_delta_out_of_range_raises(self):
        bad = {r: 0.0 for r in RACE_IDS}
        bad["white"] = -0.50  # outside [-0.15, 0.15]
        with pytest.raises(ValueError, match=r"\[-0.15"):
            self._make(deltas_race=bad).validate()

    def test_delta_non_finite_raises(self):
        bad = {r: 0.0 for r in RACE_IDS}
        bad["white"] = math.nan
        with pytest.raises(ValueError, match="finite"):
            self._make(deltas_race=bad).validate()

    def test_covariance_wrong_row_count_raises(self):
        short_cov = [[0.0] * 5 for _ in range(4)]  # 4 rows instead of 5
        with pytest.raises(ValueError, match=r"5×5"):
            self._make(covariance=short_cov).validate()

    def test_covariance_ragged_row_raises(self):
        ragged = [row[:] for row in COV_5X5]
        ragged[0] = ragged[0][:-1]  # row 0 one element short
        with pytest.raises(ValueError, match="elements"):
            self._make(covariance=ragged).validate()

    def test_invalid_source_raises(self):
        with pytest.raises(ValueError, match="source"):
            self._make(source="twitter_only").validate()

    def test_empty_source_raises(self):
        with pytest.raises(ValueError, match="source"):
            self._make(source="").validate()

    def test_invalid_bin_token_raises(self):
        bad = {r: "neutral" for r in RACE_IDS}
        bad["white"] = "very_negative"  # not a valid bin token
        with pytest.raises(ValueError, match="delta bin token"):
            self._make(delta_bins_race=bad).validate()

    def test_non_canonical_race_in_deltas_raises(self):
        bad = {r: 0.0 for r in RACE_IDS}
        bad["hispanic"] = 0.01  # not a canonical race bloc ID
        with pytest.raises(ValueError, match="canonical bloc"):
            self._make(deltas_race=bad).validate()

    def test_covariance_asymmetric_raises(self):
        asym = [row[:] for row in COV_5X5]
        asym[0][1] = 0.05  # cov[0][1] != cov[1][0]
        with pytest.raises(ValueError, match="symmetric"):
            self._make(covariance=asym).validate()

    def test_invalid_party_raises(self):
        with pytest.raises(ValueError, match="party"):
            self._make(party="libertarian").validate()


MU_SHIFTED_RACE = {r: 0.50 for r in RACE_IDS}


class TestEquilibriumData:
    def _make(self, **overrides):
        defaults = dict(
            method="cvxpy_dqcp",
            party="democrat",
            shock="kavanaugh_2018",
            weights=RACE_WEIGHTS,
            mu_shifted=MU_SHIFTED_RACE,
            feasible=True,
            target_met=False,
            target=0.535,
        )
        defaults.update(overrides)
        return EquilibriumData(**defaults)

    def test_roundtrip(self):
        assert_roundtrip(self._make())

    def test_roundtrip_no_shock(self):
        d = assert_roundtrip(self._make(shock=None, party="republican", target=0.520))
        assert d["shock"] is None

    def test_roundtrip_target_met(self):
        mu_high = {r: 0.60 for r in RACE_IDS}
        assert_roundtrip(self._make(mu_shifted=mu_high, feasible=True, target_met=True))

    def test_weights_not_summing_raises(self):
        bad = {**RACE_WEIGHTS, "white": 0.99}
        with pytest.raises(ValueError, match="sum"):
            self._make(weights=bad).validate()

    def test_invalid_party_raises(self):
        with pytest.raises(ValueError, match="party"):
            self._make(party="independent").validate()

    def test_mu_shifted_boundary_values(self):
        # 0.0 (total desertion) and 1.0 (unanimous support) are both valid vote shares
        assert_roundtrip(self._make(mu_shifted={r: 0.0 for r in RACE_IDS}))
        assert_roundtrip(self._make(mu_shifted={r: 1.0 for r in RACE_IDS}))

    def test_mu_shifted_out_of_range_raises(self):
        bad_mu = {**MU_SHIFTED_RACE, "white": 1.5}
        with pytest.raises(ValueError, match=r"mu_shifted\["):
            self._make(mu_shifted=bad_mu).validate()

    def test_mismatched_keys_raises(self):
        # mu_shifted missing one bloc that weights has
        short_mu = {r: 0.50 for r in RACE_IDS[:-1]}  # drops "other_race"
        with pytest.raises(ValueError, match="identical key sets"):
            self._make(mu_shifted=short_mu).validate()

    def test_extra_mu_key_raises(self):
        extra_mu = {**MU_SHIFTED_RACE, "evangelical": 0.30}  # extra key not in weights
        with pytest.raises(ValueError, match="identical key sets"):
            self._make(mu_shifted=extra_mu).validate()


class TestSimulationData:
    def _make(self, **overrides):
        defaults = dict(
            n_simulations=10_000,
            seed=42,
            win_probability=0.35,
            win_probability_low=0.05,
            win_probability_high=0.95,
            percentiles={r: [0.10, 0.25, 0.35, 0.45, 0.60] for r in RACE_IDS},
        )
        defaults.update(overrides)
        return SimulationData(**defaults)

    def test_roundtrip(self):
        d = assert_roundtrip(self._make())
        assert d["win_probability_low"] == pytest.approx(0.05)
        assert d["win_probability_high"] == pytest.approx(0.95)

    def test_empty_percentiles(self):
        assert_roundtrip(self._make(percentiles={}))

    def test_win_probability_boundary_values(self):
        assert_roundtrip(
            self._make(win_probability=0.0, win_probability_low=0.0, win_probability_high=0.0)
        )
        assert_roundtrip(
            self._make(win_probability=1.0, win_probability_low=1.0, win_probability_high=1.0)
        )

    def test_win_probability_out_of_range_raises(self):
        with pytest.raises(ValueError, match="win_probability"):
            self._make(win_probability=1.5).validate()

    def test_win_probability_negative_raises(self):
        with pytest.raises(ValueError, match="win_probability"):
            self._make(win_probability=-0.01).validate()

    def test_wrong_percentile_count_raises(self):
        with pytest.raises(ValueError, match="5 values"):
            self._make(percentiles={"evangelical": [0.1, 0.5, 0.9]}).validate()

    def test_percentile_value_out_of_range_raises(self):
        bad = {r: [0.10, 0.25, 0.35, 0.45, 0.60] for r in RACE_IDS}
        bad["white"] = [0.10, 0.25, 0.35, 0.45, 1.5]  # last value > 1
        with pytest.raises(ValueError, match=r"percentiles\["):
            self._make(percentiles=bad).validate()

    def test_percentiles_not_sorted_raises(self):
        bad = {r: [0.10, 0.25, 0.35, 0.45, 0.60] for r in RACE_IDS}
        bad["white"] = [0.10, 0.25, 0.80, 0.45, 0.60]  # p50 > p75 — not non-decreasing
        with pytest.raises(ValueError, match="non-decreasing"):
            self._make(percentiles=bad).validate()

    def test_zero_n_simulations_raises(self):
        with pytest.raises(ValueError, match="n_simulations"):
            self._make(n_simulations=0).validate()

    def test_win_probability_ci_inverted_raises(self):
        with pytest.raises(ValueError, match="win_probability_low"):
            self._make(win_probability_low=0.80, win_probability_high=0.20).validate()

    def test_win_probability_ci_boundary_values(self):
        assert_roundtrip(self._make(win_probability_low=0.0, win_probability_high=1.0))
        assert_roundtrip(
            self._make(win_probability=0.5, win_probability_low=0.5, win_probability_high=0.5)
        )

    def test_win_probability_ci_out_of_range_raises(self):
        with pytest.raises(ValueError, match="win_probability_low"):
            self._make(win_probability_low=-0.01, win_probability_high=0.5).validate()

    def test_win_probability_ci_high_out_of_range_raises(self):
        with pytest.raises(ValueError, match="win_probability_low"):
            self._make(win_probability_low=0.5, win_probability_high=1.5).validate()

    def test_win_probability_outside_ci_raises(self):
        with pytest.raises(ValueError, match="within CI bounds"):
            self._make(
                win_probability=0.10, win_probability_low=0.20, win_probability_high=0.80
            ).validate()


class TestMetricsTablesData:
    def _make(self, **overrides):
        defaults = dict(
            tables={
                "baseline_weights": {r: w for r, w in RACE_WEIGHTS.items()},
                "loco_mae": {"mean": 0.025, "evangelical": 0.038},
            }
        )
        defaults.update(overrides)
        return MetricsTablesData(**defaults)

    def test_roundtrip(self):
        assert_roundtrip(self._make())

    def test_empty_tables_valid(self):
        assert_roundtrip(self._make(tables={}))

    def test_json_scalars_valid(self):
        assert_roundtrip(
            self._make(
                tables={
                    "int_val": 42,
                    "float_val": 3.14,
                    "bool_val": True,
                    "null_val": None,
                    "str_val": "result",
                }
            )
        )

    def test_nested_structures_valid(self):
        assert_roundtrip(
            self._make(
                tables={
                    "nested": {"a": [1, 2, 3], "b": {"c": 0.5}},
                    "list_of_lists": [[0.1, 0.9], [0.4, 0.6]],
                }
            )
        )

    def test_empty_key_raises(self):
        with pytest.raises(ValueError, match="empty key"):
            self._make(tables={"": {"x": 1}}).validate()

    def test_non_serializable_raises(self):
        import datetime

        with pytest.raises(ValueError, match="JSON-serializable"):
            self._make(tables={"bad": datetime.datetime.now()}).validate()

    def test_error_message_names_key(self):
        import datetime

        with pytest.raises(ValueError, match="bad_table"):
            self._make(tables={"bad_table": datetime.date.today()}).validate()
