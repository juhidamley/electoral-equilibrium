"""Tests for electoral/core/schema.py — all five validation helpers.

Structure per helper:
  - ≥2 positive tests: valid inputs must pass silently (no exception)
  - ≥2 negative tests: invalid inputs must raise ValueError whose message
    names the context (class) and field so the caller can diagnose quickly.

Error message assertions use re.search so they are robust to minor rephrasing
but strict about the presence of the class name, field name, and key facts.
"""

from __future__ import annotations

import re
import pytest

from electoral.core.schema import (
    assert_required_keys,
    assert_shares_sum_to_one,
    assert_sorted_unique,
    assert_unique,
    assert_valid_share,
)


# ── assert_required_keys ──────────────────────────────────────────────────────


class TestAssertRequiredKeys:

    # ── positive ─────────────────────────────────────────────────────────────

    def test_all_required_keys_present(self):
        assert_required_keys(
            {"stage": "voter_panel", "run_key": "base_2026", "data": {}},
            ["stage", "run_key", "data"],
            context="StageArtifact",
        )

    def test_extra_keys_are_ignored(self):
        # Payload may contain keys beyond what is required.
        assert_required_keys(
            {"cycles": [2020], "races": [], "extra_field": "ok"},
            ["cycles", "races"],
            context="VoterPanelData",
        )

    def test_empty_required_list_always_passes(self):
        assert_required_keys({}, [], context="AnyClass")

    def test_single_required_key_present(self):
        assert_required_keys({"weights": {"a": 1.0}}, ["weights"], context="BaselinePortfolioData")

    # ── negative ─────────────────────────────────────────────────────────────

    def test_one_missing_key_raises(self):
        with pytest.raises(ValueError, match="StageArtifact") as exc_info:
            assert_required_keys(
                {"stage": "voter_panel"},
                ["stage", "run_key", "data"],
                context="StageArtifact",
            )
        msg = str(exc_info.value)
        assert "run_key" in msg, f"Missing field not named in error: {msg}"
        assert "data" in msg, f"Missing field not named in error: {msg}"

    def test_multiple_missing_keys_raises(self):
        with pytest.raises(ValueError, match="BaselinePortfolioData") as exc_info:
            assert_required_keys(
                {},
                ["weights", "mu_eff", "target"],
                context="BaselinePortfolioData",
            )
        msg = str(exc_info.value)
        # All missing keys must appear in the message.
        for key in ("weights", "mu_eff", "target"):
            assert key in msg, f"Key {key!r} not named in error: {msg}"

    def test_empty_payload_missing_all_keys_raises(self):
        with pytest.raises(ValueError, match="VoterPanelData"):
            assert_required_keys({}, ["cycles", "races", "genders"], context="VoterPanelData")

    def test_error_message_lists_present_keys(self):
        with pytest.raises(ValueError) as exc_info:
            assert_required_keys(
                {"present_key": 1},
                ["present_key", "absent_key"],
                context="SomeClass",
            )
        # The message should mention what WAS present to help diagnosis.
        assert "absent_key" in str(exc_info.value)


# ── assert_unique ─────────────────────────────────────────────────────────────


class TestAssertUnique:

    # ── positive ─────────────────────────────────────────────────────────────

    def test_all_unique_strings(self):
        assert_unique(
            ["african_american", "latino", "asian", "white", "other_race"],
            name="races",
            context="VoterPanelData",
        )

    def test_all_unique_integers(self):
        assert_unique([2000, 2004, 2008, 2012, 2016, 2020], name="cycles", context="VoterPanelData")

    def test_single_element_is_unique(self):
        assert_unique(["only_one"], name="races", context="VoterPanelData")

    def test_empty_list_is_unique(self):
        assert_unique([], name="items", context="AnyClass")

    # ── negative ─────────────────────────────────────────────────────────────

    def test_one_duplicate_raises(self):
        with pytest.raises(ValueError, match="VoterPanelData") as exc_info:
            assert_unique(
                ["evangelical", "catholic", "evangelical"],
                name="religions",
                context="VoterPanelData",
            )
        msg = str(exc_info.value)
        assert "religions" in msg, f"Field name not in error: {msg}"
        assert "evangelical" in msg, f"Duplicate value not named in error: {msg}"

    def test_multiple_duplicates_raises(self):
        with pytest.raises(ValueError, match="ShockResponseData") as exc_info:
            assert_unique(
                ["african_american", "white", "african_american", "white"],
                name="blocs",
                context="ShockResponseData",
            )
        msg = str(exc_info.value)
        assert "blocs" in msg
        assert "african_american" in msg or "white" in msg

    def test_adjacent_duplicates_raises(self):
        with pytest.raises(ValueError):
            assert_unique([1, 2, 2, 3], name="cycles", context="VoterPanelData")

    def test_non_adjacent_duplicates_raises(self):
        with pytest.raises(ValueError, match="duplicate"):
            assert_unique([2016, 2020, 2016], name="cycles", context="VoterPanelData")


# ── assert_sorted_unique ──────────────────────────────────────────────────────


class TestAssertSortedUnique:

    # ── positive ─────────────────────────────────────────────────────────────

    def test_strictly_increasing_integers(self):
        assert_sorted_unique(
            [2000, 2004, 2008, 2016, 2020], name="cycles", context="VoterPanelData"
        )

    def test_two_element_sorted(self):
        assert_sorted_unique([2016, 2020], name="cycles", context="VoterPanelData")

    def test_single_element(self):
        assert_sorted_unique([2020], name="cycles", context="VoterPanelData")

    def test_empty_list(self):
        assert_sorted_unique([], name="cycles", context="VoterPanelData")

    # ── negative ─────────────────────────────────────────────────────────────

    def test_unsorted_raises(self):
        with pytest.raises(ValueError, match="VoterPanelData") as exc_info:
            assert_sorted_unique([2020, 2016, 2024], name="cycles", context="VoterPanelData")
        msg = str(exc_info.value)
        assert "cycles" in msg, f"Field name not in error: {msg}"
        # The offending adjacent pair (2020 >= 2016) should be identified.
        assert "2020" in msg, f"Offending value not in error: {msg}"

    def test_duplicate_raises_via_uniqueness_check(self):
        # Duplicate is caught before the sort check; error mentions "duplicate".
        with pytest.raises(ValueError, match="duplicate"):
            assert_sorted_unique([2016, 2016, 2020], name="cycles", context="VoterPanelData")

    def test_reversed_list_raises(self):
        with pytest.raises(ValueError, match="strictly increasing"):
            assert_sorted_unique([2024, 2020, 2016], name="cycles", context="VoterPanelData")

    def test_equal_adjacent_raises(self):
        # Equal adjacent elements violate strict increase (also caught as duplicate).
        with pytest.raises(ValueError):
            assert_sorted_unique([1, 2, 2, 3], name="cycles", context="VoterPanelData")


# ── assert_valid_share ────────────────────────────────────────────────────────


class TestAssertValidShare:

    # ── positive ─────────────────────────────────────────────────────────────

    def test_zero_is_valid_boundary(self):
        assert_valid_share(0.0, name="vote_share", context="VoterPanelData")

    def test_one_is_valid_boundary(self):
        assert_valid_share(1.0, name="stratum_share", context="VoterPanelData")

    def test_interior_value_valid(self):
        assert_valid_share(0.52, name="mu_eff", context="BaselinePortfolioData")

    def test_small_positive_valid(self):
        assert_valid_share(0.01, name="weight", context="BaselinePortfolioData")

    def test_large_interior_valid(self):
        assert_valid_share(0.999, name="win_probability", context="SimulationData")

    # ── negative ─────────────────────────────────────────────────────────────

    def test_negative_value_raises(self):
        with pytest.raises(ValueError, match="VoterPanelData") as exc_info:
            assert_valid_share(-0.01, name="vote_share", context="VoterPanelData")
        msg = str(exc_info.value)
        assert "vote_share" in msg, f"Field name not in error: {msg}"
        assert "-0.01" in msg, f"Bad value not in error: {msg}"

    def test_greater_than_one_raises(self):
        with pytest.raises(ValueError, match="BaselinePortfolioData") as exc_info:
            assert_valid_share(1.001, name="stratum_share", context="BaselinePortfolioData")
        msg = str(exc_info.value)
        assert "stratum_share" in msg
        assert "1.001" in msg

    def test_large_negative_raises(self):
        with pytest.raises(ValueError, match=r"\[0\.0, 1\.0\]"):
            assert_valid_share(-99.0, name="weight", context="EquilibriumData")

    def test_value_just_above_one_raises(self):
        with pytest.raises(ValueError):
            assert_valid_share(1.0 + 1e-9, name="prob", context="SimulationData")


# ── assert_shares_sum_to_one ──────────────────────────────────────────────────


class TestAssertSharesSumToOne:

    # ── positive ─────────────────────────────────────────────────────────────

    def test_five_race_weights_exact(self):
        assert_shares_sum_to_one(
            {
                "african_american": 0.15,
                "latino": 0.12,
                "asian": 0.06,
                "white": 0.57,
                "other_race": 0.10,
            },
            context="BaselinePortfolioData.weights",
        )

    def test_two_blocs_exact(self):
        assert_shares_sum_to_one(
            {"african_american": 0.15, "white": 0.85},
            context="BaselinePortfolioData.weights",
        )

    def test_sum_within_default_tolerance(self):
        # Floating-point arithmetic often introduces sub-1e-6 rounding errors.
        val = 1.0 / 3.0
        assert_shares_sum_to_one(
            {"a": val, "b": val, "c": val},
            context="VoterPanelData.stratum_share",
        )

    def test_sum_within_custom_tolerance(self):
        assert_shares_sum_to_one(
            {"evangelical": 0.2401, "secular": 0.7600},
            context="VoterPanelData",
            tol=1e-3,
        )

    def test_single_bloc_at_one(self):
        assert_shares_sum_to_one({"only_bloc": 1.0}, context="TestClass")

    # ── negative ─────────────────────────────────────────────────────────────

    def test_sum_below_one_raises(self):
        with pytest.raises(ValueError, match="BaselinePortfolioData") as exc_info:
            assert_shares_sum_to_one(
                {"african_american": 0.15, "white": 0.57},
                context="BaselinePortfolioData",
            )
        msg = str(exc_info.value)
        # Message must report the actual sum so the caller can debug.
        assert re.search(r"0\.7[0-9]+", msg), f"Actual sum not in error: {msg}"

    def test_sum_above_one_raises(self):
        with pytest.raises(ValueError, match="VoterPanelData") as exc_info:
            assert_shares_sum_to_one(
                {"evangelical": 0.60, "catholic": 0.55},
                context="VoterPanelData",
            )
        msg = str(exc_info.value)
        assert re.search(r"1\.1[0-9]*", msg), f"Actual sum not in error: {msg}"

    def test_empty_dict_raises(self):
        with pytest.raises(ValueError, match="empty"):
            assert_shares_sum_to_one({}, context="BaselinePortfolioData.weights")

    def test_sum_below_tolerance_boundary_raises(self):
        # Sum = 1.0 - 2e-6, which exceeds the default tol of 1e-6.
        with pytest.raises(ValueError):
            assert_shares_sum_to_one(
                {"a": 0.5, "b": 0.5 - 2e-6},
                context="EquilibriumData.weights",
            )

    def test_error_message_includes_keys(self):
        with pytest.raises(ValueError) as exc_info:
            assert_shares_sum_to_one(
                {"women": 0.40, "men": 0.40},
                context="VoterPanelData.stratum_share",
            )
        msg = str(exc_info.value)
        # At least one bloc key should appear to aid debugging.
        assert "men" in msg or "women" in msg, f"No keys in error message: {msg}"
