"""Tests for electoral/metrics/synthetic_distribution.py (Phase 5, Step 5.4).

WHY THIS EXISTS: the module it tests is itself a test of sorts -- a
post-generation audit that verdicts whether a synthetic corpus's empirical
bin-magnitude distribution matches the target in configs/synthetic_events.json.
The task that built it required a self-test: feed it a deliberately skewed
batch and a realistic one, confirm it flags the first and passes the second.
That requirement is encoded here, plus the boundary/plumbing tests a drift
guard of this kind needs (matching the discipline of
tests/test_bin_midpoints_sync.py, which this module's `delta_to_bin` addition
to electoral/core/types.py should eventually be folded into).
"""

from __future__ import annotations

import pytest

from electoral.core.types import (
    BIN_MIDPOINTS,
    CANONICAL_GENDERS,
    CANONICAL_RACES,
    CANONICAL_RELIGIONS,
    DELTA_BINS,
    bin_to_magnitude_category,
    delta_to_bin,
)
from electoral.metrics.synthetic_distribution import (
    DEFAULT_TARGET_CONFIG_PATH,
    _CRAMERS_V_WARN,
    audit_corpus,
    extract_bloc_bins,
    load_target,
    measure_distribution,
)


class TestDeltaToBinBoundaries:
    """delta_to_bin() / bin_to_magnitude_category() are new additions to
    electoral/core/types.py, needed because no reverse-of-bin_to_delta
    classifier existed anywhere in the repo before this task (confirmed by
    repo-wide search). Boundary values are exact ranges from the DELTA_BINS
    comment, not derived by bisecting adjacent midpoints (they're not
    symmetric), so every edge is worth pinning down explicitly.
    """

    def test_every_midpoint_round_trips_to_its_own_token(self):
        for token, midpoint in BIN_MIDPOINTS.items():
            assert delta_to_bin(midpoint) == token

    @pytest.mark.parametrize(
        "delta,expected",
        [
            (0.0, "neutral"),
            (0.00125, "neutral"),
            (-0.00125, "neutral"),
            (0.00126, "slight_pos"),
            (-0.00126, "slight_neg"),
            (0.005, "slight_pos"),
            (0.00501, "mild_pos"),
            (0.0125, "mild_pos"),
            (0.01251, "mod_pos"),
            (0.0225, "mod_pos"),
            (0.02251, "strong_pos"),
            (0.0375, "strong_pos"),
        ],
    )
    def test_exact_boundary_edges(self, delta, expected):
        assert delta_to_bin(delta) == expected

    def test_out_of_range_values_clamp_to_nearest_extreme_bin(self):
        # Real measured_delta values (data/ground_truth/panel_deltas.json)
        # are empirical and unclipped, unlike LLM output -- this function is
        # also used to classify those, so it must not raise on them.
        assert delta_to_bin(1.0) == "strong_pos"
        assert delta_to_bin(-1.0) == "strong_neg"

    def test_bin_to_magnitude_category_covers_every_token(self):
        for token in DELTA_BINS:
            category = bin_to_magnitude_category(token)
            assert category in {"neutral", "slight", "mild", "moderate", "strong"}
        # The one token prefix that doesn't match its category name directly.
        assert bin_to_magnitude_category("mod_pos") == "moderate"
        assert bin_to_magnitude_category("mod_neg") == "moderate"

    def test_unknown_token_raises(self):
        with pytest.raises(ValueError, match="Unknown delta bin token"):
            bin_to_magnitude_category("extremely_strong_pos")


RACES = CANONICAL_RACES
RELIGIONS = CANONICAL_RELIGIONS
GENDERS = CANONICAL_GENDERS
ALL_BLOCS = list(RACES) + list(RELIGIONS) + list(GENDERS)
N_BLOCS = len(ALL_BLOCS)  # 15


def _record(shock_id: str, tokens: list[str]) -> dict:
    """Build one record whose 15 bloc-level bins are exactly `tokens`, in
    ALL_BLOCS order, split across the three delta_bins_* dicts like the real
    generation schema (scripts/synthetic/generate_deepseek.py).
    """
    assert len(tokens) == N_BLOCS
    by_bloc = dict(zip(ALL_BLOCS, tokens))
    return {
        "shock_id": shock_id,
        "delta_bins_race": {b: by_bloc[b] for b in RACES},
        "delta_bins_religion": {b: by_bloc[b] for b in RELIGIONS},
        "delta_bins_gender": {b: by_bloc[b] for b in GENDERS},
    }


class TestLoadTarget:
    def test_reads_the_real_config(self):
        # Panel-measured values (see scripts/synthetic/check_panel_target_calibration.py),
        # not the original intuition-derived 30/45/18/5/2 -- this test asserting
        # against a stale copy is exactly the drift this file exists to catch.
        target = load_target(DEFAULT_TARGET_CONFIG_PATH)
        assert target == {
            "neutral": 0.34,
            "slight": 0.31,
            "mild": 0.28,
            "moderate": 0.06,
            "strong": 0.01,
        }

    def test_sums_to_one(self):
        target = load_target(DEFAULT_TARGET_CONFIG_PATH)
        assert sum(target.values()) == pytest.approx(1.0)

    def test_missing_category_raises(self, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text('{"_meta": {"bin_distribution_target": {"neutral": 1.0}}}')
        with pytest.raises(ValueError, match="missing categories"):
            load_target(bad)

    def test_target_not_summing_to_one_raises(self, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text(
            '{"_meta": {"bin_distribution_target": '
            '{"neutral": 0.5, "slight": 0.5, "mild": 0.5, "moderate": 0.0, "strong": 0.0}}}'
        )
        with pytest.raises(ValueError, match="sums to"):
            load_target(bad)


class TestExtractBlocBins:
    def test_flattens_all_three_strata(self):
        rec = _record("s1", ["neutral"] * N_BLOCS)
        bins = extract_bloc_bins(rec)
        assert set(bins) == set(ALL_BLOCS)
        assert all(v == "neutral" for v in bins.values())

    def test_missing_strata_key_does_not_crash(self):
        rec = {"shock_id": "s1", "delta_bins_race": {"white": "neutral"}}
        bins = extract_bloc_bins(rec)
        assert bins == {"white": "neutral"}


class TestMeasureDistributionBasics:
    def test_exact_target_match_passes_with_low_cramers_v(self):
        # All 5 categories need positive expected count for chi-square/V to
        # be valid at all (Cochran's rule) -- use a realistic-shaped target,
        # not a degenerate 2-category one. Deliberately NOT the real config's
        # current value (see TestLoadTarget.test_reads_the_real_config for
        # that) -- this test is about measure_distribution()'s generic
        # behavior, not config sync, so any plausible target works here.
        target = {"neutral": 0.30, "slight": 0.45, "mild": 0.18, "moderate": 0.05, "strong": 0.02}
        tokens = (
            ["neutral"] * 300
            + ["slight_pos"] * 450
            + ["mild_pos"] * 180
            + ["mod_pos"] * 50
            + ["strong_pos"] * 20
        )
        assert len(tokens) == 1000
        check = measure_distribution(tokens, target, label="exact")
        assert check.verdict == "PASS", check.reasons
        assert check.chi_square_valid
        assert check.cramers_v < 0.01  # exact match -> ~0 effect size

    def test_empty_input_does_not_crash(self):
        target = load_target(DEFAULT_TARGET_CONFIG_PATH)
        check = measure_distribution([], target, label="empty")
        assert check.n == 0
        assert (
            check.verdict == "WARN"
        )  # nothing measured -- neither confirmed-fine nor confirmed-bad
        assert not check.chi_square_valid
        assert "nothing measured" in check.reasons[0]


class TestSelfTestSkewedVsRealistic:
    """The task's explicit self-test requirement: a deliberately skewed batch
    must be flagged, a realistic one must pass. Both fixtures are
    deterministic (no RNG) so the test can't be flaky.

    TARGET here is an illustrative example, independent of the real config
    (see TestLoadTarget.test_reads_the_real_config for the current real
    value) -- the point of this class is that measure_distribution() PASSes
    a well-calibrated batch and FAILs a badly-calibrated one against WHATEVER
    target it's given, not that this specific target matches production.
    """

    TARGET = {"neutral": 0.30, "slight": 0.45, "mild": 0.18, "moderate": 0.05, "strong": 0.02}

    def _realistic_batch(self) -> list[dict]:
        # 200 records x 15 blocs = 3000 bin observations, hand-built to land
        # within a point or two of every target category -- not sampled, so
        # the test result doesn't depend on a random seed.
        # Per-record composition (15 bins): 4-5 neutral, 6-7 slight, 2-3 mild,
        # 0-1 moderate, rarely strong -- rotated across records so the corpus
        # aggregate tracks the target without every record being identical
        # (a corpus of 200 identical records would itself be a red flag a
        # real audit should care about, even though it isn't what this
        # particular check measures).
        patterns = [
            ["neutral"] * 5
            + ["slight_pos"] * 6
            + ["slight_neg"] * 1
            + ["mild_pos"] * 2
            + ["mild_neg"] * 1,
            ["neutral"] * 4
            + ["slight_pos"] * 4
            + ["slight_neg"] * 3
            + ["mild_pos"] * 2
            + ["mod_pos"] * 1
            + ["mild_neg"] * 1,
            ["neutral"] * 5
            + ["slight_pos"] * 5
            + ["slight_neg"] * 2
            + ["mild_pos"] * 1
            + ["mild_neg"] * 1
            + ["strong_pos"] * 1,
            ["neutral"] * 4
            + ["slight_pos"] * 7
            + ["slight_neg"] * 0
            + ["mild_pos"] * 2
            + ["mild_neg"] * 2,
        ]
        assert all(len(p) == N_BLOCS for p in patterns)
        records = []
        for i in range(200):
            pattern = patterns[i % len(patterns)]
            # Cyclic-shift which bloc gets which position, per record. Without
            # this, whichever bloc sits first in ALL_BLOCS gets index 0 of
            # every pattern forever (here, always "neutral") -- a fixture
            # artifact, not a real per-bloc skew, and it previously produced
            # a false per-bloc FAIL that had nothing to do with the thing
            # being tested.
            shifted = pattern[i % N_BLOCS :] + pattern[: i % N_BLOCS]
            records.append(_record(f"realistic_shock_{i % 20}", shifted))
        return records

    def _skewed_batch(self) -> list[dict]:
        # Every record spreads its 15 bins roughly evenly across all 9 tokens
        # -- exactly the "corpus spread evenly across bins" failure mode the
        # task's WHY section describes: correct labels, wrong base rate.
        tokens_9 = [
            "strong_neg",
            "mod_neg",
            "mild_neg",
            "slight_neg",
            "neutral",
            "slight_pos",
            "mild_pos",
            "mod_pos",
            "strong_pos",
        ]
        pattern = (tokens_9 * ((N_BLOCS // len(tokens_9)) + 1))[:N_BLOCS]
        assert len(pattern) == N_BLOCS
        records = [_record(f"skewed_shock_{i % 20}", pattern) for i in range(200)]
        return records

    def test_realistic_batch_passes_overall(self):
        report = audit_corpus(self._realistic_batch(), target=self.TARGET)
        assert report.overall.verdict == "PASS", report.overall.reasons
        assert report.overall.chi_square_valid
        # NOT chi_square_p > 0.05: raw p is not the verdict driver (see
        # tolerance rationale #2) -- at this n, p is tiny even for a
        # practically-negligible deviation. Cramer's V is what should be small.
        assert report.overall.cramers_v < _CRAMERS_V_WARN

    def test_skewed_batch_fails_overall(self):
        report = audit_corpus(self._skewed_batch(), target=self.TARGET)
        assert report.overall.verdict == "FAIL", report.overall.reasons
        assert "strong" in report.overall.offending_bins
        assert "moderate" in report.overall.offending_bins
        assert report.overall.chi_square_valid
        assert report.overall.chi_square_p < 0.01

    def test_skewed_batch_overstates_strong_and_moderate(self):
        report = audit_corpus(self._skewed_batch(), target=self.TARGET)
        # 2/9 tokens are strong, 2/9 are moderate -> ~22.2% each, vs 2%/5% target.
        assert report.overall.observed_pct["strong"] > 15.0
        assert report.overall.observed_pct["moderate"] > 15.0

    def test_realistic_batch_per_bloc_and_per_shock_also_pass(self):
        report = audit_corpus(self._realistic_batch(), target=self.TARGET)
        assert report.worst_verdict in ("PASS", "WARN")  # allow minor per-bloc noise, not FAIL
        n_fail = sum(1 for c in report.per_bloc.values() if c.verdict == "FAIL")
        assert n_fail == 0, {
            b: c.reasons for b, c in report.per_bloc.items() if c.verdict == "FAIL"
        }

    def test_skewed_batch_every_bloc_fails(self):
        report = audit_corpus(self._skewed_batch(), target=self.TARGET)
        assert all(c.verdict == "FAIL" for c in report.per_bloc.values())
        assert report.worst_verdict == "FAIL"


class TestAggregateCanHidePerBlocSkew:
    """The exact scenario the task's WHY section warns about: one bloc is
    badly skewed but 14 well-calibrated blocs dilute it to invisibility in
    the corpus-wide aggregate. Confirms per-bloc breakdown catches what
    overall alone would miss.
    """

    def test_one_bad_bloc_among_many_good_ones(self):
        # Illustrative target/tokens pair, deliberately self-consistent and
        # independent of the real config (see TestLoadTarget for that) --
        # this test is about the per-bloc-vs-aggregate mechanic, not config sync.
        target = {"neutral": 0.30, "slight": 0.45, "mild": 0.18, "moderate": 0.05, "strong": 0.02}
        good_tokens = (
            ["neutral"] * 30
            + ["slight_pos"] * 45
            + ["mild_pos"] * 18
            + ["mod_pos"] * 5
            + ["strong_pos"] * 2
        )
        assert len(good_tokens) == 100
        bad_tokens = ["strong_pos"] * 100  # one bloc, always maximal

        records = []
        for i in range(100):
            by_bloc = {}
            for j, bloc in enumerate(ALL_BLOCS):
                if bloc == "white":
                    by_bloc[bloc] = bad_tokens[i]
                else:
                    by_bloc[bloc] = good_tokens[(i + j) % len(good_tokens)]
            records.append(
                {
                    "shock_id": f"shock_{i % 10}",
                    "delta_bins_race": {b: by_bloc[b] for b in RACES},
                    "delta_bins_religion": {b: by_bloc[b] for b in RELIGIONS},
                    "delta_bins_gender": {b: by_bloc[b] for b in GENDERS},
                }
            )

        report = audit_corpus(records, target=target)
        # "white" alone should FAIL hard...
        assert report.per_bloc["white"].verdict == "FAIL"
        assert report.per_bloc["white"].observed_pct["strong"] == pytest.approx(100.0)
        # ...while the corpus-wide aggregate (14/15 blocs fine) is much closer
        # to target, demonstrating exactly why per-bloc breakdown is required
        # and not optional.
        assert (
            report.overall.observed_pct["strong"] < report.per_bloc["white"].observed_pct["strong"]
        )
        assert report.worst_verdict == "FAIL"
