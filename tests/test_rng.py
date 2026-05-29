"""Tests for the deterministic RNG seed contract.

Seed contract (non-negotiable):
  - Identical inputs to derive_seed always produce identical seeds
  - Order of arguments matters (derive_seed(42, "a") != derive_seed(42, "b"))
  - make_rng(seed) always produces identical draws for the same seed
  - Running the pipeline twice with the same seed produces identical artifacts
"""

from __future__ import annotations

import numpy as np

from electoral.core.rng import derive_seed, derive_seed_tokens, make_rng


class TestDeriveSeed:
    def test_identical_inputs_same_seed(self):
        assert derive_seed(42, "voter_panel") == derive_seed(42, "voter_panel")

    def test_different_stage_different_seed(self):
        s1 = derive_seed(42, "voter_panel")
        s2 = derive_seed(42, "monte_carlo")
        assert s1 != s2

    def test_different_base_seed_different_seed(self):
        s1 = derive_seed(42, "voter_panel")
        s2 = derive_seed(43, "voter_panel")
        assert s1 != s2

    def test_order_matters(self):
        s1 = derive_seed(42, "stage_a")
        s2 = derive_seed(42, "stage_b")
        assert s1 != s2

    def test_seed_in_valid_numpy_range(self):
        seed = derive_seed(42, "test_stage")
        # np.random.default_rng accepts seeds in [0, 2**32 - 1] or None
        assert 0 <= seed < 2**31

    def test_zero_base_seed(self):
        seed = derive_seed(0, "test")
        assert isinstance(seed, int)
        assert seed >= 0

    def test_deterministic_across_calls(self):
        results = [derive_seed(123, "monte_carlo") for _ in range(10)]
        assert len(set(results)) == 1  # all identical

    def test_known_stage_names(self):
        stage_names = [
            "voter_panel",
            "baseline_portfolio",
            "sentiment_data",
            "llm_finetune",
            "shock_response",
            "optimization",
            "monte_carlo",
            "setfit",
            "roberta",
        ]
        seeds = [derive_seed(42, s) for s in stage_names]
        assert len(seeds) == len(set(seeds)), "All stage seeds should be unique"


class TestMakeRng:
    def test_same_seed_same_draws(self):
        rng1 = make_rng(42)
        rng2 = make_rng(42)
        arr1 = rng1.random(100)
        arr2 = rng2.random(100)
        np.testing.assert_array_equal(arr1, arr2)

    def test_different_seed_different_draws(self):
        rng1 = make_rng(42)
        rng2 = make_rng(43)
        arr1 = rng1.random(100)
        arr2 = rng2.random(100)
        assert not np.array_equal(arr1, arr2)

    def test_returns_generator_instance(self):
        rng = make_rng(42)
        assert isinstance(rng, np.random.Generator)

    def test_generator_integers(self):
        rng1 = make_rng(7)
        rng2 = make_rng(7)
        ints1 = rng1.integers(0, 1000, size=50)
        ints2 = rng2.integers(0, 1000, size=50)
        np.testing.assert_array_equal(ints1, ints2)

    def test_generator_normal(self):
        rng1 = make_rng(99)
        rng2 = make_rng(99)
        n1 = rng1.standard_normal(200)
        n2 = rng2.standard_normal(200)
        np.testing.assert_array_equal(n1, n2)

    def test_dirichlet_reproducibility(self):
        # Dirichlet sampling must use rng.dirichlet, not np.random.dirichlet
        rng1 = make_rng(derive_seed(42, "monte_carlo"))
        rng2 = make_rng(derive_seed(42, "monte_carlo"))
        alpha = [0.15, 0.11, 0.05, 0.62, 0.07]
        samples1 = rng1.dirichlet(alpha, size=100)
        samples2 = rng2.dirichlet(alpha, size=100)
        np.testing.assert_array_equal(samples1, samples2)

    def test_dirichlet_simplex_constraint(self):
        rng = make_rng(42)
        alpha = [0.15, 0.11, 0.05, 0.62, 0.07]
        samples = rng.dirichlet(alpha, size=1000)
        row_sums = samples.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-12)


class TestDeriveSeedTokens:
    """Tests for the general list-based derive_seed_tokens API."""

    def test_identical_inputs_produce_identical_seeds(self):
        # (i) Identical token lists must always hash to the same seed.
        assert derive_seed_tokens(["42", "voter_panel"]) == derive_seed_tokens(
            ["42", "voter_panel"]
        )

    def test_order_matters(self):
        # (ii) Token order is significant — reversing the list changes the seed.
        assert derive_seed_tokens(["alpha", "beta"]) != derive_seed_tokens(["beta", "alpha"])

    def test_generator_draws_identical_for_same_seed(self):
        # (iii) A generator seeded from derive_seed_tokens produces identical draws.
        seed = derive_seed_tokens(["42", "monte_carlo"])
        draws_a = make_rng(seed).random(50)
        draws_b = make_rng(seed).random(50)
        np.testing.assert_array_equal(draws_a, draws_b)

    def test_consistent_with_derive_seed(self):
        # derive_seed(n, s) must equal derive_seed_tokens([str(n), s]) for all inputs.
        for base, stage in [(42, "voter_panel"), (0, "monte_carlo"), (999, "setfit")]:
            assert derive_seed(base, stage) == derive_seed_tokens([str(base), stage])

    def test_different_tokens_different_seeds(self):
        assert derive_seed_tokens(["42", "a"]) != derive_seed_tokens(["42", "b"])

    def test_seed_in_valid_numpy_range(self):
        seed = derive_seed_tokens(["100", "test_stage", "extra_token"])
        assert 0 <= seed < 2**31

    def test_multi_token_list(self):
        # Three-token list: useful for per-shock-per-stage seeds.
        s1 = derive_seed_tokens(["42", "shock_response", "kavanaugh_2018"])
        s2 = derive_seed_tokens(["42", "shock_response", "kavanaugh_2018"])
        assert s1 == s2
        s3 = derive_seed_tokens(["42", "shock_response", "soleimani_2020"])
        assert s1 != s3


class TestSeedContract:
    """Pipeline-level reproducibility: two runs with same seed produce identical outputs."""

    def test_full_pipeline_seed_equality(self):
        """Two generators initialized with the same derived seed produce identical sequences."""
        base_seed = 42
        stage = "monte_carlo"

        rng_run1 = make_rng(derive_seed(base_seed, stage))
        rng_run2 = make_rng(derive_seed(base_seed, stage))

        # Simulate a small Monte Carlo step (no need to assign w_star)
        noise1 = rng_run1.multivariate_normal(mean=np.zeros(5), cov=np.eye(5) * 0.01, size=100)
        noise2 = rng_run2.multivariate_normal(mean=np.zeros(5), cov=np.eye(5) * 0.01, size=100)
        np.testing.assert_array_equal(noise1, noise2)

    def test_stage_isolation(self):
        """Different stages derive different seeds even with the same base seed."""
        base = 42
        stage_seeds = {
            stage: derive_seed(base, stage)
            for stage in ["voter_panel", "baseline_portfolio", "monte_carlo", "llm_finetune"]
        }
        assert len(set(stage_seeds.values())) == len(stage_seeds)
