"""Typed, frozen payload dataclasses for every pipeline stage.

Design rules (artifact-first):
  - All classes are frozen=True: mutation impossible after construction
  - to_dict() / from_dict() use only native Python types (dict, list, str, int, float, bool, None)
  - validate() raises ValueError with context-specific messages
  - No pandas or NumPy objects in payload fields
  - Election cycles are int YYYY; demographic IDs are lowercase snake_case strings
"""
from __future__ import annotations

import dataclasses
import json
import math
from typing import Any

from electoral.core.schema import (
    assert_required_keys,
    assert_shares_sum_to_one,
    assert_sorted_unique,
    assert_unique,
    assert_valid_share,
)
from electoral.core.types import (
    CANONICAL_GENDERS,
    CANONICAL_RACES,
    CANONICAL_RELIGIONS,
    DELTA_BINS,
    LAYER_WEIGHT_KEYS,
    VALID_SOURCES,
)

_VALID_DELTA_BINS: frozenset[str] = frozenset(DELTA_BINS)
_VALID_PARTIES: frozenset[str] = frozenset(["democrat", "republican"])


# ── Envelope ────────────────────────────────────────────────────────────────

@dataclasses.dataclass(frozen=True)
class StageArtifact:
    """Artifact envelope that wraps every stage payload.

    Schema:
        { "stage": str, "run_key": str, "metadata": {...}, "data": {...} }

    The data field contains a stage payload's to_dict() output.
    Tabular stages additionally write a .parquet file alongside the JSON.
    """

    stage: str
    run_key: str
    metadata: dict[str, Any]
    data: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> StageArtifact:
        return cls(
            stage=str(d["stage"]),
            run_key=str(d["run_key"]),
            metadata=dict(d.get("metadata", {})),
            data=dict(d["data"]),
        )

    def validate(self) -> None:
        assert_required_keys(
            {"stage": self.stage, "run_key": self.run_key, "data": self.data},
            ["stage", "run_key", "data"],
            context="StageArtifact",
        )
        if not self.stage:
            raise ValueError("StageArtifact.stage must be non-empty")
        if not self.run_key:
            raise ValueError("StageArtifact.run_key must be non-empty")


# ── Stage 1: Voter Panel ─────────────────────────────────────────────────────

@dataclasses.dataclass(frozen=True)
class VoterPanelData:
    """Cleaned longitudinal voter panel: three independent marginal stratum tables.

    No cross-tabulations; no sparse intersection cells. Each stratum is
    independently mutually exclusive and exhaustive over ~100% of the electorate.
    """

    cycles: list[int]           # sorted unique election years YYYY
    races: list[str]            # 5 canonical RaceIds
    religions: list[str]        # 7 canonical ReligionIds
    genders: list[str]          # 3 canonical GenderIds
    n_rows_race: int            # rows in panel_race.parquet
    n_rows_religion: int        # rows in panel_religion.parquet
    n_rows_gender: int          # rows in panel_gender.parquet
    layer_weights: dict[str, float]  # {"lambda_1": x, "lambda_2": x, "lambda_3": x}; sum to 1
    source: str | None          # e.g. "ARDA+GSS+NEP"

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> VoterPanelData:
        return cls(
            cycles=list(payload["cycles"]),
            races=list(payload["races"]),
            religions=list(payload["religions"]),
            genders=list(payload["genders"]),
            n_rows_race=int(payload["n_rows_race"]),
            n_rows_religion=int(payload["n_rows_religion"]),
            n_rows_gender=int(payload["n_rows_gender"]),
            layer_weights=dict(payload["layer_weights"]),
            source=payload.get("source"),
        )

    def validate(self) -> None:
        assert_sorted_unique(self.cycles, name="cycles", context="VoterPanelData")
        assert_unique(self.races, name="races", context="VoterPanelData")
        assert_unique(self.religions, name="religions", context="VoterPanelData")
        assert_unique(self.genders, name="genders", context="VoterPanelData")
        for name, val in [
            ("n_rows_race", self.n_rows_race),
            ("n_rows_religion", self.n_rows_religion),
            ("n_rows_gender", self.n_rows_gender),
        ]:
            if val < 0:
                raise ValueError(
                    f"VoterPanelData.{name} must be non-negative, got {val}"
                )
        assert_required_keys(
            self.layer_weights,
            list(LAYER_WEIGHT_KEYS),
            context="VoterPanelData.layer_weights",
        )
        assert_shares_sum_to_one(
            self.layer_weights, context="VoterPanelData.layer_weights"
        )


# ── Stage 2: Baseline Portfolio ───────────────────────────────────────────────

@dataclasses.dataclass(frozen=True)
class BaselinePortfolioData:
    """Steady-state voter equilibrium: optimal coalition weights + moment estimates."""

    method: str                         # e.g. "cvxpy_dqcp"
    party: str                          # "democrat" or "republican"
    weights: dict[str, float]           # race_id → coalition weight; 5 keys; sums to 1.0
    mu_race: dict[str, float]           # race_id → within-race vote share (Stratum 1)
    mu_religion: dict[str, float]       # religion_id → within-religion vote share (Stratum 2)
    mu_gender: dict[str, float]         # gender_id → within-gender vote share (Stratum 3)
    mu_eff: float                       # scalar effective loyalty (weighted sum, all strata)
    layer_weights: dict[str, float]     # {"lambda_1": x, "lambda_2": x, "lambda_3": x}
    target: float                       # V_eq threshold (~0.52-0.53 Dem, ~0.49-0.51 Rep)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> BaselinePortfolioData:
        return cls(
            method=str(payload["method"]),
            party=str(payload["party"]),
            weights=dict(payload["weights"]),
            mu_race=dict(payload["mu_race"]),
            mu_religion=dict(payload["mu_religion"]),
            mu_gender=dict(payload["mu_gender"]),
            mu_eff=float(payload["mu_eff"]),
            layer_weights=dict(payload["layer_weights"]),
            target=float(payload["target"]),
        )

    def validate(self) -> None:
        if self.party not in _VALID_PARTIES:
            raise ValueError(
                f"BaselinePortfolioData.party must be 'democrat' or 'republican', "
                f"got {self.party!r}"
            )
        assert_shares_sum_to_one(
            self.weights, context="BaselinePortfolioData.weights"
        )
        for k, v in self.weights.items():
            assert_valid_share(v, name=f"weights[{k}]", context="BaselinePortfolioData")
        for k, v in self.mu_race.items():
            assert_valid_share(v, name=f"mu_race[{k}]", context="BaselinePortfolioData")
        for k, v in self.mu_religion.items():
            assert_valid_share(v, name=f"mu_religion[{k}]", context="BaselinePortfolioData")
        for k, v in self.mu_gender.items():
            assert_valid_share(v, name=f"mu_gender[{k}]", context="BaselinePortfolioData")
        assert_valid_share(self.mu_eff, name="mu_eff", context="BaselinePortfolioData")
        assert_required_keys(
            self.layer_weights,
            list(LAYER_WEIGHT_KEYS),
            context="BaselinePortfolioData.layer_weights",
        )
        assert_shares_sum_to_one(
            self.layer_weights, context="BaselinePortfolioData.layer_weights"
        )
        if not (0.5 < self.target < 0.7):
            raise ValueError(
                f"BaselinePortfolioData.target must be in (0.5, 0.7), got {self.target}"
            )


# ── Stage 3a: News/Social RoBERTa sentiment ───────────────────────────────────

@dataclasses.dataclass(frozen=True)
class SentimentData:
    """RoBERTa-derived elasticity scores used as LLM fine-tuning input features.

    NOTE: These scores are intermediate features, not final shock estimates.
    They are consumed by the LLM fine-tuning pipeline.
    """

    model: str                                  # e.g. "cardiffnlp/twitter-roberta-base-sentiment"
    shocks: list[str]                           # shock event identifiers (unique)
    scores: dict[str, dict[str, float]]         # scores[bloc_id][shock_id] = elasticity in [-1, 1]

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SentimentData:
        return cls(
            model=str(payload["model"]),
            shocks=list(payload["shocks"]),
            scores={
                bloc: dict(shock_scores)
                for bloc, shock_scores in payload["scores"].items()
            },
        )

    def validate(self) -> None:
        assert_unique(self.shocks, name="shocks", context="SentimentData")
        for bloc_id, shock_scores in self.scores.items():
            for shock_id, score in shock_scores.items():
                if not (-1.0 <= score <= 1.0):
                    raise ValueError(
                        f"SentimentData.scores[{bloc_id!r}][{shock_id!r}] = {score} "
                        f"is not in [-1.0, 1.0]"
                    )


# ── Stage 3b: Social media sentiment (platform-aggregated) ───────────────────

@dataclasses.dataclass(frozen=True)
class SocialMediaSentimentData:
    """Platform-aggregated sentiment paired with lagged favorability polls.

    Used as a high-frequency weak supervisor to augment the small election-cycle
    training set. Platform user-base demographics serve as proxies for voter blocs.
    """

    shock: str                              # shock event identifier
    platforms: list[str]                    # e.g. ["bluesky", "apify", "facebook"]
    window_hours: int                       # collection window around shock (e.g. 72)
    scores: dict[str, dict[str, float]]     # scores[platform][bloc_proxy] = elasticity [-1, 1]
    n_posts: dict[str, int]                 # n_posts[platform] = number of posts collected
    lagged_delta: dict[str, float] | None   # favorability shift at t+14 days; None if unavailable

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SocialMediaSentimentData:
        raw_lagged = payload.get("lagged_delta")
        return cls(
            shock=str(payload["shock"]),
            platforms=list(payload["platforms"]),
            window_hours=int(payload["window_hours"]),
            scores={
                platform: dict(proxy_scores)
                for platform, proxy_scores in payload["scores"].items()
            },
            n_posts={k: int(v) for k, v in payload["n_posts"].items()},
            lagged_delta={k: float(v) for k, v in raw_lagged.items()}
            if raw_lagged is not None
            else None,
        )

    def validate(self) -> None:
        if not self.shock:
            raise ValueError("SocialMediaSentimentData.shock must be non-empty")
        assert_unique(
            self.platforms, name="platforms", context="SocialMediaSentimentData"
        )
        if self.window_hours <= 0:
            raise ValueError(
                f"SocialMediaSentimentData.window_hours must be positive, "
                f"got {self.window_hours}"
            )
        for platform, proxy_scores in self.scores.items():
            for proxy, score in proxy_scores.items():
                if not (-1.0 <= score <= 1.0):
                    raise ValueError(
                        f"SocialMediaSentimentData.scores[{platform!r}][{proxy!r}] "
                        f"= {score} is not in [-1.0, 1.0]"
                    )
        for platform, count in self.n_posts.items():
            if count < 0:
                raise ValueError(
                    f"SocialMediaSentimentData.n_posts[{platform!r}] = {count} "
                    f"must be non-negative"
                )


# ── Stage 3c: LLM fine-tuning dataset ────────────────────────────────────────

@dataclasses.dataclass(frozen=True)
class LLMFineTuneData:
    """Metadata about the assembled LLM fine-tuning dataset."""

    base_model: str                 # e.g. "mistralai/Mistral-7B-v0.3"
    lora_rank: int                  # LoRA rank parameter (16 or 32)
    n_examples: int                 # number of fine-tuning examples in train.jsonl
    cycles_used: list[int]          # sorted unique election cycles in training set
    adapter_path: str | None        # path to saved QLoRA adapter weights (None if not yet trained)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> LLMFineTuneData:
        return cls(
            base_model=str(payload["base_model"]),
            lora_rank=int(payload["lora_rank"]),
            n_examples=int(payload["n_examples"]),
            cycles_used=list(payload["cycles_used"]),
            adapter_path=payload.get("adapter_path"),
        )

    def validate(self) -> None:
        if self.lora_rank <= 0:
            raise ValueError(
                f"LLMFineTuneData.lora_rank must be positive, got {self.lora_rank}"
            )
        if self.n_examples <= 0:
            raise ValueError(
                f"LLMFineTuneData.n_examples must be positive, got {self.n_examples}"
            )
        assert_sorted_unique(
            self.cycles_used, name="cycles_used", context="LLMFineTuneData"
        )


# ── Stage 3d: Prediction market data ─────────────────────────────────────────

@dataclasses.dataclass(frozen=True)
class PredictionMarketData:
    """Incentive-compatible aggregate beliefs from real-money prediction markets.

    IMPORTANT: Prediction markets are CALIBRATION BENCHMARKS ONLY.
    delta_prob is NOT a training feature; it is not stored as a demographic signal.
    Use: (1) post-hoc calibration of model win-probability vs market price;
         (2) real-time display in the web app as "Market consensus: X%".
    """

    shock: str                          # shock event identifier
    party: str                          # "democrat" or "republican"
    pre_shock_prob: float               # market-implied win probability 24h before shock
    post_shock_1h: float | None         # market-implied win probability 1h after shock
    post_shock_24h: float | None        # 24h after
    post_shock_72h: float | None        # 72h after
    delta_prob: float                   # post_shock_24h - pre_shock_prob (calibration only)
    sources: list[str]                  # e.g. ["polymarket", "predictit"]
    contract_ids: dict[str, str]        # source → contract identifier (from market_contracts.json)
    volume: dict[str, float] | None     # source → total $ volume in 72h window

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PredictionMarketData:
        def _opt_float(v: Any) -> float | None:
            return float(v) if v is not None else None

        raw_vol = payload.get("volume")
        return cls(
            shock=str(payload["shock"]),
            party=str(payload["party"]),
            pre_shock_prob=float(payload["pre_shock_prob"]),
            post_shock_1h=_opt_float(payload.get("post_shock_1h")),
            post_shock_24h=_opt_float(payload.get("post_shock_24h")),
            post_shock_72h=_opt_float(payload.get("post_shock_72h")),
            delta_prob=float(payload["delta_prob"]),
            sources=list(payload["sources"]),
            contract_ids=dict(payload["contract_ids"]),
            volume={k: float(v) for k, v in raw_vol.items()}
            if raw_vol is not None
            else None,
        )

    def validate(self) -> None:
        if not self.shock:
            raise ValueError("PredictionMarketData.shock must be non-empty")
        if self.party not in _VALID_PARTIES:
            raise ValueError(
                f"PredictionMarketData.party must be 'democrat' or 'republican', "
                f"got {self.party!r}"
            )
        assert_valid_share(
            self.pre_shock_prob,
            name="pre_shock_prob",
            context="PredictionMarketData",
        )
        for attr_name in ("post_shock_1h", "post_shock_24h", "post_shock_72h"):
            val = getattr(self, attr_name)
            if val is not None:
                if not (0.0 <= val <= 1.0):
                    raise ValueError(
                        f"PredictionMarketData.{attr_name} = {val} must be in [0.0, 1.0]"
                    )
        if not math.isfinite(self.delta_prob):
            raise ValueError(
                f"PredictionMarketData.delta_prob must be finite, got {self.delta_prob}"
            )
        assert_unique(
            self.sources, name="sources", context="PredictionMarketData"
        )


# ── Stage 4: LLM shock response ───────────────────────────────────────────────

@dataclasses.dataclass(frozen=True)
class ShockResponseData:
    """Combined shock-derived voter share deviations across all three strata.

    The LLM outputs 9-token categorical bins per stratum cell (constrained
    decoding via outlines). Numeric deltas are midpoints of each bin.
    Covariance is 5×5 (race-level only) estimated via Ledoit-Wolf shrinkage.
    """

    shock: str
    cycle: int                              # most-recent historical cycle used
    party: str                              # "democrat" or "republican"
    delta_bins_race: dict[str, str]         # race_id → delta bin token (5 keys)
    delta_bins_religion: dict[str, str]     # religion_id → delta bin token (7 keys)
    delta_bins_gender: dict[str, str]       # gender_id → delta bin token (3 keys)
    deltas_race: dict[str, float]           # race_id → numeric midpoint of bin (5 keys)
    deltas_religion: dict[str, float]       # religion_id → numeric midpoint (7 keys)
    deltas_gender: dict[str, float]         # gender_id → numeric midpoint (3 keys)
    delta_eff: float                        # scalar: lambda_1*Σ(w*Δrace) + lambda_2*Σ(v*Δrel) + lambda_3*Σ(g*Δgen)
    covariance: list[list[float]]           # 5×5 race-level covariance of deltas_race
    source: str                             # "llm_unified" | "roberta_news_only" | "roberta_social_only"

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ShockResponseData:
        return cls(
            shock=str(payload["shock"]),
            cycle=int(payload["cycle"]),
            party=str(payload["party"]),
            delta_bins_race=dict(payload["delta_bins_race"]),
            delta_bins_religion=dict(payload["delta_bins_religion"]),
            delta_bins_gender=dict(payload["delta_bins_gender"]),
            deltas_race={k: float(v) for k, v in payload["deltas_race"].items()},
            deltas_religion={k: float(v) for k, v in payload["deltas_religion"].items()},
            deltas_gender={k: float(v) for k, v in payload["deltas_gender"].items()},
            delta_eff=float(payload["delta_eff"]),
            covariance=[list(row) for row in payload["covariance"]],
            source=str(payload["source"]),
        )

    def validate(self) -> None:
        if not self.shock:
            raise ValueError("ShockResponseData.shock must be non-empty")
        if not (1900 < self.cycle < 2100):
            raise ValueError(
                f"ShockResponseData.cycle must be a valid year (YYYY), got {self.cycle}"
            )
        if self.party not in _VALID_PARTIES:
            raise ValueError(
                f"ShockResponseData.party must be 'democrat' or 'republican', "
                f"got {self.party!r}"
            )
        for stratum, bins_dict in [
            ("delta_bins_race", self.delta_bins_race),
            ("delta_bins_religion", self.delta_bins_religion),
            ("delta_bins_gender", self.delta_bins_gender),
        ]:
            for k, v in bins_dict.items():
                if v not in _VALID_DELTA_BINS:
                    raise ValueError(
                        f"ShockResponseData.{stratum}[{k!r}] = {v!r} is not a valid "
                        f"delta bin token. Must be one of {DELTA_BINS}"
                    )
        for stratum, deltas_dict in [
            ("deltas_race", self.deltas_race),
            ("deltas_religion", self.deltas_religion),
            ("deltas_gender", self.deltas_gender),
        ]:
            for k, v in deltas_dict.items():
                if not math.isfinite(v):
                    raise ValueError(
                        f"ShockResponseData.{stratum}[{k!r}] = {v} must be finite"
                    )
                if not (-0.15 <= v <= 0.15):
                    raise ValueError(
                        f"ShockResponseData.{stratum}[{k!r}] = {v} is outside "
                        f"[-0.15, 0.15]"
                    )
        n = len(self.deltas_race)
        if len(self.covariance) != n:
            raise ValueError(
                f"ShockResponseData.covariance must be {n}×{n}, "
                f"got {len(self.covariance)} rows"
            )
        for i, row in enumerate(self.covariance):
            if len(row) != n:
                raise ValueError(
                    f"ShockResponseData.covariance row {i} must have {n} elements, "
                    f"got {len(row)}"
                )
        if self.source not in VALID_SOURCES:
            raise ValueError(
                f"ShockResponseData.source must be one of {sorted(VALID_SOURCES)}, "
                f"got {self.source!r}"
            )
        if not math.isfinite(self.delta_eff):
            raise ValueError(
                f"ShockResponseData.delta_eff must be finite, got {self.delta_eff}"
            )


# ── Stage 5: Equilibrium (post-shock optimizer output) ───────────────────────

@dataclasses.dataclass(frozen=True)
class EquilibriumData:
    """Rebalanced voter coalition after shock, from the CVXPY DQCP optimizer.

    Objective: max Φ((μ̃_eff(w) - V_eq) / sqrt(λ₁² · wᵀΣ_Δw))
    This maximizes P(win) rather than minimizing variance (correct for deficit scenarios).
    """

    method: str                     # e.g. "cvxpy_dqcp"
    party: str                      # "democrat" or "republican"
    shock: str | None               # shock event identifier (None for baseline)
    weights: dict[str, float]       # race_id → rebalanced coalition weight; 5 keys; sums to 1.0
    mu_eff_shifted: float           # post-shock scalar effective loyalty (all strata combined)
    feasible: bool                  # False if no w on the simplex can push μ̃_eff above V_eq
    target_met: bool                # True if mu_eff_shifted >= target
    target: float                   # V_eq threshold

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> EquilibriumData:
        return cls(
            method=str(payload["method"]),
            party=str(payload["party"]),
            shock=payload.get("shock"),
            weights={k: float(v) for k, v in payload["weights"].items()},
            mu_eff_shifted=float(payload["mu_eff_shifted"]),
            feasible=bool(payload["feasible"]),
            target_met=bool(payload["target_met"]),
            target=float(payload["target"]),
        )

    def validate(self) -> None:
        if self.party not in _VALID_PARTIES:
            raise ValueError(
                f"EquilibriumData.party must be 'democrat' or 'republican', "
                f"got {self.party!r}"
            )
        assert_shares_sum_to_one(
            self.weights, context="EquilibriumData.weights"
        )
        for k, v in self.weights.items():
            assert_valid_share(v, name=f"weights[{k}]", context="EquilibriumData")
        if not math.isfinite(self.mu_eff_shifted):
            raise ValueError(
                f"EquilibriumData.mu_eff_shifted must be finite, "
                f"got {self.mu_eff_shifted}"
            )
        if not (0.5 < self.target < 0.7):
            raise ValueError(
                f"EquilibriumData.target must be in (0.5, 0.7), got {self.target}"
            )


# ── Stage 6: Monte Carlo simulation ──────────────────────────────────────────

@dataclasses.dataclass(frozen=True)
class SimulationData:
    """Monte Carlo simulation summary using Logistic-Normal ILR parameterization.

    NOT Dirichlet (forces negative off-diagonal covariances — cannot model wave elections).
    Uses ILR (isometric log-ratio) with Helmert contrast matrix for simplex sampling.
    90% CI: win_probability_low (p5) to win_probability_high (p95).
    """

    n_simulations: int              # number of Monte Carlo draws (≥10,000 for production)
    seed: int                       # RNG seed used for this simulation run
    win_probability: float          # point estimate: fraction of draws meeting V_eq
    win_probability_low: float      # 5th percentile (p=0.05 lower CI bound)
    win_probability_high: float     # 95th percentile (p=0.95 upper CI bound)
    percentiles: dict[str, list[float]]  # bloc_id → [p5, p25, p50, p75, p95]

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SimulationData:
        return cls(
            n_simulations=int(payload["n_simulations"]),
            seed=int(payload["seed"]),
            win_probability=float(payload["win_probability"]),
            win_probability_low=float(payload["win_probability_low"]),
            win_probability_high=float(payload["win_probability_high"]),
            percentiles={
                k: [float(p) for p in v]
                for k, v in payload["percentiles"].items()
            },
        )

    def validate(self) -> None:
        if self.n_simulations <= 0:
            raise ValueError(
                f"SimulationData.n_simulations must be positive, "
                f"got {self.n_simulations}"
            )
        for attr in ("win_probability", "win_probability_low", "win_probability_high"):
            val = getattr(self, attr)
            if not (0.0 <= val <= 1.0):
                raise ValueError(
                    f"SimulationData.{attr} = {val} must be in [0.0, 1.0]"
                )
        for bloc_id, pcts in self.percentiles.items():
            if len(pcts) != 5:
                raise ValueError(
                    f"SimulationData.percentiles[{bloc_id!r}] must have exactly 5 values "
                    f"(p5, p25, p50, p75, p95), got {len(pcts)}"
                )
            for i, p in enumerate(pcts):
                if not (0.0 <= p <= 1.0):
                    raise ValueError(
                        f"SimulationData.percentiles[{bloc_id!r}][{i}] = {p} "
                        f"must be in [0.0, 1.0]"
                    )


# ── Stage 7: Performance metrics tables ──────────────────────────────────────

@dataclasses.dataclass(frozen=True)
class MetricsTablesData:
    """Summary performance metrics in manuscript-ready table format."""

    tables: dict[str, Any]   # tables["table_key"] = JSON-serializable table payload

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> MetricsTablesData:
        return cls(tables=dict(payload["tables"]))

    def validate(self) -> None:
        for k, v in self.tables.items():
            if not k:
                raise ValueError("MetricsTablesData.tables has empty key")
            try:
                json.dumps(v)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"MetricsTablesData.tables[{k!r}] is not JSON-serializable: {exc}"
                ) from exc
