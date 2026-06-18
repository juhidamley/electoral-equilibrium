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
from typing import Annotated, Any, Literal

from pydantic import BaseModel, BeforeValidator, ConfigDict

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

_VALID_PARTIES: frozenset[str] = frozenset(["democrat", "republican"])
# Union of all three strata — used to validate any field keyed by demographic bloc.
_ALL_CANONICAL_BLOCS: frozenset[str] = frozenset(
    list(CANONICAL_RACES) + list(CANONICAL_RELIGIONS) + list(CANONICAL_GENDERS)
)


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

    cycles: list[int]  # sorted unique election years YYYY
    races: list[str]  # 5 canonical RaceIds
    religions: list[str]  # 7 canonical ReligionIds
    genders: list[str]  # 3 canonical GenderIds
    n_rows_race: int  # rows in panel_race.parquet
    n_rows_religion: int  # rows in panel_religion.parquet
    n_rows_gender: int  # rows in panel_gender.parquet
    layer_weights: dict[str, float]  # {"lambda_1": x, "lambda_2": x, "lambda_3": x}; sum to 1
    source: str | None  # e.g. "ARDA+GSS+NEP"

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
        for r in self.races:
            if r not in CANONICAL_RACES:
                raise ValueError(
                    f"VoterPanelData.races[{r!r}] is not a canonical race ID. "
                    f"Must be one of {list(CANONICAL_RACES)}"
                )
        for r in self.religions:
            if r not in CANONICAL_RELIGIONS:
                raise ValueError(
                    f"VoterPanelData.religions[{r!r}] is not a canonical religion ID. "
                    f"Must be one of {list(CANONICAL_RELIGIONS)}"
                )
        for g in self.genders:
            if g not in CANONICAL_GENDERS:
                raise ValueError(
                    f"VoterPanelData.genders[{g!r}] is not a canonical gender ID. "
                    f"Must be one of {list(CANONICAL_GENDERS)}"
                )
        for name, val in [
            ("n_rows_race", self.n_rows_race),
            ("n_rows_religion", self.n_rows_religion),
            ("n_rows_gender", self.n_rows_gender),
        ]:
            if val < 0:
                raise ValueError(f"VoterPanelData.{name} must be non-negative, got {val}")
        assert_required_keys(
            self.layer_weights,
            list(LAYER_WEIGHT_KEYS),
            context="VoterPanelData.layer_weights",
        )
        assert_shares_sum_to_one(self.layer_weights, context="VoterPanelData.layer_weights")


# ── Stage 2: Baseline Portfolio ───────────────────────────────────────────────


@dataclasses.dataclass(frozen=True)
class BaselinePortfolioData:
    """Steady-state voter equilibrium: optimal coalition weights + moment estimates."""

    method: str  # e.g. "cvxpy_dqcp"
    party: str  # "democrat" or "republican"
    weights: dict[str, float]  # race_id → coalition weight; 5 keys; sums to 1.0
    mu_race: dict[str, float]  # race_id → within-race vote share (Stratum 1)
    mu_religion: dict[str, float]  # religion_id → within-religion vote share (Stratum 2)
    mu_gender: dict[str, float]  # gender_id → within-gender vote share (Stratum 3)
    mu_eff: float  # scalar effective loyalty (weighted sum, all strata)
    layer_weights: dict[str, float]  # {"lambda_1": x, "lambda_2": x, "lambda_3": x}
    target: float  # V_eq threshold (~0.52-0.53 Dem, ~0.49-0.51 Rep)

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
        for k, v in self.weights.items():
            if k not in CANONICAL_RACES:
                raise ValueError(f"BaselinePortfolioData.weights[{k!r}] is not a canonical race ID")
            assert_valid_share(v, name=f"weights[{k}]", context="BaselinePortfolioData")
        assert_shares_sum_to_one(self.weights, context="BaselinePortfolioData.weights")
        for k, v in self.mu_race.items():
            if k not in CANONICAL_RACES:
                raise ValueError(f"BaselinePortfolioData.mu_race[{k!r}] is not a canonical race ID")
            assert_valid_share(v, name=f"mu_race[{k}]", context="BaselinePortfolioData")
        for k, v in self.mu_religion.items():
            if k not in CANONICAL_RELIGIONS:
                raise ValueError(
                    f"BaselinePortfolioData.mu_religion[{k!r}] is not a canonical religion ID"
                )
            assert_valid_share(v, name=f"mu_religion[{k}]", context="BaselinePortfolioData")
        for k, v in self.mu_gender.items():
            if k not in CANONICAL_GENDERS:
                raise ValueError(
                    f"BaselinePortfolioData.mu_gender[{k!r}] is not a canonical gender ID"
                )
            assert_valid_share(v, name=f"mu_gender[{k}]", context="BaselinePortfolioData")
        assert_valid_share(self.mu_eff, name="mu_eff", context="BaselinePortfolioData")
        assert_required_keys(
            self.layer_weights,
            list(LAYER_WEIGHT_KEYS),
            context="BaselinePortfolioData.layer_weights",
        )
        assert_shares_sum_to_one(self.layer_weights, context="BaselinePortfolioData.layer_weights")
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

    model: str  # e.g. "cardiffnlp/twitter-roberta-base-sentiment"
    shocks: list[str]  # shock event identifiers (unique)
    scores: dict[str, dict[str, float]]  # scores[bloc_id][shock_id] = elasticity in [-1, 1]

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SentimentData:
        return cls(
            model=str(payload["model"]),
            shocks=list(payload["shocks"]),
            scores={bloc: dict(shock_scores) for bloc, shock_scores in payload["scores"].items()},
        )

    def validate(self) -> None:
        assert_unique(self.shocks, name="shocks", context="SentimentData")
        for bloc_id in self.scores:
            if bloc_id not in _ALL_CANONICAL_BLOCS:
                raise ValueError(f"SentimentData.scores[{bloc_id!r}] is not a canonical bloc ID")
        shocks_set = set(self.shocks)
        for bloc_id, shock_scores in self.scores.items():
            actual = set(shock_scores.keys())
            if actual != shocks_set:
                missing = sorted(shocks_set - actual)
                extra = sorted(actual - shocks_set)
                raise ValueError(
                    f"SentimentData.scores[{bloc_id!r}]: shock keys must match "
                    f"SentimentData.shocks exactly. "
                    f"Missing: {missing}, Extra: {extra}"
                )
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

    shock: str  # shock event identifier
    platforms: list[str]  # e.g. ["bluesky", "apify", "facebook"]
    window_hours: int  # collection window around shock (e.g. 72)
    scores: dict[str, dict[str, float]]  # scores[platform][bloc_proxy] = elasticity [-1, 1]
    n_posts: dict[str, int]  # n_posts[platform] = number of posts collected
    lagged_delta: dict[str, float] | None  # favorability shift at t+14 days; None if unavailable

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
                platform: dict(proxy_scores) for platform, proxy_scores in payload["scores"].items()
            },
            n_posts={k: int(v) for k, v in payload["n_posts"].items()},
            lagged_delta=(
                {k: float(v) for k, v in raw_lagged.items()} if raw_lagged is not None else None
            ),
        )

    def validate(self) -> None:
        if not self.shock:
            raise ValueError("SocialMediaSentimentData.shock must be non-empty")
        assert_unique(self.platforms, name="platforms", context="SocialMediaSentimentData")
        platforms_set = set(self.platforms)
        scores_platforms = set(self.scores.keys())
        if scores_platforms != platforms_set:
            missing = sorted(platforms_set - scores_platforms)
            extra = sorted(scores_platforms - platforms_set)
            raise ValueError(
                f"SocialMediaSentimentData.scores keys must match platforms exactly. "
                f"Missing: {missing}, extra: {extra}"
            )
        n_posts_platforms = set(self.n_posts.keys())
        if n_posts_platforms != platforms_set:
            missing = sorted(platforms_set - n_posts_platforms)
            extra = sorted(n_posts_platforms - platforms_set)
            raise ValueError(
                f"SocialMediaSentimentData.n_posts keys must match platforms exactly. "
                f"Missing: {missing}, extra: {extra}"
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

    base_model: str  # e.g. "mistralai/Mistral-7B-v0.3"
    lora_rank: int  # LoRA rank parameter (16 or 32)
    n_examples: int  # number of fine-tuning examples in train.jsonl
    cycles_used: list[int]  # sorted unique election cycles in training set
    adapter_path: str | None  # path to saved QLoRA adapter weights (None if not yet trained)

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
            raise ValueError(f"LLMFineTuneData.lora_rank must be positive, got {self.lora_rank}")
        if self.n_examples <= 0:
            raise ValueError(f"LLMFineTuneData.n_examples must be positive, got {self.n_examples}")
        assert_sorted_unique(self.cycles_used, name="cycles_used", context="LLMFineTuneData")


# ── Stage 3d: Prediction market data ─────────────────────────────────────────


@dataclasses.dataclass(frozen=True)
class PredictionMarketData:
    """Incentive-compatible aggregate beliefs from real-money prediction markets.

    IMPORTANT: Prediction markets are CALIBRATION BENCHMARKS ONLY.
    delta_prob is NOT a training feature; it is not stored as a demographic signal.
    Use: (1) post-hoc calibration of model win-probability vs market price;
         (2) real-time display in the web app as "Market consensus: X%".
    """

    shock: str  # shock event identifier
    party: str  # "democrat" or "republican"
    pre_shock_prob: float  # market-implied win probability 24h before shock
    post_shock_1h: float | None  # market-implied win probability 1h after shock
    post_shock_24h: float | None  # 24h after
    post_shock_72h: float | None  # 72h after
    delta_prob: float  # post_shock_24h - pre_shock_prob (calibration only)
    sources: list[str]  # e.g. ["polymarket", "predictit"]
    contract_ids: dict[str, str]  # source → contract identifier (from market_contracts.json)
    volume: dict[str, float] | None  # source → total $ volume in 72h window

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
            volume={k: float(v) for k, v in raw_vol.items()} if raw_vol is not None else None,
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
        assert_unique(self.sources, name="sources", context="PredictionMarketData")
        contract_keys = set(self.contract_ids.keys())
        sources_set = set(self.sources)
        if contract_keys != sources_set:
            missing = sorted(sources_set - contract_keys)
            extra = sorted(contract_keys - sources_set)
            raise ValueError(
                f"PredictionMarketData.contract_ids keys must match sources exactly. "
                f"Missing: {missing}, extra: {extra}"
            )


# ── Stage 4: LLM shock response ───────────────────────────────────────────────


@dataclasses.dataclass(frozen=True)
class ShockResponseData:
    """LLM-estimated within-bloc Democratic vote share changes after a political shock.

    Per-stratum layout separates raw bin tokens from converted numeric deltas:
      delta_bins_*  — raw constrained-decoding tokens (one of DELTA_BINS per bloc).
      deltas_*      — numeric Δμ midpoints derived from the bin tokens.
      covariance    — always 5×5 for race blocs only; religion and gender do not
                      enter Σ_Δ (see DECISIONS.md §Optimization).
    """

    shock: str
    cycle: int  # most-recent historical cycle used as context
    party: str  # "democrat" or "republican"
    delta_bins_race: dict[str, str]  # race_id → 9-token bin label
    delta_bins_religion: dict[str, str]  # religion_id → 9-token bin label
    delta_bins_gender: dict[str, str]  # gender_id → 9-token bin label
    deltas_race: dict[str, float]  # race_id → Δμ converted from bin midpoint
    deltas_religion: dict[str, float]  # religion_id → Δμ converted from bin midpoint
    deltas_gender: dict[str, float]  # gender_id → Δμ converted from bin midpoint
    delta_eff: float  # λ₁Σw_iΔμ_race + λ₂Σv_RΔμ_rel + λ₃Σg_GΔμ_gen
    covariance: list[list[float]]  # 5×5 race-bloc covariance (Ledoit-Wolf)
    source: str  # "llm_unified"|"roberta_news_only"|"roberta_social_only"

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
        _valid_bin_tokens = frozenset(DELTA_BINS)
        for field_name, bins_dict, canonical in (
            ("delta_bins_race", self.delta_bins_race, CANONICAL_RACES),
            ("delta_bins_religion", self.delta_bins_religion, CANONICAL_RELIGIONS),
            ("delta_bins_gender", self.delta_bins_gender, CANONICAL_GENDERS),
        ):
            for bloc_id, token in bins_dict.items():
                if bloc_id not in canonical:
                    raise ValueError(
                        f"ShockResponseData.{field_name}[{bloc_id!r}] is not a "
                        f"canonical bloc ID for that stratum"
                    )
                if token not in _valid_bin_tokens:
                    raise ValueError(
                        f"ShockResponseData.{field_name}[{bloc_id!r}] = {token!r} "
                        f"is not a valid delta bin token. Must be one of {DELTA_BINS}"
                    )
        for field_name, deltas_dict, canonical in (
            ("deltas_race", self.deltas_race, CANONICAL_RACES),
            ("deltas_religion", self.deltas_religion, CANONICAL_RELIGIONS),
            ("deltas_gender", self.deltas_gender, CANONICAL_GENDERS),
        ):
            for bloc_id, v in deltas_dict.items():
                if bloc_id not in canonical:
                    raise ValueError(
                        f"ShockResponseData.{field_name}[{bloc_id!r}] is not a "
                        f"canonical bloc ID for that stratum"
                    )
                if not math.isfinite(v):
                    raise ValueError(
                        f"ShockResponseData.{field_name}[{bloc_id!r}] = {v} must be finite"
                    )
                if not (-0.15 <= v <= 0.15):
                    raise ValueError(
                        f"ShockResponseData.{field_name}[{bloc_id!r}] = {v} "
                        f"is outside [-0.15, 0.15]"
                    )
        if not math.isfinite(self.delta_eff):
            raise ValueError(f"ShockResponseData.delta_eff = {self.delta_eff} must be finite")
        if len(self.covariance) != 5:
            raise ValueError(
                f"ShockResponseData.covariance must be 5×5, got {len(self.covariance)} rows"
            )
        for i, row in enumerate(self.covariance):
            if len(row) != 5:
                raise ValueError(
                    f"ShockResponseData.covariance row {i} must have 5 elements, " f"got {len(row)}"
                )
        for i in range(5):
            for j in range(i + 1, 5):
                if abs(self.covariance[i][j] - self.covariance[j][i]) > 1e-9:
                    raise ValueError(
                        f"ShockResponseData.covariance is not symmetric at "
                        f"[{i},{j}]: {self.covariance[i][j]} != {self.covariance[j][i]}"
                    )
        if self.source not in VALID_SOURCES:
            raise ValueError(
                f"ShockResponseData.source must be one of {sorted(VALID_SOURCES)}, "
                f"got {self.source!r}"
            )


# ── LLM output schema (Pydantic — used by outlines constrained decoding) ─────

DeltaBin = Literal[
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

# outlines FSM can emit space-separated character tokens (e.g. 'sl i g h t _ ne g')
# when the tokenizer maps bin labels as subword pieces. Strip spaces before the
# Literal check so validation succeeds and the correct bin string is stored.
_NormalizedDeltaBin = Annotated[
    DeltaBin,
    BeforeValidator(lambda v: v.replace(" ", "") if isinstance(v, str) else v),
]


class _RaceBins(BaseModel):
    african_american: _NormalizedDeltaBin
    latino: _NormalizedDeltaBin
    asian: _NormalizedDeltaBin
    white: _NormalizedDeltaBin
    other_race: _NormalizedDeltaBin


class _ReligionBins(BaseModel):
    evangelical: _NormalizedDeltaBin
    catholic: _NormalizedDeltaBin
    protestant: _NormalizedDeltaBin
    secular: _NormalizedDeltaBin
    jewish: _NormalizedDeltaBin
    muslim: _NormalizedDeltaBin
    other_rel: _NormalizedDeltaBin


class _GenderBins(BaseModel):
    women: _NormalizedDeltaBin
    men: _NormalizedDeltaBin
    other_gender: _NormalizedDeltaBin


class ShockResponseSchema(BaseModel):
    model_config = ConfigDict(frozen=True)

    delta_bins_race: _RaceBins
    delta_bins_religion: _ReligionBins
    delta_bins_gender: _GenderBins
    delta_eff: float


# ── Stage 5: Equilibrium (post-shock optimizer output) ───────────────────────


@dataclasses.dataclass(frozen=True)
class EquilibriumData:
    """Rebalanced voter coalition after shock, from the CVXPY DQCP optimizer.

    Objective: max Φ((μ̃_eff(w) - V_eq) / sqrt(λ₁² · wᵀΣ_Δw))
    This maximizes P(win) rather than minimizing variance (correct for deficit scenarios).

    weights and mu_shifted must share identical key sets so the frontend can zip
    them together to display coalition-weight and vote-share loyalty shifts per bloc.
    """

    method: str  # e.g. "cvxpy_dqcp"
    party: str  # "democrat" or "republican"
    shock: str | None  # shock event identifier (None for baseline)
    weights: dict[str, float]  # bloc_id → rebalanced coalition weight w̃_i; sums to 1.0
    mu_shifted: dict[str, float]  # bloc_id → post-shock within-bloc vote share μ̃_i^(P)
    feasible: bool  # False if no w on the simplex can push μ̃_eff above V_eq
    target_met: bool  # True if the rebalanced μ̃_eff meets V_eq
    target: float  # V_eq threshold

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> EquilibriumData:
        return cls(
            method=str(payload["method"]),
            party=str(payload["party"]),
            shock=payload.get("shock"),
            weights={k: float(v) for k, v in payload["weights"].items()},
            mu_shifted={k: float(v) for k, v in payload["mu_shifted"].items()},
            feasible=bool(payload["feasible"]),
            target_met=bool(payload["target_met"]),
            target=float(payload["target"]),
        )

    def validate(self) -> None:
        if self.party not in _VALID_PARTIES:
            raise ValueError(
                f"EquilibriumData.party must be 'democrat' or 'republican', got {self.party!r}"
            )
        assert_shares_sum_to_one(self.weights, context="EquilibriumData.weights")
        for k, v in self.weights.items():
            assert_valid_share(v, name=f"weights[{k}]", context="EquilibriumData")
        for k, v in self.mu_shifted.items():
            assert_valid_share(v, name=f"mu_shifted[{k}]", context="EquilibriumData")
        weights_keys = set(self.weights.keys())
        mu_keys = set(self.mu_shifted.keys())
        if weights_keys != mu_keys:
            missing = sorted(weights_keys - mu_keys)
            extra = sorted(mu_keys - weights_keys)
            raise ValueError(
                f"EquilibriumData.weights and mu_shifted must have identical key sets. "
                f"Missing from mu_shifted: {missing}, extra in mu_shifted: {extra}"
            )
        if not (0.5 < self.target < 0.7):
            raise ValueError(f"EquilibriumData.target must be in (0.5, 0.7), got {self.target}")


# ── Stage 6: Monte Carlo simulation ──────────────────────────────────────────


@dataclasses.dataclass(frozen=True)
class SimulationData:
    """Monte Carlo simulation summary using Logistic-Normal ILR parameterization.

    NOT Dirichlet (forces negative off-diagonal covariances — cannot model wave elections).
    Uses ILR (isometric log-ratio) with Helmert contrast matrix for simplex sampling.
    percentiles[bloc_id] = [p5, p25, p50, p75, p95] of the within-bloc weight distribution.
    """

    n_simulations: int  # number of Monte Carlo draws (≥10,000 for production)
    seed: int  # RNG seed used for this simulation run
win_probability: float  # point estimate: fraction of draws meeting V_eq
win_probability_low: float  # 5th percentile of bootstrap distribution of win_probability
win_probability_high: float  # 95th percentile of bootstrap distribution of win_probability
    percentiles: dict[str, list[float]]  # bloc_id → [p5, p25, p50, p75, p95]

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SimulationData:
        return cls(
            n_simulations=int(payload["n_simulations"]),
            seed=int(payload["seed"]),
            win_probability=float(payload["win_probability"]),
            win_probability_low=float(payload.get("win_probability_low", 0.0)),
            win_probability_high=float(payload.get("win_probability_high", 1.0)),
            percentiles={k: [float(p) for p in v] for k, v in payload["percentiles"].items()},
        )

    def validate(self) -> None:
        if self.n_simulations <= 0:
            raise ValueError(
                f"SimulationData.n_simulations must be positive, got {self.n_simulations}"
            )
        if not (0.0 <= self.win_probability <= 1.0):
            raise ValueError(
                f"SimulationData.win_probability = {self.win_probability} must be in [0.0, 1.0]"
            )
        if not (0.0 <= self.win_probability_low <= self.win_probability_high <= 1.0):
            raise ValueError(
                f"SimulationData: win_probability_low={self.win_probability_low} "
                f"must be <= win_probability_high={self.win_probability_high} "
                "and both in [0.0, 1.0]"
            )
        if not (self.win_probability_low <= self.win_probability <= self.win_probability_high):
            raise ValueError(
                f"SimulationData: win_probability={self.win_probability} must be within "
                f"CI bounds [{self.win_probability_low}, {self.win_probability_high}]"
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
            for i in range(len(pcts) - 1):
                if pcts[i] > pcts[i + 1]:
                    raise ValueError(
                        f"SimulationData.percentiles[{bloc_id!r}] must be non-decreasing; "
                        f"got {pcts[i]} > {pcts[i + 1]} at positions [{i},{i + 1}]"
                    )


# ── Stage 7: Performance metrics tables ──────────────────────────────────────


@dataclasses.dataclass(frozen=True)
class MetricsTablesData:
    """Summary performance metrics in manuscript-ready table format."""

    tables: dict[str, Any]  # tables["table_key"] = JSON-serializable table payload

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
