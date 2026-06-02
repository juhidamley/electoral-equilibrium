"""Pipeline stage functions — one stub per stage.

Each function accepts PipelineConfig and returns the typed artifact for that stage.
Every week replaces a stub with real kernel logic.

Stage dependency graph:
  build_voter_panel  ──►  build_baseline_portfolio
                     └──►  build_sentiment_data ──►  build_llm_finetune
                                                 └──►  build_shock_response
                                                           └──►  build_optimization
                                                                      └──►  run_simulations
"""

from __future__ import annotations

from electoral.artifacts import (
    BaselinePortfolioData,
    EquilibriumData,
    LLMFineTuneData,
    SentimentData,
    ShockResponseData,
    SimulationData,
    StageArtifact,
    VoterPanelData,
)
from electoral.config import PipelineConfig
from electoral.core.io import write_artifact
from electoral.core.rng import make_rng
from electoral.kernels.data import build_voter_panel as _build_voter_panel_kernel


def build_voter_panel(config: PipelineConfig) -> VoterPanelData:
    """Week 1: ingest raw survey exports → validated longitudinal voter panel."""
    payload, panel = _build_voter_panel_kernel(config)

    # Write stratum-split panel parquets alongside the JSON envelope.
    from pathlib import Path
    from electoral.core.types import CANONICAL_RACES, CANONICAL_RELIGIONS, CANONICAL_GENDERS

    panel_dir = Path(config.output_dir) / "panel"
    panel_dir.mkdir(parents=True, exist_ok=True)
    panel[panel["bloc"].isin(CANONICAL_RACES)].to_parquet(
        panel_dir / "panel_race.parquet", index=False
    )
    panel[panel["bloc"].isin(CANONICAL_RELIGIONS)].to_parquet(
        panel_dir / "panel_religion.parquet", index=False
    )
    panel[panel["bloc"].isin(CANONICAL_GENDERS)].to_parquet(
        panel_dir / "panel_gender.parquet", index=False
    )

    envelope = StageArtifact(
        stage="voter_panel",
        run_key=config.run_key,
        metadata={"seed": config.derive_seed("voter_panel")},
        data=payload.to_dict(),
    )
    write_artifact(f"{config.output_dir}/voter_panel.json", envelope.to_dict())
    return payload


def build_baseline_portfolio(
    config: PipelineConfig,
    panel: VoterPanelData,
) -> BaselinePortfolioData:
    """Week 2: ML-derived baseline demographic distribution + V_eq."""
    placeholder_weights = {r: 1.0 / len(config.races) for r in config.races}
    placeholder_mu_race = {r: 0.50 for r in config.races}
    placeholder_mu_religion = {r: 0.50 for r in config.religions}
    placeholder_mu_gender = {r: 0.50 for r in config.genders}
    payload = BaselinePortfolioData(
        method="placeholder",
        party=config.party,
        weights=placeholder_weights,
        mu_race=placeholder_mu_race,
        mu_religion=placeholder_mu_religion,
        mu_gender=placeholder_mu_gender,
        mu_eff=0.50,
        layer_weights=panel.layer_weights,
        target=config.target,
    )
    payload.validate()
    envelope = StageArtifact(
        stage="baseline_portfolio",
        run_key=config.run_key,
        metadata={"seed": config.derive_seed("baseline_portfolio")},
        data=payload.to_dict(),
    )
    write_artifact(f"{config.output_dir}/baseline_portfolio.json", envelope.to_dict())
    return payload


def build_sentiment_data(
    config: PipelineConfig,
    panel: VoterPanelData,
) -> SentimentData:
    """Week 3: RoBERTa pipeline → per-bloc sentiment elasticity."""
    payload = SentimentData(
        model="cardiffnlp/twitter-roberta-base-sentiment",
        shocks=[],
        scores={},
    )
    payload.validate()
    envelope = StageArtifact(
        stage="sentiment_data",
        run_key=config.run_key,
        metadata={"seed": config.derive_seed("sentiment_data")},
        data=payload.to_dict(),
    )
    write_artifact(f"{config.output_dir}/sentiment_data.json", envelope.to_dict())
    return payload


def build_llm_finetune(
    config: PipelineConfig,
    sentiment: SentimentData,
) -> LLMFineTuneData:
    """Week 4: QLoRA fine-tuning of Mistral 7B on the unified dataset."""
    payload = LLMFineTuneData(
        base_model="mistralai/Mistral-7B-v0.3",
        lora_rank=16,
        n_examples=1,
        cycles_used=[2020],
        adapter_path=None,
    )
    payload.validate()
    envelope = StageArtifact(
        stage="llm_finetune",
        run_key=config.run_key,
        metadata={"seed": config.derive_seed("llm_finetune")},
        data=payload.to_dict(),
    )
    write_artifact(f"{config.output_dir}/llm_finetune.json", envelope.to_dict())
    return payload


def build_shock_response(
    config: PipelineConfig,
    event: str,
    intensity: float,
) -> ShockResponseData:
    """Week 4/5: LLM constrained decoding → per-bloc Δμ estimates."""
    all_blocs = list(config.races) + list(config.religions) + list(config.genders)
    n = len(all_blocs)
    payload = ShockResponseData(
        shock=event,
        cycle=2020,
        deltas={bloc: 0.0 for bloc in all_blocs},
        covariance=[[0.0] * n for _ in range(n)],
        source="llm_unified",
    )
    payload.validate()
    envelope = StageArtifact(
        stage="shock_response",
        run_key=config.run_key,
        metadata={"event": event, "intensity": intensity},
        data=payload.to_dict(),
    )
    write_artifact(f"{config.output_dir}/shock_response.json", envelope.to_dict())
    return payload


def build_optimization(
    config: PipelineConfig,
    shock: ShockResponseData,
) -> EquilibriumData:
    """Week 5: CVXPY DQCP optimizer → rebalanced coalition weights.

    weights and mu_shifted are keyed by race blocs only — the optimizer decision
    variables. Religion and gender weights (v_R, g_G) are fixed and not optimized.
    """
    placeholder_weights = {r: 1.0 / len(config.races) for r in config.races}
    payload = EquilibriumData(
        method="placeholder",
        party=config.party,
        shock=shock.shock,
        weights=placeholder_weights,
        mu_shifted={r: 0.50 for r in config.races},
        feasible=True,
        target_met=False,
        target=config.target,
    )
    payload.validate()
    envelope = StageArtifact(
        stage="optimization",
        run_key=config.run_key,
        metadata={},
        data=payload.to_dict(),
    )
    write_artifact(f"{config.output_dir}/optimization.json", envelope.to_dict())
    return payload


def run_simulations(
    config: PipelineConfig,
    equilibrium: EquilibriumData,
    n_simulations: int = 10_000,
) -> SimulationData:
    """Week 5: Logistic-Normal ILR Monte Carlo → 90% CI on win probability."""
    rng = make_rng(config.derive_seed("monte_carlo"))
    _ = rng  # seeded but not yet used in stub
    payload = SimulationData(
        n_simulations=n_simulations,
        seed=config.derive_seed("monte_carlo"),
        win_probability=0.50,
        percentiles={r: [0.1, 0.3, 0.5, 0.7, 0.9] for r in config.races},
    )
    payload.validate()
    envelope = StageArtifact(
        stage="simulation",
        run_key=config.run_key,
        metadata={"n_simulations": n_simulations},
        data=payload.to_dict(),
    )
    write_artifact(f"{config.output_dir}/simulation.json", envelope.to_dict())
    return payload
