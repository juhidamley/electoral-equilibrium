"""Prefect DAG orchestrating all pipeline stages.

Run with: just smoke  (or: python -m electoral.pipeline --config configs/smoke.json)

Prefect benefits over hand-written orchestration:
  - Automatic retry on failure (retries=2, retry_delay_seconds=30)
  - Visual dashboard: prefect server start
  - Persistent state — crash restarts from last successful stage
  - Parallel task execution: task_baseline_portfolio and task_sentiment_data
    run concurrently after task_voter_panel completes (same upstream future)

Stage dependency graph:
  task_voter_panel  ──►  task_baseline_portfolio
                    └──►  task_sentiment_data ──►  task_llm_finetune
                                               └──►  task_shock_response
                                                          └──►  task_optimization
                                                                     └──►  task_simulation
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Conditional Prefect import — if not installed, use no-op decorators so the
# module imports cleanly and the pipeline still runs as plain Python.
try:
    from prefect import flow, task  # type: ignore[import]

    _HAS_PREFECT = True
except ImportError:
    _HAS_PREFECT = False

    def task(fn=None, **kwargs):  # type: ignore[misc]
        if fn is not None:
            return fn

        def decorator(f):
            return f

        return decorator

    def flow(fn=None, **kwargs):  # type: ignore[misc]
        if fn is not None:
            return fn

        def decorator(f):
            return f

        return decorator


from electoral.config import PipelineConfig
from electoral.kernels.sentiment import merge_all_posts
from electoral.stages import (
    build_baseline_portfolio,
    build_llm_finetune,
    build_optimization,
    build_sentiment_data,
    build_shock_response,
    build_voter_panel,
    run_simulations,
)

# All tasks share the same retry policy. Adjust per-task once real kernels land.
_RETRY = dict(retries=2, retry_delay_seconds=30)


@task(**_RETRY)
def task_voter_panel(config: PipelineConfig):
    return build_voter_panel(config)


@task(**_RETRY)
def task_baseline_portfolio(config: PipelineConfig, panel):
    return build_baseline_portfolio(config, panel)


@task(**_RETRY)
def task_merge_posts(config: PipelineConfig):
    """Merge per-platform JSONL files → rawdata/merged/{shock_id}/posts.jsonl."""
    return merge_all_posts()


@task(**_RETRY)
def task_sentiment_data(config: PipelineConfig, panel):
    return build_sentiment_data(config, panel)


@task(**_RETRY)
def task_llm_finetune(config: PipelineConfig, sentiment):
    return build_llm_finetune(config, sentiment)


@task(**_RETRY)
def task_shock_response(config: PipelineConfig, event: str, intensity: float):
    return build_shock_response(config, event, intensity)


@task(**_RETRY)
def task_optimization(config: PipelineConfig, shock):
    return build_optimization(config, shock)


@task(**_RETRY)
def task_simulation(config: PipelineConfig, equilibrium):
    return run_simulations(config, equilibrium)


@flow(name="electoral-equilibrium")
def run_pipeline(
    config_path: str,
    shock_event: str = "smoke_test",
    shock_intensity: float = 0.5,
) -> dict:
    """Full Electoral Equilibrium pipeline flow.

    task_baseline_portfolio and task_sentiment_data both receive the same
    `panel` future so Prefect's ConcurrentTaskRunner executes them in parallel.

    In historical mode task_shock_response is called after task_llm_finetune
    completes; in continuous mode it follows task_sentiment_data directly.
    Python execution order in the flow body is the implicit ordering guarantee.

    Returns a dict of all stage artifacts for testing and inspection.
    """
    config = PipelineConfig.from_json(config_path)
    config.validate()

    # Stage 1: voter panel — no upstream dependencies
    panel = task_voter_panel(config)

    # Stage 2: merge per-platform social files → rawdata/merged/ (runs after panel)
    task_merge_posts(config)

    # Stages 3 + 4: both depend only on panel → run concurrently under Prefect
    baseline = task_baseline_portfolio(config, panel)
    sentiment = task_sentiment_data(config, panel)

    # Stage 4: LLM fine-tuning — historical mode only
    if config.pipeline_mode == "historical":
        finetune = task_llm_finetune(config, sentiment)
    else:
        finetune = None

    # Stage 5: shock response — follows finetune (historical) or sentiment (continuous)
    # Python call order here ensures Prefect does not start this task until the
    # preceding branch completes.
    shock = task_shock_response(config, shock_event, shock_intensity)

    # Stages 6 + 7: linear chain
    equilibrium = task_optimization(config, shock)
    simulation = task_simulation(config, equilibrium)

    return {
        "voter_panel": panel,
        "baseline_portfolio": baseline,
        "sentiment_data": sentiment,
        "llm_finetune": finetune,
        "shock_response": shock,
        "optimization": equilibrium,
        "simulation": simulation,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Electoral Equilibrium pipeline")
    parser.add_argument(
        "--config",
        default="configs/smoke.json",
        help="Path to PipelineConfig JSON (default: configs/smoke.json)",
    )
    parser.add_argument(
        "--shock",
        default="smoke_test",
        help="Shock event identifier (default: smoke_test)",
    )
    parser.add_argument(
        "--intensity",
        type=float,
        default=0.5,
        help="Shock intensity 0–1 (default: 0.5)",
    )
    args = parser.parse_args()

    if not Path(args.config).exists():
        print(f"Error: config not found: {args.config}", file=sys.stderr)
        sys.exit(1)

    if _HAS_PREFECT:
        run_pipeline(
            config_path=args.config,
            shock_event=args.shock,
            shock_intensity=args.intensity,
        )
    else:
        # Fallback: run as plain Python without Prefect (CI / minimal installs)
        config = PipelineConfig.from_json(args.config)
        config.validate()
        panel = build_voter_panel(config)
        merge_all_posts()
        build_baseline_portfolio(config, panel)
        sentiment = build_sentiment_data(config, panel)
        if config.pipeline_mode == "historical":
            build_llm_finetune(config, sentiment)
        shock = build_shock_response(config, args.shock, args.intensity)
        equilibrium = build_optimization(config, shock)
        run_simulations(config, equilibrium)
        print(f"Pipeline completed (run_key={config.run_key!r})")


if __name__ == "__main__":
    main()
