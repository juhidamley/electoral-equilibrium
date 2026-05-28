"""Prefect DAG orchestrating all pipeline stages.

Run with: just smoke  (or: python -m electoral.pipeline --config configs/smoke.json)

Prefect benefits over hand-written orchestration:
  - Automatic retry on failure (retries=2)
  - Visual dashboard: prefect server start
  - Persistent state — crash restarts from last successful stage
  - Parallel task execution where stages are independent
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
from electoral.stages import (
    build_baseline_portfolio,
    build_llm_finetune,
    build_optimization,
    build_sentiment_data,
    build_voter_panel,
    run_simulations,
)


@task(retries=2, retry_delay_seconds=30)
def task_voter_panel(config: PipelineConfig):
    return build_voter_panel(config)


@task(retries=2, retry_delay_seconds=30)
def task_baseline_portfolio(config: PipelineConfig, panel):
    return build_baseline_portfolio(config, panel)


@task(retries=2, retry_delay_seconds=30)
def task_sentiment_data(config: PipelineConfig, panel):
    return build_sentiment_data(config, panel)


@task(retries=1, retry_delay_seconds=60)
def task_llm_finetune(config: PipelineConfig, sentiment):
    return build_llm_finetune(config, sentiment)


@flow(name="electoral-equilibrium")
def run_pipeline(config_path: str) -> dict:
    """Full Electoral Equilibrium pipeline flow.

    Returns a dict of all stage artifacts for testing/inspection.
    """
    config = PipelineConfig.from_json(config_path)
    config.validate()

    panel = task_voter_panel(config)
    baseline = task_baseline_portfolio(config, panel)
    sentiment = task_sentiment_data(config, panel)

    # LLM fine-tuning is skipped in continuous mode (adapter already exists)
    if config.pipeline_mode == "historical":
        finetune = task_llm_finetune(config, sentiment)
    else:
        finetune = None

    return {
        "voter_panel": panel,
        "baseline_portfolio": baseline,
        "sentiment_data": sentiment,
        "llm_finetune": finetune,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Electoral Equilibrium pipeline")
    parser.add_argument(
        "--config",
        default="configs/smoke.json",
        help="Path to PipelineConfig JSON (default: configs/smoke.json)",
    )
    args = parser.parse_args()

    if not Path(args.config).exists():
        print(f"Error: config not found: {args.config}", file=sys.stderr)
        sys.exit(1)

    if _HAS_PREFECT:
        run_pipeline(config_path=args.config)
    else:
        # Fallback: run as plain Python without Prefect
        config = PipelineConfig.from_json(args.config)
        config.validate()
        panel = build_voter_panel(config)
        build_baseline_portfolio(config, panel)
        build_sentiment_data(config, panel)
        print(f"✓ Smoke pipeline completed (run_key={config.run_key!r})")


if __name__ == "__main__":
    main()