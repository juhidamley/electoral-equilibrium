"""Kernel: LLM fine-tuning stage.

build_llm_finetune() is idempotent — if an adapter already exists at the
configured path it returns immediately without re-training.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from electoral.artifacts import LLMFineTuneData
from electoral.config import PipelineConfig

log = logging.getLogger(__name__)

_DEFAULT_BASE_MODEL = "mistralai/Mistral-7B-v0.3"
_DEFAULT_ADAPTER = "models/mistral-r16"
_DEFAULT_LORA_RANK = 16
_DEFAULT_N_EXAMPLES = 557


def build_llm_finetune(config: PipelineConfig) -> LLMFineTuneData:
    """Return LLMFineTuneData for the configured adapter, training if needed.

    Idempotent: if ``adapter_path/adapter_config.json`` already exists, training
    is skipped and the existing adapter is returned as-is.
    """
    adapter_path = Path(
        getattr(config, "adapter_path", None) or _DEFAULT_ADAPTER
    )
    base_model: str = getattr(config, "base_model", None) or _DEFAULT_BASE_MODEL
    lora_rank: int = int(getattr(config, "lora_rank", _DEFAULT_LORA_RANK))

    adapter_config_file = adapter_path / "adapter_config.json"

    if adapter_path.exists() and adapter_config_file.exists():
        log.info("adapter found at %s, skipping training", adapter_path)
        _log_eval_mae(adapter_path)
        return LLMFineTuneData(
            base_model=base_model,
            lora_rank=lora_rank,
            n_examples=_DEFAULT_N_EXAMPLES,
            cycles_used=[2020],
            adapter_path=str(adapter_path),
        )

    log.info("adapter not found at %s — starting training", adapter_path)
    _run_training(config, adapter_path, base_model, lora_rank)

    _log_eval_mae(adapter_path)
    return LLMFineTuneData(
        base_model=base_model,
        lora_rank=lora_rank,
        n_examples=_DEFAULT_N_EXAMPLES,
        cycles_used=[2020],
        adapter_path=str(adapter_path),
    )


def _run_training(
    config: PipelineConfig,
    adapter_path: Path,
    base_model: str,
    lora_rank: int,
) -> None:
    from electoral.llm.trainer import TrainConfig, train_hpc

    train_cfg = TrainConfig(
        base_model=base_model,
        lora_rank=lora_rank,
        lora_alpha=lora_rank * 2,
        learning_rate=2e-4,
        epochs=int(getattr(config, "epochs", 3)),
        batch_size=int(getattr(config, "batch_size", 4)),
        train_path=str(getattr(config, "train_path", "data/finetune/train.jsonl")),
        eval_path=str(getattr(config, "eval_path", "data/finetune/eval.jsonl")),
        output_dir=str(adapter_path),
        backend="hpc",
    )
    train_hpc(train_cfg)


def _log_eval_mae(adapter_path: Path) -> None:
    """Log eval MAE from trainer_state.json if present."""
    state_file = adapter_path / "trainer_state.json"
    if not state_file.exists():
        return
    try:
        state = json.loads(state_file.read_text(encoding="utf-8"))
        best_metric = state.get("best_metric")
        if best_metric is not None:
            log.info("adapter eval MAE (from trainer_state.json): %.4f", best_metric)
    except Exception as exc:
        log.debug("could not read trainer_state.json: %s", exc)
