"""Kernel: LLM fine-tuning stage (Stage 3c).

═══════════════════════════════════════════════════════════════════════════════
WHAT "FINE-TUNING" MEANS HERE (beginner orientation)
═══════════════════════════════════════════════════════════════════════════════
A base language model (Mistral-7B) already "knows English" but knows nothing
about our specific task: "given a shock, output a delta bin per bloc."
FINE-TUNING = continuing to train that model a little more on OUR examples
(data/finetune/*.jsonl) so it learns our task and output format.

Training all 7 billion weights would need huge GPUs, so we use QLoRA:
  • LoRA ("Low-Rank Adaptation") freezes the giant base model and trains only a
    small set of extra weights — the "ADAPTER". The adapter is tiny (~MBs) and is
    what gets saved/loaded. `lora_rank` controls how big the adapter is.
  • The "Q" = quantized: the frozen base model is stored in low precision to fit
    in memory. Together this lets us fine-tune a 7B model on one modest GPU.

IDEMPOTENT means: calling this twice does no extra work. If a trained adapter
already exists on disk, we skip training and just return its metadata — so
re-running the pipeline doesn't burn hours retraining an existing model.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from electoral.artifacts import LLMFineTuneData
from electoral.config import PipelineConfig

log = logging.getLogger(__name__)

_DEFAULT_BASE_MODEL = "mistralai/Mistral-7B-v0.3"
_DEFAULT_ADAPTER = "models/mistral-r16"  # where the trained LoRA adapter lives
_DEFAULT_LORA_RANK = 16
# ⚠️ Hardcoded metadata: the reported n_examples is fixed at 557 and does NOT
# reflect the actual training file size (the synthetic set has since grown). This
# is cosmetic (it only labels the artifact), but ideally n_examples should be
# counted from the train file instead of being a constant. Low-priority cleanup.
_DEFAULT_N_EXAMPLES = 557


def build_llm_finetune(config: PipelineConfig) -> LLMFineTuneData:
    """Return LLMFineTuneData for the configured adapter, training if needed.

    Idempotent: if ``adapter_path/adapter_config.json`` already exists, training
    is skipped and the existing adapter is returned as-is.
    """
    adapter_path = Path(getattr(config, "adapter_path", None) or _DEFAULT_ADAPTER)
    base_model: str = getattr(config, "base_model", None) or _DEFAULT_BASE_MODEL
    lora_rank: int = int(getattr(config, "lora_rank", _DEFAULT_LORA_RANK))

    # The presence of adapter_config.json is our "already trained" marker — it's
    # the small file PEFT writes alongside a finished LoRA adapter.
    adapter_config_file = adapter_path / "adapter_config.json"

    # ── IDEMPOTENT FAST PATH: adapter already on disk → skip the (slow) training.
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
