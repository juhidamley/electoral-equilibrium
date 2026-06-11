"""QLoRA fine-tuning of Mistral 7B for political shock → delta-bin prediction.

Run on CMC HPC A100 via:
    python -m electoral.llm.trainer \\
        --config configs/base.json \\
        --train-data data/finetune/train.jsonl \\
        --eval-data  data/finetune/eval.jsonl \\
        --output-dir $SCRATCH/adapters/mistral-7b-electoral \\
        --lora-rank 16 \\
        --epochs 3

Requires: transformers, peft, bitsandbytes, torch.
These are NOT in pyproject.toml's default deps because they are HPC-only.
Install on HPC with: pip install transformers peft bitsandbytes

The prompt format is Mistral instruction template:
  [INST] ... shock description + party ... [/INST]
  JSON object with delta_bins for all 15 demographic blocs.

Adapter saved to --output-dir along with trainer_state.json.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from electoral.core.rng import derive_seed

log = logging.getLogger(__name__)


# ── TrainConfig ───────────────────────────────────────────────────────────────


@dataclass
class TrainConfig:
    base_model: str = "mistralai/Mistral-7B-v0.3"
    lora_rank: int = 16
    lora_alpha: int = 32
    learning_rate: float = 2e-4
    epochs: int = 3
    batch_size: int = 4
    train_path: str = "data/finetune/train.jsonl"
    eval_path: str = "data/finetune/eval.jsonl"
    output_dir: str = "models/mistral-r16"
    backend: str = "hpc"


def load_config(path: str | Path) -> TrainConfig:
    """Read a TrainConfig JSON file. Unknown keys are silently ignored."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    fields = {f.name for f in dataclasses.fields(TrainConfig)}
    return TrainConfig(**{k: v for k, v in data.items() if k in fields})


def save_config(config: TrainConfig, path: str | Path) -> None:
    """Write a TrainConfig to JSON."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(dataclasses.asdict(config), indent=2),
        encoding="utf-8",
    )


# ── Prompt template ───────────────────────────────────────────────────────────

_SYSTEM_PROMPT_TEMPLATE = (
    "You are a political science model that predicts how shock events affect "
    "demographic bloc support for the {party} candidate. Given RoBERTa sentiment "
    "scores from news and social media, output delta bins for each demographic stratum."
)

_BLOCS_ORDERED = [
    "african_american",
    "latino",
    "asian",
    "white",
    "other_race",
    "evangelical",
    "catholic",
    "protestant",
    "secular",
    "jewish",
    "muslim",
    "other_rel",
    "women",
    "men",
    "other_gender",
]


_RACE_BLOCS = _BLOCS_ORDERED[:5]
_RELIGION_BLOCS = _BLOCS_ORDERED[5:12]
_GENDER_BLOCS = _BLOCS_ORDERED[12:]


def format_prompt(example: dict[str, Any]) -> str:
    """Build the [INST]...[/INST] prompt from a finetune record.

    Six required elements:
      (i)   System context establishing political science framing and party.
      (ii)  Event description, year, and source.
      (iii) News RoBERTa scores as a labeled JSON block.
      (iv)  Social bio-weighted scores as a labeled JSON block.
      (v)   Year and source (included in the event block).
      (vi)  Closing instruction with exact canonical keys for all three strata,
            the nine bin labels, delta_eff definition, and party substituted from
            example["party"] — never hardcoded.

    Accepts both new schema (news_roberta_scores, social_roberta_scores keys)
    and legacy schema (missing score fields → empty dicts).
    """
    party = example.get("party", "democrat")
    description = example.get("description", "")
    year = str(example.get("year", ""))
    source = example.get("source", "")
    news_scores = example.get("news_roberta_scores", {})
    social_scores = example.get("social_roberta_scores", {})

    # (i) System context — party name substituted, never hardcoded
    system = _SYSTEM_PROMPT_TEMPLATE.format(party=party)

    # (ii) + (v) Event description, year, source
    event_lines = [f"## Event\n{description}"]
    if year:
        event_lines.append(f"Year: {year}")
    if source:
        event_lines.append(f"Source: {source}")
    event_block = "\n".join(event_lines)

    # (iii) News RoBERTa scores
    news_block = (
        "## News RoBERTa Scores (per bloc)\n"
        + json.dumps(news_scores, indent=2, ensure_ascii=False)
    )

    # (iv) Social bio-weighted scores
    social_block = (
        "## Social Bio-Weighted Scores (per bloc)\n"
        + json.dumps(social_scores, indent=2, ensure_ascii=False)
    )

    # (vi) Closing instruction — exact canonical keys, nine bin labels, party substituted
    _BIN_VALUES = (
        "strong_neg, mod_neg, mild_neg, slight_neg, neutral, "
        "slight_pos, mild_pos, mod_pos, strong_pos"
    )
    closing = (
        f'You are analyzing the impact on the {party} candidate\'s coalition.\n'
        "Output only a JSON object with the following keys:\n"
        '  "delta_bins_race": keys african_american, latino, asian, white, other_race\n'
        '  "delta_bins_religion": keys evangelical, catholic, protestant, secular, '
        'jewish, muslim, other_rel\n'
        '  "delta_bins_gender": keys women, men, other_gender\n'
        '  "delta_eff": float scalar\n'
        f"Values for bin fields must be one of: {_BIN_VALUES}.\n"
        "delta_eff is a float: the pipeline verifies it matches "
        "λ₁·Σ(w_i·Δμ_i^race) + λ₂·Σ(v_R·Δμ_R^rel) + λ₃·Σ(g_G·Δμ_G^gen).\n"
        "For non-gendered shocks, gender bins may be identical across women/men."
    )

    return (
        f"[INST] {system}\n\n"
        f"{event_block}\n\n"
        f"{news_block}\n\n"
        f"{social_block}\n\n"
        f"{closing} [/INST]"
    )


def format_completion(record: dict[str, Any]) -> str:
    """Build the JSON completion target from a finetune record.

    Supports both new schema (delta_bins_race/religion/gender + delta_eff)
    and legacy flat schema (delta_bins flat dict).
    """
    if "delta_bins_race" in record:
        output = {
            "delta_bins_race": {
                k: record["delta_bins_race"][k]
                for k in _RACE_BLOCS
                if k in record["delta_bins_race"]
            },
            "delta_bins_religion": {
                k: record["delta_bins_religion"][k]
                for k in _RELIGION_BLOCS
                if k in record.get("delta_bins_religion", {})
            },
            "delta_bins_gender": {
                k: record["delta_bins_gender"][k]
                for k in _GENDER_BLOCS
                if k in record.get("delta_bins_gender", {})
            },
            "delta_eff": float(record.get("delta_eff", 0.0)),
        }
    else:
        flat = record.get("delta_bins", {})
        output = {
            "delta_bins_race": {k: flat[k] for k in _RACE_BLOCS if k in flat},
            "delta_bins_religion": {k: flat[k] for k in _RELIGION_BLOCS if k in flat},
            "delta_bins_gender": {k: flat[k] for k in _GENDER_BLOCS if k in flat},
            "delta_eff": 0.0,
        }
    return json.dumps(output, ensure_ascii=False)


def format_training_text(record: dict[str, Any]) -> str:
    """Format a finetune JSONL record as a full <s>prompt\ncompletion</s> string."""
    return f"<s>{format_prompt(record)}\n{format_completion(record)}</s>"


# ── Data loading ──────────────────────────────────────────────────────────────


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ── Evaluation ───────────────────────────────────────────────────────────────


def _eval_mae(model: Any, tokenizer: Any, eval_records: list[dict], device: str) -> float:
    """Greedy-decode predictions on eval set and return MAE in delta units."""
    from electoral.llm.eval import mae_in_delta_units
    from electoral.core.types import DELTA_BINS

    total_mae = 0.0
    n = 0

    model.eval()
    import torch

    with torch.no_grad():
        for rec in eval_records:
            prompt = format_prompt(rec)
            inputs = tokenizer(f"<s>{prompt}\n", return_tensors="pt").to(device)
            out = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                temperature=0.0,
                pad_token_id=tokenizer.eos_token_id,
            )
            generated = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            try:
                pred = json.loads(generated.strip())
                if not isinstance(pred, dict):
                    continue
                # Flatten nested stratum dicts to {bloc: bin} for MAE comparison
                if "delta_bins_race" in pred:
                    flat_pred = {
                        **pred.get("delta_bins_race", {}),
                        **pred.get("delta_bins_religion", {}),
                        **pred.get("delta_bins_gender", {}),
                    }
                else:
                    flat_pred = pred
                flat_pred = {k: v for k, v in flat_pred.items() if v in DELTA_BINS}
                # Build flat true bins from whichever schema the record uses
                if "delta_bins_race" in rec:
                    flat_true = {
                        **rec.get("delta_bins_race", {}),
                        **rec.get("delta_bins_religion", {}),
                        **rec.get("delta_bins_gender", {}),
                    }
                else:
                    flat_true = rec.get("delta_bins", {})
                if flat_pred and flat_true:
                    total_mae += mae_in_delta_units(flat_pred, flat_true)
                    n += 1
            except (json.JSONDecodeError, ValueError, KeyError):
                continue

    return total_mae / n if n > 0 else float("inf")


# ── Training ──────────────────────────────────────────────────────────────────


def train(
    train_path: Path,
    eval_path: Path,
    output_dir: Path,
    base_model: str,
    lora_rank: int,
    lora_alpha: int,
    epochs: int,
    batch_size: int,
    grad_accum: int,
    lr: float,
    seed: int,
) -> dict[str, Any]:
    """Run QLoRA fine-tuning. Returns trainer_state dict."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
        from transformers import DataCollatorForLanguageModeling
        from peft import LoraConfig, get_peft_model, TaskType
        from datasets import Dataset
    except ImportError as exc:
        raise ImportError(
            "QLoRA training requires: transformers, peft, bitsandbytes, datasets. "
            "Install with: pip install transformers peft bitsandbytes datasets"
        ) from exc

    try:
        from transformers import BitsAndBytesConfig
        import bitsandbytes  # noqa: F401
        use_4bit = True
        log.info("bitsandbytes available — using 4-bit NF4 quantization")
    except ImportError:
        use_4bit = False
        log.warning("bitsandbytes not available — training in full precision (slower)")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Training on device: %s", device)

    # ── Load data ─────────────────────────────────────────────────────────────
    train_records = load_jsonl(train_path)
    eval_records = load_jsonl(eval_path)
    log.info("Train: %d examples, Eval: %d examples", len(train_records), len(eval_records))

    train_texts = [format_training_text(r) for r in train_records]

    # ── Load tokenizer ────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ── Tokenize dataset ──────────────────────────────────────────────────────
    def tokenize(example: dict) -> dict:
        return tokenizer(
            example["text"],
            truncation=True,
            max_length=512,
            padding=False,
        )

    hf_dataset = Dataset.from_dict({"text": train_texts})
    tokenized = hf_dataset.map(tokenize, remove_columns=["text"])

    # ── Load model ────────────────────────────────────────────────────────────
    quant_config = None
    if use_4bit:
        from transformers import BitsAndBytesConfig
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=quant_config,
        device_map="auto" if device == "cuda" else None,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )

    # ── LoRA config ───────────────────────────────────────────────────────────
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── Training args ─────────────────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        fp16=(device == "cuda"),
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="no",  # manual eval below
        seed=derive_seed(seed, "llm_finetune"),
        report_to="none",
        dataloader_drop_last=False,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=data_collator,
    )

    log.info("Starting training...")
    train_result = trainer.train()
    log.info("Training complete. Loss: %.4f", train_result.training_loss)

    # ── Save adapter ──────────────────────────────────────────────────────────
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    log.info("Adapter saved to %s", output_dir)

    # ── Evaluate MAE ─────────────────────────────────────────────────────────
    log.info("Evaluating on %d examples...", len(eval_records))
    eval_mae = _eval_mae(model, tokenizer, eval_records, device)
    log.info("Eval MAE: %.4f", eval_mae)

    trainer_state = {
        "base_model": base_model,
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
        "epochs": epochs,
        "n_train": len(train_records),
        "n_eval": len(eval_records),
        "training_loss": float(train_result.training_loss),
        "eval_mae": float(eval_mae),
        "seed": seed,
    }
    state_path = output_dir / "trainer_state.json"
    state_path.write_text(json.dumps(trainer_state, indent=2), encoding="utf-8")
    log.info("trainer_state.json written to %s", state_path)

    return trainer_state


def train_hpc(config: TrainConfig, grad_accum: int = 4, seed: int = 42) -> dict[str, Any]:
    """USC Laguna / HPC backend: wraps train() with TrainConfig fields.

    grad_accum and seed are not stored in TrainConfig (they are infrastructure
    concerns, not model hyperparameters). Pass them explicitly or accept defaults.
    """
    log.info(
        "train_hpc: starting  model=%s  rank=%d  output=%s",
        config.base_model,
        config.lora_rank,
        config.output_dir,
    )
    state = train(
        train_path=Path(config.train_path),
        eval_path=Path(config.eval_path),
        output_dir=Path(config.output_dir),
        base_model=config.base_model,
        lora_rank=config.lora_rank,
        lora_alpha=config.lora_alpha,
        epochs=config.epochs,
        batch_size=config.batch_size,
        grad_accum=grad_accum,
        lr=config.learning_rate,
        seed=seed,
    )
    log.info(
        "train_hpc: complete  output=%s  loss=%.4f  eval_mae=%.4f",
        config.output_dir,
        state.get("training_loss", float("nan")),
        state.get("eval_mae", float("nan")),
    )
    return state


def train_mlx(config: TrainConfig) -> dict[str, Any]:
    """M5 MacBook backend: runs mlx_lm.lora via subprocess.

    Data directory: data/finetune/mlx/ (must contain train.jsonl and valid.jsonl
    in the flat text format expected by mlx_lm — one training string per line).
    Adapter saved to config.output_dir.

    mlx_lm takes --iters (not --epochs). Iteration count is derived from the
    training file line count: iters = epochs * max(1, n_lines // batch_size).
    """
    import subprocess

    mlx_data_dir = "data/finetune/mlx"
    train_file = Path(mlx_data_dir) / "train.jsonl"

    # Estimate iterations from training file size
    try:
        n_lines = sum(1 for ln in open(train_file, encoding="utf-8") if ln.strip())
        steps_per_epoch = max(1, n_lines // config.batch_size)
    except OSError:
        log.warning("train_mlx: could not read %s — defaulting to 100 iters/epoch", train_file)
        steps_per_epoch = 100
    iters = config.epochs * steps_per_epoch

    adapter_path = Path(config.output_dir)
    adapter_path.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "mlx_lm.lora",
        "--model", config.base_model,
        "--train",
        "--data", mlx_data_dir,
        "--batch-size", str(config.batch_size),
        "--iters", str(iters),
        "--learning-rate", str(config.learning_rate),
        "--lora-layers", str(config.lora_rank),
        "--adapter-path", str(adapter_path),
    ]

    log.info(
        "train_mlx: starting  model=%s  rank=%d  iters=%d  output=%s",
        config.base_model,
        config.lora_rank,
        iters,
        adapter_path,
    )
    log.info("train_mlx: command: %s", " ".join(cmd))

    result = subprocess.run(cmd, check=False)

    state: dict[str, Any] = {
        "base_model": config.base_model,
        "lora_rank": config.lora_rank,
        "lora_alpha": config.lora_alpha,
        "epochs": config.epochs,
        "iters": iters,
        "returncode": result.returncode,
        "output_dir": str(adapter_path),
    }

    if result.returncode != 0:
        log.error("train_mlx: mlx_lm.lora exited with code %d", result.returncode)
    else:
        log.info("train_mlx: complete  output=%s", adapter_path)

    state_path = adapter_path / "trainer_state.json"
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    return state


# ── CLI ───────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="QLoRA fine-tuning of Mistral 7B for shock→delta-bin prediction."
    )
    parser.add_argument("--config", required=True,
                        help="Path to a TrainConfig JSON (train_r16.json) or pipeline base.json.")
    parser.add_argument("--backend", default=None,
                        help="Override TrainConfig.backend (mlx | hpc).")
    parser.add_argument("--train-data", default=None, help="Override train_path from config.")
    parser.add_argument("--eval-data", default=None, help="Override eval_path from config.")
    parser.add_argument("--output-dir", default=None, help="Override output_dir from config.")
    parser.add_argument("--lora-rank", type=int, default=None)
    parser.add_argument("--lora-alpha", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--base-model", default=None,
                        help="Override base_model from config.")
    args = parser.parse_args(argv)

    # Load config — TrainConfig JSON (has lora_rank) or pipeline base.json (has seed only).
    seed = 42
    try:
        with open(args.config, encoding="utf-8") as f:
            raw_cfg = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("Could not read %s: %s — using defaults", args.config, exc)
        raw_cfg = {}

    if "lora_rank" in raw_cfg:
        tcfg = load_config(args.config)
    else:
        seed = raw_cfg.get("seed", 42)
        tcfg = TrainConfig()

    # CLI flags override config values when explicitly provided.
    if args.backend is not None:
        tcfg.backend = args.backend
    if args.train_data is not None:
        tcfg.train_path = args.train_data
    if args.eval_data is not None:
        tcfg.eval_path = args.eval_data
    if args.output_dir is not None:
        tcfg.output_dir = args.output_dir
    if args.lora_rank is not None:
        tcfg.lora_rank = args.lora_rank
    if args.lora_alpha is not None:
        tcfg.lora_alpha = args.lora_alpha
    if args.epochs is not None:
        tcfg.epochs = args.epochs
    if args.batch_size is not None:
        tcfg.batch_size = args.batch_size
    if args.lr is not None:
        tcfg.learning_rate = args.lr
    if args.base_model is not None:
        tcfg.base_model = args.base_model

    log.info(
        "TrainConfig: model=%s rank=%d alpha=%d lr=%g epochs=%d batch=%d backend=%s",
        tcfg.base_model, tcfg.lora_rank, tcfg.lora_alpha,
        tcfg.learning_rate, tcfg.epochs, tcfg.batch_size, tcfg.backend,
    )

    try:
        if tcfg.backend == "mlx":
            state = train_mlx(tcfg)
        else:
            # "hpc" or any unrecognised value — default to HPC/transformers path
            if tcfg.backend not in ("hpc", "mlx"):
                log.warning("Unknown backend '%s' — falling back to hpc", tcfg.backend)
            state = train_hpc(tcfg, grad_accum=args.grad_accum, seed=seed)
    except ImportError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    mae = state.get("eval_mae", float("inf"))
    if mae != float("inf") and mae > 0.04:
        log.warning("Held-out MAE %.4f > 0.04 — consider submitting rank-32 job", mae)
        return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
