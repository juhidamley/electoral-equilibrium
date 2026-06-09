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
import json
import logging
import sys
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# ── Prompt template ───────────────────────────────────────────────────────────

_SYSTEM_PROMPT = (
    "You are an expert political scientist specialising in US electoral demographics. "
    "Given a political shock event and the party whose voters are being analysed, "
    "predict the directional change in Democratic vote share for each demographic "
    "group using exactly one of these magnitude tokens: "
    "strong_neg, mod_neg, mild_neg, slight_neg, neutral, "
    "slight_pos, mild_pos, mod_pos, strong_pos."
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


def format_prompt(description: str, party: str) -> str:
    """Build the [INST]...[/INST] instruction prompt."""
    return (
        f"[INST] {_SYSTEM_PROMPT}\n\n"
        f"Shock event: {description}\n"
        f"Party perspective: {party}\n\n"
        "Output a JSON object with keys for each demographic group and a delta bin token "
        "as the value. [/INST]"
    )


def format_completion(delta_bins: dict[str, str]) -> str:
    """Build the expected JSON completion string."""
    ordered = {k: delta_bins[k] for k in _BLOCS_ORDERED if k in delta_bins}
    return json.dumps(ordered, ensure_ascii=False)


def format_training_text(record: dict[str, Any]) -> str:
    """Format a synthetic JSONL record as a full training string."""
    prompt = format_prompt(record["description"], record["party"])
    completion = format_completion(record["delta_bins"])
    return f"<s>{prompt}\n{completion}</s>"


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
            prompt = format_prompt(rec["description"], rec["party"])
            inputs = tokenizer(f"<s>{prompt}\n", return_tensors="pt").to(device)
            out = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id,
            )
            generated = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            try:
                pred_bins = json.loads(generated.strip())
                if not isinstance(pred_bins, dict):
                    continue
                # Validate tokens
                pred_bins = {k: v for k, v in pred_bins.items() if v in DELTA_BINS}
                true_bins = rec.get("delta_bins", {})
                if pred_bins and true_bins:
                    total_mae += mae_in_delta_units(pred_bins, true_bins)
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
        seed=seed,
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


# ── CLI ───────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="QLoRA fine-tuning of Mistral 7B for shock→delta-bin prediction."
    )
    parser.add_argument("--config", required=True, help="Path to configs/base.json")
    parser.add_argument("--train-data", required=True, help="JSONL training set")
    parser.add_argument("--eval-data", required=True, help="JSONL evaluation set")
    parser.add_argument("--output-dir", required=True, help="Directory for adapter weights")
    parser.add_argument("--lora-rank", type=int, default=16, help="LoRA rank (default 16)")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha (default 32)")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs (default 3)")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument(
        "--base-model",
        default="mistralai/Mistral-7B-v0.3",
        help="HuggingFace model ID for the base model",
    )
    args = parser.parse_args(argv)

    # Load seed from config (or use 42)
    seed = 42
    try:
        with open(args.config, encoding="utf-8") as f:
            cfg = json.load(f)
        seed = cfg.get("seed", 42)
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("Could not load config %s: %s — using seed=42", args.config, exc)

    try:
        state = train(
            train_path=Path(args.train_data),
            eval_path=Path(args.eval_data),
            output_dir=Path(args.output_dir),
            base_model=args.base_model,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            epochs=args.epochs,
            batch_size=args.batch_size,
            grad_accum=args.grad_accum,
            lr=args.lr,
            seed=seed,
        )
    except ImportError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    mae = state.get("eval_mae", float("inf"))
    if mae > 0.04:
        log.warning("Held-out MAE %.4f > 0.04 — consider submitting rank-32 job", mae)
        return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
