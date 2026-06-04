"""elasticity: Sentiment-to-vote-share regression and fine-tuning dataset assembly.

Two components:
  1. estimate_elasticity() — OLS regression: sentiment score → vote_share_delta
     Produces β coefficients (elasticity) per bloc.
     Use these to calibrate how much RoBERTa sentiment predicts actual vote shifts.

  2. assemble_finetune_dataset() — Converts RoBERTa scores to LLM fine-tuning JSONL.
     Maps each shock's per-bloc sentiment score to a 9-token delta bin label.
     Writes instruction-response pairs for Mistral/Qwen QLoRA (Stage 1 training data).

Delta bin taxonomy (9 tokens, CLAUDE.md):
    strong_neg  [-1.0, -0.50)
    mod_neg     [-0.50, -0.30)
    mild_neg    [-0.30, -0.15)
    slight_neg  [-0.15, -0.05)
    neutral     [-0.05,  0.05]
    slight_pos   (0.05,  0.15]
    mild_pos     (0.15,  0.30]
    mod_pos      (0.30,  0.50]
    strong_pos   (0.50,  1.0]
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from electoral.artifacts import LLMFineTuneData, SentimentData
from electoral.core.types import CANONICAL_GENDERS, CANONICAL_RACES, CANONICAL_RELIGIONS

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_FINETUNE_DIR = _REPO_ROOT / "data" / "finetune"

_ALL_BLOCS: list[str] = (
    list(CANONICAL_RACES) + list(CANONICAL_RELIGIONS) + list(CANONICAL_GENDERS)
)

# 9-token bin thresholds (lower bound inclusive, upper bound exclusive except neutral)
_BIN_THRESHOLDS: list[tuple[float, float, str]] = [
    (-1.001, -0.50, "strong_neg"),
    (-0.50, -0.30, "mod_neg"),
    (-0.30, -0.15, "mild_neg"),
    (-0.15, -0.05, "slight_neg"),
    (-0.05, 0.05, "neutral"),
    (0.05, 0.15, "slight_pos"),
    (0.15, 0.30, "mild_pos"),
    (0.30, 0.50, "mod_pos"),
    (0.50, 1.001, "strong_pos"),
]


def score_to_bin(score: float) -> str:
    """Map a sentiment score in [-1, 1] to a 9-token delta bin label.

    Scores outside [-1, 1] are clamped to the nearest bin.
    """
    for lo, hi, label in _BIN_THRESHOLDS:
        if lo <= score < hi:
            return label
    return "strong_pos" if score >= 1.0 else "strong_neg"


def estimate_elasticity(
    sentiment_data: SentimentData,
    vote_deltas: dict[str, dict[str, float]],
) -> dict[str, float]:
    """OLS regression of RoBERTa sentiment scores → historical vote_share_deltas.

    Fits one β coefficient per bloc using all available (shock, bloc) pairs where
    both sentiment and vote delta are observed. Blocs with fewer than 3 data points
    are skipped (coefficient → NaN).

    Args:
        sentiment_data: RoBERTa sentiment scores per bloc per shock.
        vote_deltas: vote_deltas[bloc_id][shock_id] = actual Δvote_share.
            These come from the historical panel: delta between the election cycle
            immediately before and after each shock event.

    Returns:
        Dict mapping bloc_id → β coefficient (elasticity).
        NaN entries indicate insufficient data for that bloc.
    """
    elasticities: dict[str, float] = {}
    shocks = sentiment_data.shocks

    for bloc in _ALL_BLOCS:
        x_vals: list[float] = []
        y_vals: list[float] = []

        bloc_sentiment = sentiment_data.scores.get(bloc, {})
        bloc_deltas = vote_deltas.get(bloc, {})

        for shock_id in shocks:
            x = bloc_sentiment.get(shock_id)
            y = bloc_deltas.get(shock_id)
            if x is None or y is None:
                continue
            if not (np.isfinite(x) and np.isfinite(y)):
                continue
            x_vals.append(x)
            y_vals.append(y)

        if len(x_vals) < 3:
            logger.debug(
                "elasticity: bloc='%s' has only %d observations (< 3); skipping.",
                bloc, len(x_vals),
            )
            elasticities[bloc] = float("nan")
            continue

        x_arr = np.array(x_vals, dtype=float)
        y_arr = np.array(y_vals, dtype=float)

        # OLS: β = (XᵀX)⁻¹Xᵀy with intercept
        x_design = np.column_stack([np.ones_like(x_arr), x_arr])
        try:
            coef, _, _, _ = np.linalg.lstsq(x_design, y_arr, rcond=None)
            beta = float(coef[1])
        except np.linalg.LinAlgError as exc:
            logger.warning("OLS failed for bloc '%s': %s", bloc, exc)
            beta = float("nan")

        elasticities[bloc] = beta
        logger.debug("elasticity: bloc='%s' β=%.4f n=%d", bloc, beta, len(x_vals))

    n_valid = sum(1 for v in elasticities.values() if np.isfinite(v))
    logger.info(
        "estimate_elasticity: %d/%d blocs have valid β coefficients.",
        n_valid, len(_ALL_BLOCS),
    )
    return elasticities


def assemble_finetune_dataset(
    sentiment_data: SentimentData,
    shocks_config: list[dict[str, Any]],
    output_path: str | Path | None = None,
    base_model: str = "mistralai/Mistral-7B-v0.3",
    lora_rank: int = 16,
    cycles_used: list[int] | None = None,
) -> LLMFineTuneData:
    """Convert RoBERTa sentiment scores to Mistral/Qwen QLoRA fine-tuning JSONL.

    Each shock produces one instruction-response example per stratum:
      instruction: "Shock: {description}. Predict partisan sentiment delta bins..."
      output: "Race blocs:\n  african_american: slight_pos\n  ..."

    Shocks without a text description in shocks_config are skipped.

    Args:
        sentiment_data: SentimentData from score_news_for_shocks().
        shocks_config: Parsed contents of configs/shocks.json.
        output_path: Where to write train.jsonl. Defaults to data/finetune/train.jsonl.
        base_model: Base model identifier for LLMFineTuneData.
        lora_rank: LoRA rank parameter.
        cycles_used: Election cycles whose data contributed to sentiment scores.

    Returns:
        LLMFineTuneData artifact (does NOT write to disk the artifact JSON itself;
        that is handled by the pipeline stage).
    """
    output_path = Path(output_path) if output_path else _DEFAULT_FINETUNE_DIR / "train.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build shock description lookup
    shock_desc: dict[str, str] = {}
    for entry in shocks_config:
        sid = entry.get("id", "")
        desc = entry.get("description") or entry.get("text") or entry.get("name") or ""
        if sid and desc:
            shock_desc[sid] = desc

    n_examples = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for shock_id in sentiment_data.shocks:
            description = shock_desc.get(shock_id)
            if not description:
                logger.debug(
                    "assemble_finetune_dataset: no description for shock '%s'; skipping.",
                    shock_id,
                )
                continue

            instruction = (
                f"Shock event: {description}\n"
                "Based on social media sentiment and historical patterns, "
                "predict the partisan loyalty delta bin for each voter bloc. "
                "Use one of: strong_neg, mod_neg, mild_neg, slight_neg, neutral, "
                "slight_pos, mild_pos, mod_pos, strong_pos."
            )

            # Build output string per stratum
            lines: list[str] = []

            lines.append("Race blocs:")
            for bloc in CANONICAL_RACES:
                score = sentiment_data.scores.get(bloc, {}).get(shock_id, 0.0)
                lines.append(f"  {bloc}: {score_to_bin(score)}")

            lines.append("Religion blocs:")
            for bloc in CANONICAL_RELIGIONS:
                score = sentiment_data.scores.get(bloc, {}).get(shock_id, 0.0)
                lines.append(f"  {bloc}: {score_to_bin(score)}")

            lines.append("Gender blocs:")
            for bloc in CANONICAL_GENDERS:
                score = sentiment_data.scores.get(bloc, {}).get(shock_id, 0.0)
                lines.append(f"  {bloc}: {score_to_bin(score)}")

            output_text = "\n".join(lines)

            example = {
                "instruction": instruction,
                "input": "",
                "output": output_text,
                "shock_id": shock_id,
                "metadata": {"model": sentiment_data.model},
            }
            f.write(json.dumps(example, ensure_ascii=False))
            f.write("\n")
            n_examples += 1

    logger.info(
        "assemble_finetune_dataset: wrote %d examples to %s", n_examples, output_path
    )

    return LLMFineTuneData(
        base_model=base_model,
        lora_rank=lora_rank,
        n_examples=n_examples,
        cycles_used=sorted(set(cycles_used)) if cycles_used else [],
        adapter_path=None,
    )
