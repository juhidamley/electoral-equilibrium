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
    neutral     [-0.05,  0.05)
    slight_pos  [0.05,  0.15)
    mild_pos    [0.15,  0.30)
    mod_pos     [0.30,  0.50)
    strong_pos  [0.50,  1.0]
"""

from __future__ import annotations

import dataclasses
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from electoral.artifacts import (
    LLMFineTuneData,
    PredictionMarketData,
    SentimentData,
    SocialMediaSentimentData,
)
from electoral.core.types import CANONICAL_GENDERS, CANONICAL_RACES, CANONICAL_RELIGIONS

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_FINETUNE_DIR = _REPO_ROOT / "data" / "finetune"

_ALL_BLOCS: list[str] = list(CANONICAL_RACES) + list(CANONICAL_RELIGIONS) + list(CANONICAL_GENDERS)

BIN_MIDPOINTS: dict[str, float] = {
    "strong_neg": -0.120,
    "mod_neg": -0.070,
    "mild_neg": -0.035,
    "slight_neg": -0.012,
    "neutral": 0.000,
    "slight_pos": +0.012,
    "mild_pos": +0.035,
    "mod_pos": +0.070,
    "strong_pos": +0.120,
}

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
                bloc,
                len(x_vals),
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
        n_valid,
        len(_ALL_BLOCS),
    )
    return elasticities


@dataclasses.dataclass
class ElasticityFit:
    """Per-bloc Ridge regression result from fit_elasticity().

    coefficients[bloc][source] = β coefficient
    intercepts[bloc]           = intercept term
    alpha_chosen[bloc]         = Ridge α selected by LOOCV
    cv_r2[bloc]                = leave-one-out R² (NaN if insufficient data)
    sources                    = ordered list of source names used as features
    n_obs[bloc]                = number of (shock, bloc) pairs used
    """

    coefficients: dict[str, dict[str, float]]
    intercepts: dict[str, float]
    alpha_chosen: dict[str, float]
    cv_r2: dict[str, float]
    sources: list[str]
    n_obs: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


# Default Ridge α grid — spans four decades around 1.0
_DEFAULT_ALPHA_GRID = [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]
_MIN_OBS = 3  # minimum shocks needed to fit per bloc


def fit_elasticity(
    news_scores: dict[str, dict[str, float]],
    social_scores: dict[str, dict[str, dict[str, float]]],
    polling_deltas: dict[str, dict[str, float]],
    alpha_grid: list[float] | None = None,
) -> ElasticityFit:
    """Ridge regression of sentiment scores → observed vote-share deltas, per bloc.

    Fits one Ridge model per bloc. Each model has one feature column per source
    (news + each platform in social_scores). Missing feature values for a
    (shock, source) pair are imputed as 0.0. Archive posts in social_scores are
    treated identically to live posts — they share the same platform keys and
    have already been scored through the same RoBERTa pipeline.

    Cross-validation uses leave-one-shock-out (LOOCV) to select the Ridge α.
    Blocs with fewer than _MIN_OBS observations receive NaN coefficients and are
    logged at DEBUG level.

    Args:
        news_scores: {bloc: {shock_id: score ∈ [-1,1]}}
            Aggregated RoBERTa scores from the news pipeline.
        social_scores: {platform: {bloc: {shock_id: score ∈ [-1,1]}}}
            Aggregated RoBERTa scores per platform (live + archive combined).
            Platform names become feature column names alongside "news".
        polling_deltas: {bloc: {shock_id: Δvote_share}}
            Observed baseline-adjusted vote-share changes. These are the
            regression targets.
        alpha_grid: Ridge regularisation strengths to try in CV.
            Defaults to _DEFAULT_ALPHA_GRID.

    Returns:
        ElasticityFit with per-bloc, per-source coefficients and diagnostics.
    """
    try:
        from sklearn.linear_model import RidgeCV  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "scikit-learn is required for fit_elasticity. " "Install with: pip install scikit-learn"
        ) from exc

    alphas = alpha_grid or _DEFAULT_ALPHA_GRID

    # Ordered source list: news first, then platforms alphabetically
    platforms = sorted(social_scores.keys())
    sources = ["news"] + platforms

    # Collect all shock IDs that appear in polling_deltas
    all_shock_ids: set[str] = set()
    for bloc_deltas in polling_deltas.values():
        all_shock_ids.update(bloc_deltas.keys())

    coefficients: dict[str, dict[str, float]] = {}
    intercepts: dict[str, float] = {}
    alpha_chosen: dict[str, float] = {}
    cv_r2: dict[str, float] = {}
    n_obs: dict[str, int] = {}

    for bloc in _ALL_BLOCS:
        bloc_deltas = polling_deltas.get(bloc, {})

        # Shocks that have a valid polling delta for this bloc
        valid_shocks = [
            s for s in all_shock_ids if s in bloc_deltas and np.isfinite(bloc_deltas[s])
        ]

        n = len(valid_shocks)
        n_obs[bloc] = n

        if n < _MIN_OBS:
            logger.debug(
                "fit_elasticity: bloc='%s' has %d obs (< %d); skipping.",
                bloc,
                n,
                _MIN_OBS,
            )
            nan_coef = {src: float("nan") for src in sources}
            coefficients[bloc] = nan_coef
            intercepts[bloc] = float("nan")
            alpha_chosen[bloc] = float("nan")
            cv_r2[bloc] = float("nan")
            continue

        # Build design matrix X (n × n_sources) and target y (n,)
        X_rows: list[list[float]] = []
        y: list[float] = []

        for shock_id in valid_shocks:
            row: list[float] = []

            # News feature
            news_bloc = news_scores.get(bloc, {})
            row.append(news_bloc.get(shock_id, 0.0))

            # Per-platform social features
            for platform in platforms:
                plat_bloc = social_scores.get(platform, {}).get(bloc, {})
                row.append(plat_bloc.get(shock_id, 0.0))

            X_rows.append(row)
            y.append(bloc_deltas[shock_id])

        X = np.array(X_rows, dtype=float)
        y_arr = np.array(y, dtype=float)

        # RidgeCV with leave-one-shock-out cross-validation
        # cv=n means LOOCV when n == number of samples
        ridge = RidgeCV(alphas=alphas, fit_intercept=True, cv=n)
        ridge.fit(X, y_arr)

        coefficients[bloc] = {src: float(c) for src, c in zip(sources, ridge.coef_)}
        intercepts[bloc] = float(ridge.intercept_)
        alpha_chosen[bloc] = float(ridge.alpha_)

        # Compute LOOCV R² manually (RidgeCV.best_score_ is available in sklearn ≥1.0)
        best_score = getattr(ridge, "best_score_", None)
        cv_r2[bloc] = float(best_score) if best_score is not None else float("nan")

        logger.debug(
            "fit_elasticity: bloc='%s' n=%d alpha=%.3f r2=%.3f coefs=%s",
            bloc,
            n,
            ridge.alpha_,
            cv_r2[bloc],
            {src: f"{c:+.4f}" for src, c in coefficients[bloc].items()},
        )

    n_valid = sum(1 for b in _ALL_BLOCS if np.isfinite(coefficients[b]["news"]))
    logger.info(
        "fit_elasticity: %d/%d blocs have valid coefficients (%d sources: %s)",
        n_valid,
        len(_ALL_BLOCS),
        len(sources),
        sources,
    )

    return ElasticityFit(
        coefficients=coefficients,
        intercepts=intercepts,
        alpha_chosen=alpha_chosen,
        cv_r2=cv_r2,
        sources=sources,
        n_obs=n_obs,
    )


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

    logger.info("assemble_finetune_dataset: wrote %d examples to %s", n_examples, output_path)

    return LLMFineTuneData(
        base_model=base_model,
        lora_rank=lora_rank,
        n_examples=n_examples,
        cycles_used=sorted(set(cycles_used)) if cycles_used else [],
        adapter_path=None,
    )


# ── Unified fine-tuning dataset ───────────────────────────────────────────────

# Shocks whose date year == 2020 are held out as eval
_EVAL_YEAR = 2020

# Pre-2014 shocks have no prediction market coverage
_MARKET_LAUNCH_YEAR = 2014


def _shock_year(shock_id: str, shocks_lookup: dict[str, dict]) -> int | None:
    """Return the year of a shock from the shocks registry, or None."""
    shock = shocks_lookup.get(shock_id, {})
    date_str = (shock.get("date_window") or {}).get("shock_date") or shock.get("date") or ""
    try:
        return int(date_str[:4])
    except (ValueError, TypeError):
        return None


def _aggregate_social_scores(
    social_by_shock: dict[str, SocialMediaSentimentData],
    shock_id: str,
) -> dict[str, float]:
    """Average per-platform social scores into a single per-bloc score."""
    sms = social_by_shock.get(shock_id)
    if sms is None:
        return {b: 0.0 for b in _ALL_BLOCS}

    sums: dict[str, float] = {b: 0.0 for b in _ALL_BLOCS}
    counts: dict[str, int] = {b: 0 for b in _ALL_BLOCS}
    for platform_scores in sms.scores.values():
        for bloc, score in platform_scores.items():
            if bloc in sums:
                sums[bloc] += score
                counts[bloc] += 1

    return {b: (sums[b] / counts[b] if counts[b] > 0 else 0.0) for b in _ALL_BLOCS}


def _market_block(pmd: PredictionMarketData | None) -> dict | None:
    """Serialise the calibration-only market block for a JSONL record."""
    if pmd is None:
        return None
    return {
        "pre_shock_prob": pmd.pre_shock_prob,
        "post_shock_1h": pmd.post_shock_1h,
        "post_shock_24h": pmd.post_shock_24h,
        "post_shock_72h": pmd.post_shock_72h,
        "delta_prob": pmd.delta_prob,
        "sources": pmd.sources,
    }


def build_finetune_dataset(
    sentiment_data: SentimentData,
    social_by_shock: dict[str, SocialMediaSentimentData],
    shocks_config: list[dict[str, Any]],
    market_data: dict[str, PredictionMarketData] | None = None,
    synthetic_jsonl_path: str | Path | None = None,
    train_output: str | Path | None = None,
    eval_output: str | Path | None = None,
    base_model: str = "mistralai/Mistral-7B-v0.3",
    lora_rank: int = 16,
) -> LLMFineTuneData:
    """Build the unified fine-tuning JSONL dataset.

    One record per shock event with numerical RoBERTa features, 9-token delta
    bins, prediction market calibration block, and differential sample weight.

    Record schema
    -------------
    {
      "shock_id":              str,
      "description":           str,
      "party":                 "democrat" | "republican",
      "year":                  int,
      "source":                "real" | "synthetic",
      "weight":                1.5 | 1.0 | 0.5,
      "news_roberta_scores":   {bloc_id: float, ...},   # 15 blocs
      "social_roberta_scores": {bloc_id: float, ...},   # 15 blocs, averaged across platforms
      "delta_bins":            {bloc_id: bin_str, ...},  # 15 blocs, from news score
      "prediction_market":     {...} | null              # calibration only, null pre-2014
    }

    Weighting rule
    --------------
    market-backed (has PredictionMarketData)  → weight = 1.5
    poll-only (real, no market data)          → weight = 1.0
    synthetic (from synthetic_jsonl_path)     → weight = 0.5

    Split rule
    ----------
    year == 2020  → eval.jsonl
    otherwise     → train.jsonl

    Validation (raises ValueError on failure)
    ---------
    - No duplicate shock IDs within each split
    - All 15 canonical bloc keys present in every scores dict
    - No shock_id appears in both train and eval

    Args:
        sentiment_data:        News RoBERTa scores per bloc per shock.
        social_by_shock:       Social media scores per shock.
        shocks_config:         Parsed configs/shocks.json.
        market_data:           Optional PredictionMarketData per shock.
        synthetic_jsonl_path:  Path to data/finetune/synthetic.jsonl.
        train_output:          Destination for train split.
        eval_output:           Destination for eval split.
        base_model, lora_rank: Stored in the returned LLMFineTuneData.

    Returns:
        LLMFineTuneData artifact with n_examples, cycles_used.
    """
    train_output = Path(train_output) if train_output else _DEFAULT_FINETUNE_DIR / "train.jsonl"
    eval_output = Path(eval_output) if eval_output else _DEFAULT_FINETUNE_DIR / "eval.jsonl"
    train_output.parent.mkdir(parents=True, exist_ok=True)

    # Build shock description and party lookup
    shocks_lookup: dict[str, dict] = {s["id"]: s for s in shocks_config}
    shock_desc: dict[str, str] = {}
    shock_party: dict[str, str] = {}
    for s in shocks_config:
        sid = s["id"]
        shock_desc[sid] = s.get("description") or s.get("text") or s.get("name") or sid
        shock_party[sid] = s.get("party", "democrat")

    market_data = market_data or {}

    # ── Build real records ────────────────────────────────────────────────────
    train_records: list[dict] = []
    eval_records: list[dict] = []

    for shock_id in sentiment_data.shocks:
        year = _shock_year(shock_id, shocks_lookup)

        news_scores = {
            b: float(sentiment_data.scores.get(b, {}).get(shock_id, 0.0)) for b in _ALL_BLOCS
        }
        social_scores = _aggregate_social_scores(social_by_shock, shock_id)
        pmd = market_data.get(shock_id)

        # null market block for pre-2014 shocks regardless of data presence
        market_block = _market_block(pmd) if (year is None or year >= _MARKET_LAUNCH_YEAR) else None
        weight = 1.5 if market_block is not None else 1.0

        record = {
            "shock_id": shock_id,
            "description": shock_desc.get(shock_id, shock_id),
            "party": shock_party.get(shock_id, "democrat"),
            "year": year,
            "source": "real",
            "weight": weight,
            "news_roberta_scores": news_scores,
            "social_roberta_scores": social_scores,
            "delta_bins": {b: score_to_bin(news_scores[b]) for b in _ALL_BLOCS},
            "prediction_market": market_block,
        }

        if year == _EVAL_YEAR:
            eval_records.append(record)
        else:
            train_records.append(record)

    # ── Append synthetic records to train only ────────────────────────────────
    if synthetic_jsonl_path is not None:
        synth_path = Path(synthetic_jsonl_path)
        if synth_path.exists():
            with open(synth_path, encoding="utf-8") as f:
                for lineno, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        raw = json.loads(line)
                    except json.JSONDecodeError as exc:
                        logger.warning("synthetic.jsonl line %d: %s", lineno, exc)
                        continue
                    # Synthetic records use delta_bins directly; build zero score vectors
                    bins = raw.get("delta_bins") or {}
                    bin_midpoints = {
                        b: BIN_MIDPOINTS.get(bins.get(b, "neutral"), 0.0) for b in _ALL_BLOCS
                    }
                    train_records.append(
                        {
                            "shock_id": raw.get("id", f"synthetic_{lineno}"),
                            "description": raw.get("description", ""),
                            "party": raw.get("party", "democrat"),
                            "year": None,
                            "source": "synthetic",
                            "weight": 0.5,
                            "news_roberta_scores": bin_midpoints,
                            "social_roberta_scores": {b: 0.0 for b in _ALL_BLOCS},
                            "delta_bins": {b: bins.get(b, "neutral") for b in _ALL_BLOCS},
                            "prediction_market": None,
                        }
                    )
        else:
            logger.warning("synthetic_jsonl_path not found: %s", synth_path)

    # ── Validate ──────────────────────────────────────────────────────────────
    def _validate_split(records: list[dict], split_name: str) -> None:
        seen_ids: set[str] = set()
        for rec in records:
            sid = rec["shock_id"]
            if sid in seen_ids:
                raise ValueError(
                    f"build_finetune_dataset: duplicate shock_id '{sid}' in {split_name}"
                )
            seen_ids.add(sid)
            for field in ("news_roberta_scores", "social_roberta_scores", "delta_bins"):
                missing = [b for b in _ALL_BLOCS if b not in rec[field]]
                if missing:
                    raise ValueError(
                        f"build_finetune_dataset: {split_name} record '{sid}' "
                        f"missing bloc keys in {field}: {missing}"
                    )

    _validate_split(train_records, "train")
    _validate_split(eval_records, "eval")

    # No overlap
    train_ids = {r["shock_id"] for r in train_records}
    eval_ids = {r["shock_id"] for r in eval_records}
    overlap = train_ids & eval_ids
    if overlap:
        raise ValueError(
            f"build_finetune_dataset: {len(overlap)} shock IDs appear in both "
            f"train and eval: {sorted(overlap)}"
        )

    # ── Write ─────────────────────────────────────────────────────────────────
    def _write(records: list[dict], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    _write(train_records, train_output)
    _write(eval_records, eval_output)

    n_train = len(train_records)
    n_eval = len(eval_records)
    logger.info(
        "build_finetune_dataset: train=%d (real=%d synthetic=%d) eval=%d → %s / %s",
        n_train,
        sum(1 for r in train_records if r["source"] == "real"),
        sum(1 for r in train_records if r["source"] == "synthetic"),
        n_eval,
        train_output,
        eval_output,
    )

    all_years = [r["year"] for r in train_records + eval_records if r["year"] is not None]
    cycles_used = sorted({y for y in all_years if y % 4 == 0})

    return LLMFineTuneData(
        base_model=base_model,
        lora_rank=lora_rank,
        n_examples=n_train + n_eval,
        cycles_used=cycles_used,
        adapter_path=None,
    )
