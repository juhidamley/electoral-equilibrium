#!/usr/bin/env python3
"""train_setfit.py — train SetFit bio classifiers for all three demographic strata.

Reads:    data/bio_labels/labeled_bios.jsonl  (produced by build_bio_labels.py)
Writes:   models/setfit_race/
          models/setfit_religion/
          models/setfit_gender/

Each directory contains:
  - The fine-tuned sentence-transformer backbone + sklearn head (SetFit format)
  - label_config.json  →  {"id2label": {"0": "evangelical", ...}, "label2id": {...}}

Pre-flight:
    pip install setfit>=1.0 datasets>=2.18 scikit-learn>=1.3

Usage:
    # Build bio labels first (needs JUHIDRIVE archives):
    python scripts/build_bio_labels.py

    # Then train (runs fine on M5 CPU; ~10 min per stratum):
    python scripts/train_setfit.py
    python scripts/train_setfit.py --epochs 3 --num-iterations 40 --verbose
    python scripts/train_setfit.py --strata religion  # single stratum only
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

logger = logging.getLogger(__name__)

LABELS_PATH = REPO_ROOT / "data" / "bio_labels" / "labeled_bios.jsonl"
MODELS_DIR = REPO_ROOT / "models"
BACKBONE = "sentence-transformers/all-MiniLM-L6-v2"

# Require at least this many examples per class to include it in training.
MIN_SAMPLES_PER_CLASS = 8

# Stratum definitions. gender_signal uses "F"/"M" internally; the server maps
# these back to "gender:women" / "gender:men" in the /classify response.
STRATA: dict[str, dict] = {
    "race": {
        "dir": MODELS_DIR / "setfit_race",
        "field": "race_bloc",
        "labels": ["african_american", "latino", "asian", "white", "other_race"],
    },
    "religion": {
        "dir": MODELS_DIR / "setfit_religion",
        "field": "religion_bloc",
        "labels": [
            "evangelical",
            "catholic",
            "protestant",
            "secular",
            "jewish",
            "muslim",
            "other_rel",
        ],
    },
    "gender": {
        "dir": MODELS_DIR / "setfit_gender",
        "field": "gender_signal",
        "labels": ["F", "M"],
    },
}


# ── Data loading ──────────────────────────────────────────────────────────────


def load_labeled_bios(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"Bio labels not found: {path}\n"
            "Run:  python scripts/build_bio_labels.py\n"
            "(Requires JUHIDRIVE archives; run on M5 with drive mounted.)"
        )
    records: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    logger.warning("Skipping malformed line: %s", exc)
    logger.info("Loaded %d labeled bios from %s", len(records), path)
    return records


def _build_split(
    records: list[dict],
    field: str,
    valid_labels: list[str],
) -> tuple[list[str], list[int], list[str]]:
    """Return (texts, label_ids, kept_label_names) filtered to classes with enough data.

    Classes with fewer than MIN_SAMPLES_PER_CLASS examples are dropped so
    stratified splitting doesn't fail on tiny classes.
    """
    texts_raw: list[str] = []
    labels_raw: list[str] = []
    for rec in records:
        label = rec.get(field)
        if label and label in valid_labels:
            texts_raw.append(rec["bio"])
            labels_raw.append(label)

    counts = Counter(labels_raw)
    kept = sorted(lbl for lbl, cnt in counts.items() if cnt >= MIN_SAMPLES_PER_CLASS)
    if not kept:
        return [], [], []

    texts_out: list[str] = []
    labels_out: list[int] = []
    label2id = {lbl: i for i, lbl in enumerate(kept)}
    for txt, lbl in zip(texts_raw, labels_raw):
        if lbl in label2id:
            texts_out.append(txt)
            labels_out.append(label2id[lbl])

    logger.info(
        "Stratum '%s' kept classes: %s",
        field,
        {lbl: counts[lbl] for lbl in kept},
    )
    return texts_out, labels_out, kept


# ── Training ──────────────────────────────────────────────────────────────────


def train_stratum(
    stratum_name: str,
    records: list[dict],
    cfg: dict,
    args: argparse.Namespace,
) -> float | None:
    """Train one SetFit model. Returns macro-F1 on eval set, or None if skipped."""
    try:
        from setfit import SetFitModel, SetFitTrainer, TrainingArguments
    except ImportError:
        raise RuntimeError("setfit is not installed. Run:  pip install 'setfit>=1.0'")
    try:
        from datasets import Dataset
    except ImportError:
        raise RuntimeError("datasets is not installed. Run:  pip install 'datasets>=2.18'")
    from sklearn.metrics import classification_report, f1_score
    from sklearn.model_selection import train_test_split

    texts, label_ids, label_names = _build_split(records, cfg["field"], cfg["labels"])

    if not texts:
        logger.warning("Stratum '%s': no examples above threshold — skipping.", stratum_name)
        return None
    if len(set(label_ids)) < 2:
        logger.warning(
            "Stratum '%s': only one class ('%s') — skipping (need ≥2 classes).",
            stratum_name,
            label_names[0],
        )
        return None

    n_total = len(texts)
    logger.info(
        "Stratum '%s': %d examples, %d classes, backbone=%s",
        stratum_name,
        n_total,
        len(label_names),
        BACKBONE,
    )

    arr_texts = np.array(texts, dtype=object)
    arr_labels = np.array(label_ids, dtype=int)

    try:
        X_tr, X_ev, y_tr, y_ev = train_test_split(
            arr_texts,
            arr_labels,
            test_size=0.20,
            random_state=42,
            stratify=arr_labels,
        )
    except ValueError:
        # Stratify fails when any class has <2 members after split — fall back
        logger.warning("Stratum '%s': stratified split failed, using random split.", stratum_name)
        X_tr, X_ev, y_tr, y_ev = train_test_split(
            arr_texts,
            arr_labels,
            test_size=0.20,
            random_state=42,
        )

    train_ds = Dataset.from_dict({"text": X_tr.tolist(), "label": y_tr.tolist()})
    eval_ds = Dataset.from_dict({"text": X_ev.tolist(), "label": y_ev.tolist()})

    logger.info("Loading backbone %s for stratum '%s' ...", BACKBONE, stratum_name)
    model = SetFitModel.from_pretrained(
        BACKBONE,
        labels=label_names,
    )

    def compute_metrics(y_pred, y_test):
        return {"macro_f1": float(f1_score(y_test, y_pred, average="macro", zero_division=0))}

    training_args = TrainingArguments(
        num_epochs=args.epochs,
        batch_size=16,
        num_iterations=args.num_iterations,
        seed=42,
        output_dir=str(cfg["dir"] / "checkpoints"),
    )

    trainer = SetFitTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        metric=compute_metrics,
        column_mapping={"text": "text", "label": "label"},
    )

    logger.info("Training stratum '%s' ...", stratum_name)
    trainer.train()

    metrics = trainer.evaluate()
    macro_f1 = float(metrics.get("macro_f1", 0.0))
    logger.info("Stratum '%s' eval macro-F1: %.4f", stratum_name, macro_f1)

    # Full classification report for diagnostics
    preds = model.predict(X_ev.tolist())
    report = classification_report(
        y_ev,
        preds,
        target_names=label_names,
        zero_division=0,
    )
    logger.info("Stratum '%s' classification report:\n%s", stratum_name, report)

    out_dir: Path = cfg["dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir))

    # Explicit label map so pi_bio_server.py can load without SetFit installed
    label_cfg = {
        "id2label": {str(i): lbl for i, lbl in enumerate(label_names)},
        "label2id": {lbl: i for i, lbl in enumerate(label_names)},
        "stratum": stratum_name,
        "backbone": BACKBONE,
    }
    (out_dir / "label_config.json").write_text(json.dumps(label_cfg, indent=2), encoding="utf-8")

    logger.info("Saved stratum '%s' → %s", stratum_name, out_dir)
    return macro_f1


# ── CLI ───────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train SetFit bio classifiers (one per demographic stratum)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--labels",
        type=Path,
        default=LABELS_PATH,
        metavar="JSONL",
        help="Path to labeled_bios.jsonl produced by build_bio_labels.py",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="SetFit contrastive-training epochs (1 is usually enough for small datasets)",
    )
    p.add_argument(
        "--num-iterations",
        type=int,
        default=20,
        help="Sentence-pair iterations per epoch (more = slower but better embeddings)",
    )
    p.add_argument(
        "--strata",
        nargs="+",
        default=list(STRATA),
        choices=list(STRATA),
        metavar="STRATUM",
        help="Which strata to train (default: all three)",
    )
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
    )

    records = load_labeled_bios(args.labels)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    results: dict[str, float | None] = {}
    for name in args.strata:
        results[name] = train_stratum(name, records, STRATA[name], args)

    logger.info("── Training summary ─────────────────────────────────")
    target_met = True
    for name, f1 in results.items():
        if f1 is None:
            status = "SKIP"
        elif f1 >= 0.75:
            status = "OK  "
        else:
            status = "FAIL"
            target_met = False
        logger.info(
            "  %-12s  macro-F1 = %-8s  [%s]", name, f"{f1:.4f}" if f1 is not None else "N/A", status
        )

    if not target_met:
        logger.warning(
            "One or more strata below 0.75 macro-F1. "
            "Add more labeled bios via build_bio_labels.py and re-train."
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
