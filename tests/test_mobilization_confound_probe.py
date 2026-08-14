from __future__ import annotations

import json

import pytest

from scripts import prep_finetune
from scripts.synthetic.gemini_review import _mobilizing_review_ok


def _record() -> dict:
    return {
        "shock_id": "held_out_probe",
        "party": "democrat",
        "delta_eff": 0.01,
        "_seed_meta": {
            "mobilization": "mobilizing",
            "benefiting_party": "democrat",
        },
        "_probe": {"probe_id": "probe_test"},
        "_provenance": {
            "dataset_role": "evaluation_probe",
            "training_eligible": False,
        },
    }


def test_mobilizing_review_requires_rationale_and_beneficiary_sign() -> None:
    rec = _record()
    assert not _mobilizing_review_ok(rec, {"corrected": None})[0]
    assert not _mobilizing_review_ok(
        rec,
        {
            "beneficiary_rationale": "Democrats are the beneficiary.",
            "corrected": {"delta_eff": -0.01},
        },
    )[0]
    assert _mobilizing_review_ok(
        rec,
        {"beneficiary_rationale": "Democrats benefit and delta_eff is positive."},
    )[0]


def test_training_builder_fails_closed_on_probe_record(tmp_path) -> None:
    data_dir = tmp_path / "finetune"
    data_dir.mkdir()
    (data_dir / "synthetic.jsonl").write_text(json.dumps(_record()) + "\n")

    with pytest.raises(ValueError, match="permanently held out"):
        prep_finetune.main(["--data-dir", str(data_dir)])

    assert not (data_dir / "train.jsonl").exists()
    assert not (data_dir / "eval.jsonl").exists()
