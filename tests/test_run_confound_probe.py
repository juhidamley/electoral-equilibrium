from __future__ import annotations

import pytest

from scripts.run_confound_probe import CORRECT, FALLBACK, OTHER, BLOCS, net_direction, verdict_for_prediction


def _record() -> dict:
    return {
        "shock_id": "fixture",
        "description": "Negative tone toward Democrats, but backlash helps Democrats.",
        "party": "democrat",
        "_seed_meta": {"benefiting_party": "democrat"},
        "_provenance": {"dataset_role": "evaluation_probe", "training_eligible": False},
        "_probe": {
            "probe_id": "fixture_probe",
            "benefiting_party": "democrat",
            "sentiment_only_predicted_party": "republican",
            "known_correct_delta_eff_sign": "positive",
        },
    }


def test_probe_classifier_marks_mobilization_correct_fixture() -> None:
    result = verdict_for_prediction(_record(), {bloc: "slight_pos" for bloc in BLOCS})
    assert result["model_predicted_sign"] == "positive"
    assert result["verdict"] == CORRECT


def test_probe_classifier_marks_sentiment_fallback_fixture() -> None:
    result = verdict_for_prediction(_record(), {bloc: "slight_neg" for bloc in BLOCS})
    assert result["model_predicted_sign"] == "negative"
    assert result["verdict"] == FALLBACK


def test_probe_classifier_marks_neutral_prediction_ambiguous() -> None:
    result = verdict_for_prediction(_record(), {bloc: "neutral" for bloc in BLOCS})
    assert result["model_predicted_sign"] == "neutral"
    assert result["verdict"] == OTHER


def test_net_direction_requires_complete_valid_vector() -> None:
    with pytest.raises(ValueError, match="missing"):
        net_direction({})
