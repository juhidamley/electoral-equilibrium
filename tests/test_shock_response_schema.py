"""Tests for ShockResponseSchema Pydantic model and DeltaBin type alias."""

import json

import pytest
from pydantic import ValidationError

from electoral.artifacts import ShockResponseSchema

_VALID_PAYLOAD = {
    "delta_bins_race": {
        "african_american": "slight_neg",
        "latino": "neutral",
        "asian": "slight_pos",
        "white": "mod_neg",
        "other_race": "neutral",
    },
    "delta_bins_religion": {
        "evangelical": "mod_neg",
        "catholic": "slight_neg",
        "protestant": "slight_neg",
        "secular": "slight_pos",
        "jewish": "neutral",
        "muslim": "neutral",
        "other_rel": "neutral",
    },
    "delta_bins_gender": {
        "women": "slight_pos",
        "men": "mod_neg",
        "other_gender": "neutral",
    },
    "delta_eff": -0.021,
}


def test_valid_payload_validates():
    schema = ShockResponseSchema(**_VALID_PAYLOAD)
    assert schema.delta_bins_race.african_american == "slight_neg"
    assert schema.delta_bins_religion.evangelical == "mod_neg"
    assert schema.delta_bins_gender.women == "slight_pos"
    assert schema.delta_eff == pytest.approx(-0.021)


def test_invalid_bin_raises_validation_error():
    bad = {**_VALID_PAYLOAD}
    bad["delta_bins_race"] = {**_VALID_PAYLOAD["delta_bins_race"], "african_american": "negative"}
    with pytest.raises(ValidationError):
        ShockResponseSchema(**bad)


def test_delta_eff_accepts_float():
    for val in (-1.0, 0.0, 0.5, 1.0):
        s = ShockResponseSchema(**{**_VALID_PAYLOAD, "delta_eff": val})
        assert s.delta_eff == pytest.approx(val)


def test_outlines_schema_compatibility():
    """Schema can be serialised to a JSON schema dict consumable by outlines."""
    schema_dict = ShockResponseSchema.model_json_schema()
    assert isinstance(schema_dict, dict)
    # Must be round-trippable to JSON (outlines serialises it internally)
    schema_str = json.dumps(schema_dict)
    recovered = json.loads(schema_str)
    assert "properties" in recovered
    # All three stratum keys must appear
    props = recovered["properties"]
    assert "delta_bins_race" in props
    assert "delta_bins_religion" in props
    assert "delta_bins_gender" in props
    assert "delta_eff" in props
