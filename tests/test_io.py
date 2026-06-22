"""Tests for electoral/core/io.py — Parquet + JSON artifact I/O layer.

Required coverage (from devplan):
  (i)   JSON round-trip
  (ii)  Parquet round-trip with a 100-row fixture DataFrame
  (iii) Parquet file is smaller than equivalent JSON
  (iv)  Reading a non-existent path raises FileNotFoundError

Additional coverage:
  - JSON output uses indent=2 and sort_keys=True (stable diffs)
  - Parent directories are created automatically on write
  - write_artifact dispatches to both JSON and Parquet when df is supplied
  - read_artifact returns (envelope, df) tuple; df is None for non-tabular artifacts
  - Parquet preserves column dtypes and row count exactly
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from electoral.core.io import (
    read_artifact,
    read_json,
    read_parquet,
    sanitize_floats,
    write_artifact,
    write_json,
    write_parquet,
)


# ── Shared fixtures ───────────────────────────────────────────────────────────


@pytest.fixture
def panel_df() -> pd.DataFrame:
    """100-row synthetic voter panel DataFrame matching the full pipeline schema.

    Uses the 7-column schema (cycle, stratum, bloc, vote_share, stratum_share,
    turnout, source) that the real data pipeline produces.  Note: toy_panel.csv
    is a 5-column cleaning fixture; this in-memory frame is the io-layer fixture.
    """
    from electoral.core.rng import make_rng

    rng = make_rng(42)
    n = 100
    races = ["african_american", "latino", "asian", "white", "other_race"]
    strata = ["race", "religion", "gender"]
    cycles = [2016, 2020]

    return pd.DataFrame(
        {
            "cycle": rng.choice(cycles, size=n),
            "stratum": rng.choice(strata, size=n),
            "bloc": rng.choice(races, size=n),
            "vote_share": rng.uniform(0.0, 1.0, size=n).round(4),
            "stratum_share": rng.uniform(0.05, 0.40, size=n).round(4),
            "turnout": rng.uniform(0.40, 0.80, size=n).round(4),
            "source": rng.choice(["ARDA", "GSS", "NEP"], size=n),
        }
    )


@pytest.fixture
def sample_envelope() -> dict:
    return {
        "stage": "voter_panel",
        "run_key": "test_run_42",
        "metadata": {"seed": 42, "n_cycles": 2},
        "data": {
            "cycles": [2016, 2020],
            "races": ["african_american", "white"],
            "layer_weights": {"lambda_1": 0.5, "lambda_2": 0.3, "lambda_3": 0.2},
        },
    }


# ── (i) JSON round-trip ───────────────────────────────────────────────────────


class TestWriteReadJson:

    def test_json_roundtrip(self, tmp_path, sample_envelope):
        """(i) write_json then read_json returns an identical dict."""
        path = tmp_path / "voter_panel.json"
        write_json(path, sample_envelope)
        recovered = read_json(path)
        assert recovered == sample_envelope

    def test_json_roundtrip_nested_types(self, tmp_path):
        """Nested lists, ints, floats, and None all survive a round-trip."""
        payload = {
            "weights": {"african_american": 0.15, "white": 0.85},
            "cycles": [2016, 2020],
            "adapter_path": None,
            "feasible": True,
            "mu_eff": 0.5312,
        }
        path = tmp_path / "artifact.json"
        write_json(path, payload)
        assert read_json(path) == payload

    def test_json_uses_indent_2(self, tmp_path, sample_envelope):
        """Output must be indented at 2 spaces for readable diffs."""
        path = tmp_path / "out.json"
        write_json(path, sample_envelope)
        raw = path.read_text(encoding="utf-8")
        # The first nested key should be indented by exactly 2 spaces.
        assert "\n  " in raw, "indent=2 not present in output"

    def test_json_uses_sort_keys(self, tmp_path):
        """Keys must be alphabetically sorted for stable diffs and deterministic output."""
        payload = {"z_key": 1, "a_key": 2, "m_key": 3}
        path = tmp_path / "sorted.json"
        write_json(path, payload)
        raw = path.read_text(encoding="utf-8")
        parsed = json.loads(raw)
        keys_in_file = list(parsed.keys())
        assert keys_in_file == sorted(keys_in_file), f"Keys not sorted: {keys_in_file}"

    def test_json_creates_parent_directories(self, tmp_path):
        """write_json must create any missing parent directories."""
        path = tmp_path / "deep" / "nested" / "dir" / "artifact.json"
        write_json(path, {"x": 1})
        assert path.exists()

    def test_json_ends_with_newline(self, tmp_path, sample_envelope):
        """File must end with a trailing newline (POSIX convention)."""
        path = tmp_path / "out.json"
        write_json(path, sample_envelope)
        raw = path.read_text(encoding="utf-8")
        assert raw.endswith("\n"), "JSON output does not end with newline"


# ── (ii) Parquet round-trip ───────────────────────────────────────────────────


class TestWriteReadParquet:

    def test_parquet_roundtrip_shape(self, tmp_path, panel_df):
        """(ii) write_parquet then read_parquet returns a DataFrame with identical shape."""
        path = tmp_path / "panel.parquet"
        write_parquet(path, panel_df)
        recovered = read_parquet(path)
        assert recovered.shape == panel_df.shape

    def test_parquet_roundtrip_columns(self, tmp_path, panel_df):
        """Column names are preserved exactly."""
        path = tmp_path / "panel.parquet"
        write_parquet(path, panel_df)
        recovered = read_parquet(path)
        assert list(recovered.columns) == list(panel_df.columns)

    def test_parquet_roundtrip_values(self, tmp_path, panel_df):
        """All 100 rows of data values survive the round-trip unchanged."""
        path = tmp_path / "panel.parquet"
        write_parquet(path, panel_df)
        recovered = read_parquet(path)
        pd.testing.assert_frame_equal(recovered, panel_df, check_dtype=False)

    def test_parquet_roundtrip_numeric_dtypes(self, tmp_path):
        """Float and int columns survive without precision loss."""
        df = pd.DataFrame(
            {
                "cycle": [2016, 2020] * 50,
                "vote_share": [round(x, 10) for x in np.linspace(0.0, 1.0, 100)],
                "stratum_share": [0.5] * 100,
            }
        )
        path = tmp_path / "numeric.parquet"
        write_parquet(path, df)
        recovered = read_parquet(path)
        np.testing.assert_allclose(
            recovered["vote_share"].values, df["vote_share"].values, rtol=1e-9
        )

    def test_parquet_creates_parent_directories(self, tmp_path, panel_df):
        """write_parquet must create any missing parent directories."""
        path = tmp_path / "sub" / "dir" / "panel.parquet"
        write_parquet(path, panel_df)
        assert path.exists()

    def test_parquet_roundtrip_1000_rows(self, tmp_path):
        """Devplan requirement: 1,000-row DataFrame round-trips without data loss."""
        from electoral.core.rng import make_rng

        rng = make_rng(99)
        n = 1_000
        races = ["african_american", "latino", "asian", "white", "other_race"]
        df = pd.DataFrame(
            {
                "cycle": rng.choice([2016, 2018, 2020, 2022], size=n),
                "bloc": rng.choice(races, size=n),
                "vote_share": rng.uniform(0.0, 1.0, size=n).round(6),
                "stratum_share": rng.uniform(0.05, 0.40, size=n).round(6),
                "turnout": rng.uniform(0.40, 0.80, size=n).round(6),
                "source": rng.choice(["ARDA", "GSS", "NEP"], size=n),
            }
        )
        path = tmp_path / "panel_1000.parquet"
        write_parquet(path, df)
        recovered = read_parquet(path)
        assert recovered.shape == (n, len(df.columns))
        pd.testing.assert_frame_equal(recovered, df, check_dtype=False)


# ── (iii) Parquet smaller than JSON ──────────────────────────────────────────


class TestParquetCompression:

    def test_parquet_smaller_than_json(self, tmp_path, panel_df):
        """(iii) Snappy-compressed Parquet must be smaller than equivalent JSON."""
        parquet_path = tmp_path / "panel.parquet"
        json_path = tmp_path / "panel.json"

        write_parquet(parquet_path, panel_df)
        # Serialize the same data as JSON for a fair comparison.
        json_path.write_text(
            json.dumps(panel_df.to_dict(orient="records"), indent=2),
            encoding="utf-8",
        )

        parquet_size = parquet_path.stat().st_size
        json_size = json_path.stat().st_size

        assert parquet_size < json_size, (
            f"Parquet ({parquet_size} B) is not smaller than JSON ({json_size} B). "
            f"Check that snappy compression is active."
        )

    def test_parquet_compression_ratio(self, tmp_path):
        """For a highly-repetitive 100-row panel, Parquet should be < 50% of JSON size."""
        # Repetitive data is where columnar + snappy compression shines most.
        df = pd.DataFrame(
            {
                "cycle": [2016] * 50 + [2020] * 50,
                "stratum": ["race"] * 100,
                "bloc": ["african_american"] * 100,
                "vote_share": [0.87] * 100,
                "stratum_share": [0.13] * 100,
                "turnout": [0.63] * 100,
                "source": ["NEP"] * 100,
            }
        )
        parquet_path = tmp_path / "rep.parquet"
        json_path = tmp_path / "rep.json"

        write_parquet(parquet_path, df)
        json_path.write_text(json.dumps(df.to_dict(orient="records"), indent=2), encoding="utf-8")

        ratio = parquet_path.stat().st_size / json_path.stat().st_size
        assert (
            ratio < 0.5
        ), f"Compression ratio {ratio:.2f} >= 0.50; expected < 0.50 for repetitive data."


# ── (iv) FileNotFoundError on missing path ────────────────────────────────────


class TestMissingPath:

    def test_read_json_missing_raises(self, tmp_path):
        """(iv) read_json on a non-existent path must raise FileNotFoundError."""
        missing = tmp_path / "does_not_exist.json"
        with pytest.raises(FileNotFoundError, match=str(missing)):
            read_json(missing)

    def test_read_parquet_missing_raises(self, tmp_path):
        """(iv) read_parquet on a non-existent path must raise FileNotFoundError."""
        missing = tmp_path / "does_not_exist.parquet"
        with pytest.raises(FileNotFoundError, match="does_not_exist"):
            read_parquet(missing)

    def test_error_message_includes_path(self, tmp_path):
        """FileNotFoundError message must name the missing path for fast diagnosis."""
        missing = tmp_path / "artifacts" / "voter_panel.json"
        with pytest.raises(FileNotFoundError) as exc_info:
            read_json(missing)
        assert "voter_panel.json" in str(
            exc_info.value
        ), f"Path not in error message: {exc_info.value}"


# ── write_artifact / read_artifact dispatch ───────────────────────────────────


class TestWriteArtifactDispatch:

    def test_json_only_artifact_writes_one_file(self, tmp_path, sample_envelope):
        """Non-tabular artifacts (no df) must write exactly one .json file."""
        path = tmp_path / "optimization.json"
        write_artifact(path, sample_envelope)
        assert path.exists(), ".json envelope not written"
        assert not path.with_suffix(".parquet").exists(), ".parquet written when no df was provided"

    def test_tabular_artifact_writes_both_files(self, tmp_path, sample_envelope, panel_df):
        """Tabular artifacts (with df) must write both .json and .parquet."""
        path = tmp_path / "voter_panel.json"
        write_artifact(path, sample_envelope, df=panel_df)
        assert path.exists(), ".json envelope not written"
        assert path.with_suffix(".parquet").exists(), ".parquet not written alongside .json"

    def test_read_artifact_json_only_returns_none_df(self, tmp_path, sample_envelope):
        """read_artifact for a non-tabular artifact returns (envelope, None)."""
        path = tmp_path / "simulation.json"
        write_artifact(path, sample_envelope)
        envelope, df = read_artifact(path)
        assert envelope == sample_envelope
        assert df is None

    def test_read_artifact_tabular_returns_dataframe(self, tmp_path, sample_envelope, panel_df):
        """read_artifact for a tabular artifact returns (envelope, DataFrame)."""
        path = tmp_path / "voter_panel.json"
        write_artifact(path, sample_envelope, df=panel_df)
        envelope, df = read_artifact(path)
        assert envelope == sample_envelope
        assert isinstance(df, pd.DataFrame)
        assert df.shape == panel_df.shape

    def test_artifact_envelope_is_human_readable(self, tmp_path, sample_envelope):
        """The JSON envelope must be indented (human-readable) even for tabular stages."""
        path = tmp_path / "voter_panel.json"
        write_artifact(path, sample_envelope)
        raw = path.read_text(encoding="utf-8")
        assert "\n  " in raw, "Envelope JSON is not indented"


class TestSanitizeFloats:
    """The shared non-finite-float sanitizer (Week 8 JSON-safety item)."""

    def test_replaces_inf_neg_inf_nan_with_none(self):
        out = sanitize_floats({"a": float("inf"), "b": float("-inf"), "c": float("nan")})
        assert out == {"a": None, "b": None, "c": None}

    def test_finite_floats_and_other_types_untouched(self):
        payload = {"x": 0.5, "n": 3, "s": "hi", "t": True, "f": False, "z": None}
        # Comparison is exact: finite values must pass through unchanged.
        assert sanitize_floats(payload) == payload

    def test_recurses_into_nested_lists_and_dicts(self):
        out = sanitize_floats(
            {"row": [1.0, float("nan"), {"deep": float("inf")}], "ok": [0.1, 0.2]}
        )
        assert out == {"row": [1.0, None, {"deep": None}], "ok": [0.1, 0.2]}

    def test_does_not_mutate_input(self):
        original = {"v": float("nan")}
        sanitize_floats(original)
        assert original["v"] != original["v"]  # still NaN (NaN != NaN), i.e. untouched

    def test_write_json_emits_valid_json_for_nonfinite(self, tmp_path):
        """An artifact containing inf/nan must round-trip as null, not break parsing."""
        path = tmp_path / "art.json"
        write_json(path, {"win": float("inf"), "loss": float("nan"), "ok": 0.4})
        raw = path.read_text(encoding="utf-8")
        # The invalid JSON tokens must NOT appear in the file.
        assert "Infinity" not in raw and "NaN" not in raw
        # And it must parse cleanly (strict=True rejects Infinity/NaN tokens).
        parsed = json.loads(raw)
        assert parsed == {"win": None, "loss": None, "ok": 0.4}
