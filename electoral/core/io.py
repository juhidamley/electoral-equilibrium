"""Parquet + JSON artifact I/O for the Electoral Equilibrium pipeline.

Tabular stages (voter panel, scored posts, fine-tuning dataset) write Parquet.
Non-tabular stages (optimizer, simulation) write JSON.
All artifact envelopes are JSON.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    pass


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    """Write a dict to a JSON file with stable formatting."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True, ensure_ascii=False)
        f.write("\n")


def read_json(path: str | Path) -> dict[str, Any]:
    """Read a JSON file and return its contents as a dict."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Artifact not found: {path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_parquet(path: str | Path, df: Any) -> None:
    """Write a DataFrame to a Parquet file (pyarrow/snappy)."""
    try:
        import pandas  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "pandas is required for write_parquet. Install with: pip install pandas pyarrow"
        ) from e
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(str(path), engine="pyarrow", compression="snappy", index=False)


def read_parquet(path: str | Path) -> Any:
    """Read a Parquet file and return a DataFrame."""
    try:
        import pandas as pd
    except ImportError as e:
        raise ImportError(
            "pandas is required for read_parquet. Install with: pip install pandas pyarrow"
        ) from e
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Parquet artifact not found: {path}")
    return pd.read_parquet(str(path), engine="pyarrow")


def write_artifact(
    path: str | Path,
    envelope: dict[str, Any],
    df: Optional[Any] = None,
) -> None:
    """Write a stage artifact to disk.

    The envelope JSON is always written. If df is provided, the tabular data
    payload is also written as Parquet alongside the envelope.

    Args:
        path: Destination path for the JSON envelope (e.g. artifacts/voter_panel.json)
        envelope: Dict with keys: stage, run_key, metadata, data
        df: Optional DataFrame for tabular stages — written as {path.stem}.parquet
    """
    path = Path(path)
    write_json(path, envelope)
    if df is not None:
        parquet_path = path.with_suffix(".parquet")
        write_parquet(parquet_path, df)


def read_artifact(path: str | Path) -> tuple[dict[str, Any], Optional[Any]]:
    """Read a stage artifact from disk.

    Returns:
        (envelope, df) where df is None for non-tabular artifacts
    """
    path = Path(path)
    envelope = read_json(path)
    parquet_path = path.with_suffix(".parquet")
    df = None
    if parquet_path.exists():
        df = read_parquet(parquet_path)
    return envelope, df
