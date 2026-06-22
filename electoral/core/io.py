"""Parquet + JSON artifact I/O for the Electoral Equilibrium pipeline.

═══════════════════════════════════════════════════════════════════════════════
WHAT IS AN "ARTIFACT"? (the core idea of this whole pipeline)
═══════════════════════════════════════════════════════════════════════════════
The pipeline is a chain of stages: voter panel → baseline → sentiment → LLM
shock → optimizer → simulation. Each stage takes the *output of the previous
stage* and produces its own output. Those saved outputs are called **artifacts**.

Saving every stage's output to disk (instead of passing objects in memory) buys
us three things:
  1. Reproducibility/auditing — you can inspect exactly what each stage produced.
  2. Resumability — if stage 5 crashes, you don't have to re-run stages 1–4.
  3. Decoupling — the FastAPI app and the dashboard can read artifacts a batch
     job wrote hours earlier on a different machine (synced via Syncthing).

Every artifact is stored as an **envelope**: a small JSON wrapper with four keys:
    {
      "stage":    "simulation",         # which stage produced this
      "run_key":  "smoke",              # which run it belongs to
      "metadata": { ... },              # free-form extras (timings, counts)
      "data":     { ... }               # the actual typed payload (see artifacts.py)
    }
Keeping the shape uniform means generic tools (like verify_artifacts.py) can read
any artifact without knowing its specific type.

═══════════════════════════════════════════════════════════════════════════════
WHY TWO FILE FORMATS?
═══════════════════════════════════════════════════════════════════════════════
  • JSON   — human-readable text. Used for the envelope and for non-tabular
             payloads (optimizer weights, simulation results). Easy to read,
             diff, and open in any editor.
  • Parquet — a compact, columnar *binary* format for big tables (the voter
             panel, scored posts, fine-tuning rows). Far smaller and faster to
             load than JSON/CSV for thousands of rows. Requires the `pyarrow`
             library, which is why those imports are done lazily below.

Convention: a tabular stage writes BOTH an envelope `foo.json` AND the table
`foo.parquet` next to it (same filename stem). read_artifact() looks for the
sibling .parquet automatically.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    # Reserved for type-only imports (e.g. pandas) so we never pay the import
    # cost at runtime just for annotations. Nothing needed here yet.
    pass


def sanitize_floats(obj: Any) -> Any:
    """Recursively replace non-finite floats (inf, -inf, NaN) with None.

    WHY THIS EXISTS: JSON, by the spec, has no way to write "infinity" or "not a
    number". Python's json module will happily emit the bare tokens Infinity /
    -Infinity / NaN anyway (allow_nan defaults to True), but those are INVALID
    JSON — strict parsers like the TypeScript frontend and DuckDB choke on them.
    A single bad value (e.g. the eval_mae=inf bug) makes an entire artifact file
    unreadable. So before serializing, we walk the whole structure and turn any
    non-finite float into null (None), which IS valid JSON and signals "no value".

    This is a PURE function: it returns a NEW structure and never mutates the
    input. It recurses through dicts, lists, and tuples; leaves str/int/bool/None
    untouched; and only inspects floats. (bool is excluded explicitly because in
    Python `bool` is a subclass of `int`/`float`-comparable and we must keep
    True/False as-is, not coerce them.)

    Returns the cleaned copy; call it on any payload right before json.dump.
    """
    if isinstance(obj, dict):
        return {k: sanitize_floats(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_floats(v) for v in obj]
    # bool is a subclass of int, so check/skip it before the float branch.
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, float):
        # math.isfinite is False for inf, -inf, and nan. Those → None (JSON null).
        return obj if math.isfinite(obj) else None
    return obj


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    """Write a dict to a JSON file with stable, diff-friendly formatting.

    Formatting choices and *why* they matter:
      • indent=2        → pretty-printed, readable by a human.
      • sort_keys=True  → keys always come out in the same order, so two runs
                          that produce the same data produce byte-identical
                          files. That makes `git diff` and verify_artifacts.py
                          meaningful (a diff means the data really changed).
      • ensure_ascii=False → keep accented characters (é, ñ) as themselves
                          instead of \\uXXXX escapes — nicer for the news/social
                          text fields.
      • trailing "\\n"   → POSIX-friendly final newline.

    JSON SAFETY: the payload is passed through sanitize_floats() first, so any
    non-finite float (inf / -inf / NaN) becomes null. This guarantees every
    artifact we write is valid JSON that the TypeScript frontend and DuckDB can
    parse — fixing the class of bug where one eval_mae=inf corrupted a whole file.
    """
    path = Path(path)
    # Create the parent directory tree if it doesn't exist yet, so callers don't
    # have to remember to mkdir before every write.
    path.parent.mkdir(parents=True, exist_ok=True)
    # Clean non-finite floats → null before serializing (see sanitize_floats).
    safe_payload = sanitize_floats(payload)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(safe_payload, f, indent=2, sort_keys=True, ensure_ascii=False)
        f.write("\n")


def read_json(path: str | Path) -> dict[str, Any]:
    """Read a JSON file and return its contents as a dict.

    Raises FileNotFoundError with a clear message if the file is missing — we
    fail loudly rather than returning an empty dict, so a missing upstream
    artifact surfaces as an obvious error instead of silently corrupting a
    later stage.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Artifact not found: {path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_parquet(path: str | Path, df: Any) -> None:
    """Write a pandas DataFrame to a Parquet file (pyarrow engine, snappy compression).

    pandas/pyarrow are OPTIONAL dependencies (installed via the `[data]` extra),
    so we import lazily and raise a helpful message if they're missing. This
    keeps the core pipeline importable on a minimal install that never touches
    tabular stages.
    """
    try:
        import pandas  # noqa: F401  (imported only to verify it's installed)
    except ImportError as e:
        raise ImportError(
            "pandas is required for write_parquet. Install with: pip install pandas pyarrow"
        ) from e
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # index=False → don't persist the DataFrame's row index as a column; our
    # tables are keyed by real columns (cycle, bloc), not the positional index.
    # snappy → fast compression, the de-facto default for analytics Parquet.
    df.to_parquet(str(path), engine="pyarrow", compression="snappy", index=False)


def read_parquet(path: str | Path) -> Any:
    """Read a Parquet file and return a pandas DataFrame (lazy pandas import)."""
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
    """Write a stage artifact to disk (the function stages actually call).

    The envelope JSON is always written. If `df` is provided, the tabular data
    is ALSO written as a sibling Parquet file with the same stem — e.g. calling
    with path="artifacts/voter_panel.json" and a df produces both
    "artifacts/voter_panel.json" and "artifacts/voter_panel.parquet".

    Args:
        path:     Destination path for the JSON envelope.
        envelope: Dict with keys: stage, run_key, metadata, data.
        df:       Optional DataFrame for tabular stages.
    """
    path = Path(path)
    write_json(path, envelope)
    if df is not None:
        # .with_suffix(".parquet") swaps ".json" → ".parquet" on the same path,
        # giving the sibling table file read_artifact() will look for.
        parquet_path = path.with_suffix(".parquet")
        write_parquet(parquet_path, df)


def read_artifact(path: str | Path) -> tuple[dict[str, Any], Optional[Any]]:
    """Read a stage artifact from disk — the mirror image of write_artifact.

    Returns:
        (envelope, df) where df is None for non-tabular artifacts (no sibling
        .parquet file exists).
    """
    path = Path(path)
    envelope = read_json(path)
    parquet_path = path.with_suffix(".parquet")
    df = None
    # Only tabular stages have a sibling Parquet; its presence is how we know
    # this artifact carries a table.
    if parquet_path.exists():
        df = read_parquet(parquet_path)
    return envelope, df
