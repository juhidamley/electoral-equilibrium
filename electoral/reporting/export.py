from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


def export_json(path: str | Path, data: Any) -> None:
    """Write data as pretty-printed JSON. Creates parent dirs."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    log.info("wrote JSON: %s", p)


def export_csv(path: str | Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    """Write rows to CSV with the given fieldnames. Missing keys become empty.
    Extra keys not in fieldnames are ignored."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
    log.info("wrote CSV: %s (%d rows)", p, len(rows))


def _latex_escape(s: Any) -> str:
    """Escape LaTeX special characters in a cell value."""
    if s is None:
        return "--"
    text = str(s)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def _fmt(val: Any) -> str:
    """Format a value for a LaTeX cell — round floats, escape strings."""
    if isinstance(val, float):
        return f"{val:.4f}"
    if val is None:
        return "--"
    return _latex_escape(val)


def export_latex_table(
    path: str | Path,
    rows: list[dict[str, Any]],
    caption: str,
    label: str,
    fieldnames: list[str] | None = None,
) -> None:
    """Write a booktabs-style LaTeX table fragment.

    Produces a complete table environment with \\toprule / \\midrule /
    \\bottomrule. Column headers are derived from fieldnames (underscores
    replaced with spaces, title-cased).
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []

    n_cols = len(fieldnames)
    col_spec = "l" + "r" * (n_cols - 1)  # first col left, rest right-aligned

    headers = [_latex_escape(fn.replace("_", " ").title()) for fn in fieldnames]

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        f"\\caption{{{_latex_escape(caption)}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        " & ".join(headers) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        cells = [_fmt(row.get(fn)) for fn in fieldnames]
        lines.append(" & ".join(cells) + r" \\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("wrote LaTeX table: %s (%d rows)", p, len(rows))
