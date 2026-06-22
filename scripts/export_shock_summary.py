"""Export the shock-summary table for the paper, in CSV / LaTeX / JSON.

A small reporting SCRIPT (run by hand, not part of the live pipeline). It reads
the per-shock artifacts produced by a pipeline run, builds one summary row per
shock (win probability, equilibrium gap, the biggest-moving bloc, …) via
metrics/tables.py, and writes the table in three formats so it can be dropped
straight into the paper (LaTeX), a spreadsheet (CSV), or reused programmatically
(JSON). This is the table the Week-6 PR description embeds.
"""

import logging
from pathlib import Path
from electoral.config import PipelineConfig
from electoral.metrics.tables import build_shock_summary_table
from electoral.reporting.export import export_csv, export_latex_table, export_json

logging.basicConfig(level=logging.INFO, format="%(message)s")


def main():
    config = PipelineConfig.from_json("configs/base.json")

    import json

    shocks = json.loads(Path("configs/shocks.json").read_text())
    shock_ids = [
        s["id"] for s in (shocks if isinstance(shocks, list) else shocks.get("shocks", []))
    ]

    rows = build_shock_summary_table(shock_ids, config, artifact_dir="artifacts/base")

    fieldnames = [
        "shock_id",
        "category",
        "win_probability",
        "equilibrium_gap",
        "top_moving_bloc",
        "top_moving_delta",
    ]

    export_csv("artifacts/shock_summary_table.csv", rows, fieldnames)
    export_latex_table(
        "artifacts/shock_summary_table.tex",
        rows,
        fieldnames=fieldnames,
        caption="Per-shock electoral equilibrium summary: win probability, "
        "equilibrium gap, and the largest coalition weight shift.",
        label="tab:shock_summary",
    )
    export_json("artifacts/shock_summary_table.json", rows)
    print(f"Exported {len(rows)} shock rows to CSV, LaTeX, and JSON.")


if __name__ == "__main__":
    main()
