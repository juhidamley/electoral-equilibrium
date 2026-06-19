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
