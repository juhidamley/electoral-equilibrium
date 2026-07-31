"""Generate the baseline coalition (BaselinePortfolioData) for BOTH parties.

The pipeline's baseline stage (electoral/stages.py:build_baseline_portfolio) runs
for a single config.party and writes a fixed artifacts/baseline_portfolio.json, so
only one party's coalition ever exists (and the committed smoke one is a degenerate
equal-weight identity-covariance placeholder). This script runs the CURRENT baseline
kernel on the REAL voter panel for democrat AND republican, writing party-namespaced
files so both coexist.

    python scripts/gen_baseline_coalitions.py

Writes:
    artifacts/baseline_portfolio_democrat.json
    artifacts/baseline_portfolio_republican.json

Read-only w.r.t. the panel and existing artifacts (only the two new files are written).
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pandas as pd

from electoral.artifacts import StageArtifact
from electoral.config import PipelineConfig
from electoral.core.io import write_artifact
from electoral.kernels.baseline import build_baseline_portfolio as _build_baseline_kernel


def _load_panel(output_dir: str) -> pd.DataFrame:
    panel_dir = Path(output_dir) / "panel"
    names = ("panel_race.parquet", "panel_religion.parquet", "panel_gender.parquet")
    dfs = [pd.read_parquet(panel_dir / n) for n in names if (panel_dir / n).exists()]
    if not dfs:
        raise FileNotFoundError(f"No panel parquets in {panel_dir} — run build_voter_panel first.")
    return pd.concat(dfs, ignore_index=True)


def main() -> None:
    base = PipelineConfig.from_json("configs/base.json")
    party_cfg = json.loads(Path("configs/party_config.json").read_text())
    v_eq = {p: float(party_cfg[p]["V_eq"]) for p in ("democrat", "republican")}

    panel_df = _load_panel(base.output_dir)  # real panel at artifacts/panel/

    for party in ("democrat", "republican"):
        cfg = dataclasses.replace(
            base, party=party, target=v_eq[party], run_key=f"baseline_{party}"
        )
        cfg.validate()
        payload = _build_baseline_kernel(cfg, panel_df)
        envelope = StageArtifact(
            stage="baseline_portfolio",
            run_key=cfg.run_key,
            metadata={"seed": cfg.derive_seed("baseline_portfolio")},
            data=payload.to_dict(),
        )
        out = f"{cfg.output_dir}/baseline_portfolio_{party}.json"
        write_artifact(out, envelope.to_dict())
        print(f"\n{party}  ->  {out}")
        print(f"  method : {payload.method}")
        print(f"  weights: {({k: round(v, 4) for k, v in payload.weights.items()})}")
        print(f"  mu_eff : {payload.mu_eff:.4f}   target(V_eq): {cfg.target}   "
              f"margin: {payload.mu_eff - cfg.target:+.4f}")


if __name__ == "__main__":
    main()
