"""PipelineConfig: global configuration loaded from configs/*.json.

Two valid pipeline_mode values:
  "historical"  — full rebuild from raw survey data (used during SRP development)
  "continuous"  — nightly incremental update; skips voter panel rebuild (post-SRP)
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

from electoral.core.rng import derive_seed as _derive_seed
from electoral.core.types import (
    CANONICAL_GENDERS,
    CANONICAL_RACES,
    CANONICAL_RELIGIONS,
    VALID_PIPELINE_MODES,
)


@dataclasses.dataclass(frozen=True)
class PipelineConfig:
    """Immutable pipeline configuration. Load via PipelineConfig.from_json(path)."""

    run_key: str  # unique identifier for this run (e.g. "base_2026")
    seed: int  # global random seed — all stochastic ops derive from this
    party: str  # "democrat" or "republican"
    target: float  # V_eq equilibrium threshold (~0.535 Dem, ~0.520 Rep)
    data_path: str  # root path for panel/archive data
    output_dir: str  # root path for artifact outputs
    pipeline_mode: str  # "historical" or "continuous"
    races: list[str]  # canonical race identifiers (5)
    religions: list[str]  # canonical religion identifiers (7)
    genders: list[str]  # canonical gender identifiers (3)
    pi_bio_server: str  # Tailscale URL for Pi bio classifier endpoint
    pi_npu_enabled: bool  # True if Hailo NPU is available on Pi

    @classmethod
    def from_json(cls, path: str | Path) -> PipelineConfig:
        """Load config from a JSON file. Extra JSON keys are silently ignored."""
        with open(path, encoding="utf-8") as f:
            raw: dict[str, Any] = json.load(f)

        known = {f.name for f in dataclasses.fields(cls)}
        filtered = {k: v for k, v in raw.items() if k in known}

        # Provide defaults for optional fields not present in smoke.json
        defaults: dict[str, Any] = {
            "data_path": "data/",
            "output_dir": "artifacts/",
            "pipeline_mode": "historical",
            "races": list(CANONICAL_RACES),
            "religions": list(CANONICAL_RELIGIONS),
            "genders": list(CANONICAL_GENDERS),
            "pi_bio_server": "http://100.x.x.x:9000",
            "pi_npu_enabled": False,
        }
        for k, v in defaults.items():
            if k not in filtered:
                filtered[k] = v

        return cls(**filtered)

    def derive_seed(self, stage_name: str) -> int:
        """Return a deterministic per-stage sub-seed."""
        return _derive_seed(self.seed, stage_name)

    def validate(self) -> None:
        """Raise ValueError if the config is invalid."""
        if self.party not in ("democrat", "republican"):
            raise ValueError(
                f"PipelineConfig.party must be 'democrat' or 'republican', " f"got {self.party!r}"
            )
        # Bound is (0.40, 0.70), NOT (0.50, 0.70): the Republican EC-adjusted V_eq
        # is legitimately BELOW 0.50 (~0.4934) — Republicans win the Electoral
        # College with under 50% of the two-party vote due to geographic
        # efficiency (see configs/party_config.json / derive_ec_veq). The old 0.50
        # floor silently rejected every Republican run.
        if not (0.40 < self.target < 0.70):
            raise ValueError(f"PipelineConfig.target must be in (0.40, 0.70), got {self.target}")
        if self.pipeline_mode not in VALID_PIPELINE_MODES:
            raise ValueError(
                f"PipelineConfig.pipeline_mode must be one of "
                f"{sorted(VALID_PIPELINE_MODES)}, got {self.pipeline_mode!r}"
            )
        if self.seed < 0:
            raise ValueError(f"PipelineConfig.seed must be non-negative, got {self.seed}")
