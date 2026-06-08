"""aggregator: volume-weighted market price aggregation → PredictionMarketData artifact.

volume_weighted_price(market_prices, volumes) weights each platform's price by
its trading volume; equal-weights when volume is unavailable (Metaculus, Manifold).

Usage
-----
    from electoral.markets.collector import MarketCollector
    from electoral.markets.aggregator import MarketAggregator

    collector = MarketCollector()
    prices = collector.collect("election_2024")
    agg = MarketAggregator()
    artifact = agg.aggregate("election_2024", "democrat", prices)
    agg.write(artifact)
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path

from electoral.artifacts import PredictionMarketData

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ARTIFACTS_DIR = _REPO_ROOT / "artifacts" / "markets"


def volume_weighted_price(
    market_prices: dict[str, float | None],
    volumes: dict[str, float | None] | None = None,
) -> float | None:
    """Compute volume-weighted average of platform prices.

    Platforms with None price are excluded. Platforms with None volume (or absent
    from volumes dict) receive equal weight relative to platforms that DO have volume.
    If all volumes are None, falls back to simple equal-weight average.

    Args:
        market_prices: {platform: price_in_[0,1] | None}
        volumes: {platform: dollar_volume | None}; None means equal-weight.

    Returns:
        Volume-weighted price in [0, 1], or None if no prices available.
    """
    valid = {
        p: price for p, price in market_prices.items() if price is not None and math.isfinite(price)
    }
    if not valid:
        return None

    vols = volumes or {}

    # Separate platforms with known volume from those without
    known_vol: dict[str, float] = {}
    unknown_vol: list[str] = []
    for platform in valid:
        v = vols.get(platform)
        if v is not None and math.isfinite(v) and v > 0:
            known_vol[platform] = v
        else:
            unknown_vol.append(platform)

    if not known_vol:
        # All unknown → simple average
        return sum(valid.values()) / len(valid)

    # Assign equal-weight to unknowns based on average known volume
    avg_known = sum(known_vol.values()) / len(known_vol)
    weights: dict[str, float] = {**known_vol}
    for p in unknown_vol:
        weights[p] = avg_known

    total_weight = sum(weights[p] for p in valid)
    if total_weight == 0:
        return sum(valid.values()) / len(valid)

    return sum(valid[p] * weights[p] for p in valid) / total_weight


def _aggregate_offset(
    platform_prices: dict[str, dict[str, float | None]],
    offset_key: str,
    volumes: dict[str, float | None] | None,
) -> float | None:
    """Aggregate a single time-offset price across all platforms."""
    prices = {platform: prices.get(offset_key) for platform, prices in platform_prices.items()}
    return volume_weighted_price(prices, volumes)


class MarketAggregator:
    """Aggregate multi-platform market prices into PredictionMarketData artifacts."""

    def __init__(self, artifacts_dir: str | Path = _ARTIFACTS_DIR) -> None:
        self._artifacts_dir = Path(artifacts_dir)

    def aggregate(
        self,
        shock_id: str,
        party: str,
        platform_prices: dict[str, dict[str, float | None]],
        volumes: dict[str, float | None] | None = None,
    ) -> PredictionMarketData | None:
        """Build a PredictionMarketData artifact from collected platform prices.

        Args:
            shock_id: Shock event identifier.
            party: "democrat" or "republican".
            platform_prices: Output from MarketCollector.collect() —
                {platform: {pre_shock_prob, post_shock_1h, post_shock_24h, post_shock_72h}}.
            volumes: Optional {platform: dollar_volume} for weighting.

        Returns:
            PredictionMarketData artifact, or None if no data available.
        """
        if not platform_prices:
            logger.debug("aggregate('%s'): no platform prices — returning None", shock_id)
            return None

        pre = _aggregate_offset(platform_prices, "pre_shock_prob", volumes)
        post_1h = _aggregate_offset(platform_prices, "post_shock_1h", volumes)
        post_24h = _aggregate_offset(platform_prices, "post_shock_24h", volumes)
        post_72h = _aggregate_offset(platform_prices, "post_shock_72h", volumes)

        if pre is None:
            logger.warning(
                "aggregate('%s'): no pre_shock_prob across any platform — cannot compute delta",
                shock_id,
            )
            return None

        delta = (post_24h - pre) if post_24h is not None else 0.0

        sources = sorted(platform_prices.keys())
        # contract_ids are resolved from the prices dict keys — the collector
        # only fetches for non-null IDs, so sources == platforms with real IDs.
        # Write platform name as the contract_id placeholder (collector stores the
        # real ID; we don't repeat it here to avoid duplication).
        contract_ids = {s: s for s in sources}

        artifact = PredictionMarketData(
            shock=shock_id,
            party=party,
            pre_shock_prob=pre,
            post_shock_1h=post_1h,
            post_shock_24h=post_24h,
            post_shock_72h=post_72h,
            delta_prob=delta,
            sources=sources,
            contract_ids=contract_ids,
            volume=volumes,
        )
        artifact.validate()
        logger.info(
            "aggregate('%s', %s): pre=%.3f post_24h=%s delta=%+.3f sources=%s",
            shock_id,
            party,
            pre,
            f"{post_24h:.3f}" if post_24h is not None else "None",
            delta,
            sources,
        )
        return artifact

    def write(self, artifact: PredictionMarketData) -> Path:
        """Write PredictionMarketData to artifacts/markets/{shock_id}.json."""
        self._artifacts_dir.mkdir(parents=True, exist_ok=True)
        out_path = self._artifacts_dir / f"{artifact.shock}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(artifact.to_dict(), f, indent=2, ensure_ascii=False)
        logger.info("Wrote PredictionMarketData → %s", out_path)
        return out_path

    def collect_and_write_all(
        self,
        shock_ids: list[str],
        party: str,
        platform_prices_all: dict[str, dict[str, dict[str, float | None]]],
        volumes: dict[str, dict[str, float | None]] | None = None,
    ) -> dict[str, PredictionMarketData]:
        """Aggregate and write artifacts for all shocks. Returns {shock_id: artifact}."""
        results: dict[str, PredictionMarketData] = {}
        for shock_id in shock_ids:
            prices = platform_prices_all.get(shock_id)
            if not prices:
                continue
            vol = (volumes or {}).get(shock_id)
            artifact = self.aggregate(shock_id, party, prices, volumes=vol)
            if artifact is not None:
                self.write(artifact)
                results[shock_id] = artifact
        logger.info(
            "collect_and_write_all: %d/%d shocks produced artifacts",
            len(results),
            len(shock_ids),
        )
        return results
