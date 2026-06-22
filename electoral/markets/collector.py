"""collector: fetch prediction market prices for each shock event.

BIG PICTURE — and a CRITICAL rule: a "prediction market" lets people bet on
outcomes (e.g. "will the Democrat win?"); the live price ≈ the crowd's implied
probability. We fetch those prices ONLY to display alongside our model and to
check our calibration after a shock resolves — they are NEVER training inputs.
Why never inputs: a market price is a probability of WINNING (Δπ), which is a
different quantity from our model's vote-share margins (Δμ). Mixing them up
(Δπ ≠ Δμ) would corrupt the model. Markets are a yardstick, not an ingredient.

Contract IDs are read from configs/market_contracts.json (never resolved live) so
runs are reproducible — each shock maps to a fixed {platform: contract_id}.

Contract identifiers are loaded from configs/market_contracts.json — never
resolved dynamically. Every shock maps to {platform: contract_id_or_null}.

Supported platforms
-------------------
Polymarket (2020+)  : clob-api.polymarket.com REST API, no auth
PredictIt  (2014+)  : www.predictit.org/api/marketdata/markets/{id}
Metaculus  (2015+)  : www.metaculus.com/api2/questions/{id}/
Manifold   (2022+)  : manifold.markets/api/v0/market/{slug}
IEM                 : offline CSV parse (download from tippie.uiowa.edu/iem)

Prices are collected at four offsets relative to the shock date:
  t - 24h, t + 1h, t + 24h, t + 72h

Usage
-----
    collector = MarketCollector()
    data = collector.collect(shock_id="election_2020", party="democrat")
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONTRACTS_PATH = _REPO_ROOT / "configs" / "market_contracts.json"
_SHOCKS_PATH = _REPO_ROOT / "configs" / "shocks.json"

# Seconds to wait between API calls to be polite
_REQUEST_DELAY = 0.5
_TIMEOUT = 15


def _get_json(url: str, headers: dict[str, str] | None = None) -> Any:
    """HTTP GET → parsed JSON. Returns None on any error."""
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "electoral-equilibrium-research/1.0",
            "Accept": "application/json",
            **(headers or {}),
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError, OSError) as exc:
        logger.debug("GET %s failed: %s", url, exc)
        return None


def _ts(dt: datetime) -> int:
    """UTC datetime → Unix timestamp (int)."""
    return int(dt.replace(tzinfo=timezone.utc).timestamp())


# ── Per-platform fetchers ─────────────────────────────────────────────────────


def _fetch_polymarket(contract_id: str, shock_dt: datetime) -> dict[str, float | None]:
    """Fetch prices at t-24h, t+1h, t+24h, t+72h from Polymarket CLOB API."""
    offsets = {
        "pre_shock_prob": -24,
        "post_shock_1h": 1,
        "post_shock_24h": 24,
        "post_shock_72h": 72,
    }
    result: dict[str, float | None] = {k: None for k in offsets}

    for key, hours in offsets.items():
        target_dt = shock_dt + timedelta(hours=hours)
        # Price history: 1-hour candles around the target time
        start_ts = _ts(target_dt - timedelta(hours=1))
        end_ts = _ts(target_dt + timedelta(hours=1))
        url = (
            f"https://clob-api.polymarket.com/prices-history"
            f"?market={urllib.parse.quote(contract_id)}"
            f"&startTs={start_ts}&endTs={end_ts}&fidelity=60"
        )
        data = _get_json(url)
        if data and isinstance(data, dict):
            history = data.get("history") or []
            if history:
                # Use the last price in the window as the point estimate
                try:
                    result[key] = float(history[-1].get("p") or history[-1].get("price"))
                except (TypeError, ValueError):
                    pass
        time.sleep(_REQUEST_DELAY)

    return result


def _fetch_predictit(contract_id: str, shock_dt: datetime) -> dict[str, float | None]:
    """Fetch current market data from PredictIt. Returns last trade price."""
    # PredictIt's public API returns current/recent data only — no historical time series.
    # For historical shocks, use the downloadable CSV from predictit.org/research.
    url = f"https://www.predictit.org/api/marketdata/markets/{contract_id}"
    data = _get_json(url)
    result: dict[str, float | None] = {
        "pre_shock_prob": None,
        "post_shock_1h": None,
        "post_shock_24h": None,
        "post_shock_72h": None,
    }
    if not data:
        return result

    contracts = data.get("Contracts") or []
    for contract in contracts:
        last_price = contract.get("LastTradePrice")
        if last_price is not None:
            try:
                price = float(last_price)
                # For live shocks, post_shock_1h is the best available point estimate
                result["post_shock_1h"] = price
                result["post_shock_24h"] = price
                break
            except (TypeError, ValueError):
                pass

    time.sleep(_REQUEST_DELAY)
    return result


def _fetch_metaculus(question_id: str, shock_dt: datetime) -> dict[str, float | None]:
    """Fetch community prediction from Metaculus at shock time offsets."""
    url = f"https://www.metaculus.com/api2/questions/{question_id}/"
    data = _get_json(url)
    result: dict[str, float | None] = {
        "pre_shock_prob": None,
        "post_shock_1h": None,
        "post_shock_24h": None,
        "post_shock_72h": None,
    }
    if not data:
        return result

    # Metaculus returns community_prediction as a probability in [0, 1]
    pred = (data.get("community_prediction", {}) or {}).get("full", {}) or {}
    q = pred.get("q2") or pred.get("q3")  # median or mean
    if q is not None:
        try:
            p = float(q)
            result["pre_shock_prob"] = p
            result["post_shock_24h"] = p
        except (TypeError, ValueError):
            pass

    time.sleep(_REQUEST_DELAY)
    return result


def _fetch_manifold(market_slug: str, shock_dt: datetime) -> dict[str, float | None]:
    """Fetch probability from Manifold at shock time offsets."""
    url = f"https://manifold.markets/api/v0/market/{urllib.parse.quote(market_slug)}"
    data = _get_json(url)
    result: dict[str, float | None] = {
        "pre_shock_prob": None,
        "post_shock_1h": None,
        "post_shock_24h": None,
        "post_shock_72h": None,
    }
    if not data:
        return result

    prob = data.get("probability")
    if prob is not None:
        try:
            p = float(prob)
            result["post_shock_24h"] = p
        except (TypeError, ValueError):
            pass

    # Manifold also provides bets history — fetch points around shock_dt
    bets_url = (
        f"https://manifold.markets/api/v0/bets"
        f"?contractSlug={urllib.parse.quote(market_slug)}&limit=500"
    )
    bets = _get_json(bets_url) or []
    if isinstance(bets, list):
        shock_ts = _ts(shock_dt)
        # Find bet closest to t-24h
        pre_target = shock_ts - 86400
        best_pre: tuple[int, float] | None = None
        for bet in bets:
            created = bet.get("createdTime", 0)
            prob_after = bet.get("probAfter")
            if prob_after is None:
                continue
            if created <= shock_ts:
                dist = abs(created - pre_target)
                if best_pre is None or dist < best_pre[0]:
                    best_pre = (dist, float(prob_after))
        if best_pre:
            result["pre_shock_prob"] = best_pre[1]

    time.sleep(_REQUEST_DELAY)
    return result


def _parse_iem_csv(csv_path: Path, shock_dt: datetime) -> dict[str, float | None]:
    """Parse an IEM historical CSV for a single contract around shock_dt.

    IEM CSV format (from tippie.uiowa.edu/iem):
      Date, LastPrice, Volume, ...
    """
    import csv as _csv

    result: dict[str, float | None] = {
        "pre_shock_prob": None,
        "post_shock_1h": None,
        "post_shock_24h": None,
        "post_shock_72h": None,
    }

    if not csv_path.exists():
        logger.warning("IEM CSV not found: %s", csv_path)
        return result

    rows: list[tuple[datetime, float]] = []
    try:
        with open(csv_path, encoding="utf-8", newline="") as f:
            reader = _csv.DictReader(f)
            for row in reader:
                date_str = row.get("Date") or row.get("date") or ""
                price_str = row.get("LastPrice") or row.get("last_price") or row.get("Price") or ""
                try:
                    dt = datetime.strptime(date_str.strip(), "%Y-%m-%d").replace(
                        tzinfo=timezone.utc
                    )
                    price = float(price_str.strip())
                    rows.append((dt, price))
                except (ValueError, AttributeError):
                    continue
    except OSError as exc:
        logger.warning("IEM CSV read error: %s", exc)
        return result

    rows.sort(key=lambda r: r[0])

    def _closest(target: datetime) -> float | None:
        best: tuple[timedelta, float] | None = None
        for dt, price in rows:
            diff = abs(dt - target)
            if best is None or diff < best[0]:
                best = (diff, price)
        return best[1] if best else None

    result["pre_shock_prob"] = _closest(shock_dt - timedelta(hours=24))
    result["post_shock_1h"] = _closest(shock_dt + timedelta(hours=1))
    result["post_shock_24h"] = _closest(shock_dt + timedelta(hours=24))
    result["post_shock_72h"] = _closest(shock_dt + timedelta(hours=72))
    return result


# ── Main collector ────────────────────────────────────────────────────────────


class MarketCollector:
    """Collect prediction market prices for a shock event.

    Loads contract identifiers from configs/market_contracts.json.
    Returns a MarketPrices dict keyed by platform → price_offsets dict.
    """

    def __init__(
        self,
        contracts_path: str | Path = _CONTRACTS_PATH,
        shocks_path: str | Path = _SHOCKS_PATH,
        iem_csv_dir: str | Path | None = None,
    ) -> None:
        self._contracts_path = Path(contracts_path)
        self._shocks_path = Path(shocks_path)
        self._iem_csv_dir = Path(iem_csv_dir) if iem_csv_dir else _REPO_ROOT / "data" / "iem"
        self._contracts = self._load_contracts()
        self._shocks = self._load_shocks()

    def _load_contracts(self) -> dict[str, dict[str, str | None]]:
        if not self._contracts_path.exists():
            logger.warning("market_contracts.json not found at %s", self._contracts_path)
            return {}
        raw = json.loads(self._contracts_path.read_text(encoding="utf-8"))
        # Support both flat dict and {"contracts": {...}} wrapper
        if "contracts" in raw and isinstance(raw["contracts"], dict):
            return raw["contracts"]
        return {k: v for k, v in raw.items() if isinstance(v, dict)}

    def _load_shocks(self) -> dict[str, dict]:
        if not self._shocks_path.exists():
            return {}
        shocks_list = json.loads(self._shocks_path.read_text(encoding="utf-8"))
        return {s["id"]: s for s in shocks_list}

    def _shock_datetime(self, shock_id: str) -> datetime | None:
        shock = self._shocks.get(shock_id)
        if not shock:
            return None
        date_str = (shock.get("date_window", {}) or {}).get("shock_date") or shock.get("date")
        if not date_str:
            return None
        try:
            return datetime.fromisoformat(date_str).replace(tzinfo=timezone.utc)
        except ValueError:
            return None

    def collect(
        self,
        shock_id: str,
    ) -> dict[str, dict[str, float | None]]:
        """Collect prices from all platforms with non-null contract IDs for a shock.

        Returns {platform: {pre_shock_prob, post_shock_1h, post_shock_24h, post_shock_72h}}.
        Platforms with null contract IDs in market_contracts.json are skipped.
        """
        contracts = self._contracts.get(shock_id)
        if not contracts:
            logger.debug("No market contracts for shock '%s'", shock_id)
            return {}

        shock_dt = self._shock_datetime(shock_id)
        if shock_dt is None:
            logger.warning("Cannot determine shock date for '%s'", shock_id)
            return {}

        results: dict[str, dict[str, float | None]] = {}

        polymarket_id = contracts.get("polymarket")
        if polymarket_id:
            logger.info("Fetching Polymarket prices for '%s' ...", shock_id)
            results["polymarket"] = _fetch_polymarket(polymarket_id, shock_dt)

        predictit_id = contracts.get("predictit")
        if predictit_id:
            logger.info("Fetching PredictIt prices for '%s' ...", shock_id)
            results["predictit"] = _fetch_predictit(predictit_id, shock_dt)

        metaculus_id = contracts.get("metaculus")
        if metaculus_id:
            logger.info("Fetching Metaculus prices for '%s' ...", shock_id)
            results["metaculus"] = _fetch_metaculus(metaculus_id, shock_dt)

        manifold_slug = contracts.get("manifold")
        if manifold_slug:
            logger.info("Fetching Manifold prices for '%s' ...", shock_id)
            results["manifold"] = _fetch_manifold(manifold_slug, shock_dt)

        iem_id = contracts.get("iem")
        if iem_id:
            csv_path = self._iem_csv_dir / f"{iem_id}.csv"
            logger.info("Parsing IEM CSV for '%s': %s", shock_id, csv_path)
            results["iem"] = _parse_iem_csv(csv_path, shock_dt)

        logger.info("collect('%s'): %d platforms with data", shock_id, len(results))
        return results

    def collect_all(self, shock_ids: list[str] | None = None) -> dict[str, dict]:
        """Collect prices for all shock IDs that have non-null contracts.

        Returns {shock_id: platform_prices_dict}.
        """
        if shock_ids is None:
            shock_ids = list(self._shocks.keys())

        all_results: dict[str, dict] = {}
        for shock_id in shock_ids:
            prices = self.collect(shock_id)
            if prices:
                all_results[shock_id] = prices

        logger.info(
            "collect_all: %d/%d shocks had market data",
            len(all_results),
            len(shock_ids),
        )
        return all_results
