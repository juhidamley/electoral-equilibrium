"""FastAPI service: ShockEstimator inference endpoint.

Routes:
  POST /estimate              — run ShockEstimator.estimate() and return ShockResponseData
  GET  /estimate/stream       — SSE streaming: deltas → equilibrium → simulation → done
  GET  /health                — liveness check with device info
  GET  /blocs                 — canonical bloc names for all three strata

ShockEstimator is loaded once at startup via the FastAPI lifespan context and
stored in app.state.estimator.

Concurrency pools (stored on app.state):
  process_pool  — ProcessPoolExecutor(max_workers=1): CVXPY optimizer ONLY.
                  CVXPY's C solvers (SCS/ECOS/OSQP) are NOT thread-safe and will
                  segfault under ThreadPoolExecutor. Never pass None to
                  run_in_executor for the optimizer — that defaults to the event
                  loop's thread pool, which is the segfault path.
  thread_pool   — ThreadPoolExecutor(max_workers=2): LLM inference + Monte Carlo.
                  NumPy releases the GIL during array operations, so MC is safe
                  and cheap in a thread pool (no process spawn / pickling cost).
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncGenerator

import numpy as np
import torch
from fastapi import Cookie, Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from electoral.api.audit import AuditLogger
from electoral.artifacts import EquilibriumData, ShockResponseData, ShockResponseSchema
from electoral.core.types import CANONICAL_GENDERS, CANONICAL_RACES, CANONICAL_RELIGIONS
from electoral.llm.inference import ShockEstimator

log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

_ADAPTER_PATH: str = os.environ.get("ADAPTER_PATH", "models/mistral-r16")
_BASE_MODEL: str = os.environ.get("BASE_MODEL", "mistralai/Mistral-7B-v0.3")
_GLOBAL_SEED: int = int(os.environ.get("GLOBAL_SEED", "42"))

# Nominal baseline within-bloc Democratic vote shares (pre-shock).
# Approximate historical ANES/exit-poll values; replaced by panel data when available.
_NOMINAL_MU_RACE: dict[str, dict[str, float]] = {
    "democrat": {
        "african_american": 0.90,
        "asian": 0.65,
        "latino": 0.65,
        "other_race": 0.60,
        "white": 0.42,
    },
    "republican": {
        "african_american": 0.10,
        "asian": 0.35,
        "latino": 0.35,
        "other_race": 0.40,
        "white": 0.58,
    },
}

# V_eq fallbacks — overridden by configs/party_config.json when available.
_V_EQ_DEFAULT: dict[str, float] = {"democrat": 0.521, "republican": 0.495}


# ── Config loaders ────────────────────────────────────────────────────────────


# ── Dashboard auth ────────────────────────────────────────────────────────────
# Token format mirrors webapp/lib/session.ts: "{expiry_ms}.{hmac_hex}"
# HMAC-SHA256(key=DASHBOARD_SESSION_SECRET, msg=expiry_ms_string)
# crypto.subtle uses the same byte encoding, so Python and TS agree exactly.


def _verify_session_token(token: str, secret: str) -> bool:
    """Return True iff the dashboard_session token is valid and unexpired."""
    if not token or not secret:
        return False
    dot = token.find(".")
    if dot == -1:
        return False
    expiry_str = token[:dot]
    sig_hex = token[dot + 1 :]
    try:
        expiry_ms = int(expiry_str)
    except ValueError:
        return False
    if time.time() * 1000 > expiry_ms:
        return False
    try:
        sig_bytes = bytes.fromhex(sig_hex)
    except ValueError:
        return False
    expected = hmac.new(
        secret.encode("utf-8"),
        expiry_str.encode("utf-8"),
        hashlib.sha256,
    ).digest()
    return hmac.compare_digest(expected, sig_bytes)


def _require_dashboard_auth(
    dashboard_session: str | None = Cookie(default=None),
) -> None:
    """FastAPI dependency — 401 unless the signed dashboard_session cookie is valid."""
    secret = os.environ.get("DASHBOARD_SESSION_SECRET", "")
    if not secret or not dashboard_session or not _verify_session_token(dashboard_session, secret):
        raise HTTPException(status_code=401, detail="Unauthorized")


# ── Dashboard data helpers ────────────────────────────────────────────────────

_ARTIFACT_CANDIDATES = [
    Path("artifacts"),
    Path(__file__).parent.parent.parent / "artifacts",
]
_CONFIG_CANDIDATES = [
    Path("configs"),
    Path(__file__).parent.parent.parent / "configs",
]


def _find_artifact_dir() -> Path | None:
    for p in _ARTIFACT_CANDIDATES:
        if p.is_dir():
            return p
    return None


def _find_config_dir() -> Path | None:
    for p in _CONFIG_CANDIDATES:
        if p.is_dir():
            return p
    return None


def _load_json_artifact(path: Path) -> dict[str, Any] | None:
    """Load a JSON artifact file; return None on any parse or IO error."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        # Accept both bare payload dicts and StageArtifact envelopes
        return raw.get("data", raw) if isinstance(raw, dict) else None
    except Exception:
        log.debug("_load_json_artifact: skipping %s", path, exc_info=True)
        return None


def _load_shocks_index() -> dict[str, str]:
    """Return {shock_id: category} from configs/shocks.json; empty on failure."""
    cfg_dir = _find_config_dir()
    if cfg_dir is None:
        return {}
    path = cfg_dir / "shocks.json"
    try:
        entries = json.loads(path.read_text(encoding="utf-8"))
        return {e["id"]: e["category"] for e in entries if "id" in e and "category" in e}
    except Exception:
        return {}


def _load_taxonomy_categories() -> list[str]:
    """Return canonical category list from configs/shock_taxonomy.json."""
    cfg_dir = _find_config_dir()
    if cfg_dir is None:
        return []
    path = cfg_dir / "shock_taxonomy.json"
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return list(raw.get("categories", []))
    except Exception:
        return []


def _source_quality(source: Any) -> tuple[str, list[str]]:
    """Derive (quality_label, source_list) from a panel 'source' cell value."""
    if not source or (isinstance(source, float)):
        # float NaN from pandas arrives here
        return "missing", []
    s = str(source).strip()
    if not s or s == "nan":
        return "missing", []
    if s.startswith("imputed"):
        return "imputed", [s]
    parts = [p.strip() for p in s.split("+") if p.strip()]
    if len(parts) >= 2:
        return "multi-source", parts
    return "single-source", parts if parts else [s]


def _try_load_panel_df(panel_dir: Path) -> Any:
    """Load the three panel parquets and concatenate; returns None on any failure."""
    try:
        import pandas as pd  # optional dependency

        frames = []
        for name in ("panel_race.parquet", "panel_religion.parquet", "panel_gender.parquet"):
            p = panel_dir / name
            if p.exists():
                try:
                    frames.append(pd.read_parquet(p))
                except Exception:
                    log.debug("_try_load_panel_df: skipping %s", p, exc_info=True)
        return pd.concat(frames, ignore_index=True) if frames else None
    except ImportError:
        return None
    except Exception:
        log.debug("_try_load_panel_df: unexpected error", exc_info=True)
        return None


# ── Config loaders ────────────────────────────────────────────────────────────


def _load_party_config() -> dict[str, Any]:
    """Load configs/party_config.json; return empty dict on any failure."""
    for p in [
        Path("configs/party_config.json"),
        Path(__file__).parent.parent.parent / "configs" / "party_config.json",
    ]:
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                pass
    return {}


def _target_for_party(party: str, party_config: dict[str, Any]) -> float:
    """Extract V_eq for party from loaded config, or return default."""
    if party_config:
        v = party_config.get(party, {})
        if isinstance(v, dict):
            val = v.get("v_eq") or v.get("V_eq")
            if val is not None:
                return float(val)
        elif isinstance(v, (int, float)):
            return float(v)
    return _V_EQ_DEFAULT.get(party, 0.521)


# ── Top-level picklable functions for executor pools ─────────────────────────
# These MUST be module-level (not closures) for ProcessPoolExecutor pickling.


def solve_rebalanced(
    mu_tilde: dict[str, float],
    cov_list: list[list[float]],
    target: float,
    party: str,
    shock_id: str,
) -> dict[str, Any]:
    """Run the DQCP optimizer for a post-shock coalition rebalance.

    MUST run in ProcessPoolExecutor — CVXPY's C solvers (SCS/ECOS/OSQP) are NOT
    thread-safe and will segfault under ThreadPoolExecutor.
    run_in_executor(None, ...) defaults to the event loop's THREAD pool —
    never pass None for this function; that is the segfault path.
    """
    from electoral.optimization.dqcp import compute_mu_eff, solve_dqcp
    from electoral.simulation.montecarlo import _load_layer_weights

    lw = _load_layer_weights()
    lambda_1 = lw["lambda_1"]
    lambda_2 = lw["lambda_2"]
    lambda_3 = lw["lambda_3"]
    # Fixed contribution from religion + gender strata at neutral loyalty (0.50).
    # Actual raked values would come from the voter panel; neutral is conservative.
    fixed_loyalty = (lambda_2 + lambda_3) * 0.5

    cov = np.array(cov_list)
    feasible = True

    try:
        weights = solve_dqcp(
            mu_race=mu_tilde,
            cov_race=cov,
            target=target,
            lambda_1=lambda_1,
            fixed_loyalty=fixed_loyalty,
        )
    except (ValueError, RuntimeError) as exc:
        log.warning("solve_rebalanced: DQCP infeasible (%s); falling back to equal weights", exc)
        feasible = False
        n = len(mu_tilde)
        weights = {b: 1.0 / n for b in mu_tilde}

    mu_eff_shifted = compute_mu_eff(weights, mu_tilde, lambda_1, fixed_loyalty)
    target_met = feasible and (mu_eff_shifted >= target)

    return EquilibriumData(
        method="cvxpy_dqcp",
        party=party,
        shock=shock_id,
        weights=weights,
        mu_shifted=mu_tilde,
        feasible=feasible,
        target_met=target_met,
        target=target,
        mu_eff_shifted=mu_eff_shifted,
    ).to_dict()


def _run_mc_thread(
    equilibrium_dict: dict[str, Any],
    config_seed: int,
    n_simulations: int,
) -> dict[str, Any]:
    """Run the Logistic-Normal ILR Monte Carlo in the thread pool.

    Monte Carlo is pure NumPy which releases the GIL during array operations,
    so it runs safely and cheaply in ThreadPoolExecutor — no process spawn or
    pickling cost, unlike the CVXPY optimizer.
    """
    from electoral.artifacts import EquilibriumData
    from electoral.core.rng import derive_seed
    from electoral.simulation.montecarlo import run_ilr_montecarlo

    equilibrium = EquilibriumData.from_dict(equilibrium_dict)

    class _MinConfig:
        def __init__(self, seed: int) -> None:
            self.seed = seed

        def derive_seed(self, stage: str) -> int:  # noqa: F811
            return derive_seed(self.seed, stage)

    return run_ilr_montecarlo(equilibrium, _MinConfig(config_seed), n_simulations).to_dict()


# ── SSE helpers ───────────────────────────────────────────────────────────────


def _sse(name: str, payload: Any) -> str:
    """Format a named SSE frame."""
    return f"event: {name}\ndata: {json.dumps(payload)}\n\n"


# ── Lifespan ──────────────────────────────────────────────────────────────────


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    app.state.estimator = ShockEstimator(
        adapter_path=_ADAPTER_PATH,
        base_model=_BASE_MODEL,
    )

    # CVXPY's C solvers are NOT thread-safe — must use ProcessPoolExecutor.
    # run_in_executor(None, ...) defaults to the event loop's THREAD pool;
    # never pass None for the optimizer.
    app.state.process_pool = ProcessPoolExecutor(max_workers=1)

    # Monte Carlo is pure NumPy (releases GIL) — ThreadPoolExecutor is safe
    # and avoids the process spawn / pickling overhead of ProcessPoolExecutor.
    app.state.thread_pool = ThreadPoolExecutor(max_workers=2)

    app.state.party_config = _load_party_config()
    app.state.seed = _GLOBAL_SEED
    app.state.audit = AuditLogger()

    yield

    app.state.process_pool.shutdown(wait=True)
    app.state.thread_pool.shutdown(wait=True)


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Electoral Equilibrium — Shock Estimator API",
    version="0.1.0",
    lifespan=_lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8000"],
    # Wildcards for methods/headers are treated as literal "*" strings (not wildcards)
    # by browsers when allow_credentials=True — enumerate explicitly to avoid 403s
    # on CORS preflight for non-simple requests (e.g. POST with Content-Type: application/json).
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
    # Required for the dashboard session cookie to be sent cross-origin
    # (port 3000 → 8000 in local dev).  SameSite=Lax on the cookie provides
    # the same-site restriction; in production a reverse proxy collapses both
    # origins to the same domain so this is moot.
    allow_credentials=True,
)

# ── Request / response models ─────────────────────────────────────────────────


class EstimateRequest(BaseModel):
    event: dict[str, Any]
    intensity: float = 1.0
    party: str = "democrat"


# ── Routes ────────────────────────────────────────────────────────────────────


@app.post("/estimate", response_model=ShockResponseSchema)
def estimate(request: EstimateRequest) -> Any:
    """Run ShockEstimator.estimate() and return constrained delta bins.

    The response_model is ShockResponseSchema — the same Pydantic model used
    by outlines for constrained decoding — so the API contract and the
    generation constraint are always identical.
    """
    event_text = (
        request.event.get("description", "")
        if isinstance(request.event, dict)
        else str(request.event)
    )
    if not event_text or len(event_text.strip()) < 10:
        raise HTTPException(status_code=422, detail="Event description too short")
    if not (0.1 <= request.intensity <= 3.0):
        raise HTTPException(status_code=422, detail="Intensity out of range")

    estimator: ShockEstimator = app.state.estimator
    t0 = time.perf_counter()
    try:
        result: ShockResponseData = estimator.estimate(request.event, intensity=request.intensity)
    except Exception:
        log.exception("Unhandled error in /estimate")
        raise HTTPException(status_code=500, detail="Internal inference error")
    llm_ms = int((time.perf_counter() - t0) * 1000)

    app.state.audit.log_estimate(
        event_text=event_text,
        intensity=request.intensity,
        deltas=result.deltas_race,
        feasible=None,
        target_met=None,
        win_prob=None,
        llm_ms=llm_ms,
        optimizer_ms=0,
        montecarlo_ms=0,
        backend="single",
        party=request.party,
    )

    # Convert frozen dataclass → nested dict matching ShockResponseSchema shape.
    return {
        "delta_bins_race": result.delta_bins_race,
        "delta_bins_religion": result.delta_bins_religion,
        "delta_bins_gender": result.delta_bins_gender,
        "delta_eff": result.delta_eff,
    }


@app.get("/estimate/stream")
async def estimate_stream(
    event: str,
    intensity: float = 1.0,
    party: str = "democrat",
) -> StreamingResponse:
    """Progressive SSE stream: deltas → equilibrium → simulation → done.

    Yields four named events in order:
      event: deltas       — ShockResponseData (LLM stage)
      event: equilibrium  — EquilibriumData   (CVXPY DQCP optimizer)
      event: simulation   — SimulationData    (Logistic-Normal ILR Monte Carlo)
      event: done         — empty payload     (stream complete)

    On any stage failure, yields:
      event: stream_error — {"stage": "...", "message": "..."}
      event: done

    Concurrency contract:
      LLM runs in the thread pool (ThreadPoolExecutor) — safe since inference
        releases the GIL during PyTorch/CUDA operations.
      CVXPY optimizer MUST run in the process pool (ProcessPoolExecutor) —
        CVXPY's C solvers are NOT thread-safe and segfault under threads.
      Monte Carlo runs in the thread pool — pure NumPy releases the GIL.
    """
    if not event or len(event.strip()) < 10:
        raise HTTPException(status_code=422, detail="Event description too short (min 10 chars)")
    if not (0.1 <= intensity <= 3.0):
        raise HTTPException(status_code=422, detail="Intensity must be in [0.1, 3.0]")
    if party not in ("democrat", "republican"):
        raise HTTPException(status_code=422, detail="Party must be 'democrat' or 'republican'")

    estimator: ShockEstimator = app.state.estimator
    process_pool: ProcessPoolExecutor = app.state.process_pool
    thread_pool: ThreadPoolExecutor = app.state.thread_pool
    party_config: dict[str, Any] = app.state.party_config
    seed: int = app.state.seed
    audit: AuditLogger = app.state.audit

    target = _target_for_party(party, party_config)
    nominal_mu = _NOMINAL_MU_RACE[party]

    async def _generate() -> AsyncGenerator[str, None]:
        loop = asyncio.get_running_loop()
        # ── Stage 1: LLM (thread pool) ────────────────────────────────────────
        t0 = time.perf_counter()
        try:
            shock: ShockResponseData = await loop.run_in_executor(
                thread_pool,
                estimator.estimate,
                {"description": event, "party": party},
                intensity,
            )
        except Exception as exc:
            log.exception("SSE /estimate/stream: LLM stage failed")
            yield _sse("stream_error", {"stage": "deltas", "message": str(exc)})
            yield _sse("done", {})
            return

        llm_ms = int((time.perf_counter() - t0) * 1000)
        log.info("SSE deltas: %.0fms", llm_ms)
        yield _sse("deltas", shock.to_dict())

        # ── Compute mu_tilde (nominal + delta per bloc) ────────────────────────
        mu_tilde = {
            b: float(max(0.01, min(0.99, nominal_mu[b] + shock.deltas_race.get(b, 0.0))))
            for b in CANONICAL_RACES
        }
        cov_list: list[list[float]] = shock.covariance

        # ── Stage 2: CVXPY optimizer (process pool) ────────────────────────────
        # CRITICAL: must use process_pool, NOT thread_pool or run_in_executor(None).
        # CVXPY's C solvers (CLARABEL/ECOS/SCS) are NOT thread-safe and segfault
        # when called from ThreadPoolExecutor.
        t1 = time.perf_counter()
        try:
            equilibrium_dict: dict[str, Any] = await loop.run_in_executor(
                process_pool,
                solve_rebalanced,
                mu_tilde,
                cov_list,
                target,
                party,
                shock.shock,
            )
        except Exception as exc:
            log.exception("SSE /estimate/stream: optimizer stage failed")
            yield _sse("stream_error", {"stage": "equilibrium", "message": str(exc)})
            yield _sse("done", {})
            return

        opt_ms = int((time.perf_counter() - t1) * 1000)
        log.info("SSE equilibrium: %.0fms", opt_ms)
        yield _sse("equilibrium", equilibrium_dict)

        # ── Stage 3: Monte Carlo (thread pool) ────────────────────────────────
        # Pure NumPy releases the GIL, so ThreadPoolExecutor is safe and
        # avoids the process-spawn/pickling overhead of ProcessPoolExecutor.
        t2 = time.perf_counter()
        try:
            simulation_dict: dict[str, Any] = await loop.run_in_executor(
                thread_pool,
                _run_mc_thread,
                equilibrium_dict,
                seed,
                10_000,
            )
        except Exception as exc:
            log.exception("SSE /estimate/stream: Monte Carlo stage failed")
            yield _sse("stream_error", {"stage": "simulation", "message": str(exc)})
            yield _sse("done", {})
            return

        mc_ms = int((time.perf_counter() - t2) * 1000)
        log.info(
            "SSE simulation: %.0fms  total: %.0fms", mc_ms, int((time.perf_counter() - t0) * 1000)
        )
        yield _sse("simulation", simulation_dict)

        audit.log_estimate(
            event_text=event,
            intensity=intensity,
            deltas=shock.deltas_race,
            feasible=equilibrium_dict.get("feasible"),
            target_met=equilibrium_dict.get("target_met"),
            win_prob=simulation_dict.get("win_probability"),
            llm_ms=llm_ms,
            optimizer_ms=opt_ms,
            montecarlo_ms=mc_ms,
            backend="stream",
            party=party,
        )

        yield _sse("done", {})

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/market-prior")
def market_prior(party: str = "democrat", event: str = "") -> dict[str, Any]:
    """Return cached prediction market consensus probability for an event/party pair.

    Serves pre-computed artifacts from artifacts/markets/{shock_id}.json written
    by scripts/collect_markets.py (offline job — no live API calls here).
    Matches the free-form event text against known shock IDs via keyword overlap.
    Returns {"probability": <float>} when a match exists, else {"probability": null}.

    Per CLAUDE.md: prediction markets are calibration benchmarks only, not
    training inputs.  The frontend shows this alongside the model output as
    "Market consensus: X%" with a disclaimer explaining the difference.
    """
    try:
        prob = _lookup_market_artifact(event, party)
        return {"probability": prob}
    except Exception:
        log.warning("market_prior: lookup failed", exc_info=True)
        return {"probability": None}


def _lookup_market_artifact(event_text: str, party: str) -> float | None:
    """Find a pre-computed market probability matching event_text and party.

    Searches artifacts/markets/*.json (written by the offline collector).
    Uses keyword overlap between the event text and each shock_id to find
    the best match.  Returns None if no artifact matches or none is found.
    """
    from electoral.artifacts import PredictionMarketData

    for candidate in [
        Path("artifacts/markets"),
        Path(__file__).parent.parent.parent / "artifacts" / "markets",
    ]:
        if candidate.exists():
            artifacts_dir = candidate
            break
    else:
        return None

    # Normalise event text to a set of meaningful keywords
    event_words = set(
        w
        for w in re.sub(r"[^a-z0-9\s]", "", event_text.lower()).split()
        if len(w) > 2  # skip short stop-words
    )
    if not event_words:
        return None

    best_overlap = 0
    best_prob: float | None = None

    for json_path in artifacts_dir.glob("*.json"):
        shock_id = json_path.stem
        # Shock IDs are snake_case; drop pure-digit tokens (years)
        id_words = {w for w in shock_id.split("_") if not w.isdigit()}
        overlap = len(event_words & id_words)
        if overlap == 0 or overlap <= best_overlap:
            continue
        try:
            raw = json.loads(json_path.read_text(encoding="utf-8"))
            artifact = PredictionMarketData.from_dict(raw)
        except Exception:
            continue
        if artifact.party != party:
            continue
        # Prefer post-shock 24h; fall back to 1h then pre-shock
        prob = artifact.post_shock_24h or artifact.post_shock_1h or artifact.pre_shock_prob
        if prob is None:
            continue
        best_overlap = overlap
        best_prob = prob

    return best_prob


@app.get("/api/audit")
def get_audit(
    limit: int = 100,
    search: str | None = None,
    _: None = Depends(_require_dashboard_auth),
) -> list[dict[str, Any]]:
    """Return the most recent estimate audit rows (newest first).

    search filters event_text with a LIKE %search% — parameterized, not interpolated.
    """
    return app.state.audit.recent(limit, search=search)


@app.get("/api/coverage")
def api_coverage(
    _: None = Depends(_require_dashboard_auth),
) -> dict[str, Any]:
    """Data coverage matrix: quality label + vote_share per (bloc, cycle) cell.

    Quality labels: "multi-source" | "single-source" | "imputed" | "missing".
    Derives labels from the panel parquets written by build_voter_panel().
    Returns {"status": "no_data"} if the artifacts are absent rather than 404/500.
    """
    artifact_dir = _find_artifact_dir()
    if artifact_dir is None:
        return {"status": "no_data", "cells": []}

    # ── Load VoterPanelData to determine expected cycles × blocs ─────────────
    panel_payload = _load_json_artifact(artifact_dir / "voter_panel.json")
    if panel_payload is None:
        return {"status": "no_data", "cells": []}

    try:
        from electoral.artifacts import VoterPanelData

        vpd = VoterPanelData.from_dict(panel_payload)
    except Exception:
        log.warning("api_coverage: could not parse VoterPanelData", exc_info=True)
        return {"status": "no_data", "cells": []}

    all_blocs: list[str] = list(vpd.races) + list(vpd.religions) + list(vpd.genders)
    cycles: list[int] = list(vpd.cycles)

    # ── Try to load panel parquets for per-cell vote_share + source ───────────
    panel_dir = artifact_dir / "panel"
    panel_df = _try_load_panel_df(panel_dir)

    # Build lookup {(cycle, bloc): (vote_share, source)}
    lookup: dict[tuple[int, str], tuple[float | None, str | None]] = {}
    if panel_df is not None:
        for _, row in panel_df.iterrows():
            try:
                key = (int(row["cycle"]), str(row["bloc"]))
                vs = float(row["vote_share"]) if row.get("vote_share") is not None else None
                src = str(row.get("source", "")) or None
                lookup[key] = (vs, src)
            except Exception:
                continue

    # ── Build quality matrix ──────────────────────────────────────────────────
    cells: list[dict[str, Any]] = []
    for cycle in cycles:
        for bloc in all_blocs:
            key = (cycle, bloc)
            if key in lookup:
                vote_share, source = lookup[key]
                quality, sources = _source_quality(source)
            else:
                quality, sources, vote_share = "missing", [], None
            cells.append(
                {
                    "cycle": cycle,
                    "bloc": bloc,
                    "quality": quality,
                    "vote_share": vote_share,
                    "sources": sources,
                }
            )

    return {"status": "ok", "cells": cells}


@app.get("/api/sentiment-dist")
def api_sentiment_dist(
    category: str | None = None,
    _: None = Depends(_require_dashboard_auth),
) -> dict[str, Any]:
    """Per-bloc RoBERTa elasticity score arrays, optionally filtered by shock category.

    Returns {"status": "no_data"} when the SentimentData artifact is absent.
    An unrecognised category returns an empty result with a note rather than 500.
    """
    artifact_dir = _find_artifact_dir()
    if artifact_dir is None:
        return {"status": "no_data", "blocs": {}}

    payload = _load_json_artifact(artifact_dir / "sentiment_data.json")
    if payload is None:
        return {"status": "no_data", "blocs": {}}

    try:
        from electoral.artifacts import SentimentData

        sd = SentimentData.from_dict(payload)
    except Exception:
        log.warning("api_sentiment_dist: could not parse SentimentData", exc_info=True)
        return {"status": "no_data", "blocs": {}}

    # ── Category validation ────────────────────────────────────────────────────
    note: str | None = None
    valid_categories = _load_taxonomy_categories()
    if category is not None:
        if valid_categories and category not in valid_categories:
            return {
                "status": "ok",
                "model": sd.model,
                "shocks": [],
                "blocs": {},
                "note": (f"Unknown category {category!r}. " f"Valid: {sorted(valid_categories)}"),
            }

    # ── Filter shocks by category ──────────────────────────────────────────────
    if category is not None:
        shock_index = _load_shocks_index()
        filtered_shocks = [s for s in sd.shocks if shock_index.get(s) == category]
        if not filtered_shocks:
            note = f"No shocks found for category {category!r}"
    else:
        filtered_shocks = list(sd.shocks)

    # ── Build per-bloc arrays restricted to filtered_shocks ───────────────────
    shock_set = set(filtered_shocks)
    blocs_out: dict[str, dict[str, float]] = {}
    for bloc, shock_scores in sd.scores.items():
        filtered = {sid: v for sid, v in shock_scores.items() if sid in shock_set}
        if filtered:
            blocs_out[bloc] = filtered

    return {
        "status": "ok",
        "model": sd.model,
        "shocks": filtered_shocks,
        "blocs": blocs_out,
        "note": note,
    }


@app.get("/api/bio-coverage")
def api_bio_coverage(
    _: None = Depends(_require_dashboard_auth),
) -> dict[str, Any]:
    """Per-shock bio-classifier coverage: keyword / SetFit / fallback counts.

    Scans rawdata/social/ for JSONL files that carry bio classification results
    (payload.inference_method field).  Falls back gracefully to {"status": "no_data"}
    when no classified posts exist yet.

    inference_method values (from electoral/nlp/bio_classifier.py):
      "keyword_bio"    — resolved by lexicon keyword match
      "setfit_bio"     — resolved by SetFit model on Pi NPU
      "language_prior" — resolved by language-based fallback
      None             — no bio signal (excluded from counts)
    """
    rawdata_candidates = [
        Path("rawdata") / "social",
        Path(__file__).parent.parent.parent / "rawdata" / "social",
    ]
    rawdata_dir: Path | None = None
    for candidate in rawdata_candidates:
        if candidate.is_dir():
            rawdata_dir = candidate
            break

    counts: dict[str, dict[str, int]] = {}
    n_files_scanned = 0

    if rawdata_dir is not None:
        # Walk rawdata/social/{platform}/{shock_id}/*.jsonl
        for jsonl_path in rawdata_dir.rglob("*.jsonl"):
            n_files_scanned += 1
            # Derive shock_id from path: rawdata/social/{platform}/{shock_id}/file.jsonl
            # or rawdata/social/archive/{shock_id}/file.jsonl
            parts = jsonl_path.relative_to(rawdata_dir).parts
            shock_id = parts[1] if len(parts) >= 3 else parts[0] if parts else "unknown"

            try:
                with jsonl_path.open(encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            record = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        payload_obj = record.get("payload", record)
                        method = payload_obj.get("inference_method")
                        if method is None:
                            continue
                        bucket = counts.setdefault(
                            shock_id,
                            {"keyword": 0, "setfit": 0, "fallback": 0, "total": 0},
                        )
                        if method == "keyword_bio":
                            bucket["keyword"] += 1
                        elif method == "setfit_bio":
                            bucket["setfit"] += 1
                        elif method == "language_prior":
                            bucket["fallback"] += 1
                        bucket["total"] += 1
            except Exception:
                log.debug("api_bio_coverage: error scanning %s", jsonl_path, exc_info=True)

    if not counts:
        return {
            "status": "no_data",
            "shocks": {},
            "note": f"No classified posts found (scanned {n_files_scanned} file(s))",
        }

    return {"status": "ok", "shocks": counts}


# ── Training-log helpers ───────────────────────────────────────────────────────

_TRAINING_LOG_BASES = [
    Path("models"),
    Path("checkpoints"),
    Path(__file__).parent.parent.parent / "models",
    Path(__file__).parent.parent.parent / "checkpoints",
]


def _find_training_log_paths() -> list[tuple[str, Path]]:
    """Return [(run_name, trainer_state_path), ...] from models/ and checkpoints/."""
    seen: set[Path] = set()
    results: list[tuple[str, Path]] = []
    for base in _TRAINING_LOG_BASES:
        if not base.is_dir():
            continue
        for ts_path in sorted(base.rglob("trainer_state.json")):
            resolved = ts_path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            results.append((ts_path.parent.name, ts_path))
    return results


def _parse_trainer_state(raw: dict[str, Any]) -> dict[str, Any]:
    """Normalise a trainer_state.json dict into a run summary for the dashboard."""
    if "log_history" in raw:
        # HuggingFace Trainer checkpoint format
        log_history: list[dict[str, Any]] = []
        for entry in raw.get("log_history", []):
            step = entry.get("step")
            if step is None:
                continue
            # Loss key: standard HF uses "loss"; our QLoRA trainer uses "embedding_loss"
            train_loss = next(
                (entry[k] for k in ("loss", "train_loss", "embedding_loss") if k in entry),
                None,
            )
            eval_loss = next(
                (entry[k] for k in ("eval_loss", "eval_mae") if k in entry),
                None,
            )
            log_history.append(
                {
                    "step": step,
                    "epoch": entry.get("epoch"),
                    "train_loss": train_loss,
                    "eval_loss": eval_loss,
                    "lr": entry.get("learning_rate"),
                }
            )
        train_entries = [e for e in log_history if e["train_loss"] is not None]
        return {
            "type": "checkpoint",
            "log_history": log_history,
            "summary": {
                "n_steps": raw.get("global_step"),
                "epochs": raw.get("epoch") or raw.get("num_train_epochs"),
                "final_train_loss": train_entries[-1]["train_loss"] if train_entries else None,
                "base_model": raw.get("base_model"),
            },
        }
    else:
        # Summary format written by our fine-tuning script on completion
        return {
            "type": "summary",
            "log_history": [],
            "summary": {
                k: raw[k]
                for k in (
                    "epochs",
                    "training_loss",
                    "eval_mae",
                    "n_train",
                    "n_eval",
                    "lora_rank",
                    "lora_alpha",
                    "base_model",
                    "seed",
                )
                if k in raw
            },
        }


# ── Convergence helper ────────────────────────────────────────────────────────

_MC_CONVERGENCE_N = (1_000, 5_000, 10_000)

_EQUILIBRIUM_PATTERNS = [
    "equilibrium_*.json",
    "*/equilibrium_*.json",
    "smoke/equilibrium_*.json",
]


def _find_equilibrium_path(artifact_dir: Path) -> Path | None:
    """Return the first equilibrium artifact found, preferring shock-specific ones."""
    for pattern in _EQUILIBRIUM_PATTERNS:
        candidates = sorted(artifact_dir.glob(pattern))
        if candidates:
            return candidates[0]
    return None


# ── New dashboard data routes ─────────────────────────────────────────────────


@app.get("/api/training-logs")
def api_training_logs(
    _: None = Depends(_require_dashboard_auth),
) -> dict[str, Any]:
    """Per-step train/eval loss history for every model run found in models/ and checkpoints/.

    Two formats are handled transparently:
      HuggingFace checkpoint: has log_history → per-step series (step, epoch, train_loss, eval_loss, lr)
      Summary format: our fine-tuning script's end-of-run summary → single-point totals

    Returns {"status": "no_data"} when neither source is found.
    """
    paths = _find_training_log_paths()
    if not paths:
        return {"status": "no_data", "runs": []}

    runs: list[dict[str, Any]] = []
    for run_name, ts_path in paths:
        try:
            raw = json.loads(ts_path.read_text(encoding="utf-8"))
        except Exception:
            log.debug("api_training_logs: skipping %s", ts_path, exc_info=True)
            continue
        if not isinstance(raw, dict):
            continue
        parsed = _parse_trainer_state(raw)
        runs.append(
            {
                "run_name": run_name,
                "path": str(ts_path.parent),
                **parsed,
            }
        )

    if not runs:
        return {"status": "no_data", "runs": []}
    return {"status": "ok", "runs": runs}


@app.get("/api/convergence")
def api_convergence(
    _: None = Depends(_require_dashboard_auth),
) -> dict[str, Any]:
    """Win-probability convergence at N=1k/5k/10k with 90% bootstrap CI bounds.

    Reads the first available EquilibriumData artifact and runs the ILR Monte Carlo
    at each sample size in _MC_CONVERGENCE_N.  Returns a series of
    {n, win_probability, p5, p95} points — intended for a convergence line chart
    that verifies the MC estimate stabilises before production N=10k.

    Returns {"status": "no_data"} if no equilibrium artifact exists or MC fails.
    """
    artifact_dir = _find_artifact_dir()
    if artifact_dir is None:
        return {"status": "no_data"}

    eq_path = _find_equilibrium_path(artifact_dir)
    if eq_path is None:
        return {"status": "no_data"}

    eq_payload = _load_json_artifact(eq_path)
    if eq_payload is None:
        return {"status": "no_data"}

    try:
        from electoral.artifacts import EquilibriumData

        equilibrium = EquilibriumData.from_dict(eq_payload)
    except Exception:
        log.warning("api_convergence: could not parse EquilibriumData", exc_info=True)
        return {"status": "no_data"}

    try:
        from electoral.core.rng import derive_seed
        from electoral.simulation.montecarlo import run_ilr_montecarlo

        class _Cfg:
            seed = _GLOBAL_SEED

            def derive_seed(self, stage: str) -> int:
                return derive_seed(self.seed, stage)

        cfg = _Cfg()
        series: list[dict[str, Any]] = []
        for n in _MC_CONVERGENCE_N:
            sim = run_ilr_montecarlo(equilibrium, cfg, n_simulations=n)
            series.append(
                {
                    "n": n,
                    "win_probability": sim.win_probability,
                    "p5": sim.win_probability_low,
                    "p95": sim.win_probability_high,
                }
            )
    except Exception:
        log.warning("api_convergence: MC failed", exc_info=True)
        return {"status": "no_data"}

    return {
        "status": "ok",
        "shock": equilibrium.shock,
        "party": equilibrium.party,
        "series": series,
    }


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness check — returns model path and active device."""
    device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    return {"status": "ok", "model": _ADAPTER_PATH, "device": device}


@app.get("/blocs")
def blocs() -> dict[str, list[str]]:
    """Return canonical bloc names for all three demographic strata."""
    return {
        "race": list(CANONICAL_RACES),
        "religion": list(CANONICAL_RELIGIONS),
        "gender": list(CANONICAL_GENDERS),
    }


# ── Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    _port = int(os.environ.get("API_PORT", "8001"))
    uvicorn.run(app, host="0.0.0.0", port=_port)
