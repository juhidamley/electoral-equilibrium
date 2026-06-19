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
from fastapi import FastAPI, HTTPException
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
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request / response models ─────────────────────────────────────────────────


class EstimateRequest(BaseModel):
    event: dict[str, Any]
    intensity: float = 1.0


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
def get_audit(limit: int = 100) -> list[dict[str, Any]]:
    """Return the most recent estimate audit rows (newest first)."""
    return app.state.audit.recent(limit)


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
