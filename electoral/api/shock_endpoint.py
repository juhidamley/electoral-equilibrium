"""FastAPI service: the web backend that the dashboard and webapp talk to.

═══════════════════════════════════════════════════════════════════════════════
BEGINNER ORIENTATION: what a "web backend" / FastAPI is
═══════════════════════════════════════════════════════════════════════════════
This file is the SERVER. The browser (the Next.js webapp) sends HTTP requests to
it ("estimate this shock", "give me the audit log") and it sends back JSON.

FastAPI concepts you'll see below:
  • ROUTE / ENDPOINT — a Python function decorated with @app.get("/path") or
    @app.post(...). When a request hits that URL, FastAPI calls the function and
    turns whatever it returns into a JSON HTTP response.
  • LIFESPAN — a startup/shutdown hook. We load the (large, slow) ML model ONCE
    when the server boots and stash it on `app.state`, instead of reloading it on
    every request. Cleanup (closing the worker pools) runs at shutdown.
  • DEPENDENCY (Depends(...)) — a function FastAPI runs before your route, e.g.
    _require_dashboard_auth, which rejects the request with 401 if the caller
    isn't logged in. Reusable, declarative "run this check first" plumbing.
  • async / await — FastAPI runs on an event loop (one thread juggling many
    requests). An `async def` route can `await` slow work WITHOUT blocking other
    requests. But CPU-heavy work (the optimizer, Monte Carlo, the LLM) would hog
    that single thread — so we hand it off to worker pools (see below).
  • SSE (Server-Sent Events) — a way to stream results to the browser
    incrementally over one long-lived HTTP response, instead of making it wait
    for everything. /estimate/stream sends each stage's result the moment it's
    ready (deltas → equilibrium → simulation → done), so the UI fills in
    progressively. The wire format is plain text: "event: <name>\\ndata: <json>".

Routes:
  POST /estimate              — run the LLM once → ShockResponseData (JSON)
  GET  /estimate/stream       — SSE streaming: deltas → equilibrium → simulation → done
  GET  /api/market-prior      — market consensus price for calibration display
  GET  /api/audit[/count]     — recent estimate log rows / total count (dashboard)
  GET  /api/coverage          — panel data-coverage matrix (dashboard)
  GET  /api/sentiment-dist    — per-bloc sentiment distributions (dashboard)
  GET  /api/bio-coverage      — bio-classifier method counts (dashboard)
  GET  /api/training-logs     — per-epoch train/val loss from HPC logs (dashboard)
  GET  /api/convergence       — win-prob convergence at N=1k/5k/10k (dashboard)
  GET  /health                — liveness check with device info
  GET  /blocs                 — canonical bloc names for all three strata
(The /api/* dashboard routes require a valid session cookie via _require_dashboard_auth.)

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

import ast
import asyncio
import dataclasses
import hashlib
import hmac
import json
import logging
import math
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncGenerator, Optional

import torch
from fastapi import Cookie, Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from electoral.api.audit import AuditLogger
from electoral.artifacts import ShockResponseData, ShockResponseSchema
from electoral.core.io import sanitize_floats
from electoral.core.types import CANONICAL_GENDERS, CANONICAL_RACES, CANONICAL_RELIGIONS
from electoral.llm.inference import ShockEstimator
from electoral.models.benchmarks import DEM_RACE_BENCHMARKS

log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

_ADAPTER_PATH: str = os.environ.get("ADAPTER_PATH", "models/mistral-r16")
_BASE_MODEL: str = os.environ.get("BASE_MODEL", "mistralai/Mistral-7B-v0.3")
_GLOBAL_SEED: int = int(os.environ.get("GLOBAL_SEED", "42"))

# Nominal baseline within-bloc Democratic vote shares (pre-shock).
# Sourced from the documented NEP exit-poll 2000–2024 benchmark set
# (electoral.models.benchmarks.DEM_RACE_BENCHMARKS, ESPINOSA.md §Q1.1–Q1.3) — the
# single source of truth shared with scripts/inspect_moments.py — rather than the
# prior undocumented eyeballed constants. Republican = 1 − democrat (two-party).
_NOMINAL_MU_RACE: dict[str, dict[str, float]] = {
    "democrat": dict(DEM_RACE_BENCHMARKS),
    "republican": {b: round(1.0 - v, 4) for b, v in DEM_RACE_BENCHMARKS.items()},
}

# Nominal within-bloc loyalties for the FIXED strata (religion, gender), used by
# the live API to compute the religion+gender contribution to μ_eff. Approximate
# priors (same spirit as _NOMINAL_MU_RACE); the offline kernel uses the real
# baseline artifact instead. Republican = 1 − democrat by construction.
_NOMINAL_MU_RELIGION: dict[str, dict[str, float]] = {
    "democrat": {
        "evangelical": 0.24,
        "catholic": 0.50,
        "protestant": 0.45,
        "secular": 0.70,
        "jewish": 0.68,
        "muslim": 0.65,
        "other_rel": 0.58,
    },
    "republican": {
        "evangelical": 0.76,
        "catholic": 0.50,
        "protestant": 0.55,
        "secular": 0.30,
        "jewish": 0.32,
        "muslim": 0.35,
        "other_rel": 0.42,
    },
}
_NOMINAL_MU_GENDER: dict[str, dict[str, float]] = {
    "democrat": {"women": 0.55, "men": 0.45, "other_gender": 0.70},
    "republican": {"women": 0.45, "men": 0.55, "other_gender": 0.30},
}

# V_eq fallbacks — used only if configs/party_config.json can't be loaded.
# Kept IN SYNC with the canonical EC-adjusted values in that file (0.5066 Dem /
# 0.4934 Rep) so the fallback can't silently disagree with production.
_V_EQ_DEFAULT: dict[str, float] = {"democrat": 0.5066, "republican": 0.4934}


# ── Config loaders ────────────────────────────────────────────────────────────


# ── Dashboard auth ────────────────────────────────────────────────────────────
# HOW THE LOGIN WORKS: when a user logs in (in the webapp), the server hands them
# a signed "session token" stored in a cookie. On every later dashboard request
# the browser sends that cookie back, and we re-check the signature here. The
# token is "{expiry_ms}.{hmac_hex}":
#   • expiry_ms = when the session expires (Unix milliseconds).
#   • hmac_hex  = an HMAC-SHA256 signature of the expiry, keyed by a SECRET only
#     the server knows (DASHBOARD_SESSION_SECRET).
# Because the attacker doesn't know the secret, they can't forge a valid
# signature — they can't just edit the expiry to extend their session. The TS
# side (webapp/lib/session.ts) builds tokens the exact same way, so both agree.


def _verify_session_token(token: str, secret: str) -> bool:
    """Return True iff the dashboard_session token is valid and unexpired."""
    # Fail closed: a missing token or (critically) a missing server secret means
    # "not authenticated", never "allow". Never default-allow on misconfiguration.
    if not token or not secret:
        return False
    # Split "{expiry}.{signature}" on the first dot.
    dot = token.find(".")
    if dot == -1:
        return False
    expiry_str = token[:dot]
    sig_hex = token[dot + 1 :]
    try:
        expiry_ms = int(expiry_str)
    except ValueError:
        return False
    # Reject expired sessions (current time past the token's expiry).
    if time.time() * 1000 > expiry_ms:
        return False
    try:
        sig_bytes = bytes.fromhex(sig_hex)
    except ValueError:
        return False
    # Recompute what the signature SHOULD be from the expiry + our secret...
    expected = hmac.new(
        secret.encode("utf-8"),
        expiry_str.encode("utf-8"),
        hashlib.sha256,
    ).digest()
    # ...and compare. compare_digest is a CONSTANT-TIME comparison: it always
    # takes the same time regardless of where two byte strings first differ. A
    # naive `==` can leak the secret via timing (an attacker measures how long
    # comparisons take to guess the signature byte by byte). This is the secure way.
    return hmac.compare_digest(expected, sig_bytes)


def _require_dashboard_auth(
    dashboard_session: Optional[str] = Cookie(default=None),
) -> None:
    """FastAPI DEPENDENCY — gate a route behind login.

    Declaring `_: None = Depends(_require_dashboard_auth)` on a route makes
    FastAPI run this FIRST and raise 401 (Unauthorized) if the session cookie is
    missing or invalid — so the route body only runs for authenticated callers.
    `Cookie(default=None)` tells FastAPI to pull the value from the request's
    `dashboard_session` cookie.
    """
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


def _find_artifact_dir() -> Optional[Path]:
    for p in _ARTIFACT_CANDIDATES:
        if p.is_dir():
            return p
    return None


def _find_config_dir() -> Optional[Path]:
    for p in _CONFIG_CANDIDATES:
        if p.is_dir():
            return p
    return None


def _load_json_artifact(path: Path) -> Optional[dict[str, Any]]:
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
    fixed_loyalty: float | None = None,
) -> dict[str, Any]:
    """Run the coalition optimizer for the SSE stream and return its dict form.

    Thin adapter over the single canonical optimizer
    (electoral.optimization.cvx.solve_rebalanced). It stays a MODULE-LEVEL
    function on purpose: ProcessPoolExecutor pickles it by qualified name, and a
    closure/lambda would not be picklable.

    MUST run in ProcessPoolExecutor — CVXPY's C solvers are NOT thread-safe and
    will segfault under ThreadPoolExecutor. run_in_executor(None, ...) defaults to
    the event loop's THREAD pool — never pass None for this function.

    Note: by going through the canonical optimizer, the API now enforces the same
    per-bloc [0.05, 0.60] weight bounds the kernel always used (previously this
    path ran unbounded), and uses the same λ-weighted μ_eff objective.
    `fixed_loyalty` is the real religion+gender contribution (None → neutral 0.5).
    """
    from electoral.optimization.cvx import solve_rebalanced as _solve_rebalanced

    return _solve_rebalanced(
        mu_tilde, cov_list, target, party=party, shock=shock_id, fixed_loyalty=fixed_loyalty
    ).to_dict()


def _run_mc_thread(
    equilibrium_dict: dict[str, Any],
    config_seed: int,
    n_simulations: int,
    cov_delta: list[list[float]] | None = None,
) -> dict[str, Any]:
    """Run the Logistic-Normal ILR Monte Carlo in the thread pool.

    Monte Carlo is pure NumPy which releases the GIL during array operations,
    so it runs safely and cheaply in ThreadPoolExecutor — no process spawn or
    pickling cost, unlike the CVXPY optimizer.

    cov_delta is the shock's real 5×5 race covariance (Σ_Δ); passing it lets the
    win-probability CI reflect actual cross-bloc correlation instead of the
    isotropic diagonal fallback. None → fall back to the diagonal.
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

    return run_ilr_montecarlo(
        equilibrium, _MinConfig(config_seed), n_simulations, cov_delta=cov_delta
    ).to_dict()


# ── SSE helpers ───────────────────────────────────────────────────────────────


def _sse(name: str, payload: Any) -> str:
    """Format a named SSE frame.

    sanitize_floats first: a non-finite value (inf/nan) would serialize to the
    invalid-JSON token Infinity/NaN, which the browser's JSON.parse rejects —
    breaking the whole stream. Mapping them to null keeps every frame parseable.
    """
    return f"event: {name}\ndata: {json.dumps(sanitize_floats(payload))}\n\n"


# ── Lifespan ──────────────────────────────────────────────────────────────────


# @asynccontextmanager turns this generator into a startup/shutdown handler.
# Everything BEFORE `yield` runs once when the server boots; everything AFTER
# runs once when it shuts down. FastAPI is told to use it via FastAPI(lifespan=...)
# below. The pattern: set up expensive shared resources on app.state, hand
# control to the running server (yield), then tear them down cleanly.
@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    # Load the fine-tuned LLM ONCE at startup (it's seconds-to-load and GBs in
    # memory). Every request reuses this one instance via app.state.estimator.
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

    app.state.party_config = _load_party_config()  # V_eq thresholds per party
    app.state.seed = _GLOBAL_SEED  # global RNG seed for reproducible Monte Carlo

    # Real Σ_Δ: the 5×5 Ledoit-Wolf race covariance from the historical panel,
    # computed ONCE here because it's shock-independent. The live estimator emits an
    # identity placeholder; we override shock.covariance with this so the deltas
    # frame, optimizer, and Monte Carlo all see real bloc covariance. The builder
    # logs a WARNING and returns a small diagonal if the panel parquet is absent —
    # we re-check here and log loudly so a fallback can never masquerade as real.
    from electoral.kernels.shock import build_sigma_delta_from_panel

    _artifact_dir = _find_artifact_dir()
    _panel_path = (
        (_artifact_dir / "panel" / "panel_race.parquet")
        if _artifact_dir is not None
        else Path("artifacts/panel/panel_race.parquet")
    )
    app.state.sigma_delta = build_sigma_delta_from_panel(_panel_path)
    _sd = torch.tensor(app.state.sigma_delta, dtype=torch.float64)
    _is_diagonal = bool(torch.allclose(_sd, torch.diag(torch.diag(_sd))))
    if _is_diagonal:
        log.warning(
            "Σ_Δ is DIAGONAL — panel parquet missing/insufficient at %s. Win-prob CI "
            "reflects the isotropic fallback, NOT real bloc covariance. (Modal: ensure "
            "panel_race.parquet is added to the image; see deploy/modal_app.py.)",
            _panel_path,
        )
    else:
        log.info("Σ_Δ loaded: real Ledoit-Wolf 5×5 (non-diagonal) from %s", _panel_path)
    _audit_db = os.environ.get("AUDIT_DB_PATH", "data/audit.duckdb")
    app.state.audit = AuditLogger(_audit_db)  # persistent on Modal Volume; local fallback

    yield  # ← server runs here, handling requests, until it's told to stop

    # Shutdown: close the worker pools cleanly. wait=True lets any in-flight
    # optimizer/MC job finish before the process exits (no half-done work).
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
    sigma_delta: list[list[float]] = app.state.sigma_delta  # real 5×5 Ledoit-Wolf Σ_Δ

    target = _target_for_party(party, party_config)
    nominal_mu = _NOMINAL_MU_RACE[party]
    nominal_rel = _NOMINAL_MU_RELIGION[party]
    nominal_gen = _NOMINAL_MU_GENDER[party]

    async def _generate() -> AsyncGenerator[str, None]:
        # This async generator yields SSE frames one at a time; FastAPI streams
        # each `yield`ed string to the browser immediately. Between stages we run
        # the heavy work in a worker pool so the event loop stays free.
        loop = asyncio.get_running_loop()
        # ── Stage 1: LLM (thread pool) ────────────────────────────────────────
        # loop.run_in_executor(pool, fn, *args) runs the blocking fn(*args) on a
        # SEPARATE worker (in `pool`) and `await`s its result without freezing the
        # event loop — so other requests keep being served while the LLM thinks.
        # Here: estimator.estimate({"description": event, "party": party}, intensity).
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

        # Recompute delta_eff from the predicted bins rather than using the LLM's
        # own scalar prediction, which drifts from the bins it actually emitted.
        # Nominal loyalties serve as coalition-weight proxies (per-party, so the
        # asymmetry is correct: a scandal hurts Democrat AA more than Republican AA).
        # The deltas already carry the intensity multiplier from ShockEstimator.estimate.
        from electoral.simulation.montecarlo import _load_layer_weights
        _lw = _load_layer_weights()
        _eff = (
            _lw["lambda_1"] * sum(
                nominal_mu[b] * shock.deltas_race.get(b, 0.0) for b in CANONICAL_RACES
            )
            + _lw["lambda_2"] * sum(
                nominal_rel[R] * shock.deltas_religion.get(R, 0.0) for R in CANONICAL_RELIGIONS
            )
            + _lw["lambda_3"] * sum(
                nominal_gen[G] * shock.deltas_gender.get(G, 0.0) for G in CANONICAL_GENDERS
            )
        )
        # Override the estimator's identity-placeholder covariance with the real
        # Ledoit-Wolf Σ_Δ computed once at startup. This single replace makes the
        # deltas frame, the optimizer (cov_list below), and the Monte Carlo all use
        # real bloc covariance instead of identity.
        shock = dataclasses.replace(shock, delta_eff=float(_eff), covariance=sigma_delta)
        yield _sse("deltas", shock.to_dict())

        # ── Compute mu_tilde (nominal + delta per bloc) ────────────────────────
        mu_tilde = {
            b: float(max(0.01, min(0.99, nominal_mu[b] + shock.deltas_race.get(b, 0.0))))
            for b in CANONICAL_RACES
        }
        cov_list: list[list[float]] = shock.covariance

        # Religion+gender contribution to μ_eff, from nominal loyalties shifted by
        # this shock's religion/gender deltas — so the API optimizer uses the same
        # μ_eff basis as the kernel (the "μ_eff basis" fix), not neutral 0.5.
        from electoral.optimization.dqcp import compute_fixed_loyalty

        fixed_loyalty = compute_fixed_loyalty(
            nominal_rel,
            nominal_gen,
            _lw["lambda_2"],
            _lw["lambda_3"],
            deltas_religion=shock.deltas_religion,
            deltas_gender=shock.deltas_gender,
        )

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
                fixed_loyalty,
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
                cov_list,  # the shock's real Σ_Δ → CI reflects true bloc correlation
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


def _lookup_market_artifact(event_text: str, party: str) -> Optional[float]:
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
    best_prob: Optional[float] = None

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


_AUDIT_MAX_LIMIT = 500


@app.get("/api/audit")
def get_audit(
    limit: int = 100,
    search: Optional[str] = None,
    _: None = Depends(_require_dashboard_auth),
) -> list[dict[str, Any]]:
    """Return the most recent estimate audit rows (newest first).

    limit is clamped to _AUDIT_MAX_LIMIT to prevent full-table scans.
    search filters event_text case-insensitively (ILIKE) — parameterized, never
    string-interpolated, to prevent SQL injection.
    Returns all columns including party.
    """
    limit = min(max(1, limit), _AUDIT_MAX_LIMIT)
    return app.state.audit.recent(limit, search=search)


@app.get("/api/audit/count")
def get_audit_count(
    _: None = Depends(_require_dashboard_auth),
) -> dict[str, int]:
    """Return the total number of estimate rows without fetching all data."""
    return {"count": app.state.audit.count()}


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
    lookup: dict[tuple[int, str], tuple[Optional[float], Optional[str]]] = {}
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
    category: Optional[str] = None,
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
    note: Optional[str] = None
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
    rawdata_dir: Optional[Path] = None
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


# ── HPC training-log helpers ──────────────────────────────────────────────────
# Logs are trainer stdout synced from the HPC via Syncthing into rawdata/hpc_logs/.
# Files are SLURM stdout (.out) or plain .log files from the fine-tuning job.

_HPC_LOGS_CANDIDATES = [
    Path("rawdata") / "hpc_logs",
    Path(__file__).parent.parent.parent / "rawdata" / "hpc_logs",
]

# Regex to extract dict-like substrings from log lines.
# Matches the HF Trainer repr() output: {'loss': 0.356, 'epoch': 0.08, ...}
# Also handles eval-log dicts: {'eval_loss': 0.234, ...}
_LOG_DICT_RE = re.compile(r"\{[^{}]+\}")

# Completion markers that indicate a run finished normally (not OOM/quota crash).
_COMPLETION_MARKERS = frozenset(
    [
        "training completed",
        "train metrics",  # HF Trainer's final summary block header
        "fine-tuning complete",
        "model saved",
        "saved model to",
    ]
)


def _find_hpc_logs_dir() -> Optional[Path]:
    for p in _HPC_LOGS_CANDIDATES:
        if p.is_dir():
            return p
    return None


def _safe_log_path(base: Path, run_id: str) -> Optional[Path]:
    """Resolve base/run_id and reject if it escapes base — no ../ traversal."""
    try:
        target = (base / run_id).resolve()
        target.relative_to(base.resolve())  # raises ValueError if outside
        return target
    except (ValueError, OSError):
        return None


def _safe_float_log(val: Any) -> Optional[float]:
    """Float conversion that maps inf/NaN → None.  Never serialises as JSON Infinity."""
    if val is None:
        return None
    try:
        f = float(val)
        return None if not math.isfinite(f) else f
    except (TypeError, ValueError):
        return None


def _parse_hpc_log_file(path: Path) -> dict[str, Any]:
    """Parse one HPC training log file into a normalised run dict.

    Reads the file line-by-line so a truncated/crashed log returns whatever
    epochs were completed rather than failing the whole request.

    Handles:
      HF Trainer step logs:  {'loss': X, 'epoch': Y, 'step': Z, 'learning_rate': W}
      HF Trainer eval logs:  {'eval_loss': X, 'epoch': Y, 'step': Z}
      inf values in dicts    — Python repr() emits 'inf'; replaced before ast.literal_eval
    """
    log_history: list[dict[str, Any]] = []
    complete = False
    had_eval_entry = False  # any eval line seen (even if val=inf)
    had_finite_val = False  # at least one finite val_loss

    try:
        with path.open(encoding="utf-8", errors="replace") as fh:
            for raw_line in fh:
                line_lower = raw_line.lower().strip()

                # Completion detection — set once, never cleared
                if not complete and any(m in line_lower for m in _COMPLETION_MARKERS):
                    complete = True

                # Try to parse every {...} fragment on the line
                for m in _LOG_DICT_RE.finditer(raw_line):
                    fragment = m.group()
                    # Python repr() emits bare 'inf'; replace with 1e999 (→ float inf)
                    # so ast.literal_eval can parse it without SyntaxError.
                    fragment_safe = re.sub(r"\binf\b", "1e999", fragment)
                    try:
                        d = ast.literal_eval(fragment_safe)
                    except (ValueError, SyntaxError):
                        continue
                    if not isinstance(d, dict):
                        continue

                    step = d.get("step")
                    if step is None:
                        continue

                    epoch = _safe_float_log(d.get("epoch"))

                    # Training loss — our QLoRA trainer logs as "embedding_loss"
                    train_loss = next(
                        (
                            _safe_float_log(d[k])
                            for k in ("loss", "train_loss", "embedding_loss")
                            if k in d
                        ),
                        None,
                    )

                    # Eval loss — may be inf if in-loop eval was broken
                    raw_val = next((d[k] for k in ("eval_loss", "eval_mae") if k in d), None)
                    eval_loss: Optional[float] = None
                    if raw_val is not None:
                        had_eval_entry = True
                        eval_loss = _safe_float_log(raw_val)
                        if eval_loss is not None:
                            had_finite_val = True

                    if train_loss is None and eval_loss is None:
                        continue

                    log_history.append(
                        {
                            "step": int(step),
                            "epoch": epoch,
                            "train_loss": train_loss,
                            "val_loss": eval_loss,  # null when inf/NaN (never JSON Infinity)
                            "lr": _safe_float_log(d.get("learning_rate")),
                        }
                    )
    except OSError:
        log.debug("_parse_hpc_log_file: could not read %s", path, exc_info=True)

    # Deduplicate by step (keep last occurrence in case of log restarts)
    seen_steps: dict[int, dict[str, Any]] = {}
    for entry in log_history:
        seen_steps[entry["step"]] = entry
    log_history = sorted(seen_steps.values(), key=lambda e: e["step"])

    train_entries = [e for e in log_history if e["train_loss"] is not None]
    eval_entries = [e for e in log_history if e["val_loss"] is not None]

    return {
        "run_id": path.stem,
        "complete": complete,
        # val_available = False means Panel 4 should show "val n/a" rather than
        # an empty chart that looks like missing data.
        "val_available": had_finite_val,
        "val_attempted": had_eval_entry,  # eval ran but may have produced inf
        "log_history": log_history,
        "summary": {
            "n_steps_parsed": log_history[-1]["step"] if log_history else 0,
            "n_epochs_parsed": round(max((e["epoch"] or 0 for e in log_history), default=0), 3),
            "final_train_loss": train_entries[-1]["train_loss"] if train_entries else None,
            "final_val_loss": eval_entries[-1]["val_loss"] if eval_entries else None,
        },
    }


# ── Convergence helpers ───────────────────────────────────────────────────────

_MC_CONVERGENCE_N = (1_000, 5_000, 10_000)

_SIMULATION_PATTERNS = [
    "sim_*.json",
    "*/sim_*.json",
    "smoke/sim_*.json",
    "simulation.json",
    "smoke/simulation.json",
]

_EQUILIBRIUM_PATTERNS = [
    "equilibrium_*.json",
    "*/equilibrium_*.json",
    "smoke/equilibrium_*.json",
]


def _find_most_recent_artifact(artifact_dir: Path, patterns: list[str]) -> Optional[Path]:
    """Return the newest file matching any of the glob patterns."""
    candidates: list[Path] = []
    for pat in patterns:
        candidates.extend(artifact_dir.glob(pat))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


# ── Dashboard data routes ─────────────────────────────────────────────────────


@app.get("/api/training-logs")
def api_training_logs(
    run_id: Optional[str] = None,
    _: None = Depends(_require_dashboard_auth),
) -> dict[str, Any]:
    """Per-epoch train/val loss parsed from HPC SLURM stdout logs in rawdata/hpc_logs/.

    Reads *.log and *.out files synced from the HPC via Syncthing.  Parses
    HuggingFace Trainer dict-style log lines defensively — a run that crashed
    mid-epoch (OOM, quota timeout) returns whatever steps were logged, with
    complete=false.

    val_loss is null (not JSON Infinity) whenever the in-loop eval produced
    inf or NaN (the eval_mae=inf bug); val_available=false signals Panel 4
    to show "val n/a" rather than an empty chart.

    Path traversal: if run_id is given, it is resolved under rawdata/hpc_logs/
    only — any ../ attempt returns 400.
    """
    hpc_dir = _find_hpc_logs_dir()
    if hpc_dir is None:
        return {"status": "no_data", "runs": [], "note": "rawdata/hpc_logs/ not found"}

    # Collect target files
    if run_id is not None:
        safe = _safe_log_path(hpc_dir, run_id)
        if safe is None:
            raise HTTPException(status_code=400, detail="Invalid run_id — path traversal rejected")
        # Accept the run_id as a bare filename or with a known extension
        candidates: list[Path] = []
        for ext in ("", ".log", ".out", ".txt"):
            p = safe if ext == "" else hpc_dir / (run_id + ext)
            if p.is_file():
                candidates.append(p)
        if not candidates:
            return {"status": "no_data", "runs": [], "note": f"run_id {run_id!r} not found"}
    else:
        candidates = sorted(
            p for p in hpc_dir.iterdir() if p.is_file() and p.suffix in (".log", ".out", ".txt", "")
        )

    runs: list[dict[str, Any]] = []
    for log_path in candidates:
        # Double-check traversal even for directory scan (symlinks etc.)
        try:
            log_path.resolve().relative_to(hpc_dir.resolve())
        except ValueError:
            log.warning("api_training_logs: skipping %s (outside hpc_logs)", log_path)
            continue
        runs.append(_parse_hpc_log_file(log_path))

    if not runs:
        return {"status": "no_data", "runs": []}
    return {"status": "ok", "runs": runs}


@app.get("/api/convergence")
def api_convergence(
    _: None = Depends(_require_dashboard_auth),
) -> dict[str, Any]:
    """Win-probability convergence at N=1k/5k/10k with 90% bootstrap CI.

    Loads the most recent SimulationData artifact for shock/party context, then
    re-runs run_ilr_montecarlo at N=1 000, 5 000, 10 000 from the corresponding
    EquilibriumData.  Each N is an independent MC run — win-flag draws are not
    retained between calls, so this is NOT a subsample of the production N=10k
    run; the response includes a note to that effect.

    Returns {"status": "no_data"} if artifacts are absent or MC fails.
    """
    artifact_dir = _find_artifact_dir()
    if artifact_dir is None:
        return {"status": "no_data"}

    # ── Load most-recent SimulationData for shock/party context ──────────────
    sim_path = _find_most_recent_artifact(artifact_dir, _SIMULATION_PATTERNS)
    sim_payload = _load_json_artifact(sim_path) if sim_path else None

    # ── Load most-recent EquilibriumData for MC input ─────────────────────────
    eq_path = _find_most_recent_artifact(artifact_dir, _EQUILIBRIUM_PATTERNS)
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

    # ── Re-run MC at each N ───────────────────────────────────────────────────
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

    # Extract shock/party from sim artifact if available, else from equilibrium
    shock = (sim_payload or {}).get("shock") or equilibrium.shock
    party = (sim_payload or {}).get("party") or equilibrium.party

    return {
        "status": "ok",
        "shock": shock,
        "party": party,
        "series": series,
        # Transparency: all three runs share derive_seed("monte_carlo") so the N=1k
        # and N=5k draws are prefix-nested within the N=10k draw — they are NOT
        # independent. This makes the convergence curve smoother/monotone but is
        # less statistically rigorous than varying seeds. Documented here so the
        # panel caption can be accurate.
        "note": (
            "N=1k, 5k, and 10k runs share the same ILR Monte Carlo seed "
            "(derive_seed('monte_carlo')), so the smaller runs are prefix-nested "
            "subsamples of the N=10k draw — not independent. Convergence will "
            "appear smoother than true repeated sampling."
        ),
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
