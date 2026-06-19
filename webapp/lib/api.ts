// Typed API client for the Electoral Equilibrium FastAPI backend.
//
// estimateShock validates its response with EstimateResponseSchema, which
// expects the full { shock, equilibrium, simulation } composite payload.
// The current /estimate endpoint (electoral/llm/inference.py) returns only
// ShockResponseData — so until the Week 8 endpoint-widening task lands,
// this function will throw an ApiError with a schema-validation message.
// That is intentional: fail loudly at the boundary rather than propagate
// partially-shaped data into the UI. When the endpoint is widened, this
// client requires no changes.

import {
  EstimateResponseSchema,
  EquilibriumDataSchema,
  ShockResponseDataSchema,
  SimulationDataSchema,
  type EstimateResponse,
} from "./schemas";
import type { EquilibriumData, Party, ShockResponseData, SimulationData } from "./types";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export class ApiError extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

export async function estimateShock(
  event: string,
  intensity: number,
): Promise<EstimateResponse> {
  let res: Response;
  try {
    res = await fetch(`${API_URL}/estimate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ event: { description: event }, intensity }),
    });
  } catch (e) {
    // network-level failure (server down, DNS, CORS preflight)
    throw new ApiError(
      e instanceof Error ? e.message : "Network request failed",
      0,
    );
  }

  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body?.detail ?? detail;
    } catch {
      // body not JSON; keep statusText
    }
    throw new ApiError(detail, res.status);
  }

  let json: unknown;
  try {
    json = await res.json();
  } catch {
    throw new ApiError("Response was not valid JSON", res.status);
  }

  const parsed = EstimateResponseSchema.safeParse(json);
  if (!parsed.success) {
    throw new ApiError(
      `Response failed schema validation: ${parsed.error.issues
        .map((i) => `${i.path.join(".")}: ${i.message}`)
        .join("; ")}`,
      res.status,
    );
  }
  return parsed.data;
}

export async function getBlocs(): Promise<string[]> {
  try {
    const res = await fetch(`${API_URL}/blocs`);
    if (!res.ok) throw new ApiError(res.statusText, res.status);
    const data = await res.json();
    // /blocs returns { race: [...], religion: [...], gender: [...] }
    return [
      ...(data.race ?? []),
      ...(data.religion ?? []),
      ...(data.gender ?? []),
    ];
  } catch (e) {
    if (e instanceof ApiError) throw e;
    throw new ApiError(e instanceof Error ? e.message : "getBlocs failed", 0);
  }
}

export async function healthCheck(): Promise<boolean> {
  try {
    const res = await fetch(`${API_URL}/health`);
    if (!res.ok) return false;
    const data = await res.json();
    return data?.status === "ok";
  } catch {
    return false;
  }
}

// ── SSE streaming client ──────────────────────────────────────────────────────
//
// Opens an EventSource to GET /estimate/stream and dispatches named SSE events
// to the provided callbacks. Zod-validates each payload at the API boundary —
// parse failures call onError rather than propagating undefined/NaN into the UI.
//
// Returns the EventSource so the caller can close it on unmount:
//   const es = estimateShockStream(...);
//   useEffect(() => () => es.close(), []);
//
// Named SSE events from the backend:
//   deltas       → ShockResponseData (LLM stage, ~2s)
//   equilibrium  → EquilibriumData   (CVXPY DQCP optimizer, ~1s)
//   simulation   → SimulationData    (Logistic-Normal ILR Monte Carlo, ~0.5s)
//   done         → stream complete   (no data payload)
//   stream_error → { stage, message } (stage failure; stream ends with "done")
//
// Note: the browser's built-in EventSource fires its own "error" event for
// connection-level failures (network down, CORS, HTTP 4xx/5xx). This is
// handled by es.onerror below and is distinct from the named "stream_error"
// SSE frames emitted by the backend for stage-level failures.

export function estimateShockStream(
  event: string,
  intensity: number,
  party: Party,
  callbacks: {
    onDeltas?: (data: ShockResponseData) => void;
    onEquilibrium?: (data: EquilibriumData) => void;
    onSimulation?: (data: SimulationData) => void;
    onError?: (message: string) => void;
    onDone?: () => void;
  },
): EventSource {
  const url = new URL(`${API_URL}/estimate/stream`);
  url.searchParams.set("event", event);
  url.searchParams.set("intensity", String(intensity));
  url.searchParams.set("party", party);

  const es = new EventSource(url.toString());

  es.addEventListener("deltas", (e: MessageEvent) => {
    const parsed = ShockResponseDataSchema.safeParse(JSON.parse(e.data));
    if (parsed.success) {
      callbacks.onDeltas?.(parsed.data);
    } else {
      const msg = parsed.error.issues.map((i) => `${i.path.join(".")}: ${i.message}`).join("; ");
      callbacks.onError?.(`deltas schema error: ${msg}`);
    }
  });

  es.addEventListener("equilibrium", (e: MessageEvent) => {
    let data: unknown;
    try {
      data = JSON.parse(e.data);
    } catch {
      callbacks.onError?.("equilibrium: invalid JSON payload");
      return;
    }
    const parsed = EquilibriumDataSchema.safeParse(data);
    if (parsed.success) {
      callbacks.onEquilibrium?.(parsed.data);
    } else {
      const msg = parsed.error.issues.map((i) => `${i.path.join(".")}: ${i.message}`).join("; ");
      callbacks.onError?.(`equilibrium schema error: ${msg}`);
    }
  });

  es.addEventListener("simulation", (e: MessageEvent) => {
    const parsed = SimulationDataSchema.safeParse(JSON.parse(e.data));
    if (parsed.success) {
      callbacks.onSimulation?.(parsed.data);
    } else {
      const msg = parsed.error.issues.map((i) => `${i.path.join(".")}: ${i.message}`).join("; ");
      callbacks.onError?.(`simulation schema error: ${msg}`);
    }
  });

  es.addEventListener("stream_error", (e: MessageEvent) => {
    try {
      const data = JSON.parse(e.data) as { stage?: string; message?: string };
      callbacks.onError?.(`[${data.stage ?? "unknown"}] ${data.message ?? "stage failed"}`);
    } catch {
      callbacks.onError?.("stream_error: unparseable payload");
    }
  });

  es.addEventListener("done", () => {
    callbacks.onDone?.();
    es.close();
  });

  // Connection-level errors (network failure, CORS, HTTP 4xx/5xx before stream starts).
  es.onerror = () => {
    callbacks.onError?.("SSE connection error");
    es.close();
  };

  return es;
}
