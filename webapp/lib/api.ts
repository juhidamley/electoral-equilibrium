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

import { EstimateResponseSchema, type EstimateResponse } from "./schemas";

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
