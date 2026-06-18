import { estimateShock, getBlocs, healthCheck, ApiError } from "../api";

// A valid EstimateResponse fixture matching the Zod schema exactly.
const RACES = ["african_american", "asian", "latino", "other_race", "white"];
const RELIGIONS = [
  "evangelical",
  "catholic",
  "protestant",
  "secular",
  "jewish",
  "muslim",
  "other_rel",
];
const GENDERS = ["women", "men", "other_gender"];
const rec = (keys: string[], v: unknown) =>
  Object.fromEntries(keys.map((k) => [k, v]));

const validResponse = {
  shock: {
    shock: "test_shock",
    cycle: 2028,
    party: "democrat",
    delta_bins_race: rec(RACES, "neutral"),
    delta_bins_religion: rec(RELIGIONS, "neutral"),
    delta_bins_gender: rec(GENDERS, "neutral"),
    deltas_race: rec(RACES, 0.0),
    deltas_religion: rec(RELIGIONS, 0.0),
    deltas_gender: rec(GENDERS, 0.0),
    delta_eff: 0.02,
    covariance: Array.from({ length: 5 }, (_, i) =>
      Array.from({ length: 5 }, (_, j) => (i === j ? 0.001 : 0.0)),
    ),
    source: "llm_unified",
  },
  equilibrium: {
    method: "cvxpy_dqcp",
    party: "democrat",
    shock: "test_shock",
    weights: rec(RACES, 0.2),
    mu_shifted: rec(RACES, 0.5),
    feasible: true,
    target_met: false,
    target: 0.5066,
  },
  simulation: {
    n_simulations: 10000,
    seed: 42,
    win_probability: 0.62,
    win_probability_low: 0.55,
    win_probability_high: 0.69,
    percentiles: rec(RACES, [0.1, 0.15, 0.2, 0.25, 0.3]),
  },
};

function mockFetchOnce(body: unknown, ok = true, status = 200) {
  global.fetch = jest.fn().mockResolvedValueOnce({
    ok,
    status,
    statusText: ok ? "OK" : "Error",
    json: async () => body,
  }) as unknown as typeof fetch;
}

describe("estimateShock", () => {
  afterEach(() => jest.restoreAllMocks());

  it("calls the correct URL with POST and parses a valid response", async () => {
    mockFetchOnce(validResponse);
    const result = await estimateShock("Test event", 1.0);

    expect(global.fetch).toHaveBeenCalledTimes(1);
    const [url, opts] = (global.fetch as jest.Mock).mock.calls[0];
    expect(url).toContain("/estimate");
    expect(opts.method).toBe("POST");
    expect(result.simulation.win_probability).toBe(0.62);
  });

  it("throws ApiError on a malformed (schema-invalid) response", async () => {
    const bad = JSON.parse(JSON.stringify(validResponse));
    bad.simulation.win_probability = 5; // out of [0, 1]
    mockFetchOnce(bad);
    await expect(estimateShock("Test", 1.0)).rejects.toThrow(ApiError);
  });

  it("throws ApiError with status on a non-OK HTTP response", async () => {
    mockFetchOnce({ detail: "Event description too short" }, false, 422);
    await expect(estimateShock("hi", 1.0)).rejects.toMatchObject({
      name: "ApiError",
      status: 422,
    });
  });
});

describe("healthCheck", () => {
  afterEach(() => jest.restoreAllMocks());

  it("returns true when status is ok", async () => {
    mockFetchOnce({ status: "ok" });
    expect(await healthCheck()).toBe(true);
  });

  it("returns false on network failure", async () => {
    global.fetch = jest
      .fn()
      .mockRejectedValueOnce(new Error("down")) as unknown as typeof fetch;
    expect(await healthCheck()).toBe(false);
  });
});

describe("getBlocs", () => {
  afterEach(() => jest.restoreAllMocks());

  it("flattens race/religion/gender arrays into a single list", async () => {
    mockFetchOnce({
      race: ["african_american", "white"],
      religion: ["evangelical"],
      gender: ["women", "men"],
    });
    const blocs = await getBlocs();
    expect(blocs).toEqual([
      "african_american",
      "white",
      "evangelical",
      "women",
      "men",
    ]);
  });

  it("throws ApiError on non-OK response", async () => {
    mockFetchOnce({}, false, 500);
    await expect(getBlocs()).rejects.toMatchObject({
      name: "ApiError",
      status: 500,
    });
  });
});
