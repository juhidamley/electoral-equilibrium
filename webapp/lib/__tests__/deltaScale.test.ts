import {
  DELTA_AXIS_DOMAIN,
  DELTA_AXIS_MAX,
  DELTA_AXIS_TICKS,
  formatDeltaPP,
} from "../deltaScale";

// Real electoral.core.types.BIN_MIDPOINTS values (Step 2.1 rescale), so this
// test fails loudly if the frontend and backend scales ever drift apart.
const BIN_MIDPOINTS: Record<string, number> = {
  strong_neg: -0.03,
  mod_neg: -0.0175,
  mild_neg: -0.00875,
  slight_neg: -0.003,
  neutral: 0.0,
  slight_pos: 0.003,
  mild_pos: 0.00875,
  mod_pos: 0.0175,
  strong_pos: 0.03,
};

describe("DELTA_AXIS_DOMAIN / DELTA_AXIS_MAX", () => {
  it("matches the backend's post-intensity clip ceiling, not the raw bin ceiling", () => {
    expect(DELTA_AXIS_MAX).toBe(0.0375);
    expect(DELTA_AXIS_DOMAIN).toEqual([-0.0375, 0.0375]);
  });

  it("comfortably contains every bin midpoint", () => {
    for (const v of Object.values(BIN_MIDPOINTS)) {
      expect(Math.abs(v)).toBeLessThanOrEqual(DELTA_AXIS_MAX);
    }
  });

  it("every tick falls within the domain", () => {
    for (const t of DELTA_AXIS_TICKS) {
      expect(t).toBeGreaterThanOrEqual(DELTA_AXIS_DOMAIN[0]);
      expect(t).toBeLessThanOrEqual(DELTA_AXIS_DOMAIN[1]);
    }
  });
});

describe("formatDeltaPP", () => {
  it("keeps every one of the 9 bins textually distinct at one decimal place", () => {
    const labels = Object.values(BIN_MIDPOINTS).map(formatDeltaPP);
    expect(new Set(labels).size).toBe(labels.length);
  });

  it("does not collapse slight_pos/slight_neg into neutral (the bug this module fixes)", () => {
    expect(formatDeltaPP(BIN_MIDPOINTS.slight_pos)).not.toBe(formatDeltaPP(BIN_MIDPOINTS.neutral));
    expect(formatDeltaPP(BIN_MIDPOINTS.slight_neg)).not.toBe(formatDeltaPP(BIN_MIDPOINTS.neutral));
  });

  it("formats neutral without a sign prefix", () => {
    expect(formatDeltaPP(0)).toBe("0.0pp");
  });

  it("formats positive values with an explicit + prefix and negative with -", () => {
    expect(formatDeltaPP(0.03)).toBe("+3.0pp");
    expect(formatDeltaPP(-0.03)).toBe("-3.0pp");
  });

  it("matches the example labels named in the brief", () => {
    expect(formatDeltaPP(-0.03)).toBe("-3.0pp");
    expect(formatDeltaPP(-0.0175)).toBe("-1.8pp"); // toFixed(1) rounding of -1.75
  });
});
