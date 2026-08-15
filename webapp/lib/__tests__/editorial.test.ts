import {
  EDITORIAL_BLOC_LABELS,
  EDITORIAL_SECTIONS,
  netVerdict,
  netVerdictCopy,
  probabilityRead,
  shiftWords,
  toDemocraticDirection,
} from "../editorial";

describe("editorial bloc display map", () => {
  it("contains all 14 displayed blocs and excludes other_gender", () => {
    const displayed = EDITORIAL_SECTIONS.flatMap((section) => [...section.blocs]);
    expect(displayed).toHaveLength(14);
    expect(new Set(displayed).size).toBe(14);
    expect(displayed).not.toContain("other_gender");
    for (const bloc of displayed) expect(EDITORIAL_BLOC_LABELS[bloc]).toBeTruthy();
  });
});

describe("net verdict boundaries", () => {
  test.each([
    [0, "Little changes."],
    [0.000999, "Little changes."],
    [0.001, "A slight edge to the Democrats."],
    [0.004999, "A slight edge to the Democrats."],
    [0.005, "An edge to the Democrats."],
    [0.014999, "An edge to the Democrats."],
    [0.015, "A clear shift toward the Democrats."],
    [-0.005, "An edge to the Republicans."],
  ])("maps %s at the named threshold", (delta, expected) => {
    expect(netVerdict(delta as number, "democrat")).toBe(expected);
  });

  it("maps a positive net on the Democratic scale to Democrats", () => {
    expect(toDemocraticDirection(0.005, "democrat")).toBeGreaterThan(0);
    expect(netVerdict(0.005, "democrat")).toContain("Democrats");
  });

  it("keeps verdict and context coherent across every bin", () => {
    const small = [0, 0.001, -0.004999];
    const large = [0.005, -0.014999, 0.015, -0.0375];
    for (const delta of small) {
      expect(netVerdictCopy(delta, "democrat").context).toContain("shift is small");
    }
    for (const delta of large) {
      expect(netVerdictCopy(delta, "democrat").context).toContain("comparatively large movement");
      expect(netVerdictCopy(delta, "democrat").context).not.toContain("shift is small");
    }
  });
});

describe("win-dial coherence", () => {
  it("acknowledges a shift toward the non-leading party with exactly one period", () => {
    const line = probabilityRead(0.7, "democrat", "republican");
    expect(line).toBe("Not enough to threaten the Democrats' lead");
    expect(`${line}.`).toBe("Not enough to threaten the Democrats' lead.");
  });

  it("says a shift toward the leader widens that lead", () => {
    expect(probabilityRead(0.7, "democrat", "democrat")).toBe("And it widens the Democrats' lead");
  });
});

describe("per-bloc wording boundaries", () => {
  test.each([
    [0, "barely moves"],
    [0.000999, "barely moves"],
    [0.001, "nudges Dem."],
    [0.008749, "nudges Dem."],
    [0.00875, "shifts Dem."],
    [0.017499, "shifts Dem."],
    [0.0175, "moves Dem. sharply"],
    [-0.00875, "shifts Rep."],
  ])("maps %s at the canonical boundary", (delta, expected) => {
    expect(shiftWords(delta as number, "democrat").text).toBe(expected);
  });
});
