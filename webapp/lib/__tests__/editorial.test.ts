import {
  EDITORIAL_BLOC_LABELS,
  EDITORIAL_SECTIONS,
  netVerdict,
  shiftWords,
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
