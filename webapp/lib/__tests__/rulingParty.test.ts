import { injectRulingContext, willInjectRulingContext } from "../rulingParty";

describe("injectRulingContext", () => {
  it("returns text unchanged when ruling party is 'none' (default behavior)", () => {
    const e = "A sudden recession is declared in the final month of the campaign";
    expect(injectRulingContext(e, "none")).toBe(e);
  });

  it("appends the Republican clause when Republican holds power", () => {
    expect(
      injectRulingContext("A sudden recession is declared", "republican"),
    ).toBe("A sudden recession is declared under the current Republican administration");
  });

  it("appends the Democratic clause when Democrats hold power", () => {
    expect(
      injectRulingContext("A sudden recession is declared", "democrat"),
    ).toBe("A sudden recession is declared under the current Democratic administration");
  });

  it("inserts the clause before trailing sentence punctuation", () => {
    expect(injectRulingContext("A recession hits.", "republican")).toBe(
      "A recession hits under the current Republican administration.",
    );
    expect(injectRulingContext("Is a recession coming?", "democrat")).toBe(
      "Is a recession coming under the current Democratic administration?",
    );
  });

  it("does NOT double up when the text already names an administration", () => {
    const e = "A major financial scandal involving the sitting administration is revealed";
    expect(injectRulingContext(e, "republican")).toBe(e);
  });

  it.each([
    "The incumbent faces a corruption probe",
    "Turmoil at the White House over a leaked memo",
    "The party in power passes a sweeping tax cut",
    "The current government collapses amid protests",
  ])("skips injection when governing context is already present: %s", (e) => {
    expect(injectRulingContext(e, "democrat")).toBe(e);
  });

  it("does not false-match unrelated uses of 'power'", () => {
    const e = "A storm leaves the city without power for a week";
    expect(injectRulingContext(e, "republican")).toBe(
      "A storm leaves the city without power for a week under the current Republican administration",
    );
  });

  it("returns whitespace-only input unchanged", () => {
    expect(injectRulingContext("   ", "republican")).toBe("   ");
  });

  it("willInjectRulingContext reflects whether a change occurs", () => {
    expect(willInjectRulingContext("A recession hits", "republican")).toBe(true);
    expect(willInjectRulingContext("A recession hits", "none")).toBe(false);
    expect(
      willInjectRulingContext("Scandal at the White House", "democrat"),
    ).toBe(false);
  });
});
