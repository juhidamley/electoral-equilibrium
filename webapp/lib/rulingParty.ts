// ── Ruling-party (governing-context) injection ────────────────────────────────
//
// WHY THIS EXISTS: many shocks are incumbent-relative — a recession, an
// administration scandal, a policy reversal all read differently depending on
// which party holds power. The fine-tuned Stage-1 model was trained on FREE TEXT,
// not on a separate "ruling party" field, so the only way to give it governing
// context is to weave that context INTO the event description itself. This module
// is the single source of that transformation.
//
// It is deliberately frontend-only and backend-agnostic: the string it produces is
// passed to the existing /estimate/stream `event` param unchanged. No schema or
// endpoint change is required.

export type RulingParty = "democrat" | "republican" | "none";

// The clause appended to the event text for each governing party. Phrased as a
// natural adjunct so it reads as ordinary prose to the model, e.g.
//   "A sudden recession is declared" → "... under the current Republican administration".
const RULING_CLAUSE: Record<Exclude<RulingParty, "none">, string> = {
  democrat: "under the current Democratic administration",
  republican: "under the current Republican administration",
};

// If the event text already establishes governing context, we must NOT append a
// second, possibly contradictory clause. This matches the common ways a writer
// signals "who is in power": an explicit administration/White House reference, an
// incumbent/sitting-government reference, or a party "in power" phrasing. Kept
// deliberately conservative to avoid suppressing useful injection on unrelated text
// (e.g. "left the city without power" does not match "\bin power\b").
const GOVERNING_CONTEXT_RE =
  /\b(administration|white house|incumbent|sitting (?:president|government)|in power|ruling party|party in charge|current government)\b/i;

/**
 * Weave ruling-party context into a free-text event description.
 *
 * Rules (see task spec):
 *   1. ruling === "none"  → return the text UNCHANGED (preserves current behavior).
 *   2. text already names an administration / governing context → return UNCHANGED
 *      (don't double it).
 *   3. otherwise → append the governing-context clause, inserting it BEFORE any
 *      trailing sentence punctuation so the result stays grammatical:
 *        "A recession hits."  → "A recession hits under the current Republican administration."
 *        "A recession hits"   → "A recession hits under the current Republican administration"
 *
 * Empty/whitespace-only input is returned unchanged so the caller's ≥10-char guard
 * still governs submission.
 */
export function injectRulingContext(event: string, ruling: RulingParty): string {
  const trimmed = event.trim();
  if (ruling === "none" || trimmed.length === 0) return event;
  if (GOVERNING_CONTEXT_RE.test(trimmed)) return event;

  const clause = RULING_CLAUSE[ruling];
  const trailing = /[.!?]+$/.exec(trimmed);
  if (trailing) {
    const body = trimmed.slice(0, trimmed.length - trailing[0].length).trimEnd();
    return `${body} ${clause}${trailing[0]}`;
  }
  return `${trimmed} ${clause}`;
}

/**
 * True when calling injectRulingContext would actually change the text — used by the
 * UI to show a "sent as" preview only when an injection will occur.
 */
export function willInjectRulingContext(event: string, ruling: RulingParty): boolean {
  return injectRulingContext(event, ruling) !== event;
}
