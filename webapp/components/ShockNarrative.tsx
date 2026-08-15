"use client";

// ============================================================================
// ShockNarrative — turns the model's per-bloc delta bins into plain English.
// ============================================================================
// This is the FIRST result the user sees (it populates on the "deltas" SSE event,
// before the charts). The model outputs cryptic labels like {"evangelical":
// "strong_neg", "latino": "mild_neg", ...}; a normal person can't read that. This
// component translates them into two sentences:
//   "This shock is predicted to HURT the Democratic coalition with: Evangelicals
//    (significant), Catholics (mild). And HELP with: Latino voters (moderate)…"
//
// HOW: map each bin → a direction (hurt/help/neutral) + a magnitude word, split
// the blocs into a "hurt" list and a "help" list, sort each by severity, and
// render. If everything is neutral, show a "minimal impact" message instead.
// It's a pure presentational component: props in → text out, no state.

import type { Party } from "@/lib/types";
import { BLOC_LABEL } from "@/lib/blocs";

interface ShockNarrativeProps {
  deltaBins: Record<string, string> | null;
  deltas?: Record<string, number> | null; // signed per-bloc deltas (for the neutral band)
  party: Party;
  loading?: boolean;
}

// A bloc reads as "no measurable effect" when its bin is the neutral token OR its
// signed delta is within ±NEUTRAL_THRESHOLD of zero. The latter catches a slight_*
// bin whose actual delta has shrunk below the noise floor (e.g. at low intensity),
// which would otherwise be reported as a real hurt/help.
const NEUTRAL_THRESHOLD = 0.006;

// ── Bin → direction + magnitude ───────────────────────────────────────────────

type Dir = "hurt" | "help" | "neutral";

const BIN_PHRASE: Record<string, { dir: Dir; mag: string }> = {
  strong_neg: { dir: "hurt",    mag: "significant" },
  mod_neg:    { dir: "hurt",    mag: "moderate"    },
  mild_neg:   { dir: "hurt",    mag: "mild"        },
  slight_neg: { dir: "hurt",    mag: "slight"      },
  neutral:    { dir: "neutral", mag: ""             },
  slight_pos: { dir: "help",    mag: "slight"       },
  mild_pos:   { dir: "help",    mag: "mild"         },
  mod_pos:    { dir: "help",    mag: "moderate"     },
  strong_pos: { dir: "help",    mag: "significant"  },
};

// Sort order: significant > moderate > mild > slight
const MAG_RANK: Record<string, number> = {
  significant: 0,
  moderate:    1,
  mild:        2,
  slight:      3,
};

// BLOC_LABEL imported from lib/blocs.ts — single source of truth shared with CoalitionChart.

const PARTY_LABEL: Record<Party, string> = {
  democrat:    "Democratic",
  republican:  "Republican",
};

// ── Helpers ───────────────────────────────────────────────────────────────────

interface BlocEntry { label: string; mag: string }

function sortBySeverity(entries: BlocEntry[]): BlocEntry[] {
  return [...entries].sort(
    (a, b) => (MAG_RANK[a.mag] ?? 99) - (MAG_RANK[b.mag] ?? 99),
  );
}

function formatList(entries: BlocEntry[]): React.ReactNode[] {
  return entries.map((e, i) => (
    <span key={e.label}>
      {i > 0 && ", "}
      {e.label}
      {e.mag && <span className="text-gray-500"> ({e.mag})</span>}
    </span>
  ));
}

// ── Component ─────────────────────────────────────────────────────────────────

import React from "react";

export default function ShockNarrative({
  deltaBins,
  deltas,
  party,
  loading,
}: ShockNarrativeProps) {
  // Loading state — no bins yet.
  if (loading && deltaBins === null) {
    return (
      <p className="text-sm italic text-gray-400 animate-pulse">
        Analyzing event…
      </p>
    );
  }

  // Nothing to show before first submit.
  if (deltaBins === null) return null;

  // Partition into hurt / help / neutral. A bloc is neutral when its bin is the
  // neutral token OR its signed delta is sub-threshold; neutral blocs must NOT
  // appear in hurt or help. Unknown tokens are skipped entirely.
  const hurt: BlocEntry[] = [];
  const help: BlocEntry[] = [];
  const neutral: BlocEntry[] = [];

  for (const [bloc, bin] of Object.entries(deltaBins)) {
    const phrase = BIN_PHRASE[bin]; // undefined if unexpected token → skip
    if (!phrase) continue;
    const label = BLOC_LABEL[bloc] ?? bloc; // raw id as fallback
    const d = deltas?.[bloc];
    const subThreshold = typeof d === "number" && Math.abs(d) < NEUTRAL_THRESHOLD;

    if (phrase.dir === "neutral" || subThreshold) {
      neutral.push({ label, mag: "" });
      continue;
    }
    if (phrase.dir === "hurt") hurt.push({ label, mag: phrase.mag });
    else help.push({ label, mag: phrase.mag });
  }

  // All-neutral case (nothing landed in hurt or help).
  if (hurt.length === 0 && help.length === 0) {
    return (
      <p className="text-sm text-gray-600">
        The model predicts minimal coalition impact from this event.
      </p>
    );
  }

  const partyLabel = PARTY_LABEL[party];
  const sortedHurt = sortBySeverity(hurt);
  const sortedHelp = sortBySeverity(help);

  return (
    <div className="space-y-1.5 text-sm leading-relaxed">
      {sortedHurt.length > 0 && (
        <p className="text-red-800">
          This shock is predicted to{" "}
          <strong>hurt</strong> the {partyLabel} coalition&apos;s standing
          with: {formatList(sortedHurt)}.
        </p>
      )}
      {sortedHelp.length > 0 && (
        <p className="text-green-800">
          {sortedHurt.length > 0 ? "And " : "This shock is predicted to "}
          <strong>help</strong> with: {formatList(sortedHelp)}.
        </p>
      )}
      {neutral.length > 0 && (
        <p className="text-gray-600">
          {sortedHurt.length > 0 || sortedHelp.length > 0 ? "And " : ""}
          <strong>little measurable effect</strong> on: {formatList(neutral)}.
        </p>
      )}
    </div>
  );
}
