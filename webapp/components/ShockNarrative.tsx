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
  party: Party;
  loading?: boolean;
}

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

  // Partition into hurt / help, skipping neutral and unknown tokens.
  const hurt: BlocEntry[] = [];
  const help: BlocEntry[] = [];

  for (const [bloc, bin] of Object.entries(deltaBins)) {
    const phrase = BIN_PHRASE[bin]; // undefined if unexpected token → skip
    if (!phrase || phrase.dir === "neutral") continue;
    const entry: BlocEntry = {
      label: BLOC_LABEL[bloc] ?? bloc, // raw id as fallback
      mag: phrase.mag,
    };
    if (phrase.dir === "hurt") hurt.push(entry);
    else help.push(entry);
  }

  // All-neutral case.
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
    </div>
  );
}
