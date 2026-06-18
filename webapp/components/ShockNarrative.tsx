"use client";

import type { Party } from "@/lib/types";

interface ShockNarrativeProps {
  deltaBins: Record<string, string> | null;
  party: Party;
  loading: boolean;
}

// ── Token → phrase mapping ────────────────────────────────────────────────────

type Direction = "help" | "hurt" | "neutral";

const BIN_META: Record<string, { direction: Direction; magnitude: string }> = {
  strong_neg: { direction: "hurt",    magnitude: "significant" },
  mod_neg:    { direction: "hurt",    magnitude: "moderate"    },
  mild_neg:   { direction: "hurt",    magnitude: "mild"        },
  slight_neg: { direction: "hurt",    magnitude: "slight"      },
  neutral:    { direction: "neutral", magnitude: ""            },
  slight_pos: { direction: "help",    magnitude: "slight"      },
  mild_pos:   { direction: "help",    magnitude: "mild"        },
  mod_pos:    { direction: "help",    magnitude: "moderate"    },
  strong_pos: { direction: "help",    magnitude: "significant" },
};

// ── Bloc → readable label ─────────────────────────────────────────────────────

const BLOC_LABELS: Record<string, string> = {
  african_american: "Black voters",
  asian:            "Asian voters",
  latino:           "Latino voters",
  other_race:       "other racial groups",
  white:            "white voters",
  evangelical:      "Evangelicals",
  catholic:         "Catholics",
  protestant:       "Protestants",
  secular:          "secular voters",
  jewish:           "Jewish voters",
  muslim:           "Muslim voters",
  other_rel:        "other religious groups",
  women:            "women",
  men:              "men",
  other_gender:     "other gender groups",
};

function label(blocId: string): string {
  return BLOC_LABELS[blocId] ?? blocId.replace(/_/g, " ");
}

// ── Component ─────────────────────────────────────────────────────────────────

export default function ShockNarrative({ deltaBins, party, loading }: ShockNarrativeProps) {
  if (loading && !deltaBins) {
    return (
      <div className="space-y-2 animate-pulse">
        <div className="h-4 w-3/4 rounded bg-gray-200" />
        <div className="h-4 w-1/2 rounded bg-gray-200" />
      </div>
    );
  }

  if (!deltaBins) return null;

  // Group blocs by direction, collecting (label, magnitude) pairs.
  const hurt: { name: string; magnitude: string }[] = [];
  const help: { name: string; magnitude: string }[] = [];

  for (const [bloc, bin] of Object.entries(deltaBins)) {
    const meta = BIN_META[bin];
    if (!meta || meta.direction === "neutral") continue;
    const entry = { name: label(bloc), magnitude: meta.magnitude };
    if (meta.direction === "hurt") hurt.push(entry);
    else help.push(entry);
  }

  const partyLabel = party === "democrat" ? "Democratic" : "Republican";

  if (hurt.length === 0 && help.length === 0) {
    return (
      <p className="text-sm text-gray-600">
        The model predicts minimal coalition impact from this event.
      </p>
    );
  }

  return (
    <div className="space-y-1.5 text-sm text-gray-700">
      {hurt.length > 0 && (
        <p>
          This shock is predicted to{" "}
          <strong className="text-red-700">hurt</strong> the {partyLabel}{" "}
          coalition&apos;s standing with:{" "}
          {hurt.map((e, i) => (
            <span key={e.name}>
              {i > 0 && ", "}
              {e.name}
              {e.magnitude && (
                <span className="text-gray-500"> ({e.magnitude})</span>
              )}
            </span>
          ))}
          .
        </p>
      )}
      {help.length > 0 && (
        <p>
          {hurt.length > 0 ? "And " : "This shock is predicted to "}
          <strong className="text-green-700">help</strong> with:{" "}
          {help.map((e, i) => (
            <span key={e.name}>
              {i > 0 && ", "}
              {e.name}
              {e.magnitude && (
                <span className="text-gray-500"> ({e.magnitude})</span>
              )}
            </span>
          ))}
          .
        </p>
      )}
    </div>
  );
}
