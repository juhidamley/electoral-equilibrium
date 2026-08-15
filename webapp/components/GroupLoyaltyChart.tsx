import type { Party } from "@/lib/types";
import {
  EDITORIAL_BLOC_LABELS,
  EDITORIAL_SECTIONS,
  shiftWords,
  toDemocraticDirection,
  toDemocraticLoyalty,
} from "@/lib/editorial";
import { DELTA_AXIS_MAX, DELTA_AXIS_TICKS, formatDeltaPP } from "@/lib/deltaScale";

interface Props {
  deltasRace: Record<string, number> | null;
  deltasReligion: Record<string, number> | null;
  deltasGender: Record<string, number> | null;
  // Race-only post-shock modeled-party loyalty levels from equilibrium.mu_shifted.
  // Religion and gender have no corresponding levels in the current API contract.
  shiftedRace: Record<string, number> | null;
  party: Party;
  loading?: boolean;
}

const PARTY_CLASS: Record<Party, string> = {
  democrat: "text-party-dem",
  republican: "text-party-rep",
};

function percent(value: number): number {
  return Math.max(0, Math.min(100, value * 100));
}

export default function GroupLoyaltyChart({
  deltasRace,
  deltasReligion,
  deltasGender,
  shiftedRace,
  party,
  loading,
}: Props) {
  const values: Record<string, number> = {
    ...(deltasRace ?? {}),
    ...(deltasReligion ?? {}),
    ...(deltasGender ?? {}),
  };

  if (loading && !deltasRace) {
    return (
      <section aria-label="Group loyalty loading" className="space-y-4">
        <h2 className="editorial-kicker">Group loyalty</h2>
        {Array.from({ length: 6 }).map((_, index) => (
          <div key={index} className="h-4 animate-pulse rounded bg-rule-light" />
        ))}
      </section>
    );
  }
  if (!deltasRace) return null;

  return (
    <section>
      <div className="mb-5 flex flex-wrap items-end justify-between gap-2">
        <div>
          <h2 className="editorial-kicker">Group loyalty, before and after</h2>
          <p className="mt-1 text-xs italic text-muted">
            <span aria-hidden="true">○</span> before &nbsp; <span aria-hidden="true">●</span> after (race); other groups show the available change estimate.
          </p>
        </div>
      </div>

      {EDITORIAL_SECTIONS.map((section) => {
        const hasLevels = section.title === "By race" && shiftedRace !== null;
        return (
          <div key={section.title} className="mb-7">
            <div className="mb-3 flex items-center gap-3">
              <h3 className="editorial-kicker whitespace-nowrap text-gold">{section.title}</h3>
              <div className="h-px flex-1 bg-rule-light" />
            </div>
            {!hasLevels && (
              <p className="mb-3 text-xs italic text-muted">
                Baseline loyalty levels are not returned for this group; tracks show the available shift from no change.
              </p>
            )}
            <div className="space-y-4">
              {section.blocs.map((bloc) => {
                const rawDelta = values[bloc] ?? 0;
                const democraticDelta = toDemocraticDirection(rawDelta, party);
                const shifted = shiftedRace?.[bloc];
                const rawBaseline = shifted === undefined ? null : shifted - rawDelta;
                // The server constructs mu_shifted = baseline + deltas_race. Recover
                // baseline before converting party-relative levels to the shared axis.
                // A clipped endpoint cannot support that reconstruction honestly.
                const hasRaceLevel = hasLevels && rawBaseline !== null && rawBaseline >= 0 && rawBaseline <= 1;
                const after = hasRaceLevel ? toDemocraticLoyalty(shifted!, party) : null;
                const before = hasRaceLevel
                  ? toDemocraticLoyalty(rawBaseline!, party)
                  : null;
                const position = 50 + (democraticDelta / DELTA_AXIS_MAX) * 50;
                const left = Math.min(50, position);
                const width = Math.abs(position - 50);
                const words = shiftWords(rawDelta, party);
                const verdictClass = words.party ? PARTY_CLASS[words.party] : "text-muted";
                const aria = hasRaceLevel
                  ? `${EDITORIAL_BLOC_LABELS[bloc]}: Democratic loyalty moves from ${(before! * 100).toFixed(1)} to ${(after! * 100).toFixed(1)} percent; ${words.text}, ${formatDeltaPP(rawDelta)}.`
                  : `${EDITORIAL_BLOC_LABELS[bloc]}: ${words.text}, ${formatDeltaPP(rawDelta)} toward the modeled party.`;

                return (
                  <div
                    key={bloc}
                    className="grid grid-cols-1 gap-1.5 sm:grid-cols-[150px_minmax(180px,1fr)_128px] sm:items-center sm:gap-4"
                  >
                    <div className="text-[14.5px] text-ink">{EDITORIAL_BLOC_LABELS[bloc]}</div>
                    <div className="relative h-5" role="img" aria-label={aria}>
                      <div className="absolute left-0 right-0 top-1/2 h-px bg-[var(--dial-track)]" />
                      <div className="absolute bottom-0 left-1/2 top-0 w-px bg-control" />
                      {hasRaceLevel ? (
                        <>
                          <div
                            className={`absolute top-[9px] h-0.5 ${democraticDelta >= 0 ? "bg-party-dem" : "bg-party-rep"}`}
                            style={{ left: `${Math.min(percent(before!), percent(after!))}%`, width: `${Math.abs(percent(after!) - percent(before!))}%` }}
                          />
                          <div
                            className={`absolute top-[6px] h-2 w-2 -translate-x-1/2 rounded-full border ${democraticDelta >= 0 ? "border-party-dem" : "border-party-rep"} bg-surface`}
                            style={{ left: `${percent(before!)}%` }}
                          />
                          <div
                            className={`absolute top-[5px] h-2.5 w-2.5 -translate-x-1/2 rounded-full ${democraticDelta >= 0 ? "bg-party-dem" : "bg-party-rep"}`}
                            style={{ left: `${percent(after!)}%` }}
                          />
                        </>
                      ) : (
                        <>
                          <div
                            className={`absolute top-[9px] h-0.5 ${democraticDelta >= 0 ? "bg-party-dem" : "bg-party-rep"}`}
                            style={{ left: `${left}%`, width: `${width}%` }}
                          />
                          <div
                            className={`absolute top-[5px] h-2.5 w-2.5 -translate-x-1/2 rounded-full ${democraticDelta >= 0 ? "bg-party-dem" : "bg-party-rep"}`}
                            style={{ left: `${Math.max(0, Math.min(100, position))}%` }}
                          />
                        </>
                      )}
                    </div>
                    <div className={`text-[13px] italic sm:text-right ${verdictClass}`}>
                      {words.text} <span className="opacity-65">{formatDeltaPP(rawDelta).replace("pp", "")}</span>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        );
      })}

      <div className="grid grid-cols-1 sm:grid-cols-[150px_minmax(180px,1fr)_128px] sm:gap-4">
        <div />
        <div>
          <div className="flex justify-between text-[11px] italic">
            <span className="text-party-rep">all Republican</span>
            <span className="text-muted">50/50</span>
            <span className="text-party-dem">all Democrat</span>
          </div>
          <div className="mt-1 flex justify-between text-[10px] text-faint">
            <span>{formatDeltaPP(-DELTA_AXIS_MAX)}</span>
            {DELTA_AXIS_TICKS.filter((tick) => tick === 0).map((tick) => <span key={tick}>no change</span>)}
            <span>{formatDeltaPP(DELTA_AXIS_MAX)}</span>
          </div>
        </div>
        <div />
      </div>
    </section>
  );
}
