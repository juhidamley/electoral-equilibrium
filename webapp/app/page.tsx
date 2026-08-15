"use client";

import { useEffect, useRef, useState } from "react";
import { Link2 } from "lucide-react";
import CoalitionChart from "@/components/CoalitionChart";
import EditorialWinDial from "@/components/EditorialWinDial";
import EmpiricalSupportPanel from "@/components/EmpiricalSupport";
import ErrorBanner from "@/components/ErrorBanner";
import GroupLoyaltyChart from "@/components/GroupLoyaltyChart";
import ShockInput from "@/components/ShockInput";
import ShockNarrative from "@/components/ShockNarrative";
import WinGauge from "@/components/WinGauge";
import { estimateShockStream } from "@/lib/api";
import { netVerdict } from "@/lib/editorial";
import { injectRulingContext, type RulingParty } from "@/lib/rulingParty";
import type {
  EmpiricalSupport,
  EquilibriumData,
  Party,
  Refinement,
  SimulationData,
} from "@/lib/types";

export default function HomePage() {
  const [party, setParty] = useState<Party>("democrat");
  const [rulingParty, setRulingParty] = useState<RulingParty>("none");
  const [event, setEvent] = useState("");
  const [submittedEvent, setSubmittedEvent] = useState("");
  const [intensity, setIntensity] = useState(1);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [deltaBins, setDeltaBins] = useState<Record<string, string> | null>(null);
  const [deltas, setDeltas] = useState<Record<string, number> | null>(null);
  const [deltasRace, setDeltasRace] = useState<Record<string, number> | null>(null);
  const [deltasReligion, setDeltasReligion] = useState<Record<string, number> | null>(null);
  const [deltasGender, setDeltasGender] = useState<Record<string, number> | null>(null);
  const [deltaEff, setDeltaEff] = useState<number | null>(null);
  const [equilibrium, setEquilibrium] = useState<EquilibriumData | null>(null);
  const [simulation, setSimulation] = useState<SimulationData | null>(null);
  const [empiricalSupport, setEmpiricalSupport] = useState<EmpiricalSupport | null>(null);
  const [refinement, setRefinement] = useState<Refinement | null>(null);
  const [copied, setCopied] = useState(false);
  const esRef = useRef<EventSource | null>(null);

  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const selectedParty = params.get("party");
    if (selectedParty === "democrat" || selectedParty === "republican") setParty(selectedParty);
    const ruling = params.get("ruling");
    if (ruling === "democrat" || ruling === "republican" || ruling === "none") setRulingParty(ruling);
    const sharedEvent = params.get("event");
    if (sharedEvent) setEvent(sharedEvent);
    const rawIntensity = Number(params.get("intensity"));
    if (rawIntensity >= 0.5 && rawIntensity <= 2) setIntensity(rawIntensity);
  }, []);

  useEffect(() => () => esRef.current?.close(), []);

  function clearResults() {
    setDeltaBins(null);
    setDeltas(null);
    setDeltasRace(null);
    setDeltasReligion(null);
    setDeltasGender(null);
    setDeltaEff(null);
    setEquilibrium(null);
    setSimulation(null);
    setEmpiricalSupport(null);
    setRefinement(null);
  }

  function handleSubmit() {
    esRef.current?.close();
    setLoading(true);
    setError(null);
    clearResults();
    setSubmittedEvent(event.trim());
    const eventText = injectRulingContext(event, rulingParty);
    esRef.current = estimateShockStream(eventText, intensity, party, {
      onDeltas: (data) => {
        setDeltaBins({ ...data.delta_bins_race, ...data.delta_bins_religion, ...data.delta_bins_gender });
        setDeltas({ ...data.deltas_race, ...data.deltas_religion, ...data.deltas_gender });
        setDeltasRace(data.deltas_race);
        setDeltasReligion(data.deltas_religion);
        setDeltasGender(data.deltas_gender);
        setDeltaEff(data.delta_eff);
      },
      onEquilibrium: setEquilibrium,
      onSimulation: setSimulation,
      onEmpiricalSupport: setEmpiricalSupport,
      onRefinement: setRefinement,
      onDone: () => {
        setLoading(false);
        esRef.current = null;
      },
      onError: (message) => {
        console.error("SSE stream error", message);
        setError("The estimation service is unavailable. Please try again.");
        setLoading(false);
        esRef.current = null;
      },
    });
  }

  function handleShare() {
    const params = new URLSearchParams({ party, event, intensity: intensity.toFixed(1) });
    if (rulingParty !== "none") params.set("ruling", rulingParty);
    navigator.clipboard.writeText(`${window.location.origin}/?${params}`).then(() => {
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1800);
    });
  }

  const hasResult = deltaEff !== null;

  return (
    <main className="min-h-screen bg-canvas px-4 py-8 text-body sm:py-14">
      <article className="mx-auto max-w-[680px] border border-rule bg-surface shadow-[0_18px_55px_rgba(80,62,30,0.08)]">
        <div className="h-[5px] bg-maroon" />
        <div className="px-5 py-7 sm:px-10 sm:py-10">
          <p className="editorial-kicker mb-8 text-gold">
            Electoral Equilibrium · Scenario Estimate
          </p>

          <ShockInput
            party={party}
            setParty={setParty}
            rulingParty={rulingParty}
            setRulingParty={setRulingParty}
            event={event}
            setEvent={setEvent}
            intensity={intensity}
            setIntensity={setIntensity}
            loading={loading}
            onSubmit={handleSubmit}
          />

          <ErrorBanner message={error} />

          {(loading || hasResult) && (
            <div className="mt-10 border-t border-rule pt-8">
              {loading && !hasResult ? (
                <div className="space-y-4" aria-live="polite">
                  <p className="editorial-kicker animate-pulse text-gold">Reading the political terrain…</p>
                  <div className="h-7 w-3/4 animate-pulse bg-rule-light" />
                  <div className="h-4 w-full animate-pulse bg-rule-light" />
                  <div className="h-4 w-5/6 animate-pulse bg-rule-light" />
                </div>
              ) : (
                <>
                  <section>
                    <p className="editorial-kicker mb-2 text-gold">The event</p>
                    <p className="text-[17px] italic leading-relaxed text-ink">“{submittedEvent}”</p>
                  </section>

                  <section className="mt-8 grid grid-cols-1 gap-7 sm:grid-cols-[1fr_168px] sm:items-center">
                    <div>
                      <h1 className="text-[26px] leading-tight text-ink">
                        {netVerdict(deltaEff ?? 0, party)}
                      </h1>
                      <p className="mt-4 text-[15px] leading-relaxed text-body">
                        The shift is small — the norm. Few voters leave the side they already favor;
                        elections turn on the margins that do.
                      </p>
                    </div>
                    <EditorialWinDial
                      winProbability={simulation?.win_probability ?? null}
                      modeledParty={party}
                      loading={loading && !simulation}
                    />
                  </section>

                  <div className="my-9 h-px bg-rule" />
                  <GroupLoyaltyChart
                    deltasRace={deltasRace}
                    deltasReligion={deltasReligion}
                    deltasGender={deltasGender}
                    party={party}
                    loading={loading}
                  />

                  <p className="mt-9 border-t border-rule-light pt-5 text-[12.5px] italic leading-relaxed text-muted">
                    Each line runs from no change to where the event moves a group. A research estimate
                    of small directional shifts — calibrated to a decade of survey data; best read as
                    direction and rough size, not exact vote counts.
                  </p>

                  <details className="mt-6 border-t border-rule-light pt-4">
                    <summary className="editorial-kicker cursor-pointer text-gold">
                      Methodology &amp; model internals
                    </summary>
                    <div className="mt-5 space-y-6 text-sm">
                      <ShockNarrative deltaBins={deltaBins} deltas={deltas} party={party} />
                      <CoalitionChart
                        baseline={null}
                        shifted={equilibrium?.mu_shifted ?? null}
                        rebalanced={equilibrium?.weights ?? null}
                        deltasRace={deltasRace}
                        deltasReligion={deltasReligion}
                        deltasGender={deltasGender}
                        feasible={equilibrium?.feasible ?? true}
                        targetMet={equilibrium?.target_met ?? null}
                        muEffShifted={equilibrium?.mu_eff_shifted ?? null}
                        target={equilibrium?.target ?? null}
                        party={party}
                        loading={loading}
                      />
                      <EmpiricalSupportPanel empiricalSupport={empiricalSupport} refinement={refinement} />
                      <WinGauge
                        winProbability={simulation?.win_probability ?? null}
                        winProbabilityLow={simulation?.win_probability_low}
                        winProbabilityHigh={simulation?.win_probability_high}
                        percentiles={simulation?.percentiles ?? null}
                        loading={loading && !simulation}
                      />
                    </div>
                  </details>

                  <button
                    type="button"
                    onClick={handleShare}
                    className="mt-6 inline-flex items-center gap-1.5 text-xs italic text-muted hover:text-maroon"
                  >
                    <Link2 className="h-3.5 w-3.5" aria-hidden="true" />
                    {copied ? "Link copied" : "Share this estimate"}
                  </button>
                </>
              )}
            </div>
          )}
        </div>
      </article>
      <p className="mx-auto mt-5 max-w-[680px] text-center text-xs italic text-muted">
        CMC Summer Research Project · Juhi Damley
      </p>
    </main>
  );
}
