"use client";

// Progressive reveal via SSE — three named events populate the UI sequentially:
//   1. "deltas"      → ShockNarrative + opaque CoalitionChart bars (immediate, ~2s)
//   2. "equilibrium" → translucent rebalance overlay added IN PLACE to CoalitionChart
//   3. "simulation"  → WinGauge win probability + CI strip
//   4. "done"        → loading=false, EventSource closed
//
// Rendering contract:
//   • When deltaBins+shifted are set but equilibrium is null, CoalitionChart
//     renders ONLY the opaque baseline bars. No translucent layer yet.
//   • When equilibrium arrives, CoalitionChart adds the translucent layer without
//     remounting — pass it as a separate prop so the chart uses conditional layer
//     rendering, not a key change that would re-animate the opaque bars.
//   • WinGauge stays in a skeleton/loading state until simulation is non-null.

import { useEffect, useRef, useState } from "react";

import type { EquilibriumData, Party, SimulationData } from "@/lib/types";
import CoalitionChart from "@/components/CoalitionChart";
import ErrorBanner from "@/components/ErrorBanner";
import ShockInput from "@/components/ShockInput";
import ShockNarrative from "@/components/ShockNarrative";
import WinGauge from "@/components/WinGauge";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

// Backend guard: event descriptions shorter than 10 chars receive a 422.
// Mirror it here so the user gets immediate feedback instead of a round-trip.

export default function HomePage() {
  const [party, setParty] = useState<Party>("democrat");
  const [event, setEvent] = useState<string>("");
  const [intensity, setIntensity] = useState<number>(1.0);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  // Populated by SSE "deltas" event — enables ShockNarrative and opaque bars.
  const [deltaBins, setDeltaBins] = useState<Record<string, string> | null>(null);
  const [shifted, setShifted] = useState<Record<string, number> | null>(null);

  // Populated by SSE "equilibrium" event — adds translucent rebalance layer.
  const [equilibrium, setEquilibrium] = useState<EquilibriumData | null>(null);

  // Populated by SSE "simulation" event — drives WinGauge.
  const [simulation, setSimulation] = useState<SimulationData | null>(null);

  // Ref holds the active EventSource so handleSubmit can close a stale connection
  // and the unmount cleanup can close any in-flight stream.
  const esRef = useRef<EventSource | null>(null);

  useEffect(() => {
    return () => {
      esRef.current?.close();
    };
  }, []);

  const handleSubmit = () => {
    // Close any in-flight stream before opening a new one.
    esRef.current?.close();

    setLoading(true);
    setError(null);
    setDeltaBins(null);
    setShifted(null);
    setEquilibrium(null);
    setSimulation(null);

    const url = new URL(`${API_URL}/estimate/stream`);
    url.searchParams.set("event", event);
    url.searchParams.set("intensity", String(intensity));
    url.searchParams.set("party", party);

    const es = new EventSource(url.toString());
    esRef.current = es;

    es.addEventListener("deltas", (e) => {
      const data = JSON.parse((e as MessageEvent).data);
      // delta_bins_race is a Record<RaceBloc, DeltaBin>; merge all three
      // strata for ShockNarrative which may display any bloc type.
      setDeltaBins({
        ...data.delta_bins_race,
        ...data.delta_bins_religion,
        ...data.delta_bins_gender,
      });
      setShifted(data.deltas_race);
    });

    es.addEventListener("equilibrium", (e) => {
      const data = JSON.parse((e as MessageEvent).data) as EquilibriumData;
      setEquilibrium(data);
    });

    es.addEventListener("simulation", (e) => {
      const data = JSON.parse((e as MessageEvent).data) as SimulationData;
      setSimulation(data);
    });

    es.addEventListener("done", () => {
      setLoading(false);
      es.close();
      esRef.current = null;
    });

    es.onerror = (rawEvent) => {
      console.error("SSE stream error", rawEvent);
      setError("The estimation service is unavailable. Please try again.");
      setLoading(false);
      es.close();
      esRef.current = null;
    };
  };

  return (
    <main className="mx-auto max-w-6xl px-4 py-8">
      <h1 className="mb-6 text-2xl font-semibold">Electoral Equilibrium</h1>

      <ShockInput
        party={party}
        setParty={setParty}
        event={event}
        setEvent={setEvent}
        intensity={intensity}
        setIntensity={setIntensity}
        loading={loading}
        onSubmit={handleSubmit}
      />

      <ErrorBanner message={error} />

      {/* Results region — shown as soon as loading starts so components can
          display their own skeleton states before their SSE event arrives.
          Gated on (deltaBins || loading) so nothing renders before first submit. */}
      {(deltaBins || loading) && (
        <section className="mt-8 space-y-6">
          {/* ShockNarrative: populates on "deltas" (~2s) — first visible result */}
          <ShockNarrative
            deltaBins={deltaBins}
            party={party}
            loading={loading && !deltaBins}
          />

          {/* Grid: chart (2/3) + gauge (1/3) side-by-side on desktop, stacked on mobile.
              CoalitionChart renders opaque baseline bars on "deltas"; the translucent
              rebalance overlay is added IN PLACE on "equilibrium" via the rebalanced
              prop — no key change, no remount, no re-animation of opaque bars. */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div className="lg:col-span-2">
              {/* baseline=μ (pre-shock loyalty) is not yet in the SSE payload —
                  the deltas event carries Δ (deltas_race), not μ. Until the
                  backend widens the stream to include μ, baseline is null and
                  CoalitionChart renders only the shifted layer. */}
              <CoalitionChart
                baseline={null}
                shifted={shifted}
                rebalanced={equilibrium?.weights ?? null}
                feasible={equilibrium?.feasible ?? true}
                party={party}
                loading={loading && !deltaBins}
              />
            </div>
            <div className="lg:col-span-1">
              {/* WinGauge stays in skeleton state until "simulation" event arrives */}
              <WinGauge simulation={simulation} loading={loading && !simulation} />
            </div>
          </div>
        </section>
      )}
    </main>
  );
}
