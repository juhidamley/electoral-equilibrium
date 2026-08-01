"use client";

// Progressive reveal via SSE — three named events populate the UI sequentially:
//   1. "deltas"      → ShockNarrative (delta bins for all strata)
//   2. "equilibrium" → CoalitionChart (mu_shifted opaque bars + weights translucent overlay,
//                       target_met, mu_eff_shifted for gap display)
//   3. "simulation"  → WinGauge win probability + CI strip
//   4. "done"        → loading=false, EventSource closed
//
// Layout:
//   • Sticky header — project name, description, methodology link
//   • Two-column on desktop (lg+): left sidebar = inputs, right = results
//   • Single column on mobile (< lg)
//   • Persistent footer disclaimer

import { useEffect, useRef, useState } from "react";
import * as Collapsible from "@radix-ui/react-collapsible";
import { ChevronDown, LayoutDashboard, Link2 } from "lucide-react";

import type {
  EmpiricalSupport,
  EquilibriumData,
  Party,
  Refinement,
  SimulationData,
} from "@/lib/types";
import { injectRulingContext, type RulingParty } from "@/lib/rulingParty";
import { estimateShockStream } from "@/lib/api";
import CoalitionChart from "@/components/CoalitionChart";
import EmpiricalSupportPanel from "@/components/EmpiricalSupport";
import ErrorBanner from "@/components/ErrorBanner";
import ShockInput from "@/components/ShockInput";
import ShockNarrative from "@/components/ShockNarrative";
import WinGauge from "@/components/WinGauge";

// ── "How it works" collapsible ────────────────────────────────────────────────

function HowItWorks() {
  const [open, setOpen] = useState(false);
  return (
    <Collapsible.Root
      open={open}
      onOpenChange={setOpen}
      className="overflow-hidden rounded-md border border-gray-200 bg-white"
    >
      <Collapsible.Trigger className="flex w-full items-center justify-between px-4 py-3 text-left text-sm font-medium text-gray-700 hover:bg-gray-50 focus:outline-none">
        How it works
        <ChevronDown
          className={`h-4 w-4 flex-none text-gray-400 transition-transform duration-200 ${
            open ? "rotate-180" : ""
          }`}
        />
      </Collapsible.Trigger>
      <Collapsible.Content>
        <ol className="space-y-2.5 px-4 pb-4 pt-1 text-sm text-gray-600">
          <li>
            <span className="font-medium text-gray-800">1. Shock → delta bins.</span>{" "}
            A fine-tuned Mistral model predicts how each demographic group's party loyalty
            shifts (9-bin ordinal) after the hypothetical event.
          </li>
          <li>
            <span className="font-medium text-gray-800">2. Equilibrium coalition.</span>{" "}
            A CVXPY DQCP solver finds the party&apos;s stable strategic coalition weighting
            — given baseline bloc loyalties and demographic constraints — that the shock is
            then evaluated against. This weighting is the same regardless of the shock; step
            3 measures how the shock plays out against it.
          </li>
          <li>
            <span className="font-medium text-gray-800">3. Simulation.</span>{" "}
            10,000 Logistic-Normal ILR Monte Carlo draws propagate covariance uncertainty
            into a 90% win-probability confidence interval.
          </li>
        </ol>
      </Collapsible.Content>
    </Collapsible.Root>
  );
}

// ── "What do these mean?" explainer ───────────────────────────────────────────
// Plain-language definitions for the two charts + the gauge. The closing caveat
// is the honest disclaimer that coalition emphasis is NOT population share.

function WhatDoTheseMean() {
  const [open, setOpen] = useState(false);
  return (
    <Collapsible.Root
      open={open}
      onOpenChange={setOpen}
      className="overflow-hidden rounded-md border border-gray-200 bg-white"
    >
      <Collapsible.Trigger className="flex w-full items-center justify-between px-4 py-3 text-left text-sm font-medium text-gray-700 hover:bg-gray-50 focus:outline-none">
        What do these mean?
        <ChevronDown
          className={`h-4 w-4 flex-none text-gray-400 transition-transform duration-200 ${
            open ? "rotate-180" : ""
          }`}
        />
      </Collapsible.Trigger>
      <Collapsible.Content>
        <dl className="space-y-2.5 px-4 pb-2 pt-1 text-sm text-gray-600">
          <div>
            <dt className="inline font-medium text-gray-800">Loyalty shift (μ̃).</dt>{" "}
            <dd className="inline">
              How much each bloc&apos;s support for the selected party moves after
              this hypothetical shock.
            </dd>
          </div>
          <div>
            <dt className="inline font-medium text-gray-800">Equilibrium coalition weighting (w̃).</dt>{" "}
            <dd className="inline">
              How heavily the party&apos;s most defensible strategy leans on each bloc,
              given baseline loyalties. This does not change with the shock — the
              shock&apos;s effect shows up in the loyalty shifts and win probability
              instead. Strategic weighting, NOT each bloc&apos;s share of the population
              or electorate.
            </dd>
          </div>
          <div>
            <dt className="inline font-medium text-gray-800">Equilibrium status.</dt>{" "}
            <dd className="inline">
              Whether a weighting exists that keeps effective loyalty above the
              model&apos;s win threshold.
            </dd>
          </div>
          <div>
            <dt className="inline font-medium text-gray-800">Win probability.</dt>{" "}
            <dd className="inline">
              The model&apos;s simulated probability of clearing the threshold
              under this hypothetical — a conditional estimate, not a real-world
              forecast.
            </dd>
          </div>
        </dl>
        <p className="px-4 pb-4 pt-1 text-xs leading-snug text-gray-400">
          Coalition emphasis reflects the optimizer&apos;s math, bounded to a
          demographically plausible range (0.5×–1.5× each bloc&apos;s actual
          electorate share). It is the same for every shock — see
          &quot;Equilibrium coalition weighting&quot; above.
        </p>
      </Collapsible.Content>
    </Collapsible.Root>
  );
}

// ── Page ──────────────────────────────────────────────────────────────────────
//
// This is the top-level page component and the frontend's "state machine". A few
// React fundamentals to read it:
//   • useState(initial) → returns [value, setValue]. Calling setValue re-renders
//     the component with the new value. Each piece of changing data is its own
//     state variable below.
//   • The KEY IDEA here (progressive reveal): the three result states start null
//     and get filled one at a time as SSE events arrive (deltas → equilibrium →
//     simulation). Each setX triggers a re-render, so each chart "pops in" the
//     moment its data lands — the user isn't staring at a blank screen.
//   • useRef(initial) → a mutable "box" (.current) that survives re-renders but
//     does NOT trigger one when changed. We use it to hold the live EventSource
//     so we can close it later (a connection isn't UI, so it shouldn't be state).
//   • useEffect(fn, []) → run fn once after mount; its returned function runs on
//     unmount (cleanup). Used below to close the stream when the user leaves.

export default function HomePage() {
  // ── Form inputs (driven by the ShockInput controls) ──
  const [party, setParty] = useState<Party>("democrat");
  // Governing context — distinct from `party` (the modeled coalition). "none" is the
  // default and leaves the event text untouched. See lib/rulingParty.ts.
  const [rulingParty, setRulingParty] = useState<RulingParty>("none");
  const [event, setEvent] = useState<string>("");
  const [intensity, setIntensity] = useState<number>(1.0);
  // ── Request lifecycle ──
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  // ── Results — each filled by its own SSE event, null until then ──
  const [deltaBins, setDeltaBins] = useState<Record<string, string> | null>(null);       // ← "deltas" (bins)
  const [deltas, setDeltas] = useState<Record<string, number> | null>(null);             // ← "deltas" (numeric, all 15 blocs merged)
  const [deltasRace, setDeltasRace] = useState<Record<string, number> | null>(null);         // ← "deltas" race stratum
  const [deltasReligion, setDeltasReligion] = useState<Record<string, number> | null>(null); // ← "deltas" religion stratum
  const [deltasGender, setDeltasGender] = useState<Record<string, number> | null>(null);     // ← "deltas" gender stratum
  const [equilibrium, setEquilibrium] = useState<EquilibriumData | null>(null);          // ← "equilibrium"
  const [simulation, setSimulation] = useState<SimulationData | null>(null);              // ← "simulation"
  const [empiricalSupport, setEmpiricalSupport] = useState<EmpiricalSupport | null>(null); // ← "empirical_support"
  const [refinement, setRefinement] = useState<Refinement | null>(null);                  // ← "refinement"
  const [copied, setCopied] = useState(false);  // transient "Copied!" feedback on the share button

  // Holds the active SSE connection so we can close a stale one before starting a
  // new request and on unmount. A ref (not state) because it's plumbing, not UI.
  const esRef = useRef<EventSource | null>(null);

  // Pre-fill form from shared URL params (once on mount, no auto-submit).
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);

    const p = params.get("party");
    if (p === "democrat" || p === "republican") setParty(p);

    const r = params.get("ruling");
    if (r === "democrat" || r === "republican" || r === "none") setRulingParty(r);

    const e = params.get("event");
    if (e) setEvent(e);

    const iRaw = params.get("intensity");
    if (iRaw !== null) {
      const n = Number(iRaw);
      if (!Number.isNaN(n) && n >= 0.5 && n <= 2.0) setIntensity(n);
    }
  }, []);

  // Close in-flight EventSource on unmount.
  useEffect(() => {
    return () => {
      esRef.current?.close();
    };
  }, []);

  const handleShare = () => {
    // Share the RAW event + ruling party (not the injected text) so the recipient's
    // page reconstructs an identical request, injection included.
    const params = new URLSearchParams({
      party,
      event,
      intensity: String(intensity),
    });
    if (rulingParty !== "none") params.set("ruling", rulingParty);
    const shareUrl = `${window.location.origin}${window.location.pathname}?${params}`;
    navigator.clipboard.writeText(shareUrl).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  };

  // Driver: open the SSE stream and wire each event to the matching setX. This is
  // the whole "state machine" — the callbacks fire as the backend streams stages.
  const handleSubmit = () => {
    esRef.current?.close();  // cancel any previous in-flight stream first
    setLoading(true);
    setError(null);
    // Clear prior results so stale charts don't linger while the new run streams in.
    setDeltaBins(null);
    setDeltas(null);
    setDeltasRace(null);
    setDeltasReligion(null);
    setDeltasGender(null);
    setEquilibrium(null);
    setSimulation(null);
    setEmpiricalSupport(null);
    setRefinement(null);

    // Weave governing context into the free-text event before it reaches the model.
    // "none" → unchanged. The model was trained on free text, so this is the only
    // channel for ruling-party context (no separate backend field exists).
    const eventText = injectRulingContext(event, rulingParty);

    const es = estimateShockStream(eventText, intensity, party, {
      onDeltas: (data) => {
        setDeltaBins({
          ...data.delta_bins_race,
          ...data.delta_bins_religion,
          ...data.delta_bins_gender,
        });
        setDeltas({
          ...data.deltas_race,
          ...data.deltas_religion,
          ...data.deltas_gender,
        });
        setDeltasRace(data.deltas_race);
        setDeltasReligion(data.deltas_religion);
        setDeltasGender(data.deltas_gender);
      },
      onEquilibrium: (data) => setEquilibrium(data),
      onSimulation: (data) => setSimulation(data),
      onEmpiricalSupport: (data) => setEmpiricalSupport(data),
      onRefinement: (data) => setRefinement(data),
      onDone: () => {
        setLoading(false);
        esRef.current = null;
      },
      onError: (msg) => {
        console.error("SSE stream error", msg);
        setError("The estimation service is unavailable. Please try again.");
        setLoading(false);
        esRef.current = null;
      },
    }, false);  // refinement toggle removed — sentiment injection is server-side (always on for covered events)
    esRef.current = es;
  };

  return (
    <div className="flex min-h-screen flex-col">
      {/* ── Sticky header ──────────────────────────────────────────────────── */}
      <header className="sticky top-0 z-10 border-b border-gray-200 bg-white/80 backdrop-blur-md">
        <div className="mx-auto flex max-w-6xl items-start justify-between gap-4 px-4 py-3">
          <div className="min-w-0">
            <h1 className="text-base font-bold leading-tight text-gray-900 sm:text-lg">
              Electoral Equilibrium
            </h1>
            <p className="mt-0.5 hidden text-xs leading-snug text-gray-500 sm:block">
              A bipartisan model of how electoral coalitions shift under hypothetical
              political shocks.
            </p>
          </div>

          {/* Link to the analyst dashboard. Plain <a> = full navigation; the
              /dashboard route is a server component that gates on the session
              cookie and shows its own login form when unauthenticated. */}
          <a
            href="/dashboard"
            className="flex flex-none items-center gap-1.5 rounded-md border border-gray-200 bg-white px-3 py-1.5 text-xs font-medium text-gray-600 shadow-sm transition-colors hover:bg-gray-50 hover:text-gray-900 sm:text-sm"
          >
            <LayoutDashboard className="h-3.5 w-3.5" />
            Dashboard
          </a>
        </div>
      </header>

      {/* ── Body ───────────────────────────────────────────────────────────── */}
      <div className="mx-auto w-full max-w-6xl flex-1 px-4 py-6">
        {/*
         * Two-column on desktop (lg+):
         *   left  — fixed-width sidebar: ShockInput + collapsible
         *   right — flex-1: error banner + results region
         * Single column on mobile: sidebar stacks above results.
         */}
        <div className="flex flex-col gap-6 lg:flex-row lg:items-start lg:gap-8">

          {/* ── Left sidebar — controls ─────────────────────────────────── */}
          <aside className="w-full space-y-4 lg:w-80 xl:w-96 lg:flex-none">
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

            <HowItWorks />

            {/* Share button — disabled until event meets the backend's ≥10-char guard */}
            <button
              onClick={handleShare}
              disabled={event.trim().length < 10}
              className="flex w-full items-center justify-center gap-1.5 rounded-md border border-gray-200 bg-white px-3 py-2 text-sm text-gray-600 hover:bg-gray-50 disabled:cursor-not-allowed disabled:opacity-40"
            >
              <Link2 className="h-3.5 w-3.5" />
              {copied ? "Copied!" : "Share this result"}
            </button>
          </aside>

          {/* ── Right column — results ──────────────────────────────────── */}
          <div className="min-w-0 flex-1 space-y-6">
            <ErrorBanner message={error} />

            {/*
             * Results region — shown as soon as loading starts so components
             * can display their own skeleton states before their SSE event
             * arrives. Gated on (deltaBins || loading) to suppress empty state
             * before first submit.
             */}
            {(deltaBins || loading) && (
              <>
                {/* ShockNarrative: populates on "deltas" (~2s) */}
                <ShockNarrative
                  deltaBins={deltaBins}
                  deltas={deltas}
                  party={party}
                  loading={loading && !deltaBins}
                />

                {/*
                 * CoalitionChart: skeleton until "equilibrium" event.
                 * Both opaque (mu_shifted) and translucent (weights) layers
                 * arrive together — no key change, no remount.
                 */}
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

                {/* Real-reaction grounding: per-bloc predicted-vs-real comparison
                    (+ refined prediction for aligned events). Renders nothing for
                    uncovered events. */}
                <EmpiricalSupportPanel
                  empiricalSupport={empiricalSupport}
                  refinement={refinement}
                />

                {/* Plain-language definitions for the two panels + the gauge */}
                <WhatDoTheseMean />

                {/* WinGauge: skeleton until "simulation" event */}
                <WinGauge
                  winProbability={simulation?.win_probability ?? null}
                  winProbabilityLow={simulation?.win_probability_low}
                  winProbabilityHigh={simulation?.win_probability_high}
                  percentiles={simulation?.percentiles ?? null}
                  loading={loading && !simulation}
                />
              </>
            )}
          </div>
        </div>
      </div>

      {/* ── Persistent disclaimer ───────────────────────────────────────────── */}
      <footer className="border-t border-gray-200 bg-white">
        <div className="mx-auto max-w-6xl px-4 py-4">
          <p className="text-xs leading-relaxed text-gray-400">
            This is a research tool, not a forecast. It estimates the directional effect
            of <em>hypothetical</em> events on coalition structure — it does not predict
            real election outcomes. Built as a CMC Summer Research Project by Juhi Damley.
          </p>
        </div>
      </footer>
    </div>
  );
}
