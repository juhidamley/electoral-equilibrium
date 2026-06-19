"use client";

import * as Slider from "@radix-ui/react-slider";
import { Loader2 } from "lucide-react";

import type { Party } from "@/lib/types";

interface ShockInputProps {
  party: Party;
  setParty: (p: Party) => void;
  event: string;
  setEvent: (e: string) => void;
  intensity: number;
  setIntensity: (i: number) => void;
  onSubmit: () => void;
  loading: boolean;
}

// Mirrors the backend 422 guard — descriptions shorter than 10 chars are rejected.
const MIN_EVENT_LENGTH = 10;

const PRESETS = [
  { label: "Security",           color: "amber",   text: "An assassination attempt is made on the leading presidential candidate" },
  { label: "Geopolitical",       color: "indigo",  text: "The US enters a major armed conflict in the Middle East" },
  { label: "Moral / Scandal",    color: "rose",    text: "A major financial scandal involving the sitting administration is revealed" },
  { label: "Electoral Surprise", color: "violet",  text: "A major October Surprise leak dominates the news cycle" },
  { label: "Economic",           color: "emerald", text: "A sudden recession is declared in the final month of the campaign" },
];

// Static class map — dynamic string interpolation is invisible to Tailwind JIT.
const COLOR_CLASSES: Record<string, string> = {
  amber:   "border-amber-400 text-amber-700 hover:bg-amber-50",
  indigo:  "border-indigo-400 text-indigo-700 hover:bg-indigo-50",
  rose:    "border-rose-400 text-rose-700 hover:bg-rose-50",
  violet:  "border-violet-400 text-violet-700 hover:bg-violet-50",
  emerald: "border-emerald-400 text-emerald-700 hover:bg-emerald-50",
};

export default function ShockInput({
  party,
  setParty,
  event,
  setEvent,
  intensity,
  setIntensity,
  onSubmit,
  loading,
}: ShockInputProps) {
  const eventTooShort = event.trim().length > 0 && event.trim().length < MIN_EVENT_LENGTH;
  const submitDisabled = loading || event.trim().length < MIN_EVENT_LENGTH;

  const partyColor = party === "democrat" ? "blue" : "red";

  return (
    <div className="space-y-6">
      {/* ── (i) Party toggle — primary decision, visually dominant ── */}
      <div>
        <p className="mb-2 text-xs font-semibold uppercase tracking-widest text-gray-500">
          Modeling party
        </p>
        <div className="inline-flex rounded-lg border border-gray-200 p-1 shadow-sm">
          <button
            type="button"
            onClick={() => setParty("democrat")}
            className={[
              "rounded-md px-6 py-2.5 text-sm font-semibold transition-colors",
              party === "democrat"
                ? "bg-blue-600 text-white shadow-sm"
                : "bg-transparent text-gray-500 hover:text-gray-700",
            ].join(" ")}
          >
            Democrat
          </button>
          <button
            type="button"
            onClick={() => setParty("republican")}
            className={[
              "rounded-md px-6 py-2.5 text-sm font-semibold transition-colors",
              party === "republican"
                ? "bg-red-600 text-white shadow-sm"
                : "bg-transparent text-gray-500 hover:text-gray-700",
            ].join(" ")}
          >
            Republican
          </button>
        </div>
      </div>

      {/* ── (ii) Event description textarea ── */}
      <div>
        <label
          htmlFor="shock-event"
          className="mb-1.5 block text-sm font-medium text-gray-700"
        >
          Political shock
        </label>
        <textarea
          id="shock-event"
          rows={3}
          value={event}
          onChange={(e) => setEvent(e.target.value)}
          placeholder="Describe a hypothetical political event..."
          className="w-full resize-y rounded-md border border-gray-300 px-3 py-2 text-sm shadow-sm placeholder:text-gray-400 focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500"
        />
        {eventTooShort && (
          <p className="mt-1 text-xs text-amber-600">
            Enter at least {MIN_EVENT_LENGTH} characters
          </p>
        )}

        {/* ── Preset pills — fill textarea only, do not submit ── */}
        <div className="mt-3">
          <p className="mb-1.5 text-xs font-medium text-gray-400">Or try a preset:</p>
          <div className="flex flex-wrap gap-2">
            {PRESETS.map((preset) => (
              <button
                key={preset.label}
                type="button"
                onClick={() => setEvent(preset.text)}
                className={[
                  "rounded-full border px-3 py-1 text-xs font-medium transition-colors",
                  COLOR_CLASSES[preset.color],
                ].join(" ")}
              >
                {preset.label}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* ── (iii) Intensity slider ── */}
      <div>
        <div className="mb-2 flex items-center justify-between">
          <label className="text-sm font-medium text-gray-700">Intensity</label>
          <span className="text-sm tabular-nums text-gray-500">
            {intensity.toFixed(1)}
          </span>
        </div>
        <Slider.Root
          min={0.5}
          max={2.0}
          step={0.1}
          value={[intensity]}
          onValueChange={([v]) => setIntensity(v)}
          className="relative flex h-5 w-full touch-none select-none items-center"
        >
          <Slider.Track className="relative h-1.5 w-full grow overflow-hidden rounded-full bg-gray-200">
            <Slider.Range
              className={`absolute h-full ${partyColor === "blue" ? "bg-blue-500" : "bg-red-500"}`}
            />
          </Slider.Track>
          <Slider.Thumb
            className={[
              "block h-4 w-4 rounded-full border-2 bg-white shadow-md",
              "focus:outline-none focus:ring-2 focus:ring-offset-1",
              partyColor === "blue"
                ? "border-blue-500 focus:ring-blue-500"
                : "border-red-500 focus:ring-red-500",
            ].join(" ")}
            aria-label="Intensity"
          />
        </Slider.Root>
        <div className="mt-1 flex justify-between text-xs text-gray-400">
          <span>0.5 — minor</span>
          <span>2.0 — major</span>
        </div>
      </div>

      {/* ── (iv) Submit button ── */}
      <button
        type="button"
        onClick={onSubmit}
        disabled={submitDisabled}
        className={[
          "flex w-full items-center justify-center gap-2 rounded-md px-4 py-3",
          "text-sm font-semibold text-white shadow-sm transition-opacity",
          "disabled:cursor-not-allowed disabled:opacity-50",
          party === "democrat" ? "bg-blue-600 hover:bg-blue-700" : "bg-red-600 hover:bg-red-700",
        ].join(" ")}
      >
        {loading ? (
          <>
            <Loader2 className="h-4 w-4 animate-spin" />
            Estimating…
          </>
        ) : (
          "Run estimate"
        )}
      </button>
    </div>
  );
}
