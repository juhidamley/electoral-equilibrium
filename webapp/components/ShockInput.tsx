"use client";

import * as Slider from "@radix-ui/react-slider";
import { Loader2 } from "lucide-react";
import type { Party } from "@/lib/types";
import type { RulingParty } from "@/lib/rulingParty";

interface ShockInputProps {
  party: Party;
  setParty: (party: Party) => void;
  rulingParty: RulingParty;
  setRulingParty: (party: RulingParty) => void;
  event: string;
  setEvent: (event: string) => void;
  intensity: number;
  setIntensity: (intensity: number) => void;
  onSubmit: () => void;
  loading: boolean;
}

const MIN_EVENT_LENGTH = 10;
const PARTIES: { value: Party; label: string }[] = [
  { value: "democrat", label: "Democrat" },
  { value: "republican", label: "Republican" },
];

function SegmentedControl({
  label,
  value,
  onChange,
}: {
  label: string;
  value: RulingParty;
  onChange: (party: Party) => void;
}) {
  return (
    <div className="min-w-0 flex-1">
      <p className="editorial-kicker mb-2 text-[10px] text-muted">{label}</p>
      <div className="grid grid-cols-2 overflow-hidden rounded-sm border border-control">
        {PARTIES.map((option) => (
          <button
            key={option.value}
            type="button"
            aria-pressed={value === option.value}
            onClick={() => onChange(option.value)}
            className={`px-3 py-2 text-[13px] transition-colors ${
              value === option.value
                ? "bg-maroon text-surface"
                : "bg-surface text-body hover:bg-rule-light"
            }`}
          >
            {option.label}
          </button>
        ))}
      </div>
    </div>
  );
}

export default function ShockInput({
  party,
  setParty,
  rulingParty,
  setRulingParty,
  event,
  setEvent,
  intensity,
  setIntensity,
  onSubmit,
  loading,
}: ShockInputProps) {
  const eventTooShort = event.trim().length > 0 && event.trim().length < MIN_EVENT_LENGTH;
  const disabled = loading || event.trim().length < MIN_EVENT_LENGTH;

  return (
    <section>
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 sm:gap-6">
        <SegmentedControl
          label="Party in power"
          value={rulingParty}
          onChange={setRulingParty}
        />
        <SegmentedControl label="Viewed from" value={party} onChange={setParty} />
      </div>

      <div className="mt-7">
        <label htmlFor="shock-event" className="editorial-kicker mb-2 block text-gold">
          The event
        </label>
        <textarea
          id="shock-event"
          rows={3}
          value={event}
          onChange={(e) => setEvent(e.target.value)}
          placeholder="Describe a hypothetical political event…"
          className="w-full resize-y border-0 border-b border-rule-light bg-transparent px-0 py-2 text-[17px] italic leading-relaxed text-ink outline-none placeholder:text-faint focus:border-gold"
        />
        {eventTooShort && <p className="mt-1 text-xs italic text-party-rep">Enter at least 10 characters.</p>}
      </div>

      <div className="mt-5 grid grid-cols-1 items-end gap-4 sm:grid-cols-[1fr_150px]">
        <div>
          <div className="mb-2 flex justify-between text-xs text-muted">
            <label>Event intensity</label>
            <span>{intensity.toFixed(1)}</span>
          </div>
          <Slider.Root
            min={0.5}
            max={2}
            step={0.1}
            value={[intensity]}
            onValueChange={([next]) => setIntensity(next)}
            className="relative flex h-5 touch-none items-center"
          >
            <Slider.Track className="relative h-1 w-full grow bg-rule">
              <Slider.Range className="absolute h-full bg-gold" />
            </Slider.Track>
            <Slider.Thumb
              aria-label="Event intensity"
              className="block h-3.5 w-3.5 rounded-full border-2 border-maroon bg-surface outline-none focus:ring-2 focus:ring-gold"
            />
          </Slider.Root>
        </div>
        <button
          type="button"
          onClick={onSubmit}
          disabled={disabled}
          className="flex h-10 items-center justify-center gap-2 bg-maroon px-4 text-sm text-surface transition-opacity disabled:cursor-not-allowed disabled:opacity-40"
        >
          {loading && <Loader2 className="h-4 w-4 animate-spin" />}
          {loading ? "Estimating…" : "Run estimate"}
        </button>
      </div>
    </section>
  );
}
