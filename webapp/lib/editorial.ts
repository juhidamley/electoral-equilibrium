import type { Party } from "./types";
import {
  GENDER_BLOCS,
  RACE_BLOCS,
  RELIGION_BLOCS,
} from "./blocs";

export const EDITORIAL_BLOC_LABELS: Record<string, string> = {
  white: "White voters",
  african_american: "Black voters",
  latino: "Latino voters",
  asian: "Asian voters",
  other_race: "Other / multiracial voters",
  evangelical: "Evangelical Christians",
  catholic: "Catholic voters",
  protestant: "Mainline Protestants",
  secular: "Secular / non-religious",
  jewish: "Jewish voters",
  muslim: "Muslim voters",
  other_rel: "Other faiths",
  women: "Women",
  men: "Men",
};

// Display-only omission, per owner decision: the source category is sub-1% of
// the population. It remains in API data, validation, modeling, and aggregation.
export const EDITORIAL_GENDER_BLOCS = GENDER_BLOCS.filter(
  (bloc) => bloc !== "other_gender",
);

export const EDITORIAL_SECTIONS = [
  { title: "By race", blocs: RACE_BLOCS },
  { title: "By religion", blocs: RELIGION_BLOCS },
  { title: "By gender", blocs: EDITORIAL_GENDER_BLOCS },
] as const;

export function oppositeParty(party: Party): Party {
  return party === "democrat" ? "republican" : "democrat";
}

export function partyName(party: Party, adjective = false): string {
  if (adjective) return party === "democrat" ? "Democratic" : "Republican";
  return party === "democrat" ? "Democrats" : "Republicans";
}

export type NetVerdictBin = "little" | "slight" | "edge" | "clear";

// These thresholds are on the canonical ±0.0375 loyalty-delta scale. Keep the
// verdict headline and its explanatory copy on this one shared classification.
export function netVerdictBin(deltaEff: number): NetVerdictBin {
  const magnitude = Math.abs(deltaEff);
  if (magnitude < 0.001) return "little";
  if (magnitude < 0.005) return "slight";
  if (magnitude < 0.015) return "edge";
  return "clear";
}

export interface NetVerdictCopy {
  bin: NetVerdictBin;
  headline: string;
  context: string;
  beneficiary: Party | null;
}

export function netVerdictCopy(deltaEff: number, modeledParty: Party): NetVerdictCopy {
  const bin = netVerdictBin(deltaEff);
  const beneficiary = deltaEff === 0 ? null : deltaEff > 0 ? modeledParty : oppositeParty(modeledParty);
  const smallContext = "The shift is small — the norm. Few voters leave the side they already favor; elections turn on the margins that do.";
  const largeContext = "For a single event, this is a comparatively large movement — though still far smaller than the gap between the parties' existing coalitions.";

  if (bin === "little") return { bin, headline: "Little changes.", context: smallContext, beneficiary: null };
  if (bin === "slight") return { bin, headline: `A slight edge to the ${partyName(beneficiary!)}.`, context: smallContext, beneficiary };
  if (bin === "edge") return { bin, headline: `An edge to the ${partyName(beneficiary!)}.`, context: largeContext, beneficiary };
  return { bin, headline: `A clear shift toward the ${partyName(beneficiary!)}.`, context: largeContext, beneficiary };
}

export function netVerdict(deltaEff: number, modeledParty: Party): string {
  return netVerdictCopy(deltaEff, modeledParty).headline;
}

export function probabilityRead(
  probability: number,
  modeledParty: Party,
  shiftParty: Party | null,
): string {
  const leader = probability >= 0.5 ? modeledParty : oppositeParty(modeledParty);
  if (shiftParty && shiftParty !== leader) return `Not enough to threaten the ${partyName(leader)}' lead`;
  if (shiftParty === leader && probability >= 0.6) return `And it widens the ${partyName(leader)}' lead`;
  if (probability >= 0.6) return `The ${partyName(leader)}' edge holds`;
  return "The race remains close";
}

export interface ShiftWords {
  text: string;
  party: Party | null;
}

// Boundaries align with the canonical neutral/mild/moderate representative
// values in deltaScale.ts and the backend decode table.
export function shiftWords(deltaForModeledParty: number, modeledParty: Party): ShiftWords {
  const magnitude = Math.abs(deltaForModeledParty);
  if (magnitude < 0.001) return { text: "barely moves", party: null };
  const beneficiary = deltaForModeledParty > 0 ? modeledParty : oppositeParty(modeledParty);
  const short = beneficiary === "democrat" ? "Dem." : "Rep.";
  if (magnitude < 0.00875) return { text: `nudges ${short}`, party: beneficiary };
  if (magnitude < 0.0175) return { text: `shifts ${short}`, party: beneficiary };
  return { text: `moves ${short} sharply`, party: beneficiary };
}

// The stream is party-relative: ShockResponseData.party is the selected
// "Viewed from" party. Convert its positive-toward-modeled-party sign to the
// shared Democrat-right / Republican-left editorial axis.
export function toDemocraticDirection(deltaForModeledParty: number, modeledParty: Party): number {
  return modeledParty === "democrat" ? deltaForModeledParty : -deltaForModeledParty;
}

// mu_shifted uses the same party-relative loyalty convention as the deltas.
// Convert a [0, 1] modeled-party loyalty level to a Democratic loyalty level.
export function toDemocraticLoyalty(levelForModeledParty: number, modeledParty: Party): number {
  return modeledParty === "democrat" ? levelForModeledParty : 1 - levelForModeledParty;
}
