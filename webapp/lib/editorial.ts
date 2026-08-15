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

// Net thresholds are expressed on the canonical ±0.0375 loyalty-delta scale.
// They preserve the brief's distinction between effectively neutral, slight,
// ordinary, and unusually clear aggregate movement.
export function netVerdict(deltaEff: number, modeledParty: Party): string {
  const magnitude = Math.abs(deltaEff);
  if (magnitude < 0.001) return "Little changes.";
  const beneficiary = deltaEff > 0 ? modeledParty : oppositeParty(modeledParty);
  if (magnitude < 0.005) return `A slight edge to the ${partyName(beneficiary)}.`;
  if (magnitude < 0.015) return `An edge to the ${partyName(beneficiary)}.`;
  return `A clear shift toward the ${partyName(beneficiary)}.`;
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

export function toDemocraticDirection(deltaForModeledParty: number, modeledParty: Party): number {
  return modeledParty === "democrat" ? deltaForModeledParty : -deltaForModeledParty;
}
