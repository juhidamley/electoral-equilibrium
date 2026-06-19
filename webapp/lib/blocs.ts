// Shared demographic bloc constants — imported by ShockNarrative AND CoalitionChart.
// Any change to labels or the race-bloc list must be made here only.

export const RACE_BLOCS = [
  "african_american",
  "asian",
  "latino",
  "other_race",
  "white",
] as const;

export type RaceBlocId = (typeof RACE_BLOCS)[number];

export const BLOC_LABEL: Record<string, string> = {
  // Race
  african_american: "Black voters",
  asian:            "Asian voters",
  latino:           "Latino voters",
  other_race:       "voters of other backgrounds",
  white:            "White voters",
  // Religion
  evangelical:      "Evangelicals",
  catholic:         "Catholics",
  protestant:       "Protestants",
  secular:          "secular voters",
  jewish:           "Jewish voters",
  muslim:           "Muslim voters",
  other_rel:        "other religious voters",
  // Gender
  women:            "Women",
  men:              "Men",
  other_gender:     "other-gender voters",
};
