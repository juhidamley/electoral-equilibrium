# Elicitation Instrument — Shock Descriptions

12 shocks, each with a neutral one-paragraph description written independently
for this instrument (NOT copied from `configs/shocks.json`'s `description`
field, which feeds the model's own prompt — reusing that text would blind
experts to a description the model itself sees, defeating the point of an
independent read). Facts (dates, entities, scale) are preserved; interpretive
or valence-loaded framing is deliberately avoided. Present these to experts in
**randomized order per expert** (see `protocol.md` §Blinding for the exact
randomization procedure) — the order below is for reference only, not the
presentation order.

Each entry states: the `shock_id` (for internal tracking/scoring only — do not
explain the ID's own connotation to experts beyond the description text),
category, existing survey-layer coverage (for the coordinator's own tracking;
irrelevant to experts and not shown to them), and the reasoning tag this shock
was selected to test.

---

### 1. `sept_11_2001` — Security — coverage: none — tests: rally-effect baseline

> On the morning of September 11, 2001, hijackers affiliated with al-Qaeda
> crashed four commercial airliners into the World Trade Center towers in New
> York City, the Pentagon in Arlington, Virginia, and a field in Shanksville,
> Pennsylvania. Nearly 3,000 people were killed. It was the deadliest act of
> terrorism on US soil in the nation's history. President George W. Bush
> addressed the nation that evening; the United States and allied forces
> began military operations in Afghanistan the following month.

### 2. `hurricane_katrina_2005` — Domestic Policy — coverage: none — tests: rally-effect baseline (domestic)

> On August 29, 2005, Hurricane Katrina made landfall on the US Gulf Coast as
> a Category 3 storm. Levee failures flooded roughly 80% of New Orleans,
> including neighborhoods that were majority Black and lower-income. Over
> 1,800 people died across the affected region. The federal disaster response,
> coordinated primarily through FEMA under the Bush administration, faced
> sustained criticism over its speed and effectiveness in the weeks that
> followed.

### 3. `ayatollah_assassination` — Geopolitical (constructed/hypothetical) — coverage: none — tests: reasoning without precedent

> **This is a constructed hypothetical scenario, not a historical event — no
> subsequent news coverage exists to consult.** Assume: a US airstrike kills
> Ali Khamenei, the Supreme Leader of Iran, who has held that position since
> 1989. The strike is publicly acknowledged by the US government within 24
> hours. No other details of the surrounding circumstances are specified.

### 4. `financial_crisis_2008` — Economic — coverage: CES (consistency check) — tests: straightforward incumbent-blame

> On September 15, 2008, the investment bank Lehman Brothers filed for
> bankruptcy, triggering a cascading global financial crisis. Credit markets
> froze, major financial institutions required emergency federal intervention,
> and the US unemployment rate rose toward 10% over the following year as
> millions of Americans lost homes to foreclosure. The crisis occurred under
> the incumbent Bush administration, two months before the 2008 presidential
> election.

### 5. `bin_laden_killing_2011` — Geopolitical — coverage: CES (consistency check) — tests: straightforward success/credit

> On May 2, 2011, a US Navy SEAL team killed Osama bin Laden, the founder and
> leader of al-Qaeda and the principal architect of the September 11 attacks,
> during a raid on a compound in Abbottabad, Pakistan. The operation was
> authorized by President Obama and announced by him in a televised address
> that evening.

### 6. `russia_ukraine_invasion_2022` — Geopolitical — coverage: CES (consistency check) — tests: low/uncertain partisan signal

> On February 24, 2022, Russia launched a full-scale military invasion of
> Ukraine, marking the largest conventional land war in Europe since World War
> II. The Biden administration and both parties in Congress supported
> sanctions on Russia and military aid to Ukraine in the invasion's immediate
> aftermath. The war has continued with no swift resolution.

### 7. `charlottesville_2017` — Moral/Scandal — coverage: panel data exists but is statistically unusable (too many co-occurring events in its bracket window — see `protocol.md`) — tests: contested presidential response; this shock has NO other usable ground truth, elicitation is the only source

> On August 12, 2017, a "Unite the Right" rally of white nationalist and
> white supremacist groups in Charlottesville, Virginia, turned violent; a
> counter-protester, Heather Heyer, was killed when a rally attendee drove a
> car into a crowd. Two days later, President Trump stated there were "very
> fine people on both sides" of the confrontation, a remark that drew
> criticism from members of both parties as well as defenses from some of his
> supporters.

### 8. `kavanaugh_2018` — Moral/Scandal — coverage: panel Tier B (consistency check) — tests: double-mobilization (both sides energized)

> In September 2018, Christine Blasey Ford testified before the Senate
> Judiciary Committee that Supreme Court nominee Brett Kavanaugh had sexually
> assaulted her when they were both in high school in the early 1980s.
> Kavanaugh testified in his own defense, denying the allegation. The
> confirmation process drew sustained national attention over several weeks;
> the Senate confirmed Kavanaugh to the Supreme Court on October 6, 2018, by a
> vote of 50-48.

### 9. `family_separation_2018` — Immigration — coverage: panel Tier B (consistency check) — tests: high-salience, cross-party disapproval in general polling but bloc-level intensity may vary

> Beginning in spring 2018, the Trump administration's "zero tolerance"
> border enforcement policy resulted in the separation of migrant children
> from their parents after unauthorized crossings at the US-Mexico border.
> Reporting in June 2018 established that more than 2,500 children had been
> separated from their families under the policy. Widespread public criticism
> followed; the administration ended the practice via executive order on June
> 20, 2018.

### 10. `blm_george_floyd_2020` — Criminal Justice — coverage: panel Tier B (consistency check) — tests: double-mobilization (both sides energized)

> On May 25, 2020, Minneapolis police officer Derek Chauvin killed George
> Floyd, a Black man, by kneeling on his neck for over nine minutes during an
> arrest; the killing was recorded on video by bystanders. In the weeks that
> followed, protests against police violence took place in hundreds of US
> cities, among the largest sustained protest movements in the nation's
> history. Some protests included property destruction and clashes with
> police; the large majority were peaceful.

### 11. `dobbs_2022` — Electoral/Voting Rights — coverage: CES (consistency check) — tests: the paradigmatic "mobilizing backlash" case — a conservative-aligned legal outcome that may energize the opposing side more than its own base

> On June 24, 2022, the US Supreme Court ruled in *Dobbs v. Jackson Women's
> Health Organization* to overturn *Roe v. Wade*, the 1973 decision that had
> established a constitutional right to abortion. The ruling returned the
> authority to regulate or ban abortion to individual states. Several states
> implemented near-total abortion bans within weeks of the decision, while
> others moved to protect or expand abortion access.

### 12. `jan_6_insurrection_2021` — Electoral/Voting Rights — coverage: CES (consistency check) — tests: sharply contested partisan interpretation of the same event

> On January 6, 2021, a crowd of supporters of President Trump, who had
> rallied near the White House to protest the certification of Joe Biden's
> Electoral College victory, marched to and breached the US Capitol building
> while Congress was in the process of certifying the election results.
> Congress was evacuated; the building was cleared after several hours and
> certification resumed that night. Five people died in connection with the
> events of the day; more than 140 police officers were injured.

---

## Selection rationale summary

| # | shock_id | Category | Existing coverage | Why selected |
|---|---|---|---|---|
| 1 | sept_11_2001 | Security | None | No survey layer reaches 2001; unambiguous rally-effect baseline; maximally recognizable |
| 2 | hurricane_katrina_2005 | Domestic Policy | None | No survey layer reaches 2005; domestic (non-foreign-policy) incumbent-blame baseline with a racial-equity dimension relevant to bloc-level reasoning |
| 3 | ayatollah_assassination | Geopolitical (hypothetical) | None | Cannot exist in any survey — pure reasoning test with no precedent to recall; already a designed pilot scenario in `configs/shocks.json` |
| 4 | financial_crisis_2008 | Economic | CES | Straightforward incumbent-blame economic shock; CES year-pair (2007→2008) gives a real, if coarse, consistency check |
| 5 | bin_laden_killing_2011 | Geopolitical | CES | Straightforward credit-to-incumbent shock, opposite valence direction from #4; CES consistency check |
| 6 | russia_ukraine_invasion_2022 | Geopolitical | CES | The one shock selected specifically to test whether the model (and experts) correctly predict LOW/NULL partisan signal rather than over-predicting a partisan effect on every input |
| 7 | charlottesville_2017 | Moral/Scandal | Panel (Tier C — unusable) | The one shock in this set where elicitation is the ONLY validation route at all — its panel bracket window is too crowded (5 co-occurring shocks) for any other method to score it |
| 8 | kavanaugh_2018 | Moral/Scandal | Panel Tier B | Double-mobilization: plausibly energized both pro-confirmation and #MeToo-aligned blocs simultaneously — exactly the confound this validation route is meant to catch |
| 9 | family_separation_2018 | Immigration | Panel Tier B | High-salience, emotionally direct shock; tests whether bloc-level predictions track general-polling disapproval or diverge from it |
| 10 | blm_george_floyd_2020 | Criminal Justice | Panel Tier B | Double-mobilization, same reasoning as #8, different domain (policing rather than judicial confirmation) |
| 11 | dobbs_2022 | Electoral/Voting Rights | CES | The paradigmatic mobilizing-backlash case named explicitly in the Step 1.4 brief — a nominally conservative "win" plausibly energizing the opposing coalition more than its own |
| 12 | jan_6_insurrection_2021 | Electoral/Voting Rights | CES | Same event, two starkly different partisan readings in contemporaneous coverage — tests whether bloc-level predictions track the same divide |

**Coverage mix**: 3 shocks with zero survey-layer coverage of any kind, 1 shock
where the panel technically has data but it is statistically unusable (so
elicitation is its only real validation), 3 shocks with panel Tier B
consistency-check data, 5 shocks with CES consistency-check data.

**Category mix**: Security, Domestic Policy, Geopolitical (×3), Economic,
Moral/Scandal (×2), Immigration, Criminal Justice, Electoral/Voting Rights
(×2) — 9 of the taxonomy's category labels represented across 12 shocks.

**Valence/dynamic mix**: 2 straightforward single-direction shocks (#4, #5,
opposite valence from each other), 1 explicit null/low-signal test (#6), 1
novel/no-precedent reasoning test (#3), 2 rally-effect baselines (#1, #2), and
4 contested-or-double-mobilization shocks (#7, #8, #10, #11/#12 pair) —
weighted toward the mobilizing-backlash dynamic since that is the confound
most likely to break a naive model.
