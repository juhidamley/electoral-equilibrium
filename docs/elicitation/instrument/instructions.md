# Expert Elicitation — Instructions

Thank you for participating. This document explains what you're predicting,
how to fill in the response form, and the ground rules. Please read this
whole sheet before you start — it's short, and the ground rules (§4) matter
for your responses to be usable.

## 1. What you're predicting

For each of 12 political/news events ("shocks"), you'll predict how the event
shifted the **Democratic Party's support** within each of 15 demographic
groups ("blocs") — 5 race/ethnicity groups, 7 religious groups, 3 gender
groups. Every prediction is framed the same way, regardless of which party
the event seems to "belong" to: **does this bloc's support for the Democratic
Party go up, go down, or stay about the same, as a result of this specific
event?**

You are NOT predicting which party wins an election, or general public
opinion. You're predicting a **within-bloc shift in Democratic support**,
attributable to this one event. Two people might be asking "who's the
Republican leadership going to look for the villain among Party members?" —
but only one of those two questions is the one we're asking. If a bloc's
support doesn't plausibly move at all from this specific event, "no effect"
is a completely legitimate, useful answer.

## 2. The 12 shocks

Read `shock_descriptions.md` for the full text. Each entry is a neutral,
factual paragraph — no analysis, no framing about who it helps or hurts. You
will receive the shocks **in a randomized order** (randomized separately for
you vs. other experts) to avoid order effects. Read each description once,
form your prediction from what you already know about the event and about
American political behavior, and fill in the grid. **Please do not look up
additional information, polling, or news coverage about these events before
or while responding** — we want your standing expert judgment, not a
research report (see §4).

One shock (`ayatollah_assassination`) is a constructed hypothetical with no
real-world precedent — this is intentional. Answer it the same way, reasoning
from the scenario as described.

## 3. Filling in the response grid

You'll get a spreadsheet (`response_form_template.csv`, or an .xlsx version
if you prefer — ask the coordinator) with one row per (shock, bloc) — 180
rows total. For **each row**, fill in:

| Column | What to enter |
|---|---|
| `direction` | One of: `DEM_GAIN`, `REP_GAIN`, `NO_EFFECT`, `NO_PREDICTION` |
| `direction_confidence` | One of: `HIGH`, `MEDIUM`, `LOW` — leave blank if `direction` is `NO_PREDICTION` |
| `magnitude_bucket` | One of the 5 buckets below — leave blank if `direction` is `NO_PREDICTION` (a `NO_EFFECT`/negligible-magnitude row is fine to fill as `NEGLIGIBLE`) |
| `notes` | Optional. Anything you want on record — a caveat, a reason you split from your usual intuition, whatever. Not required. |

**`NO_PREDICTION` is a real, useful answer — use it freely.** If you don't
have a confident view on how, say, this event moved Buddhist voters'
Democratic support specifically, say so. A forced guess on a bloc you don't
have a real basis to judge manufactures false signal and actively hurts the
analysis; it does not help you "finish faster" in any way that matters. We
expect most experts to use `NO_PREDICTION` on a meaningful fraction of rows,
especially for smaller/less-familiar blocs — that's expected and fine.

### Magnitude buckets

These are anchored to **real measured shifts from panel survey data**, not
a theoretical range — actual within-person Democratic-support shifts around
comparable events have mostly fallen between 0.001 and 0.03 (i.e. a tenth of
a percentage point to three percentage points of a bloc's own composition).
Use this table:

| Bucket | Approximate shift | Read as |
|---|---|---|
| `NEGLIGIBLE` | ~0 to 0.3 percentage points | Barely detectable, if at all |
| `SMALL` | ~0.3 to 1.0 percentage points | A real but modest shift |
| `MODERATE` | ~1.0 to 2.0 percentage points | A shift you'd expect to show up clearly in a well-powered survey |
| `LARGE` | ~2.0 to 3.0 percentage points | A big shift for a single event — among the larger effects we've measured for any bracketed shock in this project |
| `VERY_LARGE` | > 3.0 percentage points | Larger than nearly anything we've observed in survey data for a single event. Use this if you genuinely believe it, but it's worth a `notes` entry explaining why this bloc/event is exceptional |

**If magnitude estimation feels much harder than direction estimation, that's
expected and fine — prioritize getting `direction` and `direction_confidence`
right.** Direction is the primary thing this study needs; magnitude is
collected as a secondary, exploratory signal. You are not being scored or
compared on magnitude accuracy the way you are on direction.

## 4. Ground rules (blinding)

These matter — please follow them exactly:

1. **Do not look at any model output, any other expert's responses, or any
   ground-truth survey data (panel, CES, or exit-poll numbers) before or
   while completing this form.** If you've seen any of this project's
   ground-truth deltas before, tell the coordinator before starting — that
   shock (or bloc) needs to be excluded from your responses.
2. **Do not discuss your in-progress predictions with other participating
   experts**, including Juhi or Prof. Espinosa, until everyone has submitted.
3. **Work independently.** Don't research the shocks beyond what you already
   know (§2) — we're eliciting your existing expert judgment, not a
   literature review.
4. Once you submit, your responses are locked (see the pre-registration
   procedure in `protocol.md` §3) — you can't revise them after seeing
   anyone else's answers or any model output. If you realize you made an
   entry error before the lock, tell the coordinator immediately; corrections
   are only possible before locking.

## 5. Time estimate and submission

Budget **60-90 minutes**, in one sitting or several — the form can be saved
and resumed at any point before you submit it as final. When you're done,
send the completed file back to the coordinator (Juhi), who will lock it
(timestamp + hash + git commit) before looking at any model predictions.

## 6. Your identity

You'll be assigned an `expert_code` (e.g. `E1`) — use it exactly as given in
the `expert_code` column of every row. Your name is not used in any of the
analysis files; the coordinator keeps the code-to-name mapping separately.
