# Expert Elicitation Protocol — Step 1.4

Status: **DESIGN ONLY. Nothing in this protocol has been run.** No experts
have been contacted, no predictions collected, no model comparison performed.
This document specifies the full procedure so it can be executed without
redesign after a ~6-week gap, and so a reviewer can verify every methodological
choice was fixed before any data existed.

## 0. Why this is the primary direction-validation route

The three survey-based ground-truth layers (`data/ground_truth/panel_deltas.json`,
`ces_deltas.json`, `exit_poll_deltas.json`, built in Steps 1.1-1.3) cannot
support per-bloc direction validation at the scale this project needs:

- **The VOTER panel** — the only true within-person layer — has only
  **19 of 270 (shock, bloc) cells** that pass all four trust filters
  (`data/ground_truth/trustworthy_subset.md`). This is a **statistical power**
  failure, not a window-quality failure: even the tightest, cleanest bracket
  (Tier A, 46-day window) only clears the bar for one bloc (catholic).
  Widening bloc sample sizes isn't available on demand — the panel is a fixed,
  already-collected dataset.
- **CES** (annual cross-sections) and **exit polls** (cross-cycle) are
  structurally **never** single-shock-attributable (`ces_deltas.json` and
  `exit_poll_deltas.json` set `single_shock_attributable: false`
  unconditionally, for every cell) — CES windows run 267-1029 days and exit
  polls compare different electorates 4-20 years apart.
- **The panel and exit polls actively disagree** on the one case where both
  exist at meaningful scale: the 2016→2020 comparison for `election_2020`
  disagrees in *sign* on 5 of 6 comparable blocs (`ground_truth_layers.md`).
  Two independent survey-based sources contradicting each other is itself
  evidence that survey layers alone cannot arbitrate direction here.

**Expert elicitation is the only validation method whose unit matches the
model's own**: a single shock, asked about in isolation, with a direct
per-bloc directional judgment. It cannot replace survey ground truth — it has
its own weaknesses (small N, no guarantee experts are right, potential shared
priors) — but it tests something no survey layer can: whether a domain expert,
reasoning about one event at a time the way the model is asked to, agrees with
the model's per-bloc direction.

---

## 1. Candidate shock selection

**12 shocks**, selected from `configs/shocks.json`. Full descriptions and the
selection-rationale table are in `instrument/shock_descriptions.md`; summary
here:

| # | shock_id | Category | Coverage | Selected because |
|---|---|---|---|---|
| 1 | sept_11_2001 | Security | none | No survey layer reaches 2001; unambiguous rally-effect baseline |
| 2 | hurricane_katrina_2005 | Domestic Policy | none | No survey layer reaches 2005; domestic incumbent-blame baseline |
| 3 | ayatollah_assassination | Geopolitical (hypothetical) | none | No precedent exists anywhere to recall — pure reasoning test |
| 4 | financial_crisis_2008 | Economic | CES | Straightforward incumbent-blame; CES consistency check |
| 5 | bin_laden_killing_2011 | Geopolitical | CES | Straightforward credit-to-incumbent, opposite valence from #4 |
| 6 | russia_ukraine_invasion_2022 | Geopolitical | CES | The null/low-partisan-signal test case |
| 7 | charlottesville_2017 | Moral/Scandal | panel (Tier C — unusable) | **Elicitation is the only usable validation for this shock at all** |
| 8 | kavanaugh_2018 | Moral/Scandal | panel Tier B | Double-mobilization (both sides plausibly energized) |
| 9 | family_separation_2018 | Immigration | panel Tier B | High-salience; tests bloc-level vs. general-polling divergence |
| 10 | blm_george_floyd_2020 | Criminal Justice | panel Tier B | Double-mobilization, different domain from #8 |
| 11 | dobbs_2022 | Electoral/Voting Rights | CES | The paradigmatic mobilizing-backlash case |
| 12 | jan_6_insurrection_2021 | Electoral/Voting Rights | CES | Same event, starkly divergent partisan readings |

**Coverage mix**: 3 shocks with zero prior survey-layer coverage, 1 shock
where survey data exists but is statistically unusable, 3 with panel Tier B
consistency-check data, 5 with CES consistency-check data. Every shock with
existing (even partial) survey coverage is retained specifically so its
elicitation result can be cross-checked against the corresponding
`panel_deltas.json`/`ces_deltas.json` cell — see §5.4.

**Why 12, not 15**: at 15 blocs/shock, 15 shocks would be 225 response rows;
piloting the response format (§2) against a 60-90 minute time budget (§4) set
the ceiling at 12 shocks × 15 blocs = 180 rows. Diversity was prioritized over
count — see the category/valence mix table in `shock_descriptions.md`.

## 2. The blinded instrument

Files: `instrument/shock_descriptions.md` (the 12 neutral descriptions),
`instrument/instructions.md` (expert-facing instruction sheet, human-readable
version of everything in this section), `instrument/response_form_template.csv`
(the fillable grid).

**Design decisions and their rationale:**

- **Party framing is fixed to "Democratic support," always.** Rather than
  asking about "the party being modeled" per shock (which would double
  response burden if asked both ways, or introduce inconsistent framing if
  chosen per-shock), every prediction is phrased as "does this bloc's
  Democratic support go up, down, or stay flat." This exactly matches the
  ground-truth layers' `dem_loyalty` sign convention
  (`(7-pid7)/6`, positive = more Democratic) — a model prediction made with
  `party="republican"` must be sign-flipped before comparison, exactly as
  `electoral/metrics/ground_truth_accuracy.py`'s `normalize_prediction()`
  already does for the survey-layer scorer. One scoring convention, one
  elicitation convention, no translation layer to get wrong later.
- **Neutral descriptions, written fresh** (`shock_descriptions.md`), not
  copied from `configs/shocks.json`'s `description` field — that field feeds
  the model's own prompt, and reusing its exact phrasing would mean experts
  and the model see identically-framed text, which is not the same as an
  independent read.
- **Direction and magnitude are separate response fields, with confidence
  attached to direction only.** Per the design brief: direction is the
  primary thing this study needs, and magnitude is plausibly much harder for
  a human to estimate well. Collecting them separately means a low-confidence
  or skipped magnitude never invalidates a usable direction judgment.
- **Magnitude buckets are anchored to ±0.03, not ±0.15.** The old ±0.15 range
  is the *theoretical* bin range used elsewhere in this codebase
  (`electoral/core/types.py`'s `DELTA_BINS`); it is not what the panel
  actually measures. Real within-person deltas from `panel_deltas.json` run
  roughly 0.0002 to 0.03 in magnitude (e.g. `travel_ban_2017`'s `other_race`
  cell: -0.0201 unweighted). Anchoring bucket boundaries to the ±0.15 range
  would put nearly every real answer in the bottom bucket and make the scale
  useless for discriminating expert judgments — see the bucket table in
  `instructions.md` §3.
- **`NO_PREDICTION` is always available and explicitly encouraged** on any
  row, including direction. Forcing a response on a bloc/shock pair an expert
  has no real basis to judge manufactures signal that isn't there — see §5.3
  for how missing responses are handled downstream.
- **Randomized presentation order, per expert.** Each expert receives the 12
  shocks in an order determined by `random.Random(expert_code).shuffle(shock_ids)`
  (a fixed, reproducible seed keyed to their assigned code, not a fresh random
  draw each time) — this avoids a shared fixed order that could let a later
  shock's judgment be anchored by an earlier one in a way that's correlated
  across experts. The coordinator generates each expert's ordered copy of
  `response_form_template.csv` at recruitment time (see `scripts/score_elicitation.py`
  for the exact shuffle call once it's exercised).

**Blinding requirements** (full text in `instructions.md` §4): experts must
not see model output, each other's responses, or any ground-truth survey
deltas before submitting; must not discuss in-progress predictions with each
other or with the coordinators; must not research beyond existing knowledge.
Any prior exposure to ground truth for a specific shock/bloc must be
self-disclosed and that cell excluded.

## 3. Pre-registration mechanism

Predictions must be locked — timestamped and immutable — before any model
output is compared against them, and a reviewer must be able to verify this
independently.

**Procedure:**

1. Each expert returns their completed `response_form_template.csv` (renamed
   `{expert_code}_responses.csv`) to the coordinator.
2. The coordinator computes `sha256(file_bytes)` for the exact file received
   and appends a row to `docs/elicitation/pre_registration_log.csv`:
   `expert_code, filename, sha256, received_at (ISO8601, coordinator's local
   clock), n_rows_completed`. This log is **append-only** — rows are never
   edited or deleted, only added; corrections are new rows with a `supersedes`
   note, never in-place edits.
3. The coordinator commits the response file to
   `docs/elicitation/responses/{expert_code}_responses.csv` in git, with a
   commit message of the form `elicitation: lock {expert_code} responses
   (sha256 {first 12 chars})`, using a **GPG-signed commit** (`git commit -S`)
   if a signing key is available, so the commit's authenticity and timestamp
   are independently verifiable via `git log --show-signature`.
4. This is repeated independently for each expert as their response arrives
   — experts are locked as they finish, not batched, so no expert's timing
   depends on another's.
5. **No model prediction file may be generated, run, or referenced by the
   coordinator until every expert who is going to respond has been locked**
   (or the recruitment window has closed — see §4). This is a procedural rule
   for the humans involved, not something code can enforce — but
   `scripts/score_elicitation.py` adds a second, mechanical check: it refuses
   to score if the model-predictions file's mtime (or an explicit
   `--model-predictions-generated-at` timestamp) precedes any expert's
   `received_at` in the log, i.e. it checks the *right* ordering
   (predictions locked before model output exists), not just *a* ordering.

**What a reviewer can verify**: `git log` on `docs/elicitation/responses/`
shows commit timestamps (and signatures, if used) for every expert's locked
file; `pre_registration_log.csv`'s `sha256` column lets anyone confirm a given
response file hasn't been altered since it was logged; the two together (git
history + independent hash log) mean tampering would require rewriting git
history AND the log AND getting away with a hash mismatch — a high bar for an
append-only, small research project.

## 4. Expert recruitment plan

**Target: 4 experts beyond Juhi and Prof. Espinosa. Minimum usable: 3.**

- **Why 3 minimum**: with 2 experts, inter-rater agreement collapses to a
  single pairwise number with no way to tell "these two happen to agree" from
  a real consensus, and majority-vote consensus-building (§5.3) isn't
  possible at all with an even split. 3 is the minimum N where a majority
  vote is always decisive (no ties) and a multi-rater agreement statistic
  (Fleiss' κ, §5.1) is meaningful rather than degenerate.
- **Why 4 is the target, not the minimum**: response attrition is real even
  for a 60-90 minute ask — targeting 4 assumes ~1 may not finish or may
  respond to only part of the form, still leaving 3 usable.

**Qualifications** (any ONE of the following counts):
- Faculty or postdoctoral researcher in political science, with a research or
  teaching focus that includes American political behavior, public opinion,
  or voting behavior.
- PhD candidate (ABD or later) in political science or a closely related
  field (sociology, communication), with the same substantive focus.
- Evidence of the above: e.g. has taught or TA'd a course on American
  political behavior/public opinion, or has authored/co-authored work in the
  area — a lightweight bar, not a formal credentialing process.

Experts are **not** required to be specialists in all 15 blocs individually —
broad training in mass political behavior is the bar, and `NO_PREDICTION`
(§2) exists precisely so an expert with weaker priors on, say, Muslim or
other_rel voters isn't forced to fabricate an opinion.

**Time ask**: 60-90 minutes, statable up front to prospective experts so they
can decline before starting rather than abandon partway through (see
`instructions.md` §5). Splittable across sessions.

**Partial responses**: a response file is eligible for the **primary
analysis** if the expert completed `direction` (non-blank, may be
`NO_PREDICTION`) for **at least 80% of rows (≥144 of 180)**. Below that
threshold, the response is retained (never discarded — it's still real data,
locked the same way) but reported only in a supplementary note, not folded
into the primary consensus computation, since a heavily incomplete response
could distort per-shock coverage unevenly (e.g. an expert who only answered
questions about 3 of the 12 shocks isn't really comparable to one who
sampled all 12 lightly). 80% was chosen as a bar that tolerates realistic
`NO_PREDICTION` usage (expected to be nontrivial, per §2) while excluding
responses so incomplete they're effectively a different, smaller study.

**Juhi and Prof. Espinosa's own responses**: both may complete the same
instrument (useful as an internal check and to keep the option open), but are
**flagged `non_independent: true` in the response schema and analyzed
completely separately from the recruited-expert consensus** — never pooled
into the inter-expert agreement statistic or the model-comparison consensus
in §5. Having built the model, their priors about "what the model probably
predicts" are not independent of it in the way the whole design requires.
Their responses may be reported alongside the main result as a labeled,
separate data point ("the model's own authors predicted X; the independent
expert consensus predicted Y") but must never be silently blended in.

## 5. The agreement metric — defined before any data exists

### 5.1 Inter-expert agreement (computed FIRST, always)

For every (shock, bloc) cell where **at least 2 independent (non-flagged)
experts** gave a `direction` other than `NO_PREDICTION`, compute:

- **Fleiss' κ** across all such cells, treating `direction` as a 3-category
  nominal judgment (`DEM_GAIN` / `REP_GAIN` / `NO_EFFECT`); `NO_PREDICTION`
  responses are excluded from that rater's contribution to that cell (not
  coded as a 4th category — an abstention isn't a judgment to agree or
  disagree with).
- **Mean pairwise percent agreement** as a more interpretable companion
  number (fraction of expert pairs, across all scored cells, whose `direction`
  values match exactly).

**Pre-registered interpretation** (standard Landis & Koch 1977 bands, not
invented for this project): κ < 0.20 = none/slight, 0.21-0.40 = fair,
0.41-0.60 = moderate, 0.61-0.80 = substantial, 0.81-1.00 = almost perfect.

**Pre-registered threshold: κ ≥ 0.40 (moderate or better) is required before
proceeding to any model-vs-consensus comparison.** If κ < 0.40, **that is
the headline finding, reported as such**: the experts do not agree with each
other enough for their consensus to function as ground truth, independent of
whatever the model predicts. Do not compute or report a model-vs-consensus
number in that case — it would be comparing the model against noise dressed
up as an answer key. This ordering (inter-expert agreement checked and
gated on, before anything about the model is examined) is the whole point of
specifying this in advance.

### 5.2 Consensus construction (only if §5.1's threshold is met)

For each (shock, bloc) cell with ≥2 non-abstaining, non-flagged experts:
- If **≥60% of non-abstaining experts** agree on the same `direction` value,
  that value is the **consensus direction** for the cell.
- Otherwise, the cell is **`SPLIT — no consensus`** and is excluded from
  model scoring (§5.3 covers how splits are still reported, not silently
  dropped from the write-up).
- Cells with 0 or 1 non-abstaining expert get no consensus (`INSUFFICIENT_DATA`)
  and are excluded the same way, with the reason recorded.

Magnitude consensus (secondary, exploratory only — see §2): report the
**median magnitude bucket** among experts who gave a non-blank bucket, for
cells with a direction consensus; no threshold is pre-registered for
magnitude agreement, consistent with treating magnitude as exploratory.

### 5.3 Handling splits and declines (explicit, not swept in)

- **`SPLIT` cells** (no ≥60% majority) are reported as their own category in
  every output — e.g. "of 96 scorable cells, 61 had consensus, 24 were split,
  11 had insufficient data." A high split rate is itself informative (it
  says the blocs/shocks with splits are ones where expert judgment itself is
  genuinely uncertain) and must be surfaced, not folded into a denominator
  that makes the model's score look better or worse than it is.
- **Non-`NO_PREDICTION` cells from a single expert** never get "topped up"
  or imputed from other cells/blocs — a missing response stays missing.

### 5.4 Model-vs-expert-consensus agreement (the primary result)

For every (shock, bloc) cell with a consensus direction (§5.2), score the
model's predicted `sign(deltas_{race,religion,gender}[bloc])` (party-sign-flip
applied per §2) against the consensus direction. Compute:

- **Direction accuracy**: fraction of consensus cells where the model's sign
  matches. Computed only over cells where consensus itself is directional
  (`DEM_GAIN` or `REP_GAIN`) — `NO_EFFECT`-consensus cells are reported
  separately (does the model correctly predict *no* effect where experts
  agree there is none?) rather than folded into the same accuracy number,
  since getting a null case right is a different skill from getting a
  directional case right.
- **Pre-registered validation threshold**: **≥70% direction accuracy on
  directional-consensus cells "validates"; 55-70% is "weak/inconclusive —
  report but do not claim validation"; <55% (not distinguishable from a coin
  flip on a binary sign call) "does not validate."** 70% (not the more
  conventional 80%) was chosen deliberately high above the 50% chance
  baseline given how few consensus cells this design is likely to produce
  (12 shocks × 15 blocs, filtered through §5.1-5.3 — probably well under 100
  scorable cells, plausibly under 50): a small-N result needs a wide margin
  above chance to be convincing at all, and a threshold barely above 50%
  would be noise-dominated at this scale.
- **§5.4 secondary check — consistency with existing survey layers**:
  for the 9 shocks in this set that DO have partial panel or CES coverage,
  separately report whether the expert consensus direction agrees with the
  corresponding `panel_deltas.json`/`ces_deltas.json` cell's sign (where that
  cell itself isn't suppressed). This is not a validation gate — survey layers
  disagreeing with elicitation is exactly as informative as the panel-vs-exit-poll
  disagreement already found in Step 1.3, not a reason to distrust either — but
  it's a cross-check worth reporting for every shock where it's available, per
  the original brief.

---

## 6. What "done" looks like when this is executed

1. `docs/elicitation/responses/{expert_code}_responses.csv` exists for ≥3
   non-flagged experts (+ optionally Juhi's and Espinosa's, separately
   flagged), each locked per §3.
2. `pre_registration_log.csv` has one row per locked file.
3. `scripts/score_elicitation.py --responses docs/elicitation/responses/
   --model-predictions <path>` is run, which:
   - refuses to proceed if any prediction file predates a lock timestamp it
     shouldn't (§3);
   - reports inter-expert κ and pairwise agreement FIRST, gated per §5.1;
   - if gated open, builds consensus (§5.2), reports split/insufficient rates
     (§5.3), and scores the model against consensus (§5.4) with the
     pre-registered thresholds applied automatically, not eyeballed after
     the fact.
