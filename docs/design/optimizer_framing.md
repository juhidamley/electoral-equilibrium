# What the optimizer stage computes — canonical framing

**Status:** canonical, as of Step 3.1 (2026-07-31). Supersedes every earlier
description of the optimizer as a per-shock coalition-shift recommender —
see the audit list at the bottom of this document for exactly where that
older framing still lives in the repo, website, and paper drafts.
Decision record: `DECISIONS.md`, "Step 3.1 — optimizer framing (Option C)."

This document is written to be lifted verbatim — unreworded — into the
paper, the website copy, and the README. Where those three need different
lengths, pull a prefix of this text; do not rephrase it independently in
each place, or the framing will drift again the way it did the first time.

---

## 1. What the equilibrium coalition is

The optimizer computes the **equilibrium coalition**: the race-bloc
weighting `w*` that maximizes a party's probability of clearing its win
threshold `V_eq`, given that party's **baseline** bloc loyalties and the
historical covariance structure between blocs (`Σ_Δ`), subject to a
demographic-plausibility constraint — each bloc's weight is bounded to
`[0.5×, 1.5×]` of its actual electorate share.

This is computed **once per party**, not once per shock. It answers: *given
who actually supports this party and by how much, what is the most
defensible strategic emphasis across race blocs, within demographically
realistic bounds?* It is a property of the party's standing coalition, not
of any single hypothetical event.

## 2. Why it's stable

The equilibrium coalition is **stable against single-shock perturbations**
by measured, not assumed, fact. The gap between blocs' baseline loyalties
(e.g. ~0.94 for the highest-loyalty bloc vs. ~0.45 for the lowest, under
Democratic framing) is far larger than any delta a single shock can
plausibly produce at the panel-measured scale (±0.03). Bisection on the
optimizer directly (`solve_rebalanced`) found that coalition weights first
move at a delta magnitude of **~0.0633** — while the hard clip ceiling on
any shock's per-bloc delta is **0.0375**, a full **1.7× below** the
movement threshold and unreachable at any legal intensity. This was
confirmed three independent ways: adversarial synthetic construction (every
bloc pinned to its most extreme bin, intensity multipliers up to 10×), real
model inference run at the API's maximum legal intensity, and both party
framings. In every case, the equilibrium coalition did not move.

## 3. How shocks are evaluated against the equilibrium

A shock is not asked to move the equilibrium coalition. It is evaluated
**against** the fixed equilibrium coalition, through two quantities that
*do* respond to it:

- **`mu_eff_shifted`** — the equilibrium coalition's effective loyalty,
  recomputed with the shock's predicted per-bloc loyalty shifts applied.
  This moves proportionally with the shock (measured: mean movement ratio
  ≈0.25, matching the delta-rescale factor almost exactly — this stage is
  linear).
- **Win probability**, from the Monte Carlo simulation over that same fixed
  coalition. This is where the real, nonlinear signal lives: under a
  baseline with a tight enough margin over `V_eq` (demonstrated under
  Republican framing), a shock produces genuine, non-degenerate movement in
  win probability — mean movement 0.0503, with a real (non-zero-width)
  confidence interval of roughly 0.012–0.015 across tested shocks. Under a
  baseline with a large margin (Democratic framing, in the currently
  configured baseline), win probability saturates near its ceiling and a
  single shock's effect on it is not visible — a property of the *margin*,
  not of the optimizer being unresponsive.

**The chain is: shock → per-bloc deltas → `mu_eff_shifted` (linear
response) → win probability (nonlinear response, margin-dependent) — all
evaluated against one fixed equilibrium coalition.** The coalition weights
themselves are not part of that responsive chain, by design.

---

## What this system does NOT claim

Stated plainly, because the earlier framing implied otherwise in several
places (see the audit list below):

- **The system does not recommend a per-shock coalition change.** It does
  not answer "given this shock, how should the coalition shift?" There is
  no such recommendation to give — the equilibrium coalition is, by
  measured fact, the same coalition before and after the shock, within the
  demographically plausible bound the model enforces.
- **Coalition weights are not a shock-response output.** Any UI element,
  paper sentence, or docstring that pairs "optimizer weights" with a
  specific shock as if the weights were computed *for* that shock is
  describing something the system does not do.
- **"Optimizer-recommended emphasis," "strategic weighting for this shock,"
  and similar phrasing are inaccurate as currently worded.** The emphasis
  is real and the math behind it is real, but it is a property of the
  party's baseline coalition, not a recommendation conditioned on the
  hypothetical the user just typed in.
- **Win probability and `mu_eff_shifted` are the shock-conditional
  outputs.** Only these should be described as responding to "this shock."

---

## The finding, stated positively

**Under demographically plausible coalition constraints, the optimal
coalition is stable against single-shock effects.**

This is a substantive result about electoral coalitions, not a limitation
of the model to disclose apologetically. A campaign's strategically
defensible emphasis across race blocs — bounded to stay within a realistic
band of each bloc's actual electorate share — does not rationally lurch in
response to any single news event; it is anchored by the standing structure
of who supports the party and by how much. A shock-responsive optimizer,
one whose recommended coalition emphasis swung with every hypothetical
event, would have been the less interesting and less defensible result: it
would have implied that a demographically bounded, risk-adjusted strategic
allocation should be re-litigated after every scandal or endorsement, which
is not a claim this project can or should make. Stability here is the
finding, and it is a more interesting one than the alternative would have
been.

---

## Decision record

See `DECISIONS.md`, "Step 3.1 — optimizer framing (Option C)," for the full
decision log entry: the empirical basis, the alternatives considered (in
particular, why widening the demographic-plausibility bounds to force
weight movement was rejected), and next steps.

---

## Audit list — places using the old framing (worklist for Steps 3.2/3.3)

**Not edited in this step.** This is a discovery pass only — file and line,
so the next steps have a concrete list. Grouped by surface.

### Website (`webapp/`)

| File : line | Current text | Why it's in scope |
|---|---|---|
| `webapp/app/layout.tsx:11` | Site meta description: "...models how a party's winning voter coalition must **rebalance** after a hypothetical political shock." | The single most-visible framing statement on the site (SEO description, OpenGraph). States per-shock coalition rebalancing directly. |
| `webapp/app/page.tsx:15` (page `<title>` inherited from layout) — see also line 3 tagline area | "modeling voter-coalition shifts after political shocks" | Same framing, page-level title. |
| `webapp/app/page.tsx:107-112` | `dt`: "Coalition emphasis (w̃)." / `dd`: "The optimizer's **recommendation** for how heavily the campaign should lean on each bloc to stay above the win threshold — this is a **strategic weighting**..." | Core "What do these mean?" explainer copy — the most direct claim that weights are a per-shock recommendation. |
| `webapp/app/page.tsx:132-135` | "Coalition emphasis reflects the optimizer's math and is not currently constrained to realistic demographic shares; treat it as relative strategic weighting..." | **Also factually stale, independent of the framing issue**: the optimizer *is* constrained (`WEIGHT_LOWER_MULT`/`WEIGHT_UPPER_MULT` in `cvx.py`) — that constraint is the entire mechanism behind the Step 2.2/2.3 finding. This line should be corrected for accuracy, not just reframed. |
| `webapp/components/CoalitionChart.tsx:12` | Header comment: "RIGHT: coalition emphasis (w̃, [0,1]) — from 'equilibrium' event" | Describes weights as arriving per-event (per-shock) in the SSE stream; worth checking whether the underlying event actually is re-sent per shock even though its value doesn't change, which would itself be worth noting in the fix. |
| `webapp/components/CoalitionChart.tsx:120` | Tooltip: "Coalition emphasis: **{weight}%**" shown alongside per-bloc, per-shock loyalty-shift tooltip content | Presented side-by-side with shock-conditional numbers without distinguishing that this one isn't. |
| `webapp/components/CoalitionChart.tsx:404-407` | `"No feasible coalition path under this shock."` | Feasibility genuinely can vary with the shock (a target can become unreachable within bounds) — likely fine as-is, but sits directly beside the mislabeled panel below it and should be reviewed together. |
| `webapp/components/CoalitionChart.tsx:423-424` | `title="Optimizer-recommended emphasis"` / `subtitle="Strategic weighting (w̃) — not population share"` | **The exact phrase named in the brief.** Primary UI label needing a rewrite. |
| `webapp/components/ShockInput.tsx:75` | "Whose coalition the optimizer **rebalances**." | Form-field helper text, states per-shock rebalancing directly. |
| `webapp/components/PercentileStrip.tsx:159-165` | "...A wide bar means there's more uncertainty about how heavily to lean on that bloc; a narrow bar means **the recommendation is stable**." | Frames Monte Carlo spread on coalition weight as recommendation stability — should instead be framed as (in)stability of the *estimate* of a fixed quantity, not of a per-shock recommendation. |

### README (`README.md`)

| Line | Current text |
|---|---|
| 3 | Tagline: "A stochastic optimization framework for modeling how voter coalitions **must rebalance after political shocks**." |
| 13 | Goal #1: "How a party's coalition **must structurally shift** across racial, religious, and gender strata to maintain a mathematical path to victory" |
| 15 | "The system is **prescriptive**, not just predictive. It doesn't ask *who is winning*. It asks *what would have to change* for someone to still win if the world shifted overnight." — this line in particular is the project's own stated self-conception and needs the most careful rewrite, not a word-swap. |
| 60 | Table cell: "Optimizer decision variables — the only weights the optimizer **rebalances**" — accurate in isolation (race *is* the only stratum the optimizer treats as a decision variable) but sits in a table whose surrounding rows reinforce the per-shock framing. |
| 248 | Decision-log table row restating the optimizer's objective — accurate on the math, worth a one-line addition noting the equilibrium/stability framing rather than a rewrite. |

### Paper drafts

| File : line | Current text |
|---|---|
| `research_outline.tex:32-33` (abstract) | "...a three-stage framework that estimates how a party's voter coalition **must rebalance** after a hypothetical political shock. ... a quasi-convex optimizer **recovers the coalition weights that maximize win probability**..." | Abstract-level claim; highest-visibility text in the paper. |
| `research_outline.tex:47` (Introduction) | "...demographic blocs shift their support unevenly, so a campaign's *optimal* coalition emphasis **changes after a shock**." | States the exact claim Step 2.2/2.3 falsified for coalition weights specifically (mu_eff and win probability *do* change; weights do not). |
| `research_outline.tex:301-303` | "...the optimizer **concentrates emphasis differently per party**..." | This one is about *party*-level differences (confirmed real and distinct from the shock-invariance finding — democrat vs. republican framing land at different corners). Likely needs only a clarifying addition distinguishing "differs by party" from "does not differ by shock," not a full rewrite. |
| `research_summary.tex:25-28` | "...after a shock a party's *optimal* coalition emphasis **shifts**. Electoral Equilibrium asks a single question: given a hypothetical shock..., **what race-bloc coalition weighting best keeps a party above its win threshold**..." | Two-page summary's framing sentence — same issue as the abstract, needs to lead with equilibrium/stability instead. |

### Internal code docstrings (lower priority — precise to a reader who already knows the pipeline, but reinforce the old framing to anyone skimming)

| File : line | Current text |
|---|---|
| `electoral/stages.py:150` | `"""Week 5: CVXPY DQCP optimizer → rebalanced coalition weights."""` |
| `electoral/artifacts.py:739, 751, 754` | `EquilibriumData` docstring: `"Rebalanced voter coalition after shock..."`; field comments `"rebalanced coalition weight"`, `"rebalanced μ̃_eff"` |
| `electoral/optimization/cvx.py:46-57` | `WEIGHT_LOWER_MULT`/`WEIGHT_UPPER_MULT` rationale comment — describes *why the bounds exist* (still valid under Option C) but is the natural place to add a one-line cross-reference to this document, since it's the exact mechanism this whole document is about. Also still marked **"PROVISIONAL... pending advisor review"** — Option C means these stay as-is; see the decision log entry for what that implies for this comment. |

**Not included** (checked and excluded as false positives — a different,
unrelated sense of "rebalance"): `research_summary.tex` and
`research_outline.tex`'s several references to the **synthetic training
corpus** being "rebalanced" (label-imbalance correction in the fine-tuning
data), and `configs/synthetic_events.json`'s `"rebalance_note"` (same
sense). These describe data preprocessing, not the optimizer's coalition
output, and should not be touched by this framing fix.
