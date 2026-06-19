# Electoral Equilibrium — Synthetic Shock Event Taxonomy

Design document for the synthetic fine-tuning data pipeline. Defines the axes
along which shock events vary, so generation is *systematically balanced*
rather than skewed toward one ideological flavor or one electoral direction.

The goal: a model that accepts an arbitrary political shock and estimates
per-bloc vote-share deltas with calibrated accuracy. That requires training
data that spans the space of shocks, not 5,000 variations of one template.

---

## Why four axes

The naive instinct ("generate democratic, republican, tankie, capitalist
events") conflates two genuinely independent things:

- **What the event *is*** (its ideological character) — e.g. a far-left
  nationalization proposal, a libertarian deregulation push.
- **What the event *does*** (its electoral effect) — who it helps, who it
  hurts, which blocs it splits.

These are not the same. A far-left ("tankie") event can *hurt* Democrats with
moderates and Latinos while energizing seculars. If we tag events only by
ideological flavor and assume the flavor predicts the outcome, we train the
model on a falsehood. The whole point is for the model to learn that cause and
effect diverge.

So events are tagged on four axes, and the generator samples to fill all
combinations.

---

## Axis 1 — Event domain (what sphere)

| Code | Domain | Examples |
|------|--------|----------|
| `econ` | Economic | recession, inflation spike, tariffs, jobs report, stock crash, UBI, min wage |
| `foreign` | Foreign policy / security | war, assassination, treaty, troop deployment, refugee crisis, trade war |
| `social` | Social / cultural | SCOTUS ruling, abortion, guns, immigration, LGBTQ rights, affirmative action |
| `candidate` | Candidate / campaign | nominee change, VP pick, debate, gaffe, health scare, endorsement |
| `institutional` | Institutional / legal | impeachment, indictment, court packing, election law, filibuster |
| `scandal` | Scandal / corruption | financial, sexual, abuse of power, foreign entanglement |
| `exogenous` | Disaster / exogenous | pandemic, natural disaster, terrorist attack, infrastructure failure |

## Axis 2 — Ideological valence (what flavor)

This is the *character* of the event, independent of who it helps.

| Code | Valence |
|------|---------|
| `far_left` | Socialist / anti-capitalist / nationalization ("tankie") |
| `progressive` | Left-progressive (Green New Deal, reparations, M4A) |
| `center_left` | Mainstream Democratic |
| `bipartisan` | Cross-party consensus / non-ideological |
| `center_right` | Mainstream Republican |
| `conservative` | Right-conservative (tax cuts, deregulation, social conservatism) |
| `far_right` | Nationalist / authoritarian / ethnonationalist |
| `libertarian` | Pro-market / anti-state / civil-libertarian ("capitalist") |
| `populist` | Anti-elite, scrambles left-right (can be left- or right-populist) |

## Axis 3 — Electoral effect (the LABEL TARGET — what it does)

This is what the model predicts. It is NOT inferable from Axis 2 alone.

| Code | Effect |
|------|--------|
| `helps_dem` | Net positive for the Democratic ticket |
| `helps_rep` | Net positive for the Republican ticket |
| `splits` | Helps some blocs, hurts others (no clean net) |
| `realigns` | Large structural shift, crosses bloc loyalties |
| `neutral` | Minimal net effect (tests the model's restraint) |

## Axis 4 — Affected party perspective (whose ticket)

The `party` field in the schema. The same real-world event is scored
differently depending on whose campaign is being modeled.

| Code | Perspective |
|------|-------------|
| `democrat` | Modeling a Democratic ticket |
| `republican` | Modeling a Republican ticket |

---

## Coverage targets

To avoid the current skew (everything reads as "Dem progressive → minorities
slight_pos"), generation should aim for rough balance:

- **Axis 3 (effect)**: at least 20% each of helps_dem, helps_rep, splits;
  realigns and neutral can be ~10% each. *This is the most important balance* —
  the model currently can't predict Republican-favoring shocks because it
  hasn't seen them.
- **Axis 4 (party)**: ~50/50 democrat/republican perspective.
- **Axis 1 (domain)**: spread across all seven, weighted toward econ, social,
  foreign, candidate (the highest-frequency real shock types).
- **Axis 2 (valence)**: full spread including the tails (far_left, far_right,
  libertarian, populist) — these are underrepresented in mainstream training
  corpora and are where the model most needs synthetic coverage.

The generator should track counts per axis and oversample underfilled cells.

---

## A worked example of cause ≠ effect

Event: *"Democratic nominee proposes nationalizing the oil and gas industry."*

- Axis 1: `econ`
- Axis 2: `far_left`
- Axis 3: `splits` — NOT `helps_dem`
- Axis 4: `democrat`

Plausible bloc deltas (for the Democratic ticket):
- secular: `slight_pos` (energized base)
- african_american: `neutral`
- white: `mod_neg` (working-class energy-state backlash)
- latino: `slight_neg` (TX/NM energy employment)
- evangelical: `mild_neg`
- delta_eff: `~ -0.04`

A naive "far-left = good for Dems" labeling would score this `helps_dem` and
teach the model the wrong lesson. The taxonomy forces the generator (and the
reviewers) to reason about effect separately from flavor.

---

## Seed events

Below are hand-designed seed events spanning the axes. The generator expands
each ~20x with varied phrasing and slight delta perturbations, then the
DeepSeek → Gemini → Opus review chain validates the labels.

Each seed is `{description, domain, valence, expected_effect, party}`. The
`expected_effect` is a *hint* for the generator and a *check* for the
reviewers — if the generated deltas contradict it badly, that's a flag.

(Seed list maintained as configs/synthetic_events.json — see that file.)
