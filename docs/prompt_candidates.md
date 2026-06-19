# Prompt Template Candidates — Mistral 7B Electoral Fine-Tuning

**Task:** Given a shock event description, party perspective, and per-bloc RoBERTa sentiment
scores from news and social media, predict the directional change in Democratic vote share
for all 15 demographic blocs using 9-token magnitude bins, grouped by stratum.

**Bin vocabulary:** `strong_neg` `mod_neg` `mild_neg` `slight_neg` `neutral`
`slight_pos` `mild_pos` `mod_pos` `strong_pos`

**Output schema:**
```json
{
  "delta_bins_race":     {"african_american": "...", "latino": "...", "asian": "...", "white": "...", "other_race": "..."},
  "delta_bins_religion": {"evangelical": "...", "catholic": "...", "protestant": "...", "secular": "...", "jewish": "...", "muslim": "...", "other_rel": "..."},
  "delta_bins_gender":   {"women": "...", "men": "...", "other_gender": "..."},
  "delta_eff":           0.0
}
```

---

## Template A — Structured JSON Input ✅ SELECTED

### System prompt

```
You are a political science model that predicts how shock events affect
demographic bloc support for political candidates. Given RoBERTa sentiment
scores from news and social media, output delta bins for each demographic
stratum.
```

### User prompt

```
{
  "shock_id": "election_2020",
  "description": "2020 Presidential election: Biden vs Trump during
    COVID-19 pandemic. Biden campaigns on pandemic response competence,
    healthcare expansion, and racial justice. Trump campaigns on economic
    recovery and law and order.",
  "party": "democrat",
  "news_roberta_scores": {
    "african_american":  0.61,
    "latino":            0.29,
    "asian":             0.47,
    "white":             0.11,
    "other_race":        0.22,
    "evangelical":      -0.68,
    "catholic":          0.04,
    "protestant":       -0.26,
    "secular":           0.73,
    "jewish":            0.44,
    "muslim":            0.38,
    "other_rel":         0.07,
    "women":             0.54,
    "men":              -0.19,
    "other_gender":      0.51
  },
  "social_roberta_scores": {
    "african_american":  0.69,
    "latino":            0.18,
    "asian":             0.55,
    "white":             0.08,
    "other_race":        0.19,
    "evangelical":      -0.79,
    "catholic":         -0.06,
    "protestant":       -0.33,
    "secular":           0.81,
    "jewish":            0.52,
    "muslim":            0.46,
    "other_rel":         0.03,
    "women":             0.62,
    "men":              -0.28,
    "other_gender":      0.59
  }
}

Output only a JSON object with keys: delta_bins_race, delta_bins_religion,
delta_bins_gender (each a dict of bloc→bin label), and delta_eff (float).
```

### Expected output

```json
{
  "delta_bins_race": {
    "african_american": "slight_neg",
    "latino":           "mild_neg",
    "asian":            "slight_pos",
    "white":            "slight_pos",
    "other_race":       "neutral"
  },
  "delta_bins_religion": {
    "evangelical":  "mod_neg",
    "catholic":     "neutral",
    "protestant":   "slight_neg",
    "secular":      "mild_pos",
    "jewish":       "slight_pos",
    "muslim":       "slight_pos",
    "other_rel":    "neutral"
  },
  "delta_bins_gender": {
    "women":        "slight_pos",
    "men":          "slight_neg",
    "other_gender": "mild_pos"
  },
  "delta_eff": -0.008
}
```

### Pros
- Input schema mirrors the training data JSON (`ShockResponseData`) directly —
  no transformation between the scored corpus and the training file.
- All 15 scores are machine-readable at inference time; no natural-language parsing.
- Fixed, predictable prompt length (~280–320 tokens) regardless of shock complexity.
- Pure JSON output is directly constrainable by `outlines` from the first output token.
- Clear input/output boundary: everything before `[/INST]` is context; everything
  after is prediction target.

### Cons
- No reasoning trace in the output — debugging a wrong bin requires checking scores,
  not model reasoning.
- Cold inference on the base un-fine-tuned model is unreliable (see Selection §4).

---

## Template B — Narrative Framing

### System prompt

```
You are an election forecaster. Given a political shock and sentiment
signals across voter groups, predict how each demographic bloc's support
for the Democratic party will shift. Output a JSON object.
```

### User prompt

```
In the 2020 Presidential election (party: democrat), Biden faced Trump
during the COVID-19 pandemic. News coverage was strongly positive among
Black communities (+0.61) and secular voters (+0.73), weakly positive
among Latinos (+0.29) and Asian Americans (+0.47), and strongly negative
among Evangelical Christians (-0.68). Social media echoed these patterns:
secular (+0.81), evangelical (-0.79), Black (+0.69), Latino (+0.18).
Women leaned positive in both channels (+0.54 / +0.62); men leaned
negative (-0.19 / -0.28). Catholic and other-religion blocs were near
neutral (news: +0.04 / +0.07; social: -0.06 / +0.03).

Predict the delta bins for all 15 demographic blocs.
Output only a JSON object with keys: delta_bins_race, delta_bins_religion,
delta_bins_gender (each a dict of bloc→bin label), and delta_eff (float).
```

### Expected output

Same structure as Template A.

### Pros
- Natural language format is easier for the un-fine-tuned base model to follow
  from world knowledge; useful for zero-shot qualitative baselines.
- Narrative descriptions can capture context that scores alone miss (e.g., "during
  COVID-19" implies a cross-cutting shock that affects all blocs simultaneously).

### Cons
- Scores are embedded in prose — exact float values require parsing of natural
  language sentences, introducing ambiguity and inconsistency across shocks.
- Prompt length scales with shock complexity and the author's writing style.
  A major multi-issue election generates 3–4× more tokens than a simple scandal.
- Cannot be auto-generated at inference time from `RoBERTaScorer` output without
  a separate summarization step; requires human authoring of the narrative portion.
- Inconsistent formatting between training examples degrades fine-tuned model
  performance compared to a fixed schema.

---

## Template C — Chain-of-Thought

### System prompt

```
You are a political science model that predicts how shock events affect
demographic bloc support for political candidates. Given RoBERTa sentiment
scores from news and social media, output delta bins for each demographic
stratum.
```

### User prompt

```
{
  "shock_id": "election_2020",
  "description": "2020 Presidential election ...",
  "party": "democrat",
  "news_roberta_scores": { ... same as Template A ... },
  "social_roberta_scores": { ... same as Template A ... }
}

Reason step by step about each demographic group. For each group, explain
in one sentence how the scores indicate a shift, assign a bin, then output
the final JSON with keys delta_bins_race, delta_bins_religion,
delta_bins_gender, and delta_eff.
```

### Expected output (partial reasoning + final JSON)

```
african_american: News (+0.61) and social (+0.69) scores are moderately
positive but below the 2016 Obama baseline, suggesting modest enthusiasm
gap. → slight_neg

latino: Social score (+0.18) is weak despite positive news coverage (+0.29),
consistent with observed underperformance in south Florida and the Rio
Grande Valley. → mild_neg

...

{
  "delta_bins_race": { ... },
  "delta_bins_religion": { ... },
  "delta_bins_gender": { ... },
  "delta_eff": -0.008
}
```

### Pros
- Reasoning tokens may improve bin accuracy on ambiguous shocks where news and
  social scores conflict (e.g., news: +0.5, social: -0.3).
- Output is interpretable for debugging and supervisor review without reading
  raw score tables.

### Cons
- **Breaks constrained decoding.** `outlines` applies a token-level JSON grammar
  constraint from the first output token. Chain-of-thought requires free-text
  reasoning tokens before the JSON block, which are incompatible with hard
  token-level constraints unless a two-pass approach is used (unconstrained
  reasoning → constrained extraction), doubling inference cost and latency.
- Reasoning length is unpredictable: a complex multi-bloc shock can generate
  500+ reasoning tokens; a simple scandal may generate 50. This variance
  makes `max_new_tokens` impossible to set reliably and breaks batching.
- During fine-tuning, the loss signal is dominated by the reasoning tokens
  rather than the bin predictions, requiring more data to reach the same
  bin accuracy. With 500–2,000 training examples, reasoning supervision
  is not feasible.
- Training examples require gold-standard reasoning traces that do not exist
  in the current dataset.

---

## Selection

**Winner: Template A**

### 1. Schema fidelity
The JSON input structure mirrors `ShockResponseData` and the finetune JSONL schema
exactly. Training examples assembled by `scripts/prep_finetune.py` are already
structured as JSON objects with `news_roberta_scores` and `social_roberta_scores`
dicts keyed by the 15 canonical bloc IDs. Template A requires zero transformation
between the scored corpus and the training file.

### 2. Constrained decoding compatibility
`outlines` applies a token-level JSON grammar constraint from the first output token.
Template A's output is a pure JSON object with no preamble, making it directly
constrainable to the 9-bin vocabulary per value field. Templates B and C both require
free-text tokens before or during the JSON output, breaking this constraint.
With constrained decoding, the bin tokens are hard-enforced — the fine-tuned model
cannot emit `"atheist"`, `"male"`, or other out-of-vocabulary labels regardless of
what the base model would have predicted.

### 3. Input/output boundary
Template A places the full input context before `[/INST]` and expects only the JSON
target after it. This is the canonical Mistral instruction format and produces a clean
causal language modelling objective during training: loss is computed only on the
JSON completion tokens, not on the input scores.

### 4. Base model testing (2026-06-10)
Five training examples were tested on the un-fine-tuned `mistralai/Mistral-7B-v0.3`
with Template A prompts. Findings:
- The model produced valid JSON structure on 4 of 5 examples.
- **Wrong bloc names:** output used `"other_religion"`, `"atheist"`, `"male"`,
  `"female"` instead of canonical IDs (`other_rel`, `secular`, `men`, `women`).
- **`delta_eff` degenerates:** all 5 examples output `"delta_eff": 0.0`, indicating
  the base model has no learned prior for this scalar.
- **Bin vocabulary partially correct:** bins like `"negative"` and `"positive"` instead
  of `"slight_neg"` / `"slight_pos"` appeared on 3 of 5 examples.

These failures confirm two things: (a) fine-tuning is required to learn the canonical
bloc names and 9-token bin vocabulary, and (b) constrained decoding must enforce
valid bin labels at inference time. Template A's fixed JSON structure makes both
requirements straightforward to satisfy.

### Template C rejected
CoT increases output length unpredictably and breaks `outlines` constrained decoding
(see §C cons). Even without constrained decoding, the training set (500–2,000 examples)
is too small to supervise both reasoning quality and bin accuracy simultaneously.

### Template B rejected
Inline score embedding is inconsistent and cannot be auto-generated at inference time
without a separate summarization step. The variable prompt length creates training
distribution mismatch between shocks of different complexity.

---

*Template A implemented in `electoral/llm/trainer.py` as `format_prompt()` and
`format_completion()`.*
