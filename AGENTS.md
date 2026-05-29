# AGENTS.md — Electoral Equilibrium

Engineering conventions, contributor discipline, and AI-tool routing for the
Electoral Equilibrium SRP (CMC, Summer 2026). This document is the contract
between contributors and the pipeline. Read it before opening a PR.

---

## 1. Contributor Conventions

### Branch naming

Every branch follows a single pattern:

```
feature/weekN-short-description
```

Examples: `feature/week1-voter-panel-loader`, `feature/week3-roberta-scorer`,
`feature/week5-cvxpy-dqcp-optimizer`.

- `weekN` is the SRP week number (1–8). No exceptions.
- `short-description` is kebab-case, under 40 characters, describes the deliverable.
- Hotfixes use `fix/weekN-description`. Documentation-only changes use `docs/description`.
- Never commit directly to `main`. Every change ships via a PR, even solo work.

### Code style

All Python code must pass the following before a PR can be merged:

```bash
just lint      # ruff check + black --check
just fmt       # ruff --fix + black (apply autofix)
```

- **Black** enforces formatting. Line length: 100 (configured in `pyproject.toml`).
- **Ruff** enforces style and import hygiene. Target: `py39` (`pyproject.toml`).
- Both are run in CI and as pre-merge checks. Do not bypass them with `# noqa` unless
  the suppression is accompanied by a comment explaining the invariant being protected.
- Type annotations are required on all public functions. Use `from __future__ import annotations`
  for forward-reference compatibility.

### Identifier conventions

These are non-negotiable. Every deviation breaks downstream validation.

| Kind | Convention | Example |
|---|---|---|
| Bloc IDs | lowercase snake_case | `african_american`, `other_rel` |
| Party field | lowercase string | `"democrat"` or `"republican"` |
| Election cycles | `int` YYYY | `2020`, not `"2020"` |
| Share/weight values | `float` in [0.0, 1.0] | `0.52`, not `52` |
| Stage names in `derive_seed` | lowercase snake_case | `"voter_panel"`, `"monte_carlo"` |

Any string that is not in `CANONICAL_RACES`, `CANONICAL_RELIGIONS`, or
`CANONICAL_GENDERS` (`electoral/core/types.py`) will be rejected by `validate()`
with a `ValueError`. Do not invent new bloc identifiers without a `DECISIONS.md` entry.

### Write-once file ownership

Each file in `rawdata/` and `artifacts/` is written by exactly one machine and
never overwritten by another. Syncthing propagates files across machines; it does
not resolve write conflicts — concurrent writes to the same file will corrupt data.

| Directory | Owner | Machine |
|---|---|---|
| `rawdata/social/` | Social media collectors | Intel Mac |
| `rawdata/articles/` | News scraper (`electoral/nlp/scraper.py`) | Intel Mac |
| `rawdata/merged/` | `merge_posts()` Prefect task | Intel Mac |
| `data/embeddings/` | RoBERTa scorer (`electoral/nlp/scorer.py`) | HPC ($SCRATCH → M5 via rsync) |
| `data/finetune/` | Fine-tune dataset assembly (`electoral/nlp/elasticity.py`) | M5 |
| `data/bio_labels/` | Bio classifier (`electoral/nlp/bio_classifier.py`) | Pi (via HTTP) / Intel Mac |
| `artifacts/` | Pipeline artifact I/O (`electoral/core/io.py`) | M5 |
| `adapters/mistral-7b-electoral/` | `scripts/hpc_submit.sh` auto-copy | HPC → M5 via Syncthing |

If you need to write a file from a non-owning machine, open a DECISIONS.md entry
first. Silence from Syncthing does not mean the conflict was resolved correctly.

---

## 2. Pull Request Discipline

### PR format

PR titles follow the commit message convention: one sentence, imperative mood,
no period. Include the week number.

```
[Week 2] Implement GP classifier baseline and V_eq derivation
[Week 3] Add RoBERTa embedding cache and bloc-weighted aggregation
[Fix] Correct ILR back-transform in Logistic-Normal Monte Carlo
```

### Mandatory checklist

Every PR description must include this checklist, completed honestly.
A PR with unchecked boxes will not be merged without explanation.

```
- [ ] All new stochastic components derive randomness via `derive_seed` + `make_rng`
      from `electoral/core/rng.py`. No bare `np.random` calls, no `random.seed()`.
- [ ] Unit tests added for all new kernel logic. Tests live in `tests/test_*.py`.
- [ ] New code produces and consumes typed artifact dataclasses from
      `electoral/artifacts.py`. No raw dicts passed between stages.
- [ ] All new dependencies added to `pyproject.toml` with pinned minimum versions.
      Dependencies not in `pyproject.toml` are invisible to HPC and CI.
- [ ] `Justfile` updated if a new developer command was added.
- [ ] `DECISIONS.md` updated if a non-obvious architectural choice was made.
- [ ] `just smoke` completes without error on the author's machine.
- [ ] `just test` passes with no unexpected failures (skipped tests are OK;
      previously passing tests that now fail are not).
```

### Definition of "complete"

A stage is not complete when the code exists. A stage is complete when:

1. Its kernel function is implemented (not a stub that returns placeholder data).
2. Its artifact `validate()` passes on real output.
3. Its test file (`tests/test_*.py`) has at least four non-skipped tests covering
   the happy path and at least one invariant violation.
4. `just test` passes in full.
5. `just smoke` runs the stage end-to-end without raising.

Checklist items 2–5 are verified in the PR, not assumed from code review alone.

---

## 3. Reproducibility Contract

Reproducibility is not a goal — it is a hard constraint. A pipeline run that produces
different output on two invocations with the same seed is broken, regardless of whether
the numerical difference is "small."

### Seed derivation

The global seed lives in `configs/base.json` as `"seed": 42`. Every stochastic
operation in the pipeline must derive its generator from this seed using:

```python
from electoral.core.rng import derive_seed, make_rng

rng = make_rng(derive_seed(config.seed, "my_stage_name"))
```

`derive_seed(base_seed, stage_name)` computes a deterministic SHA-256-based sub-seed.
The stage name must be a stable snake_case string — changing it changes the seed.
Stage names in use:

| Stage | `stage_name` string |
|---|---|
| Voter panel | `"voter_panel"` |
| Baseline portfolio | `"baseline_portfolio"` |
| Sentiment / RoBERTa | `"sentiment_data"` |
| SetFit bio classifier | `"setfit"` |
| LLM fine-tune | `"llm_finetune"` |
| Shock response | `"shock_response"` |
| Monte Carlo | `"monte_carlo"` |
| Synthetic data generation | `"synthetic"` |

Never pass a dynamic string (timestamp, UUID, shock ID) as the `stage_name`. If a
stage needs per-shock randomness, use `derive_seed(derive_seed(config.seed, "shock_response"), shock_id)`.

### Banned patterns

```python
# BANNED: global state
np.random.seed(42)
random.seed(42)

# BANNED: bare call
x = np.random.randn(100)

# BANNED: unseeded choice
rng = np.random.default_rng()  # no seed argument

# CORRECT
rng = make_rng(derive_seed(config.seed, "monte_carlo"))
x = rng.standard_normal(100)
```

Any PR introducing a bare `np.random` call will be rejected at review.

### Artifact equality test

`tests/test_rng.py` must include a test that runs the full smoke pipeline twice
with `configs/smoke.json` and asserts that every artifact JSON produced in run 1
is byte-identical to run 2. This test is the canary for accidental non-determinism.

If `just smoke` output differs across two runs on the same machine, the pipeline is
broken. Do not paper over the difference by adjusting tolerances — find and fix the
source of non-determinism.

### HPC reproducibility note

HPC SLURM jobs (`scripts/score_array.sh`, `scripts/hpc_submit.sh`) run on
multi-GPU nodes where CUDA non-determinism may be introduced by parallel operations.
Pin `CUBLAS_WORKSPACE_CONFIG=:4096:8` and call `torch.use_deterministic_algorithms(True)`
in any HPC training script. Document any unavoidable non-determinism in `DECISIONS.md`
with the specific CUDA operation and a quantitative bound on the expected variance.

---

## 4. Agent-Based Development Protocol

### AI tool routing

This project uses two AI assistants with distinct, non-overlapping roles.
Using the wrong tool for the wrong task wastes time and produces worse results.

| Task class | Tool | Rationale |
|---|---|---|
| Paper summaries, daily briefs, boilerplate generation, literature search | **Gemini Pro** (Google AI Studio) | Free tier (60 RPM, 1k req/day), fast, sufficient for non-architectural tasks |
| CVXPY DQCP formulation and debugging | **Claude** | Deep reasoning on quasi-convexity proofs, constraint specification, and solver error diagnosis |
| QLoRA fine-tuning logic, PEFT/HuggingFace integration | **Claude** | Multi-file context needed; must understand `electoral/llm/trainer.py` in relation to `electoral/artifacts.py` |
| ILR Monte Carlo implementation and correctness | **Claude** | Mathematical correctness over Aitchison geometry; subtle covariance propagation via Jacobian |
| FastAPI async inference pipeline (`/estimate/stream` SSE) | **Claude** | `ProcessPoolExecutor` vs `ThreadPoolExecutor` routing; CVXPY thread-safety constraints |
| LaTeX methodology section, quasiconvexity proof writeup | **Claude** | Requires understanding of DECISIONS.md §Optimization and the paper's epistemic framing |
| Debugging Syncthing conflicts, SLURM job failures | **Claude** | Requires understanding of the full machine topology and write-ownership rules |

Tag Claude in a PR comment as `@claude` when requesting a review of architectural
decisions. Claude has access to `CLAUDE.md`, `DECISIONS.md`, and the full repo
context, so reference specific file paths and line numbers in your query.

### API key hygiene

- Never commit API keys, tokens, or credentials to the repository.
- Keys live in `.env` (gitignored) and are injected via environment variables.
- Vercel environment variables are managed via `vercel env pull .env.local`.
- Modal secrets are managed via `modal secret create`.
- The Pi Tailscale IP lives in `configs/base.json` as `"pi_bio_server": "http://100.x.x.x:9000"` — the `x.x.x` is intentional obfuscation for the repo; the real IP is in `.env`.
- If a key is accidentally committed, rotate it immediately. Do not assume a force-push
  is sufficient — treat the key as permanently compromised.

---

## 5. Validation as Infrastructure

Schema validation is not defensive programming — it is a first-class pipeline stage
that catches data corruption before it propagates to expensive downstream operations
(HPC GPU hours, optimizer solve time, Monte Carlo iterations).

### Principle

Every stage must validate its inputs and outputs. Validation failure must raise a
`ValueError` with a message that names the class, the field, and the violation.
Generic `assert` statements are not acceptable.

```python
# WRONG: silent or uninformative
assert sum(weights.values()) == 1.0

# WRONG: assert with message but wrong exception type
assert sum(weights.values()) == 1.0, "weights must sum to 1"

# CORRECT: informative ValueError naming class and field
if abs(sum(weights.values()) - 1.0) > 1e-9:
    raise ValueError(
        f"BaselinePortfolioData.weights must sum to 1.0 ± 1e-9; "
        f"got {sum(weights.values()):.10f}"
    )
```

Validation helpers in `electoral/core/schema.py` (`assert_shares_sum_to_one`,
`assert_required_keys`, `assert_valid_share`, etc.) exist precisely so this logic
is not duplicated. Use them; extend them when a new invariant is needed.

### Where validation runs

| Location | What is validated |
|---|---|
| `VoterPanelData.validate()` | Stratum shares sum to 1.0 ± 1e-6 per cycle; all bloc IDs canonical |
| `BaselinePortfolioData.validate()` | Weights sum to 1.0 ± 1e-9; mu values in [0, 1]; party is valid |
| `ShockResponseData.validate()` | All delta bins are valid 9-token strings; covariance is 5×5 |
| `EquilibriumData.validate()` | Weights sum to 1.0 ± 1e-9; mu_eff_shifted is a scalar float |
| `SimulationData.validate()` | win_probability_low ≤ win_probability ≤ win_probability_high |
| `StageArtifact.validate()` | stage field is a known stage name; run_key is non-empty |

Every new dataclass added to `electoral/artifacts.py` must implement `validate()`.
Every new field added to an existing dataclass must be covered by a validation rule
and a negative test in `tests/test_artifact_roundtrip.py`.

### Validation at system boundaries

Validation is mandatory at:
- The entry point of every Prefect task (validate the input artifact before doing any work).
- The exit point of every Prefect task (validate the output artifact before writing it).
- The `/estimate` FastAPI endpoint (validate the request body before touching the pipeline).
- The HPC SLURM job entry point (validate that input files exist and are non-empty
  before requesting GPU resources).

Do not validate in the middle of a function. Validate at boundaries. A function that
validates midway through has already partially mutated state, making error recovery
ambiguous.

---

## Appendix: Machine Topology Quick Reference

```
M5 MacBook Pro (48 GB)       — interactive dev, cleaning model, Claude Code,
                                Bluesky firehose collection, artifact output
Intel Mac (2019)              — always-on: news scraper (launchd 2am), social
                                media archiving, rawdata/ write owner
Raspberry Pi 5 + Hailo NPU   — SetFit bio classifier server (port 9000),
                                accessible via Tailscale at $PI_TAILSCALE_IP
CMC HPC (A100)               — QLoRA fine-tuning (scripts/hpc_submit.sh),
                                RoBERTa scoring array (scripts/score_array.sh)
```

Syncthing syncs `data/` (processed outputs only) between M5, Intel Mac, and Pi.
It does not sync `rawdata/` (too large), `adapters/` (rsync from HPC directly),
or anything under `.venv/`.

`INFERENCE_BACKEND` env var switches the `/estimate` endpoint between Modal
(default, serverless GPU) and HPC vLLM (post-SRP, when CMC external connectivity
is confirmed). Changing this variable is the only required deployment change.
