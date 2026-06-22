# Electoral Equilibrium

> A stochastic optimization framework for modeling how voter coalitions must rebalance after political shocks.

**Juhi Damley** · Summer Research Program 2026 · Claremont McKenna College  
Supervised by **Prof. Gaston Espinosa** 
---

## What it does

A user types any hypothetical political event — an October surprise, a financial scandal, an assassination attempt — and the system immediately shows:

1. How a party's coalition must structurally shift across racial, religious, and gender strata to maintain a mathematical path to victory
2. The probability of achieving that equilibrium, with a 90% confidence interval

The system is prescriptive, not just predictive. It doesn't ask *who is winning*. It asks *what would have to change for someone to still win if the world shifted overnight*.

---

## How it works

Three stages in sequence, streamed to the frontend via SSE:

```
Shock text
    │
    ▼
┌─────────────────────────────────────────┐
│  Stage 1 — LLM (Mistral 7B + QLoRA)    │
│  Estimates Δμ per stratum from text     │
│  Output: delta_bins_{race,religion,     │
│          gender} + delta_eff scalar     │
└─────────────────────┬───────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────┐
│  Stage 2 — Optimizer (CVXPY DQCP)      │
│  Maximizes P(win) over race weights     │
│  Quasi-convex Sharpe-ratio objective    │
│  Output: new w* + feasibility flag      │
└─────────────────────┬───────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────┐
│  Stage 3 — Monte Carlo (Logistic-Normal │
│  ILR, N=10,000+)                        │
│  Output: P(win) point estimate +        │
│          90% CI band                    │
└─────────────────────────────────────────┘
```

---

## Demographic architecture

Three independent parallel strata. Each covers ~100% of the electorate from its own marginal tables. No cross-tabulations. No sparse intersection cells.

| Stratum | Groups | Role |
|---|---|---|
| **Race/Ethnicity** | African American, Latino, Asian American, White, Other | Optimizer decision variables — the only weights the optimizer rebalances |
| **Religion** | Evangelical, Catholic, Mainline Protestant, Secular, Jewish, Muslim, Other | Fixed from voter panel — enters loyalty estimate but not optimized |
| **Gender** | Women, Men, Other | Fixed from voter panel — same |

**Effective loyalty:**

```
μ_eff(w) = λ₁·Σ wᵢ·μᵢ_race
         + λ₂·Σ vᴿ·μᴿ_rel
         + λ₃·Σ gᴳ·μᴳ_gen
```

λ₁ + λ₂ + λ₃ = 1, calibrated from historical election data. Stored in `configs/layer_weights.json`.

**Win condition:** `μ_eff(w) ≥ V_eq`  
V_eq derived empirically from voter panel: ~0.52–0.53 for Democrats, ~0.49–0.51 for Republicans.

> **Note:** The additive formula assumes demographic identities contribute independently. This is an acknowledged approximation (ecological fallacy). MRP is the theoretically correct alternative but requires full joint cross-tab data not available in this timeline. Raking (`electoral/kernels/raking.py`) runs as an optional calibration step and the paper compares additive vs. raked outputs.

---

## Optimizer

**Objective:** maximize the probability of exceeding V_eq — not minimize variance.

```
max_w Φ( (μ̃_eff(w) - V_eq) / sqrt(λ₁²·wᵀΣ_Δw) )
subject to: Σwᵢ = 1, wᵢ ≥ 0
```

Φ = standard normal CDF. This is a quasi-convex (Sharpe-ratio form) fractional program solved via CVXPY's Disciplined Quasiconvex Programming (DQCP).

```python
problem.solve(qcp=True)
assert problem.is_dcp(qcp=True)  # must pass — required unit test
```

Minimizing variance is wrong for campaigns at a structural deficit — it flees from the only mechanism (high-variance demographic bets) that could produce a tail-event win.

---

## Monte Carlo

**Logistic-Normal distribution with ILR parameterization.** 
ILR approach:
1. Map `w*` to ILR coordinates via Helmert contrast matrix
2. Propagate `Σ_Δ` to ILR space via Jacobian
3. Draw N samples from `N(z*, Σ_ILR)` in `R^(K-1)`
4. Back-transform via softmax
5. Compute win flags → point estimate + 5th/95th percentile CI

Zero-weight blocs are reported as `infeasible_bloc`, never floored.

---

## Data sources

| Source | Platform | Demographic proxy | Access |
|---|---|---|---|
| AT Protocol archive + firehose | Bluesky | Far-left, post-Twitter migration | HuggingFace datasets + live atproto |
| Apify X scraper | X / Twitter | Urban, Catholic/Evangelical | Free tier, 500/run |
| News scraper | Christianity Today, CBN, Univision, NYT, WaPo, Fox | Outlet-specific | Intel Mac, launchd |
| Historical archives | Kavanaugh 56M, 2020 election, MeToo, Congress, SMAPP | Various | Pushshift / HuggingFace |
| USC Telegram 2024 | Telegram | Evangelical, MAGA | HuggingFace |
| Discord-Unveiled-Compressed | Discord | Young male | HuggingFace (arXiv:2502.00627) |
| Reddit (Pushshift + API) | Reddit | Subreddit-level proxies | Pushshift pre-2023 + Reddit API |
| 3DLNews2 | Local news | 14k outlets, 1995–2024 | W&M NewsLab |
| ARDA, GSS, NEP, Pew, Gallup | Voter panel | Ground truth loyalty by stratum | Academic |

**Prediction markets** (Polymarket, Kalshi, PredictIt): calibration benchmark only — not training inputs. `Δπ ≠ Δμ`: a change in win probability is not a change in vote-share margin.

---

## Setup

### Requirements

- Python 3.11+
- Node 18+ (webapp)
- [uv](https://github.com/astral-sh/uv) (recommended) or pip
- A CMC HPC account with A100 access (for fine-tuning)
- Tailscale (for Pi connectivity — mDNS blocked on CMC enterprise WiFi)

### Install

```bash
git clone https://github.com/jdamley28/electoral-equilibrium
cd electoral-equilibrium
uv sync                    # installs all Python deps from pyproject.toml
cd webapp && npm install   # installs Next.js deps
```

### Configure

```bash
cp configs/base.json configs/local.json
# Edit configs/local.json:
#   "seed": 42
#   "party": "democrat"
#   "pi_bio_server": "http://100.x.x.x:9000"   # your Pi's Tailscale IP
#   "inference_backend": "modal"                 # or "hpc" or "local"
```

### Run (development)

```bash
just dev          # starts FastAPI backend + Next.js frontend
just smoke        # runs the smoke test with a toy config
just test         # full test suite
```

See `Justfile` for all available commands.

---

## Compute stack

| Machine | Role |
|---|---|
| M5 MacBook Pro (48GB) | Interactive dev, cleaning model, Bluesky collection |
| Intel MacBook (2019) | News scraper 24/7, nightly pipeline (launchd 2am) |
| Raspberry Pi 5 + Hailo NPU | SetFit bio classifier — accessible via Tailscale |
| CMC HPC (A100) | QLoRA fine-tuning, SLURM scoring array jobs |

Syncthing ties M5, Intel Mac, and Pi together. **Never install Syncthing on the HPC.**  
HPC data transfer uses explicit `rsync` before each SLURM submission.

---

## Repo structure

```
electoral-equilibrium/
├── CLAUDE.md               # Claude Code context (read automatically)
├── DECISIONS.md            # architectural decision log
├── Justfile                # task runner
├── configs/                # all runtime config (base.json, party_config.json, etc.)
├── electoral/              # core Python package
│   ├── artifacts.py        # all frozen dataclass schemas
│   ├── config.py           # PipelineConfig + derive_seed
│   ├── pipeline.py         # Prefect DAG
│   ├── core/               # io, schema, types, rng
│   ├── kernels/            # stage internals (data, baseline, sentiment, etc.)
│   ├── nlp/                # scraper, bio_classifier, scorer, social collectors
│   ├── markets/            # prediction market collector + aggregator
│   ├── portfolios/         # CVXPY optimizer
│   ├── simulation/         # Logistic-Normal ILR Monte Carlo
│   └── api/                # FastAPI endpoint
├── webapp/                 # Next.js frontend (Vercel)
│   └── components/         # CoalitionChart, WinGauge, ShockNarrative, ...
├── collectors/             # always-on Intel Mac collectors
│   ├── bluesky.py
│   ├── apify.py
│   └── reddit.py
├── scripts/                # HPC jobs, cleaning, synthetic data generation
├── deploy/                 # Modal + HPC deployment
├── data/
│   ├── panel/              # panel_race/religion/gender.parquet
│   ├── archives/           # historical social media datasets
│   ├── finetune/           # train/eval/synthetic JSONL
│   └── audit.duckdb        # inference audit log (DuckDB, local read-only)
└── tests/                  # pytest suite
```

---

## Key architectural decisions

Full rationale for every decision is in `DECISIONS.md`. Short version:

| Decision | Choice | Why |
|---|---|---|
| Demographic structure | Parallel three-stratum | Nested cross-tabs produce sparse cells that contaminate the Monte Carlo covariance |
| Optimizer objective | Max P(win), not min-variance | Min-variance guarantees a loss when baseline loyalty is below V_eq |
| Monte Carlo distribution | Logistic-Normal ILR | Dirichlet forces negative covariances; delta method floor distorts Aitchison geometry |
| Synthetic data gate | Soft MMD weight + PCA + PCD | Hard KS gate filters out novel co-movements the model needs to learn |
| Cleaning model | Local open-weight (Qwen2.5-7B) | Gemini API has silent weight updates that break reproducibility |
| Prediction markets | Calibration only | Δπ ≠ Δμ — win probability ≠ vote-share margin |
| Pi connectivity | Tailscale | CMC enterprise WiFi blocks mDNS |

---

## Paper

This codebase is the implementation for the SRP research paper. Required methodology section deliverables:

- Formal quasiconvexity proof for the DQCP objective
- Explicit additive independence assumption statement + empirical fit test on held-out cycles
- Additive vs. raked model comparison
- MMD + PCA alignment + PCD synthetic data validation
- Ecological fallacy acknowledgment + MRP deferred rationale

---

## How to Reproduce This Run

This section documents how to reproduce the paper-baseline artifacts frozen at `artifacts/paper_baseline/`. The canonical run was executed on CMC Hopper (A100) under Python 3.11. Local development uses Python 3.9 (system) — numeric results across Python minor versions are expected to match within `atol=1e-9` for seeded operations; the 3.11 HPC run is the paper's ground truth.

### 1. Clone and install

```bash
git clone https://github.com/jdamley28/electoral-equilibrium
cd electoral-equilibrium
git checkout feature/week5-llm-finetune   # branch the paper baseline was tagged from

python3.11 -m venv .venv
source .venv/bin/activate

pip install -e ".[dev]"       # core + dev extras (pytest, black, ruff)
pip install -e ".[llm]"       # LLM stack: transformers, peft, outlines, sentencepiece
pip install -e ".[data]"      # pandas, pyarrow, vaderSentiment, zstandard
```

Verify the suite is green before proceeding:

```bash
pytest -x -q
```

### 2. Obtain the trained adapter

The LoRA adapter is **not checked into git** (binary, ~300 MB). It lives on JUHIDRIVE and the HPC:

| Location | Path |
|---|---|
| M5 MacBook (local) | `/Volumes/JUHIDRIVE/electoralData/models/mistral-r16/` |
| CMC Hopper | `/hopper/home/jdamley28/electoral-equilibrium/models/mistral-r16/` |

To copy to a new machine from the HPC:
```bash
scp -r jdamley28@hopper.hpc.cmc.edu:/hopper/home/jdamley28/electoral-equilibrium/models/mistral-r16 \
    ./models/mistral-r16
```

Then set the environment variable so inference picks it up:
```bash
export ADAPTER_PATH="$(pwd)/models/mistral-r16"
```

### 3. HPC setup (CMC Hopper, A100)

The paper baseline was run on the `main` partition (A100, 32 GB VRAM). All SLURM details are in `scripts/hpc/hopper_submit.sh`. To replicate the environment manually on a Hopper login shell:

```bash
module load cuda/13.2
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/hopper/software/cuda/13.2/lib64

source ~/miniconda3/etc/profile.d/conda.sh
conda activate electoral

export HF_HOME=/hopper/home/jdamley28/electoral-equilibrium/.hf_cache
export TRANSFORMERS_CACHE=/hopper/home/jdamley28/electoral-equilibrium/.hf_cache
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="/hopper/home/jdamley28/electoral-equilibrium:${PYTHONPATH:-}"
```

rsync data to the HPC before submitting:
```bash
rsync -avz --exclude='.venv' --exclude='rawdata' \
    ./data/ jdamley28@hopper.hpc.cmc.edu:/hopper/home/jdamley28/electoral-equilibrium/data/
```

### 4. Run the paper baseline

> **Preconditions:** The adapter must have been retrained on the augmented fine-tune dataset (post-Newsom synthetic data) and validated to fix the Newsom-2028 direction error. The `configs/paper_baseline.json` config must exist with Espinosa-confirmed `target` and `V_eq` semantics. Run `pytest -x -q` first to confirm the green clean-checkout suite.

```bash
# Once configs/paper_baseline.json exists and preconditions are met:
python -m electoral.pipeline --config configs/paper_baseline.json \
    --output-dir artifacts/paper_baseline/
```

Or via the Justfile target (once added):
```bash
just paper-baseline
```

### 5. Expected per-stage runtimes

> **Note:** Runtimes below are placeholders. They will be filled in after the first successful paper baseline run on Hopper. The run is currently blocked on the adapter retraining precondition.

| Stage | Expected runtime | Notes |
|---|---|---|
| voter_panel | ~5 s | Parquet reads, raking |
| baseline_portfolio | ~10 s | GP classifier + LOCO CV |
| sentiment_data | ~2 min | RoBERTa scoring on all archives |
| llm_finetune | ~3 h (HPC) | QLoRA on A100; skipped in `historical` mode if adapter exists |
| shock_response | ~30 s | Constrained decoding, Ledoit-Wolf |
| optimization | ~5 s | CVXPY DQCP solve |
| simulation | ~20 s | 10,000 ILR Monte Carlo draws |

If a stage hangs beyond 2× the expected time, check GPU utilization (`nvidia-smi`) and the SLURM log (`logs/finetune_<jobid>.log`).

### 6. Verify reproducibility

After producing a fresh run, compare it field-by-field against the frozen baseline:

```bash
python scripts/verify_artifacts.py artifacts/paper_baseline/ <fresh_output_dir>/
```

The script exits 0 on full match, 1 on any mismatch. All data-payload fields are compared; the only unconditionally skipped key is the envelope-level `metadata` (which may contain a creation timestamp). Any numeric diff larger than `atol=1e-9` is logged as a finding.

To adjust the float tolerance (e.g., for cross-BLAS comparison):
```bash
python scripts/verify_artifacts.py --atol 1e-6 artifacts/paper_baseline/ <fresh_dir>/
```

---

## License

Academic research use. Contact jdamley28@cmc.edu before reuse.
