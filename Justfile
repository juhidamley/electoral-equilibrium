# Electoral Equilibrium — task runner
# Install: brew install just
# Usage:   just <target>
#
# Targets overview:
#   smoke       Run pipeline end-to-end on the smoke config (fast, for local dev checks)
#   test        Run the full pytest suite with verbose output
#   score       Submit RoBERTa SLURM array job on CMC HPC
#   train       Submit QLoRA fine-tuning SLURM job on CMC HPC
#   deploy      Deploy to Modal (GPU inference) + Vercel (frontend)
#   sample      Submit stratified archive sampling SLURM array on CMC Hopper HPC
#   sample-laguna Submit same array on USC Laguna HPC
#   clean       Run LLM cleaning on sampled social/news data (M5, local open-weight model)
#   scrape      Scrape news articles for a shock event via Google News RSS (Intel Mac)
#   continuous  Run nightly incremental pipeline (post-SRP mode, launchd @ 2am)

# ── Local development ────────────────────────────────────────────────────────

# Run pipeline on smoke config (completes in <10 seconds)
smoke:
    python -m electoral.pipeline --config configs/smoke.json

# Run full test suite
test:
    pytest tests/ -v --tb=short

# Run tests with coverage report
test-cov:
    pytest tests/ -v --tb=short --cov=electoral --cov-report=term-missing

# ── HPC batch jobs ───────────────────────────────────────────────────────────

# Submit RoBERTa scoring SLURM array (scores all shock events in parallel)
score:
    rsync -avz rawdata/merged/ hpc:$SCRATCH/electoral/merged/
    rsync -avz rawdata/articles/ hpc:$SCRATCH/electoral/articles/
    sbatch scripts/score_array.sh
    @echo "Monitor with: squeue -u $USER"

# Submit QLoRA fine-tuning job (r=16 first; compare eval before submitting r=32)
train:
    rsync -avz data/finetune/ hpc:$SCRATCH/electoral/finetune/
    sbatch scripts/hpc_submit.sh
    @echo "Monitor with: squeue -u $USER && tail -f slurm-*.out"

# Submit stratified archive sampling SLURM array (draws ~5k posts per shock/archive pair)
# Archives must already be present on HPC scratch — they are downloaded directly there.
sample:
    #!/usr/bin/env bash
    set -euo pipefail
    N=$(python scripts/sample_archives.py --list-tasks | tail -1)
    echo "Submitting sampling array: $N tasks (0-$((N-1)))"
    sbatch --array=0-$((N-1)) scripts/hpc/sample_archives.slurm
    echo "Monitor with: squeue -u $USER"

# Submit same sampling array on USC Laguna HPC (partition=compute, scratch at /scratch/JDamley28@cmc.edu/)
sample-laguna:
    #!/usr/bin/env bash
    set -euo pipefail
    N=$(python scripts/sample_archives.py --list-tasks | tail -1)
    echo "Submitting Laguna sampling array: $N tasks (0-$((N-1)))"
    sbatch --array=0-$((N-1)) scripts/hpc/sample_archives_laguna.slurm
    echo "Monitor with: squeue -u $USER"

# ── Data preparation (runs locally on M5 or Windows) ─────────────────────────

# LLM cleaning: steps 1-4 (Gemini 2.0 Flash for off-topic filter; steps 2-4 deterministic)
# Requires GEMINI_API_KEY or GEMINI env var. Use --dry-run to skip Gemini (steps 2-4 only).
clean:
    python scripts/clean_with_llm.py \
        --input-dir /Volumes/JUHIDRIVE/electoralData/sampled/ \
        --output-dir /Volumes/JUHIDRIVE/electoralData/cleaned/

# Scrape news articles for a shock event via Google News RSS (runs on Intel Mac).
# Override sources with: just scrape SHOCK=dobbs_2022 SOURCES="nyt wapo reuters"
# Quick test (5 articles/source): just scrape-test SHOCK=ayatollah_assassination
SHOCK := "ayatollah_assassination"
SOURCES := ""
DATE_START := "2026-02-25"
DATE_END := "2026-03-15"

scrape:
    python -m electoral.nlp.scraper \
        --shock-id {{SHOCK}} \
        $([ -n "{{SOURCES}}" ] && echo "--sources {{SOURCES}}") \
        --date-range {{DATE_START}} {{DATE_END}}

scrape-test:
    python -m electoral.nlp.scraper \
        --shock-id {{SHOCK}} \
        --date-range {{DATE_START}} {{DATE_END}} \
        --test

# ── Deployment ────────────────────────────────────────────────────────────────

# Deploy backend to Modal (serverless GPU) + frontend to Vercel
deploy:
    modal deploy deploy/modal_app.py
    @echo "Set NEXT_PUBLIC_API_URL to the Modal URL, then: vercel --prod"

# Preview deploy to Vercel only (no Modal)
deploy-preview:
    vercel

# ── Post-SRP continuous mode ─────────────────────────────────────────────────

# Nightly incremental pipeline: collect new shocks, score, append to fine-tune dataset
# Triggered by launchd on Intel Mac at 2am; triggered manually here for testing
continuous:
    python -m electoral.pipeline --config configs/continuous.json

# ── Utility ──────────────────────────────────────────────────────────────────

# Format and lint
lint:
    ruff check electoral/ tests/ scripts/ collectors/
    black --check electoral/ tests/ scripts/ collectors/

# Apply autofix
fmt:
    ruff check --fix electoral/ tests/ scripts/ collectors/
    black electoral/ tests/ scripts/ collectors/

# Remove generated artifacts (keeps source data)
clean-artifacts:
    rm -rf artifacts/smoke/ artifacts/*.json artifacts/*.parquet