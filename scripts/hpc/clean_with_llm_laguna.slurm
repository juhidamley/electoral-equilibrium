#!/bin/bash
#SBATCH --job-name=electoral-clean
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/JDamley28@cmc.edu/electoralData/logs/clean_%j.log
#
# NOT an array job — clean_with_llm.py iterates over all shocks sequentially.
# Parallelisation is intentionally avoided: Gemini 2.0 Flash free tier is
# capped at 15 RPM and the script enforces 4 s sleeps between batches.
# Running concurrent jobs would immediately 429 and burn the rate limit.
#
# Submit:
#   sbatch scripts/hpc/clean_with_llm_laguna.slurm
#
# Monitor:
#   squeue -u JDamley28@cmc.edu
#   tail -f /scratch/JDamley28@cmc.edu/electoralData/logs/clean_<JOBID>.log

set -euo pipefail

HOME_DIR=/home1/JDamley28@cmc.edu
REPO_DIR=${HOME_DIR}/electoral-equilibrium
SCRATCH_ROOT=/scratch/JDamley28@cmc.edu/electoralData
INPUT_DIR=${SCRATCH_ROOT}/sampled
OUTPUT_DIR=${SCRATCH_ROOT}/cleaned
LOG_DIR=${SCRATCH_ROOT}/logs

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Job $SLURM_JOB_ID starting on $(hostname)"
echo "  Input:  $INPUT_DIR"
echo "  Output: $OUTPUT_DIR"

# ── Environment ───────────────────────────────────────────────────────────────
module load conda
source /apps/conda/miniforge3/25.11.0-1/etc/profile.d/conda.sh
conda activate electoral

# Pull GEMINI_API_KEY from ~/.bashrc (written there during Laguna onboarding).
# eval + grep is used instead of sourcing the full ~/.bashrc because sourcing
# an interactive shell config inside a batch job triggers interactive-only
# blocks (e.g. PS1 setup) that error on non-interactive nodes.
GEMINI_KEY=$(grep -E '^export GEMINI_API_KEY=' "$HOME/.bashrc" | tail -1 | sed "s/export GEMINI_API_KEY=//;s/['\"]//g")
if [ -z "$GEMINI_KEY" ]; then
    echo "ERROR: GEMINI_API_KEY not found in ~/.bashrc — aborting" >&2
    exit 1
fi
export GEMINI_API_KEY="$GEMINI_KEY"

# ── Run cleaning pipeline ─────────────────────────────────────────────────────
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting clean_with_llm.py"

python "$REPO_DIR/scripts/clean_with_llm.py" \
    --input-dir  "$INPUT_DIR"  \
    --output-dir "$OUTPUT_DIR"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Cleaning complete"
echo "Cleaned files: $(find "$OUTPUT_DIR" -name '*.jsonl' 2>/dev/null | wc -l)"
