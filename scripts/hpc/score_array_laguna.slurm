#!/bin/bash
#SBATCH --job-name=score_shocks
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=/scratch/JDamley28@cmc.edu/electoralData/logs/score_%A_%a.log

# Array bounds are set at submission time via Justfile:
#   just score-laguna
# which counts shock_id directories in rawdata/merged/ and runs:
#   sbatch --array=0-$((N-1)) scripts/hpc/score_array_laguna.slurm
# Do NOT hardcode --array here — the count changes as new shocks are processed.

set -euo pipefail

REPO_DIR=/home1/JDamley28@cmc.edu/electoral-equilibrium
SCRATCH=/scratch/JDamley28@cmc.edu/electoralData

MERGED_DIR=${SCRATCH}/merged
OUTPUT_DIR=${SCRATCH}/scored
CACHE_DIR=${SCRATCH}/embeddings_cache
LOG_DIR=${SCRATCH}/logs

mkdir -p "$OUTPUT_DIR" "$CACHE_DIR" "$LOG_DIR"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Task ${SLURM_ARRAY_TASK_ID} / array ${SLURM_ARRAY_JOB_ID} starting on $(hostname)"

# ── Resolve shock_id from sorted directory listing ───────────────────────────

SHOCK_ID=$(python3 -c "
import os, sys
merged = '${MERGED_DIR}'
shocks = sorted(e for e in os.listdir(merged) if os.path.isdir(os.path.join(merged, e)))
idx = ${SLURM_ARRAY_TASK_ID}
if idx >= len(shocks):
    print(f'ERROR: task index {idx} out of range (have {len(shocks)} shocks)', file=sys.stderr)
    sys.exit(1)
print(shocks[idx])
")

echo "[$(date '+%Y-%m-%d %H:%M:%S')] shock_id=${SHOCK_ID}  merged=${MERGED_DIR}/${SHOCK_ID}/posts.jsonl"

# ── Environment ───────────────────────────────────────────────────────────────

module load conda
source /apps/conda/miniforge3/25.11.0-1/etc/profile.d/conda.sh
conda activate electoral

export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"

# ── Score ─────────────────────────────────────────────────────────────────────

python "${REPO_DIR}/scripts/score_shock.py" \
    --shock-id    "${SHOCK_ID}" \
    --merged-dir  "${MERGED_DIR}" \
    --output-dir  "${OUTPUT_DIR}" \
    --cache-dir   "${CACHE_DIR}" \
    --batch-size  64 \
    --resume

# ── Summary ───────────────────────────────────────────────────────────────────

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Task ${SLURM_ARRAY_TASK_ID} (${SHOCK_ID}) complete"
if [[ -f "${OUTPUT_DIR}/${SHOCK_ID}/scored.jsonl" ]]; then
    N_SCORED=$(wc -l < "${OUTPUT_DIR}/${SHOCK_ID}/scored.jsonl")
    echo "  scored.jsonl: ${N_SCORED} lines"
fi
