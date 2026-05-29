#!/usr/bin/env bash
# SLURM array job: RoBERTa scoring across shock events.
# Submit from HPC login node:
#   sbatch scripts/score_array.sh
#
# Each array task scores one shock_id from configs/shocks.json.
# Output: data/embeddings/{shock_id}.parquet per task.

#SBATCH --job-name=electoral-score
#SBATCH --array=0-24           # adjust upper bound to len(shocks)-1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=logs/score_%A_%a.out
#SBATCH --error=logs/score_%A_%a.err

set -euo pipefail

# ── Environment ───────────────────────────────────────────────────────────────
module load python/3.11 cuda/12.1 2>/dev/null || true

REPO_ROOT="$HOME/projects/electoral-equilibrium"
cd "$REPO_ROOT"

source .venv/bin/activate || source venv/bin/activate

# ── Resolve shock_id for this array task ──────────────────────────────────────
SHOCK_IDS=$(python3 -c "
import json
with open('configs/shocks.json') as f:
    shocks = json.load(f)
for s in shocks:
    if s.get('active', True):
        print(s['id'])
")

SHOCK_ID=$(echo "$SHOCK_IDS" | sed -n "$((SLURM_ARRAY_TASK_ID + 1))p")

if [ -z "$SHOCK_ID" ]; then
    echo "No shock for array task $SLURM_ARRAY_TASK_ID — exiting"
    exit 0
fi

echo "Scoring shock: $SHOCK_ID (array task $SLURM_ARRAY_TASK_ID)"

# ── Run scorer ────────────────────────────────────────────────────────────────
python3 -m electoral.nlp.scorer \
    --shock-id "$SHOCK_ID" \
    --input "rawdata/merged/${SHOCK_ID}/posts.jsonl" \
    --output "data/embeddings/${SHOCK_ID}.parquet" \
    --config configs/base.json

echo "Done: data/embeddings/${SHOCK_ID}.parquet"
