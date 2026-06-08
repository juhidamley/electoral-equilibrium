#!/bin/bash
#SBATCH --job-name=sample_archives
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=/hopper/home/jdamley28/electoral/logs/sample_%A_%a.log

# Array bounds are set at submission time via Justfile:
#   just sample
# which computes N from shocks.json and runs:
#   sbatch --array=0-$((N-1)) scripts/hpc/sample_archives.slurm
# Do NOT hardcode --array here — the count changes as shocks.json grows.

set -euo pipefail

REPO_DIR=/hopper/home/jdamley28/electoral-equilibrium
ARCHIVE_ROOT=/hopper/scratch/jdamley28/electoral/archives
OUTPUT_DIR=/hopper/scratch/jdamley28/electoral/sampled

mkdir -p "$OUTPUT_DIR" "/hopper/home/jdamley28/electoral/logs"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Task $SLURM_ARRAY_TASK_ID / array $SLURM_ARRAY_JOB_ID starting"

source /hopper/home/jdamley28/miniconda3/etc/profile.d/conda.sh
conda activate base
pip install --quiet -r "$REPO_DIR/requirements.txt"

export HF_XET_HIGH_PERFORMANCE=1

python "$REPO_DIR/scripts/sample_archives.py" \
    --task-id "$SLURM_ARRAY_TASK_ID" \
    --archive-root "$ARCHIVE_ROOT" \
    --output-dir "$OUTPUT_DIR" \
    --seed 42 \
    --verbose

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Task $SLURM_ARRAY_TASK_ID complete"
echo "Output: $(ls -lh "$OUTPUT_DIR"/**/*.jsonl 2>/dev/null | wc -l) files written so far"
