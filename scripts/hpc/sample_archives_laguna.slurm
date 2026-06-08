#!/bin/bash
#SBATCH --job-name=sample_archives
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=/scratch/JDamley28@cmc.edu/electoralData/logs/sample_%A_%a.log

# Array bounds are set at submission time via Justfile:
#   just sample-laguna
# which computes N from shocks.json and runs:
#   sbatch --array=0-$((N-1)) scripts/hpc/sample_archives_laguna.slurm
# Do NOT hardcode --array here — the count changes as shocks.json grows.

set -euo pipefail

HOME_DIR=/home1/JDamley28@cmc.edu
REPO_DIR=${HOME_DIR}/electoral-equilibrium
ARCHIVE_ROOT=/scratch/JDamley28@cmc.edu/electoralData/archives
OUTPUT_DIR=/scratch/JDamley28@cmc.edu/electoralData/sampled
LOG_DIR=/scratch/JDamley28@cmc.edu/electoralData/logs

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Task $SLURM_ARRAY_TASK_ID / array $SLURM_ARRAY_JOB_ID starting on $(hostname)"

module load conda
source /apps/conda/miniforge3/25.11.0-1/etc/profile.d/conda.sh
conda activate base
pip install --quiet -r "$REPO_DIR/requirements.txt"

export HF_XET_HIGH_PERFORMANCE=1

python "$REPO_DIR/scripts/sample_archives.py" \
    --task-id "$SLURM_ARRAY_TASK_ID" \
    --archive-root "$ARCHIVE_ROOT" \
    --output-dir "$OUTPUT_DIR" \
    --target-per-shock 200000 \
    --seed 42 \
    --verbose

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Task $SLURM_ARRAY_TASK_ID complete"
echo "Output files written: $(find "$OUTPUT_DIR" -name '*.jsonl' 2>/dev/null | wc -l)"
