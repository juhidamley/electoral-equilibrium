#!/bin/bash
#SBATCH --job-name=filter_reddit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --output=/hopper/home/jdamley28/electoral/logs/filter_reddit_%A_%a.log

# Array bounds are set at submission time:
#   N=$(python scripts/filter_reddit_monthly.py --list-tasks --input-dir $INPUT_DIR | tail -1)
#   sbatch --array=0-$((N-1)) scripts/hpc/filter_reddit_monthly.slurm
# Do NOT hardcode --array here — the count changes as more monthly dumps are added.

set -euo pipefail

REPO_DIR=/hopper/home/jdamley28/electoral-equilibrium
INPUT_DIR=/hopper/scratch/jdamley28/reddit_monthly
OUTPUT_DIR=/hopper/home/jdamley28/reddit_monthly_filtered

mkdir -p "$OUTPUT_DIR" "/hopper/home/jdamley28/electoral/logs"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Task $SLURM_ARRAY_TASK_ID / array $SLURM_ARRAY_JOB_ID starting"
echo "Input:  $INPUT_DIR"
echo "Output: $OUTPUT_DIR"

source /hopper/home/jdamley28/miniconda3/etc/profile.d/conda.sh
conda activate base
pip install --quiet -r "$REPO_DIR/requirements.txt"

export HF_XET_HIGH_PERFORMANCE=1

python "$REPO_DIR/scripts/filter_reddit_monthly.py" \
    --task-id "$SLURM_ARRAY_TASK_ID" \
    --input-dir "$INPUT_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --verbose

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Task $SLURM_ARRAY_TASK_ID complete"
