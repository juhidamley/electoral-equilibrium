#!/bin/bash
#SBATCH --job-name=sample_reddit_filtered
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=/scratch/JDamley28@cmc.edu/electoralData/logs/sample_reddit_filtered_%A_%a.out
#SBATCH --error=/scratch/JDamley28@cmc.edu/electoralData/logs/sample_reddit_filtered_%A_%a.err

# Submit AFTER filter_reddit_monthly_laguna.slurm completes.
# Runs sample_archives.py scoped to reddit_monthly_filtered tasks only.
#
# Submission:
#   N=$(python scripts/sample_archives.py --list-tasks \
#         --archive-filter reddit_monthly_filtered \
#         --archive-root /scratch/JDamley28@cmc.edu/electoralData/archives | tail -1)
#   sbatch --array=0-$((N-1)) scripts/hpc/sample_reddit_monthly_laguna.slurm
#
# Or as a dependency on the filter job:
#   sbatch --dependency=afterok:<filter_job_id> --array=0-$((N-1)) \
#       scripts/hpc/sample_reddit_monthly_laguna.slurm

set -euo pipefail

HOME_DIR=/home1/JDamley28@cmc.edu
REPO_DIR=${HOME_DIR}/electoral-equilibrium
ARCHIVE_ROOT=/scratch/JDamley28@cmc.edu/electoralData/archives
OUTPUT_DIR=/scratch/JDamley28@cmc.edu/electoralData/sampled
LOG_DIR=/scratch/JDamley28@cmc.edu/electoralData/logs

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Task $SLURM_ARRAY_TASK_ID / array $SLURM_ARRAY_JOB_ID starting on $(hostname)"
echo "Archive root: $ARCHIVE_ROOT"
echo "Output dir:   $OUTPUT_DIR"

module load conda
source /apps/conda/miniforge3/25.11.0-1/etc/profile.d/conda.sh
conda activate base
pip install --quiet -r "$REPO_DIR/requirements.txt"

export HF_XET_HIGH_PERFORMANCE=1

python "$REPO_DIR/scripts/sample_archives.py" \
    --task-id "$SLURM_ARRAY_TASK_ID" \
    --archive-filter reddit_monthly_filtered \
    --archive-root "$ARCHIVE_ROOT" \
    --output-dir "$OUTPUT_DIR" \
    --target-per-shock 200000 \
    --seed 42 \
    --verbose

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Task $SLURM_ARRAY_TASK_ID complete"
echo "Output files written: $(find "$OUTPUT_DIR" -name '*.jsonl' 2>/dev/null | wc -l)"
