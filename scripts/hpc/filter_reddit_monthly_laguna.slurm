#!/bin/bash
#SBATCH --job-name=filter_reddit
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=/scratch/JDamley28@cmc.edu/logs/filter_reddit_%A_%a.out
#SBATCH --error=/scratch/JDamley28@cmc.edu/logs/filter_reddit_%A_%a.err

# Array bounds are set at submission time:
#   N=$(python scripts/filter_reddit_monthly.py --list-tasks --input-dir $INPUT_DIR | tail -1)
#   sbatch --array=0-$((N-1)) scripts/hpc/filter_reddit_monthly_laguna.slurm
# Do NOT hardcode --array here — the count changes as more monthly dumps are added.

set -euo pipefail

REPO_DIR=/home/JDamley28@cmc.edu/electoral-equilibrium
INPUT_DIR=/scratch/JDamley28@cmc.edu/electoralData/archives/reddit/reddit_monthly
OUTPUT_DIR=/scratch/JDamley28@cmc.edu/electoralData/archives/reddit/reddit_monthly_filtered

mkdir -p "$OUTPUT_DIR" "/scratch/JDamley28@cmc.edu/logs"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Task $SLURM_ARRAY_TASK_ID / array $SLURM_ARRAY_JOB_ID starting"
echo "Input:  $INPUT_DIR"
echo "Output: $OUTPUT_DIR"

source /home/JDamley28@cmc.edu/miniconda3/etc/profile.d/conda.sh
conda activate base
pip install --quiet -r "$REPO_DIR/requirements.txt"

# Map SLURM_ARRAY_TASK_ID to the sorted list of bz2 files in INPUT_DIR
INPUT_FILE=$(python3 -c "from pathlib import Path; files=sorted(Path('${INPUT_DIR}').glob('*.bz2')); print(files[${SLURM_ARRAY_TASK_ID}])")

echo "Processing: $INPUT_FILE"

python "$REPO_DIR/scripts/filter_reddit_monthly.py" \
    --input-file "$INPUT_FILE" \
    --output-dir "$OUTPUT_DIR" \
    --workers 8 \
    --verbose

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Task $SLURM_ARRAY_TASK_ID complete"
