#!/bin/bash
#SBATCH --job-name=discord-find
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=/scratch/JDamley28@cmc.edu/electoralData/logs/discord_find_%j.log
#
# Single-pass filter: streams dataset.zst via zstd | tarfile (Python),
# reads first message from each server file, matches against political keywords.
# No temp files — memory use is O(1) with respect to archive size.
#
# Submit:
#   sbatch scripts/hpc/discord_find_servers_laguna.slurm
#
# Output:
#   /scratch/JDamley28@cmc.edu/electoralData/archives/discord/target_servers.txt
#   TSV: guild_id  channel_name  keywords_matched

set -euo pipefail

HOME_DIR=/home1/JDamley28@cmc.edu
REPO_DIR=${HOME_DIR}/electoral-equilibrium
DISCORD_DIR=/scratch/JDamley28@cmc.edu/electoralData/archives/discord

mkdir -p "$DISCORD_DIR" /scratch/JDamley28@cmc.edu/electoralData/logs

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Job $SLURM_JOB_ID starting on $(hostname)"
echo "  Dataset: $DISCORD_DIR/dataset.zst"
echo "  Output:  $DISCORD_DIR/target_servers.txt"

# ── Environment ───────────────────────────────────────────────────────────────

module load conda
source /apps/conda/miniforge3/25.11.0-1/etc/profile.d/conda.sh
conda activate electoral

# ── Run ───────────────────────────────────────────────────────────────────────

python "$REPO_DIR/scripts/discord_find_servers.py" \
    --dataset "$DISCORD_DIR/dataset.zst" \
    --output  "$DISCORD_DIR/target_servers.txt" \
    --zstd    ~/.conda/envs/electoral/bin/zstd

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Complete"
echo "Matched servers: $(tail -n +2 "$DISCORD_DIR/target_servers.txt" | wc -l)"
echo "Next step: review target_servers.txt then submit discord_extract.slurm"
