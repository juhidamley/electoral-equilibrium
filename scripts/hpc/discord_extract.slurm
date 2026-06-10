#!/bin/bash
#SBATCH --job-name=discord-extract
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=/scratch/JDamley28@cmc.edu/electoralData/logs/discord_extract_%j.log
#
# Single-pass extraction of all target Discord servers from the 110 GB archive.
# Streams dataset.zst once through zstd | tar, pulling every guild JSON file
# listed in guild_ids.txt.  No array job, no repeated decompression.
#
# Prerequisites:
#   guild_ids.txt — one guild_id per line, no header.
#   Generate from target_servers.txt:
#     tail -n +2 target_servers.txt | cut -f1 > guild_ids.txt
#
# Submit:
#   sbatch scripts/hpc/discord_extract.slurm

set -euo pipefail

HOME_DIR=/home1/JDamley28@cmc.edu
DISCORD_DIR=/scratch/JDamley28@cmc.edu/electoralData/archives/discord
DATASET=${DISCORD_DIR}/dataset.zst
GUILD_IDS=${DISCORD_DIR}/guild_ids.txt
EXTRACTED_DIR=${DISCORD_DIR}/extracted
ZSTD=~/.conda/envs/electoral/bin/zstd

mkdir -p "$EXTRACTED_DIR" /scratch/JDamley28@cmc.edu/electoralData/logs

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Job $SLURM_JOB_ID starting on $(hostname)"

if [[ ! -f "$GUILD_IDS" ]]; then
    echo "ERROR: guild_ids.txt not found: $GUILD_IDS" >&2
    echo "  Generate with: tail -n +2 target_servers.txt | cut -f1 > guild_ids.txt" >&2
    exit 1
fi

N=$(wc -l < "$GUILD_IDS")
echo "  Guild IDs  : $N"
echo "  Dataset    : $DATASET"
echo "  Extract to : $EXTRACTED_DIR"

# ── Environment ───────────────────────────────────────────────────────────────

module load conda
source /apps/conda/miniforge3/25.11.0-1/etc/profile.d/conda.sh
conda activate electoral

# ── Build tar extraction list ─────────────────────────────────────────────────

# Translate guild IDs to tar member paths (./GUILD_ID.json).
# Written to a temp file so we never hit shell ARG_MAX with thousands of IDs.
EXTRACT_LIST=$(mktemp /tmp/discord_extract_XXXXXX.txt)
trap 'rm -f "$EXTRACT_LIST"' EXIT

awk '{print "./" $1 ".json"}' "$GUILD_IDS" > "$EXTRACT_LIST"
echo "  Members to extract: $(wc -l < "$EXTRACT_LIST")"

# ── Single streaming pass ─────────────────────────────────────────────────────

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Streaming extraction started"

# -f -           read archive from stdin (piped zstd output)
# -T EXTRACT_LIST  extract only the listed members (exact names, no wildcards needed)
# --ignore-failed-read  warn but continue if a listed member is absent from archive
$ZSTD -d -c "$DATASET" | \
    tar -x -C "$EXTRACTED_DIR" -f - -T "$EXTRACT_LIST" --ignore-failed-read \
    2>&1 | grep -v "^tar: Removing leading" || true

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Extraction complete"
EXTRACTED=$(find "$EXTRACTED_DIR" -name '*.json' | wc -l)
echo "Extracted: $EXTRACTED / $N files"
