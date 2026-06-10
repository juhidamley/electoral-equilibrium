#!/bin/bash
#SBATCH --job-name=discord_extract
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --output=/scratch/JDamley28@cmc.edu/electoralData/logs/discord_extract_%A_%a.log

# Array bounds are set at submission time:
#   sbatch --array=0-$(( $(wc -l < target_servers.txt) - 1 )) discord_extract.slurm
# Do NOT hardcode --array here so the bound stays dynamic.

set -euo pipefail

DISCORD_DIR=/scratch/JDamley28@cmc.edu/electoralData/archives/discord
DATASET=$DISCORD_DIR/dataset.zst
TARGET_FILE=$DISCORD_DIR/target_servers.txt

# Read the guild ID for this array task (skip comment/blank lines)
mapfile -t GUILD_IDS < <(grep -v '^\s*#' "$TARGET_FILE" | grep -v '^\s*$' | awk '{print $1}')

if [[ $SLURM_ARRAY_TASK_ID -ge ${#GUILD_IDS[@]} ]]; then
    echo "SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID exceeds guild ID list length ${#GUILD_IDS[@]}. Exiting."
    exit 1
fi

GUILD_ID="${GUILD_IDS[$SLURM_ARRAY_TASK_ID]}"
OUTPUT=$DISCORD_DIR/extracted/${GUILD_ID}.json

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Task $SLURM_ARRAY_TASK_ID — guild $GUILD_ID"

if [[ -f "$OUTPUT" ]]; then
    echo "Output already exists: $OUTPUT — skipping."
    exit 0
fi

module load conda
source /apps/conda/miniforge3/25.11.0-1/etc/profile.d/conda.sh
conda activate base

echo "Extracting guild $GUILD_ID from $DATASET..."

/usr/bin/pzstd -d -c "$DATASET" 2>/dev/null | python - "$GUILD_ID" "$OUTPUT" <<'EOF'
import sys
import json

guild_id = sys.argv[1]
output_path = sys.argv[2]

messages = []
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        obj = json.loads(line)
        if str(obj.get("guild_id") or obj.get("server_id") or obj.get("guildId", "")) == guild_id:
            messages.append(obj)
    except json.JSONDecodeError:
        continue

with open(output_path, "w") as f:
    json.dump(messages, f)

print(f"Extracted {len(messages)} messages for guild {guild_id}")
EOF

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Extraction complete: $OUTPUT"
echo "File size: $(du -sh "$OUTPUT" | cut -f1)"
