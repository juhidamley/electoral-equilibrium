#!/bin/bash
#SBATCH --job-name=discord_metadata
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=/scratch/JDamley28@cmc.edu/electoralData/logs/discord_metadata_%j.log
#SBATCH --dependency=afterok:$DISCORD_DOWNLOAD_JID

set -euo pipefail

DISCORD_DIR=/scratch/JDamley28@cmc.edu/electoralData/archives/discord
DATASET=$DISCORD_DIR/dataset.zst
ZSTD=~/.conda/envs/electoral/bin/zstd

mkdir -p "$DISCORD_DIR"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Listing archive members"

module load conda
source /apps/conda/miniforge3/25.11.0-1/etc/profile.d/conda.sh
conda activate electoral

# Stream the tar index — no full decompression needed.
# Format: ./GUILD_ID.json  (one file per server)
$ZSTD -d -c "$DATASET" | tar -t | grep "\.json$" > "$DISCORD_DIR/all_servers.txt"
echo "Total servers: $(wc -l < "$DISCORD_DIR/all_servers.txt")"

# Extract guild IDs from filenames (./GUILD_ID.json → GUILD_ID) and write server_list.txt.
# Individual server extraction command for reference:
#   $ZSTD -d -c dataset.zst | tar -x --to-stdout ./GUILD_ID.json
python - "$DISCORD_DIR/all_servers.txt" "$DISCORD_DIR/server_list.txt" <<'EOF'
import sys
from pathlib import Path

input_path = sys.argv[1]
output_path = sys.argv[2]

guild_ids = []
with open(input_path) as f:
    for line in f:
        name = line.strip()
        if not name:
            continue
        guild_id = Path(name).stem   # "./123456789.json" -> "123456789"
        if guild_id:
            guild_ids.append(guild_id)

guild_ids.sort()
with open(output_path, "w") as f:
    for gid in guild_ids:
        f.write(gid + "\n")

print(f"Total distinct guild IDs: {len(guild_ids)}")
print(f"Wrote guild ID list to: {output_path}")
EOF

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Metadata extraction complete"
echo "Review $DISCORD_DIR/server_list.txt and populate $DISCORD_DIR/target_servers.txt"
