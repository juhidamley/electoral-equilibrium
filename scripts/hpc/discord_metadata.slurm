#!/bin/bash
#SBATCH --job-name=discord_metadata
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=/hopper/home/jdamley28/discord/logs/metadata_%j.log
#SBATCH --dependency=afterok:$DISCORD_DOWNLOAD_JID

set -euo pipefail

DISCORD_DIR=/hopper/home/jdamley28/discord
DATASET=$DISCORD_DIR/dataset.zst
TMPFILE=$(mktemp /tmp/discord_head_XXXXXX.json)

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Inspecting dataset structure"

source /hopper/home/jdamley28/miniconda3/etc/profile.d/conda.sh
conda activate base

# Decompress first 100MB to inspect schema
echo "Decompressing first 100MB..."
/usr/bin/pzstd -d -c "$DATASET" 2>/dev/null | head -c 104857600 > "$TMPFILE"

echo "First 3 lines of decompressed data:"
head -n 3 "$TMPFILE"

echo ""
echo "Extracting server names and guild IDs..."

python - "$TMPFILE" "$DISCORD_DIR/server_list.txt" <<'EOF'
import sys
import json
import re

input_path = sys.argv[1]
output_path = sys.argv[2]

servers = {}  # guild_id -> server_name

# Try newline-delimited JSON first, then wrapped JSON array
with open(input_path, "r", errors="replace") as f:
    raw = f.read()

# Attempt NDJSON parse
found = 0
for line in raw.splitlines():
    line = line.strip()
    if not line:
        continue
    try:
        obj = json.loads(line)
        guild_id = str(obj.get("guild_id") or obj.get("server_id") or obj.get("guildId", ""))
        name = obj.get("guild_name") or obj.get("server_name") or obj.get("guildName", "")
        if guild_id:
            servers[guild_id] = name
            found += 1
    except json.JSONDecodeError:
        pass

# If no hits, try to find guild_id / guild_name via regex (partial/corrupt JSON)
if not servers:
    guild_ids = re.findall(r'"guild_id"\s*:\s*"?(\d+)"?', raw)
    guild_names = re.findall(r'"guild_name"\s*:\s*"([^"]*)"', raw)
    for gid, gname in zip(guild_ids, guild_names):
        servers[gid] = gname

with open(output_path, "w") as f:
    for guild_id, name in sorted(servers.items()):
        f.write(f"{guild_id}\t{name}\n")

print(f"Total distinct servers found: {len(servers)}")
print(f"Wrote server list to: {output_path}")
EOF

rm -f "$TMPFILE"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Metadata extraction complete"
echo "Total servers: $(wc -l < "$DISCORD_DIR/server_list.txt")"
echo "Review $DISCORD_DIR/server_list.txt and populate $DISCORD_DIR/target_servers.txt"
