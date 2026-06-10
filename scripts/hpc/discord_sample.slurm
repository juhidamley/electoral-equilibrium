#!/bin/bash
#SBATCH --job-name=discord_sample
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=4:00:00
#SBATCH --output=/scratch/JDamley28@cmc.edu/electoralData/logs/discord_sample_%j.log
#SBATCH --dependency=afterok:$DISCORD_EXTRACT_ARRAY_JID

set -euo pipefail

DISCORD_DIR=/scratch/JDamley28@cmc.edu/electoralData/archives/discord

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting Discord sampling pass"

module load conda
source /apps/conda/miniforge3/25.11.0-1/etc/profile.d/conda.sh
conda activate base
pip install ijson --quiet

python - "$DISCORD_DIR" <<'EOF'
import sys
import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path

discord_dir = Path(sys.argv[1])
extracted_dir = discord_dir / "extracted"
sampled_dir = discord_dir / "sampled"
sampled_dir.mkdir(exist_ok=True)

DATE_MIN = datetime(2019, 1, 1, tzinfo=timezone.utc)
DATE_MAX = datetime(2024, 11, 30, 23, 59, 59, tzinfo=timezone.utc)
MAX_PER_SERVER = 50_000
SEED = 42

random.seed(SEED)

json_files = sorted(extracted_dir.glob("*.json"))
if not json_files:
    print("No extracted JSON files found. Run discord_extract first.")
    sys.exit(1)

total_kept = 0
total_dropped_date = 0
total_dropped_empty = 0

for json_path in json_files:
    guild_id = json_path.stem
    output_path = sampled_dir / f"{guild_id}.jsonl"

    if output_path.exists():
        print(f"[{guild_id}] Already sampled — skipping.")
        continue

    print(f"[{guild_id}] Loading {json_path.name}...")
    try:
        with open(json_path, "rb") as f:
            import ijson
            raw = list(ijson.items(f, 'item'))
    except (json.JSONDecodeError, OSError) as e:
        print(f"[{guild_id}] ERROR reading file: {e}")
        continue

    if not isinstance(raw, list):
        raw = [raw]

    kept = []
    for msg in raw:
        text = (msg.get("content") or msg.get("text") or "").strip()
        if not text:
            total_dropped_empty += 1
            continue

        ts_raw = (
            msg.get("timestamp")
            or msg.get("created_at")
            or msg.get("date")
            or ""
        )
        try:
            if isinstance(ts_raw, (int, float)):
                ts = datetime.fromtimestamp(ts_raw, tz=timezone.utc)
            else:
                ts = datetime.fromisoformat(str(ts_raw).replace("Z", "+00:00"))
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
        except (ValueError, OSError, OverflowError):
            total_dropped_date += 1
            continue

        if not (DATE_MIN <= ts <= DATE_MAX):
            total_dropped_date += 1
            continue

        kept.append({
            "id": str(msg.get("id") or msg.get("message_id") or ""),
            "text": text,
            "created_at": ts.isoformat(),
            "lang": "en",
            "platform": "discord",
            "archive_id": "discord_unveiled",
            "server_id": guild_id,
        })

    if len(kept) > MAX_PER_SERVER:
        kept = random.sample(kept, MAX_PER_SERVER)

    with open(output_path, "w") as f:
        for post in kept:
            f.write(json.dumps(post, ensure_ascii=False) + "\n")

    total_kept += len(kept)
    print(f"[{guild_id}] kept={len(kept)} → {output_path.name}")

print("")
print(f"Sampling complete.")
print(f"  Total posts kept:          {total_kept:,}")
print(f"  Dropped (date filter):     {total_dropped_date:,}")
print(f"  Dropped (empty content):   {total_dropped_empty:,}")
print(f"  Output directory:          {sampled_dir}")
EOF

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Sampling pass complete"
