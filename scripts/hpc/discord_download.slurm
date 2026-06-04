#!/bin/bash
#SBATCH --job-name=discord_download
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=/hopper/home/jdamley28/discord/logs/download_%j.log

set -euo pipefail

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting Discord dataset download"

source /hopper/home/jdamley28/miniconda3/etc/profile.d/conda.sh
conda activate base

export HF_HUB_ENABLE_HF_TRANSFER=1

python - <<'EOF'
from huggingface_hub import hf_hub_download

local_path = hf_hub_download(
    repo_id="SaisExperiments/Discord-Unveiled-Compressed",
    filename="dataset.zst",
    repo_type="dataset",
    local_dir="/hopper/home/jdamley28/discord",
)
print(f"Downloaded to: {local_path}")
EOF

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Download complete"
echo "File size: $(du -sh /hopper/home/jdamley28/discord/dataset.zst | cut -f1)"
