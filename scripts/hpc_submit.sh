#!/usr/bin/env bash
# Submit QLoRA fine-tuning job to CMC HPC (A100 node via SLURM).
# Run from the login node after rsync'ing the fine-tuning dataset:
#   sbatch scripts/hpc_submit.sh
#
# The adapter is copied to $HOME/projects/... before the job exits
# so it survives $SCRATCH wipe. Then sync to M5 via Syncthing or scp.

#SBATCH --job-name=electoral-finetune
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --output=logs/finetune_%j.out
#SBATCH --error=logs/finetune_%j.err

set -euo pipefail

# ── Environment ───────────────────────────────────────────────────────────────
module load python/3.11 cuda/12.1 2>/dev/null || true

REPO_ROOT="$HOME/projects/electoral-equilibrium"
SCRATCH_ADAPTER="$SCRATCH/adapters/mistral-7b-electoral"
DEST_ADAPTER="$HOME/projects/electoral-equilibrium/adapters/mistral-7b-electoral"

cd "$REPO_ROOT"
source .venv/bin/activate || source venv/bin/activate

# ── Idempotency check ─────────────────────────────────────────────────────────
if [ -d "$DEST_ADAPTER" ] && [ -f "$DEST_ADAPTER/adapter_config.json" ]; then
    echo "Adapter already exists at $DEST_ADAPTER — skipping training"
    exit 0
fi

# ── Fine-tuning ───────────────────────────────────────────────────────────────
echo "Starting QLoRA fine-tuning — $(date)"

python3 -m electoral.llm.trainer \
    --config configs/base.json \
    --train-data data/finetune/train.jsonl \
    --eval-data  data/finetune/eval.jsonl \
    --output-dir "$SCRATCH_ADAPTER" \
    --lora-rank 16 \
    --epochs 3

echo "Training complete — $(date)"

# ── Copy adapter to persistent storage before scratch wipe ───────────────────
mkdir -p "$DEST_ADAPTER"
cp -r "$SCRATCH_ADAPTER/." "$DEST_ADAPTER/"
echo "Adapter saved to $DEST_ADAPTER"

# ── If rank-16 underfit, prepare rank-32 job (do not auto-submit) ────────────
python3 -c "
import json, sys
with open('$DEST_ADAPTER/trainer_state.json') as f:
    state = json.load(f)
mae = state.get('eval_mae', 0)
print(f'Held-out MAE: {mae:.4f}')
if mae > 0.04:
    print('WARNING: MAE > 0.04 — consider submitting rank-32 job')
    sys.exit(2)
" || true

echo "hpc_submit.sh complete"
