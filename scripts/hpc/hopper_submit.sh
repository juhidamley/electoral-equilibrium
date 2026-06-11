#!/bin/bash
#SBATCH --job-name=electoral-finetune
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --output=/hopper/home/jdamley28/electoral-equilibrium/logs/finetune_%j.log

set -euo pipefail

REPO_DIR=/hopper/home/jdamley28/electoral-equilibrium
ADAPTER_DIR=$REPO_DIR/models/mistral-r16

export HF_HOME=$REPO_DIR/.hf_cache
export TRANSFORMERS_CACHE=$REPO_DIR/.hf_cache
export TOKENIZERS_PARALLELISM=false

mkdir -p "$REPO_DIR/logs" "$REPO_DIR/.hf_cache"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Job ${SLURM_JOB_ID} starting on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

source ~/miniconda3/etc/profile.d/conda.sh
conda activate electoral

export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"

cd "$REPO_DIR"
python -m electoral.llm.trainer \
  --config configs/train_r16.json \
  --train-data data/finetune/train.jsonl \
  --eval-data data/finetune/eval.jsonl \
  --output-dir models/mistral-r16-hopper \
  --lora-rank 16 \
  --lora-alpha 32 \
  --epochs 3 \
  --batch-size 4 \
  --grad-accum 4

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Job ${SLURM_JOB_ID} complete"
