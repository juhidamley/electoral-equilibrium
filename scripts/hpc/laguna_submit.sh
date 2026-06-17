#!/bin/bash
#SBATCH --job-name=electoral-finetune
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --output=/scratch/JDamley28@cmc.edu/electoralData/logs/finetune_%j.log

# USC Laguna GPU nodes have NVIDIA L40S (48GB VRAM), not A100/V100.
# QLoRA r=16 fits comfortably in 48GB; r=32 also fits if needed.
# Validate without queuing: sbatch --test-only scripts/hpc/laguna_submit.sh

set -euo pipefail

SCRATCH=/scratch/JDamley28@cmc.edu/electoralData
REPO_DIR=/home1/JDamley28@cmc.edu/electoral-equilibrium

export TRANSFORMERS_CACHE=${SCRATCH}/hf_cache
export HF_HOME=${SCRATCH}/hf_cache
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export CUDA_LAUNCH_BLOCKING=1
export OMP_NUM_THREADS=1
export CUDA_LAUNCH_BLOCKING=1

mkdir -p "${SCRATCH}/logs" "${SCRATCH}/hf_cache"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Job ${SLURM_JOB_ID} starting on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

module load conda
source /apps/conda/miniforge3/25.11.0-1/etc/profile.d/conda.sh
conda activate electoral

export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"

cd "${REPO_DIR}"
python -m electoral.llm.trainer \
      --config $REPO_DIR/configs/train_r16.json \
      --train-data $SCRATCH/finetune/train.jsonl \
      --eval-data $SCRATCH/finetune/eval.jsonl \
      --output-dir $SCRATCH/models/mistral-r16 \
      --lora-rank 16 \
      --lora-alpha 32 \
      --epochs 3 \
      --batch-size 4 \
      --grad-accum 4

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Job ${SLURM_JOB_ID} complete"
