#!/bin/bash
#SBATCH --job-name=electoral-finetune
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
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

mkdir -p "${SCRATCH}/logs" "${SCRATCH}/hf_cache"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Job ${SLURM_JOB_ID} starting on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

module load conda
source /apps/conda/miniforge3/25.11.0-1/etc/profile.d/conda.sh
conda activate electoral

export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"

cd "${REPO_DIR}"
python -m electoral.llm.trainer --backend hpc --config configs/train_r16.json

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Job ${SLURM_JOB_ID} complete"
