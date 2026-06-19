#!/usr/bin/env bash
# =============================================================================
# deploy/hpc_vllm.sh — vLLM serving of Mistral-7B + mistral-r16 on CMC Hopper
#
# STATUS: DEACTIVATED — scaffolded for future use, NOT wired into any active
# deployment path.  Activate when the vLLM latency task lands.
# See: docs/devplan.pdf §7 "HPC vLLM backend"
#
# WHY DEACTIVATED:
#   The current production path is Modal (deploy/modal_app.py), which handles
#   cold-start and GPU provisioning automatically.  The HPC vLLM backend is
#   intended as a lower-latency, zero-cost-per-request alternative once Hopper
#   allocation is confirmed and the vLLM PEFT adapter serving is validated.
#   See deploy/backend_router.py for the switchover mechanism.
#
# WHEN ACTIVATING:
#   1. Confirm Hopper A100 allocation and partition name (currently "gpu").
#   2. Sync adapter to Hopper:
#        rsync -az models/mistral-r16/ jdamley28@hopper.hpc.cmc.edu:\
#            /hopper/home/jdamley28/electoral-equilibrium/models/mistral-r16/
#   3. Install vLLM in the conda env:
#        conda activate electoral && pip install vllm>=0.6.0
#   4. Uncomment the SBATCH + python block below.
#   5. Remove this "exit 1" guard at the bottom.
#   6. Submit: sbatch deploy/hpc_vllm.sh
#   7. Expose the port via SSH tunnel or Tailscale, then set VLLM_URL in the
#      backend_router and flip INFERENCE_BACKEND=hpc_vllm.
#
# SERVING CONTRACT (when active):
#   • Endpoint: http://<hopper-node>:8080/v1/completions  (OpenAI-compatible)
#   • Model name in requests: "electoral-r16"  (the --lora-modules alias)
#   • The backend_router translates Electoral API calls to vLLM format.
#   • shock_endpoint.py does NOT call vLLM directly — it always talks to the
#     router, which handles the format translation.
#
# =============================================================================
#
# ── SBATCH directives (uncomment when activating) ────────────────────────────
#
# #!/bin/bash
# #SBATCH --job-name=electoral-vllm
# #SBATCH --partition=gpu
# #SBATCH --nodes=1
# #SBATCH --ntasks=1
# #SBATCH --cpus-per-task=8
# #SBATCH --mem=64G
# #SBATCH --gres=gpu:a100:1
# #SBATCH --time=23:00:00
# #SBATCH --output=/hopper/home/jdamley28/electoral-equilibrium/logs/vllm_%j.log
#
# set -euo pipefail
#
# REPO_DIR=/hopper/home/jdamley28/electoral-equilibrium
# ADAPTER_DIR=$REPO_DIR/models/mistral-r16
# HOST=0.0.0.0
# PORT=8080
#
# export HF_HOME=$REPO_DIR/.hf_cache
# export TRANSFORMERS_CACHE=$REPO_DIR/.hf_cache
# export TOKENIZERS_PARALLELISM=false
#
# module load cuda/13.2
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/hopper/software/cuda/13.2/lib64
#
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate electoral
#
# mkdir -p "$REPO_DIR/logs"
# echo "[$(date '+%Y-%m-%d %H:%M:%S')] Job ${SLURM_JOB_ID} starting on $(hostname)"
# nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
#
# # Serve the base model with the r=16 adapter as a named LoRA module.
# # --enable-lora + --lora-modules requires vLLM >= 0.6.0
# python -m vllm.entrypoints.openai.api_server \
#     --model mistralai/Mistral-7B-v0.3 \
#     --enable-lora \
#     --lora-modules electoral-r16="$ADAPTER_DIR" \
#     --host $HOST \
#     --port $PORT \
#     --dtype float16 \
#     --max-model-len 2048 \
#     --gpu-memory-utilization 0.85 \
#     --max-lora-rank 16 \
#     --disable-log-requests
#
# echo "[$(date '+%Y-%m-%d %H:%M:%S')] Job ${SLURM_JOB_ID}: vLLM server stopped"

# ── Active guard — remove when uncommenting the block above ──────────────────
echo "hpc_vllm.sh is DEACTIVATED (scaffolded only). See header comment." >&2
exit 1
