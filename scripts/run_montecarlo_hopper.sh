#!/bin/bash
#SBATCH --job-name=electoral-mc
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --output=/hopper/home/jdamley28/electoral-equilibrium/logs/montecarlo_%j.log

set -euo pipefail

REPO_DIR=/hopper/home/jdamley28/electoral-equilibrium
SHOCK_ID="${SHOCK_ID:-ayatollah_assassination}"
N_SIMS="${N_SIMS:-50000}"

mkdir -p "${REPO_DIR}/logs"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate electoral

cd "$REPO_DIR"

python -m electoral.simulation.montecarlo \
    --shock-artifact       "artifacts/shock_${SHOCK_ID}.json" \
    --equilibrium-artifact "artifacts/equilibrium_${SHOCK_ID}.json" \
    --output               "artifacts/sim_${SHOCK_ID}.json" \
    --n-simulations        "$N_SIMS"

echo "Job complete: artifacts/sim_${SHOCK_ID}.json"
