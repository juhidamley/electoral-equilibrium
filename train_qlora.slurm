#!/bin/bash
# Phase 6 Step 6.4: NEW hybrid adapter only. This script never deploys an adapter.
# PLACEHOLDER RESOURCE PROFILE: replace every value below with the prior successful
# Hopper QLoRA job's partition/account/time/GPU profile before sbatch.
#SBATCH --job-name=electoral-hybrid-003
# PLACEHOLDER partition; sensible default from the prior A100 scaffold:
#SBATCH --partition=gpu
# PLACEHOLDER account; uncomment and replace if Hopper requires an account:
##SBATCH --account=PLACEHOLDER_ACCOUNT
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
# PLACEHOLDER wall time; sensible QLoRA default:
#SBATCH --time=12:00:00
# PLACEHOLDER GRES; sensible one-A100 default:
#SBATCH --gres=gpu:a100:1
#SBATCH --output=logs/hybrid_003_%j.out
#SBATCH --error=logs/hybrid_003_%j.err

set -euo pipefail

REPO_DIR="${HOME}/electoral-equilibrium"
TRAIN_REL="data/finetune/phase6_step6_4_hybrid_20260814/train.jsonl"
EVAL_REL="data/finetune/phase6_step6_4_hybrid_20260814/eval.jsonl"
CONFIG_REL="configs/train_mistral-r16-hybrid-0.03-20260814.json"
ADAPTER_REL="models/mistral-r16-hybrid-0.03-20260814"
CORPUS_SHA256="77f848f3d64739c45dfc4ed91112e00afe57b5cb1a1ebde2c22a0ed90d51d734"
TRAIN_SHA256="49752dd011f0739b60af4deed1a9d2ea9381d9e48799fc65e618605ef917e321"
EVAL_SHA256="5796ea238fa9c38a87261c50c4a010fcc926b1722bed5b6bc9315b630b36ae34"
CONFIG_SHA256="f933490354628e3b3b55705109b55653f68118ae279197ed10f4ea0a138efeb2"

cd "$REPO_DIR"
mkdir -p logs

# PLACEHOLDER ENVIRONMENT: replace these with the exact module/activation lines from
# the prior successful Hopper QLoRA run. Keep outlines pinned: an unbounded
# outlines>=0.0.46 can resolve to a Python-3.10-only build in an older environment.
module load python/3.11 cuda/12.1  # PLACEHOLDER: match prior successful QLoRA run
source "${HOME}/miniconda3/etc/profile.d/conda.sh"  # PLACEHOLDER path
conda activate electoral  # PLACEHOLDER environment; install outlines==0.0.46
python - <<'PY'
from importlib.metadata import version
actual = version("outlines")
if actual != "0.0.46":
    raise SystemExit(f"ENV_GATE=FAIL outlines={actual}; required exact pin is 0.0.46")
print(f"ENV_GATE=PASS outlines={actual}")
PY

export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-${REPO_DIR}/.hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
export TOKENIZERS_PARALLELISM=false

check_hash() {
    local expected="$1" path="$2" actual
    test -f "$path" || { echo "HASH_GATE=FAIL missing=$path" >&2; exit 10; }
    actual="$(sha256sum "$path" | awk '{print $1}')"
    if [[ "$actual" != "$expected" ]]; then
        echo "HASH_GATE=FAIL path=$path expected=$expected actual=$actual" >&2
        exit 11
    fi
    echo "HASH_GATE=PASS path=$path sha256=$actual"
}

check_hash "$TRAIN_SHA256" "$TRAIN_REL"
check_hash "$EVAL_SHA256" "$EVAL_REL"
check_hash "$CONFIG_SHA256" "$CONFIG_REL"

# Fail closed rather than overwrite any adapter, including a partial prior attempt.
if [[ -e "$ADAPTER_REL" ]]; then
    echo "NO_OVERWRITE_GATE=FAIL fresh output already exists: $ADAPTER_REL" >&2
    exit 12
fi
for protected in models/mistral-r16-v3 models/mistral-r16-grounded models/mistral-r16-grounded-v2; do
    [[ "$ADAPTER_REL" != "$protected" ]] || { echo "protected adapter path selected" >&2; exit 13; }
done

echo "TRAIN_START=$(date -u +%Y-%m-%dT%H:%M:%SZ) host=$(hostname) job=${SLURM_JOB_ID:-none}"
echo "base_model=mistralai/Mistral-7B-v0.3 rank=16 alpha=32 dropout=0.05 lr=0.0002 epochs=3 batch=4 grad_accum=4 seed=42 corpus_sha256=$CORPUS_SHA256"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

set +e
python -m electoral.llm.trainer \
    --config "$CONFIG_REL" \
    --backend hpc \
    --train-data "$TRAIN_REL" \
    --eval-data "$EVAL_REL" \
    --output-dir "$ADAPTER_REL" \
    --grad-accum 4
trainer_rc=$?
set -e
# trainer rc=2 means training completed but its diagnostic eval-MAE warning fired.
if [[ "$trainer_rc" -ne 0 && "$trainer_rc" -ne 2 ]]; then
    echo "TRAIN_GATE=FAIL trainer_rc=$trainer_rc" >&2
    exit "$trainer_rc"
fi
test -f "$ADAPTER_REL/adapter_config.json" || { echo "TRAIN_GATE=FAIL adapter missing" >&2; exit 14; }

export ADAPTER_REL CONFIG_REL CORPUS_SHA256 TRAIN_SHA256 EVAL_SHA256 CONFIG_SHA256 trainer_rc
python - <<'PY'
import hashlib, json, os, subprocess
from datetime import datetime, timezone
from pathlib import Path

adapter = Path(os.environ["ADAPTER_REL"])
config = json.loads(Path(os.environ["CONFIG_REL"]).read_text())
provenance = {
    "run_id": "phase6_step6_4_hybrid_0.03_20260814",
    "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    "git_commit": subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip(),
    "base_model": config["base_model"],
    "qlora": {
        "quantization": "4bit_nf4_double_quant",
        "rank": config["lora_rank"],
        "alpha": config["lora_alpha"],
        "dropout": 0.05,
        "targets": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "learning_rate": config["learning_rate"],
        "epochs": config["epochs"],
        "batch_size": config["batch_size"],
        "gradient_accumulation_steps": config["gradient_accumulation_steps"],
        "seed": config["seed"],
    },
    "corpus": {
        "sha256": os.environ["CORPUS_SHA256"],
        "scale": "delta_scale_0.03",
        "records": 382,
        "panel_materialized": 0,
        "panel_policy": "post-training calibration only",
        "probe_in_training_or_split_eval": 0,
    },
    "splits": {
        "train_records": 306,
        "eval_records": 76,
        "train_sha256": os.environ["TRAIN_SHA256"],
        "eval_sha256": os.environ["EVAL_SHA256"],
    },
    "config_sha256": os.environ["CONFIG_SHA256"],
    "trainer_returncode": int(os.environ["trainer_rc"]),
    "deployment_changed": False,
}
(adapter / "_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
PY

echo "TRAIN_GATE=PASS adapter=$ADAPTER_REL trainer_rc=$trainer_rc production_untouched=true"
