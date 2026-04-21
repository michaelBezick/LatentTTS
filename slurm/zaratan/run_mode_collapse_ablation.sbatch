#!/usr/bin/env bash
#SBATCH --output=logs/slurm/%x-%j.out

REPO_ROOT="${REPO_ROOT:?REPO_ROOT not set — submit via submit_mode_collapse_ablation.sh}"
GENERATOR_TYPE="${GENERATOR_TYPE:-coconut}"
ATTENTION_PRM_ID="${ATTENTION_PRM_ID:-${PRM_ID:-outputs/coconut-soft-attention/best}}"
BASELINE_PRM_ID="${BASELINE_PRM_ID:-checkpoints/latentRM}"
ATTENTION_INTERACTION_TYPE="${ATTENTION_INTERACTION_TYPE:-attention}"
BASELINE_INTERACTION_TYPE="${BASELINE_INTERACTION_TYPE:-none}"
DATA_PATH="${DATA_PATH:-data/gsm_valid.json}"
NUM_RETURN_SEQUENCES="${NUM_RETURN_SEQUENCES:-64}"
MODEL_DTYPE="${MODEL_DTYPE:-bf16}"
SEED="${SEED:-200}"
PAIRWISE_COSINE_POOL="${PAIRWISE_COSINE_POOL:-mean}"
PAIRWISE_COLLAPSE_THRESHOLD="${PAIRWISE_COLLAPSE_THRESHOLD:-0.9}"
RESULTS_DIR="${RESULTS_DIR:-${REPO_ROOT}/logs/mode-collapse}"

mkdir -p "${REPO_ROOT}/logs/slurm"
cd "${REPO_ROOT}"

RUN_DIR="${RESULTS_DIR}/${SLURM_JOB_ID:-manual}"
mkdir -p "${RUN_DIR}"

echo "hostname: $(hostname)"
echo "repo_root: ${REPO_ROOT}"
echo "attention_prm_id: ${ATTENTION_PRM_ID}"
echo "baseline_prm_id: ${BASELINE_PRM_ID}"
echo "attention_interaction_type: ${ATTENTION_INTERACTION_TYPE}"
echo "baseline_interaction_type: ${BASELINE_INTERACTION_TYPE}"
echo "data_path: ${DATA_PATH}"
echo "num_return_sequences: ${NUM_RETURN_SEQUENCES}"
echo "seed: ${SEED}"
echo "pairwise_cosine_pool: ${PAIRWISE_COSINE_POOL}"
echo "pairwise_collapse_threshold: ${PAIRWISE_COLLAPSE_THRESHOLD}"
echo "results_dir: ${RUN_DIR}"
echo "slurm_job_id: ${SLURM_JOB_ID:-unknown}"

source ~/.bashrc.conda
conda activate latenttts

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

ATTN_LOG="${RUN_DIR}/attention.log"
BASELINE_LOG="${RUN_DIR}/baseline.log"
ATTN_JSON="${RUN_DIR}/attention_metrics.json"
BASELINE_JSON="${RUN_DIR}/baseline_metrics.json"
COMPARISON_TXT="${RUN_DIR}/comparison.txt"

echo "Running attention interaction condition..."
srun python -m src.infer_gpt2_rm \
    --generator_type="${GENERATOR_TYPE}" \
    --prm_mode=best_of_n \
    --model_dtype="${MODEL_DTYPE}" \
    --prm_id="${ATTENTION_PRM_ID}" \
    --data_path="${DATA_PATH}" \
    --num_return_sequences="${NUM_RETURN_SEQUENCES}" \
    --seed="${SEED}" \
    --communication_type="${ATTENTION_INTERACTION_TYPE}" \
    --report_pairwise_cosine=True \
    --pairwise_cosine_pool="${PAIRWISE_COSINE_POOL}" \
    --pairwise_collapse_threshold="${PAIRWISE_COLLAPSE_THRESHOLD}" \
    --pairwise_cosine_output_path="${ATTN_JSON}" | tee "${ATTN_LOG}"

echo "Running no-interaction baseline condition..."
srun python -m src.infer_gpt2_rm \
    --generator_type="${GENERATOR_TYPE}" \
    --prm_mode=best_of_n \
    --model_dtype="${MODEL_DTYPE}" \
    --prm_id="${BASELINE_PRM_ID}" \
    --data_path="${DATA_PATH}" \
    --num_return_sequences="${NUM_RETURN_SEQUENCES}" \
    --seed="${SEED}" \
    --communication_type="${BASELINE_INTERACTION_TYPE}" \
    --report_pairwise_cosine=True \
    --pairwise_cosine_pool="${PAIRWISE_COSINE_POOL}" \
    --pairwise_collapse_threshold="${PAIRWISE_COLLAPSE_THRESHOLD}" \
    --pairwise_cosine_output_path="${BASELINE_JSON}" | tee "${BASELINE_LOG}"

python - "${ATTN_JSON}" "${BASELINE_JSON}" "${COMPARISON_TXT}" <<'PY'
import json
import sys

attn_path, baseline_path, comparison_path = sys.argv[1:4]

with open(attn_path, "r", encoding="utf-8") as f:
    attn = json.load(f)
with open(baseline_path, "r", encoding="utf-8") as f:
    baseline = json.load(f)

def fmt_pct(x):
    return "n/a" if x is None else f"{x*100:.4f}%"

def fmt_float(x):
    return "n/a" if x is None else f"{x:.6f}"

attn_pair = attn.get("pairwise_cosine") or {}
base_pair = baseline.get("pairwise_cosine") or {}

lines = [
    "Mode Collapse Ablation (separate no-interaction baseline)",
    f"Attention interaction: {attn_path}",
    f"No-interaction baseline: {baseline_path}",
    "",
    f"Accuracy            | attention={fmt_pct(attn.get('accuracy'))} | baseline={fmt_pct(baseline.get('accuracy'))}",
    f"Coverage            | attention={fmt_pct(attn.get('coverage'))} | baseline={fmt_pct(baseline.get('coverage'))}",
    f"Voting Accuracy     | attention={fmt_pct(attn.get('voting_accuracy'))} | baseline={fmt_pct(baseline.get('voting_accuracy'))}",
    "",
    f"Pairwise Cos Pre    | attention={fmt_float(attn_pair.get('pre_pairwise_cos_mean'))} | baseline={fmt_float(base_pair.get('pre_pairwise_cos_mean'))}",
    f"Pairwise Cos Post   | attention={fmt_float(attn_pair.get('post_pairwise_cos_mean'))} | baseline={fmt_float(base_pair.get('post_pairwise_cos_mean'))}",
    f"Pairwise Cos Delta  | attention={fmt_float(attn_pair.get('delta_pairwise_cos_mean'))} | baseline={fmt_float(base_pair.get('delta_pairwise_cos_mean'))}",
    f"Collapse Rate Post  | attention={fmt_pct(attn_pair.get('post_collapse_rate'))} | baseline={fmt_pct(base_pair.get('post_collapse_rate'))}",
]

with open(comparison_path, "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")

print("\n".join(lines))
print(f"\nWrote comparison report to {comparison_path}")
PY
