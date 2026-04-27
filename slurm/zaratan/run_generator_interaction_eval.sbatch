#!/usr/bin/env bash
#SBATCH --output=logs/slurm/%x-%j.out

REPO_ROOT="${REPO_ROOT:?REPO_ROOT not set — submit via submit_generator_interaction_eval.sh}"
GENERATOR_TYPE="${GENERATOR_TYPE:-coconut}"
PRM_ID="${PRM_ID:-checkpoints/latentRM}"
DATA_PATH="${DATA_PATH:-data/gsm_valid.json}"
NUM_RETURN_SEQUENCES="${NUM_RETURN_SEQUENCES:-8}"
SAMPLING_BY="${SAMPLING_BY:-dropout}"
NOISE_STD="${NOISE_STD:-}"
DROPOUT_P="${DROPOUT_P:-}"
GENERATOR_INTERACTION_TYPE="${GENERATOR_INTERACTION_TYPE:-gated_router}"
GENERATOR_INTERACTION_CHECKPOINT="${GENERATOR_INTERACTION_CHECKPOINT:-outputs/coconut-generator-interaction-gated-router-ce/best}"
GENERATOR_ADAPTER_CHECKPOINT="${GENERATOR_ADAPTER_CHECKPOINT:-outputs/coconut-generator-interaction-gated-router-ce/best/generator_adapter}"
GENERATOR_INTERACTION_ATTENTION_HEADS="${GENERATOR_INTERACTION_ATTENTION_HEADS:-4}"
GENERATOR_INTERACTION_TOPK="${GENERATOR_INTERACTION_TOPK:-2}"
GENERATOR_INTERACTION_GATE_BIAS="${GENERATOR_INTERACTION_GATE_BIAS:--4.0}"
GENERATOR_INTERACTION_SCALE="${GENERATOR_INTERACTION_SCALE:-1.0}"
GENERATOR_INTERACTION_EVERY="${GENERATOR_INTERACTION_EVERY:-1}"
RM_INTERACTION_TYPE="${RM_INTERACTION_TYPE:-none}"
MODEL_DTYPE="${MODEL_DTYPE:-bf16}"
SEED="${SEED:-200}"

mkdir -p "${REPO_ROOT}/logs/slurm"
cd "${REPO_ROOT}"

echo "hostname: $(hostname)"
echo "repo_root: ${REPO_ROOT}"
echo "generator_type: ${GENERATOR_TYPE}"
echo "prm_id: ${PRM_ID}"
echo "data_path: ${DATA_PATH}"
echo "num_return_sequences: ${NUM_RETURN_SEQUENCES}"
echo "sampling_by: ${SAMPLING_BY}"
echo "generator_interaction_type: ${GENERATOR_INTERACTION_TYPE}"
echo "generator_interaction_checkpoint: ${GENERATOR_INTERACTION_CHECKPOINT}"
echo "generator_adapter_checkpoint: ${GENERATOR_ADAPTER_CHECKPOINT}"
echo "rm_interaction_type: ${RM_INTERACTION_TYPE}"
echo "seed: ${SEED}"
echo "slurm_job_id: ${SLURM_JOB_ID:-unknown}"

source ~/.bashrc.conda
conda activate latenttts

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

cmd=(
    python -m src.infer_gpt2_rm
    --generator_type="${GENERATOR_TYPE}"
    --prm_mode=best_of_n
    --model_dtype="${MODEL_DTYPE}"
    --prm_id="${PRM_ID}"
    --data_path="${DATA_PATH}"
    --num_return_sequences="${NUM_RETURN_SEQUENCES}"
    --seed="${SEED}"
    --sampling_by="${SAMPLING_BY}"
    --communication_type="${RM_INTERACTION_TYPE}"
    --generator_communication_type="${GENERATOR_INTERACTION_TYPE}"
    --generator_communication_attention_heads="${GENERATOR_INTERACTION_ATTENTION_HEADS}"
    --generator_communication_topk="${GENERATOR_INTERACTION_TOPK}"
    --generator_communication_gate_bias="${GENERATOR_INTERACTION_GATE_BIAS}"
    --generator_communication_interaction_scale="${GENERATOR_INTERACTION_SCALE}"
    --generator_communication_every="${GENERATOR_INTERACTION_EVERY}"
)

if [[ -n "${GENERATOR_INTERACTION_CHECKPOINT}" ]]; then
    cmd+=(--generator_communication_checkpoint="${GENERATOR_INTERACTION_CHECKPOINT}")
fi
if [[ -n "${GENERATOR_ADAPTER_CHECKPOINT}" ]]; then
    cmd+=(--generator_adapter_checkpoint="${GENERATOR_ADAPTER_CHECKPOINT}")
fi
if [[ -n "${NOISE_STD}" ]]; then
    cmd+=(--noise_std="${NOISE_STD}")
fi
if [[ -n "${DROPOUT_P}" ]]; then
    cmd+=(--dropout_p="${DROPOUT_P}")
fi

printf 'Running: %s\n' "${cmd[*]}"
srun "${cmd[@]}"
