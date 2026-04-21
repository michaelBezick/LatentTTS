#!/usr/bin/env bash
#SBATCH --output=logs/slurm/%x-%j.out

REPO_ROOT="${REPO_ROOT:?REPO_ROOT not set — submit via submit_generator_comm_eval.sh}"
GENERATOR_TYPE="${GENERATOR_TYPE:-coconut}"
PRM_ID="${PRM_ID:-checkpoints/latentRM}"
DATA_PATH="${DATA_PATH:-data/gsm_valid.json}"
NUM_RETURN_SEQUENCES="${NUM_RETURN_SEQUENCES:-8}"
GENERATOR_COMMUNICATION_TYPE="${GENERATOR_COMMUNICATION_TYPE:-attention}"
GENERATOR_COMMUNICATION_CHECKPOINT="${GENERATOR_COMMUNICATION_CHECKPOINT:-outputs/coconut-generator-attention/best}"
GENERATOR_COMMUNICATION_ATTENTION_HEADS="${GENERATOR_COMMUNICATION_ATTENTION_HEADS:-4}"
GENERATOR_COMMUNICATION_TOPK="${GENERATOR_COMMUNICATION_TOPK:-2}"
GENERATOR_COMMUNICATION_EVERY="${GENERATOR_COMMUNICATION_EVERY:-1}"
RM_COMMUNICATION_TYPE="${RM_COMMUNICATION_TYPE:-none}"
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
echo "generator_communication_type: ${GENERATOR_COMMUNICATION_TYPE}"
echo "generator_communication_checkpoint: ${GENERATOR_COMMUNICATION_CHECKPOINT}"
echo "rm_communication_type: ${RM_COMMUNICATION_TYPE}"
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
    --communication_type="${RM_COMMUNICATION_TYPE}"
    --generator_communication_type="${GENERATOR_COMMUNICATION_TYPE}"
    --generator_communication_attention_heads="${GENERATOR_COMMUNICATION_ATTENTION_HEADS}"
    --generator_communication_topk="${GENERATOR_COMMUNICATION_TOPK}"
    --generator_communication_every="${GENERATOR_COMMUNICATION_EVERY}"
)

if [[ -n "${GENERATOR_COMMUNICATION_CHECKPOINT}" ]]; then
    cmd+=(--generator_communication_checkpoint="${GENERATOR_COMMUNICATION_CHECKPOINT}")
fi

printf 'Running: %s\n' "${cmd[*]}"
srun "${cmd[@]}"
