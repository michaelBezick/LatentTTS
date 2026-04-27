#!/usr/bin/env bash
#SBATCH --output=logs/slurm/%x-%j.out

REPO_ROOT="${REPO_ROOT:?REPO_ROOT not set — submit via submit_generator_interaction_eval_rm_free.sh}"
GENERATOR_TYPE="${GENERATOR_TYPE:-coconut}"
DATA_PATH="${DATA_PATH:-data/gsm_valid.json}"
NUM_RETURN_SEQUENCES="${NUM_RETURN_SEQUENCES:-8}"
LATENT_LENGTH="${LATENT_LENGTH:-6}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
DO_SAMPLE="${DO_SAMPLE:-true}"
SAMPLING_BY="${SAMPLING_BY:-noise}"
NOISE_STD="${NOISE_STD:-0.1}"
DROPOUT_P="${DROPOUT_P:-0.2}"
GENERATOR_INTERACTION_TYPE="${GENERATOR_INTERACTION_TYPE:-attention}"
GENERATOR_INTERACTION_CHECKPOINT="${GENERATOR_INTERACTION_CHECKPOINT:-outputs/coconut-generator-interaction-verifiable-rl/best}"
GENERATOR_INTERACTION_ATTENTION_HEADS="${GENERATOR_INTERACTION_ATTENTION_HEADS:-4}"
GENERATOR_INTERACTION_TOPK="${GENERATOR_INTERACTION_TOPK:-2}"
GENERATOR_INTERACTION_GATE_BIAS="${GENERATOR_INTERACTION_GATE_BIAS:--4.0}"
GENERATOR_INTERACTION_SCALE="${GENERATOR_INTERACTION_SCALE:-1.0}"
GENERATOR_INTERACTION_EVERY="${GENERATOR_INTERACTION_EVERY:-1}"
MODEL_DTYPE="${MODEL_DTYPE:-bf16}"
SEED="${SEED:-200}"

mkdir -p "${REPO_ROOT}/logs/slurm"
cd "${REPO_ROOT}"

echo "hostname: $(hostname)"
echo "repo_root: ${REPO_ROOT}"
echo "generator_type: ${GENERATOR_TYPE}"
echo "data_path: ${DATA_PATH}"
echo "num_return_sequences: ${NUM_RETURN_SEQUENCES}"
echo "sampling_by: ${SAMPLING_BY}"
echo "generator_interaction_type: ${GENERATOR_INTERACTION_TYPE}"
echo "generator_interaction_checkpoint: ${GENERATOR_INTERACTION_CHECKPOINT}"
echo "seed: ${SEED}"
echo "slurm_job_id: ${SLURM_JOB_ID:-unknown}"

source ~/.bashrc.conda
conda activate latenttts

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

cmd=(
    python -m src.infer_gpt2
    --model_type="${GENERATOR_TYPE}"
    --model_dtype="${MODEL_DTYPE}"
    --data_path="${DATA_PATH}"
    --n_samples="${NUM_RETURN_SEQUENCES}"
    --latent_length="${LATENT_LENGTH}"
    --max_new_tokens="${MAX_NEW_TOKENS}"
    --do_sample="${DO_SAMPLE}"
    --sampling_by="${SAMPLING_BY}"
    --generator_communication_type="${GENERATOR_INTERACTION_TYPE}"
    --generator_communication_attention_heads="${GENERATOR_INTERACTION_ATTENTION_HEADS}"
    --generator_communication_topk="${GENERATOR_INTERACTION_TOPK}"
    --generator_communication_gate_bias="${GENERATOR_INTERACTION_GATE_BIAS}"
    --generator_communication_interaction_scale="${GENERATOR_INTERACTION_SCALE}"
    --generator_communication_every="${GENERATOR_INTERACTION_EVERY}"
    --seed="${SEED}"
)

if [[ -n "${GENERATOR_INTERACTION_CHECKPOINT}" ]]; then
    cmd+=(--generator_communication_checkpoint="${GENERATOR_INTERACTION_CHECKPOINT}")
fi
if [[ -n "${NOISE_STD}" ]]; then
    cmd+=(--noise_std="${NOISE_STD}")
fi
if [[ -n "${DROPOUT_P}" ]]; then
    cmd+=(--dropout_p="${DROPOUT_P}")
fi

printf 'Running: %s\n' "${cmd[*]}"
srun "${cmd[@]}"
