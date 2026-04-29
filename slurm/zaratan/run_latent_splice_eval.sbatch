#!/usr/bin/env bash
#SBATCH --output=logs/slurm/%x-%j.out

REPO_ROOT="${REPO_ROOT:?REPO_ROOT not set — submit via submit_latent_splice_eval.sh}"
GENERATOR_TYPE="${GENERATOR_TYPE:-coconut}"
PRM_ID="${PRM_ID:-checkpoints/latentRM}"
DATA_PATH="${DATA_PATH:-data/gsm_valid.json}"
NUM_RETURN_SEQUENCES="${NUM_RETURN_SEQUENCES:-8}"
SAMPLING_BY="${SAMPLING_BY:-dropout}"
NOISE_STD="${NOISE_STD:-}"
DROPOUT_P="${DROPOUT_P:-}"
MODEL_DTYPE="${MODEL_DTYPE:-bf16}"
SEED="${SEED:-200}"
ACCEPTANCE_MARGIN="${ACCEPTANCE_MARGIN:-0.0}"
SPLICE_SCORE_CHUNK_SIZE="${SPLICE_SCORE_CHUNK_SIZE:-64}"
SPLICE_DECODE_BATCH_SIZE="${SPLICE_DECODE_BATCH_SIZE:-64}"

mkdir -p "${REPO_ROOT}/logs/slurm"
cd "${REPO_ROOT}"

echo "hostname: $(hostname)"
echo "repo_root: ${REPO_ROOT}"
echo "generator_type: ${GENERATOR_TYPE}"
echo "prm_id: ${PRM_ID}"
echo "data_path: ${DATA_PATH}"
echo "num_return_sequences: ${NUM_RETURN_SEQUENCES}"
echo "sampling_by: ${SAMPLING_BY}"
echo "model_dtype: ${MODEL_DTYPE}"
echo "seed: ${SEED}"
echo "acceptance_margin: ${ACCEPTANCE_MARGIN}"
echo "splice_score_chunk_size: ${SPLICE_SCORE_CHUNK_SIZE}"
echo "splice_decode_batch_size: ${SPLICE_DECODE_BATCH_SIZE}"
echo "slurm_job_id: ${SLURM_JOB_ID:-unknown}"

source ~/.bashrc.conda
conda activate latenttts

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

cmd=(
    python -m src.infer_latent_splice
    --generator_type="${GENERATOR_TYPE}"
    --model_dtype="${MODEL_DTYPE}"
    --prm_id="${PRM_ID}"
    --data_path="${DATA_PATH}"
    --num_return_sequences="${NUM_RETURN_SEQUENCES}"
    --seed="${SEED}"
    --sampling_by="${SAMPLING_BY}"
    --acceptance_margin="${ACCEPTANCE_MARGIN}"
    --splice_score_chunk_size="${SPLICE_SCORE_CHUNK_SIZE}"
    --splice_decode_batch_size="${SPLICE_DECODE_BATCH_SIZE}"
)

if [[ -n "${NOISE_STD}" ]];  then cmd+=(--noise_std="${NOISE_STD}");   fi
if [[ -n "${DROPOUT_P}" ]];  then cmd+=(--dropout_p="${DROPOUT_P}");   fi

printf 'Running: %s\n' "${cmd[*]}"
srun "${cmd[@]}"
