#!/usr/bin/env bash
#SBATCH --output=logs/slurm/%x-%j.out

REPO_ROOT="${REPO_ROOT:?REPO_ROOT not set — submit via submit_best_of_n_eval.sh}"
GENERATOR_TYPE="${GENERATOR_TYPE:-coconut}"
PRM_ID="${PRM_ID:-checkpoints/latentRM}"
DATA_PATH="${DATA_PATH:-data/gsm_valid.json}"
NUM_RETURN_SEQUENCES="${NUM_RETURN_SEQUENCES:-64}"
INTERACTION_TYPE="${INTERACTION_TYPE:-attention}"
MODEL_DTYPE="${MODEL_DTYPE:-bf16}"

mkdir -p "${REPO_ROOT}/logs/slurm"
cd "${REPO_ROOT}"

echo "hostname: $(hostname)"
echo "repo_root: ${REPO_ROOT}"
echo "generator_type: ${GENERATOR_TYPE}"
echo "prm_id: ${PRM_ID}"
echo "data_path: ${DATA_PATH}"
echo "num_return_sequences: ${NUM_RETURN_SEQUENCES}"
echo "slurm_job_id: ${SLURM_JOB_ID:-unknown}"

source ~/.bashrc.conda
conda activate latenttts

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

printf 'Running: python -m src.infer_gpt2_rm\n'
srun python -m src.infer_gpt2_rm \
    --generator_type="${GENERATOR_TYPE}" \
    --prm_mode=best_of_n \
    --model_dtype="${MODEL_DTYPE}" \
    --prm_id="${PRM_ID}" \
    --data_path="${DATA_PATH}" \
    --num_return_sequences="${NUM_RETURN_SEQUENCES}" \
    --communication_type="${INTERACTION_TYPE}"
