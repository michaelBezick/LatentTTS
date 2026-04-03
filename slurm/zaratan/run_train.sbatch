#!/usr/bin/env bash
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

REPO_ROOT="${REPO_ROOT:?REPO_ROOT not set — submit via submit_train.sh}"
TRAIN_CONFIG="${TRAIN_CONFIG:-training_args/train_coconut_soft_attention.yaml}"
NUM_GPUS="${NUM_GPUS:-4}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29500}"

mkdir -p "${REPO_ROOT}/logs/slurm"
cd "${REPO_ROOT}"

echo "hostname: $(hostname)"
echo "repo_root: ${REPO_ROOT}"
echo "train_config: ${TRAIN_CONFIG}"
echo "num_gpus: ${NUM_GPUS}"
echo "slurm_job_id: ${SLURM_JOB_ID:-unknown}"

source ~/.bashrc.conda
conda activate latenttts

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

printf 'Running: accelerate launch -m src.train %s\n' "${TRAIN_CONFIG}"
srun accelerate launch \
    --num_processes "${NUM_GPUS}" \
    --num_machines 1 \
    --main_process_port "${MAIN_PROCESS_PORT}" \
    -m src.train \
    "${TRAIN_CONFIG}"
