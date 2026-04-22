#!/usr/bin/env bash
#SBATCH --output=logs/slurm/%x-%j.out

REPO_ROOT="${REPO_ROOT:?REPO_ROOT not set — submit via submit_train_generator_interaction.sh}"
TRAIN_CONFIG="${TRAIN_CONFIG:-training_args/train_coconut_generator_interaction_verifiable_rl.yaml}"
NUM_GPUS="${NUM_GPUS:-4}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29500}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
WANDB_MODE="${WANDB_MODE:-offline}"
WANDB_INIT_TIMEOUT="${WANDB_INIT_TIMEOUT:-300}"
WANDB__SERVICE_WAIT="${WANDB__SERVICE_WAIT:-60}"
WANDB_DIR="${WANDB_DIR:-${REPO_ROOT}/wandb-offline}"
WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-${WANDB_DIR}/cache}"
WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-${WANDB_DIR}/config}"
WANDB_DATA_DIR="${WANDB_DATA_DIR:-${WANDB_DIR}/data}"
WANDB_PROJECT="${WANDB_PROJECT:-latenttts-generator-interaction}"

mkdir -p "${REPO_ROOT}/logs/slurm"
mkdir -p "${WANDB_DIR}" "${WANDB_CACHE_DIR}" "${WANDB_CONFIG_DIR}" "${WANDB_DATA_DIR}"
cd "${REPO_ROOT}"

echo "hostname: $(hostname)"
echo "repo_root: ${REPO_ROOT}"
echo "train_config: ${TRAIN_CONFIG}"
echo "num_gpus: ${NUM_GPUS}"
echo "wandb_mode: ${WANDB_MODE}"
echo "wandb_init_timeout: ${WANDB_INIT_TIMEOUT}"
echo "wandb_service_wait: ${WANDB__SERVICE_WAIT}"
echo "wandb_dir: ${WANDB_DIR}"
echo "wandb_project: ${WANDB_PROJECT}"
echo "slurm_job_id: ${SLURM_JOB_ID:-unknown}"

source ~/.bashrc.conda
conda activate latenttts

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE
export WANDB_INIT_TIMEOUT
export WANDB__SERVICE_WAIT
export WANDB_DIR
export WANDB_CACHE_DIR
export WANDB_CONFIG_DIR
export WANDB_DATA_DIR
export WANDB_PROJECT

if [[ "${WANDB_MODE}" == "offline" ]]; then
    printf 'Offline W&B runs can be synced later with: wandb sync %s/wandb/offline-run-*\n' "${WANDB_DIR}"
fi

printf 'Running: accelerate launch -m src.train_generator_interaction %s\n' "${TRAIN_CONFIG}"
srun accelerate launch \
    --num_processes "${NUM_GPUS}" \
    --num_machines 1 \
    --main_process_port "${MAIN_PROCESS_PORT}" \
    --mixed_precision "${MIXED_PRECISION}" \
    -m src.train_generator_interaction \
    "${TRAIN_CONFIG}"
