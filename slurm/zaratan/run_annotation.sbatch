#!/usr/bin/env bash
#SBATCH --output=logs/slurm/%x-%j.out

REPO_ROOT="${REPO_ROOT:?REPO_ROOT not set — submit via submit_annotation.sh}"
ANNOTATION_SCRIPT="${ANNOTATION_SCRIPT:-run_annotation.sh}"
USE_WANDB="${USE_WANDB:-True}"
WANDB_MODE="${WANDB_MODE:-offline}"
WANDB_DIR="${WANDB_DIR:-${REPO_ROOT}/wandb-offline}"
WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-${WANDB_DIR}/cache}"
WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-${WANDB_DIR}/config}"
WANDB_DATA_DIR="${WANDB_DATA_DIR:-${WANDB_DIR}/data}"
WANDB_PROJECT="${WANDB_PROJECT:-annotation-latent-data}"

mkdir -p "${REPO_ROOT}/logs/slurm"
mkdir -p "${WANDB_DIR}" "${WANDB_CACHE_DIR}" "${WANDB_CONFIG_DIR}" "${WANDB_DATA_DIR}"
cd "${REPO_ROOT}"

echo "hostname: $(hostname)"
echo "repo_root: ${REPO_ROOT}"
echo "annotation_script: ${ANNOTATION_SCRIPT}"
echo "slurm_job_id: ${SLURM_JOB_ID:-unknown}"
echo "dataset_piece: ${DATASET_PIECE:-1.0}  dataset_indice: ${DATASET_INDICE:-0}"
echo "num_processes: ${NUM_PROCESSES:-1}"
echo "use_wandb: ${USE_WANDB}"
echo "wandb_mode: ${WANDB_MODE}"
echo "wandb_dir: ${WANDB_DIR}"
echo "wandb_project: ${WANDB_PROJECT}"

source ~/.bashrc.conda
conda activate latenttts

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export USE_WANDB
export WANDB_MODE
export WANDB_DIR
export WANDB_CACHE_DIR
export WANDB_CONFIG_DIR
export WANDB_DATA_DIR
export WANDB_PROJECT

# Forward sharding/multi-GPU vars set by submit_annotation.sh.
export DATASET_PIECE="${DATASET_PIECE:-1.0}"
export DATASET_INDICE="${DATASET_INDICE:-0}"
export NUM_PROCESSES="${NUM_PROCESSES:-1}"

printf 'Offline W&B runs can be synced later with: wandb sync %s/wandb/offline-run-*\n' "${WANDB_DIR}"

printf 'Running: bash %s\n' "${ANNOTATION_SCRIPT}"
srun bash "${ANNOTATION_SCRIPT}"
