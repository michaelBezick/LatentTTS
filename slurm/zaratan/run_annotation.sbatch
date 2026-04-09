#!/usr/bin/env bash
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

REPO_ROOT="${REPO_ROOT:?REPO_ROOT not set — submit via submit_annotation.sh}"
ANNOTATION_SCRIPT="${ANNOTATION_SCRIPT:-run_annotation.sh}"

mkdir -p "${REPO_ROOT}/logs/slurm"
cd "${REPO_ROOT}"

echo "hostname: $(hostname)"
echo "repo_root: ${REPO_ROOT}"
echo "annotation_script: ${ANNOTATION_SCRIPT}"
echo "slurm_job_id: ${SLURM_JOB_ID:-unknown}"
echo "dataset_piece: ${DATASET_PIECE:-1.0}  dataset_indice: ${DATASET_INDICE:-0}"
echo "num_processes: ${NUM_PROCESSES:-1}"

source ~/.bashrc.conda
conda activate latenttts

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

# Forward sharding/multi-GPU vars set by submit_annotation.sh.
export DATASET_PIECE="${DATASET_PIECE:-1.0}"
export DATASET_INDICE="${DATASET_INDICE:-0}"
export NUM_PROCESSES="${NUM_PROCESSES:-1}"

printf 'Running: bash %s\n' "${ANNOTATION_SCRIPT}"
srun bash "${ANNOTATION_SCRIPT}"
