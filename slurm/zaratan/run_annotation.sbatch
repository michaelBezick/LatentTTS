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

source ~/.bashrc.conda
conda activate latenttts

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

printf 'Running: bash %s\n' "${ANNOTATION_SCRIPT}"
srun bash "${ANNOTATION_SCRIPT}"
