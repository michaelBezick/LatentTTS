#!/usr/bin/env bash
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:?Set REPO_ROOT before submitting.}"
ANNOTATION_SCRIPT="${ANNOTATION_SCRIPT:-run_annotation.sh}"
CONDA_ENV="${CONDA_ENV:-}"
VENV_ACTIVATE="${VENV_ACTIVATE:-}"
MODULES_TO_LOAD="${MODULES_TO_LOAD:-}"

mkdir -p "${REPO_ROOT}/logs/slurm"
cd "${REPO_ROOT}"

if [[ -n "${MODULES_TO_LOAD}" ]]; then
    if command -v module >/dev/null 2>&1; then
        source /etc/profile || true
        for module_name in ${MODULES_TO_LOAD}; do
            module load "${module_name}"
        done
    else
        echo "Warning: MODULES_TO_LOAD was set, but the 'module' command is unavailable."
    fi
fi

if [[ -f "${HOME}/.bashrc" ]]; then
    # shellcheck disable=SC1090
    source "${HOME}/.bashrc"
fi

if [[ -n "${CONDA_ENV}" ]]; then
    eval "$(conda shell.bash hook)"
    conda activate "${CONDA_ENV}"
elif [[ -n "${VENV_ACTIVATE}" ]]; then
    # shellcheck disable=SC1090
    source "${VENV_ACTIVATE}"
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-1}}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONUNBUFFERED=1

printf 'Running command: bash %q\n' "${ANNOTATION_SCRIPT}"
srun bash "${ANNOTATION_SCRIPT}"
