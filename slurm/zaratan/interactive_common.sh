#!/usr/bin/env bash

set -euo pipefail

setup_cluster_environment() {
    local repo_root="$1"

    if [[ -n "${MODULES_TO_LOAD:-}" ]]; then
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

    if [[ -n "${CONDA_ENV:-}" ]]; then
        eval "$(conda shell.bash hook)"
        conda activate "${CONDA_ENV}"
    elif [[ -n "${VENV_ACTIVATE:-}" ]]; then
        # shellcheck disable=SC1090
        source "${VENV_ACTIVATE}"
    fi

    export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-1}}"
    export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
    export PYTHONUNBUFFERED=1

    cd "${repo_root}"
}


print_interactive_context() {
    echo "==== Zaratan interactive context ===="
    echo "hostname: $(hostname)"
    echo "repo_root: $(pwd)"
    echo "slurm_job_id: ${SLURM_JOB_ID:-none}"
    echo "cuda_visible_devices: ${CUDA_VISIBLE_DEVICES:-unset}"
    echo "===================================="
}
