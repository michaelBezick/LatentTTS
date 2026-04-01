#!/usr/bin/env bash

set -euo pipefail

source_with_nounset_disabled() {
    local target_file="$1"
    local had_nounset=0

    if [[ $- == *u* ]]; then
        had_nounset=1
        set +u
    fi

    # shellcheck disable=SC1090
    source "${target_file}"

    if [[ ${had_nounset} -eq 1 ]]; then
        set -u
    fi
}


setup_cluster_environment() {
    local repo_root="$1"

    if [[ -n "${MODULES_TO_LOAD:-}" ]]; then
        if command -v module >/dev/null 2>&1; then
            if [[ -f /etc/profile ]]; then
                source_with_nounset_disabled /etc/profile || true
            fi
            for module_name in ${MODULES_TO_LOAD}; do
                module load "${module_name}"
            done
        else
            echo "Warning: MODULES_TO_LOAD was set, but the 'module' command is unavailable."
        fi
    fi

    if [[ -f "${HOME}/.bashrc" ]]; then
        source_with_nounset_disabled "${HOME}/.bashrc"
    fi

    if [[ -n "${CONDA_ENV:-}" ]]; then
        eval "$(conda shell.bash hook)"
        conda activate "${CONDA_ENV}"
    elif [[ -n "${VENV_ACTIVATE:-}" ]]; then
        source_with_nounset_disabled "${VENV_ACTIVATE}"
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
