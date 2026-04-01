#!/usr/bin/env bash
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:?Set REPO_ROOT before submitting.}"
PYTHON_MODULE="${PYTHON_MODULE:?Set PYTHON_MODULE before submitting.}"
MODULE_ARGS="${MODULE_ARGS:-}"
NUM_GPUS="${NUM_GPUS:-1}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CONDA_ENV="${CONDA_ENV:-}"
VENV_ACTIVATE="${VENV_ACTIVATE:-}"
MODULES_TO_LOAD="${MODULES_TO_LOAD:-}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29500}"
EXTRA_ACCELERATE_ARGS="${EXTRA_ACCELERATE_ARGS:-}"
USE_ACCELERATE="${USE_ACCELERATE:-1}"

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

mkdir -p "${REPO_ROOT}/logs/slurm"
cd "${REPO_ROOT}"

if [[ -n "${MODULES_TO_LOAD}" ]]; then
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

if [[ -n "${CONDA_ENV}" ]]; then
    eval "$(conda shell.bash hook)"
    conda activate "${CONDA_ENV}"
elif [[ -n "${VENV_ACTIVATE}" ]]; then
    source_with_nounset_disabled "${VENV_ACTIVATE}"
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-1}}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONUNBUFFERED=1

read -r -a module_args <<< "${MODULE_ARGS}"
read -r -a extra_accelerate_args <<< "${EXTRA_ACCELERATE_ARGS}"

echo "==== Zaratan job context ===="
echo "hostname: $(hostname)"
echo "repo_root: ${REPO_ROOT}"
echo "python_module: ${PYTHON_MODULE}"
echo "module_args: ${MODULE_ARGS}"
echo "num_gpus: ${NUM_GPUS}"
echo "slurm_job_id: ${SLURM_JOB_ID:-unknown}"
echo "============================="

if [[ "${USE_ACCELERATE}" == "1" ]]; then
    launch_cmd=(
        "${PYTHON_BIN}" -m accelerate.commands.launch
        --num_processes "${NUM_GPUS}"
        --num_machines 1
        --main_process_port "${MAIN_PROCESS_PORT}"
        "${extra_accelerate_args[@]}"
        -m "${PYTHON_MODULE}"
        "${module_args[@]}"
    )
else
    launch_cmd=(
        "${PYTHON_BIN}" -m "${PYTHON_MODULE}"
        "${module_args[@]}"
    )
fi

printf 'Running command:'
printf ' %q' "${launch_cmd[@]}"
printf '\n'

srun "${launch_cmd[@]}"
