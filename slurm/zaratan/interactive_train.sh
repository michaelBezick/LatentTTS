#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/interactive_common.sh"

TRAIN_CONFIG="${1:-${TRAIN_CONFIG:-training_args/train_coconut_soft_attention.yaml}}"

ACCOUNT="${ACCOUNT:-}"
PARTITION="${PARTITION:-}"
QOS="${QOS:-}"
TIME_LIMIT="${TIME_LIMIT:-04:00:00}"
NUM_GPUS="${NUM_GPUS:-4}"
GPU_TYPE="${GPU_TYPE:-a100}"
CPUS_PER_TASK="${CPUS_PER_TASK:-32}"
MEMORY="${MEMORY:-240G}"
CONDA_ENV="${CONDA_ENV:-latenttts}"
VENV_ACTIVATE="${VENV_ACTIVATE:-}"
MODULES_TO_LOAD="${MODULES_TO_LOAD:-}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29500}"
EXTRA_ACCELERATE_ARGS="${EXTRA_ACCELERATE_ARGS:-}"
OPEN_SHELL_ONLY="${OPEN_SHELL_ONLY:-0}"

salloc_args=(
    "--nodes=1"
    "--ntasks=1"
    "--cpus-per-task=${CPUS_PER_TASK}"
    "--time=${TIME_LIMIT}"
    "--mem=${MEMORY}"
)

if [[ -n "${ACCOUNT}" ]]; then
    salloc_args+=("--account=${ACCOUNT}")
fi
if [[ -n "${PARTITION}" ]]; then
    salloc_args+=("--partition=${PARTITION}")
fi
if [[ -n "${QOS}" ]]; then
    salloc_args+=("--qos=${QOS}")
fi
if [[ -n "${GPU_TYPE}" ]]; then
    salloc_args+=("--gres=gpu:${GPU_TYPE}:${NUM_GPUS}")
else
    salloc_args+=("--gres=gpu:${NUM_GPUS}")
fi

read -r -a extra_accelerate_args <<< "${EXTRA_ACCELERATE_ARGS}"

command_script=$(cat <<EOF
set -euo pipefail
export CONDA_ENV="${CONDA_ENV}"
export VENV_ACTIVATE="${VENV_ACTIVATE}"
export MODULES_TO_LOAD="${MODULES_TO_LOAD}"
cd "${REPO_ROOT}"
source "${SCRIPT_DIR}/interactive_common.sh"
setup_cluster_environment "${REPO_ROOT}"
print_interactive_context
if [[ "${OPEN_SHELL_ONLY}" == "1" ]]; then
  echo "Interactive shell ready. Run your training command manually."
  exec bash -i
fi
cmd=(python -m accelerate.commands.launch --num_processes "${NUM_GPUS}" --num_machines 1 --main_process_port "${MAIN_PROCESS_PORT}" ${EXTRA_ACCELERATE_ARGS} -m src.train "${TRAIN_CONFIG}")
printf 'Running command:'
printf ' %q' "\${cmd[@]}"
printf '\n'
"\${cmd[@]}"
EOF
)

printf 'Requesting interactive allocation with: salloc'
printf ' %q' "${salloc_args[@]}"
printf '\n'

salloc "${salloc_args[@]}" srun --pty bash -lc "${command_script}"
