#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/interactive_common.sh"

ANNOTATION_SCRIPT="${ANNOTATION_SCRIPT:-run_annotation.sh}"

ACCOUNT="${ACCOUNT:-}"
PARTITION="${PARTITION:-}"
QOS="${QOS:-}"
TIME_LIMIT="${TIME_LIMIT:-02:00:00}"
NUM_GPUS="${NUM_GPUS:-1}"
GPU_TYPE="${GPU_TYPE:-a100}"
CPUS_PER_TASK="${CPUS_PER_TASK:-16}"
MEMORY="${MEMORY:-120G}"
CONDA_ENV="${CONDA_ENV:-latenttts}"
VENV_ACTIVATE="${VENV_ACTIVATE:-}"
MODULES_TO_LOAD="${MODULES_TO_LOAD:-}"
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
  echo "Interactive shell ready. Run your annotation command manually."
  exec bash -i
fi
printf 'Running command: bash %q\n' "${ANNOTATION_SCRIPT}"
bash "${ANNOTATION_SCRIPT}"
EOF
)

printf 'Requesting interactive allocation with: salloc'
printf ' %q' "${salloc_args[@]}"
printf '\n'

salloc "${salloc_args[@]}" srun --pty bash -lc "${command_script}"
