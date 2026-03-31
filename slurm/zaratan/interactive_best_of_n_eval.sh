#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/interactive_common.sh"

PRM_ID="${PRM_ID:-checkpoints/latentRM}"
DATA_PATH="${DATA_PATH:-data/gsm_valid.json}"
NUM_RETURN_SEQUENCES="${NUM_RETURN_SEQUENCES:-64}"
COMMUNICATION_TYPE="${COMMUNICATION_TYPE:-attention}"
MODEL_DTYPE="${MODEL_DTYPE:-bf16}"
GENERATOR_TYPE="${GENERATOR_TYPE:-coconut}"

ACCOUNT="${ACCOUNT:-}"
PARTITION="${PARTITION:-}"
QOS="${QOS:-}"
TIME_LIMIT="${TIME_LIMIT:-01:00:00}"
NUM_GPUS="${NUM_GPUS:-1}"
GPU_TYPE="${GPU_TYPE:-a100}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
MEMORY="${MEMORY:-64G}"
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
  echo "Interactive shell ready. Run your evaluation command manually."
  exec bash -i
fi
cmd=(python -m src.infer_gpt2_rm --generator_type="${GENERATOR_TYPE}" --prm_mode=best_of_n --model_dtype="${MODEL_DTYPE}" --prm_id="${PRM_ID}" --data_path="${DATA_PATH}" --num_return_sequences="${NUM_RETURN_SEQUENCES}" --communication_type="${COMMUNICATION_TYPE}")
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
