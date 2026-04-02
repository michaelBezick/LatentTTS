#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
SBATCH_SCRIPT="${SCRIPT_DIR}/run_python_module.sbatch"
LOG_DIR="${REPO_ROOT}/logs/slurm"

TRAIN_CONFIG="${1:-${TRAIN_CONFIG:-training_args/train_coconut_soft_attention.yaml}}"

JOB_NAME="${JOB_NAME:-latenttts-train}"
ACCOUNT="${ACCOUNT:-}"
PARTITION="${PARTITION:-}"
QOS="${QOS:-}"
TIME_LIMIT="${TIME_LIMIT:-24:00:00}"
NUM_GPUS="${NUM_GPUS:-4}"
GPU_TYPE="${GPU_TYPE:-a100}"
CPUS_PER_TASK="${CPUS_PER_TASK:-32}"
MEMORY="${MEMORY:-240G}"
CONDA_ENV="${CONDA_ENV:-latenttts}"
VENV_ACTIVATE="${VENV_ACTIVATE:-}"
MODULES_TO_LOAD="${MODULES_TO_LOAD:-}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29500}"
EXTRA_ACCELERATE_ARGS="${EXTRA_ACCELERATE_ARGS:-}"

export REPO_ROOT
export PYTHON_MODULE="src.train"
export MODULE_ARGS="${TRAIN_CONFIG}"
export NUM_GPUS
export CONDA_ENV
export VENV_ACTIVATE
export MODULES_TO_LOAD
export MAIN_PROCESS_PORT
export EXTRA_ACCELERATE_ARGS
export USE_ACCELERATE=1

mkdir -p "${LOG_DIR}"

sbatch_args=(
    "--job-name=${JOB_NAME}"
    "--nodes=1"
    "--ntasks=1"
    "--chdir=${REPO_ROOT}"
    "--cpus-per-task=${CPUS_PER_TASK}"
    "--time=${TIME_LIMIT}"
    "--mem=${MEMORY}"
    "--output=${LOG_DIR}/%x-%j.out"
    "--error=${LOG_DIR}/%x-%j.err"
    "--export=ALL"
)

if [[ -n "${ACCOUNT}" ]]; then
    sbatch_args+=("--account=${ACCOUNT}")
fi
if [[ -n "${PARTITION}" ]]; then
    sbatch_args+=("--partition=${PARTITION}")
fi
if [[ -n "${QOS}" ]]; then
    sbatch_args+=("--qos=${QOS}")
fi
if [[ -n "${GPU_TYPE}" ]]; then
    sbatch_args+=("--gres=gpu:${GPU_TYPE}:${NUM_GPUS}")
else
    sbatch_args+=("--gres=gpu:${NUM_GPUS}")
fi

printf 'Submitting command: sbatch'
printf ' %q' "${sbatch_args[@]}"
printf ' %q\n' "${SBATCH_SCRIPT}"

sbatch "${sbatch_args[@]}" "${SBATCH_SCRIPT}"
