#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
SBATCH_SCRIPT="${SCRIPT_DIR}/run_annotation.sbatch"

ANNOTATION_SCRIPT="${ANNOTATION_SCRIPT:-run_annotation.sh}"

JOB_NAME="${JOB_NAME:-latenttts-annotate}"
ACCOUNT="${ACCOUNT:-}"
PARTITION="${PARTITION:-}"
QOS="${QOS:-}"
TIME_LIMIT="${TIME_LIMIT:-12:00:00}"
NUM_GPUS="${NUM_GPUS:-1}"
GPU_TYPE="${GPU_TYPE:-a100}"
CPUS_PER_TASK="${CPUS_PER_TASK:-16}"
MEMORY="${MEMORY:-120G}"
CONDA_ENV="${CONDA_ENV:-latenttts}"
VENV_ACTIVATE="${VENV_ACTIVATE:-}"
MODULES_TO_LOAD="${MODULES_TO_LOAD:-}"

export REPO_ROOT
export ANNOTATION_SCRIPT
export NUM_GPUS
export CONDA_ENV
export VENV_ACTIVATE
export MODULES_TO_LOAD

sbatch_args=(
    "--job-name=${JOB_NAME}"
    "--nodes=1"
    "--ntasks=1"
    "--cpus-per-task=${CPUS_PER_TASK}"
    "--time=${TIME_LIMIT}"
    "--mem=${MEMORY}"
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
