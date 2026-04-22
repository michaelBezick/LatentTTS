#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
LOG_DIR="${REPO_ROOT}/logs/slurm"

TRAIN_CONFIG="${1:-${TRAIN_CONFIG:-training_args/train_coconut_generator_interaction_verifiable_rl.yaml}}"

ACCOUNT="${ACCOUNT:-}"
PARTITION="${PARTITION:-}"
QOS="${QOS:-}"
TIME_LIMIT="${TIME_LIMIT:-24:00:00}"
NUM_GPUS="${NUM_GPUS:-4}"
GPU_TYPE="${GPU_TYPE:-a100}"
CPUS_PER_TASK="${CPUS_PER_TASK:-32}"
MEMORY="${MEMORY:-240G}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29500}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_INIT_TIMEOUT="${WANDB_INIT_TIMEOUT:-300}"
WANDB__SERVICE_WAIT="${WANDB__SERVICE_WAIT:-60}"

export REPO_ROOT
export TRAIN_CONFIG
export NUM_GPUS
export MAIN_PROCESS_PORT
export MIXED_PRECISION
export WANDB_MODE
export WANDB_INIT_TIMEOUT
export WANDB__SERVICE_WAIT

mkdir -p "${LOG_DIR}"

sbatch_args=(
    "--job-name=latenttts-gen-interaction-train"
    "--nodes=1" "--ntasks=1"
    "--chdir=${REPO_ROOT}"
    "--cpus-per-task=${CPUS_PER_TASK}"
    "--time=${TIME_LIMIT}"
    "--mem=${MEMORY}"
    "--output=${LOG_DIR}/%x-%j.out"
    "--export=ALL,WANDB_MODE=${WANDB_MODE},WANDB_INIT_TIMEOUT=${WANDB_INIT_TIMEOUT},WANDB__SERVICE_WAIT=${WANDB__SERVICE_WAIT}"
)

if [[ -n "${ACCOUNT}" ]];   then sbatch_args+=("--account=${ACCOUNT}");     fi
if [[ -n "${PARTITION}" ]]; then sbatch_args+=("--partition=${PARTITION}"); fi
if [[ -n "${QOS}" ]];       then sbatch_args+=("--qos=${QOS}");             fi
if [[ -n "${GPU_TYPE}" ]]; then
    sbatch_args+=("--gres=gpu:${GPU_TYPE}:${NUM_GPUS}")
else
    sbatch_args+=("--gres=gpu:${NUM_GPUS}")
fi

echo "Submitting generator interaction training job: ${TRAIN_CONFIG}"
echo "wandb_mode: ${WANDB_MODE}"
echo "wandb_init_timeout: ${WANDB_INIT_TIMEOUT}"
echo "wandb_service_wait: ${WANDB__SERVICE_WAIT}"
sbatch "${sbatch_args[@]}" "${SCRIPT_DIR}/run_train_generator_interaction.sbatch"
