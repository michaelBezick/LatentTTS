#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
LOG_DIR="${REPO_ROOT}/logs/slurm"

GENERATOR_TYPE="${GENERATOR_TYPE:-coconut}"
PRM_ID="${PRM_ID:-checkpoints/latentRM}"
DATA_PATH="${DATA_PATH:-data/gsm_valid.json}"
NUM_RETURN_SEQUENCES="${NUM_RETURN_SEQUENCES:-8}"
GENERATOR_COMMUNICATION_TYPE="${GENERATOR_COMMUNICATION_TYPE:-attention}"
GENERATOR_COMMUNICATION_CHECKPOINT="${GENERATOR_COMMUNICATION_CHECKPOINT:-outputs/coconut-generator-attention/best}"
GENERATOR_COMMUNICATION_ATTENTION_HEADS="${GENERATOR_COMMUNICATION_ATTENTION_HEADS:-4}"
GENERATOR_COMMUNICATION_TOPK="${GENERATOR_COMMUNICATION_TOPK:-2}"
GENERATOR_COMMUNICATION_EVERY="${GENERATOR_COMMUNICATION_EVERY:-1}"
RM_COMMUNICATION_TYPE="${RM_COMMUNICATION_TYPE:-none}"
MODEL_DTYPE="${MODEL_DTYPE:-bf16}"
SEED="${SEED:-200}"

ACCOUNT="${ACCOUNT:-}"
PARTITION="${PARTITION:-}"
QOS="${QOS:-}"
TIME_LIMIT="${TIME_LIMIT:-04:00:00}"
NUM_GPUS="${NUM_GPUS:-1}"
GPU_TYPE="${GPU_TYPE:-a100}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
MEMORY="${MEMORY:-64G}"

export REPO_ROOT
export GENERATOR_TYPE
export PRM_ID
export DATA_PATH
export NUM_RETURN_SEQUENCES
export GENERATOR_COMMUNICATION_TYPE
export GENERATOR_COMMUNICATION_CHECKPOINT
export GENERATOR_COMMUNICATION_ATTENTION_HEADS
export GENERATOR_COMMUNICATION_TOPK
export GENERATOR_COMMUNICATION_EVERY
export RM_COMMUNICATION_TYPE
export MODEL_DTYPE
export SEED

mkdir -p "${LOG_DIR}"

sbatch_args=(
    "--job-name=latenttts-gen-comm-eval"
    "--nodes=1" "--ntasks=1"
    "--chdir=${REPO_ROOT}"
    "--cpus-per-task=${CPUS_PER_TASK}"
    "--time=${TIME_LIMIT}"
    "--mem=${MEMORY}"
    "--output=${LOG_DIR}/%x-%j.out"
    "--export=ALL"
)

if [[ -n "${ACCOUNT}" ]];   then sbatch_args+=("--account=${ACCOUNT}");     fi
if [[ -n "${PARTITION}" ]]; then sbatch_args+=("--partition=${PARTITION}"); fi
if [[ -n "${QOS}" ]];       then sbatch_args+=("--qos=${QOS}");             fi
if [[ -n "${GPU_TYPE}" ]]; then
    sbatch_args+=("--gres=gpu:${GPU_TYPE}:${NUM_GPUS}")
else
    sbatch_args+=("--gres=gpu:${NUM_GPUS}")
fi

echo "Submitting generator communication eval job..."
sbatch "${sbatch_args[@]}" "${SCRIPT_DIR}/run_generator_comm_eval.sbatch"
