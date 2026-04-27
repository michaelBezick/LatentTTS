#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
LOG_DIR="${REPO_ROOT}/logs/slurm"

GENERATOR_TYPE="${GENERATOR_TYPE:-coconut}"
PRM_ID="${PRM_ID:-checkpoints/latentRM}"
DATA_PATH="${DATA_PATH:-data/gsm_valid.json}"
NUM_RETURN_SEQUENCES="${NUM_RETURN_SEQUENCES:-8}"
GENERATOR_INTERACTION_TYPE="${GENERATOR_INTERACTION_TYPE:-gated_router}"
GENERATOR_INTERACTION_CHECKPOINT="${GENERATOR_INTERACTION_CHECKPOINT:-outputs/coconut-generator-interaction-gated-router-ce/best}"
GENERATOR_ADAPTER_CHECKPOINT="${GENERATOR_ADAPTER_CHECKPOINT:-outputs/coconut-generator-interaction-gated-router-ce/best/generator_adapter}"
GENERATOR_INTERACTION_ATTENTION_HEADS="${GENERATOR_INTERACTION_ATTENTION_HEADS:-4}"
GENERATOR_INTERACTION_TOPK="${GENERATOR_INTERACTION_TOPK:-2}"
GENERATOR_INTERACTION_GATE_BIAS="${GENERATOR_INTERACTION_GATE_BIAS:--4.0}"
GENERATOR_INTERACTION_SCALE="${GENERATOR_INTERACTION_SCALE:-1.0}"
GENERATOR_INTERACTION_EVERY="${GENERATOR_INTERACTION_EVERY:-1}"
RM_INTERACTION_TYPE="${RM_INTERACTION_TYPE:-none}"
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
export GENERATOR_INTERACTION_TYPE
export GENERATOR_INTERACTION_CHECKPOINT
export GENERATOR_ADAPTER_CHECKPOINT
export GENERATOR_INTERACTION_ATTENTION_HEADS
export GENERATOR_INTERACTION_TOPK
export GENERATOR_INTERACTION_GATE_BIAS
export GENERATOR_INTERACTION_SCALE
export GENERATOR_INTERACTION_EVERY
export RM_INTERACTION_TYPE
export MODEL_DTYPE
export SEED

mkdir -p "${LOG_DIR}"

sbatch_args=(
    "--job-name=latenttts-gen-interaction-eval"
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

echo "Submitting generator interaction eval job..."
sbatch "${sbatch_args[@]}" "${SCRIPT_DIR}/run_generator_interaction_eval.sbatch"
