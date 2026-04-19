#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
LOG_DIR="${REPO_ROOT}/logs/slurm"

GENERATOR_TYPE="${GENERATOR_TYPE:-coconut}"
PRM_ID="${PRM_ID:-outputs/coconut-soft-attention/best}"
DATA_PATH="${DATA_PATH:-data/gsm_valid.json}"
NUM_RETURN_SEQUENCES="${NUM_RETURN_SEQUENCES:-64}"
MODEL_DTYPE="${MODEL_DTYPE:-bf16}"
SEED="${SEED:-200}"
PAIRWISE_COSINE_POOL="${PAIRWISE_COSINE_POOL:-mean}"
PAIRWISE_COLLAPSE_THRESHOLD="${PAIRWISE_COLLAPSE_THRESHOLD:-0.9}"
RESULTS_DIR="${RESULTS_DIR:-${REPO_ROOT}/logs/mode-collapse}"

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
export MODEL_DTYPE
export SEED
export PAIRWISE_COSINE_POOL
export PAIRWISE_COLLAPSE_THRESHOLD
export RESULTS_DIR

mkdir -p "${LOG_DIR}"

sbatch_args=(
    "--job-name=latenttts-collapse-ablation"
    "--nodes=1" "--ntasks=1"
    "--chdir=${REPO_ROOT}"
    "--cpus-per-task=${CPUS_PER_TASK}"
    "--time=${TIME_LIMIT}"
    "--mem=${MEMORY}"
    "--output=${LOG_DIR}/%x-%j.out"
    "--error=${LOG_DIR}/%x-%j.err"
    "--export=ALL"
)

if [[ -n "${ACCOUNT}" ]]; then sbatch_args+=("--account=${ACCOUNT}"); fi
if [[ -n "${PARTITION}" ]]; then sbatch_args+=("--partition=${PARTITION}"); fi
if [[ -n "${QOS}" ]]; then sbatch_args+=("--qos=${QOS}"); fi
if [[ -n "${GPU_TYPE}" ]]; then
    sbatch_args+=("--gres=gpu:${GPU_TYPE}:${NUM_GPUS}")
else
    sbatch_args+=("--gres=gpu:${NUM_GPUS}")
fi

echo "Submitting mode-collapse ablation job..."
sbatch "${sbatch_args[@]}" "${SCRIPT_DIR}/run_mode_collapse_ablation.sbatch"
