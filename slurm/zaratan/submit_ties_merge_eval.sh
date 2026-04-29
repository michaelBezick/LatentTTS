#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
LOG_DIR="${REPO_ROOT}/logs/slurm"

GENERATOR_TYPE="${GENERATOR_TYPE:-coconut}"
PRM_ID="${PRM_ID:-checkpoints/latentRM}"
DATA_PATH="${DATA_PATH:-data/gsm_valid.json}"
NUM_RETURN_SEQUENCES="${NUM_RETURN_SEQUENCES:-8}"
SAMPLING_BY="${SAMPLING_BY:-dropout}"
NOISE_STD="${NOISE_STD:-}"
DROPOUT_P="${DROPOUT_P:-}"
MODEL_DTYPE="${MODEL_DTYPE:-bf16}"
SEED="${SEED:-200}"
TOP_K_RATIO="${TOP_K_RATIO:-0.2}"
MERGE_MODE="${MERGE_MODE:-all_pairs}"
USE_TASK_VECTORS="${USE_TASK_VECTORS:-False}"
MERGE_SCORE_CHUNK_SIZE="${MERGE_SCORE_CHUNK_SIZE:-64}"

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
export SAMPLING_BY
export NOISE_STD
export DROPOUT_P
export MODEL_DTYPE
export SEED
export TOP_K_RATIO
export MERGE_MODE
export USE_TASK_VECTORS
export MERGE_SCORE_CHUNK_SIZE

mkdir -p "${LOG_DIR}"

sbatch_args=(
    "--job-name=latenttts-ties-merge-eval"
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

echo "Submitting TIES merge eval job..."
sbatch "${sbatch_args[@]}" "${SCRIPT_DIR}/run_ties_merge_eval.sbatch"
