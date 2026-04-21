#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
LOG_DIR="${REPO_ROOT}/logs/slurm"

GENERATOR_TYPE="${GENERATOR_TYPE:-coconut}"
DATA_PATH="${DATA_PATH:-data/gsm_valid.json}"
NUM_RETURN_SEQUENCES="${NUM_RETURN_SEQUENCES:-8}"
LATENT_LENGTH="${LATENT_LENGTH:-6}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
DO_SAMPLE="${DO_SAMPLE:-true}"
SAMPLING_BY="${SAMPLING_BY:-noise}"
NOISE_STD="${NOISE_STD:-0.1}"
DROPOUT_P="${DROPOUT_P:-0.2}"
GENERATOR_INTERACTION_TYPE="${GENERATOR_INTERACTION_TYPE:-attention}"
GENERATOR_INTERACTION_CHECKPOINT="${GENERATOR_INTERACTION_CHECKPOINT:-outputs/coconut-generator-interaction-verifiable-rl/best}"
GENERATOR_INTERACTION_ATTENTION_HEADS="${GENERATOR_INTERACTION_ATTENTION_HEADS:-4}"
GENERATOR_INTERACTION_TOPK="${GENERATOR_INTERACTION_TOPK:-2}"
GENERATOR_INTERACTION_EVERY="${GENERATOR_INTERACTION_EVERY:-1}"
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
export DATA_PATH
export NUM_RETURN_SEQUENCES
export LATENT_LENGTH
export MAX_NEW_TOKENS
export DO_SAMPLE
export SAMPLING_BY
export NOISE_STD
export DROPOUT_P
export GENERATOR_INTERACTION_TYPE
export GENERATOR_INTERACTION_CHECKPOINT
export GENERATOR_INTERACTION_ATTENTION_HEADS
export GENERATOR_INTERACTION_TOPK
export GENERATOR_INTERACTION_EVERY
export MODEL_DTYPE
export SEED

mkdir -p "${LOG_DIR}"

sbatch_args=(
    "--job-name=latenttts-gen-interaction-eval-rm-free"
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

echo "Submitting RM-free generator interaction eval job..."
sbatch "${sbatch_args[@]}" "${SCRIPT_DIR}/run_generator_interaction_eval_rm_free.sbatch"
