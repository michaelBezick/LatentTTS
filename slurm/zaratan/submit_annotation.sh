#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
LOG_DIR="${REPO_ROOT}/logs/slurm"

ACCOUNT="${ACCOUNT:-}"
PARTITION="${PARTITION:-}"
QOS="${QOS:-}"
TIME_LIMIT="${TIME_LIMIT:-48:00:00}"
NUM_GPUS="${NUM_GPUS:-1}"
GPU_TYPE="${GPU_TYPE:-a100}"
CPUS_PER_TASK="${CPUS_PER_TASK:-16}"
MEMORY="${MEMORY:-120G}"

# Sharding: set NUM_SHARDS > 1 to launch that many parallel annotation jobs,
# each processing 1/NUM_SHARDS of the training dataset.
# e.g.  NUM_SHARDS=4 ./submit_annotation.sh
NUM_SHARDS="${NUM_SHARDS:-4}"

export REPO_ROOT

mkdir -p "${LOG_DIR}"

sbatch_args=(
    "--nodes=1" "--ntasks=1"
    "--chdir=${REPO_ROOT}"
    "--cpus-per-task=${CPUS_PER_TASK}"
    "--time=${TIME_LIMIT}"
    "--mem=${MEMORY}"
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

if [[ "${NUM_SHARDS}" -gt 1 ]]; then
    DATASET_PIECE=$(python3 -c "print(1/${NUM_SHARDS})")
    echo "Submitting ${NUM_SHARDS} sharded annotation jobs (each covers ${DATASET_PIECE} of training data)..."
    for (( IDX=0; IDX<NUM_SHARDS; IDX++ )); do
        export DATASET_PIECE DATASET_INDICE="${IDX}"
        sbatch \
            "${sbatch_args[@]}" \
            "--job-name=latenttts-annotate-shard${IDX}" \
            "--output=${LOG_DIR}/latenttts-annotate-shard${IDX}-%j.out" \
            "${SCRIPT_DIR}/run_annotation.sbatch"
    done
else
    echo "Submitting annotation job..."
    sbatch \
        "${sbatch_args[@]}" \
        "--job-name=latenttts-annotate" \
        "--output=${LOG_DIR}/latenttts-annotate-%j.out" \
        "${SCRIPT_DIR}/run_annotation.sbatch"
fi
