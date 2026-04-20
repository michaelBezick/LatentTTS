#!/bin/bash

# Sharding: split the training dataset across parallel SLURM jobs.
#   DATASET_PIECE  – fraction of the dataset each shard covers (e.g. 0.25 for 4 shards)
#   DATASET_INDICE – which shard this job processes (0-based)
# Multi-GPU within a single job:
#   NUM_PROCESSES  – number of GPUs / accelerate processes (default: 1)
DATASET_PIECE="${DATASET_PIECE:-1.0}"
DATASET_INDICE="${DATASET_INDICE:-0}"
NUM_PROCESSES="${NUM_PROCESSES:-1}"
USE_WANDB="${USE_WANDB:-False}"
WANDB_PROJECT="${WANDB_PROJECT:-annotation-latent-data}"

default_params=(
    "--save_path=latent-data/coconut"
    "--batch_size=64"
    "--model_type=coconut"
    "--seed=42"
    "--generation_latent_do_sample_by=dropout"
    "--generation_dropout_p=0.2"
    "--use_wandb=${USE_WANDB}"
    "--wandb_project=${WANDB_PROJECT}"
    "--num_latents=8"
    "--n_samples_per_step=8"
)

# Pass sharding args only when actually sharding to keep logs readable.
shard_params=()
if [[ "${DATASET_PIECE}" != "1.0" ]]; then
    shard_params+=("--dataset_piece=${DATASET_PIECE}" "--dataset_indice=${DATASET_INDICE}")
fi

mkdir -p latent-data/coconut
# training data
accelerate launch --mixed_precision=bf16 --main_process_port 0 \
    --num_processes "${NUM_PROCESSES}" \
    -m src.annotate_data \
    ${default_params[@]} \
    ${shard_params[@]} \
    --n_samples=8 \
    --data_path=data/gsm_train.json \
    --name="train${DATASET_PIECE:+"-piece${DATASET_INDICE}"}"


# validation data for evaluation
Ns=(4 64)
for N in ${Ns[@]}; do
    accelerate launch --mixed_precision=bf16 --main_process_port 0 \
        --num_processes "${NUM_PROCESSES}" \
        -m src.annotate_data \
        ${default_params[@]} \
        --n_samples_per_step=0 \
        --n_samples=${N} \
        --data_path=data/gsm_valid.json \
        --name="valid-${N}"
done
