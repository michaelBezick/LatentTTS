# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Python-only research codebase for **generator-side interaction learning in latent reasoning models**. The main pipeline is: generate annotated latent data → train a generator-side interaction module → compare best-of-N generator eval with and without interaction. RM-side interaction training/eval is in-tree but experimental; treat generator-side as the mainline workflow.

There is no `pyproject.toml`, `pytest`, linter, or CI; validation is done by running experiment entrypoints.

## Environment

- Python 3.11, install with `pip install -r requirements.txt`.
- The default local shell may be in Conda `base` rather than `latenttts`. Prefer `conda run -n latenttts ...` unless explicitly activated. For activation: `source ~/.bashrc.conda && conda activate latenttts`.
- Missing modules like `transformers` usually mean the command ran in `base` instead of `latenttts`.
- Code expects local checkpoints at runtime, not HF model IDs: `checkpoints/coconut`, `checkpoints/codi`, `checkpoints/colar`, and `checkpoints/latentRM` (see `src/model_registry.py`).
- `checkpoints/` and `latent-data/` are gitignored; these heavy artifacts are not tracked.

## Commands

**Annotate data (local):**
```bash
./run_annotation.sh
# or directly:
python -m src.annotate_data --model_type coconut --n_samples 8 --batch_size 1024 --latent_length 6
```
Produces `.safetensors` shards under `latent-data/coconut/<name>/`. Supports `DATASET_PIECE`, `DATASET_INDICE` env vars for sharding and `NUM_PROCESSES` for multi-GPU.

**Train generator-side interaction (primary):**
```bash
accelerate launch -m src.train_generator_interaction training_args/train_coconut_generator_interaction_verifiable_rl.yaml
```

**Train RM-side interaction (experimental):**
```bash
accelerate launch -m src.train training_args/train_coconut_soft_attention.yaml
```

**Eval — best-of-N baseline (no interaction):**
```bash
python -m src.infer_gpt2_rm --generator_type=coconut --prm_mode=best_of_n --model_dtype=bf16 \
  --prm_id=checkpoints/latentRM --data_path=data/gsm_valid.json \
  --num_return_sequences=8 --generator_communication_type=none --communication_type=none
```

**Eval — best-of-N with learned interaction:**
```bash
python -m src.infer_gpt2_rm --generator_type=coconut --prm_mode=best_of_n --model_dtype=bf16 \
  --prm_id=checkpoints/latentRM --data_path=data/gsm_valid.json \
  --num_return_sequences=8 --generator_communication_type=attention \
  --generator_communication_checkpoint=outputs/coconut-generator-interaction-attention/best
```

For one-off runs and debugging, prefer direct `python -m src.*` invocations over shell wrappers (`run_annotation.sh`, etc.), which hardcode arrays and `CUDA_VISIBLE_DEVICES`.

## SLURM (Zaratan)

Main wrappers in `slurm/zaratan/`: `submit_annotation.sh`, `submit_train_generator_interaction.sh`, `submit_generator_interaction_eval.sh`. Example:
```bash
ACCOUNT=<acc> PARTITION=<part> QOS=<qos> NUM_GPUS=4 \
  ./slurm/zaratan/submit_train_generator_interaction.sh \
    training_args/train_coconut_generator_interaction_verifiable_rl.yaml
```
sbatch scripts always `source ~/.bashrc.conda && conda activate latenttts`. This checkout is local-only; edits must be pushed to GitHub and pulled on Zaratan.

## Architecture

**Data flow:**
```
data/*.json → [annotate_data.py] → latent-data/coconut/<name>/*.safetensors
                                        ↓
                            [train_generator_interaction.py]
                            (frozen generator + trainable communication module)
                                        ↓
                              outputs/<name>/best/
                                        ↓
                              [infer_gpt2_rm.py] → accuracy metrics
```

**Key modules:**
- `src/model_registry.py` — maps model type (`"coconut"`, `"codi"`, `"colar"`) to class, checkpoint dir, and answer extractor.
- `src/generation_mixin.py` — extends HF `GenerationMixin` with latent-token decoding, dropout/noise perturbation, and cross-path communication during generation.
- `src/models/communication.py` — four communication designs: `IdentityCommunication`, `MeanBroadcastCommunication`, `CrossPathAttentionCommunication` (primary), `TopKRouterCommunication`. Attached only when `communication_type != "none"`.
- `src/annotate_data.py` — generates N trajectories per input with perturbations, estimates per-step rewards, writes `.safetensors` shards.
- `src/dataset.py` — `CachedPickleDatasetV2` loads shards; `DataCollatorForContrastiveLatentRM` handles trajectory grouping; `InferenceCollator` handles left-padded batched generation.
- `src/train_generator_interaction.py` — primary training entrypoint; two objectives: `"prm"` (direct path ranking) and `"verifiable_rl"` (policy gradient with RM critic).
- `src/train.py` / `src/trainer.py` — experimental RM-side interaction training.
- `src/infer_gpt2_rm.py` — best-of-N reranking evaluation; reports pass@k, voting accuracy, and selected-response accuracy.

## Config and interface conventions

- Training YAML files are parsed by `fire` + `HfArgumentParser`. Pass the YAML path as a positional argument.
- Inference scripts use `--flag=value` Fire CLIs.
- Key config fields: `communication_type` (`none`/`mean`/`attention`/`router`), `communication_every`, `sampling_by` (`dropout`/`noise`), `objective` (`prm`/`verifiable_rl`), `use_wandb`.
- Shell wrappers use `*_INTERACTION_*` env vars, but the Python/YAML interface still uses `communication_*` keys internally.

## Correctness conventions

- Answer extraction is model-specific (`src/model_registry.py`): `coconut` expects trailing `#` format; `codi` uses regex last-number extraction; `colar` parses `Answer:` with `###` as latent end marker.
- GPT-2-family models use `<|latent|>`, `<|start-latent|>`, `<|end-latent|>` special tokens. CoLaR/LLaMA uses `###`.
- Communication requires grouped trajectories with equal latent-token counts; mismatches raise runtime errors in `apply_communication_to_latent_embeddings`.
- Batched inference uses **left padding**; collators set `tokenizer.padding_side = "left"` and generation depends on it.
- Generator-side interaction eval is implemented only for sampled `best_of_n`; beam search rejects `generator_communication_type != "none"`.
- For communication-aware RM training, keep trajectory grouping intact so collators pass `trajectory_group_size`.
- Best model selection writes the chosen checkpoint to `output_dir/best`.
