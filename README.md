# Latent Thought Interaction for Strong Latent Evolution

`LatentTTS` is the inherited repo name. The active direction of this fork is learning generator-side interaction mechanisms that help parallel latent thoughts evolve into stronger candidates before final selection.

The current first-class workflow is:

1. Generate annotated latent data.
2. Train a generator-side interaction module on top of a frozen base generator.
3. Compare best-of-N evaluation with and without generator-side interaction.

RM-side interaction training and reranking remain in-tree, but they are experimental rather than the mainline path.

## Status

- Supported main workflow: COCONUT generator-side interaction.
- Experimental backbones: CODI and CoLaR.
- Experimental scorer-side workflow: RM-side interaction training and reranking.
- Validation in this repo is done by running experiment entrypoints. There is no `pytest`, linter, or CI workflow in-tree.

## Installation

```bash
conda create -n latenttts python=3.11 -y
conda activate latenttts
pip install -r requirements.txt
```

Recommended environment:

- Python 3.11
- CUDA-capable GPU for training and evaluation
- Local checkpoints under `checkpoints/`

## Required Assets

The code expects local checkpoint directories, not HF model IDs at runtime.

```bash
huggingface-cli download ModalityDance/latent-tts-coconut --local-dir checkpoints/coconut
huggingface-cli download ModalityDance/latent-tts-codi --local-dir checkpoints/codi
huggingface-cli download ModalityDance/latent-tts-colar --local-dir checkpoints/colar
huggingface-cli download ModalityDance/latent-tts-rm --local-dir checkpoints/latentRM
```

Datasets live under `data/`. Large generated artifacts such as `checkpoints/` and `latent-data/` are intentionally untracked.

## Main Workflow

### 1. Annotate latent data

```bash
./run_annotation.sh
```

This writes `.safetensors` shards under `latent-data/coconut/<name>/`.

Useful local overrides:

- `DATASET_PIECE` and `DATASET_INDICE` for sharded annotation
- `NUM_PROCESSES` for multi-GPU annotation

### 2. Train generator-side interaction

Local training:

```bash
accelerate launch -m src.train_generator_interaction \
  training_args/train_coconut_generator_interaction_attention.yaml
```

This workflow freezes the generator, learns only the generator-side interaction module, and selects checkpoints by `eval_max_path_score`.

The current primary config is:

- `training_args/train_coconut_generator_interaction_attention.yaml`

Note: the underlying Python and YAML interface still uses `communication_*` keys for compatibility, even though the user-facing workflow is described as interaction.

### 3. Evaluate baseline vs interaction

Generator-side interaction is implemented only for sampled `best_of_n`. Beam search rejects generator-side interaction.

Baseline evaluation with no generator interaction:

```bash
python -m src.infer_gpt2_rm \
  --generator_type=coconut \
  --prm_mode=best_of_n \
  --model_dtype=bf16 \
  --prm_id=checkpoints/latentRM \
  --data_path=data/gsm_valid.json \
  --num_return_sequences=8 \
  --generator_communication_type=none \
  --communication_type=none
```

Interaction evaluation with a learned generator module:

```bash
python -m src.infer_gpt2_rm \
  --generator_type=coconut \
  --prm_mode=best_of_n \
  --model_dtype=bf16 \
  --prm_id=checkpoints/latentRM \
  --data_path=data/gsm_valid.json \
  --num_return_sequences=8 \
  --generator_communication_type=attention \
  --generator_communication_checkpoint=outputs/coconut-generator-interaction-attention/best \
  --communication_type=none
```

For a fair generator-side comparison, keep the dataset, `num_return_sequences`, seed, PRM checkpoint, and RM-side setting fixed across both runs.

## Zaratan SLURM

The maintained Zaratan wrappers are in `slurm/zaratan/`.

Primary generator-side interaction training:

```bash
ACCOUNT=<your_account> \
PARTITION=<your_partition> \
QOS=<your_qos> \
NUM_GPUS=4 \
./slurm/zaratan/submit_train_generator_interaction.sh \
  training_args/train_coconut_generator_interaction_attention.yaml
```

Primary generator-side interaction eval:

```bash
ACCOUNT=<your_account> \
PARTITION=<your_partition> \
QOS=<your_qos> \
GENERATOR_INTERACTION_TYPE=attention \
GENERATOR_INTERACTION_CHECKPOINT=outputs/coconut-generator-interaction-attention/best \
RM_INTERACTION_TYPE=none \
./slurm/zaratan/submit_generator_interaction_eval.sh
```

Generator baseline eval through the same wrapper:

```bash
ACCOUNT=<your_account> \
PARTITION=<your_partition> \
QOS=<your_qos> \
GENERATOR_INTERACTION_TYPE=none \
GENERATOR_INTERACTION_CHECKPOINT="" \
RM_INTERACTION_TYPE=none \
./slurm/zaratan/submit_generator_interaction_eval.sh
```

Other maintained wrappers:

- `./slurm/zaratan/submit_annotation.sh`
- `./slurm/zaratan/submit_train.sh` for experimental RM-side interaction training
- `./slurm/zaratan/submit_best_of_n_eval.sh` for experimental RM-side reranking eval
- `./slurm/zaratan/submit_mode_collapse_ablation.sh` for experimental collapse analysis

## Experimental Workflows

### RM-side interaction

Train an RM-side interaction model:

```bash
accelerate launch -m src.train training_args/train_coconut_soft_attention.yaml
```

This path is still useful for ablations, but it is not the main workflow of the fork.

### Additional backbones

`codi` and `colar` remain in the model registry and inference code, but they should be treated as experimental support for now.

## Project Structure

```text
LatentTTS/
├── src/
│   ├── model_registry.py
│   ├── annotate_data.py
│   ├── train_generator_interaction.py
│   ├── train.py
│   ├── infer_gpt2.py
│   ├── infer_llama.py
│   ├── infer_gpt2_rm.py
│   ├── generation_mixin.py
│   ├── trainer.py
│   ├── dataset.py
│   ├── utils.py
│   └── models/
│       ├── communication.py
│       ├── coconut.py
│       ├── codi.py
│       ├── colar.py
│       ├── gpt2.py
│       ├── llama.py
│       ├── loss.py
│       └── perturbation.py
├── slurm/
│   └── zaratan/
├── training_args/
│   ├── train_coconut_generator_interaction_attention.yaml
│   └── train_coconut_soft_attention.yaml
├── run_annotation.sh
├── data/
├── checkpoints/
└── requirements.txt
```

## Citation

If you use this repo, please cite the original base paper:

```bibtex
@misc{you2025paralleltesttimescalinglatent,
      title={Parallel Test-Time Scaling for Latent Reasoning Models},
      author={Runyang You and Yongqi Li and Meng Liu and Wenjie Wang and Liqiang Nie and Wenjie Li},
      year={2025},
      eprint={2510.07745},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2510.07745},
}
```
