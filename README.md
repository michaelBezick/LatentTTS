<a name="readme-top"></a>

<div align="center">

<h1>Latent Thought Communication for Parallel Latent Reasoning Models</h1>

</div>

---

This repository is a **communication-focused fork** of the implementation for [**Parallel Test-Time Scaling for Latent Reasoning Models**](https://huggingface.co/papers/2510.07745). The original codebase explores multiple latent reasoning trajectories with stochastic sampling and reward-model-guided search. This fork keeps that foundation, but its main goal is to study **latent thought communication** between parallel trajectories.

The current first experiment is **scorer-side soft attention**: generate multiple latent trajectories as usual, then allow those trajectories to interact through a communication module before the latent reward model ranks them. The repository still supports the original stochastic sampling and reward-guided best-of-N / beam-search pipeline for COCONUT, CODI, and CoLaR on benchmarks such as GSM8K Test, GSM8K Hard, and MultiArith.

### 🪐 Key Features

> [!IMPORTANT]
> **🧩 Full Transformers Integration**
> All models (COCONUT, CODI, and CoLaR) are **seamlessly integrated with Transformers**, providing native support for:
> - ✅ **Batch processing** for efficient parallel inference
> - ✅ **Standard Transformers APIs** (`generate()`, `from_pretrained()`, etc.)
> - ✅ **Device management** with `device_map` and multi-GPU support
> - ✅ **Easy integration** into existing Transformers-based workflows
> 
> Simply use `model.generate()` with batch inputs just like any other Transformers model!

🤝 **Latent Thought Communication**
This fork adds configurable communication modules for parallel latent trajectories, starting with **soft attention** between trajectories before reward-model scoring.

🧭 **Stochastic Sampling Methods**
Monte Carlo Dropout and Additive Gaussian Noise are still supported for generating diverse latent reasoning paths.

🌌 **Communication-Aware Latent Reward Model**
The LatentRM can now score trajectories either independently or after cross-trajectory communication, enabling experiments on whether interacting latent paths improve reranking.

🖥️ **Cluster-Friendly Experimentation**
Reusable Zaratan SLURM submit scripts are included for data annotation, multi-GPU training, and best-of-N evaluation.


## 📑 Table of Contents <span id="table-of-contents"></span>

* [🚀 Quick Start](#quick-start)
  * [Installation](#installation)
  * [Data](#data)
  * [Running](#running)
* [✨ How It Works](#how-it-works)
* [📁 Project Structure](#project-structure)
* [🤝 Community](#community)
* [🌱 Acknowledgements](#acknowledgements)
* [🔗 Related Projects](#related)
* [📚 Citation](#citation)

## 🚀 Quick Start <span id="quick-start"></span>



### 1. Installation <span id="installation"></span>

#### **Conda (recommended)**

```bash
conda create -n latenttts python=3.11 -y
conda activate latenttts
pip install -r requirements.txt
```


#### **Hardware Requirements**

* GPU: **Recommended for training and inference (CUDA-compatible)**
* Python: **3.11**
* CUDA: **Compatible with PyTorch 2.8.0**
* Frameworks: **PyTorch 2.8.0, Transformers 4.52.4, Accelerate 1.7.0**

### 2. Preparation <span id="data"></span>

#### **Dataset**

The datasets are located in the `/data` directory. These datasets are obtained from the [coconut](https://github.com/facebookresearch/coconut) project.

#### **Latent Reasoning Models**

Download the pre-trained models from HuggingFace to the `checkpoints/` directory:

```bash
# Download COCONUT model
huggingface-cli download ModalityDance/latent-tts-coconut --local-dir checkpoints/coconut

# Download CODI model
huggingface-cli download ModalityDance/latent-tts-codi --local-dir checkpoints/codi

# Download CoLaR model
huggingface-cli download ModalityDance/latent-tts-colar --local-dir checkpoints/colar

# Optionally download LatentRM (for reward-guided generation)
huggingface-cli download ModalityDance/latent-tts-rm --local-dir checkpoints/latentRM
```

#### **Data Annotation**

First, run the data annotation process to prepare training data for LatentRM:

```bash
./run_annotation.sh
```

This script will:

- Process training data and validation data with specified batch size and sampling parameters
- Generate annotated data for LatentRM training
- Save results to the specified output directory



### 3. Running <span id="running"></span>

#### **Training Configuration**

Configure your training parameters in the `training_args/` directory. For the communication experiment, the main configuration file is `train_coconut_soft_attention.yaml`:

```yaml
run_name: "coconut-soft-attention"
metric_for_best_model: "test_n_64_recall_at_1"
output_dir: "outputs/coconut-soft-attention"
loss_type: "ce"
communication_type: "attention"
```

#### **Model Training**

Navigate to your project directory and launch training:

```bash
cd your/path/to/latent-tts
accelerate launch -m src.train training_args/train_coconut_soft_attention.yaml
```

The training process will:

- Load the annotated data from the previous step
- Train the communication-aware LatentRM with the specified configuration
- Save checkpoints and evaluation results

##### **Zaratan SLURM Submission**

For UMD Zaratan, this repo now includes reusable submit wrappers in `slurm/zaratan/`.

Training the scorer-side soft-attention experiment on a single 4xA100 node:

```bash
ACCOUNT=<your_account> \
PARTITION=<your_partition> \
QOS=<your_qos> \
CONDA_ENV=latenttts \
NUM_GPUS=4 \
./slurm/zaratan/submit_train.sh training_args/train_coconut_soft_attention.yaml
```

Useful overrides:

- `GPU_TYPE=a100` or `GPU_TYPE=""` if your Zaratan partition expects a generic GPU request
- `TIME_LIMIT=24:00:00`
- `MEMORY=240G`
- `MODULES_TO_LOAD="cuda/12.1"`
- `EXTRA_ACCELERATE_ARGS="--mixed_precision bf16"`

Related helpers:

- `./slurm/zaratan/submit_annotation.sh` to launch latent-data generation
- `./slurm/zaratan/submit_best_of_n_eval.sh` to evaluate best-of-N reranking with the communication-aware RM

These wrappers intentionally keep Zaratan-specific settings parameterized, so you can adapt account, partition, QoS, conda environment, and GPU request without editing the scripts themselves.

> [!NOTE]
> Pre-trained checkpoint is available at [HuggingFace](https://huggingface.co/ModalityDance/latent-tts-rm).

#### **Evaluation and Testing**

##### **Majority Voting and Coverage Testing**

Run comprehensive evaluation using majority voting and coverage metrics:

```bash
# For LLaMA model (CoLaR)
./run_tests_llama.sh

# For GPT-2 models (COCONUT and CODI)
./run_tests.sh
```

These scripts will:

- Test different sampling strategies (dropout, noise)
- Evaluate on multiple datasets (GSM8K Test, MultiArith, GSM8K Hard)
- Generate detailed performance metrics including Pass@k, Coverage, and Voting Accuracy

##### **Beam Search and Best-of-N Testing**

For beam search evaluation:

```bash
./run_tts_with_rm.sh
```

This script will:

- Test beam search with different `beam size` (1, 2, 4, 8)
- Test Best-of-N with different `n_return_sequences` (1, 4, 16, 64)
- Generate logs for different configurations



<!--
How It Works (Methods Overview)


GOALS OF THIS SECTION:
1. Provide a clear and brief explanation of how the system or method works.
2. Make this understandable even for readers who do not yet know the technical details.

Points:
1. A high-level description of the system architecture or method.
2. Key components/modules and their roles.
3. A step-by-step workflow of the main process.
4. Figures or diagrams to illustrate the method.

Or:

you can organize in your own way as long as it meets the goals above!!!

-->

## ✨ How It Works <span id="how-it-works"></span>

🪐 **LatentTTS (communication fork)** is built around a modular pipeline for **parallel latent reasoning with interacting trajectories**.  
The original repository focused on generating diverse latent reasoning paths and selecting among them. This fork keeps that structure, but emphasizes research on whether parallel latent paths should **communicate** before selection rather than remain independent.  
The code is organized so communication can be introduced incrementally, with the current implementation focused on scorer-side interaction and future work aimed at generation-time interaction.

At a high level, the workflow proceeds as follows:

1. **Input Processing and Tokenization** — Raw problem inputs are tokenized and prepared with special latent tokens (`<|latent|>`, `<|start-latent|>`, `<|end-latent|>`), creating the scaffold for latent reasoning.  
2. **Parallel Latent Trajectory Generation** — The base model generates multiple reasoning paths in continuous latent space using **Monte Carlo Dropout** or **Additive Gaussian Noise**. This preserves the original repository's ability to explore diverse thought trajectories in parallel.  
3. **Latent Thought Communication** — In this fork's first experiment, grouped trajectories for the same prompt are passed through a communication module before reward-model scoring. The current focus is **soft attention**, but the communication layer is configurable so other mechanisms can be tested later.  
4. **Communication-Aware Reward-Guided Selection** — The **Latent Reward Model (LatentRM)** scores each trajectory after optional communication. Those scores are then used for **best-of-N** reranking and can be compared against the baseline independent-trajectory scorer. Generation-time communication remains a future extension.





## 📁 Project Structure <span id="project-structure"></span>

```
latent-tts/
├── src/                   # Source code
│   ├── models/            # Model implementations
│   │   ├── coconut.py     # COCONUT model
│   │   ├── codi.py        # CODI model
│   │   ├── colar.py       # CoLaR model
│   │   ├── communication.py # Trajectory communication modules
│   │   ├── gpt2.py        # GPT-2 base models
│   │   ├── llama.py       # LLaMA base models
│   │   ├── loss.py        # Loss functions
│   │   └── perturbation.py # Perturbation methods
│   ├── annotate_data.py   # Data annotation script
│   ├── train.py           # latentRM training script
│   ├── trainer.py         # Training utilities
│   ├── infer_gpt2.py      # GPT-2 inference
│   ├── infer_llama.py     # LLaMA inference
│   ├── infer_gpt2_rm.py   # Communication-aware latentRM inference
│   ├── dataset.py         # Dataset handling
│   ├── generation_mixin.py # Generation utilities
│   ├── paths.py           # Path utilities
│   └── utils.py           # Utility functions
├── slurm/
│   └── zaratan/           # Zaratan cluster submission scripts
├── training_args/         # Training configurations
│   ├── train_coconut.yaml # Baseline COCONUT training config
│   └── train_coconut_soft_attention.yaml # Communication-focused training config
├── data/                  # Dataset files
├── checkpoints/           # Model checkpoints
│   └── latentRM/          # latentRM checkpoint
|   └── coconut/
├── run_annotation.sh      # Data annotation script
├── run_tests.sh           # GPT-2 evaluation script
├── run_tests_llama.sh     # LLaMA evaluation script
├── run_tts_with_rm.sh     # Beam search evaluation script
└── requirements.txt       # Python dependencies
```

## 🤝 Join the Community <span id="community"></span>

We welcome researchers and engineers interested in **latent thought communication**, latent-space reasoning, and test-time scaling. If you are exploring interacting latent trajectories, communication-aware reranking, or generation-time communication mechanisms, this fork is intended to be an easy place to experiment and contribute.

> [!TIP]
> 📄 Explore the paper on [**Hugging Face Papers**](https://huggingface.co/papers/2510.07745) — it includes community discussions, citation tools, and related resources. If you find our work insightful, please consider giving it an **upvote** to support further research!

## 🌱 **Acknowledgements** <span id="acknowledgements"></span>

We would like to thank the contributors, open-source projects, and research communities whose work made **LatentTTS** possible. This fork builds directly on that foundation and extends it toward latent thought communication. We also acknowledge helpful discussions and support from the members of **Modality Dance Group** and the open-source community.

This project is licensed under the **MIT License**. Please refer to the LICENSE file for more details.


## 🔗 **Related Projects** <span id="related"></span>

### 📄 Related Papers

- **[LLMs are Single-threaded Reasoners: Demystifying the Working Mechanism of Soft Thinking](https://arxiv.org/abs/2508.03440)**  
  Check out stochastic soft thinking!

### 🌟 Awesome Collections

- **[Awesome Latent Space](https://github.com/YU-deep/Awesome-Latent-Space)**  
  A curated collection of resources on latent space methods and applications.

- **[Awesome Latent CoT](https://github.com/EIT-NLP/Awesome-Latent-CoT)**  
  A comprehensive list of latent chain-of-thought reasoning resources.

- **[Awesome Efficient Reasoning](https://github.com/hemingkx/Awesome-Efficient-Reasoning)**  
  A collection of efficient reasoning methods and techniques.


## 📚 **Citation** <span id="citation"></span>

If you use **LatentTTS** in your research or applications, please consider citing:

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

---

<div align="center">

<a href="https://github.com/ModalityDance/LatentTTS">
  <img src="https://img.shields.io/badge/⭐ Star%20us%20on%20GitHub-181717?style=for-the-badge&logo=github&logoColor=white" />
</a>

<a href="https://github.com/ModalityDance/LatentTTS/issues">
  <img src="https://img.shields.io/badge/🐞 Report%20Issues-e74c3c?style=for-the-badge&logo=github" />
</a>


</div>
