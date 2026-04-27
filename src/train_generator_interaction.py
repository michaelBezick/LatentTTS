import json
import math
import os
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Literal

import datasets
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from safetensors.torch import save_file
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, HfArgumentParser, get_scheduler

try:
    from peft import LoraConfig, get_peft_model
except ImportError:  # pragma: no cover - PEFT is a required repo dependency.
    LoraConfig = None
    get_peft_model = None

from src.generation_mixin import LatentGenerationConfig
from src.model_registry import MODELS
from src.models.coconut import COCONUTGPT2
from src.models.gpt2 import COCONUTGPT2ForTokenClassification
from src.models.communication import (
    CrossPathAttentionCommunication,
    build_communication_module,
    restore_communication_module_from_checkpoint,
)
from src.models.loss import DiversityPenaltyLoss
from src.utils import InferenceCollator, add_noise, disable_dropout, enable_dropout, set_dropout_p, set_seed


@dataclass
class GeneratorInteractionConfig:
    run_name: str = field(default="coconut-generator-interaction")
    output_dir: str = field(default="outputs/coconut-generator-interaction")
    model_id: str = field(default="checkpoints/coconut")
    prm_id: str = field(default="checkpoints/latentRM")
    objective: Literal["prm", "verifiable_rl", "ce"] = field(default="prm")
    train_data_path: str = field(default="data/gsm_train.json")
    valid_data_path: str = field(default="data/gsm_valid.json")
    model_dtype: Literal["fp32", "fp16", "bf16"] = field(default="bf16")
    seed: int = field(default=42)
    num_return_sequences: int = field(default=8)
    latent_length: int = field(default=6)
    max_new_tokens: int = field(default=128)
    communication_type: Literal["mean", "attention", "router", "gated_router"] = field(default="attention")
    communication_attention_heads: int = field(default=4)
    communication_topk: int = field(default=2)
    communication_gate_bias: float = field(default=-4.0)
    communication_interaction_scale: float = field(default=1.0)
    communication_every: int = field(default=1)
    init_communication_from: str = field(default="")
    sampling_by: Literal["dropout", "noise"] = field(default="dropout")
    dropout_p: float = field(default=0.2)
    noise_std: float = field(default=0.1)
    per_device_train_batch_size: int = field(default=8)
    per_device_eval_batch_size: int = field(default=8)
    per_device_forward_batch_size: int | None = field(default=None)
    gradient_accumulation_steps: int = field(default=1)
    learning_rate: float = field(default=1.0e-4)
    weight_decay: float = field(default=0.0)
    warmup_ratio: float = field(default=0.0)
    lr_scheduler_type: str = field(default="cosine")
    num_train_epochs: int = field(default=3)
    max_train_steps: int | None = field(default=None)
    max_grad_norm: float = field(default=1.0)
    logging_steps: int = field(default=10)
    eval_frequency: int = field(default=1)
    eval_on_start: bool = field(default=True)
    eval_only: bool = field(default=False)
    use_wandb: bool = field(default=False)
    wandb_project: str = field(default="latenttts-generator-interaction")
    wandb_entity: str = field(default="")
    metric_for_best_model: str = field(default="eval_max_path_score")
    score_temperature: float = field(default=0.25)
    reward_baseline: Literal["leave_one_out", "group_mean"] = field(default="leave_one_out")
    normalize_advantages: bool = field(default=True)
    use_prm_advantages: bool = field(default=False)
    freeze_prm: bool = field(default=False)
    train_generator_lora: bool = field(default=False)
    lora_r: int = field(default=8)
    lora_alpha: int = field(default=16)
    lora_dropout: float = field(default=0.05)
    lora_target_modules: list[str] = field(default_factory=lambda: ["c_attn", "c_proj"])
    ce_prm_weight: float = field(default=0.5)
    diversity_penalty_weight: float = field(default=0.1)
    anchor_weight: float = field(default=0.5)
    use_dense_credit: bool = field(default=False)
    traj_credit_weight: float = field(default=1.0)
    dataloader_num_workers: int = field(default=0)
    sort_by_len: bool = field(default=True)
    max_train_samples: int | None = field(default=None)
    max_eval_samples: int | None = field(default=None)


@dataclass
class LatentRolloutOutput:
    input_ids: torch.LongTensor
    attention_mask: torch.LongTensor
    latent_thoughts: torch.FloatTensor
    pre_communication_latent_thoughts: torch.FloatTensor
    post_communication_latent_thoughts: torch.FloatTensor
    attention_weights: torch.FloatTensor | None = None  # [B, L_comm, N, N]


def gaussian_latent_log_probs(
    samples: torch.Tensor,
    means: torch.Tensor,
    std: float,
) -> torch.Tensor:
    if std <= 0:
        raise ValueError(f"noise_std must be positive for verifiable RL, but got {std}")
    normalized = (samples.detach().float() - means.float()) / std
    log_normalizer = math.log(std) + 0.5 * math.log(2.0 * math.pi)
    # Per-dim mean, not sum. Summing over (latent_length, hidden) makes the
    # gradient into `means` scale as (1/std^2) * sqrt(T * D) — with T=6, D=768,
    # std=0.1, that saturates max_grad_norm and the clipped direction is
    # dominated by whichever noise realization happened to align with its
    # advantage. Mean keeps the per-element magnitude bounded and leaves the
    # trust region to max_grad_norm + anchor/KL, which is what actually
    # controls drift.
    return (-0.5 * normalized.pow(2) - log_normalizer).mean(dim=(-1, -2))


def gaussian_latent_log_probs_per_step(
    samples: torch.Tensor,
    means: torch.Tensor,
    std: float,
) -> torch.Tensor:
    """Returns [B*N, L] — log-prob averaged over hidden dim only, one value per latent step."""
    if std <= 0:
        raise ValueError(f"noise_std must be positive for verifiable RL, but got {std}")
    normalized = (samples.detach().float() - means.float()) / std
    log_normalizer = math.log(std) + 0.5 * math.log(2.0 * math.pi)
    return (-0.5 * normalized.pow(2) - log_normalizer).mean(dim=-1)  # [B*N, L]


def parse_args(*args, **kwargs) -> GeneratorInteractionConfig:
    parser = HfArgumentParser(GeneratorInteractionConfig)
    if len(args) == 1:
        if len(kwargs) > 0:
            raise ValueError(f"Invalid arguments: {args} and {kwargs}")
        if args[0].endswith(".yaml"):
            return parser.parse_yaml_file(args[0])[0]
        if args[0].endswith(".json"):
            return parser.parse_json_file(args[0])[0]
        raise ValueError(f"Invalid config path: {args[0]}")
    if len(args) == 0:
        return parser.parse_dict(kwargs)[0]
    raise ValueError(f"Invalid arguments: {args}")


def resolve_torch_dtype(model_dtype: str) -> torch.dtype | None:
    if model_dtype == "bf16":
        return torch.bfloat16
    if model_dtype == "fp16":
        return torch.float16
    return None


def get_base_generator_model(model: torch.nn.Module) -> torch.nn.Module:
    if hasattr(model, "module"):
        model = model.module
    if hasattr(model, "get_base_model"):
        return model.get_base_model()
    return model


def format_numeric_answer(answer) -> str:
    value = float(answer)
    if value.is_integer():
        return str(int(value))
    return f"{value:.10g}"


def unique_parameters(parameters):
    seen = set()
    unique = []
    for param in parameters:
        if id(param) in seen:
            continue
        seen.add(id(param))
        unique.append(param)
    return unique


def safe_parse_answer(answer: str) -> float | None:
    try:
        return float(answer.replace(",", ""))
    except Exception:
        return None


def prepare_coconut_dataset(
    data_path: str,
    tokenizer: AutoTokenizer,
    sort_by_len: bool,
    max_samples: int | None,
) -> datasets.Dataset:
    dataset = datasets.Dataset.from_json(data_path)
    dataset = dataset.map(lambda x, idx: {"idx": idx}, with_indices=True)
    dataset = dataset.map(
        lambda x: {
            "question": x["question"] + "\n<|start-latent|>",
            "answer": safe_parse_answer(x["answer"]),
        }
    )
    dataset = dataset.filter(lambda x: x["answer"] is not None)
    dataset = dataset.map(lambda x: tokenizer(x["question"]), batched=True)
    if sort_by_len:
        dataset = dataset.map(lambda x: {"length": len(x["input_ids"])})
        dataset = dataset.sort("length")
        dataset = dataset.remove_columns("length")
    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    return dataset


def rollout_latent_paths(
    model: COCONUTGPT2,
    communication_module: torch.nn.Module,
    input_ids: torch.LongTensor,
    attention_mask: torch.LongTensor,
    latent_token_id: int,
    num_return_sequences: int,
    latent_length: int,
    sampling_by: Literal["dropout", "noise"],
    noise_std: float,
    communication_every: int,
    collect_attention_weights: bool = False,
) -> LatentRolloutOutput:
    base_model = get_base_generator_model(model)
    base_batch_size = input_ids.shape[0]
    total_batch_size = base_batch_size * num_return_sequences
    input_ids = input_ids.repeat_interleave(num_return_sequences, dim=0)
    attention_mask = attention_mask.repeat_interleave(num_return_sequences, dim=0)
    inputs_embeds = base_model.get_input_embeddings()(torch.where(input_ids == latent_token_id, 0, input_ids))
    alive_mask = torch.ones(
        base_batch_size, num_return_sequences, dtype=torch.bool, device=input_ids.device
    )

    latent_steps = []
    pre_communication_steps = []
    post_communication_steps = []
    attention_weight_steps = []
    dropout_sampling = sampling_by == "dropout"
    if dropout_sampling:
        enable_dropout(model)
    try:
        for latent_step in range(latent_length):
            transformer_outputs = base_model.transformer(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )
            next_latent = transformer_outputs.last_hidden_state[:, -1, :]
            pre_communication_steps.append(next_latent)

            if latent_step % max(1, communication_every) == 0:
                grouped_latent = next_latent.view(base_batch_size, num_return_sequences, -1)
                result = communication_module(
                    grouped_latent,
                    alive_mask=alive_mask,
                    step_idx=latent_step,
                    return_weights=collect_attention_weights,
                )
                if collect_attention_weights:
                    comm_out, attn_w = result
                    if attn_w is not None:
                        attention_weight_steps.append(attn_w.detach())
                else:
                    comm_out = result
                next_latent = comm_out.view(total_batch_size, -1)
            post_communication_steps.append(next_latent)

            if sampling_by == "noise":
                latent_to_append = add_noise(next_latent, std=noise_std)
            else:
                latent_to_append = next_latent

            next_ids = torch.full(
                (total_batch_size, 1),
                latent_token_id,
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
            next_attention = torch.ones(
                (total_batch_size, 1), dtype=attention_mask.dtype, device=attention_mask.device
            )
            input_ids = torch.cat([input_ids, next_ids], dim=1)
            attention_mask = torch.cat([attention_mask, next_attention], dim=1)
            inputs_embeds = torch.cat([inputs_embeds, latent_to_append.unsqueeze(1)], dim=1)
            latent_steps.append(latent_to_append)
    finally:
        if dropout_sampling:
            disable_dropout(model)

    attn_stack = torch.stack(attention_weight_steps, dim=1) if attention_weight_steps else None
    return LatentRolloutOutput(
        input_ids=input_ids,
        attention_mask=attention_mask,
        latent_thoughts=torch.stack(latent_steps, dim=1),
        pre_communication_latent_thoughts=torch.stack(pre_communication_steps, dim=1),
        post_communication_latent_thoughts=torch.stack(post_communication_steps, dim=1),
        attention_weights=attn_stack,
    )


class GeneratorInteractionTrainer:
    def __init__(self, args: GeneratorInteractionConfig):
        self.args = args
        self.accelerator = Accelerator(
            mixed_precision=("no" if args.model_dtype == "fp32" else args.model_dtype),
            gradient_accumulation_steps=args.gradient_accumulation_steps,
        )
        set_seed(args.seed + self.accelerator.process_index)
        self.torch_dtype = resolve_torch_dtype(args.model_dtype)

        self.tokenizer = AutoTokenizer.from_pretrained(args.model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.answer_extractor = MODELS["coconut"]["answer_extractor"]
        self.latent_id = self.tokenizer.convert_tokens_to_ids("<|latent|>")
        self.start_id = self.tokenizer.convert_tokens_to_ids("<|start-latent|>")
        self.end_id = self.tokenizer.convert_tokens_to_ids("<|end-latent|>")
        if self.latent_id == self.start_id or self.latent_id == self.end_id:
            raise ValueError(
                f"latent_id={self.latent_id}, start_id={self.start_id}, end_id={self.end_id} must all differ"
            )
        if args.objective == "verifiable_rl":
            if args.sampling_by != "noise":
                raise ValueError("verifiable_rl requires sampling_by='noise' so latent log-probs are defined")
            if args.num_return_sequences <= 1:
                raise ValueError("verifiable_rl requires num_return_sequences > 1")
        if args.objective == "ce" and args.num_return_sequences <= 1:
            raise ValueError("ce requires num_return_sequences > 1 so the PRM selector can rank paths")
        self.train_prm = args.objective == "verifiable_rl" and not args.freeze_prm
        self.train_generator = args.objective == "ce" and args.train_generator_lora

        self.generator = COCONUTGPT2.from_pretrained(
            args.model_id,
            latent_id=self.latent_id,
            latent_start_id=self.start_id,
            latent_end_id=self.end_id,
            pad_token_id=self.tokenizer.pad_token_id,
            torch_dtype=self.torch_dtype,
        )
        self.communication_module = build_communication_module(
            communication_type=args.communication_type,
            d_model=self.generator.config.hidden_size,
            n_heads=args.communication_attention_heads,
            topk=args.communication_topk,
            gate_bias=args.communication_gate_bias,
            interaction_scale=args.communication_interaction_scale,
        )
        self.generator.communication_module = self.communication_module
        if args.init_communication_from:
            restored = restore_communication_module_from_checkpoint(
                module_owner=self.generator,
                checkpoint_path=args.init_communication_from,
            )
            if not restored:
                raise ValueError(
                    f"Could not restore generator interaction module from {args.init_communication_from}"
                )

        if self.train_generator:
            if get_peft_model is None or LoraConfig is None:
                raise ImportError("PEFT is required for train_generator_lora=True")
            lora_config = LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                target_modules=args.lora_target_modules,
                fan_in_fan_out=True,
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.generator = get_peft_model(self.generator, lora_config)
            self.generator.get_base_model().communication_module = self.communication_module
        else:
            for param in self.generator.parameters():
                param.requires_grad = False
        for param in self.communication_module.parameters():
            param.requires_grad = True
        self.generator.eval()

        if args.sampling_by == "dropout":
            set_dropout_p(self.generator, args.dropout_p)

        self.prm = COCONUTGPT2ForTokenClassification.from_pretrained(
            args.prm_id,
            latent_id=self.latent_id,
            latent_start_id=self.start_id,
            latent_end_id=self.end_id,
            pad_token_id=self.tokenizer.pad_token_id,
            torch_dtype=self.torch_dtype,
        )
        self.prm.config.use_cache = False
        for param in self.prm.parameters():
            param.requires_grad = self.train_prm
        if self.train_prm:
            self.prm.train()
        else:
            self.prm.eval()
        self.verifiable_generation_config = LatentGenerationConfig(
            max_new_tokens=args.max_new_tokens,
            latent_length=args.latent_length,
            latent_do_sample=False,
            communication_type="none",
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
        )

        self.data_collator = InferenceCollator(self.tokenizer)
        train_dataset = prepare_coconut_dataset(
            data_path=args.train_data_path,
            tokenizer=self.tokenizer,
            sort_by_len=args.sort_by_len,
            max_samples=args.max_train_samples,
        )
        if len(train_dataset) == 0:
            raise ValueError("Training dataset is empty")
        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=args.per_device_train_batch_size,
            collate_fn=self.data_collator,
            shuffle=True,
            num_workers=args.dataloader_num_workers,
        )

        self.eval_dataloader = None
        if args.valid_data_path:
            eval_dataset = prepare_coconut_dataset(
                data_path=args.valid_data_path,
                tokenizer=self.tokenizer,
                sort_by_len=args.sort_by_len,
                max_samples=args.max_eval_samples,
            )
            if len(eval_dataset) > 0:
                self.eval_dataloader = DataLoader(
                    eval_dataset,
                    batch_size=args.per_device_eval_batch_size,
                    collate_fn=self.data_collator,
                    shuffle=False,
                    num_workers=args.dataloader_num_workers,
                )

        trainable_params = [
            param for param in self.communication_module.parameters() if param.requires_grad
        ]
        if self.train_generator:
            trainable_params.extend(param for param in self.generator.parameters() if param.requires_grad)
        if self.train_prm:
            trainable_params.extend(param for param in self.prm.parameters() if param.requires_grad)
        trainable_params = unique_parameters(trainable_params)
        if len(trainable_params) == 0:
            raise ValueError("No trainable generator interaction parameters found")
        self.optimizer = AdamW(
            trainable_params,
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )

        per_process_train_batches = max(
            1,
            math.ceil(len(self.train_dataloader) / self.accelerator.num_processes),
        )
        self.num_update_steps_per_epoch = max(
            1,
            math.ceil(per_process_train_batches / self.accelerator.gradient_accumulation_steps),
        )
        if args.max_train_steps is not None:
            if args.max_train_steps <= 0:
                raise ValueError("max_train_steps must be positive when provided")
            self.max_train_steps = args.max_train_steps
            self.num_train_epochs = max(
                1,
                math.ceil(self.max_train_steps / self.num_update_steps_per_epoch),
            )
        else:
            self.max_train_steps = self.num_update_steps_per_epoch * args.num_train_epochs
            self.num_train_epochs = args.num_train_epochs
        self.num_warmup_steps = int(self.max_train_steps * args.warmup_ratio)
        self.lr_scheduler = get_scheduler(
            name=args.lr_scheduler_type,
            optimizer=self.optimizer,
            num_warmup_steps=self.num_warmup_steps,
            num_training_steps=self.max_train_steps,
        )

        prepare_items = [self.communication_module]
        prepare_names = ["communication_module"]
        if self.train_generator:
            prepare_items.append(self.generator)
            prepare_names.append("generator")
        if self.train_prm:
            prepare_items.append(self.prm)
            prepare_names.append("prm")
        prepare_items.extend([self.optimizer, self.train_dataloader])
        prepare_names.extend(["optimizer", "train_dataloader"])
        if self.eval_dataloader is not None:
            prepare_items.append(self.eval_dataloader)
            prepare_names.append("eval_dataloader")
        prepare_items.append(self.lr_scheduler)
        prepare_names.append("lr_scheduler")
        prepared = self.accelerator.prepare(*prepare_items)
        for name, value in zip(prepare_names, prepared):
            setattr(self, name, value)
        self.generator = self.generator.to(self.accelerator.device)
        if self.prm is not None and args.objective != "verifiable_rl":
            self.prm = self.prm.to(self.accelerator.device)
        self.diversity_loss_fct = DiversityPenaltyLoss()
        self.metric_names = self.get_metric_names()
        if args.metric_for_best_model not in {f"eval_{name}" for name in self.metric_names}:
            raise ValueError(
                f"metric_for_best_model={args.metric_for_best_model} is not supported for objective={args.objective}. "
                f"Available metrics: {[f'eval_{name}' for name in self.metric_names]}"
            )
        self.best_metric = None
        self.global_step = 0
        self.wandb_run = None
        self.use_wandb = args.use_wandb or os.getenv("WANDB_MODE") is not None

        if self.accelerator.is_main_process:
            os.makedirs(args.output_dir, exist_ok=True)
            with open(
                os.path.join(args.output_dir, "train_config.json"),
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(asdict(args), f, indent=2)
            if self.use_wandb:
                try:
                    import wandb
                except ImportError as exc:
                    raise ImportError(
                        "W&B logging requested for generator interaction training, but wandb is not installed."
                    ) from exc
                wandb_init_kwargs = {
                    "project": args.wandb_project,
                    "name": args.run_name,
                    "config": asdict(args),
                }
                if args.wandb_entity:
                    wandb_init_kwargs["entity"] = args.wandb_entity
                self.wandb_run = wandb.init(**wandb_init_kwargs)
        self.accelerator.wait_for_everyone()

    def get_metric_names(self) -> list[str]:
        if self.args.objective == "verifiable_rl":
            return [
                "loss",
                "policy_loss",
                "selector_loss",
                "diversity_loss",
                "anchor_loss",
                "mean_reward",
                "coverage",
                "voting_accuracy",
                "selected_accuracy",
            ]
        if self.args.objective == "ce":
            return [
                "loss",
                "ce_loss",
                "prm_weighted_ce_loss",
                "diversity_loss",
                "anchor_loss",
                "mean_path_score",
                "max_path_score",
                "selected_accuracy",
            ]
        return [
            "loss",
            "score_loss",
            "diversity_loss",
            "anchor_loss",
            "mean_path_score",
            "max_path_score",
        ]

    def init_metric_sums(self) -> dict[str, torch.Tensor]:
        return {
            name: torch.zeros((), device=self.accelerator.device, dtype=torch.float32)
            for name in self.metric_names
        }

    def prepare_batch_inputs(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        batch_inputs = {}
        for key in ["input_ids", "attention_mask", "answer"]:
            if key not in batch:
                continue
            value = batch[key]
            if isinstance(value, torch.Tensor):
                batch_inputs[key] = value.to(self.accelerator.device)
            else:
                batch_inputs[key] = value
        return batch_inputs

    def iter_forward_batches(self, batch: dict[str, torch.Tensor]):
        batch_size = batch["input_ids"].shape[0]
        chunk_size = self.args.per_device_forward_batch_size
        if chunk_size is None or chunk_size <= 0 or chunk_size >= batch_size:
            yield batch, batch_size
            return

        for start in range(0, batch_size, chunk_size):
            end = min(start + chunk_size, batch_size)
            yield {key: value[start:end] for key, value in batch.items()}, end - start

    def aggregate_chunk_metrics(self, metric_sums: dict[str, torch.Tensor], batch_size: int):
        return {key: value / batch_size for key, value in metric_sums.items()}

    def compute_reward_advantages(self, rewards: torch.Tensor) -> torch.Tensor:
        if self.args.reward_baseline == "group_mean":
            advantages = rewards - rewards.mean(dim=1, keepdim=True)
        else:
            baselines = (rewards.sum(dim=1, keepdim=True) - rewards) / (self.args.num_return_sequences - 1)
            advantages = rewards - baselines
        if self.args.normalize_advantages:
            # Divide by per-group std so binary-reward groups don't dominate
            # mixed-reward ones. Clamp guards groups where all paths got the
            # same reward (std=0 → advantage is already zero).
            advantages = advantages / advantages.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
        return advantages

    def compute_selector_scores(
        self,
        rollout: LatentRolloutOutput,
        detach_latents: bool,
    ) -> torch.Tensor:
        latent_embeds = rollout.latent_thoughts.detach() if detach_latents else rollout.latent_thoughts
        prm_outputs = self.prm(
            input_ids=rollout.input_ids,
            attention_mask=rollout.attention_mask,
            latent_embeds=latent_embeds,
            use_cache=False,
            return_dict=True,
        )
        prm_scores = prm_outputs.logits.squeeze(-1)
        latent_mask = rollout.input_ids == self.latent_id
        prm_scores = torch.where(latent_mask, prm_scores, 0).sum(dim=-1)
        return prm_scores.view(-1, self.args.num_return_sequences)

    def evaluate_verifiable_rewards(
        self,
        batch: dict[str, torch.Tensor],
        rollout: LatentRolloutOutput,
        selector_scores: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        input_ids = rollout.input_ids
        latent_mask = input_ids == self.latent_id
        generator = self.accelerator.unwrap_model(self.generator)
        base_model = get_base_generator_model(generator)
        inputs_embeds = base_model.get_input_embeddings()(
            torch.where(input_ids == self.latent_id, 0, input_ids)
        )
        inputs_embeds = inputs_embeds.clone()
        inputs_embeds[latent_mask] = rollout.latent_thoughts.detach().to(
            dtype=inputs_embeds.dtype
        ).reshape(-1, inputs_embeds.shape[-1])

        with torch.no_grad():
            output = generator.generate(
                input_ids=input_ids,
                attention_mask=rollout.attention_mask,
                inputs_embeds=inputs_embeds,
                generation_config=self.verifiable_generation_config,
                num_return_sequences=1,
                use_cache=True,
            )

        continuation = output[:, input_ids.shape[1] :]
        text_output = self.tokenizer.batch_decode(continuation, skip_special_tokens=True)
        extracted_answers = [self.answer_extractor(text) for text in text_output]
        rewards = torch.tensor(
            [
                extracted_answers[i] == batch["answer"][i // self.args.num_return_sequences]
                for i in range(len(extracted_answers))
            ],
            device=self.accelerator.device,
            dtype=torch.float32,
        ).view(-1, self.args.num_return_sequences)
        voting_result = []
        for group_idx in range(rewards.shape[0]):
            start = group_idx * self.args.num_return_sequences
            end = start + self.args.num_return_sequences
            selected_answer = Counter(extracted_answers[start:end]).most_common(1)[0][0]
            voting_result.append(selected_answer == batch["answer"][group_idx])
        voting_accuracy = torch.tensor(
            voting_result,
            device=self.accelerator.device,
            dtype=torch.float32,
        ).mean()
        if selector_scores is None:
            selected_accuracy = voting_accuracy
        else:
            selected_indices = selector_scores.argmax(dim=1, keepdim=True)
            selected_accuracy = rewards.gather(1, selected_indices).mean()
        reward_metrics = {
            "mean_reward": rewards.mean(),
            "coverage": rewards.max(dim=1).values.mean(),
            "voting_accuracy": voting_accuracy,
            "selected_accuracy": selected_accuracy,
        }
        return rewards, reward_metrics

    def build_answer_targets(self, batch: dict[str, torch.Tensor], num_paths: int) -> tuple[torch.Tensor, torch.Tensor]:
        target_ids = []
        for answer in batch["answer"]:
            text = "#" + format_numeric_answer(answer)
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            if self.tokenizer.eos_token_id is not None:
                ids = ids + [self.tokenizer.eos_token_id]
            target_ids.extend([ids] * num_paths)

        max_len = max(len(ids) for ids in target_ids)
        padded = torch.full(
            (len(target_ids), max_len),
            self.tokenizer.pad_token_id,
            dtype=torch.long,
            device=self.accelerator.device,
        )
        mask = torch.zeros_like(padded)
        for row, ids in enumerate(target_ids):
            values = torch.tensor(ids, dtype=torch.long, device=self.accelerator.device)
            padded[row, : values.numel()] = values
            mask[row, : values.numel()] = 1
        return padded, mask

    def compute_answer_ce_losses(
        self,
        batch: dict[str, torch.Tensor],
        rollout: LatentRolloutOutput,
    ) -> torch.Tensor:
        input_ids = rollout.input_ids
        latent_mask = input_ids == self.latent_id
        base_model = get_base_generator_model(self.generator)
        prefix_embeds = base_model.get_input_embeddings()(
            torch.where(input_ids == self.latent_id, 0, input_ids)
        )
        prefix_embeds = prefix_embeds.clone()
        prefix_embeds[latent_mask] = rollout.latent_thoughts.to(
            dtype=prefix_embeds.dtype
        ).reshape(-1, prefix_embeds.shape[-1])

        target_ids, target_mask = self.build_answer_targets(
            batch,
            num_paths=self.args.num_return_sequences,
        )
        target_embeds = base_model.get_input_embeddings()(target_ids)
        inputs_embeds = torch.cat([prefix_embeds, target_embeds], dim=1)
        attention_mask = torch.cat([rollout.attention_mask, target_mask], dim=1)
        labels = torch.full_like(
            torch.cat([input_ids, target_ids], dim=1),
            -100,
            dtype=torch.long,
        )
        labels[:, input_ids.shape[1] :] = torch.where(
            target_mask.bool(),
            target_ids,
            torch.full_like(target_ids, -100),
        )

        outputs = self.generator(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=None,
            use_cache=False,
            return_dict=True,
        )
        shift_logits = outputs.logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        flat_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction="none",
        ).view_as(shift_labels)
        valid_mask = shift_labels.ne(-100)
        return (flat_loss * valid_mask).sum(dim=-1) / valid_mask.sum(dim=-1).clamp_min(1)

    def evaluate_batch_objective(self, batch: dict[str, torch.Tensor]):
        batch_size = batch["input_ids"].shape[0]
        metric_sums = self.init_metric_sums()

        for chunk_inputs, chunk_size in self.iter_forward_batches(batch):
            with self.accelerator.autocast():
                _, metrics = self.compute_batch_objective(chunk_inputs)
            for key in metric_sums:
                metric_sums[key] += metrics[key].detach().float() * chunk_size

        return self.aggregate_chunk_metrics(metric_sums, batch_size)

    def backward_batch_objective(self, batch: dict[str, torch.Tensor]):
        batch_size = batch["input_ids"].shape[0]
        metric_sums = self.init_metric_sums()

        for chunk_inputs, chunk_size in self.iter_forward_batches(batch):
            with self.accelerator.autocast():
                loss, metrics = self.compute_batch_objective(chunk_inputs)
            self.accelerator.backward(loss * (chunk_size / batch_size))
            for key in metric_sums:
                metric_sums[key] += metrics[key].detach().float() * chunk_size

        return self.aggregate_chunk_metrics(metric_sums, batch_size)

    def log_wandb(self, metrics: dict[str, float], step: int | None = None):
        if not self.accelerator.is_main_process or self.wandb_run is None:
            return
        if step is None:
            self.wandb_run.log(metrics)
        else:
            self.wandb_run.log(metrics, step=step)

    def compute_2d_credit_weights(
        self,
        rollout: LatentRolloutOutput,
        advantages: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute time and trajectory credit weights for dense credit assignment.

        Returns:
            w_time: [B, L] normalized over latent steps (sums to 1), based on attention sharpness.
            w_traj: [B, N] normalized so mean=1 over trajectories, based on cross-path credit.
        Both tensors are detached.
        """
        B, N = advantages.shape
        L = rollout.latent_thoughts.shape[1]
        attn = rollout.attention_weights   # [B, L_comm, N, N] or None
        comm_every = max(1, self.args.communication_every)
        device = advantages.device

        if attn is not None:
            eps = 1e-8
            H = -(attn * (attn + eps).log()).sum(dim=-1)          # [B, L_comm, N]
            max_H = math.log(max(N, 2))
            sharpness = (1.0 - H / max_H).mean(dim=-1)            # [B, L_comm]
            sharpness_expanded = sharpness.repeat_interleave(comm_every, dim=1)[:, :L]  # [B, L]
            w_time = torch.softmax(sharpness_expanded, dim=-1)     # [B, L]
        else:
            w_time = torch.full((B, L), 1.0 / L, device=device)

        if attn is not None and self.args.traj_credit_weight > 0:
            pos_adv = advantages.clamp(min=0)                      # [B, N]
            mean_attn = attn.mean(dim=1)                           # [B, N, N] (query, key)
            # how much did positive-advantage trajectories attend TO each trajectory?
            w_raw = torch.einsum("bn,bnk->bk", pos_adv, mean_attn)  # [B, N]
            w_traj = 1.0 + self.args.traj_credit_weight * w_raw    # [B, N], baseline=1
            w_traj = w_traj / w_traj.mean(dim=1, keepdim=True)     # normalize: mean=1
        else:
            w_traj = torch.ones(B, N, device=device)

        return w_time.detach(), w_traj.detach()

    def compute_batch_objective(self, batch: dict[str, torch.Tensor]):
        rollout = rollout_latent_paths(
            model=self.generator,
            communication_module=self.communication_module,
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            latent_token_id=self.latent_id,
            num_return_sequences=self.args.num_return_sequences,
            latent_length=self.args.latent_length,
            sampling_by=self.args.sampling_by,
            noise_std=self.args.noise_std,
            communication_every=self.args.communication_every,
            collect_attention_weights=self.args.use_dense_credit,
        )
        if self.args.objective == "prm":
            prm_outputs = self.prm(
                input_ids=rollout.input_ids,
                attention_mask=rollout.attention_mask,
                latent_embeds=rollout.latent_thoughts,
                use_cache=False,
                return_dict=True,
            )
            prm_scores = prm_outputs.logits.squeeze(-1)
            latent_mask = rollout.input_ids == self.latent_id
            prm_scores = torch.where(latent_mask, prm_scores, 0).sum(dim=-1)
            prm_scores = prm_scores.view(-1, self.args.num_return_sequences)
            objective_loss = -(
                torch.logsumexp(prm_scores / self.args.score_temperature, dim=1)
                * self.args.score_temperature
            ).mean()
        elif self.args.objective == "ce":
            path_ce = self.compute_answer_ce_losses(batch, rollout)
            path_ce = path_ce.view(-1, self.args.num_return_sequences)
            ce_loss = path_ce.mean()
            with torch.no_grad():
                selector_scores = self.compute_selector_scores(rollout, detach_latents=True)
            selector_probs = torch.softmax(selector_scores / self.args.score_temperature, dim=1)
            prm_weighted_ce_loss = (selector_probs * path_ce).sum(dim=1).mean()
            prm_weight = min(max(self.args.ce_prm_weight, 0.0), 1.0)
            objective_loss = (1.0 - prm_weight) * ce_loss + prm_weight * prm_weighted_ce_loss
            prm_scores = selector_scores
            if torch.is_grad_enabled():
                selected_accuracy = objective_loss.new_zeros(())
            else:
                _, reward_metrics = self.evaluate_verifiable_rewards(
                    batch,
                    rollout,
                    selector_scores=selector_scores,
                )
                selected_accuracy = reward_metrics["selected_accuracy"]
        else:
            selector_scores = self.compute_selector_scores(rollout, detach_latents=True)
            rewards, reward_metrics = self.evaluate_verifiable_rewards(
                batch,
                rollout,
                selector_scores=selector_scores,
            )
            # PRM scores provide a continuous, per-sample signal correlated with
            # correctness. Binary rewards have near-zero covariance with the Gaussian
            # noise realizations, so the REINFORCE gradient is effectively dead.
            # PRM scores vary continuously within a group and give a real ranking.
            if self.args.use_prm_advantages:
                pg_advantages = self.compute_reward_advantages(selector_scores).detach()
            else:
                pg_advantages = self.compute_reward_advantages(rewards).detach()
            if self.args.use_dense_credit:
                base_batch_size = batch["input_ids"].shape[0]
                per_step_lp = gaussian_latent_log_probs_per_step(
                    samples=rollout.latent_thoughts,
                    means=rollout.post_communication_latent_thoughts,
                    std=self.args.noise_std,
                ).view(base_batch_size, self.args.num_return_sequences, -1)  # [B, N, L]
                w_time, w_traj = self.compute_2d_credit_weights(rollout, pg_advantages)
                credit = w_time.unsqueeze(1) * w_traj.unsqueeze(2)  # [B, N, L]
                weighted_lp = (credit * per_step_lp).sum(dim=-1)    # [B, N]
                policy_loss = -(pg_advantages * weighted_lp).mean()
            else:
                log_probs = gaussian_latent_log_probs(
                    samples=rollout.latent_thoughts,
                    means=rollout.post_communication_latent_thoughts,
                    std=self.args.noise_std,
                ).view(-1, self.args.num_return_sequences)
                policy_loss = -(pg_advantages * log_probs).mean()
            selector_probs = torch.softmax(selector_scores / self.args.score_temperature, dim=1)
            selector_loss = -((selector_probs * rewards).sum(dim=1)).mean()
            objective_loss = policy_loss + selector_loss
        diversity_loss = objective_loss.new_zeros(())
        if self.args.diversity_penalty_weight > 0 and self.args.num_return_sequences > 1:
            diversity_loss = self.diversity_loss_fct(
                rollout.post_communication_latent_thoughts,
                self.args.num_return_sequences,
            )
        anchor_loss = diversity_loss.new_zeros(())
        if self.args.anchor_weight > 0:
            anchor_loss = F.mse_loss(
                rollout.post_communication_latent_thoughts,
                rollout.pre_communication_latent_thoughts.detach(),
            )

        total_loss = objective_loss
        total_loss = total_loss + self.args.diversity_penalty_weight * diversity_loss
        total_loss = total_loss + self.args.anchor_weight * anchor_loss

        if self.args.objective == "prm":
            metrics = {
                "loss": total_loss.detach(),
                "score_loss": objective_loss.detach(),
                "diversity_loss": diversity_loss.detach(),
                "anchor_loss": anchor_loss.detach(),
                "mean_path_score": prm_scores.mean().detach(),
                "max_path_score": prm_scores.max(dim=1).values.mean().detach(),
            }
        elif self.args.objective == "ce":
            metrics = {
                "loss": total_loss.detach(),
                "ce_loss": ce_loss.detach(),
                "prm_weighted_ce_loss": prm_weighted_ce_loss.detach(),
                "diversity_loss": diversity_loss.detach(),
                "anchor_loss": anchor_loss.detach(),
                "mean_path_score": prm_scores.mean().detach(),
                "max_path_score": prm_scores.max(dim=1).values.mean().detach(),
                "selected_accuracy": selected_accuracy.detach(),
            }
        else:
            metrics = {
                "loss": total_loss.detach(),
                "policy_loss": policy_loss.detach(),
                "selector_loss": selector_loss.detach(),
                "diversity_loss": diversity_loss.detach(),
                "anchor_loss": anchor_loss.detach(),
                **{key: value.detach() for key, value in reward_metrics.items()},
            }
        return total_loss, metrics

    def evaluate(self) -> dict[str, float]:
        if self.eval_dataloader is None:
            return {}

        metric_sums = torch.zeros(len(self.metric_names), device=self.accelerator.device, dtype=torch.float32)
        count = torch.zeros(1, device=self.accelerator.device, dtype=torch.float32)
        self.generator.eval()
        if self.prm is not None:
            self.prm.eval()
        for batch in self.eval_dataloader:
            batch_inputs = self.prepare_batch_inputs(batch)
            with torch.no_grad():
                metrics = self.evaluate_batch_objective(batch_inputs)
            batch_size = batch_inputs["input_ids"].shape[0]
            metric_sums += torch.tensor(
                [metrics[name].float().item() * batch_size for name in self.metric_names],
                device=self.accelerator.device,
            )
            count += batch_size

        metric_sums = self.accelerator.reduce(metric_sums, reduction="sum")
        count = self.accelerator.reduce(count, reduction="sum")
        count_value = max(count.item(), 1.0)
        return {
            f"eval_{name}": (metric_sums[idx] / count_value).item()
            for idx, name in enumerate(self.metric_names)
        }

    def metric_value(self, metrics: dict[str, float]) -> float:
        if self.args.metric_for_best_model == "eval_loss":
            return -metrics["eval_loss"]
        return metrics[self.args.metric_for_best_model]

    def save_checkpoint(self, name: str, metrics: dict[str, float]):
        if not self.accelerator.is_main_process:
            return
        checkpoint_dir = os.path.join(self.args.output_dir, name)
        os.makedirs(checkpoint_dir, exist_ok=True)
        state_dict = {
            f"communication_module.{key}": value.detach().cpu().contiguous()
            for key, value in self.accelerator.unwrap_model(
                self.communication_module
            ).state_dict().items()
        }
        save_file(state_dict, os.path.join(checkpoint_dir, "model.safetensors"))
        if self.train_prm:
            prm_checkpoint_dir = os.path.join(checkpoint_dir, "prm")
            os.makedirs(prm_checkpoint_dir, exist_ok=True)
            self.accelerator.unwrap_model(self.prm).save_pretrained(
                prm_checkpoint_dir,
                safe_serialization=True,
            )
        if self.train_generator:
            generator_adapter_dir = os.path.join(checkpoint_dir, "generator_adapter")
            os.makedirs(generator_adapter_dir, exist_ok=True)
            self.accelerator.unwrap_model(self.generator).save_pretrained(
                generator_adapter_dir,
                safe_serialization=True,
            )
        with open(
            os.path.join(checkpoint_dir, "communication_config.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                {
                    "objective": self.args.objective,
                    "communication_type": self.args.communication_type,
                    "communication_attention_heads": self.args.communication_attention_heads,
                    "communication_topk": self.args.communication_topk,
                    "communication_gate_bias": self.args.communication_gate_bias,
                    "communication_interaction_scale": self.args.communication_interaction_scale,
                    "communication_every": self.args.communication_every,
                    "num_return_sequences": self.args.num_return_sequences,
                    "latent_length": self.args.latent_length,
                    "max_new_tokens": self.args.max_new_tokens,
                    "sampling_by": self.args.sampling_by,
                    "dropout_p": self.args.dropout_p,
                    "noise_std": self.args.noise_std,
                    "reward_baseline": self.args.reward_baseline,
                    "freeze_prm": self.args.freeze_prm,
                    "selector_checkpoint": "prm" if self.train_prm else None,
                    "train_generator_lora": self.args.train_generator_lora,
                    "generator_adapter_checkpoint": "generator_adapter" if self.train_generator else None,
                    "lora_r": self.args.lora_r,
                    "lora_alpha": self.args.lora_alpha,
                    "lora_dropout": self.args.lora_dropout,
                    "lora_target_modules": self.args.lora_target_modules,
                    "ce_prm_weight": self.args.ce_prm_weight,
                },
                f,
                indent=2,
            )
        with open(
            os.path.join(checkpoint_dir, "metrics.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(metrics, f, indent=2)

    def maybe_save_best(self, metrics: dict[str, float]):
        if len(metrics) == 0:
            return
        current_metric = self.metric_value(metrics)
        if self.best_metric is None or current_metric > self.best_metric:
            self.best_metric = current_metric
            self.save_checkpoint("best", metrics)
            if self.accelerator.is_main_process and self.wandb_run is not None:
                for key, value in metrics.items():
                    self.wandb_run.summary[key] = value

    def train(self):
        if self.eval_dataloader is not None and (self.args.eval_on_start or self.args.eval_only):
            eval_metrics = self.evaluate()
            if self.accelerator.is_main_process:
                print(json.dumps(eval_metrics, indent=2))
            self.log_wandb(eval_metrics, step=self.global_step)
            self.maybe_save_best(eval_metrics)
            self.accelerator.wait_for_everyone()

        if self.args.eval_only:
            if self.accelerator.is_main_process and self.wandb_run is not None:
                self.wandb_run.finish()
            return

        eval_interval = max(1, self.num_update_steps_per_epoch // max(1, self.args.eval_frequency))
        progress_bar = tqdm(
            total=self.max_train_steps,
            disable=not self.accelerator.is_main_process,
            desc=self.args.run_name,
        )
        running_metrics = {name: 0.0 for name in self.metric_names}
        accumulation_metrics = {name: 0.0 for name in self.metric_names}
        accumulation_count = 0

        for _ in range(self.num_train_epochs):
            if self.train_generator:
                self.generator.train()
            else:
                self.generator.eval()
            if self.train_prm:
                self.prm.train()
            else:
                self.prm.eval()
            for batch in self.train_dataloader:
                batch_inputs = self.prepare_batch_inputs(batch)
                accumulate_modules = [self.communication_module]
                if self.train_generator:
                    accumulate_modules.append(self.generator)
                if self.train_prm:
                    accumulate_modules.append(self.prm)
                    self.prm.train()
                accumulation_context = self.accelerator.accumulate(*accumulate_modules)
                with accumulation_context:
                    metrics = self.backward_batch_objective(batch_inputs)
                    for name in self.metric_names:
                        accumulation_metrics[name] += metrics[name].float().item()
                    accumulation_count += 1
                    if self.accelerator.sync_gradients:
                        grad_params = list(self.communication_module.parameters())
                        if self.train_generator:
                            grad_params.extend(
                                param for param in self.generator.parameters() if param.requires_grad
                            )
                        if self.train_prm:
                            grad_params.extend(self.prm.parameters())
                        grad_params = unique_parameters(
                            param for param in grad_params if param.requires_grad
                        )
                        self.accelerator.clip_grad_norm_(grad_params, self.args.max_grad_norm)
                        self.optimizer.step()
                        self.lr_scheduler.step()
                        self.optimizer.zero_grad(set_to_none=True)

                if not self.accelerator.sync_gradients:
                    continue

                for name in self.metric_names:
                    running_metrics[name] += accumulation_metrics[name] / max(1, accumulation_count)
                accumulation_metrics = {name: 0.0 for name in self.metric_names}
                accumulation_count = 0

                self.global_step += 1
                progress_bar.update(1)
                if self.accelerator.is_main_process and self.global_step % self.args.logging_steps == 0:
                    denom = float(self.args.logging_steps)
                    train_metrics = {
                        f"train_{name}": running_metrics[name] / denom for name in self.metric_names
                    }
                    train_metrics["lr"] = self.lr_scheduler.get_last_lr()[0]
                    print(json.dumps({"step": self.global_step} | train_metrics))
                    self.log_wandb(train_metrics, step=self.global_step)
                    running_metrics = {name: 0.0 for name in self.metric_names}

                should_eval = (
                    self.eval_dataloader is not None
                    and (self.global_step % eval_interval == 0 or self.global_step == self.max_train_steps)
                )
                if should_eval:
                    eval_metrics = self.evaluate()
                    if self.accelerator.is_main_process:
                        print(json.dumps({"step": self.global_step} | eval_metrics))
                    self.log_wandb(eval_metrics, step=self.global_step)
                    self.maybe_save_best(eval_metrics)
                    self.accelerator.wait_for_everyone()

                if self.global_step >= self.max_train_steps:
                    break
            if self.global_step >= self.max_train_steps:
                break

        final_metrics = {"step": self.global_step}
        if self.eval_dataloader is not None:
            final_metrics |= self.evaluate()
            self.log_wandb({k: v for k, v in final_metrics.items() if k != "step"}, step=self.global_step)
            self.maybe_save_best(final_metrics)
        self.save_checkpoint("last", final_metrics)
        if self.accelerator.is_main_process and self.wandb_run is not None:
            self.wandb_run.finish()
        self.accelerator.wait_for_everyone()
        progress_bar.close()


def main(*args, **kwargs):
    trainer_args = parse_args(*args, **kwargs)
    trainer = GeneratorInteractionTrainer(trainer_args)
    trainer.train()


if __name__ == "__main__":
    from fire import Fire

    Fire(main)
