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

from src.generation_mixin import LatentGenerationConfig
from src.model_registry import MODELS
from src.models.coconut import COCONUTGPT2
from src.models.gpt2 import COCONUTGPT2ForTokenClassification
from src.models.communication import (
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
    objective: Literal["prm", "verifiable_rl"] = field(default="prm")
    train_data_path: str = field(default="data/gsm_train.json")
    valid_data_path: str = field(default="data/gsm_valid.json")
    model_dtype: Literal["fp32", "fp16", "bf16"] = field(default="bf16")
    seed: int = field(default=42)
    num_return_sequences: int = field(default=8)
    latent_length: int = field(default=6)
    max_new_tokens: int = field(default=128)
    communication_type: Literal["mean", "attention", "router"] = field(default="attention")
    communication_attention_heads: int = field(default=4)
    communication_topk: int = field(default=2)
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
    use_wandb: bool = field(default=False)
    wandb_project: str = field(default="latenttts-generator-interaction")
    wandb_entity: str = field(default="")
    metric_for_best_model: str = field(default="eval_max_path_score")
    score_temperature: float = field(default=0.25)
    reward_baseline: Literal["leave_one_out", "group_mean"] = field(default="leave_one_out")
    diversity_penalty_weight: float = field(default=0.1)
    anchor_weight: float = field(default=0.01)
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


def gaussian_latent_log_probs(
    samples: torch.Tensor,
    means: torch.Tensor,
    std: float,
) -> torch.Tensor:
    if std <= 0:
        raise ValueError(f"noise_std must be positive for verifiable RL, but got {std}")
    normalized = (samples.detach().float() - means.float()) / std
    log_normalizer = math.log(std) + 0.5 * math.log(2.0 * math.pi)
    # REINFORCE should use the log-probability of the full latent trajectory, not
    # an average per latent dimension, otherwise the policy gradient is diluted.
    return (-0.5 * normalized.pow(2) - log_normalizer).sum(dim=(-1, -2))


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
) -> LatentRolloutOutput:
    base_batch_size = input_ids.shape[0]
    total_batch_size = base_batch_size * num_return_sequences
    input_ids = input_ids.repeat_interleave(num_return_sequences, dim=0)
    attention_mask = attention_mask.repeat_interleave(num_return_sequences, dim=0)
    inputs_embeds = model.get_input_embeddings()(torch.where(input_ids == latent_token_id, 0, input_ids))
    alive_mask = torch.ones(
        base_batch_size, num_return_sequences, dtype=torch.bool, device=input_ids.device
    )

    latent_steps = []
    pre_communication_steps = []
    post_communication_steps = []
    dropout_sampling = sampling_by == "dropout"
    if dropout_sampling:
        enable_dropout(model)
    try:
        for latent_step in range(latent_length):
            transformer_outputs = model.transformer(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )
            next_latent = transformer_outputs.last_hidden_state[:, -1, :]
            pre_communication_steps.append(next_latent)

            if latent_step % max(1, communication_every) == 0:
                grouped_latent = next_latent.view(base_batch_size, num_return_sequences, -1)
                next_latent = communication_module(
                    grouped_latent,
                    alive_mask=alive_mask,
                    step_idx=latent_step,
                ).view(total_batch_size, -1)
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

    return LatentRolloutOutput(
        input_ids=input_ids,
        attention_mask=attention_mask,
        latent_thoughts=torch.stack(latent_steps, dim=1),
        pre_communication_latent_thoughts=torch.stack(pre_communication_steps, dim=1),
        post_communication_latent_thoughts=torch.stack(post_communication_steps, dim=1),
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
            param.requires_grad = args.objective == "verifiable_rl"
        if args.objective == "verifiable_rl":
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
        if args.objective == "verifiable_rl":
            trainable_params.extend(param for param in self.prm.parameters() if param.requires_grad)
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

        if self.eval_dataloader is None:
            if args.objective == "verifiable_rl":
                (
                    self.communication_module,
                    self.prm,
                    self.optimizer,
                    self.train_dataloader,
                    self.lr_scheduler,
                ) = self.accelerator.prepare(
                    self.communication_module,
                    self.prm,
                    self.optimizer,
                    self.train_dataloader,
                    self.lr_scheduler,
                )
            else:
                (
                    self.communication_module,
                    self.optimizer,
                    self.train_dataloader,
                    self.lr_scheduler,
                ) = self.accelerator.prepare(
                    self.communication_module,
                    self.optimizer,
                    self.train_dataloader,
                    self.lr_scheduler,
                )
        else:
            if args.objective == "verifiable_rl":
                (
                    self.communication_module,
                    self.prm,
                    self.optimizer,
                    self.train_dataloader,
                    self.eval_dataloader,
                    self.lr_scheduler,
                ) = self.accelerator.prepare(
                    self.communication_module,
                    self.prm,
                    self.optimizer,
                    self.train_dataloader,
                    self.eval_dataloader,
                    self.lr_scheduler,
                )
            else:
                (
                    self.communication_module,
                    self.optimizer,
                    self.train_dataloader,
                    self.eval_dataloader,
                    self.lr_scheduler,
                ) = self.accelerator.prepare(
                    self.communication_module,
                    self.optimizer,
                    self.train_dataloader,
                    self.eval_dataloader,
                    self.lr_scheduler,
                )
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
            return rewards - rewards.mean(dim=1, keepdim=True)
        baselines = (rewards.sum(dim=1, keepdim=True) - rewards) / (self.args.num_return_sequences - 1)
        return rewards - baselines

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
        inputs_embeds = self.generator.get_input_embeddings()(
            torch.where(input_ids == self.latent_id, 0, input_ids)
        )
        inputs_embeds = inputs_embeds.clone()
        inputs_embeds[latent_mask] = rollout.latent_thoughts.detach().to(
            dtype=inputs_embeds.dtype
        ).reshape(-1, inputs_embeds.shape[-1])

        with torch.no_grad():
            output = self.generator.generate(
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
        else:
            selector_scores = self.compute_selector_scores(rollout, detach_latents=True)
            rewards, reward_metrics = self.evaluate_verifiable_rewards(
                batch,
                rollout,
                selector_scores=selector_scores,
            )
            advantages = self.compute_reward_advantages(rewards).detach()
            log_probs = gaussian_latent_log_probs(
                samples=rollout.latent_thoughts,
                means=rollout.post_communication_latent_thoughts,
                std=self.args.noise_std,
            ).view(-1, self.args.num_return_sequences)
            policy_loss = -(advantages * log_probs).mean()
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
        if self.args.objective == "verifiable_rl":
            prm_checkpoint_dir = os.path.join(checkpoint_dir, "prm")
            os.makedirs(prm_checkpoint_dir, exist_ok=True)
            self.accelerator.unwrap_model(self.prm).save_pretrained(
                prm_checkpoint_dir,
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
                    "communication_every": self.args.communication_every,
                    "num_return_sequences": self.args.num_return_sequences,
                    "latent_length": self.args.latent_length,
                    "max_new_tokens": self.args.max_new_tokens,
                    "sampling_by": self.args.sampling_by,
                    "dropout_p": self.args.dropout_p,
                    "noise_std": self.args.noise_std,
                    "reward_baseline": self.args.reward_baseline,
                    "selector_checkpoint": "prm" if self.args.objective == "verifiable_rl" else None,
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
        if self.eval_dataloader is not None and self.args.eval_on_start:
            eval_metrics = self.evaluate()
            if self.accelerator.is_main_process:
                print(json.dumps(eval_metrics, indent=2))
            self.log_wandb(eval_metrics, step=self.global_step)
            self.maybe_save_best(eval_metrics)
            self.accelerator.wait_for_everyone()

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
            self.generator.eval()
            if self.args.objective == "verifiable_rl":
                self.prm.train()
            for batch in self.train_dataloader:
                batch_inputs = self.prepare_batch_inputs(batch)
                if self.args.objective == "verifiable_rl":
                    self.prm.train()
                    accumulation_context = self.accelerator.accumulate(
                        self.communication_module,
                        self.prm,
                    )
                else:
                    accumulation_context = self.accelerator.accumulate(self.communication_module)
                with accumulation_context:
                    metrics = self.backward_batch_objective(batch_inputs)
                    for name in self.metric_names:
                        accumulation_metrics[name] += metrics[name].float().item()
                    accumulation_count += 1
                    if self.accelerator.sync_gradients:
                        grad_params = list(self.communication_module.parameters())
                        if self.args.objective == "verifiable_rl":
                            grad_params.extend(self.prm.parameters())
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
