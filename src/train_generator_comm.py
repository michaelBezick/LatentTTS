import json
import math
import os
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

from src.models.coconut import COCONUTGPT2
from src.models.gpt2 import COCONUTGPT2ForTokenClassification
from src.models.communication import (
    build_communication_module,
    restore_communication_module_from_checkpoint,
)
from src.models.loss import DiversityPenaltyLoss
from src.utils import InferenceCollator, add_noise, disable_dropout, enable_dropout, set_dropout_p, set_seed


@dataclass
class GeneratorCommunicationConfig:
    run_name: str = field(default="coconut-generator-communication")
    output_dir: str = field(default="outputs/coconut-generator-communication")
    model_id: str = field(default="checkpoints/coconut")
    prm_id: str = field(default="checkpoints/latentRM")
    train_data_path: str = field(default="data/gsm_train.json")
    valid_data_path: str = field(default="data/gsm_valid.json")
    model_dtype: Literal["fp32", "fp16", "bf16"] = field(default="bf16")
    seed: int = field(default=42)
    num_return_sequences: int = field(default=8)
    latent_length: int = field(default=6)
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
    max_grad_norm: float = field(default=1.0)
    logging_steps: int = field(default=10)
    eval_frequency: int = field(default=1)
    eval_on_start: bool = field(default=True)
    use_wandb: bool = field(default=False)
    wandb_project: str = field(default="latenttts-generator-communication")
    wandb_entity: str = field(default="")
    metric_for_best_model: Literal["eval_loss", "eval_max_path_score"] = field(
        default="eval_max_path_score"
    )
    score_temperature: float = field(default=0.25)
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


def parse_args(*args, **kwargs) -> GeneratorCommunicationConfig:
    parser = HfArgumentParser(GeneratorCommunicationConfig)
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


class GeneratorCommunicationTrainer:
    def __init__(self, args: GeneratorCommunicationConfig):
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
        self.latent_id = self.tokenizer.convert_tokens_to_ids("<|latent|>")
        self.start_id = self.tokenizer.convert_tokens_to_ids("<|start-latent|>")
        self.end_id = self.tokenizer.convert_tokens_to_ids("<|end-latent|>")
        if self.latent_id == self.start_id or self.latent_id == self.end_id:
            raise ValueError(
                f"latent_id={self.latent_id}, start_id={self.start_id}, end_id={self.end_id} must all differ"
            )

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
                    f"Could not restore generator communication module from {args.init_communication_from}"
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
            param.requires_grad = False
        self.prm.eval()

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
        if len(trainable_params) == 0:
            raise ValueError("No trainable generator communication parameters found")
        self.optimizer = AdamW(
            trainable_params,
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )

        self.num_update_steps_per_epoch = max(
            1,
            math.ceil(
                len(self.train_dataloader) / self.accelerator.gradient_accumulation_steps
            ),
        )
        self.max_train_steps = self.num_update_steps_per_epoch * args.num_train_epochs
        self.num_warmup_steps = int(self.max_train_steps * args.warmup_ratio)
        self.lr_scheduler = get_scheduler(
            name=args.lr_scheduler_type,
            optimizer=self.optimizer,
            num_warmup_steps=self.num_warmup_steps,
            num_training_steps=self.max_train_steps,
        )

        if self.eval_dataloader is None:
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
        self.prm = self.prm.to(self.accelerator.device)
        self.diversity_loss_fct = DiversityPenaltyLoss()
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
                        "W&B logging requested for generator communication training, but wandb is not installed."
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

    def evaluate_batch_objective(self, batch: dict[str, torch.Tensor]):
        batch_size = batch["input_ids"].shape[0]
        metric_sums = {
            "loss": torch.zeros((), device=self.accelerator.device, dtype=torch.float32),
            "score_loss": torch.zeros((), device=self.accelerator.device, dtype=torch.float32),
            "diversity_loss": torch.zeros((), device=self.accelerator.device, dtype=torch.float32),
            "anchor_loss": torch.zeros((), device=self.accelerator.device, dtype=torch.float32),
            "mean_path_score": torch.zeros((), device=self.accelerator.device, dtype=torch.float32),
            "max_path_score": torch.zeros((), device=self.accelerator.device, dtype=torch.float32),
        }

        for chunk_inputs, chunk_size in self.iter_forward_batches(batch):
            with self.accelerator.autocast():
                _, metrics = self.compute_batch_objective(chunk_inputs)
            for key in metric_sums:
                metric_sums[key] += metrics[key].detach().float() * chunk_size

        return self.aggregate_chunk_metrics(metric_sums, batch_size)

    def backward_batch_objective(self, batch: dict[str, torch.Tensor]):
        batch_size = batch["input_ids"].shape[0]
        metric_sums = {
            "loss": torch.zeros((), device=self.accelerator.device, dtype=torch.float32),
            "score_loss": torch.zeros((), device=self.accelerator.device, dtype=torch.float32),
            "diversity_loss": torch.zeros((), device=self.accelerator.device, dtype=torch.float32),
            "anchor_loss": torch.zeros((), device=self.accelerator.device, dtype=torch.float32),
            "mean_path_score": torch.zeros((), device=self.accelerator.device, dtype=torch.float32),
            "max_path_score": torch.zeros((), device=self.accelerator.device, dtype=torch.float32),
        }

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

        score_loss = -(
            torch.logsumexp(prm_scores / self.args.score_temperature, dim=1)
            * self.args.score_temperature
        ).mean()
        diversity_loss = prm_scores.new_zeros(())
        if self.args.diversity_penalty_weight > 0 and self.args.num_return_sequences > 1:
            diversity_loss = self.diversity_loss_fct(
                rollout.post_communication_latent_thoughts,
                self.args.num_return_sequences,
            )
        anchor_loss = prm_scores.new_zeros(())
        if self.args.anchor_weight > 0:
            anchor_loss = F.mse_loss(
                rollout.post_communication_latent_thoughts,
                rollout.pre_communication_latent_thoughts.detach(),
            )

        total_loss = score_loss
        total_loss = total_loss + self.args.diversity_penalty_weight * diversity_loss
        total_loss = total_loss + self.args.anchor_weight * anchor_loss

        metrics = {
            "loss": total_loss.detach(),
            "score_loss": score_loss.detach(),
            "diversity_loss": diversity_loss.detach(),
            "anchor_loss": anchor_loss.detach(),
            "mean_path_score": prm_scores.mean().detach(),
            "max_path_score": prm_scores.max(dim=1).values.mean().detach(),
        }
        return total_loss, metrics

    def evaluate(self) -> dict[str, float]:
        if self.eval_dataloader is None:
            return {}

        metric_sums = torch.zeros(6, device=self.accelerator.device, dtype=torch.float32)
        count = torch.zeros(1, device=self.accelerator.device, dtype=torch.float32)
        self.generator.eval()
        self.prm.eval()
        for batch in self.eval_dataloader:
            batch_inputs = {
                k: v.to(self.accelerator.device)
                for k, v in batch.items()
                if k in {"input_ids", "attention_mask"}
            }
            with torch.no_grad():
                metrics = self.evaluate_batch_objective(batch_inputs)
            batch_size = batch_inputs["input_ids"].shape[0]
            metric_sums += torch.tensor(
                [
                    metrics["loss"].float().item() * batch_size,
                    metrics["score_loss"].float().item() * batch_size,
                    metrics["diversity_loss"].float().item() * batch_size,
                    metrics["anchor_loss"].float().item() * batch_size,
                    metrics["mean_path_score"].float().item() * batch_size,
                    metrics["max_path_score"].float().item() * batch_size,
                ],
                device=self.accelerator.device,
            )
            count += batch_size

        metric_sums = self.accelerator.reduce(metric_sums, reduction="sum")
        count = self.accelerator.reduce(count, reduction="sum")
        count_value = max(count.item(), 1.0)
        return {
            "eval_loss": (metric_sums[0] / count_value).item(),
            "eval_score_loss": (metric_sums[1] / count_value).item(),
            "eval_diversity_loss": (metric_sums[2] / count_value).item(),
            "eval_anchor_loss": (metric_sums[3] / count_value).item(),
            "eval_mean_path_score": (metric_sums[4] / count_value).item(),
            "eval_max_path_score": (metric_sums[5] / count_value).item(),
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
        with open(
            os.path.join(checkpoint_dir, "communication_config.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                {
                    "communication_type": self.args.communication_type,
                    "communication_attention_heads": self.args.communication_attention_heads,
                    "communication_topk": self.args.communication_topk,
                    "communication_every": self.args.communication_every,
                    "num_return_sequences": self.args.num_return_sequences,
                    "latent_length": self.args.latent_length,
                    "sampling_by": self.args.sampling_by,
                    "dropout_p": self.args.dropout_p,
                    "noise_std": self.args.noise_std,
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
        running_loss = 0.0
        running_score_loss = 0.0
        running_diversity_loss = 0.0
        running_anchor_loss = 0.0

        for _ in range(self.args.num_train_epochs):
            self.generator.eval()
            for batch in self.train_dataloader:
                batch_inputs = {
                    k: v.to(self.accelerator.device)
                    for k, v in batch.items()
                    if k in {"input_ids", "attention_mask"}
                }
                with self.accelerator.accumulate(self.communication_module):
                    metrics = self.backward_batch_objective(batch_inputs)
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(
                            self.accelerator.unwrap_model(
                                self.communication_module
                            ).parameters(),
                            self.args.max_grad_norm,
                        )
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)

                running_loss += metrics["loss"].float().item()
                running_score_loss += metrics["score_loss"].float().item()
                running_diversity_loss += metrics["diversity_loss"].float().item()
                running_anchor_loss += metrics["anchor_loss"].float().item()

                if not self.accelerator.sync_gradients:
                    continue

                self.global_step += 1
                progress_bar.update(1)
                if self.accelerator.is_main_process and self.global_step % self.args.logging_steps == 0:
                    denom = float(self.args.logging_steps)
                    train_metrics = {
                        "train_loss": running_loss / denom,
                        "train_score_loss": running_score_loss / denom,
                        "train_diversity_loss": running_diversity_loss / denom,
                        "train_anchor_loss": running_anchor_loss / denom,
                        "lr": self.lr_scheduler.get_last_lr()[0],
                    }
                    print(json.dumps({"step": self.global_step} | train_metrics))
                    self.log_wandb(train_metrics, step=self.global_step)
                    running_loss = 0.0
                    running_score_loss = 0.0
                    running_diversity_loss = 0.0
                    running_anchor_loss = 0.0

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
    trainer = GeneratorCommunicationTrainer(trainer_args)
    trainer.train()


if __name__ == "__main__":
    from fire import Fire

    Fire(main)
