from collections import Counter
import json
import os
import time
from typing import Literal
import torch
import numpy as np
from tqdm import tqdm
import datasets
from transformers import AutoTokenizer
from accelerate import Accelerator
from torch.utils.data import DataLoader
from fire import Fire

from .generation_mixin import LatentGenerationMixin, LatentGenerationConfig
from .paths import MODELS
from .models.gpt2 import COCONUTGPT2ForTokenClassification
from .models.llama import COCONUTLlamaForTokenClassification
from .models.communication import (
    build_communication_module,
    restore_communication_module_from_checkpoint,
)
from .utils import set_seed, InferenceCollator


def synchronize_device(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def trajectory_has_any_correct(value: torch.Tensor | list[bool]) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(value.any().item())
    return any(value)


def initialize_pairwise_cosine_stats() -> dict[str, float | int]:
    return {
        "num_groups": 0,
        "pre_pairwise_cos_sum": 0.0,
        "post_pairwise_cos_sum": 0.0,
        "pre_collapse_count": 0,
        "post_collapse_count": 0,
    }


def _to_latent_tensor(latent_embeds: torch.Tensor | list[torch.Tensor] | None) -> torch.Tensor | None:
    if latent_embeds is None:
        return None
    if isinstance(latent_embeds, list):
        if len(latent_embeds) == 0:
            return None
        if not all(isinstance(x, torch.Tensor) for x in latent_embeds):
            return None
        min_len = min(x.shape[0] for x in latent_embeds)
        latent_embeds = torch.stack([x[:min_len] for x in latent_embeds], dim=0)
    if not isinstance(latent_embeds, torch.Tensor) or latent_embeds.dim() != 3:
        return None
    return latent_embeds


def _compute_group_pairwise_cosine_means(
    latent_embeds: torch.Tensor | list[torch.Tensor] | None,
    trajectory_group_size: int,
    pool: Literal["mean", "last"],
) -> torch.Tensor | None:
    latent_embeds = _to_latent_tensor(latent_embeds)
    if latent_embeds is None or trajectory_group_size <= 1:
        return None

    total_trajectories = latent_embeds.shape[0]
    usable_trajectories = (total_trajectories // trajectory_group_size) * trajectory_group_size
    if usable_trajectories == 0:
        return None

    latent_embeds = latent_embeds[:usable_trajectories]
    if pool == "last":
        pooled = latent_embeds[:, -1, :]
    else:
        pooled = latent_embeds.mean(dim=1)

    num_groups = usable_trajectories // trajectory_group_size
    pooled = pooled.reshape(num_groups, trajectory_group_size, -1).float()
    pooled = torch.nn.functional.normalize(pooled, p=2, dim=-1)

    sim_matrix = torch.matmul(pooled, pooled.transpose(1, 2))
    off_diag_mask = ~torch.eye(trajectory_group_size, device=sim_matrix.device, dtype=torch.bool).unsqueeze(0)
    off_diag = sim_matrix.masked_select(off_diag_mask).reshape(num_groups, -1)
    return off_diag.mean(dim=1)


def update_pairwise_cosine_stats(
    stats: dict[str, float | int],
    pre_latent_embeds: torch.Tensor | list[torch.Tensor] | None,
    post_latent_embeds: torch.Tensor | list[torch.Tensor] | None,
    trajectory_group_size: int,
    pool: Literal["mean", "last"],
    collapse_threshold: float,
) -> dict[str, float | int]:
    pre_means = _compute_group_pairwise_cosine_means(pre_latent_embeds, trajectory_group_size, pool)
    post_source = post_latent_embeds if post_latent_embeds is not None else pre_latent_embeds
    post_means = _compute_group_pairwise_cosine_means(post_source, trajectory_group_size, pool)
    if pre_means is None or post_means is None:
        return stats

    num_groups = min(pre_means.shape[0], post_means.shape[0])
    if num_groups == 0:
        return stats

    pre_means = pre_means[:num_groups].float().cpu()
    post_means = post_means[:num_groups].float().cpu()

    stats["num_groups"] += int(num_groups)
    stats["pre_pairwise_cos_sum"] += float(pre_means.sum().item())
    stats["post_pairwise_cos_sum"] += float(post_means.sum().item())
    stats["pre_collapse_count"] += int((pre_means > collapse_threshold).sum().item())
    stats["post_collapse_count"] += int((post_means > collapse_threshold).sum().item())
    return stats


def merge_pairwise_cosine_stats(all_stats: list[dict[str, float | int]]) -> dict[str, float | int]:
    merged = initialize_pairwise_cosine_stats()
    for stats in all_stats:
        for key in merged:
            merged[key] += stats.get(key, 0)
    return merged


def finalize_pairwise_cosine_stats(
    stats: dict[str, float | int],
    collapse_threshold: float,
) -> dict[str, float] | None:
    num_groups = int(stats["num_groups"])
    if num_groups == 0:
        return None

    pre_pairwise_cos_mean = float(stats["pre_pairwise_cos_sum"]) / num_groups
    post_pairwise_cos_mean = float(stats["post_pairwise_cos_sum"]) / num_groups
    pre_collapse_rate = float(stats["pre_collapse_count"]) / num_groups
    post_collapse_rate = float(stats["post_collapse_count"]) / num_groups
    return {
        "num_groups": num_groups,
        "pre_pairwise_cos_mean": pre_pairwise_cos_mean,
        "post_pairwise_cos_mean": post_pairwise_cos_mean,
        "delta_pairwise_cos_mean": post_pairwise_cos_mean - pre_pairwise_cos_mean,
        "collapse_threshold": collapse_threshold,
        "pre_collapse_rate": pre_collapse_rate,
        "post_collapse_rate": post_collapse_rate,
        "delta_collapse_rate": post_collapse_rate - pre_collapse_rate,
    }

@torch.no_grad()
def main(
    generator_type: Literal["coconut", "codi"] = "coconut",
    model_dtype: Literal["bf16", "fp16", "fp32"] = "fp32",
    prm_id: str = "../../latentRM-coconut",
    prm_model_family: Literal["llama", "gpt2"] = "gpt2",
    prm_mode: Literal["best_of_n", "beam_search"] = "beam_search",
    data_path: str = "data/gsm_test.json",
    num_return_sequences: int = 1,
    num_beams: int = 4,
    num_beam_candidates: int = 4,
    batch_size: int = 1024,
    latent_length: int = 6,
    max_new_tokens: int | None = 128,
    seed: int = 200,
    sort_by_len: bool = True,
    progress_bar: bool = True,
    generator_communication_type: Literal["none", "mean", "attention", "router"] = "none",
    generator_communication_checkpoint: str | None = None,
    generator_communication_attention_heads: int = 4,
    generator_communication_topk: int = 2,
    generator_communication_every: int = 1,
    communication_type: Literal["none", "mean", "attention", "router"] = "none",
    communication_attention_heads: int = 4,
    communication_topk: int = 2,
    report_pairwise_cosine: bool = False,
    pairwise_cosine_pool: Literal["mean", "last"] = "mean",
    pairwise_collapse_threshold: float = 0.9,
    pairwise_cosine_output_path: str | None = None,
):
    if prm_mode == "beam_search" and generator_communication_type != "none":
        raise ValueError(
            "Generator-side communication is only supported for sampled best_of_n inference. "
            "The beam_search path is not overridden in this repo."
        )
    new_line_after_input = generator_type == "coconut"
    if prm_mode == "beam_search":
        assert num_return_sequences == 1
        batch_size = max(1, batch_size // num_beam_candidates // num_beams)
    else:
        assert num_return_sequences > 1
        num_beam_candidates = None
        num_beams = 1
        batch_size = max(1, batch_size // num_return_sequences)

    model_id = MODELS[generator_type]["id"]
    set_seed(seed)
    accelerator = Accelerator()
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        print(f"Loading {data_path}")

    processing_class = AutoTokenizer.from_pretrained(
        MODELS[generator_type]["id"],
    )
    if processing_class.pad_token is None:
        processing_class.pad_token = processing_class.eos_token

    latent_id = processing_class.convert_tokens_to_ids("<|latent|>")
    start_id = processing_class.convert_tokens_to_ids("<|start-latent|>")
    end_id = processing_class.convert_tokens_to_ids("<|end-latent|>")
    target_id = processing_class.convert_tokens_to_ids(">>")
    assert (
        latent_id != start_id and latent_id != end_id and start_id != end_id
    ), f"latent_id, start_id, end_id must be different, but got {latent_id}, {start_id}, {end_id}"
    # print(f"<|latent|>: {latent_id}, <|start-latent|>: {start_id}, <|end-latent|>: {end_id}, >>: {target_id}")

    generation_config = LatentGenerationConfig(
        max_new_tokens=max_new_tokens,
        latent_length=latent_length,
        latent_do_sample=True,
        latent_do_sample_by="dropout",
        communication_type=generator_communication_type,
        communication_every=generator_communication_every,
        generation_mode="beam_search" if prm_mode == "beam_search" else None,
        # do_sample=False,
        num_beams=num_beams,
        num_beam_candidates=num_beam_candidates,
        # num_samples=num_samples,
        pad_token_id=processing_class.pad_token_id,
        eos_token_id=processing_class.eos_token_id,
        bos_token_id=processing_class.bos_token_id,
    )

    class LatentGPT2LMHeadModel(MODELS[generator_type]["class"], LatentGenerationMixin):
        def __init__(self, config):
            super().__init__(config)

    model = LatentGPT2LMHeadModel.from_pretrained(
        MODELS[generator_type]["id"],
        latent_id=latent_id,
        latent_start_id=start_id,
        latent_end_id=end_id,
        target_id=target_id,
        pad_token_id=processing_class.pad_token_id,
        device_map={"": accelerator.process_index},
        torch_dtype=(
            torch.bfloat16 if model_dtype == "bf16" else (torch.float16 if model_dtype == "fp16" else None)
        ),
    )
    if generator_communication_type != "none":
        model.communication_module = build_communication_module(
            communication_type=generator_communication_type,
            d_model=model.config.hidden_size,
            n_heads=generator_communication_attention_heads,
            topk=generator_communication_topk,
        )
        if generator_communication_checkpoint is not None:
            restored = restore_communication_module_from_checkpoint(
                module_owner=model,
                checkpoint_path=generator_communication_checkpoint,
            )
            if not restored:
                raise ValueError(
                    f"Could not restore generator communication module from {generator_communication_checkpoint}"
                )
        model.communication_module = model.communication_module.to(
            device=model.device, dtype=model.dtype
        )
    if prm_model_family == "llama":

        prm_processing_class = AutoTokenizer.from_pretrained(prm_id)
        prm = COCONUTLlamaForTokenClassification.from_pretrained(
            prm_id,
            latent_id=prm_processing_class.convert_tokens_to_ids("<|latent|>"),
            latent_start_id=prm_processing_class.convert_tokens_to_ids("<|start-latent|>"),
            latent_end_id=prm_processing_class.convert_tokens_to_ids("<|end-latent|>"),
            target_id=prm_processing_class.convert_tokens_to_ids(">>"),
            pad_token_id=prm_processing_class.pad_token_id,
            latent_hidden_size=model.config.hidden_size,
            device_map={"": accelerator.process_index},
            torch_dtype=(
                torch.bfloat16
                if model_dtype == "bf16"
                else (torch.float16 if model_dtype == "fp16" else None)
            ),
        )
    else:
        prm_processing_class = processing_class
        prm = COCONUTGPT2ForTokenClassification.from_pretrained(
            prm_id,
            latent_id=latent_id,
            latent_start_id=start_id,
            latent_end_id=end_id,
            target_id=target_id,
            pad_token_id=processing_class.pad_token_id,
            device_map={"": accelerator.process_index},
            torch_dtype=(
                torch.bfloat16
                if model_dtype == "bf16"
                else (torch.float16 if model_dtype == "fp16" else None)
            ),
        )
    if communication_type != "none":
        prm.communication_module = build_communication_module(
            communication_type=communication_type,
            d_model=prm.config.hidden_size,
            n_heads=communication_attention_heads,
            topk=communication_topk,
        )
        restore_communication_module_from_checkpoint(
            module_owner=prm,
            checkpoint_path=prm_id,
        )
        prm.communication_module = prm.communication_module.to(device=prm.device, dtype=prm.dtype)
    prm.eval()
    model.eval()
    dataset = datasets.Dataset.from_json(data_path)
    postfix = "\n<|start-latent|>" if new_line_after_input else "<|start-latent|>"
    dataset = dataset.map(
        lambda x, idx: {
            "idx": idx,
            "question": x["question"] + postfix,
            "answer": float(x["answer"].replace(",", "")),
        },
        with_indices=True,
    )
    dataset = dataset.map(lambda x: processing_class(x["question"]), batched=True)
    if sort_by_len:
        dataset = dataset.map(lambda x: {"length": len(x["input_ids"])})
        dataset = dataset.sort("length")
        dataset = dataset.remove_columns("length")

    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=InferenceCollator(processing_class))

    if prm_mode == "best_of_n":
        all_corrects = {}
        voting_accuracies = {}
    accuracies = {}

    if accelerator.is_main_process and progress_bar:
        pbar = tqdm(dataloader, colour="green", desc="Generating")
    else:
        pbar = dataloader

    timing_stats = {
        "num_batches": 0,
        "num_examples": 0,
        "num_trajectories": 0,
        "generation_time_sec": 0.0,
        "prm_scoring_time_sec": 0.0,
        "wall_time_sec": 0.0,
    }
    pairwise_cosine_stats = initialize_pairwise_cosine_stats()

    accelerator.wait_for_everyone()
    synchronize_device(model.device)
    run_start = time.perf_counter()

    for batch in pbar:
        timing_stats["num_batches"] += 1
        timing_stats["num_examples"] += len(batch["idx"])
        model_inputs = {
            k: v.to(model.device) for k, v in batch.items() if k in ["input_ids", "attention_mask"]
        }

        synchronize_device(model.device)
        generation_start = time.perf_counter()
        output = model.generate(
            **model_inputs,
            process_reward_model=prm if prm_mode == "beam_search" else None,
            generation_config=generation_config,
            num_return_sequences=num_return_sequences,
            return_dict_in_generate=True,
            use_cache=True,
        )
        synchronize_device(model.device)
        timing_stats["generation_time_sec"] += time.perf_counter() - generation_start
        text_output = processing_class.batch_decode(output.sequences, skip_special_tokens=True)
        timing_stats["num_trajectories"] += len(text_output)

        prm_input_texts = []
        if prm_model_family == "llama":
            for text in text_output:
                splited = text.split("\n")
                question = splited[0]
                answer = "\n".join(splited[1:])
                input_text = prm_processing_class.apply_chat_template(
                    [
                        {"role": "user", "content": question},
                        {"role": "assistant", "content": answer},
                    ],
                    tokenize=False,
                )
                prm_input_texts.append(input_text)
            inputs = prm_processing_class(prm_input_texts, return_tensors="pt", padding=True)
        else:
            inputs = prm_processing_class(text_output, return_tensors="pt", padding=True)
        if prm_mode == "best_of_n":
            synchronize_device(prm.device)
            prm_scoring_start = time.perf_counter()
            prm_outputs = prm(
                input_ids=inputs["input_ids"].to(prm.device),
                attention_mask=inputs["attention_mask"].to(prm.device),
                latent_embeds=output.latent_thoughts.to(prm.device),
                trajectory_group_size=num_return_sequences,
                return_dict=True,
            )
            prm_scores = prm_outputs.logits.squeeze(-1)  # (B * num_samples, seq)
            synchronize_device(prm.device)
            timing_stats["prm_scoring_time_sec"] += time.perf_counter() - prm_scoring_start

            if report_pairwise_cosine:
                pairwise_cosine_stats = update_pairwise_cosine_stats(
                    stats=pairwise_cosine_stats,
                    pre_latent_embeds=output.latent_thoughts,
                    post_latent_embeds=getattr(prm_outputs, "communicated_latent_embeds", None),
                    trajectory_group_size=num_return_sequences,
                    pool=pairwise_cosine_pool,
                    collapse_threshold=pairwise_collapse_threshold,
                )

            prm_scores = torch.where(
                (inputs["input_ids"] == prm.config.latent_id).to(prm.device), prm_scores, 0
            )

            prm_scores = prm_scores.sum(dim=-1)
            prm_scores = prm_scores.reshape(-1, num_return_sequences)
            top_indices = torch.topk(prm_scores, k=1, dim=1)[1]
            del prm_scores, prm_outputs
        answer_output = [
            MODELS[generator_type]["answer_extractor"](text_output[i]) for i in range(len(text_output))
        ]
        if prm_mode == "best_of_n":
            voting_result = [
                Counter(answer_output[i : i + num_return_sequences]).most_common(1)[0][0]
                for i in range(0, len(answer_output), num_return_sequences)
            ]
            voting_result = [voting_result[i] == batch["answer"][i] for i in range(len(voting_result))]

        correct = [
            answer_output[i] == batch["answer"][i // num_return_sequences] for i in range(len(answer_output))
        ]
        correct = torch.tensor(correct).view(-1, num_return_sequences)
        del output, inputs
        torch.cuda.empty_cache()
        for i, _idx in enumerate(batch["idx"]):

            if prm_mode == "best_of_n":
                all_corrects[_idx.item()] = correct[i].tolist()
                voting_accuracies[_idx.item()] = voting_result[i]
                accuracies[_idx.item()] = correct[i][top_indices[i].item()]
            else:
                accuracies[_idx.item()] = correct.item()  # only one item since num_return_sequences == 1

        if accelerator.is_main_process and progress_bar:
            _corrects = torch.tensor(list(accuracies.values())).long().sum()
            cur_num_samples = len(accuracies)
            if prm_mode == "best_of_n":
                _cov = torch.tensor(
                    [trajectory_has_any_correct(v) for v in all_corrects.values()]
                ).long().sum()
                _voting = torch.tensor(list(voting_accuracies.values())).long().sum()
                _dict = dict(
                    Cov=f"{_cov/cur_num_samples*100:.4f}% ({_cov}/{cur_num_samples})",
                    Acc=f"{_corrects/cur_num_samples*100:.4f}% ({_corrects}/{cur_num_samples})",
                    Vot=f"{_voting/cur_num_samples*100:.4f}% ({_voting}/{cur_num_samples})",
                )
            else:
                _dict = dict(
                    Acc=f"{_corrects/cur_num_samples*100:.4f}% ({_corrects}/{cur_num_samples})",
                )
            pbar.set_postfix(_dict)

    accelerator.wait_for_everyone()
    synchronize_device(model.device)
    timing_stats["wall_time_sec"] = time.perf_counter() - run_start
    if accelerator.num_processes > 1:
        accuracies = accelerator.gather_for_metrics([accuracies], use_gather_object=True)
        accuracies = {idx: value for d in accuracies for idx, value in d.items()}
        if prm_mode == "best_of_n":
            all_corrects = accelerator.gather_for_metrics([all_corrects], use_gather_object=True)
            all_corrects = {idx: value for d in all_corrects for idx, value in d.items()}
            voting_accuracies = accelerator.gather_for_metrics([voting_accuracies], use_gather_object=True)
            voting_accuracies = {idx: value for d in voting_accuracies for idx, value in d.items()}
            if report_pairwise_cosine:
                pairwise_cosine_stats = accelerator.gather_for_metrics(
                    [pairwise_cosine_stats], use_gather_object=True
                )
                pairwise_cosine_stats = merge_pairwise_cosine_stats(pairwise_cosine_stats)
        timing_stats = accelerator.gather_for_metrics([timing_stats], use_gather_object=True)
    else:
        timing_stats = [timing_stats]

    if accelerator.is_main_process:
        print(f"SEED={seed}")

        corrects = np.array(list(accuracies.values()))
        accuracy = float(corrects.mean())
        print(f"Accuracy: {accuracy*100:.4f}%")
        if prm_mode == "best_of_n":
            coverages = float(np.array([trajectory_has_any_correct(v) for v in all_corrects.values()]).mean())
            voting_accuracies = np.array(list(voting_accuracies.values()))
            voting_accuracy = float(voting_accuracies.mean())
            print(f"Coverage: {coverages*100:.4f}%")
            print(f"Voting Accuracy: {voting_accuracy*100:.4f}%")

            if report_pairwise_cosine:
                pairwise_summary = finalize_pairwise_cosine_stats(
                    pairwise_cosine_stats,
                    collapse_threshold=pairwise_collapse_threshold,
                )
                if pairwise_summary is not None:
                    print(f"Pairwise Cosine (Pre): {pairwise_summary['pre_pairwise_cos_mean']:.6f}")
                    print(f"Pairwise Cosine (Post): {pairwise_summary['post_pairwise_cos_mean']:.6f}")
                    print(f"Pairwise Cosine Delta: {pairwise_summary['delta_pairwise_cos_mean']:.6f}")
                    print(
                        f"Collapse Rate @>{pairwise_summary['collapse_threshold']:.2f} (Pre): "
                        f"{pairwise_summary['pre_collapse_rate']*100:.4f}%"
                    )
                    print(
                        f"Collapse Rate @>{pairwise_summary['collapse_threshold']:.2f} (Post): "
                        f"{pairwise_summary['post_collapse_rate']*100:.4f}%"
                    )
                    print(
                        f"Collapse Rate Delta: {pairwise_summary['delta_collapse_rate']*100:.4f}%"
                    )

        total_examples = sum(stats["num_examples"] for stats in timing_stats)
        total_trajectories = sum(stats["num_trajectories"] for stats in timing_stats)
        max_num_batches = max(stats["num_batches"] for stats in timing_stats)
        wall_time_sec = max(stats["wall_time_sec"] for stats in timing_stats)
        generation_time_sec = max(stats["generation_time_sec"] for stats in timing_stats)
        prm_scoring_time_sec = max(stats["prm_scoring_time_sec"] for stats in timing_stats)

        print(f"Wall Time (s): {wall_time_sec:.4f}")
        print(f"Generation Time (s): {generation_time_sec:.4f}")
        print(
            f"Avg Generation Time / Batch (s): {safe_divide(generation_time_sec, max_num_batches):.4f}"
        )
        print(f"Avg Wall Time / Example (ms): {safe_divide(wall_time_sec * 1000.0, total_examples):.4f}")
        print(f"Example Throughput (examples/s): {safe_divide(total_examples, wall_time_sec):.4f}")
        if total_trajectories > total_examples:
            print(
                f"Trajectory Throughput (trajectories/s): "
                f"{safe_divide(total_trajectories, wall_time_sec):.4f}"
            )
        if prm_mode == "best_of_n":
            print(f"PRM Scoring Time (s): {prm_scoring_time_sec:.4f}")
            print(
                f"Avg PRM Scoring Time / Batch (s): "
                f"{safe_divide(prm_scoring_time_sec, max_num_batches):.4f}"
            )

        if pairwise_cosine_output_path is not None:
            summary = {
                "seed": seed,
                "generator_type": generator_type,
                "prm_id": prm_id,
                "prm_mode": prm_mode,
                "generator_communication_type": generator_communication_type,
                "generator_communication_checkpoint": generator_communication_checkpoint,
                "communication_type": communication_type,
                "num_return_sequences": num_return_sequences,
                "data_path": data_path,
                "accuracy": accuracy,
                "coverage": coverages if prm_mode == "best_of_n" else None,
                "voting_accuracy": voting_accuracy if prm_mode == "best_of_n" else None,
                "wall_time_sec": wall_time_sec,
                "generation_time_sec": generation_time_sec,
                "prm_scoring_time_sec": prm_scoring_time_sec,
            }
            if report_pairwise_cosine and prm_mode == "best_of_n":
                summary["pairwise_cosine"] = finalize_pairwise_cosine_stats(
                    pairwise_cosine_stats,
                    collapse_threshold=pairwise_collapse_threshold,
                )
                summary["pairwise_cosine_pool"] = pairwise_cosine_pool
            output_dir = os.path.dirname(pairwise_cosine_output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            with open(pairwise_cosine_output_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
            print(f"Wrote pairwise cosine summary to {pairwise_cosine_output_path}")


if __name__ == "__main__":
    Fire(main)
