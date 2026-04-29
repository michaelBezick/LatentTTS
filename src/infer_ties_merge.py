from collections import Counter
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
from .model_registry import MODELS
from .models.gpt2 import COCONUTGPT2ForTokenClassification
from .models.ties_merge import ties_merge_pair, ties_merge_pair_task_vector
from .utils import set_seed, InferenceCollator


@torch.no_grad()
def _decode_with_fixed_latent(
    model,
    prompt_ids: torch.Tensor,   # [S] unpadded prompt input_ids
    merged_lat: torch.Tensor,   # [L, D] merged latent embedding
    latent_id: int,
    latent_length: int,
    gen_config: LatentGenerationConfig,
    tokenizer,
) -> str:
    """Decode text by injecting a fixed merged latent into the prefill KV-cache.

    Constructs full_ids = [prompt | latent_id * L] and full_embs = [prompt_embs |
    merged_lat]. The generation mixin sees latent_sequences_end=True on the first
    step, forces <|end-latent|>, then generates text greedily from the KV-cache
    that was built with the merged latent embeddings.
    """
    device = model.device
    L = merged_lat.shape[0]

    p_ids = prompt_ids.unsqueeze(0).to(device)                              # [1, S]
    lat_ids = torch.full((1, L), latent_id, dtype=p_ids.dtype, device=device)
    full_ids = torch.cat([p_ids, lat_ids], dim=1)                           # [1, S+L]

    emb = model.get_input_embeddings()
    p_embs = emb(p_ids)                                                     # [1, S, D]
    m_embs = merged_lat.to(device=device, dtype=model.dtype).unsqueeze(0)  # [1, L, D]
    full_embs = torch.cat([p_embs, m_embs], dim=1)                         # [1, S+L, D]

    full_attn = torch.ones(1, full_ids.shape[1], device=device, dtype=torch.long)

    out = model.generate(
        input_ids=full_ids,
        attention_mask=full_attn,
        inputs_embeds=full_embs,
        generation_config=gen_config,
        num_return_sequences=1,
        return_dict_in_generate=False,
        use_cache=True,
    )
    return tokenizer.decode(out[0], skip_special_tokens=True)


def synchronize_device(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def trajectory_has_any_correct(value) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(value.any().item())
    return any(value)


def build_best_anchored_pairs(scores: torch.Tensor, merge_mode: str) -> tuple[int, list[tuple[int, int]]]:
    """Return merge pairs that always keep the global-best trajectory as base."""
    if scores.ndim != 1:
        raise ValueError(f"Expected 1D scores, got shape {tuple(scores.shape)}")
    if scores.numel() < 2:
        raise ValueError("At least two trajectories are required for TIES merging")

    sorted_idx = scores.argsort(descending=True)
    best_idx = sorted_idx[0].item()

    if merge_mode == "greedy":
        return best_idx, [(best_idx, sorted_idx[1].item())]
    if merge_mode == "all_pairs":
        return best_idx, [(best_idx, j) for j in range(scores.numel()) if j != best_idx]
    raise ValueError(f"Unsupported merge_mode={merge_mode!r}")


def merge_count_offsets(counts: list[int]) -> list[tuple[int, int]]:
    offsets = []
    offset = 0
    for count in counts:
        offsets.append((offset, offset + count))
        offset += count
    return offsets


def choose_ties_correctness(
    base_correct: bool,
    best_original_score: float,
    merged_scores: torch.Tensor,
    acceptance_margin: float,
) -> tuple[bool, bool, float]:
    """Select diagnostic TIES correctness while preserving the base answer."""
    if len(merged_scores) == 0:
        return base_correct, False, 0.0

    max_merged_score = merged_scores.max().item()
    score_gain = max_merged_score - best_original_score
    accepted = max_merged_score > best_original_score + acceptance_margin
    return base_correct, accepted, score_gain if accepted else 0.0


def score_prm_latents(
    prm,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    latent_embeds,
    trajectory_group_size: int = 1,
) -> torch.Tensor:
    prm_out = prm(
        input_ids=input_ids,
        attention_mask=attention_mask,
        latent_embeds=latent_embeds,
        trajectory_group_size=trajectory_group_size,
        return_dict=True,
    )
    prm_scores = prm_out.logits.squeeze(-1)
    prm_scores = torch.where(
        input_ids == prm.config.latent_id,
        prm_scores,
        torch.zeros_like(prm_scores),
    ).sum(dim=-1)
    return prm_scores


@torch.no_grad()
def main(
    generator_type: Literal["coconut", "codi"] = "coconut",
    model_dtype: Literal["bf16", "fp16", "fp32"] = "fp32",
    prm_id: str = "checkpoints/latentRM",
    data_path: str = "data/gsm_valid.json",
    num_return_sequences: int = 8,
    batch_size: int = 1024,
    latent_length: int = 6,
    max_new_tokens: int | None = 128,
    seed: int = 200,
    sort_by_len: bool = True,
    progress_bar: bool = True,
    sampling_by: Literal["dropout", "noise"] = "dropout",
    noise_std: float | None = 0.1,
    dropout_p: float | None = None,
    # TIES-specific args
    top_k_ratio: float = 0.2,
    merge_mode: Literal["all_pairs", "greedy"] = "all_pairs",
    use_task_vectors: bool = False,
    acceptance_margin: float = 0.0,
    merge_score_chunk_size: int = 64,
):
    """Best-of-N inference with post-generation TIES latent merging.

    Generates num_return_sequences trajectories per example, scores all with
    latentRM, then for each pair applies a TIES merge (dominant same-sign
    coordinates averaged, rest kept from the higher-scoring trajectory) and
    re-scores. Accepts a merge only when it beats the best original score.
    Reports both base best-of-N accuracy and TIES accuracy.

    merge_mode="all_pairs"  — try all pairs anchored on the best trajectory
    merge_mode="greedy"     — try only the top-2 scoring pair (1 extra RM call)
    use_task_vectors=True   — apply TIES to (emb - mean_emb) instead of raw embs
    acceptance_margin        accept only if merged_score > best_score + margin
    """
    assert num_return_sequences > 1, "num_return_sequences must be > 1 for merging"
    batch_size = max(1, batch_size // num_return_sequences)

    set_seed(seed)
    accelerator = Accelerator()
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        print(f"Loading {data_path}")

    processing_class = AutoTokenizer.from_pretrained(MODELS[generator_type]["id"])
    if processing_class.pad_token is None:
        processing_class.pad_token = processing_class.eos_token

    latent_id = processing_class.convert_tokens_to_ids("<|latent|>")
    start_id = processing_class.convert_tokens_to_ids("<|start-latent|>")
    end_id = processing_class.convert_tokens_to_ids("<|end-latent|>")
    target_id = processing_class.convert_tokens_to_ids(">>")
    assert latent_id != start_id and latent_id != end_id and start_id != end_id

    generation_config = LatentGenerationConfig(
        max_new_tokens=max_new_tokens,
        latent_length=latent_length,
        latent_do_sample=True,
        latent_do_sample_by=sampling_by,
        noise_std=noise_std,
        dropout_p=dropout_p,
        communication_type="none",
        communication_every=1,
        num_beams=1,
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
    prm = COCONUTGPT2ForTokenClassification.from_pretrained(
        prm_id,
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
    prm.eval()
    model.eval()

    new_line_after_input = generator_type == "coconut"
    postfix = "\n<|start-latent|>" if new_line_after_input else "<|start-latent|>"
    dataset = datasets.Dataset.from_json(data_path)
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

    base_accuracies = {}
    ties_accuracies = {}
    all_corrects = {}
    total_merge_accepted = 0
    total_pairs_tried = 0
    total_merge_gain = 0.0

    timing_stats = {
        "num_batches": 0,
        "num_examples": 0,
        "generation_time_sec": 0.0,
        "prm_initial_time_sec": 0.0,
        "prm_merge_time_sec": 0.0,
        "wall_time_sec": 0.0,
    }

    if accelerator.is_main_process and progress_bar:
        pbar = tqdm(dataloader, colour="green", desc="TIES Merge Inference")
    else:
        pbar = dataloader

    accelerator.wait_for_everyone()
    synchronize_device(model.device)
    run_start = time.perf_counter()

    for batch in pbar:
        timing_stats["num_batches"] += 1
        timing_stats["num_examples"] += len(batch["idx"])
        N = num_return_sequences
        model_inputs = {
            k: v.to(model.device) for k, v in batch.items() if k in ["input_ids", "attention_mask"]
        }

        # --- Generation ---
        synchronize_device(model.device)
        gen_start = time.perf_counter()
        output = model.generate(
            **model_inputs,
            generation_config=generation_config,
            num_return_sequences=N,
            return_dict_in_generate=True,
            use_cache=True,
        )
        synchronize_device(model.device)
        timing_stats["generation_time_sec"] += time.perf_counter() - gen_start

        text_output = processing_class.batch_decode(output.sequences, skip_special_tokens=True)
        inputs = processing_class(text_output, return_tensors="pt", padding=True)

        # --- Initial RM scoring ---
        synchronize_device(prm.device)
        prm_init_start = time.perf_counter()
        prm_input_ids = inputs["input_ids"].to(prm.device)
        prm_attention_mask = inputs["attention_mask"].to(prm.device)
        prm_scores_raw = score_prm_latents(
            prm=prm,
            input_ids=prm_input_ids,
            attention_mask=prm_attention_mask,
            latent_embeds=output.latent_thoughts.to(prm.device),
            trajectory_group_size=1,
        )
        original_scores = prm_scores_raw.reshape(-1, N).cpu()  # [B, N]
        synchronize_device(prm.device)
        timing_stats["prm_initial_time_sec"] += time.perf_counter() - prm_init_start

        # --- Answers and correctness ---
        answer_output = [MODELS[generator_type]["answer_extractor"](t) for t in text_output]
        correct = torch.tensor([
            answer_output[i] == batch["answer"][i // N]
            for i in range(len(answer_output))
        ]).view(-1, N)  # [B, N]

        # --- Build TIES merge candidates ---
        latent_thoughts = output.latent_thoughts.cpu()  # [B*N, L, D]
        seq_ids_cpu = inputs["input_ids"].cpu()         # [B*N, S]
        B_orig = original_scores.shape[0]

        all_merged_latents = []  # [L, D] each
        all_merged_ids = []      # [S] each
        merge_info = []          # (b, base_traj_idx)
        merge_counts = []

        for b in range(B_orig):
            lats_b = latent_thoughts[b * N:(b + 1) * N]  # [N, L, D]
            ids_b = seq_ids_cpu[b * N:(b + 1) * N]       # [N, S]
            scores_b = original_scores[b]                 # [N]
            mean_lat = lats_b.float().mean(dim=0) if use_task_vectors else None
            base_traj_idx, pairs_to_try = build_best_anchored_pairs(scores_b, merge_mode)
            merge_counts.append(len(pairs_to_try))

            for i, j in pairs_to_try:
                emb_base = lats_b[i].float()
                emb_other = lats_b[j].float()

                if use_task_vectors:
                    merged = ties_merge_pair_task_vector(emb_base, emb_other, mean_lat, top_k_ratio)
                else:
                    merged = ties_merge_pair(emb_base, emb_other, top_k_ratio)

                all_merged_latents.append(merged.to(lats_b.dtype))
                all_merged_ids.append(ids_b[base_traj_idx])
                merge_info.append((b, base_traj_idx))

        # --- Score all merged pairs ---
        synchronize_device(prm.device)
        merge_score_start = time.perf_counter()
        all_merged_scores_parts = []
        for c_start in range(0, len(all_merged_latents), merge_score_chunk_size):
            c_end = min(c_start + merge_score_chunk_size, len(all_merged_latents))
            chunk_lats = [all_merged_latents[k].to(prm.device) for k in range(c_start, c_end)]
            chunk_ids = torch.stack(all_merged_ids[c_start:c_end]).to(prm.device)
            chunk_attn = (chunk_ids != processing_class.pad_token_id).long()
            chunk_scores = score_prm_latents(
                prm=prm,
                input_ids=chunk_ids,
                attention_mask=chunk_attn,
                latent_embeds=chunk_lats,
                trajectory_group_size=1,
            )
            all_merged_scores_parts.append(chunk_scores.cpu())
        synchronize_device(prm.device)
        timing_stats["prm_merge_time_sec"] += time.perf_counter() - merge_score_start

        all_merged_scores = (
            torch.cat(all_merged_scores_parts) if all_merged_scores_parts else torch.empty(0)
        )

        # --- Select best per example ---
        for b, (pair_offset, pair_end) in enumerate(merge_count_offsets(merge_counts)):
            _idx = batch["idx"][b].item()
            scores_b = original_scores[b]
            base_best_idx = scores_b.argmax().item()
            base_correct = correct[b][base_best_idx].item()
            base_accuracies[_idx] = base_correct
            all_corrects[_idx] = correct[b].tolist()

            merged_scores_b = all_merged_scores[pair_offset:pair_end]
            total_pairs_tried += merge_counts[b]

            ties_correct, accepted, score_gain = choose_ties_correctness(
                base_correct=base_correct,
                best_original_score=scores_b[base_best_idx].item(),
                merged_scores=merged_scores_b,
                acceptance_margin=acceptance_margin,
            )
            ties_accuracies[_idx] = ties_correct
            if accepted:
                if merge_info:
                    _, base_traj_idx = merge_info[pair_offset + merged_scores_b.argmax().item()]
                    assert base_traj_idx == base_best_idx
                total_merge_accepted += 1
                total_merge_gain += score_gain

        del output, inputs, latent_thoughts
        torch.cuda.empty_cache()

        if accelerator.is_main_process and progress_bar:
            base_sum = sum(base_accuracies.values())
            ties_sum = sum(ties_accuracies.values())
            n = len(base_accuracies)
            pbar.set_postfix({
                "Base": f"{base_sum/n*100:.2f}%",
                "TIES": f"{ties_sum/n*100:.2f}%",
                "Acc": f"{total_merge_accepted}/{total_pairs_tried}",
            })

    accelerator.wait_for_everyone()
    synchronize_device(model.device)
    timing_stats["wall_time_sec"] = time.perf_counter() - run_start

    if accelerator.num_processes > 1:
        base_accuracies = accelerator.gather_for_metrics([base_accuracies], use_gather_object=True)
        base_accuracies = {idx: v for d in base_accuracies for idx, v in d.items()}
        ties_accuracies = accelerator.gather_for_metrics([ties_accuracies], use_gather_object=True)
        ties_accuracies = {idx: v for d in ties_accuracies for idx, v in d.items()}
        all_corrects = accelerator.gather_for_metrics([all_corrects], use_gather_object=True)
        all_corrects = {idx: v for d in all_corrects for idx, v in d.items()}
        merge_stats = accelerator.gather_for_metrics(
            [{
                "accepted": total_merge_accepted,
                "pairs_tried": total_pairs_tried,
                "gain": total_merge_gain,
            }],
            use_gather_object=True,
        )
        total_merge_accepted = sum(s["accepted"] for s in merge_stats)
        total_pairs_tried = sum(s["pairs_tried"] for s in merge_stats)
        total_merge_gain = sum(s["gain"] for s in merge_stats)
        timing_stats = accelerator.gather_for_metrics([timing_stats], use_gather_object=True)
    else:
        timing_stats = [timing_stats]

    if accelerator.is_main_process:
        print(f"SEED={seed}")
        base_arr = np.array(list(base_accuracies.values()))
        ties_arr = np.array(list(ties_accuracies.values()))
        coverage = float(np.array([trajectory_has_any_correct(v) for v in all_corrects.values()]).mean())

        print(f"Base Best-of-{N} Accuracy: {base_arr.mean()*100:.4f}%")
        print(f"TIES Merge Accuracy:        {ties_arr.mean()*100:.4f}%")
        print(f"Coverage (any correct):     {coverage*100:.4f}%")
        print(f"Merge Accept Rate:          {safe_divide(total_merge_accepted, total_pairs_tried)*100:.4f}%"
              f" ({total_merge_accepted}/{total_pairs_tried})")
        if total_merge_accepted > 0:
            print(f"Avg Score Gain (accepted):  {total_merge_gain / total_merge_accepted:.6f}")
        print(
            "Merge Mode: "
            f"{merge_mode}, top_k_ratio={top_k_ratio}, use_task_vectors={use_task_vectors}, "
            f"acceptance_margin={acceptance_margin}"
        )

        wall = max(s["wall_time_sec"] for s in timing_stats)
        gen = max(s["generation_time_sec"] for s in timing_stats)
        prm_init = max(s["prm_initial_time_sec"] for s in timing_stats)
        prm_merge = max(s["prm_merge_time_sec"] for s in timing_stats)
        n_ex = sum(s["num_examples"] for s in timing_stats)
        print(f"Wall Time (s): {wall:.2f} | Gen: {gen:.2f} | PRM-init: {prm_init:.2f} | PRM-merge: {prm_merge:.2f}")
        print(f"Throughput: {safe_divide(n_ex, wall):.2f} examples/s")


if __name__ == "__main__":
    Fire(main)
