import time
from typing import Literal

import datasets
import numpy as np
import torch
from accelerate import Accelerator
from fire import Fire
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from .generation_mixin import LatentGenerationConfig, LatentGenerationMixin
from .infer_ties_merge import (
    _decode_with_fixed_latents,
    choose_decoded_ties_correctness,
    merge_count_offsets,
    safe_divide,
    score_prm_latents,
    synchronize_device,
    trajectory_has_any_correct,
)
from .model_registry import MODELS
from .models.gpt2 import COCONUTGPT2ForTokenClassification
from .utils import InferenceCollator, set_seed


def build_best_anchored_splices(
    scores: torch.Tensor,
    latent_length: int,
) -> tuple[int, list[tuple[int, int, int]]]:
    """Return (base, donor, cut) splices anchored on the PRM-best trajectory."""
    if scores.ndim != 1:
        raise ValueError(f"Expected 1D scores, got shape {tuple(scores.shape)}")
    if scores.numel() < 2:
        raise ValueError("At least two trajectories are required for latent splicing")
    if latent_length < 2:
        raise ValueError("latent_length must be at least 2 for latent splicing")

    base_idx = scores.argmax().item()
    candidates = [
        (base_idx, donor_idx, cut)
        for donor_idx in range(scores.numel())
        if donor_idx != base_idx
        for cut in range(1, latent_length)
    ]
    return base_idx, candidates


def splice_latents(
    emb_base: torch.Tensor,
    emb_donor: torch.Tensor,
    cut: int,
) -> torch.Tensor:
    """Preserve base prefix and use donor suffix at a latent-token boundary."""
    if emb_base.shape != emb_donor.shape:
        raise ValueError(f"Expected equal latent shapes, got {emb_base.shape} and {emb_donor.shape}")
    if not 0 < cut < emb_base.shape[0]:
        raise ValueError(f"Expected 0 < cut < {emb_base.shape[0]}, got {cut}")
    return torch.cat([emb_base[:cut], emb_donor[cut:]], dim=0)


@torch.no_grad()
def main(
    generator_type: Literal["coconut", "codi"] = "coconut",
    model_dtype: Literal["bf16", "fp16", "fp32"] = "bf16",
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
    acceptance_margin: float = 0.0,
    splice_score_chunk_size: int = 64,
    splice_decode_batch_size: int = 64,
):
    """Best-of-N inference with PRM-best anchored latent suffix splicing.

    For each example, generate N latent trajectories and score them with
    latentRM. Keep the PRM-best trajectory as the prefix anchor, then splice in
    every other trajectory's suffix at every internal latent boundary. Each
    spliced latent sequence is decoded and PRM-reranked against the original
    best trajectory.
    """
    assert num_return_sequences > 1, "num_return_sequences must be > 1 for splicing"
    assert latent_length > 1, "latent_length must be > 1 for splicing"
    assert splice_decode_batch_size > 0, "splice_decode_batch_size must be > 0"
    assert splice_score_chunk_size > 0, "splice_score_chunk_size must be > 0"
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

    dtype = torch.bfloat16 if model_dtype == "bf16" else (torch.float16 if model_dtype == "fp16" else None)
    model = LatentGPT2LMHeadModel.from_pretrained(
        MODELS[generator_type]["id"],
        latent_id=latent_id,
        latent_start_id=start_id,
        latent_end_id=end_id,
        target_id=target_id,
        pad_token_id=processing_class.pad_token_id,
        device_map={"": accelerator.process_index},
        torch_dtype=dtype,
    )
    prm = COCONUTGPT2ForTokenClassification.from_pretrained(
        prm_id,
        latent_id=latent_id,
        latent_start_id=start_id,
        latent_end_id=end_id,
        target_id=target_id,
        pad_token_id=processing_class.pad_token_id,
        device_map={"": accelerator.process_index},
        torch_dtype=dtype,
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
    splice_accuracies = {}
    all_corrects = {}
    splice_oracle_corrects = {}
    total_splice_accepted = 0
    total_splice_accepted_correct = 0
    total_splice_accepted_incorrect = 0
    total_splices_tried = 0
    total_splice_gain = 0.0

    timing_stats = {
        "num_batches": 0,
        "num_examples": 0,
        "generation_time_sec": 0.0,
        "splice_decode_time_sec": 0.0,
        "prm_initial_time_sec": 0.0,
        "prm_splice_time_sec": 0.0,
        "wall_time_sec": 0.0,
    }

    if accelerator.is_main_process and progress_bar:
        pbar = tqdm(dataloader, colour="green", desc="Latent Splice Inference")
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

        synchronize_device(prm.device)
        prm_init_start = time.perf_counter()
        prm_scores_raw = score_prm_latents(
            prm=prm,
            input_ids=inputs["input_ids"].to(prm.device),
            attention_mask=inputs["attention_mask"].to(prm.device),
            latent_embeds=output.latent_thoughts.to(prm.device),
            trajectory_group_size=1,
        )
        original_scores = prm_scores_raw.reshape(-1, N).cpu()
        synchronize_device(prm.device)
        timing_stats["prm_initial_time_sec"] += time.perf_counter() - prm_init_start

        answer_output = [MODELS[generator_type]["answer_extractor"](t) for t in text_output]
        correct = torch.tensor([
            answer_output[i] == batch["answer"][i // N]
            for i in range(len(answer_output))
        ]).view(-1, N)

        latent_thoughts = output.latent_thoughts.cpu()
        prompt_ids_cpu = batch["input_ids"].cpu()
        B_orig = original_scores.shape[0]

        all_spliced_latents = []
        all_spliced_prompt_ids = []
        splice_info = []
        splice_counts = []

        for b in range(B_orig):
            lats_b = latent_thoughts[b * N:(b + 1) * N]
            scores_b = original_scores[b]
            base_traj_idx, splices_to_try = build_best_anchored_splices(
                scores=scores_b,
                latent_length=lats_b.shape[1],
            )
            splice_counts.append(len(splices_to_try))
            prompt_ids_b = prompt_ids_cpu[b][prompt_ids_cpu[b] != processing_class.pad_token_id]

            for base_idx, donor_idx, cut in splices_to_try:
                assert base_idx == base_traj_idx
                spliced = splice_latents(lats_b[base_idx], lats_b[donor_idx], cut)
                all_spliced_latents.append(spliced.to(lats_b.dtype))
                all_spliced_prompt_ids.append(prompt_ids_b)
                splice_info.append((b, base_traj_idx, donor_idx, cut))

        synchronize_device(model.device)
        splice_decode_start = time.perf_counter()
        spliced_text_output = []
        for c_start in range(0, len(all_spliced_latents), splice_decode_batch_size):
            c_end = min(c_start + splice_decode_batch_size, len(all_spliced_latents))
            spliced_text_output.extend(
                _decode_with_fixed_latents(
                    model=model,
                    prompt_ids=all_spliced_prompt_ids[c_start:c_end],
                    merged_lats=all_spliced_latents[c_start:c_end],
                    latent_id=latent_id,
                    gen_config=generation_config,
                    tokenizer=processing_class,
                )
            )
        synchronize_device(model.device)
        timing_stats["splice_decode_time_sec"] += time.perf_counter() - splice_decode_start

        if spliced_text_output:
            spliced_inputs = processing_class(spliced_text_output, return_tensors="pt", padding=True)
            spliced_answer_output = [
                MODELS[generator_type]["answer_extractor"](t) for t in spliced_text_output
            ]
            spliced_correct = torch.tensor([
                spliced_answer_output[k] == batch["answer"][splice_info[k][0]]
                for k in range(len(spliced_answer_output))
            ])
        else:
            spliced_inputs = None
            spliced_correct = torch.empty(0, dtype=torch.bool)

        synchronize_device(prm.device)
        splice_score_start = time.perf_counter()
        all_spliced_scores_parts = []
        for c_start in range(0, len(all_spliced_latents), splice_score_chunk_size):
            c_end = min(c_start + splice_score_chunk_size, len(all_spliced_latents))
            chunk_lats = [all_spliced_latents[k].to(prm.device) for k in range(c_start, c_end)]
            chunk_scores = score_prm_latents(
                prm=prm,
                input_ids=spliced_inputs["input_ids"][c_start:c_end].to(prm.device),
                attention_mask=spliced_inputs["attention_mask"][c_start:c_end].to(prm.device),
                latent_embeds=chunk_lats,
                trajectory_group_size=1,
            )
            all_spliced_scores_parts.append(chunk_scores.cpu())
        synchronize_device(prm.device)
        timing_stats["prm_splice_time_sec"] += time.perf_counter() - splice_score_start

        all_spliced_scores = (
            torch.cat(all_spliced_scores_parts) if all_spliced_scores_parts else torch.empty(0)
        )

        for b, (splice_offset, splice_end) in enumerate(merge_count_offsets(splice_counts)):
            _idx = batch["idx"][b].item()
            scores_b = original_scores[b]
            base_best_idx = scores_b.argmax().item()
            base_correct = correct[b][base_best_idx].item()
            base_accuracies[_idx] = base_correct
            all_corrects[_idx] = correct[b].tolist()

            spliced_scores_b = all_spliced_scores[splice_offset:splice_end]
            spliced_correct_b = spliced_correct[splice_offset:splice_end]
            total_splices_tried += splice_counts[b]

            splice_correct, accepted, score_gain, best_spliced_correct = choose_decoded_ties_correctness(
                base_correct=base_correct,
                best_original_score=scores_b[base_best_idx].item(),
                merged_scores=spliced_scores_b,
                merged_correct=spliced_correct_b,
                acceptance_margin=acceptance_margin,
            )
            splice_accuracies[_idx] = splice_correct
            splice_oracle_corrects[_idx] = bool(
                trajectory_has_any_correct(correct[b]) or trajectory_has_any_correct(spliced_correct_b)
            )
            if accepted:
                if splice_info:
                    _, base_traj_idx, _, _ = splice_info[
                        splice_offset + spliced_scores_b.argmax().item()
                    ]
                    assert base_traj_idx == base_best_idx
                total_splice_accepted += 1
                total_splice_gain += score_gain
                if best_spliced_correct:
                    total_splice_accepted_correct += 1
                else:
                    total_splice_accepted_incorrect += 1

        del output, inputs, latent_thoughts, spliced_inputs
        torch.cuda.empty_cache()

        if accelerator.is_main_process and progress_bar:
            base_sum = sum(base_accuracies.values())
            splice_sum = sum(splice_accuracies.values())
            n = len(base_accuracies)
            pbar.set_postfix({
                "Base": f"{base_sum/n*100:.2f}%",
                "Splice": f"{splice_sum/n*100:.2f}%",
                "Accepted": f"{total_splice_accepted}/{total_splices_tried}",
            })

    accelerator.wait_for_everyone()
    synchronize_device(model.device)
    timing_stats["wall_time_sec"] = time.perf_counter() - run_start

    if accelerator.num_processes > 1:
        base_accuracies = accelerator.gather_for_metrics([base_accuracies], use_gather_object=True)
        base_accuracies = {idx: v for d in base_accuracies for idx, v in d.items()}
        splice_accuracies = accelerator.gather_for_metrics([splice_accuracies], use_gather_object=True)
        splice_accuracies = {idx: v for d in splice_accuracies for idx, v in d.items()}
        all_corrects = accelerator.gather_for_metrics([all_corrects], use_gather_object=True)
        all_corrects = {idx: v for d in all_corrects for idx, v in d.items()}
        splice_oracle_corrects = accelerator.gather_for_metrics(
            [splice_oracle_corrects],
            use_gather_object=True,
        )
        splice_oracle_corrects = {idx: v for d in splice_oracle_corrects for idx, v in d.items()}
        splice_stats = accelerator.gather_for_metrics(
            [{
                "accepted": total_splice_accepted,
                "accepted_correct": total_splice_accepted_correct,
                "accepted_incorrect": total_splice_accepted_incorrect,
                "splices_tried": total_splices_tried,
                "gain": total_splice_gain,
            }],
            use_gather_object=True,
        )
        total_splice_accepted = sum(s["accepted"] for s in splice_stats)
        total_splice_accepted_correct = sum(s["accepted_correct"] for s in splice_stats)
        total_splice_accepted_incorrect = sum(s["accepted_incorrect"] for s in splice_stats)
        total_splices_tried = sum(s["splices_tried"] for s in splice_stats)
        total_splice_gain = sum(s["gain"] for s in splice_stats)
        timing_stats = accelerator.gather_for_metrics([timing_stats], use_gather_object=True)
    else:
        timing_stats = [timing_stats]

    if accelerator.is_main_process:
        print(f"SEED={seed}")
        base_arr = np.array(list(base_accuracies.values()))
        splice_arr = np.array(list(splice_accuracies.values()))
        coverage = float(np.array([trajectory_has_any_correct(v) for v in all_corrects.values()]).mean())
        splice_oracle_coverage = float(np.array(list(splice_oracle_corrects.values())).mean())

        print(f"Base Best-of-{N} Accuracy: {base_arr.mean()*100:.4f}%")
        print(f"Latent Splice Accuracy:    {splice_arr.mean()*100:.4f}%")
        print(f"Coverage:                  {coverage*100:.4f}%")
        print(f"Splice Oracle Coverage:    {splice_oracle_coverage*100:.4f}%")
        print(f"Splice Accept Rate:        {safe_divide(total_splice_accepted, total_splices_tried)*100:.4f}%"
              f" ({total_splice_accepted}/{total_splices_tried})")
        print(
            "Accepted Splices Correct/Incorrect: "
            f"{total_splice_accepted_correct}/{total_splice_accepted_incorrect}"
        )
        if total_splice_accepted > 0:
            print(f"Avg Score Gain (accepted): {total_splice_gain / total_splice_accepted:.6f}")
        print(
            "Splice Mode: "
            f"best_anchored_suffix_all_pairs, acceptance_margin={acceptance_margin}, "
            f"splice_decode_batch_size={splice_decode_batch_size}"
        )

        wall = max(s["wall_time_sec"] for s in timing_stats)
        gen = max(s["generation_time_sec"] for s in timing_stats)
        splice_decode = max(s["splice_decode_time_sec"] for s in timing_stats)
        prm_init = max(s["prm_initial_time_sec"] for s in timing_stats)
        prm_splice = max(s["prm_splice_time_sec"] for s in timing_stats)
        n_ex = sum(s["num_examples"] for s in timing_stats)
        print(
            f"Wall Time (s): {wall:.2f} | Gen: {gen:.2f} | Splice-decode: {splice_decode:.2f} "
            f"| PRM-init: {prm_init:.2f} | PRM-splice: {prm_splice:.2f}"
        )
        print(f"Throughput: {safe_divide(n_ex, wall):.2f} examples/s")


if __name__ == "__main__":
    Fire(main)
