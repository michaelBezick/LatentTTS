import torch


def ties_merge_pair(
    emb_base: torch.Tensor,
    emb_other: torch.Tensor,
    top_k_ratio: float = 0.2,
) -> torch.Tensor:
    """TIES-style merge of two latent embeddings [L, D].

    Per latent step: averages coordinates that are top-k% by magnitude in
    *both* embeddings and agree in sign. All other coordinates fall back to
    emb_base (the higher-scoring trajectory).
    """
    L, D = emb_base.shape
    k = max(1, int(top_k_ratio * D))
    merged = emb_base.clone()
    for l in range(L):
        a = emb_base[l]
        b = emb_other[l]
        thresh_a = a.abs().topk(k).values.min()
        thresh_b = b.abs().topk(k).values.min()
        dom_a = a.abs() >= thresh_a
        dom_b = b.abs() >= thresh_b
        same_sign = (a * b) > 0
        merge_mask = dom_a & dom_b & same_sign
        merged[l] = torch.where(merge_mask, (a + b) / 2, a)
    return merged


def ties_merge_pair_task_vector(
    emb_base: torch.Tensor,
    emb_other: torch.Tensor,
    emb_mean: torch.Tensor,
    top_k_ratio: float = 0.2,
) -> torch.Tensor:
    """TIES merge in task-vector space (deviations from mean latent).

    Computes task vectors as (emb - emb_mean), applies TIES merge to the
    task vectors, then adds emb_mean back. Closer to the original paper.
    """
    tv_base = emb_base - emb_mean
    tv_other = emb_other - emb_mean
    tv_merged = ties_merge_pair(tv_base, tv_other, top_k_ratio)
    return emb_mean + tv_merged
