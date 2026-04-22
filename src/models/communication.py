import os

import torch
import torch.nn as nn


def load_local_checkpoint_state_dict(checkpoint_path: str) -> dict[str, torch.Tensor] | None:
    if os.path.isdir(checkpoint_path):
        safetensors_path = os.path.join(checkpoint_path, "model.safetensors")
        if os.path.exists(safetensors_path):
            from safetensors.torch import load_file

            return load_file(safetensors_path, device="cpu")

        pytorch_bin_path = os.path.join(checkpoint_path, "pytorch_model.bin")
        if os.path.exists(pytorch_bin_path):
            return torch.load(pytorch_bin_path, map_location="cpu")
        return None

    if os.path.isfile(checkpoint_path):
        if checkpoint_path.endswith(".safetensors"):
            from safetensors.torch import load_file

            return load_file(checkpoint_path, device="cpu")
        return torch.load(checkpoint_path, map_location="cpu")

    return None


def restore_communication_module_from_checkpoint(
    module_owner: nn.Module,
    checkpoint_path: str,
    prefix: str = "communication_module.",
) -> bool:
    communication_module = getattr(module_owner, "communication_module", None)
    if communication_module is None:
        return False

    state_dict = load_local_checkpoint_state_dict(checkpoint_path)
    if state_dict is None:
        return False

    communication_state_dict = {
        key[len(prefix) :]: value
        for key, value in state_dict.items()
        if key.startswith(prefix)
    }
    if len(communication_state_dict) == 0:
        communication_state_dict = state_dict

    missing_keys, unexpected_keys = communication_module.load_state_dict(
        communication_state_dict, strict=False
    )
    if len(unexpected_keys) > 0 and len(communication_state_dict) == len(state_dict):
        return False
    return True

class BaseCommunication(nn.Module):
    def forward(self, x, alive_mask=None, step_idx=None, return_weights=False):
        # x: [B, N, D]
        raise NotImplementedError

class IdentityCommunication(BaseCommunication):
    def forward(self, x, alive_mask=None, step_idx=None, return_weights=False):
        if return_weights:
            return x, None
        return x

class MeanBroadcastCommunication(BaseCommunication):
    def forward(self, x, alive_mask=None, step_idx=None, return_weights=False):
        if alive_mask is None:
            pooled = x.mean(dim=1, keepdim=True)
        else:
            w = alive_mask.float().unsqueeze(-1)
            pooled = (x * w).sum(dim=1, keepdim=True) / w.sum(dim=1, keepdim=True).clamp_min(1.0)
        out = x + pooled
        if return_weights:
            return out, None
        return out

class CrossPathAttentionCommunication(BaseCommunication):
    """Cross-path attention communication module.

    Design contract
    ---------------
    * **No causal mask** — attention is fully bidirectional (all-to-all across the N
      trajectory paths).  No ``attn_mask`` or ``is_causal`` flag is passed to
      ``nn.MultiheadAttention``, so every path can attend to every other path equally.

    * **No positional encoding** — ``nn.MultiheadAttention`` applies Q/K/V linear
      projections only; no positional bias is added here.  Moreover, all k paths process
      the *same* input sequence at the *same* latent step, so any positional information
      already embedded in the representations is identical across paths, keeping the
      communication purely content-based.

    * **Permutation equivariant** — standard multi-head self-attention is equivariant:
      permuting the N input paths yields the same permutation of outputs.  This is the
      correct property for a *communication* layer; each path's updated representation
      reflects information from all others while the paths remain individually
      distinguishable.

    Permutation invariance of the *final selection* is achieved externally by the reward
    model pipeline: path embeddings (enriched by this module) are scored independently
    by the RM classifier, and the best path is selected via argmax — an operation that
    is invariant to the order of its inputs.

    ``alive_mask`` semantics
    ------------------------
    ``alive_mask`` is a boolean tensor of shape ``[B, N]`` indicating which of the N
    paths are still active (generating).

    * During **RM training**, ``alive_mask`` is ``None`` (or all-ones) — no masking,
      all paths communicate freely.
    * During **generation**, ``~alive_mask`` is passed as ``key_padding_mask`` so that
      live paths do not attend to finished ones.  Finished paths may still attend to
      live ones, but their communicated outputs are never written back (the caller guards
      updates with the ``latent_sequences`` index mask).
    """

    def __init__(self, d_model, n_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x, alive_mask=None, step_idx=None, return_weights=False):
        # Mask finished paths from the KEY dimension so live paths ignore them.
        # No attn_mask is used — attention is fully bidirectional (non-causal).
        # need_weights=True disables flash attention but is only used during training.
        key_padding_mask = None if alive_mask is None else ~alive_mask
        y, attn_weights = self.attn(
            x, x, x,
            key_padding_mask=key_padding_mask,
            need_weights=return_weights,
            average_attn_weights=True,  # average over heads → [B, N, N]
        )
        out = self.ln(x + y)
        if return_weights:
            return out, attn_weights  # attn_weights: [B, N, N]
        return out

class TopKRouterCommunication(BaseCommunication):
    def __init__(self, d_model, topk=2):
        super().__init__()
        self.router = nn.Linear(d_model, d_model)
        self.score = nn.Linear(d_model, 1)
        self.topk = topk
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x, alive_mask=None, step_idx=None, return_weights=False):
        # x: [B, N, D]
        routed = self.router(x)
        scores = self.score(x).squeeze(-1)  # [B, N]
        if alive_mask is not None:
            scores = scores.masked_fill(~alive_mask, float("-inf"))
        top_idx = scores.topk(k=min(self.topk, x.size(1)), dim=1).indices
        gathered = torch.gather(
            routed,
            1,
            top_idx.unsqueeze(-1).expand(-1, -1, routed.size(-1))
        )
        summary = gathered.mean(dim=1, keepdim=True)
        out = self.ln(x + summary)
        if return_weights:
            return out, None
        return out


def build_communication_module(
    communication_type: str,
    d_model: int,
    n_heads: int = 4,
    topk: int = 2,
) -> BaseCommunication:
    if communication_type == "none":
        return IdentityCommunication()
    if communication_type == "mean":
        return MeanBroadcastCommunication()
    if communication_type == "attention":
        return CrossPathAttentionCommunication(d_model=d_model, n_heads=n_heads)
    if communication_type == "router":
        return TopKRouterCommunication(d_model=d_model, topk=topk)
    raise ValueError(f"Unsupported communication type: {communication_type}")


def apply_communication_to_latent_embeddings(
    inputs_embeds: torch.Tensor,
    input_ids: torch.LongTensor,
    latent_token_id: int,
    communication_module: BaseCommunication | None,
    trajectory_group_size: int | None,
) -> torch.Tensor:
    """Apply cross-path communication to latent token embeddings before RM scoring.

    For each latent step ``i``, the embeddings of all ``trajectory_group_size`` paths
    at that step are gathered into a ``[num_groups, trajectory_group_size, hidden_size]``
    tensor and passed through ``communication_module``.  Communication happens
    *independently per step* — there is no cross-step attention here.

    The communication module (e.g. ``CrossPathAttentionCommunication``) is *permutation
    equivariant*: permuting the trajectory order permutes its outputs identically.
    Permutation invariance of the *final selection* is achieved downstream by the RM
    scoring + argmax pipeline.

    Args:
        inputs_embeds: Full sequence embeddings, shape ``[B, S, D]``.
        input_ids: Token IDs corresponding to ``inputs_embeds``, shape ``[B, S]``.
        latent_token_id: ID of the special ``<|latent|>`` token.
        communication_module: Module implementing cross-path communication, or ``None``
            to skip communication entirely.
        trajectory_group_size: Number of trajectories per group (``N``).  The batch
            size ``B`` must be divisible by this value.  ``None`` or ``≤1`` disables
            communication.

    Returns:
        Updated ``inputs_embeds`` with latent token positions overwritten by their
        communicated counterparts.  Non-latent positions are unchanged.
    """
    if communication_module is None or trajectory_group_size is None or trajectory_group_size <= 1:
        return inputs_embeds

    total_batch, _, hidden_size = inputs_embeds.shape
    if total_batch % trajectory_group_size != 0:
        raise ValueError(
            f"Batch size {total_batch} must be divisible by trajectory_group_size={trajectory_group_size}."
        )

    latent_mask = input_ids == latent_token_id
    latent_counts = latent_mask.sum(dim=-1)
    if latent_counts.numel() == 0 or latent_counts.max().item() == 0:
        return inputs_embeds

    if not torch.all(latent_counts == latent_counts[0]):
        raise ValueError("All grouped trajectories must contain the same number of latent tokens.")

    num_groups = total_batch // trajectory_group_size
    num_latent_tokens = latent_counts[0].item()
    grouped_latents = inputs_embeds[latent_mask].view(
        total_batch, num_latent_tokens, hidden_size
    ).view(num_groups, trajectory_group_size, num_latent_tokens, hidden_size)
    alive_mask = torch.ones(
        (num_groups, trajectory_group_size), dtype=torch.bool, device=inputs_embeds.device
    )

    communicated_steps = []
    for latent_step in range(num_latent_tokens):
        communicated_steps.append(
            communication_module(
                grouped_latents[:, :, latent_step, :],
                alive_mask=alive_mask,
                step_idx=latent_step,
            )
        )

    communicated_latents = torch.stack(communicated_steps, dim=2).reshape(
        total_batch, num_latent_tokens, hidden_size
    )
    updated_inputs_embeds = inputs_embeds.clone()
    updated_inputs_embeds[latent_mask] = communicated_latents.reshape(-1, hidden_size)
    return updated_inputs_embeds
