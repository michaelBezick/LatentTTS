import torch
import torch.nn as nn

class BaseCommunication(nn.Module):
    def forward(self, x, alive_mask=None, step_idx=None):
        # x: [B, N, D]
        raise NotImplementedError

class IdentityCommunication(BaseCommunication):
    def forward(self, x, alive_mask=None, step_idx=None):
        return x

class MeanBroadcastCommunication(BaseCommunication):
    def forward(self, x, alive_mask=None, step_idx=None):
        if alive_mask is None:
            pooled = x.mean(dim=1, keepdim=True)
        else:
            w = alive_mask.float().unsqueeze(-1)
            pooled = (x * w).sum(dim=1, keepdim=True) / w.sum(dim=1, keepdim=True).clamp_min(1.0)
        return x + pooled

class CrossPathAttentionCommunication(BaseCommunication):
    def __init__(self, d_model, n_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x, alive_mask=None, step_idx=None):
        key_padding_mask = None if alive_mask is None else ~alive_mask
        y, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask, need_weights=False)
        return self.ln(x + y)

class TopKRouterCommunication(BaseCommunication):
    def __init__(self, d_model, topk=2):
        super().__init__()
        self.router = nn.Linear(d_model, d_model)
        self.score = nn.Linear(d_model, 1)
        self.topk = topk
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x, alive_mask=None, step_idx=None):
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
        return self.ln(x + summary)


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
