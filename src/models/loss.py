import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
import torch.nn.functional as F


class MaskedBCEWithLogitsLoss(BCEWithLogitsLoss):
    def __init__(self, ignore_index: int = -100):
        super().__init__()
        self.ignore_index = ignore_index
        self.func = self._filter_ignore_index

    def _filter_ignore_index(self, input: torch.Tensor, target: torch.Tensor):
        """
        Filter out the logits and labels that are ignored.
        Args:
            input (torch.Tensor): The input logits, (B, S, D)
            target (torch.Tensor): The target labels, (B, S)
        Returns:
            tuple: A tuple of filtered logits and labels, (N, D), (N,)
            where N is the number of valid tokens of all sequences.
        """
        # logits (B, S, D) labels (B, S) mask (B, S)
        mask = target != self.ignore_index
        # mask (B, S, D)
        mask_expand = mask.unsqueeze(-1).expand_as(input)
        # logits (B, S, D) -> (N, D)
        logits_filtered = input[mask_expand]
        logits_filtered = logits_filtered.view(-1, input.shape[-1])

        # labels (B, S) -> (N,)
        labels_filtered = target[mask]
        labels_filtered = labels_filtered.view(-1).to(logits_filtered.dtype)

        return logits_filtered, labels_filtered

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        # input: logits (B, S, 1), target: labels (B, S)
        logits_filtered, labels_filtered = self.func(input, target)
        loss = super().forward(logits_filtered.squeeze(-1), labels_filtered)

        return loss


class MaskedCrossEntropyLoss(CrossEntropyLoss):
    def __init__(
        self,
    ):
        super().__init__()

    def forward(self, input: torch.Tensor, target: torch.Tensor, n_samples: int):
        # logits (B, seq_len, 1) -> (a_B, n_samples, seq_len), labels (B, seq_len) -> (a_B, n_samples, seq_len)
        actual_batch_size = input.shape[0] // n_samples
        logits_filtered = input.squeeze(-1).view(actual_batch_size, n_samples, -1)
        labels_filtered = target.view(actual_batch_size, n_samples, -1)
        # set the label on -100 postion
        _mask = labels_filtered == -100
        # labels_filtered = torch.where(_mask, 0.0, labels_filtered)
        # logits_filtered = torch.where(_mask, 0.0, logits_filtered)
        log_probs = F.log_softmax(logits_filtered, dim=1)
        # mask out the padding tokens
        log_probs = log_probs * ~_mask

        # labels_filtered = labels_filtered.softmax(dim=1)
        labels_filtered = labels_filtered * ~_mask
        # dim 1 (-2) is the n_samples dimension, which is also the "classification" dimension
        loss = -(labels_filtered * log_probs).sum(dim=1).mean()

        return loss


class DiversityPenaltyLoss(nn.Module):
    """Penalizes low diversity across trajectories in the same group.

    For each group of ``N`` trajectories, mean-pools the ``L`` post-communication
    latent embeddings into a single vector per trajectory, L2-normalizes them, then
    averages all ``N*(N-1)/2`` pairwise cosine similarities.  A higher value means
    the trajectories are more similar → larger penalty → the communication module is
    pushed to maintain diversity.

    Args:
        communicated_latent_embeds: Shape ``[B, L, D]`` where
            ``B = num_groups * trajectory_group_size``.
        trajectory_group_size: Number of trajectories per group (``N``).

    Returns:
        Scalar mean pairwise cosine similarity, range approximately ``[-1, 1]``.
    """

    def forward(
        self,
        communicated_latent_embeds: torch.Tensor,
        trajectory_group_size: int,
    ) -> torch.Tensor:
        B, L, D = communicated_latent_embeds.shape
        if B % trajectory_group_size != 0:
            raise ValueError(
                f"Batch size {B} is not divisible by trajectory_group_size={trajectory_group_size}."
            )
        num_groups = B // trajectory_group_size

        # [B, D] — mean-pool over latent steps
        pooled = communicated_latent_embeds.mean(dim=1)

        # L2-normalize: [B, D]
        normed = F.normalize(pooled, p=2, dim=-1)

        # Reshape to [num_groups, N, D]
        normed = normed.view(num_groups, trajectory_group_size, D)

        # Pairwise cosine similarity via batch matrix multiply: [num_groups, N, N]
        sim_matrix = torch.bmm(normed, normed.transpose(1, 2))

        # Mask: upper triangle, excluding diagonal
        N = trajectory_group_size
        upper_mask = torch.triu(torch.ones(N, N, device=sim_matrix.device, dtype=torch.bool), diagonal=1)
        n_pairs = upper_mask.sum().item()
        if n_pairs == 0:
            return sim_matrix.new_zeros(())

        penalty = sim_matrix[:, upper_mask].mean()
        return penalty

