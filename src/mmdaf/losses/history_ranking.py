"""Historical-loss ranking regularization for QMF (Eqs. 4-7)."""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class SampleLossHistory(nn.Module):
    """Store cumulative per-sample loss sums and observation counts.

    Call ``update`` with detached single-modality cross-entropy losses during
    the historical observation window. The mean values are used as ``kappa``
    in the QMF ranking regularizer. A new instance should be created per
    modality, because the paper tracks modality-specific historical losses.
    """

    def __init__(self, num_samples: int) -> None:
        super().__init__()
        if num_samples < 1:
            raise ValueError("num_samples must be positive")
        self.register_buffer("loss_sum", torch.zeros(num_samples, dtype=torch.float32))
        self.register_buffer("count", torch.zeros(num_samples, dtype=torch.float32))

    @torch.no_grad()
    def update(self, indices: Tensor, per_sample_loss: Tensor) -> None:
        indices = indices.to(device=self.loss_sum.device, dtype=torch.long).reshape(-1)
        values = per_sample_loss.detach().to(self.loss_sum.device).reshape(-1).float()
        if indices.shape != values.shape:
            raise ValueError("indices and per_sample_loss must have the same number of elements")
        if torch.any(indices < 0) or torch.any(indices >= self.loss_sum.numel()):
            raise IndexError("sample index is outside the history buffer")
        self.loss_sum.index_add_(0, indices, values)
        self.count.index_add_(0, indices, torch.ones_like(values))

    def mean_loss(self, indices: Tensor) -> Tensor:
        indices = indices.to(device=self.loss_sum.device, dtype=torch.long).reshape(-1)
        return self.loss_sum[indices] / self.count[indices].clamp_min(1.0)

    def observed(self, indices: Tensor) -> Tensor:
        indices = indices.to(device=self.count.device, dtype=torch.long).reshape(-1)
        return self.count[indices] > 0

    @torch.no_grad()
    def reset(self) -> None:
        self.loss_sum.zero_()
        self.count.zero_()


def _per_sample_weight(weight: Tensor) -> Tensor:
    if weight.ndim == 0:
        raise ValueError("weight must include a batch dimension")
    return weight.reshape(weight.shape[0], -1).mean(dim=1)


def history_ranking_loss(
    confidence: Tensor,
    indices: Tensor,
    history: SampleLossHistory,
    partner_indices: Tensor | None = None,
) -> Tensor:
    """Implement the pairwise hinge regularizer in Eq. (6).

    By default, each item is paired with the next item in the batch (cyclic
    pairing, as in the recovered training scripts). ``partner_indices`` may
    optionally provide explicit sample IDs, which must be unique in the batch.
    Pairs are ignored until both samples have historical loss observations.
    """
    weights = _per_sample_weight(confidence)
    indices = indices.to(device=weights.device, dtype=torch.long).reshape(-1)
    if weights.shape[0] != indices.shape[0]:
        raise ValueError("confidence and indices must have matching batch sizes")
    if weights.numel() < 2:
        return confidence.sum() * 0.0

    if partner_indices is None:
        partner_indices = torch.roll(indices, shifts=-1, dims=0)
        paired_weights = torch.roll(weights, shifts=-1, dims=0)
    else:
        partner_indices = partner_indices.to(device=weights.device, dtype=torch.long).reshape(-1)
        if partner_indices.shape != indices.shape:
            raise ValueError("partner_indices must match indices shape")
        index_values = indices.detach().cpu().tolist()
        if len(set(index_values)) != len(index_values):
            raise ValueError("indices must be unique when custom partner_indices are supplied")
        lookup = {int(sample_id): position for position, sample_id in enumerate(index_values)}
        partner_values = partner_indices.detach().cpu().tolist()
        if any(int(sample_id) not in lookup for sample_id in partner_values):
            raise ValueError("custom partner_indices must refer to samples in the current batch")
        positions = torch.tensor(
            [lookup[int(sample_id)] for sample_id in partner_values],
            device=weights.device,
            dtype=torch.long,
        )
        paired_weights = weights[positions]

    k_i = history.mean_loss(indices).to(weights.device)
    k_j = history.mean_loss(partner_indices).to(weights.device)
    valid = history.observed(indices).to(weights.device) & history.observed(partner_indices).to(weights.device)
    if not torch.any(valid):
        return confidence.sum() * 0.0

    weight_difference = weights - paired_weights
    direction = torch.sign(weight_difference.detach())
    pair_loss = F.relu(direction * (k_i - k_j) + weight_difference)
    return pair_loss[valid].mean()
