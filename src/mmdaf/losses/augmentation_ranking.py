"""Ranking regularization for original-versus-degraded modality pairs (Eq. 11)."""

from __future__ import annotations

from typing import Mapping

import torch
from torch import Tensor
import torch.nn.functional as F


def _per_sample_weight(weight: Tensor) -> Tensor:
    if weight.ndim == 0:
        raise ValueError("weight must include a batch dimension")
    return weight.reshape(weight.shape[0], -1).mean(dim=1)


def augmentation_ranking_loss(
    original_weights: Mapping[str, Tensor],
    mixed_weights: Mapping[str, Tensor],
    degraded_mask: Mapping[str, Tensor],
) -> Tensor:
    """Penalize a degraded modality when its weight exceeds its original weight.

    For each sample, the hinge penalties are averaged only over modalities that
    were replaced by low-quality counterparts. The sample losses are then
    averaged over the batch, matching the ``1 / n_i`` normalization in Eq. (11).
    """
    names = tuple(original_weights.keys())
    if not names or set(names) != set(mixed_weights) or set(names) != set(degraded_mask):
        raise ValueError("weight mappings and degraded_mask must have identical non-empty keys")

    batch_size = _per_sample_weight(original_weights[names[0]]).shape[0]
    reference = original_weights[names[0]]
    penalty_sum = reference.new_zeros(batch_size)
    degraded_count = reference.new_zeros(batch_size)

    for name in names:
        original = _per_sample_weight(original_weights[name])
        mixed = _per_sample_weight(mixed_weights[name])
        mask = degraded_mask[name].to(device=original.device, dtype=torch.bool)
        if original.shape[0] != batch_size or mixed.shape != original.shape or mask.shape != (batch_size,):
            raise ValueError(f"inconsistent batch shapes for modality {name!r}")
        penalty_sum = penalty_sum + F.relu(mixed - original) * mask.to(original.dtype)
        degraded_count = degraded_count + mask.to(original.dtype)

    valid = degraded_count > 0
    if not torch.any(valid):
        return reference.sum() * 0.0
    per_sample = penalty_sum / degraded_count.clamp_min(1.0)
    return per_sample[valid].mean()
