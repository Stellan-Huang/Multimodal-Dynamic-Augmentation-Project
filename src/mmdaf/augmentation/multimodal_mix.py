"""Multimodal Mixing Augmentation (MA)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import Tensor


@dataclass
class MixedModalBatch:
    """A mixed batch and a mask identifying which modalities were degraded."""

    modalities: dict[str, Tensor]
    degraded_mask: dict[str, Tensor]
    subset_codes: Tensor


def multimodal_mixing_augmentation(
    original: Mapping[str, Tensor],
    augmented: Mapping[str, Tensor],
    generator: torch.Generator | None = None,
    subset_codes: Tensor | None = None,
) -> MixedModalBatch:
    """Replace a uniformly sampled non-empty subset of modalities per sample.

    With ``M`` modalities there are ``2**M - 1`` allowed non-empty degraded
    subsets. A bit mask selects the augmented version for each modality; all
    unselected modalities remain original. Labels are unchanged and should be
    passed through by the caller.
    """
    names = tuple(original.keys())
    if not names:
        raise ValueError("at least one modality is required")
    if set(names) != set(augmented):
        raise ValueError("original and augmented mappings must have identical keys")

    first = original[names[0]]
    if first.ndim < 1:
        raise ValueError("modality tensors must include a batch dimension")
    batch_size = first.shape[0]
    device = first.device
    num_modalities = len(names)
    if num_modalities > 62:
        raise ValueError("at most 62 modalities are supported by the integer subset code")

    for name in names:
        if original[name].shape != augmented[name].shape:
            raise ValueError(f"shape mismatch for modality {name!r}")
        if original[name].shape[0] != batch_size:
            raise ValueError("all modalities must have the same batch size")
        if original[name].device != device or augmented[name].device != device:
            raise ValueError("all modality tensors must be on the same device")

    if subset_codes is None:
        subset_codes = torch.randint(
            low=1,
            high=2**num_modalities,
            size=(batch_size,),
            device=device,
            generator=generator,
        )
    else:
        subset_codes = subset_codes.to(device=device, dtype=torch.long)
        if subset_codes.shape != (batch_size,):
            raise ValueError("subset_codes must have shape [batch]")
        if torch.any(subset_codes < 1) or torch.any(subset_codes >= 2**num_modalities):
            raise ValueError("each subset code must select a non-empty valid subset")

    mixed: dict[str, Tensor] = {}
    masks: dict[str, Tensor] = {}
    for bit, name in enumerate(names):
        mask = ((subset_codes >> bit) & 1).bool()
        view_shape = (batch_size,) + (1,) * (original[name].ndim - 1)
        mixed[name] = torch.where(
            mask.view(view_shape), augmented[name], original[name]
        )
        masks[name] = mask

    return MixedModalBatch(
        modalities=mixed,
        degraded_mask=masks,
        subset_codes=subset_codes,
    )
