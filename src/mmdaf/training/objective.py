"""Joint objective corresponding to the QMF and augmentation losses."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
import torch.nn.functional as F

from mmdaf.models.qmf import QMFOutput


@dataclass
class ObjectiveTerms:
    total: Tensor
    original_classification: Tensor
    augmented_classification: Tensor
    history_regularization: Tensor
    augmentation_ranking: Tensor


def _classification_term(output: QMFOutput, labels: Tensor) -> Tensor:
    loss = F.cross_entropy(output.fused_logits, labels)
    for logits in output.modality_logits.values():
        loss = loss + F.cross_entropy(logits, labels)
    return loss


def combined_training_objective(
    original_output: QMFOutput,
    mixed_output: QMFOutput,
    labels: Tensor,
    history_regularization: Tensor | None = None,
    augmentation_ranking: Tensor | None = None,
    lambda_history: float = 1.0,
    epsilon_ranking: float = 1.0,
) -> ObjectiveTerms:
    """Combine Eqs. (8) and (12) into the joint training objective.

    ``history_regularization`` is the sum of modality-specific QMF historical
    ranking losses. ``augmentation_ranking`` is the Eq. (11) original/degraded
    ranking loss. The caller computes both terms so that history updates and
    mixing masks remain explicit in the training loop.
    """
    original_cls = _classification_term(original_output, labels)
    augmented_cls = _classification_term(mixed_output, labels)
    zero = original_output.fused_logits.new_zeros(())
    history_term = zero if history_regularization is None else history_regularization
    ranking_term = zero if augmentation_ranking is None else augmentation_ranking
    total = (
        original_cls
        + lambda_history * history_term
        + augmented_cls
        + epsilon_ranking * ranking_term
    )
    return ObjectiveTerms(
        total=total,
        original_classification=original_cls,
        augmented_classification=augmented_cls,
        history_regularization=history_term,
        augmentation_ranking=ranking_term,
    )
