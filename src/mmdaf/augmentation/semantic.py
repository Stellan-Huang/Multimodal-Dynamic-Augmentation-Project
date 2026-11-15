"""Feature-space semantic regularization used by adversarial augmentation."""

from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as F


def semantic_feature_distance(original_features: Tensor, augmented_features: Tensor) -> Tensor:
    """Mean-squared distance between penultimate-layer feature representations.

    The paper describes ``sim`` as a negative L2 feature distance. The recovered
    training scripts operationalize the constraint with ``MSELoss``; this
    function follows that implementation convention. During adversarial
    maximization the distance is subtracted from the objective, preserving
    feature-level semantics while the input is perturbed.
    """
    if original_features.shape != augmented_features.shape:
        raise ValueError("original and augmented feature tensors must have equal shapes")
    return F.mse_loss(augmented_features, original_features)
