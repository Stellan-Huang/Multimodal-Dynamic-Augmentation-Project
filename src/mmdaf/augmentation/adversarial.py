"""Adversarial Uncertainty Weight (AW) and semantic regularization (SR)."""

from __future__ import annotations

import math
from typing import Mapping

import torch
from torch import Tensor
import torch.nn.functional as F

from mmdaf.augmentation.semantic import semantic_feature_distance
from mmdaf.models.qmf import QualityAwareMultimodalFusion


def generate_low_quality_modalities(
    model: QualityAwareMultimodalFusion,
    modalities: Mapping[str, Tensor],
    labels: Tensor,
    steps: int = 3,
    step_size: float = 1e-3,
    uncertainty_weight: float = 1.0,
    semantic_weight: float = 1.0,
    clamp: tuple[float, float] | None = None,
) -> dict[str, Tensor]:
    """Generate one adversarial low-quality counterpart for each modality.

    For modality ``m``, gradient ascent updates the input using

        CE(y, f_m(xhat_m)) - mu * w_m(xhat_m) - gamma * d(h_m(xhat_m), h_m(x_m)).

    The distance term is subtracted because the paper defines semantic
    similarity as a negative feature distance. Other modalities remain fixed
    while one modality is optimized. Model parameters are not updated by this
    function; gradients are requested only with respect to the current input.

    ``clamp`` is optional because RGB-D and tabular pipelines may use different
    normalization ranges. Supply it only when the input domain has a known
    valid interval.
    """
    if not isinstance(steps, int) or steps < 1:
        raise ValueError("steps must be a positive integer")
    if not math.isfinite(step_size) or step_size <= 0:
        raise ValueError("step_size must be finite and positive")
    if not math.isfinite(uncertainty_weight) or uncertainty_weight < 0:
        raise ValueError("uncertainty_weight must be finite and non-negative")
    if not math.isfinite(semantic_weight) or semantic_weight < 0:
        raise ValueError("semantic_weight must be finite and non-negative")
    if clamp is not None:
        if len(clamp) != 2 or not all(math.isfinite(value) for value in clamp):
            raise ValueError("clamp must contain two finite bounds")
        if clamp[0] > clamp[1]:
            raise ValueError("clamp lower bound must not exceed upper bound")
    if set(modalities) != set(model.modalities):
        raise ValueError("input modalities must match the model modalities")

    # Preserve mixed train/eval states in nested modules, not only the root flag.
    training_states = [(module, module.training) for module in model.modules()]
    model.eval()
    try:
        with torch.no_grad():
            reference_output = model(modalities)
            reference_features = {
                name: feature.detach()
                for name, feature in reference_output.modality_features.items()
            }

        augmented: dict[str, Tensor] = {}
        for name in model.modalities:
            candidate = modalities[name].detach().clone()
            for _ in range(steps):
                with torch.enable_grad():
                    candidate = candidate.detach().requires_grad_(True)
                    current = dict(modalities)
                    current[name] = candidate
                    output = model(current)
                    classification = F.cross_entropy(output.modality_logits[name], labels)
                    confidence = output.modality_weights[name].mean()
                    semantic_distance = semantic_feature_distance(
                        reference_features[name], output.modality_features[name]
                    )
                    objective = (
                        classification
                        - uncertainty_weight * confidence
                        - semantic_weight * semantic_distance
                    )
                    input_gradient = torch.autograd.grad(objective, candidate)[0]

                with torch.no_grad():
                    candidate = candidate + step_size * input_gradient
                    if clamp is not None:
                        candidate = candidate.clamp(min=clamp[0], max=clamp[1])
            augmented[name] = candidate.detach()
        return augmented
    finally:
        for module, was_training in training_states:
            module.training = was_training
