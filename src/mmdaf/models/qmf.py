"""Quality-aware Multimodal Fusion (QMF).

The module accepts one encoder and one classifier per modality. It estimates
modality uncertainty with the energy score and combines modality logits using
quality weights. Weights are deliberately not softmax-normalized: the paper's
QMF formulation uses a weighted sum of modality predictions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

import torch
from torch import Tensor, nn


Coefficient = float | Mapping[str, float]


def energy_score(logits: Tensor, temperature: float | Tensor = 1.0) -> Tensor:
    """Return the energy-based uncertainty score from Eq. (3).

    Args:
        logits: Classifier logits with shape ``[batch, classes]``.
        temperature: Positive scalar energy temperature ``tau``.

    Returns:
        A tensor of shape ``[batch]`` containing
        ``-tau * logsumexp(logits / tau)``.
    """
    if logits.ndim != 2:
        raise ValueError("logits must have shape [batch, classes]")

    if isinstance(temperature, Tensor):
        if temperature.numel() != 1:
            raise ValueError("temperature must be a scalar")
        tau = temperature.to(device=logits.device, dtype=logits.dtype).reshape(())
        if not bool((torch.isfinite(tau) & (tau > 0)).item()):
            raise ValueError("temperature must be finite and positive")
    else:
        if not math.isfinite(float(temperature)) or float(temperature) <= 0:
            raise ValueError("temperature must be finite and positive")
        tau = logits.new_tensor(float(temperature))

    return -tau * torch.logsumexp(logits / tau, dim=1)


@dataclass
class QMFOutput:
    """Structured outputs required by the fusion and augmentation modules."""

    fused_logits: Tensor
    modality_logits: dict[str, Tensor]
    modality_weights: dict[str, Tensor]
    modality_features: dict[str, Tensor]
    modality_energies: dict[str, Tensor]


class QualityAwareMultimodalFusion(nn.Module):
    """Encode modalities, classify each one, and perform energy-weighted fusion.

    ``alpha``, ``beta``, and ``temperature`` may be scalars or mappings keyed
    by modality name. The paper specifies ``alpha_m < 0``, ``beta_m >= 0``,
    and a positive modality-specific energy temperature ``tau_m``. Scalar
    values are broadcast to every modality for convenience.

    Args:
        encoders: Mapping from modality name to a feature encoder module.
        classifiers: Mapping from modality name to a classifier module.
        temperature: Positive scalar or per-modality energy temperature.
        alpha: Negative coefficient mapping uncertainty to fusion weight.
        beta: Non-negative modality-specific offset.
    """

    def __init__(
        self,
        encoders: Mapping[str, nn.Module],
        classifiers: Mapping[str, nn.Module],
        temperature: Coefficient = 1.0,
        alpha: Coefficient = -0.1,
        beta: Coefficient = 0.0,
    ) -> None:
        super().__init__()
        if not encoders:
            raise ValueError("at least one modality is required")
        if set(encoders) != set(classifiers):
            raise ValueError("encoders and classifiers must have identical keys")

        self.modalities = tuple(encoders.keys())
        self.encoders = nn.ModuleDict(dict(encoders))
        self.classifiers = nn.ModuleDict(dict(classifiers))

        temperature_values = self._expand_coefficient(temperature, "temperature")
        alpha_values = self._expand_coefficient(alpha, "alpha")
        beta_values = self._expand_coefficient(beta, "beta")

        if any(not math.isfinite(value) or value <= 0 for value in temperature_values):
            raise ValueError("each temperature must be finite and positive")
        if any(not math.isfinite(value) or value >= 0 for value in alpha_values):
            raise ValueError("each alpha coefficient must be finite and negative")
        if any(not math.isfinite(value) or value < 0 for value in beta_values):
            raise ValueError("each beta coefficient must be finite and non-negative")

        self.register_buffer(
            "temperatures", torch.tensor(temperature_values, dtype=torch.float32)
        )
        self.register_buffer("alpha", torch.tensor(alpha_values, dtype=torch.float32))
        self.register_buffer("beta", torch.tensor(beta_values, dtype=torch.float32))

    def _expand_coefficient(self, value: Coefficient, name: str) -> list[float]:
        if isinstance(value, Mapping):
            missing = set(self.modalities) - set(value)
            extra = set(value) - set(self.modalities)
            if missing or extra:
                raise ValueError(
                    f"{name} keys must match modalities; missing={sorted(missing)}, "
                    f"extra={sorted(extra)}"
                )
            return [float(value[key]) for key in self.modalities]
        return [float(value)] * len(self.modalities)

    def forward(self, modalities: Mapping[str, Tensor]) -> QMFOutput:
        """Run all modality branches and return fused and per-modality outputs."""
        if set(modalities) != set(self.modalities):
            raise ValueError(
                f"expected modalities {self.modalities}, got {tuple(modalities.keys())}"
            )

        logits_by_modality: dict[str, Tensor] = {}
        features_by_modality: dict[str, Tensor] = {}
        energies: dict[str, Tensor] = {}
        weights: dict[str, Tensor] = {}
        fused_logits: Tensor | None = None

        for index, name in enumerate(self.modalities):
            features = self.encoders[name](modalities[name])
            logits = self.classifiers[name](features)
            energy = energy_score(logits, self.temperatures[index])
            weight = (self.alpha[index] * energy + self.beta[index]).unsqueeze(1)

            features_by_modality[name] = features
            logits_by_modality[name] = logits
            energies[name] = energy
            weights[name] = weight
            weighted_logits = weight * logits
            fused_logits = (
                weighted_logits if fused_logits is None else fused_logits + weighted_logits
            )

        assert fused_logits is not None
        return QMFOutput(
            fused_logits=fused_logits,
            modality_logits=logits_by_modality,
            modality_weights=weights,
            modality_features=features_by_modality,
            modality_energies=energies,
        )
