"""Small synthetic model builder shared by the examples."""

from __future__ import annotations

from collections.abc import Mapping

from torch import nn

from mmdaf.models.qmf import QualityAwareMultimodalFusion


def build_toy_qmf(input_dims: Mapping[str, int], num_classes: int = 4, hidden_dim: int = 32):
    encoders = {
        name: nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
        for name, input_dim in input_dims.items()
    }
    classifiers = {
        name: nn.Linear(hidden_dim, num_classes) for name in input_dims
    }
    return QualityAwareMultimodalFusion(encoders, classifiers)
