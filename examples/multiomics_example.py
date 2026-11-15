"""Synthetic three-modality example (mRNA, methylation, miRNA)."""

import torch
import torch.nn.functional as F

from examples.common import build_toy_qmf
from mmdaf.augmentation import generate_low_quality_modalities, multimodal_mixing_augmentation
from mmdaf.losses import SampleLossHistory, augmentation_ranking_loss, history_ranking_loss
from mmdaf.training import combined_training_objective


def main() -> None:
    torch.manual_seed(11)
    batch_size, dataset_size, num_classes = 8, 40, 3
    model = build_toy_qmf(
        {"mrna": 64, "methylation": 32, "mirna": 16},
        num_classes=num_classes,
        hidden_dim=24,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    original = {
        "mrna": torch.randn(batch_size, 64),
        "methylation": torch.randn(batch_size, 32),
        "mirna": torch.randn(batch_size, 16),
    }
    labels = torch.randint(0, num_classes, (batch_size,))
    indices = torch.arange(batch_size)

    original_output = model(original)
    low_quality = generate_low_quality_modalities(
        model,
        original,
        labels,
        steps=2,
        step_size=5e-4,
        uncertainty_weight=0.5,
        semantic_weight=0.5,
    )
    mixed = multimodal_mixing_augmentation(original, low_quality)
    mixed_output = model(mixed.modalities)

    histories = {name: SampleLossHistory(dataset_size) for name in model.modalities}
    history_term = original_output.fused_logits.new_zeros(())
    for name in model.modalities:
        history_term = history_term + history_ranking_loss(
            original_output.modality_weights[name], indices, histories[name]
        )
        histories[name].update(
            indices,
            F.cross_entropy(original_output.modality_logits[name], labels, reduction="none"),
        )

    augmentation_term = augmentation_ranking_loss(
        original_output.modality_weights,
        mixed_output.modality_weights,
        mixed.degraded_mask,
    )
    terms = combined_training_objective(
        original_output,
        mixed_output,
        labels,
        history_regularization=history_term,
        augmentation_ranking=augmentation_term,
    )
    optimizer.zero_grad()
    terms.total.backward()
    optimizer.step()

    print("Synthetic three-modality API demonstration")
    print("Fused logits:", tuple(original_output.fused_logits.shape))
    print("Possible non-empty degraded subsets:", 2 ** len(model.modalities) - 1)
    print("Sampled subset codes:", mixed.subset_codes.tolist())
    print("Joint objective:", float(terms.total.detach()))


if __name__ == "__main__":
    main()
