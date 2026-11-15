import torch
import torch.nn as nn

from mmdaf.augmentation import generate_low_quality_modalities, multimodal_mixing_augmentation
from mmdaf.losses import SampleLossHistory, augmentation_ranking_loss, history_ranking_loss
from mmdaf.models.qmf import QualityAwareMultimodalFusion, energy_score
from mmdaf.training import combined_training_objective


def make_model(temperature=1.0):
    encoders = {
        "a": nn.Sequential(nn.Linear(5, 8), nn.ReLU()),
        "b": nn.Sequential(nn.Linear(3, 8), nn.ReLU()),
    }
    classifiers = {"a": nn.Linear(8, 3), "b": nn.Linear(8, 3)}
    return QualityAwareMultimodalFusion(
        encoders, classifiers, temperature=temperature
    )


def test_energy_score_shape_and_qmf_forward():
    logits = torch.randn(4, 3)
    assert energy_score(logits).shape == (4,)
    model = make_model()
    output = model({"a": torch.randn(4, 5), "b": torch.randn(4, 3)})
    assert output.fused_logits.shape == (4, 3)
    assert output.modality_weights["a"].shape == (4, 1)


def test_qmf_accepts_modality_specific_energy_temperatures():
    model = make_model(temperature={"a": 0.5, "b": 2.0})
    assert torch.allclose(model.temperatures, torch.tensor([0.5, 2.0]))
    output = model({"a": torch.randn(4, 5), "b": torch.randn(4, 3)})
    assert output.modality_energies["a"].shape == (4,)
    assert output.modality_energies["b"].shape == (4,)


def test_mixing_selects_nonempty_subsets():
    original = {"a": torch.zeros(4, 2), "b": torch.zeros(4, 3)}
    augmented = {"a": torch.ones(4, 2), "b": torch.ones(4, 3)}
    mixed = multimodal_mixing_augmentation(
        original, augmented, subset_codes=torch.tensor([1, 2, 3, 1])
    )
    degraded_count = mixed.degraded_mask["a"].long() + mixed.degraded_mask["b"].long()
    assert torch.all(degraded_count >= 1)
    assert torch.equal(mixed.modalities["a"][0], torch.ones(2))
    assert torch.equal(mixed.modalities["b"][0], torch.zeros(3))


def test_augmentation_ranking_is_zero_when_degraded_weight_is_lower():
    original = {"a": torch.tensor([[0.8], [0.6]])}
    mixed = {"a": torch.tensor([[0.4], [0.5]])}
    mask = {"a": torch.tensor([True, True])}
    loss = augmentation_ranking_loss(original, mixed, mask)
    assert torch.allclose(loss, torch.tensor(0.0))


def test_history_ranking_penalizes_incorrect_confidence_order():
    history = SampleLossHistory(2)
    indices = torch.tensor([0, 1])
    history.update(indices, torch.tensor([2.0, 0.0]))

    correct_order = torch.tensor([[0.2], [0.8]], requires_grad=True)
    incorrect_order = torch.tensor([[0.8], [0.2]], requires_grad=True)
    correct_loss = history_ranking_loss(correct_order, indices, history)
    incorrect_loss = history_ranking_loss(incorrect_order, indices, history)

    assert torch.allclose(correct_loss, torch.tensor(0.0))
    assert incorrect_loss > correct_loss
    incorrect_loss.backward()
    assert incorrect_order.grad is not None


def test_history_ranking_uses_observed_samples():
    history = SampleLossHistory(4)
    indices = torch.tensor([0, 1, 2, 3])
    history.update(indices, torch.tensor([0.1, 0.4, 0.2, 0.5]))
    confidence = torch.tensor([[0.8], [0.3], [0.7], [0.2]], requires_grad=True)
    loss = history_ranking_loss(confidence, indices, history)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    loss.backward()
    assert confidence.grad is not None


def test_adversarial_generation_preserves_module_modes_and_parameter_grads():
    torch.manual_seed(3)
    model = make_model()
    model.train()
    model.encoders["a"].eval()  # Deliberately create mixed nested states.
    original_modes = {module: module.training for module in model.modules()}
    original = {"a": torch.randn(4, 5), "b": torch.randn(4, 3)}
    labels = torch.randint(0, 3, (4,))

    augmented = generate_low_quality_modalities(
        model, original, labels, steps=1, step_size=1e-4,
        uncertainty_weight=0.2, semantic_weight=0.2,
    )

    assert set(augmented) == set(original)
    assert all(module.training == state for module, state in original_modes.items())
    assert all(parameter.grad is None for parameter in model.parameters())


def test_adversarial_generation_and_joint_objective_backward():
    torch.manual_seed(7)
    model = make_model()
    original = {"a": torch.randn(4, 5), "b": torch.randn(4, 3)}
    labels = torch.randint(0, 3, (4,))
    original_output = model(original)
    augmented = generate_low_quality_modalities(
        model, original, labels, steps=1, step_size=1e-4,
        uncertainty_weight=0.2, semantic_weight=0.2,
    )
    mixed = multimodal_mixing_augmentation(original, augmented)
    mixed_output = model(mixed.modalities)
    rank = augmentation_ranking_loss(
        original_output.modality_weights, mixed_output.modality_weights, mixed.degraded_mask
    )
    terms = combined_training_objective(
        original_output, mixed_output, labels, augmentation_ranking=rank
    )
    terms.total.backward()
    assert torch.isfinite(terms.total)
