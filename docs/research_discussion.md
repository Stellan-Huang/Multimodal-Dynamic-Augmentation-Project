# Research discussion: questions a faculty member may ask

This document is a preparation aid for explaining the method. Answers should be adapted to the actual role and contributions of each author.

## 1. What problem does the paper address?

Dynamic fusion models estimate how much each modality should contribute, but the training data may not contain enough examples with varying modality quality. The model can therefore struggle when a modality is degraded at test time. The proposed augmentation method expands the training scenarios with model-aware low-quality modalities and heterogeneous-quality combinations.

## 2. What is the distinction between QMF and the paper's contribution?

QMF is the uncertainty-aware dynamic-fusion foundation introduced in the paper's preliminary section. The paper's augmentation contribution is the combination of single-modal low-quality generation - Adversarial Uncertainty Weight (AW) and Semantic Similarity Regularization (SR) - with Multimodal Mixing Augmentation (MA) and Ranking Regularization (RR). A careful explanation should not claim QMF itself as the new contribution of this paper.

## 3. Why use an energy score?

The energy score is a log-sum-exp function of the classifier logits. It provides a scalar signal related to how strongly the classifier supports its output distribution. The paper maps uncertainty to fusion weight with a negative coefficient, so higher uncertainty corresponds to a lower modality weight. It is an uncertainty proxy, not a guarantee of calibrated probability.

## 4. Why use adversarial augmentation instead of only Gaussian noise?

A fixed corruption process encodes a limited set of assumptions about how quality may degrade. The adversarial step is model-aware: it searches in input space for perturbations that increase the modality classification loss and reduce its fusion weight. This can expose failure directions that are not captured by a manually selected noise family.

## 5. Why is semantic regularization needed?

Without a semantic constraint, the adversarial input could move too far from the original sample and become an arbitrary or label-inconsistent example. The feature-distance penalty keeps the perturbed representation near the original penultimate-layer representation while the classification and confidence terms create the desired quality challenge.

## 6. Why mix modalities after generating low-quality samples?

Real scenarios often contain heterogeneous quality: one modality can be degraded while another remains useful. Replacing a non-empty subset of modalities creates these combinations. With `M` modalities, there are `2^M - 1` non-empty subsets, so the strategy generalizes naturally from two modalities to three or more.

## 7. What is the difference between the two ranking losses?

The QMF historical ranking term uses each sample's average historical single-modality classification loss and regularizes its relationship with confidence. The augmentation ranking term compares a clean modality with its degraded counterpart in the same sample. The first is trajectory-based; the second is a direct clean/degraded pair constraint.

## 8. Why include per-modality classification losses as well as fused classification loss?

The fused loss trains the final multimodal decision. Per-modality losses keep each branch predictive and provide meaningful gradients and logits for modality-level uncertainty estimation. The paper's objective includes both.

## 9. What does this repository demonstrate, and what does it not demonstrate?

It demonstrates a modular PyTorch expression of the core tensor operations and their relationships: QMF fusion, input-space adversarial optimization, feature regularization, subset mixing, ranking losses, and a joint objective. The included examples use synthetic vectors and toy encoders. They do not demonstrate real dataset pipelines, benchmark performance, or exact reproduction of the paper's experiments.

## 10. How would the method extend to more modalities?

The mixing strategy scales through a bit mask over modality names. The core model, adversarial generation, and ranking functions operate on mappings, so adding a modality requires supplying its encoder, classifier, and input tensor rather than rewriting the algorithm for a fixed number of views.

## 11. What are the limitations?

The adversarial generator is model- and objective-dependent, and it does not guarantee coverage of every real-world corruption. The feature constraint depends on the selected representation and coefficient. The quality-to-weight mapping also depends on the underlying uncertainty estimator. The current portfolio implementation is not a benchmark reproduction package and does not contain the original datasets or checkpoints.

## 12. How should the project be described accurately?

Describe it as a research reference implementation and portfolio companion that organizes the core algorithmic ideas. Distinguish the QMF foundation from the paper's augmentation contribution. Do not claim exact reproduction of the original experimental results unless the complete data and protocol have been independently restored and verified. Describe your own contribution separately from the author team's contribution.
