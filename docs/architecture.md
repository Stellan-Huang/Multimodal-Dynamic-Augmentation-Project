# Architecture and module responsibilities

## Design principle

The repository is organized by research responsibility rather than by dataset-specific training script. A modality is represented by a named tensor in a mapping, so the same components can handle two, three, or more modalities.

## Core modules

| Path | Responsibility | Research role |
|---|---|---|
| `models/qmf.py` | Modality encoders, classifiers, energy scores, weighted logit fusion | QMF foundation, Eqs. (1)-(3) |
| `augmentation/adversarial.py` | Input-space gradient ascent for each modality | AW and the optimization step used with SR |
| `augmentation/semantic.py` | Feature-space semantic distance | SR |
| `augmentation/multimodal_mix.py` | Non-empty subset replacement | MA, Eq. (10) |
| `losses/history_ranking.py` | Per-sample loss history and pairwise ranking | QMF regularizer, Eqs. (4)-(7) |
| `losses/augmentation_ranking.py` | Original/degraded confidence hinge loss | RR, Eq. (11) |
| `training/objective.py` | Joint original and augmented classification objective | Eqs. (8) and (12) |

## Forward data flow

1. Each modality is encoded and classified independently.
2. The energy score is calculated from each modality's logits.
3. The energy score is mapped to a modality weight using its alpha/beta coefficients.
4. Weighted modality logits are summed to obtain the fused prediction.
5. The AW module optimizes each modality input while keeping the other inputs fixed; SR penalizes feature drift.
6. The MA module selects a non-empty subset of modalities to replace with their low-quality versions.
7. The two ranking losses are computed for their separate purposes.
8. The joint objective combines original-batch and mixed-batch classification terms.

## Extending to another modality

Provide an encoder and classifier under the same modality key, then pass the corresponding input tensor in the input mapping. The mixing and ranking modules infer the modality count from the mapping; no dataset-specific branch is needed in the core algorithm.

The included examples use toy MLP encoders on synthetic vectors. A real RGB-D or multi-omics application still needs appropriate modality-specific encoders, preprocessing, data loaders, and dataset splits.
