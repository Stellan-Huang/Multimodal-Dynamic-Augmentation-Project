# Multimodal Dynamic Augmentation

**A research portfolio and reference implementation for dynamic multimodal learning**

This repository accompanies the paper:

> Haining Li, Shan Huang, and Changqing Zhang, "Multimodal Data Augmentation for Dynamic Fusion," *2025 International Joint Conference on Neural Networks (IJCNN)*, 2025.  
> DOI: [10.1109/IJCNN64981.2025.11228982](https://doi.org/10.1109/IJCNN64981.2025.11228982)

The project presents the method's core algorithmic structure in a compact, modular PyTorch codebase. Its purpose is to help a reader inspect the research design, follow the mapping from equations to implementation, and discuss possible extensions. **It is a reference implementation, not a complete reproduction package.**

## Research problem

In real-world multimodal systems, the quality of each modality can change independently. A fusion model may assign a dynamic weight to each modality, but it can still be insufficiently prepared for quality patterns that are rare or absent in the training set.

The paper addresses this training-data limitation by generating model-aware low-quality modality samples and mixing them with original modalities to create heterogeneous-quality multimodal examples.

## Method: foundation and augmentation contribution

An important distinction when reading the paper is that **QMF is the uncertainty-aware dynamic-fusion foundation**, while the proposed data-augmentation framework consists of the single-modal and cross-modal augmentation components.

| Component | Role in the paper | Reference implementation |
|---|---|---|
| QMF | Quality-aware fusion foundation; energy-based uncertainty and weighted logit fusion | `src/mmdaf/models/qmf.py` |
| AW - Adversarial Uncertainty Weight | Input-space optimization that increases modality classification loss while decreasing its fusion weight | `src/mmdaf/augmentation/adversarial.py` |
| SR - Semantic Similarity Regularization | Feature-space constraint that keeps an adversarial sample close to its original representation | `src/mmdaf/augmentation/semantic.py` |
| MA - Multimodal Mixing Augmentation | Replaces a non-empty subset of original modalities with their low-quality counterparts | `src/mmdaf/augmentation/multimodal_mix.py` |
| RR - Ranking Regularization | Penalizes a degraded modality when its fusion weight exceeds that of its original counterpart | `src/mmdaf/losses/augmentation_ranking.py` |
| QMF historical ranking | Separately regularizes the relationship between historical single-modality loss and fusion weight | `src/mmdaf/losses/history_ranking.py` |
| Joint objective | Combines original and mixed-batch classification terms with their corresponding regularizers | `src/mmdaf/training/objective.py` |

The two ranking losses have different purposes and should not be conflated:

1. **QMF historical ranking (Eqs. 4-7):** uses per-sample historical single-modality loss to regularize the relationship between training difficulty and modality weight.
2. **Augmentation RR (Eq. 11):** directly compares the weight of a degraded modality with the weight of its original counterpart in the same sample.

See [docs/method_overview.md](docs/method_overview.md) for the equations and [docs/code_mapping.md](docs/code_mapping.md) for the paper-to-code map.

## What this repository demonstrates

- A modular PyTorch implementation of energy-based, quality-aware logit fusion.
- Input-space gradient ascent for model-aware low-quality sample generation.
- Feature-space semantic regularization.
- Generalized non-empty modality-subset mixing for two or more modalities.
- Two separate ranking losses and a joint training objective.
- A consistent interface based on named modality tensors rather than dataset-specific branches.

## Quick start

Requirements: Python 3.10+ and PyTorch 2.0+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

python -m examples.rgbd_example
python -m examples.multiomics_example
pytest
```

The examples are intentionally small and synthetic. They demonstrate the API and tensor flow; they do not load NYU Depth V2, SUN RGB-D, BRCA, or ROSMAP.

## Important scope and limitations

- This repository is a compact reference presentation of the core method, not a byte-for-byte restoration of the historical experiment scripts.
- The examples use random vectors and toy MLP encoders. Their names indicate two-modality and three-modality use cases; they are **not** real RGB-D image pipelines or real multi-omics pipelines.
- No dataset adapters, original data, preprocessing caches, trained checkpoints, or historical training launch scripts are included.
- The recovered archives contained RGB-D and multi-omics experiment code. No CREMA-D audiovisual implementation was identified in those archives, so it is not included here.
- The tests check core tensor logic and basic gradient flow. They do not validate benchmark accuracy, reproduce paper tables, or establish robustness on real data.
- Exact reproduction would require the original datasets, splits, preprocessing, hyperparameters, random seeds, checkpoints, and a verified historical software environment.

A local smoke check on 2026-11-14 used Python 3.13.5 and PyTorch 2.10.0+cpu: all 8 unit tests passed, and both synthetic examples completed. This is a code-path check only, not a reproduction result.

## Suggested reading path for faculty

1. [Method overview](docs/method_overview.md) - research motivation and mathematical formulation.
2. [Paper-to-code mapping](docs/code_mapping.md) - where each equation is represented.
3. `src/mmdaf/models/qmf.py` - uncertainty estimation and dynamic fusion.
4. `src/mmdaf/augmentation/adversarial.py` and `semantic.py` - AW and SR.
5. `src/mmdaf/augmentation/multimodal_mix.py` - MA.
6. `src/mmdaf/losses/` and `src/mmdaf/training/objective.py` - the two ranking losses and the joint objective.
7. [Research discussion](docs/research_discussion.md) - technical questions and limitations.

For application use, see [the portfolio guide](docs/portfolio_guide.md) and complete [the contribution statement template](docs/contribution_statement_template.md) with an accurate description of your own role.

## Citation

If you discuss or build on the research method, cite the original paper using [CITATION.cff](CITATION.cff).

## License and provenance

No open-source license is applied at this stage. The historical source archives contain third-party copyright and licensing notices, so this repository does not automatically relicense those materials. See [NOTICE.md](NOTICE.md) and [LICENSE_STATUS.md](LICENSE_STATUS.md).

A public repository without a license is not an OSI-licensed open-source release and does not grant others permission to reuse its code. Confirm authorship, institutional or employer agreements, and third-party provenance before public distribution or before adding a license.
