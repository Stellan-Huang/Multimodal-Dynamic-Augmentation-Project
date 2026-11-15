# Portfolio guide for faculty review

## What a reader should understand

The project is intended to demonstrate the ability to connect a research problem, mathematical objectives, and modular deep-learning code. It should not be presented as a complete benchmark reproduction.

The clearest narrative is:

1. **Problem:** modality quality varies, and a model may not have seen enough quality-diverse training examples.
2. **Method:** generate model-aware low-quality modality counterparts, preserve their feature semantics, and mix them with original modalities.
3. **Learning signal:** train on both original and mixed samples, while keeping the QMF historical ranking term distinct from the augmentation ranking term.
4. **Engineering:** express the method through reusable PyTorch modules that accept named modality tensors.
5. **Limitations:** the examples are synthetic and the original benchmark pipelines are not included.

## Recommended five-minute walkthrough

- Start with `docs/method_overview.md` and explain the problem and the two-stage augmentation idea.
- Open `src/mmdaf/models/qmf.py` to explain energy scores, modality weights, and decision-level logit fusion.
- Open `src/mmdaf/augmentation/adversarial.py` to explain why the input, rather than the model parameters, is optimized during low-quality sample generation.
- Open `src/mmdaf/augmentation/multimodal_mix.py` to explain the `2^M - 1` non-empty degraded subsets.
- Compare `history_ranking.py` with `augmentation_ranking.py` to explain why the two ranking losses serve different roles.
- Finish with the limitations and the actual contribution statement in `docs/contribution_statement_template.md`.

## What not to claim

Do not describe this repository as:

- a complete reproduction of IJCNN 2025;
- a source release of the original historical training project;
- an implementation of the original NYU Depth V2, SUN RGB-D, BRCA, ROSMAP, or CREMA-D data pipelines;
- evidence that the reported benchmark metrics have been reproduced;
- an open-source release under MIT or another license while no license has been granted.

## Suggested short project description

> This repository is a research reference implementation accompanying the IJCNN 2025 paper on multimodal data augmentation for dynamic fusion. It organizes the QMF fusion foundation and the paper's core augmentation components - adversarial uncertainty weighting, semantic similarity regularization, multimodal mixing, and augmentation ranking - into modular PyTorch code. The included examples use synthetic tensors to illustrate the API; the repository is not a full benchmark reproduction package.
