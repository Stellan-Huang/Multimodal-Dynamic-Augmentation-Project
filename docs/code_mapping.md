# Paper-to-code mapping

This table maps the published method to the compact reference implementation and records where corresponding ideas appeared in the supplied historical archives.

| Paper component | Equation(s) | Reference implementation | Recovered historical location |
|---|---:|---|---|
| Decision-level quality-aware fusion | (1) | `src/mmdaf/models/qmf.py` | `Depth-Red/models/QMF.py`; `mmdynamics-main/model.py` (`QMF`) |
| Energy uncertainty and confidence weight | (2)-(3) | `energy_score()` and `QualityAwareMultimodalFusion.forward()` | Both QMF model implementations |
| Historical single-modality loss | (4) | `SampleLossHistory` | `Depth-Red/utils/crl_utils.py` and related training-script history logic |
| Historical weight/loss ordering | (5)-(7) | `history_ranking_loss()` | `rank_loss()` in the ADA training scripts |
| Original QMF classification objective | (8) | `_classification_term()` and `combined_training_objective()` | QMF training loops |
| Adversarial Uncertainty Weight (AW) | (9) | `generate_low_quality_modalities()` | `maximize()` in the RGB-D and multi-omics ADA training scripts |
| Semantic Similarity Regularization (SR) | (9), method text | `semantic_feature_distance()` | `MSELoss` between original and adversarial penultimate features in the ADA scripts |
| Heterogeneous-quality modality mixing (MA) | (10) | `multimodal_mixing_augmentation()` | `model_forward_train()` in the image and multi-omics ADA scripts |
| Augmentation Ranking Regularization (RR) | (11) | `augmentation_ranking_loss()` | `MarginRankingLoss` comparisons between original and augmented modality weights |
| Augmentation classification objective | (12) | `combined_training_objective()` | ADA training loops |

## Important interpretation notes

- QMF is the uncertainty-aware fusion foundation described in the paper's preliminary section; it is not presented here as the paper's new augmentation contribution.
- The paper's augmentation contribution combines AW and SR for single-modal low-quality sample generation with MA and RR for heterogeneous-quality multimodal training.
- The QMF historical ranking regularizer and the augmentation ranking regularizer are separate losses. Their inputs and purposes differ.
- The reference code uses a modality mapping rather than hard-coded RGB/depth or three-omics branches.
- The reference code is a compact re-expression of the method, not a verbatim port of the historical scripts. Dataset-specific data pipelines and benchmark scripts are not included.
