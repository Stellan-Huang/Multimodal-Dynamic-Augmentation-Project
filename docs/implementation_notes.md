# Implementation notes and scope

## What this repository is

A compact research reference implementation intended to make the paper's core algorithmic design inspectable. It is suitable as a code portfolio artifact and as a basis for technical discussion.

## What this repository is not

- It is not a full reproduction package.
- It does not contain NYU Depth V2, SUN RGB-D, BRCA, ROSMAP, or CREMA-D data.
- It does not include dataset-specific data loaders, preprocessing caches, original data splits, trained weights, or historical experiment launch scripts.
- It does not claim to reproduce any table or figure from the paper.

## Implementation choices

### QMF weights and energy temperatures

The energy score is `-tau * logsumexp(logits / tau)`. The modality weight is `alpha * energy + beta`, with negative `alpha` and non-negative `beta`. The implementation accepts a shared temperature or a separate positive temperature for each modality, matching the modality-specific notation in the paper. The default values `alpha=-0.1`, `beta=0`, and `tau=1` follow the common setting visible in the recovered QMF code. Weights are not normalized across modalities because the published formulation uses a weighted sum, not a modality softmax.

### Adversarial input optimization

The AW module maximizes the per-modality classification loss minus the confidence and semantic-distance penalties. Only the modality input is differentiated; model parameters are not updated during generation. Other modalities remain fixed while a single modality is optimized. The default update is an unconstrained gradient-ascent step; an optional clamp can be supplied when the input domain has a known range. The function temporarily evaluates the model and restores each submodule's original train/eval state afterward.

### Semantic regularization

The paper text describes a negative L2 distance between penultimate features. The recovered training scripts use `nn.MSELoss()` between original and adversarial features. This reference implementation uses mean-squared feature distance to follow the recovered implementation pattern and makes that choice explicit. Since MSE includes a reduction over feature elements, its coefficient scale is not identical to an unreduced L2 norm.

### Multimodal mixing

The mixing module samples one of the `2**M - 1` non-empty degraded subsets for every sample. A boolean mask records which modalities were replaced. The labels are not changed.

### Ranking losses

`history_ranking_loss` implements the QMF historical loss/weight relationship from Eqs. (4)-(7). `augmentation_ranking_loss` implements the original-versus-degraded comparison in Eq. (11). They must not be merged into one generic ranking function.

`SampleLossHistory` stores cumulative sums and observation counts. To model the paper's specified sampling window, reset the history before the intended window, update it only during that window, and stop updating once the `T` observations have been collected. The default pair selection in `history_ranking_loss` uses cyclic neighbors in the current batch, following the recovered implementation pattern; it is not an exhaustive all-pairs implementation.

### Joint objective

The training helper adds the original-batch classification objective, the historical QMF regularizer, the mixed-batch classification objective, and the augmentation ranking term. The caller controls when the historical buffers are updated, which allows the sampling window to be explicit in the training loop.

## Historical code review observations

The supplied historical archives contained two complementary code trees:

- `Depth-Red` included RGB-D model definitions and NYU Depth V2 / SUN RGB-D experiment scripts.
- `mmdynamics-main` included a QMF model and BRCA / ROSMAP multi-omics experiment scripts.

The recovered ADA scripts contained recognizable implementations of input-space adversarial maximization, feature-space MSE regularization, modality subset mixing, and confidence ranking. No CREMA-D training implementation was identified in those archives.

The historical scripts and third-party source files are not redistributed in this repository. See `NOTICE.md` and `LICENSE_STATUS.md` for provenance and license status.

## Examples and validation

The examples use random vectors and toy MLP encoders. They illustrate the API and tensor flow only; they are not actual RGB-D images, multi-omics records, or benchmark data. The unit tests check selected tensor operations, ranking behavior, and basic gradient flow. They do not establish benchmark accuracy or reproduce the paper's reported results.

A local smoke check on 2026-11-14 used Python 3.13.5 and PyTorch 2.10.0+cpu: 8 unit tests passed and both synthetic examples completed.
