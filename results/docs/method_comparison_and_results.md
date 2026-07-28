# Method Comparison and Results

This document is the central, manually maintained summary of the seminar
project. It defines the four counterfactual methods under comparison, reports
their quantitative results, interprets the findings, and records which method
variants were retained or superseded.

It consolidates the former `final_method_summary.md`, `method_comparison.md`,
and `method_variant_rationale.md` documents.

Related documentation:

- [Fixed evaluation summary](fixed_evaluation_summary.md): canonical,
  automatically generated metrics table produced from the method
  `metadata.json` files by
  [`summarize_counterfactual_evaluation.py`](../../scripts/summarize_counterfactual_evaluation.py).
- [Method fidelity comparison](method_fidelity_comparison.md): detailed audit
  of each implementation against its reference method.
- [Final configurations](../final_configs/): method documentation and
  reproducible run commands.

> **Reporting principle:** Validity means only that the classifier predicts the
> requested target class. It does not establish medical plausibility, clinical
> causality, or that a highlighted image change is a human-interpretable disease
> marker.

## 1. Methods Compared

The project compares four counterfactual approaches for medical image
classification:

1. **CFProto (original-style):** optimization-based and prototype-guided.
2. **Goyal et al. (2019) CVE:** instance-based feature-cell replacement.
3. **SEDC-T:** region-based segment replacement.
4. **DVCE (original-style):** generative, diffusion-guided counterfactuals.

All methods use the same fixed evaluation manifests: 15 BUSI samples and 20
Pneumonia samples, with identical inputs and target classes across methods.
DVCE is more expensive per sample because it requires diffusion sampling, but
it is also evaluated on the complete manifests using the implementation closest
to the original code. For the non-robust ResNet18, Cone Projection is the
original-faithful DVCE variant.

## 2. Baseline Classifiers

| Dataset | Classes | Model | Accuracy | Weighted F1 |
| --- | --- | --- | ---: | ---: |
| BUSI | benign, malignant, normal | Pretrained ResNet18 | 0.8390 | 0.8365 |
| Pneumonia | NORMAL, PNEUMONIA | Pretrained ResNet18 | 0.8782 | 0.8732 |

## 3. Method 1: CFProto

The original-style CFProto implementation optimizes the image in pixel space
and closely follows Alibi's `CounterfactualProto`:

- FISTA optimization with shrinkage-thresholding and Nesterov momentum;
- a hinge attack loss that suppresses the original class relative to the
  strongest alternative class;
- the sum-based objective
  `c·L_attack + L2 + beta·L1 + gamma·L_AE + theta·L_proto`;
- binary search over the attack constant `c`, including tenfold escalation;
- encoder-space class prototypes derived from the classifier's own predictions
  on the training split using the mean of the k nearest neighbors; and
- Elastic Net distance (`L2 + beta·L1`) to select the best counterfactual.

The intentional differences are the framework (PyTorch instead of the original
TensorFlow 1.x graph) and dataset-/autoencoder-specific recalibration of
`gamma` and `theta`. Because all objective terms are sums, their raw magnitude
depends on the input and latent dimensions; the original MNIST example values
do not transfer directly. The implementation does not reproduce the TensorFlow
graph itself, black-box numerical gradients, categorical variables, k-d-tree
prototypes, or TrustScore filtering, which is disabled by default in Alibi.
See the complete
[implementation-to-reference comparison](../final_configs/cfproto_encoder_method_documentation.md).

| Dataset | Samples | Validity | Mean CF confidence | Mean changed-pixel fraction | Mean runtime |
| --- | ---: | ---: | ---: | ---: | ---: |
| BUSI | 15 | 0.87 | 0.6815 | 0.0529 | 46.10 s |
| Pneumonia | 20 | 1.00 | 0.5740 | 0.0180 | 46.34 s |

**Interpretation.** The two BUSI failures out of 15 are an expected consequence
of the untargeted hinge attack loss: optimization found a confident transition
away from the original class, but not into the target class fixed by the
manifest. BUSI uses `theta=0.5` and reaches 0.87 validity; Pneumonia uses
`theta=0.05` and reaches 1.00. This configuration supersedes the earlier
feature-map, bottleneck-1024, and ResNet/class-mean prototype experiments, which
are no longer reported as separate comparison rows.

## 4. Method 2: Goyal et al. (2019) Counterfactual Visual Explanations

This is an instance-based feature-space editing method based on Goyal et al.
(ICML 2019, arXiv:1904.07451). ResNet18 is divided into a spatial extractor
(`layer4`, with 7×7×512 cells) and a decision head (global average pooling and a
fully connected layer). The method retrieves the nearest correctly classified
training image from the target class in pooled feature space. It then greedily
replaces spatial cells in the query feature map with distractor cells, using
each cell at most once, until the prediction changes to the target class. The
reference implementation is the Goyal baseline in
[`facebookresearch/visual-counterfactuals`](https://github.com/facebookresearch/visual-counterfactuals).

| Dataset | Samples | Validity | Mean CF confidence | Mean edits (of 49) | Mean changed-pixel fraction | Mean runtime |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BUSI | 15 | 1.00 | 0.5279 | 14.0 | 0.2596 | 0.25 s |
| Pneumonia | 20 | 1.00 | 0.5231 | 16.15 | 0.3072 | 0.17 s |

**Interpretation.** Validity is 1.00 by construction: with the full 49-cell
budget, the pooled representation converges to that of the distractor, which
guarantees a prediction change. The informative measure is therefore the
number of edited cells: a mean of 14.0 for BUSI and 16.15 for Pneumonia, out of
49. Mean counterfactual confidence remains close to 0.5 because the greedy
search stops at the first class change, prioritizing sparsity over margin. The
edits are grounded in a real target-class image and localized to a coarse 7×7
grid. This method supersedes the earlier retrieval-only nearest-unlike-neighbor
baseline, which returned a neighbor without editing the query and was not based
on a published counterfactual method. See the
[method documentation](../final_configs/goyal_cve_method_documentation.md).

## 5. Method 3: SEDC-T-Style Segment Replacement

SEDC-T modifies image segments and queries the classifier for a target-class
change. The original-style best-first run is the method-fidelity reference. Two
variants are retained because they answer different questions:

- **Original-style best-first:** follows the referenced SEDC-T search
  mechanism.
- **Pneumonia lung-field ROI ablation:** uses the same best-first search,
  Quickshift segmentation, and Gaussian-blur replacement, but restricts
  candidate segments to a simple geometric lung-field mask.

| Variant | Dataset | Samples | Validity | Mean CF confidence | Mean changed-pixel fraction | Mean runtime |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Original-style best-first | BUSI | 15 | 0.80 | 0.6343 | 0.2640 | 6.71 s |
| Original-style best-first | Pneumonia | 20 | 0.55 | 0.6759 | 0.3270 | 13.92 s |
| Lung-field ROI ablation | Pneumonia | 20 | 0.50 | 0.7770 | 0.1745 | 15.23 s |

**Interpretation.** SEDC-T produces localized, segment-level changes that are
often straightforward to discuss visually. Validity is lower, particularly for
Pneumonia, where diffuse model evidence makes segment replacement more
difficult. The original-style best-first run should be used as the
method-fidelity reference. The ROI result must be described explicitly as a
project-specific Pneumonia ablation, not as part of the original SEDC-T method
or as a medical lung segmentation. See the
[method documentation](../final_configs/sedc_t_method_documentation.md).

## 6. Method 4: DVCE Diffusion-Guided Generation

DVCE represents the generative approach and uses the implementation closest to
the original code in [`src/dvce_core.py`](../../src/dvce_core.py):

- `gen_type=p_sample`;
- `timestep_respacing=200` and `skip_timesteps=100`;
- `classifier_lambda=0.1`, `lp_custom=1.0`, and `lp_custom_value=0.15`;
- `enforce_same_norms=True` and `clip_denoised=False`; and
- Cone Projection through `--second_model_path` with a PGD-robust ResNet18,
  `--deg_cone_projection 30`, and `--aug_num 16`.

The evaluation varies two independent factors:

1. **Guidance:** Cone Projection is the original-faithful variant for the
   non-robust ResNet18 under explanation. The no-cone configuration is retained
   only as an explicit ablation; in the original work, no-cone guidance is
   defined for robust classifiers.
2. **Diffusion checkpoint:** the original OpenAI unconditional 256×256 backbone
   and medical EMA checkpoints fine-tuned separately on Pneumonia and BUSI.

| Variant | Checkpoint | Dataset | n | Validity | Mean CF confidence | Mean absolute difference | Changed pixels (>0.05) |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Cone (original-faithful) | OpenAI | BUSI | 15 | 0.93 | 0.944 | 0.024 | 0.116 |
| Cone (original-faithful) | OpenAI | Pneumonia | 20 | 0.80 | 0.837 | 0.017 | 0.051 |
| Cone | BUSI fine-tuned | BUSI | 15 | 1.00 | 0.998 | 0.028 | 0.156 |
| Cone | Pneumonia fine-tuned | Pneumonia | 20 | 1.00 | 0.980 | 0.019 | 0.067 |
| No cone (ablation) | BUSI fine-tuned | BUSI | 15 | 1.00 | 0.998 | 0.026 | 0.136 |
| No cone (ablation) | Pneumonia fine-tuned | Pneumonia | 20 | 1.00 | 0.995 | 0.017 | 0.052 |
| No cone (ablation) | OpenAI | Pneumonia | 20 | 1.00 | 0.997 | 0.018 | 0.060 |

**Interpretation.** The core follows the original `dff_attack.py`: `p_sample`,
classifier and distance guidance on `pred_xstart`, an unclamped `_map_img`,
epsilon-norm rebalancing when `enforce_same_norms=True`, and Cone Projection
that projects the robust PGD classifier's gradient onto a cone around the
explained classifier's gradient. The fine-tuned checkpoints achieve full
validity, while the generic OpenAI checkpoint reaches 0.93 on BUSI and 0.80 on
Pneumonia. This difference is expected because the generic checkpoint was
trained on natural images rather than medical scans. The OpenAI runtimes of
700–1,173 seconds reflect a CPU-bound machine and are not comparable across
hardware; the fine-tuned runs of approximately 33–45 seconds are more
representative. See the [DVCE method documentation](../final_configs/dvce_method_documentation.md)
and [Cone Projection notes](../final_configs/dvce_cone_projection.md).

## 7. Summary Comparison

| Method | Dataset | Samples | Validity | Mean CF confidence | Mean change | Mean runtime |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| CFProto (original-style) | BUSI | 15 | 0.87 | 0.6815 | 0.0529 pixel fraction | 46.10 s |
| CFProto (original-style) | Pneumonia | 20 | 1.00 | 0.5740 | 0.0180 pixel fraction | 46.34 s |
| Goyal et al. (2019) CVE | BUSI | 15 | 1.00 | 0.5279 | 0.2596 pixel fraction, 14.0 edits | 0.25 s |
| Goyal et al. (2019) CVE | Pneumonia | 20 | 1.00 | 0.5231 | 0.3072 pixel fraction, 16.15 edits | 0.17 s |
| SEDC-T original-style best-first | BUSI | 15 | 0.80 | 0.6343 | 0.2640 pixel fraction | 6.71 s |
| SEDC-T original-style best-first | Pneumonia | 20 | 0.55 | 0.6759 | 0.3270 pixel fraction | 13.92 s |
| SEDC-T lung-field ROI ablation | Pneumonia | 20 | 0.50 | 0.7770 | 0.1745 pixel fraction | 15.23 s |
| DVCE Cone (OpenAI, original-faithful) | BUSI | 15 | 0.93 | 0.944 | 0.116 pixel fraction | 1,173.4 s |
| DVCE Cone (OpenAI, original-faithful) | Pneumonia | 20 | 0.80 | 0.837 | 0.051 pixel fraction | 700.2 s |
| DVCE Cone (fine-tuned) | BUSI | 15 | 1.00 | 0.998 | 0.156 pixel fraction | 44.6 s |
| DVCE Cone (fine-tuned) | Pneumonia | 20 | 1.00 | 0.980 | 0.067 pixel fraction | 44.9 s |

The complete automatically generated table, including every no-cone row and
its metadata path, is available in the
[fixed evaluation summary](fixed_evaluation_summary.md).

## 8. Retained and Superseded Variants

| Method family | Retained role | Notes |
| --- | --- | --- |
| CFProto (original-style) | Final prototype-guided method | FISTA with shrinkage-thresholding, hinge attack loss, encoder-space prototypes, binary search over `c`, and Elastic Net selection. Supersedes feature-map, bottleneck-1024, and class-mean experiments. |
| Goyal et al. (2019) CVE | Instance-based feature-space editing | Greedy cell swaps from a nearest-unlike distractor. Supersedes the earlier retrieval-only NUN baseline. |
| SEDC-T | Region-based, localized counterfactuals | Original-style best-first search plus the Pneumonia lung-field ROI ablation. |
| DVCE | Generative, diffusion-guided counterfactuals | Cone Projection, which is original-faithful for a non-robust ResNet18, plus a no-cone ablation; evaluated with OpenAI, Pneumonia, and BUSI checkpoints. |

## 9. Recommended Terminology

- **CFProto:** “CFProto original-style prototype-guided counterfactuals.”
- **Goyal:** “Goyal et al. (2019) Counterfactual Visual Explanations.”
- **SEDC-T:** use “original-style best-first” for the fidelity reference and
  label the ROI result as a “project-specific Pneumonia ablation.”
- **DVCE:** use Cone Projection as the primary original-faithful variant, label
  every no-cone result as an **ablation**, and state the checkpoint separately.

## 10. Main Finding

The methods expose different trade-offs:

- **CFProto** produces compact counterfactuals and is usually model-valid, with
  small, plausibility-regularized changes.
- **Goyal et al. CVE** produces sparse, localized edits grounded in real
  target-class images. Validity is guaranteed by construction, and confidence
  remains close to the decision boundary.
- **SEDC-T** is localized but less consistently valid.
- **DVCE** is generative and reaches full validity with the medically
  fine-tuned checkpoints, while the generic OpenAI checkpoint reaches
  0.80–0.93.

Model validity must not be conflated with medical plausibility; the latter
requires separate evaluation.
