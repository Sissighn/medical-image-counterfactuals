# Method Variant Rationale

This document explains which method states are retained in the project and why.
The earlier feature-map and bottleneck-ablation prototype-guided experiments
have been replaced by a single CFProto (original-style) configuration: a
faithful PyTorch port of alibi's `CounterfactualProto` using a bottleneck-256
autoencoder, with `gamma`/`theta` calibrated per dataset.

## Current Method Roles

| Method family | Retained role | Notes |
| --- | --- | --- |
| CFProto (original-style) prototype-guided optimization | Final prototype-guided method | FISTA + shrinkage-thresholding, untargeted hinge attack loss, encoder-space class prototypes, binary c-search, elastic-net selection |
| Goyal et al. 2019 CVE | Instance-based feature-space edit | Greedy cell swaps from a nearest-unlike distractor; sparse localized edits; validity guaranteed by construction |
| SEDC-T | Region-based/localized counterfactuals | Original-style best-first plus Pneumonia lung-field ROI ablation |
| DVCE | Generative counterfactual feasibility | One original-style method family with no-cone and Cone Projection states; diffusion checkpoints are OpenAI, Pneumonia-medical, and BUSI-medical |

## CFProto (Original-Style Prototype-Guided Optimization)

Use the following wording:

```text
CFProto original-style prototype-guided counterfactuals
```

The method follows alibi's `CounterfactualProto` faithfully:

- FISTA optimization with shrinkage-thresholding and Nesterov momentum,
- an untargeted hinge attack loss on the original class,
- a sum-based loss `c*L_attack + L2 + beta*L1 + gamma*L_AE + theta*L_proto`,
- binary search over the attack constant `c` (x10 escalation),
- encoder-space class prototypes from the classifier's own predictions on the
  training split (kNN mean),
- elastic-net (L2 + beta*L1) best-counterfactual selection.

Deliberate differences from the original are only the framework (PyTorch
instead of the TensorFlow 1.x graph) and per-dataset/autoencoder recalibrated
`gamma`/`theta` weights: since all loss terms are sums, their raw magnitude
depends on the input and latent dimensionality, so the original MNIST-example
values do not transfer. Not reproduced: the TensorFlow graph itself, black-box
mode with numerical gradients, categorical variables/k-d-tree prototypes, and
TrustScore filtering (disabled by default in alibi too). Full documentation:
`results/final_configs/cfproto_encoder_method_documentation.md`.

This bottleneck-256 configuration replaced the earlier feature-map,
bottleneck-1024, and ResNet/class-mean prototype experiments, which are no
longer retained as separate comparison rows.

## Goyal et al. 2019 CVE

This method follows Goyal et al. (ICML 2019, arXiv:1904.07451). The ResNet18 is
split into a spatial extractor (`layer4`, 7x7x512 cells) and a decision head
(GAP + FC). A distractor image from the target class is retrieved as the nearest
correctly classified training image in pooled feature space, then a greedy
exhaustive search swaps query feature cells for distractor cells (each cell used
at most once) until the prediction flips to the target class.

It replaces the former retrieval-NUN baseline, which only retrieved the nearest
unlike neighbor without editing the query and was not based on an original
published method. Goyal et al. is image-native, instance-based, and has a
faithful reference implementation (the baseline in the Meta repo
`facebookresearch/visual-counterfactuals`). It reuses the same feature-space
retrieval as the distractor-selection step.

Reporting note: validity is 1.00 by construction (full 49-cell budget), so the
reported quantity of interest is the number of edited cells (sparsity). Mean CF
confidence is near 0.5 because the search stops at the first flip. The edits are
grounded in real target-class content but limited to a coarse 7x7 cell grid.

## SEDC-T

SEDC-T is retained with two states because these states answer different
questions:

- original-style best-first: closer to the referenced SEDC-T search mechanism,
- Pneumonia lung-field ROI ablation: same best-first/Quickshift/Gaussian-blur
  mechanism, but candidate segments are restricted to a simple geometric
  lung-field mask.

The original-style best-first run should be used as the method-faithfulness
reference. The ROI result must be described as a project-specific Pneumonia
ablation, not as part of original SEDC-T and not as a medical lung
segmentation.

## DVCE

DVCE generation is retained as the generative feasibility method. The older
free-guidance prototype variants have been removed. The retained implementation
uses the original-code-nearer `src/dvce_core.py` path: `p_sample`,
`pred_xstart` guidance, separate classifier and LP-distance gradients,
`enforce_same_norms=True`, and `clip_denoised=False`.

There is one DVCE original-style method family:

```text
DVCE original-style medical generation
```

The retained DVCE states are:

- without Cone Projection, mainly as original-style baseline,
- with Cone Projection using a PGD-robust second medical ResNet18 classifier.

The diffusion checkpoint states are:

- OpenAI 256x256 unconditional checkpoint,
- Pneumonia fine-tuned medical checkpoint,
- BUSI fine-tuned medical checkpoint once available.

These states should not be reported with old free-guidance numbers. They need
fresh fixed-manifest runs with `src/dvce_core.py`.

## Reporting Principle

Validity means that the classifier changes to the target class. It does not
imply medical plausibility, clinical causality, or that the highlighted image
change is a human-interpretable disease marker.
