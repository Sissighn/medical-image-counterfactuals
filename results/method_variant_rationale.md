# Method Variant Rationale

This document explains which method states are retained in the project and why.
The old prototype-guided variants have been removed. The retained main
prototype-guided method is the CFProto-nearer encoder feature-map configuration;
the bottleneck256 and bottleneck1024 configurations are retained only as
autoencoder/prototype-space ablations.

## Current Method Roles

| Method family | Retained role | Notes |
| --- | --- | --- |
| CFProto-nearer prototype-guided optimization | Final prototype-guided method plus bottleneck ablations | Encoder-kNN prototypes, adaptive c-search, elastic-net selection, polynomial learning-rate decay, targeted margin loss |
| Goyal et al. 2019 CVE | Instance-based feature-space edit | Greedy cell swaps from a nearest-unlike distractor; sparse localized edits; validity guaranteed by construction |
| SEDC-T | Region-based/localized counterfactuals | Original-style best-first plus Pneumonia lung-field ROI ablation |
| DVCE | Generative counterfactual feasibility | One original-style method family with no-cone and Cone Projection states; diffusion checkpoints are OpenAI, Pneumonia-medical, and BUSI-medical |

## CFProto-Nearer Prototype-Guided Optimization

Use the following wording:

```text
CFProto-nearer prototype-guided optimization baseline
```

The method uses:

- autoencoder encoder-space target-class kNN mean prototypes,
- adaptive binary-style attack-constant search,
- elastic-net selection among valid candidates,
- polynomial learning-rate decay,
- targeted margin-style attack loss.

It is methodically aligned with CFProto, but it is not a full Alibi
`CounterfactualProto` reproduction. FISTA/shrinkage, TrustScore, the original
TensorFlow graph, and the original Alibi k-d-tree machinery are not fully
reproduced.

The encoder feature-map configuration is the main reported prototype-guided
result. The bottleneck256 and bottleneck1024 configurations are kept as
ablations because they test a more compact autoencoder latent representation.
In the current fixed-manifest results they are not improvements: they reduce
validity and increase changed pixel fraction. No older ResNet/class-mean or
Cross-Entropy prototype-guided result is retained as a main comparison row.

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
