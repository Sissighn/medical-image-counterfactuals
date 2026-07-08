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
| Retrieval-NUN | Case-based nearest unlike baseline | Retrieves real target-class training images; not a minimal edit |
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

## Retrieval-NUN

Retrieval-NUN retrieves the nearest real training image from the manifest target
class in ResNet18 penultimate feature space. Candidate images must have the
target true label and must also be predicted as the target class.

It is useful because it gives an intuitive nearest unlike case. Its limitation
is that it does not isolate the minimal image change needed for the original
sample; differences can reflect anatomy, acquisition, and dataset variation.

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
