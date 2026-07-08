# Method Variant Rationale

This document explains which method states are retained in the project and why.
The old prototype-guided variants have been removed. The only retained
prototype-guided method is the CFProto-nearer prototype-guided optimization
baseline.

## Current Method Roles

| Method family | Retained role | Notes |
| --- | --- | --- |
| CFProto-nearer prototype-guided optimization | Final prototype-guided method | Encoder-kNN prototypes, adaptive c-search, elastic-net selection, polynomial learning-rate decay, targeted margin loss |
| Retrieval-NUN | Case-based nearest unlike baseline | Retrieves real target-class training images; not a minimal edit |
| SEDC-T | Region-based/localized counterfactuals | Original-style best-first plus project/tuning variants |
| DVCE | Generative counterfactual feasibility | OpenAI checkpoint plus Pneumonia fine-tuned checkpoint state |

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

No separate older prototype-guided result is retained as a comparison row.

## Retrieval-NUN

Retrieval-NUN retrieves the nearest real training image from the manifest target
class in ResNet18 penultimate feature space. Candidate images must have the
target true label and must also be predicted as the target class.

It is useful because it gives an intuitive nearest unlike case. Its limitation
is that it does not isolate the minimal image change needed for the original
sample; differences can reflect anatomy, acquisition, and dataset variation.

## SEDC-T

SEDC-T is retained with multiple states because these states answer different
questions:

- original-style best-first: closer to the referenced SEDC-T search mechanism,
- project variant: faster practical implementation,
- tuned variants: controlled parameter checks for segment budget, replacement
  mode, and ROI effects.

The original-style best-first run should be used as the method-faithfulness
reference. ROI and tuned variants must be described as project-specific
constraints or parameter checks.

## DVCE

DVCE-style generation is retained as the generative feasibility method. The
OpenAI diffusion checkpoint and the Pneumonia fine-tuned checkpoint are reported
as checkpoint/guidance states of the same method direction, not as unrelated
methods.

## Reporting Principle

Validity means that the classifier changes to the target class. It does not
imply medical plausibility, clinical causality, or that the highlighted image
change is a human-interpretable disease marker.
