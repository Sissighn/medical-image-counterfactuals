# Final Method Summary

## Purpose

This file summarizes the implemented counterfactual methods and the current
fixed-evaluation results. It is intended as a compact public project summary for
report writing and presentation preparation.

The current project compares three implemented counterfactual directions:

```text
1. Prototype-guided optimization baseline
2. SEDC-T segment replacement
3. DVCE-style diffusion-guided generation
```

The wording is intentionally careful. The implementations are adaptations to a
PyTorch medical-image setup and should not be described as exact reproductions
of the original methods unless the mechanisms are checked one by one.

Several method states are reported because they answer different questions:

| State type | Purpose |
| --- | --- |
| Main method result | primary comparison on fixed evaluation manifests |
| Original-style / method-faithfulness check | shows how close the implementation is to the referenced method mechanism |
| Project variant or tuning ablation | tests whether additional constraints or parameter changes explain failure cases |

The detailed justification is documented in
`results/method_variant_rationale.md` and
`results/method_implementation_audit.md`.

## Baseline Classifiers

The counterfactual methods explain pretrained ResNet18 classifiers trained on
two medical image datasets:

| Dataset | Classes | Best model | Accuracy | Weighted F1 |
| --- | --- | --- | ---: | ---: |
| BUSI | benign, malignant, normal | ResNet18 pretrained | 0.8390 | 0.8365 |
| Pneumonia | NORMAL, PNEUMONIA | ResNet18 pretrained | 0.8782 | 0.8732 |

These models are not intended to be clinically optimal. Their role is to provide
reasonable learned decision functions that can be explained with counterfactual
methods.

## Method 1: Prototype-Guided Optimization Baseline

The prototype-guided optimization baseline computes class-level feature
prototypes from the trained ResNet18 embedding space. It then optimizes an input
image toward a target class while constraining the image change.

This is not presented as a full Alibi CFProto reproduction. It is a
project-specific baseline that borrows the prototype-guidance idea and is used
to provide a high-validity reference point for the other methods.

Recommended wording:

```text
Prototype-guided optimization baseline
```

Fixed-evaluation result:

| Variant | Dataset | Samples | Validity | Mean CF confidence | Mean changed pixel fraction | Mean runtime |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Baseline | BUSI | 15 | 1.00 | 0.9978 | 0.0559 | 5.24s |
| Baseline | Pneumonia | 20 | 1.00 | 0.9928 | 0.1442 | 5.69s |
| Plausibility ablation | BUSI | 15 | 1.00 | 0.9953 | 0.0315 | 4.78s |
| Plausibility ablation | Pneumonia | 20 | 1.00 | 0.9814 | 0.0934 | 4.76s |

Interpretation:

```text
High model validity, but changes can be diffuse and medically hard to localize.
The plausibility ablation reduces changed area without changing the method
core, but the method's best role remains a technical baseline.
```

## Method 2: SEDC-T Segment Replacement

The SEDC-T method segments an image and searches for region replacements that
change the classifier prediction to the target class. The project now reports an
original-style best-first mode for method fidelity and a faster project variant
for comparison.

Recommended wording:

```text
SEDC-T original-style best-first segment replacement
```

Fixed-evaluation result:

| Variant | Dataset | Samples | Validity | Mean CF confidence | Mean changed pixel fraction | Mean runtime |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Original-style best-first | BUSI | 15 | 0.80 | 0.6674 | 0.1517 | 6.59s |
| Original-style best-first | Pneumonia | 20 | 0.55 | 0.7343 | 0.1410 | 13.78s |
| Project variant | BUSI | 15 | 0.80 | 0.6376 | 0.1471 | 0.56s |
| Project variant with lung-field ROI | Pneumonia | 20 | 0.45 | 0.7639 | 0.1510 | 0.39s |

Interpretation:

```text
More localized and visually readable than the prototype-guided baseline, but
less consistently valid. The original-style best-first mode is method-faithful
but slower. The project variant is faster and can use a simple lung-field ROI,
but this ROI must be described as a project-specific constraint rather than an
original SEDC-T mechanism.
```

Tuning ablation:

```text
The best tuned Pneumonia setting reached 0.60 validity, only slightly above the
0.55 original-style result. This suggests that low Pneumonia validity is not
just a parameter issue, but reflects a limitation of segment replacement for
diffuse chest X-ray cues.
```

## Method 3: DVCE-Style Diffusion-Guided Generation

The DVCE-style prototype uses a diffusion-guided generation process together
with the medical ResNet18 classifier adapter. It covers the generative
counterfactual category.

The OpenAI checkpoint result is the current DVCE-style baseline. The Pneumonia
fine-tuned checkpoint is reported as a checkpoint/guidance ablation, not as a
separate fourth method.

Recommended wording:

```text
DVCE-style diffusion-guided feasibility prototype
```

Fixed-evaluation result with the current diffusion checkpoint:

| Dataset | Samples | Validity | Mean CF confidence | Mean changed pixels above threshold | Mean runtime |
| --- | ---: | ---: | ---: | ---: | ---: |
| BUSI | 5 | 1.00 | 0.7034 | 0.3569 | 8.86s |
| Pneumonia | 5 | 0.80 | 0.7219 | 0.1654 | 9.49s |

Additional Pneumonia fine-tuned diffusion checkpoint result:

| Variant | Dataset | Samples | Validity | Mean CF confidence | Mean changed pixels above threshold | Mean runtime |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Medical checkpoint, original settings | Pneumonia | 5 | 0.20 | 0.7673 | 0.1615 | 9.50s |
| Medical checkpoint, stronger guidance | Pneumonia | 5 | 0.40 | 0.7236 | 0.1615 | 9.30s |
| Medical checkpoint, compromise setting | Pneumonia | 5 | 0.80 | 0.6937 | 0.2469 | 15.63s |
| Medical checkpoint, strongest tested setting | Pneumonia | 5 | 1.00 | 0.8527 | 0.4453 | 25.96s |

Interpretation:

```text
Generative and capable of target-class flips, but still limited by the
diffusion prior and guidance settings. The Pneumonia fine-tuned checkpoint can
be integrated successfully, but higher validity requires stronger generation
changes and can still produce noise-like artifacts. Visual plausibility must be
evaluated separately from model validity.
```

## Overall Comparison

| Criterion | Prototype-guided baseline | SEDC-T | DVCE-style |
| --- | --- | --- | --- |
| Validity | strongest | lower, especially on Pneumonia | promising on small fixed subset |
| Locality | weak to moderate | strongest | weak to moderate |
| Runtime | moderate | original-style slower, project variant fastest | slowest |
| Visual interpretability | limited by diffuse changes | strongest | limited by artifacts/domain mismatch |
| Best role | technical baseline | main region-based method | generative feasibility method |

## Main Takeaway

The methods show a trade-off between validity and interpretability.

```text
Prototype-guided optimization is highly valid but less localized.
SEDC-T replacement is localized but not always valid; the original-style run is safer for method fidelity.
DVCE-style generation is generative but currently limited by domain mismatch.
```

The project therefore compares different explanation behaviors rather than
claiming that one method is universally best.
