# Final Method Summary

## Purpose

This file summarizes the implemented counterfactual methods and the current
fixed-evaluation results. It is intended as a compact public project summary for
report writing and presentation preparation.

The current project compares three implemented counterfactual directions:

```text
1. Prototype-guided optimization baseline
2. SEDC-T-style segment replacement
3. DVCE-style diffusion-guided generation
```

The wording is intentionally careful. The implementations are adaptations to a
PyTorch medical-image setup and should not be described as exact reproductions
of the original methods unless the mechanisms are checked one by one.

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

Recommended wording:

```text
Prototype-guided optimization baseline
```

Fixed-evaluation result:

| Dataset | Samples | Validity | Mean CF confidence | Mean changed pixel fraction | Mean runtime |
| --- | ---: | ---: | ---: | ---: | ---: |
| BUSI | 15 | 1.00 | 0.9978 | 0.0559 | 5.24s |
| Pneumonia | 20 | 1.00 | 0.9928 | 0.1442 | 5.69s |

Interpretation:

```text
High model validity, but changes can be diffuse and medically hard to localize.
Best role: technical baseline.
```

## Method 2: SEDC-T-Style Segment Replacement

The SEDC-T-style method segments an image and searches for region replacements
that change the classifier prediction to the target class. It is more localized
than direct pixel optimization.

Recommended wording:

```text
SEDC-T-style targeted segment replacement
```

Fixed-evaluation result:

| Dataset | Samples | Validity | Mean CF confidence | Mean changed pixel fraction | Mean runtime |
| --- | ---: | ---: | ---: | ---: | ---: |
| BUSI | 15 | 0.80 | 0.6376 | 0.1471 | 0.56s |
| Pneumonia | 20 | 0.45 | 0.7639 | 0.1510 | 0.39s |

Interpretation:

```text
More localized and visually readable than the prototype-guided baseline, but
less consistently valid. Pneumonia remains difficult because simple geometric
lung-field constraints are not true lung segmentation.
```

## Method 3: DVCE-Style Diffusion-Guided Generation

The DVCE-style prototype uses a diffusion-guided generation process together
with the medical ResNet18 classifier adapter. It covers the generative
counterfactual category.

Recommended wording:

```text
DVCE-style diffusion-guided feasibility prototype
```

Fixed-evaluation result with the current diffusion checkpoint:

| Dataset | Samples | Validity | Mean CF confidence | Mean changed pixels above threshold | Mean runtime |
| --- | ---: | ---: | ---: | ---: | ---: |
| BUSI | 5 | 1.00 | 0.7034 | 0.3569 | 8.86s |
| Pneumonia | 5 | 0.80 | 0.7219 | 0.1654 | 9.49s |

Interpretation:

```text
Generative and capable of target-class flips, but still limited by the
non-medical diffusion prior. Visual plausibility must be evaluated separately
from model validity.
```

## Overall Comparison

| Criterion | Prototype-guided baseline | SEDC-T-style | DVCE-style |
| --- | --- | --- | --- |
| Validity | strongest | lower, especially on Pneumonia | promising on small fixed subset |
| Locality | weak to moderate | strongest | weak to moderate |
| Runtime | moderate | fastest | slowest |
| Visual interpretability | limited by diffuse changes | strongest | limited by artifacts/domain mismatch |
| Best role | technical baseline | main region-based method | generative feasibility method |

## Main Takeaway

The methods show a trade-off between validity and interpretability.

```text
Prototype-guided optimization is highly valid but less localized.
SEDC-T-style replacement is localized but not always valid.
DVCE-style generation is generative but currently limited by domain mismatch.
```

The project therefore compares different explanation behaviors rather than
claiming that one method is universally best.
