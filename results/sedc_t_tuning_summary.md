# SEDC-T Tuning Summary

This document summarizes a small, controlled SEDC-T tuning ablation on the fixed
evaluation manifests. The goal was not to replace the original-style SEDC-T
reference, but to check whether the low Pneumonia validity is mainly a parameter
issue or a method/data limitation.

## Setup

All runs use the fixed evaluation manifests:

- BUSI: 15 correctly classified samples, 5 per class.
- Pneumonia: 20 correctly classified samples, 10 per class.
- Target class: second-best non-original class.

The original-style SEDC-T run remains the method-faithfulness reference. The
tuned runs use the faster `greedy_minimal` project variant and vary only a small
number of interpretable parameters:

- maximum number of changed segments,
- ROI constraint,
- replacement mode.

## Variant Terminology

The project does not define four equally weighted SEDC-T methods. It contains
two main SEDC-T states and several additional tuning ablations:

| Category | Search mode | ROI | Role in the paper |
| --- | --- | --- | --- |
| Original-style reference | `original_best_first` | `none` | method-faithfulness reference for BUSI and Pneumonia |
| Project variant | `greedy_minimal` | `none` for BUSI, `lung_fields` for Pneumonia | faster practical variant; Pneumonia ROI is project-specific |
| Tuning ablations | `greedy_minimal` | varied | parameter checks, not separate main methods |

The ROI constraint is not part of the original-style SEDC-T reference. It was
introduced only as a project-specific Pneumonia variant to test whether a rough
anatomical restriction makes the selected regions easier to interpret. BUSI has
no lung-field ROI, so the comparable BUSI project variant uses `roi_mode=none`.

## Results

| Variant | Dataset | ROI | Replacement | Max segments | Validity | Mean CF confidence | Mean changed pixel fraction | Mean L1 | Mean runtime |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Original-style best-first | BUSI | none | blur | 6 | 0.80 | 0.6674 | 0.1517 | 0.0135 | 6.59s |
| Project variant | BUSI | none | blur | 6 | 0.80 | 0.6376 | 0.1471 | 0.0130 | 0.56s |
| Tuned project variant | BUSI | none | blur | 10 | 0.80 | 0.6050 | 0.1698 | 0.0141 | 1.00s |
| Original-style best-first | Pneumonia | none | blur | 6 | 0.55 | 0.7343 | 0.1410 | 0.0111 | 13.78s |
| Project variant | Pneumonia | lung_fields | blur | 6 | 0.45 | 0.7639 | 0.1510 | 0.0091 | 0.39s |
| Tuned project variant | Pneumonia | none | blur | 8 | 0.60 | 0.6800 | 0.1552 | 0.0120 | 1.21s |
| Tuned project variant | Pneumonia | none | blur | 10 | 0.60 | 0.6682 | 0.1814 | 0.0135 | 1.44s |
| Tuned project variant | Pneumonia | none | mean | 8 | 0.50 | 0.7086 | 0.1712 | 0.0291 | 1.32s |
| Tuned project variant | Pneumonia | lung_fields | blur | 10 | 0.50 | 0.7653 | 0.2131 | 0.0117 | 0.71s |

## Interpretation

BUSI does not improve with a larger segment budget. Increasing `max_segments`
from 6 to 10 keeps validity at 0.80, but increases the changed pixel fraction
from 0.1471 to 0.1698. This means the method changes more of the image without
solving additional fixed BUSI cases.

Pneumonia improves only slightly. The best tuned setting is:

```text
greedy_minimal, roi_mode=none, replacement_mode=blur, max_segments=8
```

It reaches 12/20 valid counterfactuals, compared with 11/20 for the
original-style run and 9/20 for the ROI-constrained project variant. Increasing
the segment budget further to 10 does not increase validity, but increases the
changed pixel fraction. Mean replacement also does not help and produces a much
higher mean L1 difference.

## Conclusion

The tuning does not indicate that SEDC-T was simply under-tuned. It improves
Pneumonia validity from 0.55 to 0.60 at best, but the gain is small and comes
with a higher amount of changed image area. This supports the interpretation
that SEDC-T has a genuine limitation on this Pneumonia setup:

- the model decision may depend on broad or diffuse image cues,
- segment replacement may not remove or create the relevant evidence cleanly,
- geometric ROI constraints improve anatomical control but can reduce the
  search space,
- higher validity can be obtained only by allowing broader, less minimal
  changes.

For the paper, the best framing is:

```text
SEDC-T is useful as a localized, region-based model-behavior explanation, but it
is not consistently strong for generating valid and medically plausible
counterfactuals on the Pneumonia dataset.
```
