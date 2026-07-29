# Method Comparison and Results

## Experimental Scope

All methods explain the same non-robust ResNet18 for each dataset. The fixed
evaluation set contains 15 correctly classified BUSI test images and 20
correctly classified Pneumonia test images. The target is the class with the
second-highest original probability.

The evaluation separates model validity from medical plausibility. Validity
only means that the model predicts the requested target class after the
change.

## Baseline Classifiers

| Dataset | Accuracy | Weighted F1 |
| --- | ---: | ---: |
| BUSI | 0.8390 | 0.8365 |
| Pneumonia | 0.8782 | 0.8732 |

The BUSI test split contains 118 images; the Pneumonia test split contains 624.
See [`results/baseline_classifiers/README.md`](../baseline_classifiers/README.md)
for class-level results.

## Metrics

- **Validity:** fraction of samples that reach the target class.
- **CF confidence:** model probability of the predicted counterfactual class.
- **MAD (`l1_mean`):** mean absolute pixel difference.
- **RMS (`l2_mean`):** root mean squared pixel difference.
- **Change fraction:** pixel fraction above 0.03 for CFProto and Goyal-CVE,
  pixel fraction above 0.05 for DVCE, and segment-mask area for SEDC-T.
- **Runtime:** measured execution time per sample in the respective
  environment.

MAD and RMS are computed over all pixels and channels in `[0, 1]` image space.

## Quantitative Results

| Method | Dataset | n | Validity | CF confidence | Change fraction | Runtime |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| CFProto | BUSI | 15 | 0.8667 | 0.6815 | 0.0529 | 46.10 s |
| CFProto | Pneumonia | 20 | 1.0000 | 0.5740 | 0.0180 | 46.34 s |
| Goyal-CVE | BUSI | 15 | 1.0000 | 0.5279 | 0.2596 | 0.25 s |
| Goyal-CVE | Pneumonia | 20 | 1.0000 | 0.5231 | 0.3072 | 0.17 s |
| SEDC-T | BUSI | 15 | 0.8000 | 0.6343 | 0.2640 | 6.71 s |
| SEDC-T | Pneumonia | 20 | 0.5500 | 0.6759 | 0.3270 | 13.92 s |
| SEDC-T lung-field ROI | Pneumonia | 20 | 0.5000 | 0.7770 | 0.1745 | 15.23 s |
| DVCE Cone, OpenAI | BUSI | 15 | 0.9333 | 0.9439 | 0.1157 | 1173.45 s |
| DVCE Cone, OpenAI | Pneumonia | 20 | 0.8000 | 0.8369 | 0.0510 | 700.15 s |
| DVCE Cone, BUSI fine-tuned | BUSI | 15 | 1.0000 | 0.9984 | 0.1565 | 44.62 s |
| DVCE Cone, Pneumonia fine-tuned | Pneumonia | 20 | 1.0000 | 0.9800 | 0.0669 | 44.94 s |
| DVCE without Cone, BUSI fine-tuned | BUSI | 15 | 1.0000 | 0.9976 | 0.1359 | 33.24 s |
| DVCE without Cone, Pneumonia fine-tuned | Pneumonia | 20 | 1.0000 | 0.9951 | 0.0520 | 33.21 s |
| DVCE without Cone, OpenAI | Pneumonia | 20 | 1.0000 | 0.9972 | 0.0596 | 49.95 s |

The table generated directly from metadata is available in
[`fixed_evaluation_summary.md`](fixed_evaluation_summary.md).

## Interpretation by Method

### CFProto

CFProto reaches full validity on Pneumonia and high validity on BUSI while
producing the smallest mean change fractions among the non-generative methods.
Individual BUSI cases still contain broad, grainy changes, so small distance
metrics alone do not establish medical plausibility.

### Goyal-CVE

Goyal-CVE reaches full validity on both evaluation sets and is the fastest
method in the recorded runs. It inserts real feature cells from a target-class
distractor. A few cells can yield a localized explanation; many cells create
mosaic-like transfers of anatomy and acquisition differences. Mean edit counts
are 14.00 for BUSI and 16.15 for Pneumonia.

### SEDC-T

SEDC-T produces directly inspectable segment masks but reaches validity of only
0.80 on BUSI and 0.55 on Pneumonia. The geometric lung-field ROI reduces the
mean Pneumonia change fraction from 0.3270 to 0.1745 while lowering validity
from 0.55 to 0.50. It is therefore a locality ablation, not an overall
performance improvement or a medical lung segmentation.

### DVCE

The medically fine-tuned Cone and No-Cone runs reach full model validity and
high target confidence on the evaluated samples. Their qualitative figures
still contain global reconstruction, texture changes, and artificial
structures. A model-valid DVCE is therefore not automatically a plausible
medical counterfactual.

The OpenAI and medically fine-tuned DVCE runs did not use identical devices
and precision settings. Runtime and confidence differences cannot isolate a
causal checkpoint effect.

## Overall Finding

No method dominates validity, proximity, locality, runtime, and visual
plausibility simultaneously:

- CFProto changes relatively little on average but can produce
  hard-to-interpret pixel patterns.
- Goyal-CVE is fast and valid but becomes broad and mosaic-like with many
  feature-cell swaps.
- SEDC-T exposes spatial regions directly but has the highest failure rate.
- DVCE reaches high model validity but remains visually sensitive to checkpoint
  and guidance choices.

The case-by-case visual analysis is documented in
[`results/qualitative_figures/qualitative_results_interpretation.md`](../qualitative_figures/qualitative_results_interpretation.md).
