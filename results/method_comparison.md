# Counterfactual Method Comparison

## Compared Methods

This project currently compares four counterfactual explanation directions for
medical image classification:

```text
1. CFProto-nearer prototype-guided optimization (`cfproto_encoder_knn`)
2. Retrieval-based nearest-unlike-neighbor baseline
3. SEDC-T original-style / SEDC-T-style targeted segment replacement
4. DVCE-style diffusion-guided generation
```

All methods are evaluated against trained ResNet18 classifiers for BUSI and
Pneumonia. The main quantitative criterion is whether the generated
counterfactual changes the model prediction to a predefined target class.

## Fixed Evaluation Setup

The final comparison uses fixed evaluation manifests instead of allowing each
method to choose its own samples.

```text
BUSI:      15 correctly classified test samples, 5 per class
Pneumonia: 20 correctly classified test samples, 10 per class
Target:    second_best non-original class
```

Manifest files:

```text
results/evaluation_manifests/busi_balanced_5_per_class_second_best.json
results/evaluation_manifests/pneumonia_balanced_10_per_class_second_best.json
```

DVCE is currently evaluated on the first 5 fixed manifest samples per dataset
because the diffusion sampling step is substantially more expensive than the
other two methods. For Pneumonia, an additional fine-tuned diffusion checkpoint
is reported separately.

## Quantitative Results

| Method | Dataset | Samples | Validity | Mean CF confidence | Mean change | Mean runtime |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| cfproto_encoder_knn final CFProto-nearer prototype-guided method | BUSI | 15 | 1.00 | 0.5471 | 0.0084 changed pixel fraction | 7.43s |
| cfproto_encoder_knn final CFProto-nearer prototype-guided method | Pneumonia | 20 | 0.90 | 0.5767 | 0.0108 changed pixel fraction | 8.58s |
| Prototype-guided legacy ResNet/class-mean baseline | BUSI | 15 | 1.00 | 0.9978 | 0.0559 changed pixel fraction | 5.24s |
| Prototype-guided legacy ResNet/class-mean baseline | Pneumonia | 20 | 1.00 | 0.9928 | 0.1442 changed pixel fraction | 5.69s |
| Prototype-guided plausibility ablation | BUSI | 15 | 1.00 | 0.9953 | 0.0315 changed pixel fraction | 4.78s |
| Prototype-guided plausibility ablation | Pneumonia | 20 | 1.00 | 0.9814 | 0.0934 changed pixel fraction | 4.76s |
| Retrieval-based nearest-unlike-neighbor baseline | BUSI | 15 | 1.00 | 0.8191 | 0.8516 changed pixel fraction | 0.01s |
| Retrieval-based nearest-unlike-neighbor baseline | Pneumonia | 20 | 1.00 | 0.6496 | 0.8741 changed pixel fraction | 0.01s |
| SEDC-T original-style best-first | BUSI | 15 | 0.80 | 0.6674 | 0.1517 changed pixel fraction | 6.59s |
| SEDC-T original-style best-first | Pneumonia | 20 | 0.55 | 0.7343 | 0.1410 changed pixel fraction | 13.78s |
| SEDC-T project variant | BUSI | 15 | 0.80 | 0.6376 | 0.1471 changed pixel fraction | 0.56s |
| SEDC-T project variant with lung-field ROI | Pneumonia | 20 | 0.45 | 0.7639 | 0.1510 changed pixel fraction | 0.39s |
| DVCE-style diffusion-guided generation | BUSI | 5 | 1.00 | 0.7034 | 0.3569 changed pixels above threshold | 8.86s |
| DVCE-style diffusion-guided generation | Pneumonia | 5 | 0.80 | 0.7219 | 0.1654 changed pixels above threshold | 9.49s |
| DVCE-style with Pneumonia fine-tuned checkpoint | Pneumonia | 5 | 0.80 | 0.6937 | 0.2469 changed pixels above threshold | 15.63s |

Detailed fixed-evaluation metadata is stored under:

```text
results/fixed_evaluation/
```

The compact generated summary is:

```text
results/fixed_evaluation_summary.md
```

Additional SEDC-T tuning details are documented in:

```text
results/sedc_t_tuning_summary.md
```

The rationale for why multiple variants are reported is documented in:

```text
results/method_variant_rationale.md
```

## Interpretation

The final CFProto-nearer prototype-guided method is reported as
`cfproto_encoder_knn`. It uses autoencoder encoder-space local target-class
KNN prototypes, adaptive binary-style c-search, elastic-net selection, and
polynomial learning-rate decay. It is methodically closer to CFProto than the
earlier ResNet/class-mean prototype baseline, but it is still not a full Alibi
CFProto reproduction.

The older prototype-guided rows remain in the table as legacy baselines or
ablations. They should not be treated as replacements for the final
`cfproto_encoder_knn` method. Their high confidence values are useful context,
but they came from earlier prototype spaces and settings.

A conservative prototype-guided plausibility ablation keeps the same method
core but increases regularization and lowers the maximum perturbation. It keeps
validity at 1.00 on both datasets while reducing changed pixel fraction from
0.0559 to 0.0315 on BUSI and from 0.1442 to 0.0934 on Pneumonia. This provides
better presentation candidates, but does not change the method's role as a
technical baseline with limited locality.

The retrieval-based nearest-unlike-neighbor baseline also reaches 1.00 validity
on both datasets because it retrieves real training images from the requested
target class that are correctly classified by the model. This makes it visually
intuitive as a case-based comparison: the counterfactual is an actual example
from the target class. However, it is not a minimal edit of the original image.
The high changed pixel fraction is expected because the retrieved image may
differ in patient anatomy, acquisition conditions, positioning, and general
image appearance.

The SEDC-T original-style best-first run is the safer method-faithfulness
baseline. It follows the target-score best-first search more closely and uses no
ROI restriction. It reaches 12/15 valid counterfactuals on BUSI and 11/20 on
Pneumonia, but is substantially slower than the project variant.

The SEDC-T project variant is a practical implementation variant. It keeps the
same targeted segment-replacement idea, but uses a greedier search and selects
valid candidates by smaller changed area. On Pneumonia, the reported project
variant also uses a simple `lung_fields` ROI. This ROI makes the candidate
regions easier to justify anatomically, but it is not an original SEDC-T
mechanism and should be described as a project-specific constraint.

A small SEDC-T tuning ablation did not fundamentally change this conclusion. On
Pneumonia, increasing the allowed segment budget improved the best project
variant only to 12/20 valid counterfactuals (validity 0.60). Larger budgets or
mean replacement increased the amount of changed image area without producing a
substantial validity improvement. This suggests that the low Pneumonia validity
is not just a parameter issue, but reflects a limitation of segment replacement
for this classifier/dataset combination.

DVCE-style diffusion-guided generation covers the generative method category. It
can produce valid target-class counterfactuals, but the current checkpoint is
not medical-domain-specific. A Pneumonia fine-tuned checkpoint was integrated
successfully and reaches the same 4/5 validity with a compromise parameter
setting, but requires stronger changes and longer runtime. The outputs therefore
need to be discussed carefully: model validity and medical plausibility are
separate questions.

## Current Conclusion

```text
CFProto-nearer prototype-guided optimization: high BUSI validity, lower Pneumonia validity than legacy, small but diffuse changes.
Legacy prototype-guided optimization: high validity, retained as baseline/ablation.
Retrieval-NUN: real target-class examples, interpretable as cases, not minimal edits.
SEDC-T original-style: method-faithful, localized, slower, moderate validity.
SEDC-T project variant: faster and constrained, but includes adaptations.
DVCE-style generation: generative and promising, but still sensitive to checkpoint and guidance settings.
```

No method should be described as clinically causal. The methods explain model
behavior under controlled perturbations or generations.
