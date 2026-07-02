# Counterfactual Method Comparison

## Compared Methods

This project currently compares three counterfactual explanation directions for
medical image classification:

```text
1. Prototype-guided optimization baseline
2. SEDC-T-style targeted segment replacement
3. DVCE-style diffusion-guided generation
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
| Prototype-guided optimization baseline | BUSI | 15 | 1.00 | 0.9978 | 0.0559 changed pixel fraction | 5.24s |
| Prototype-guided optimization baseline | Pneumonia | 20 | 1.00 | 0.9928 | 0.1442 changed pixel fraction | 5.69s |
| SEDC-T-style segment replacement | BUSI | 15 | 0.80 | 0.6376 | 0.1471 changed pixel fraction | 0.56s |
| SEDC-T-style segment replacement | Pneumonia | 20 | 0.45 | 0.7639 | 0.1510 changed pixel fraction | 0.39s |
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

## Interpretation

The prototype-guided baseline reaches the highest validity on both datasets. It
is useful as a technical baseline, but the changes are often diffuse intensity or
texture shifts rather than clearly localized medical changes.

SEDC-T-style segment replacement is less universally valid, especially for
Pneumonia. Its strength is locality: the method identifies concrete image
regions, which makes the visual output easier to discuss. The Pneumonia results
remain harder to interpret because simple geometric lung-field constraints are
not equivalent to real lung segmentation.

DVCE-style diffusion-guided generation covers the generative method category. It
can produce valid target-class counterfactuals, but the current checkpoint is
not medical-domain-specific. A Pneumonia fine-tuned checkpoint was integrated
successfully and reaches the same 4/5 validity with a compromise parameter
setting, but requires stronger changes and longer runtime. The outputs therefore
need to be discussed carefully: model validity and medical plausibility are
separate questions.

## Current Conclusion

```text
Prototype-guided optimization: high validity, limited locality.
SEDC-T-style replacement: more localized, lower validity.
DVCE-style generation: generative and promising, but still sensitive to checkpoint and guidance settings.
```

No method should be described as clinically causal. The methods explain model
behavior under controlled perturbations or generations.
