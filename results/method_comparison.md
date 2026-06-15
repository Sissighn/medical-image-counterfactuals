# Counterfactual Method Comparison

## Compared Methods

This comparison summarizes the first quantitative comparison between:

```text
CFProto-inspired prototype-guided optimization
SEDC-T-inspired targeted segment replacement
```

Both methods were tested on the pretrained ResNet18 baselines for BUSI and Pneumonia.

## Evaluation Setup

Each method was evaluated on 20 correctly classified test samples per dataset.

CFProto settings:

```text
BUSI:
steps=800, max_delta=0.15, perturbation_resolution=12

Pneumonia:
steps=600, max_delta=0.10, perturbation_resolution=12
```

SEDC-T settings:

```text
n_segments=80
compactness=8
max_segments=12
replacement_mode=blur
blur_kernel=31

Pneumonia:
exclude_border_fraction=0.10
```

## Quantitative Results

| Method | Dataset | Validity | Mean L1 | Mean L2 | Mean changed pixel fraction | Mean runtime |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| CFProto | BUSI | 1.00 | 0.0410 | 0.0514 | 0.5334 | 13.11s |
| CFProto | Pneumonia | 1.00 | 0.0382 | 0.0458 | 0.5228 | 10.44s |
| SEDC-T | BUSI | 0.95 | 0.0145 | 0.0508 | 0.1169 | 2.69s |
| SEDC-T | Pneumonia | 0.85 | 0.0071 | 0.0242 | 0.1115 | 1.60s |

Additional SEDC-T sparsity:

| Dataset | Mean changed segments | Mean changed segment fraction |
| --- | ---: | ---: |
| BUSI | 4.85 | 0.0794 |
| Pneumonia | 6.85 | 0.0958 |

## Interpretation

CFProto achieved perfect validity on both datasets. However, it changed a large fraction of image pixels. The resulting visual explanations often appear as low-frequency intensity or contrast changes. This makes CFProto useful as a technical baseline, but its medical interpretability is limited.

SEDC-T achieved slightly lower validity, especially on Pneumonia, but produced much more localized changes. It changed only about 11 percent of pixels on average and was substantially faster than CFProto. The selected segment overlays are easier to interpret visually because they identify concrete image regions.

For BUSI, SEDC-T often selects lesion-adjacent regions, which is useful for qualitative interpretation.

For Pneumonia, excluding border segments was important. Without this constraint, SEDC-T could select non-lung border regions. With border exclusion, the selected regions moved into the lung fields, making the results more meaningful.

## Current Conclusion

CFProto:

```text
Strength: high validity
Weakness: diffuse, less interpretable intensity changes
Best role: technical baseline
```

SEDC-T:

```text
Strength: localized, sparse, faster, visually easier to interpret
Weakness: validity is not perfect and segment replacement creates artificial occlusions
Best role: main interpretable region-based method so far
```