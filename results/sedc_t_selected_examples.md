# Selected SEDC-T Examples

This file summarizes the most useful SEDC-T examples for presentation and report writing.
The selection prioritizes valid counterfactuals, sparse changes, readable visualizations, and changes that are not only numerical but also visually interpretable.

## Selection Criteria

- The counterfactual must be valid: `counterfactual_prediction == target_class`.
- The changed image region should be as small as possible.
- The selected segments should be visually meaningful and not dominated by borders, labels, or obvious artifacts.
- The counterfactual confidence should be high enough to show a real model decision change.
- Failure cases are kept separately because they are useful for discussing limitations.

## Recommended BUSI Examples

BUSI provides the stronger qualitative SEDC-T results. The selected segments are often close to the visible lesion or suspicious ultrasound structure, so these examples are suitable for the main presentation.

| Sample | Target | Prediction change | CF confidence | Changed pixels | Changed segments | Image |
| --- | --- | --- | ---: | ---: | ---: | --- |
| 00 | benign -> malignant | benign -> malignant | 0.5483 | 0.0186 | 1 | `results/sedc_t/busi_minimal_20samples/sample_00.summary.png` |
| 06 | benign -> malignant | benign -> malignant | 0.5172 | 0.0212 | 1 | `results/sedc_t/busi_minimal_20samples/sample_06.summary.png` |
| 04 | benign -> malignant | benign -> malignant | 0.6220 | 0.0628 | 3 | `results/sedc_t/busi_minimal_20samples/sample_04.summary.png` |
| 10 | benign -> normal | benign -> normal | 0.7612 | 0.0667 | 3 | `results/sedc_t/busi_minimal_20samples/sample_10.summary.png` |
| 15 | benign -> malignant | benign -> malignant | 0.8214 | 0.1776 | 7 | `results/sedc_t/busi_minimal_20samples/sample_15.summary.png` |

Interpretation:

The best BUSI examples show that SEDC-T can flip the model prediction by replacing a small number of image segments. Samples 00 and 06 are especially compact because only one segment is modified. Samples 10 and 15 are useful when a stronger counterfactual confidence is needed, although sample 15 modifies a larger image region.

## Recommended Pneumonia Examples

Pneumonia is less clean than BUSI. The method often changes regions inside the chest area, but some selected segments still overlap with heart, spine, or broad anatomical structures instead of clearly isolated lung findings. These examples can be shown, but they should be framed more carefully.

| Sample | Target | Prediction change | CF confidence | Changed pixels | Changed segments | Image |
| --- | --- | --- | ---: | ---: | ---: | --- |
| 00 | NORMAL -> PNEUMONIA | NORMAL -> PNEUMONIA | 0.5071 | 0.0656 | 3 | `results/sedc_t/pneumonia_lung_roi_minimal_20samples/sample_00.summary.png` |
| 02 | NORMAL -> PNEUMONIA | NORMAL -> PNEUMONIA | 0.5265 | 0.0614 | 4 | `results/sedc_t/pneumonia_lung_roi_minimal_20samples/sample_02.summary.png` |
| 07 | NORMAL -> PNEUMONIA | NORMAL -> PNEUMONIA | 0.5165 | 0.0671 | 4 | `results/sedc_t/pneumonia_lung_roi_minimal_20samples/sample_07.summary.png` |
| 08 | NORMAL -> PNEUMONIA | NORMAL -> PNEUMONIA | 0.5358 | 0.0555 | 3 | `results/sedc_t/pneumonia_lung_roi_minimal_20samples/sample_08.summary.png` |
| 16 | NORMAL -> PNEUMONIA | NORMAL -> PNEUMONIA | 0.5493 | 0.0478 | 4 | `results/sedc_t/pneumonia_lung_roi_minimal_20samples/sample_16.summary.png` |

Interpretation:

The selected Pneumonia examples are valid and relatively sparse. The new lung-field ROI reduces some irrelevant candidate regions, but the examples are still less medically specific than the BUSI examples. This suggests that SEDC-T is useful for model-behavior analysis, but a stronger anatomical constraint would be needed for better medical plausibility on chest X-rays.

## Failure Cases To Discuss

| Dataset | Sample | Issue | Image |
| --- | --- | --- | --- |
| BUSI | 13 | Not valid after the maximum number of segment replacements. The model remains in the original class. | `results/sedc_t/busi_minimal_20samples/sample_13.summary.png` |
| Pneumonia | 11 | Not valid after 12 selected segments. | `results/sedc_t/pneumonia_lung_roi_minimal_20samples/sample_11.summary.png` |
| Pneumonia | 13 | Not valid after 12 selected segments. | `results/sedc_t/pneumonia_lung_roi_minimal_20samples/sample_13.summary.png` |
| Pneumonia | 15 | Not valid after 12 selected segments. | `results/sedc_t/pneumonia_lung_roi_minimal_20samples/sample_15.summary.png` |
| Pneumonia | 03 and 18 | Valid but weak examples: the original model confidence is low and the selected regions are not clearly disease-specific. | `results/sedc_t/pneumonia_lung_roi_minimal_20samples/sample_03.summary.png`, `results/sedc_t/pneumonia_lung_roi_minimal_20samples/sample_18.summary.png` |

## Suggested Use In The Project

For the main report, use BUSI samples 00, 04, and 10 as strong examples. They show valid counterfactuals with localized and visually understandable changes.

For Pneumonia, use one or two examples only and explicitly describe the limitation: the method can flip the model prediction with sparse region changes, but the selected regions are not always medically convincing. This is a good argument for a future improvement using lung segmentation or a region-of-interest mask.

The comparison with the prototype-guided optimization baseline should emphasize the trade-off:

- The prototype-guided optimization baseline reached higher validity in the fixed test, but changed broad image areas and was harder to interpret.
- SEDC-T reached slightly lower validity, especially on Pneumonia, but produced much more localized and presentation-friendly explanations.
