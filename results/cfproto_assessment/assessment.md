# Prototype-Guided Baseline Assessment

## Goal

This assessment checks whether the current prototype-guided counterfactual method produces results that are not only valid for the model, but also visually meaningful and interpretable.

## Tested Runs

BUSI:

```text
Output: results/cfproto_assessment/busi_5samples
Samples: 5
Validity: 5/5
```

Pneumonia:

```text
Output: results/cfproto_assessment/pneumonia_5samples
Samples: 5
Validity: 5/5
```

## Quantitative Summary

| Dataset   | Validity | Mean L1 | Mean L2 | Mean changed pixel fraction | Mean runtime |
| --------- | -------: | ------: | ------: | --------------------------: | -----------: |
| BUSI      |     1.00 |  0.0431 |  0.0539 |                      0.5805 |       12.77s |
| Pneumonia |     1.00 |  0.0359 |  0.0438 |                      0.4474 |        9.65s |

## Interpretation

The method is technically successful: all tested counterfactuals changed the model prediction to the selected target class.

However, the visual changes are only partly interpretable. The generated perturbations are smoother than the earlier noisy pixel-level version, but they still often appear as broad intensity or contrast changes rather than localized, semantically meaningful medical changes.

For BUSI, the counterfactuals often modify larger ultrasound regions and shadows. This can indicate which image patterns influence the model, but it does not clearly show a medically realistic transformation from benign to malignant or normal.

For Pneumonia, the counterfactuals are visually cleaner than the BUSI examples. Still, the changes mostly look like diffuse intensity changes in the chest image rather than clearly localized pneumonia-like infiltrates.

## Conclusion

The current prototype-guided optimization method is useful as a first baseline counterfactual method because:

```text
1. It works with the existing PyTorch ResNet18 models.
2. It is multiclass-capable.
3. It produces valid model counterfactuals.
4. It stores predictions, confidence values, validity, change metrics, and visual overlays.
```

But it should not yet be presented as producing fully realistic or clinically meaningful counterfactual images.

The best wording for the project is:

```text
The prototype-guided method produces valid model counterfactuals, but the resulting changes are mostly low-frequency intensity perturbations. Therefore, the method is suitable as a technical baseline, while plausibility and medical interpretability remain limited.
```
