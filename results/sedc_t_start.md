# SEDC-T Start

## Source Repository

Reference implementation:

```text
https://github.com/ADMAntwerp/ImageCounterfactualExplanations
```

The repository implements SEDC and SEDC-T for image counterfactual explanations. The provided runnable script targets TensorFlow Hub MobileNetV2 and ImageNet labels, so it is not directly plug-and-play for this project.

## Method Idea

SEDC-T is a targeted, model-agnostic counterfactual method for image classification.

The main idea is:

```text
1. Segment the input image into superpixels.
2. Replace one or more segments with a baseline version of the image.
3. Query the classifier after each replacement.
4. Search for a small set of segments whose replacement changes the prediction to the target class.
```

This differs from our current CFProto-style method:

```text
CFProto-style method:
Optimizes image intensities directly and often produces diffuse changes.

SEDC-T:
Removes or replaces image regions and can therefore produce more localized explanations.
```

## Adaptation For This Project

The external implementation assumes:

```text
TensorFlow / Keras classifier
MobileNetV2
ImageNet labels
quickshift segmentation
replacement modes: mean, blur, random, inpaint
```

Our project uses:

```text
PyTorch ResNet18 checkpoints
BUSI and Pneumonia class labels
ImageFolder datasets
ImageNet mean/std normalization for model input
```

Therefore, we will implement a small PyTorch-native SEDC-T script instead of copying the TensorFlow script directly.

## Planned First Implementation

The first PyTorch SEDC-T script should:

```text
1. Load a trained ResNet18 checkpoint.
2. Select correctly classified test samples.
3. Segment each image with scikit-image.
4. Replace candidate segments with blur or mean values.
5. Search for a small segment set that changes the prediction to a target class.
6. Save the original image, perturbed image, segment mask, overlay, predictions, confidence values, and validity.
```

## Expected Benefit

SEDC-T may be more interpretable than the current prototype-guided method because it identifies image regions whose removal or replacement changes the model decision.

This could be especially useful for:

```text
BUSI:
Showing whether the classifier focuses on lesion-adjacent regions or ultrasound artifacts.

Pneumonia:
Showing whether the classifier relies on lung regions or non-lung artifacts.
```

## First Caution

SEDC-T still does not guarantee medical plausibility. Replacing image segments with blur or mean values can create artificial occlusions. However, these occlusions are often easier to interpret than diffuse pixel-level counterfactual perturbations.

## First PyTorch Implementation

A PyTorch-native first implementation was added:

```text
scripts/run_sedc_t_pytorch.py
```

The script:

```text
1. Loads a trained ResNet18 checkpoint.
2. Selects correctly classified test images.
3. Segments each image with scikit-image SLIC.
4. Greedily replaces candidate segments with a blurred version of the image.
5. Stops when the model prediction changes to the target class or when the maximum number of segments is reached.
6. Saves original image, counterfactual image, difference map, selected segment overlay, prediction metadata, and change metrics.
```

For Pneumonia, border segments can be excluded:

```text
--exclude_border_fraction 0.10
```

This is useful because the first unconstrained test selected an image border region. Excluding border regions forced the search toward lung-relevant regions.

## First Results

BUSI:

```text
Output: results/sedc_t/busi_5samples
Samples: 5
Validity: 4/5
Mean changed pixel fraction: 0.0890
Mean runtime: 3.13s
```

Pneumonia:

```text
Output: results/sedc_t/pneumonia_5samples
Samples: 5
Validity: 5/5
Mean changed pixel fraction: 0.0673
Mean runtime: 0.97s
Border exclusion: 0.10
```

## 20-Sample Evaluation

After the initial smoke tests, SEDC-T was evaluated on 20 correctly classified samples per dataset.

BUSI:

```text
Output: results/sedc_t/busi_20samples_max12
Samples: 20
Validity: 19/20
Mean changed pixel fraction: 0.1169
Mean changed segments: 4.85
Mean runtime: 2.69s
```

Pneumonia:

```text
Output: results/sedc_t/pneumonia_20samples_max12
Samples: 20
Validity: 17/20
Mean changed pixel fraction: 0.1115
Mean changed segments: 6.85
Mean runtime: 1.60s
Border exclusion: 0.10
```

The full comparison with CFProto is documented in:

```text
results/method_comparison.md
```

## Improved SEDC-T Run

The SEDC-T implementation was improved in two small ways:

```text
1. Minimality-aware candidate selection:
   If several candidate replacements already produce the target class, the method now
   chooses the candidate with the smallest changed pixel fraction.

2. Optional ROI restriction:
   Candidate segments can be restricted to a simple geometric region of interest.
   For Pneumonia, the new lung_fields ROI restricts the search toward left and right
   lung-field regions.
```

The new CLI options are:

```text
--roi_mode none|central_chest|lung_fields
--roi_min_overlap 0.35
```

Improved BUSI run:

```text
Output: results/sedc_t/busi_minimal_20samples
Samples: 20
Validity: 19/20
Mean changed pixel fraction: 0.1073
Mean changed segments: 4.85
Mean runtime: 2.19s
```

Improved Pneumonia run:

```text
Output: results/sedc_t/pneumonia_lung_roi_minimal_20samples
Samples: 20
Validity: 17/20
Mean changed pixel fraction: 0.1085
Mean changed segments: 6.90
Mean runtime: 1.02s
ROI: lung_fields
ROI minimum overlap: 0.35
Border exclusion: 0.10
```

The improved version keeps the same validity as before, but the average changed pixel fraction is slightly lower. The Pneumonia results are still not fully medically convincing, so a proper lung segmentation mask remains an important future improvement.

## Current Interpretation

Compared with the CFProto-inspired method, SEDC-T produces more localized and easier-to-read explanations. The selected regions are visible directly in the overlay plot.

The first BUSI example selected a lesion-adjacent region. The first Pneumonia example, after excluding border regions, selected regions in both lung fields.

This makes SEDC-T promising as the second method for the project. It still creates artificial occlusions because segments are blurred, but the explanations are more localized and interpretable than the diffuse intensity changes produced by the current CFProto-style method.
