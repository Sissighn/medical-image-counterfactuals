# CFProto / Prototype Counterfactual Start

## Repository Decision

Two external repositories were considered:

```text
SeldonIO/alibi
e-delaney/cfe_images_how_people_differ_from_machines
```

`SeldonIO/alibi` contains the original CFProto-style method, but its image examples and model integration are mainly designed around TensorFlow/Keras workflows. Our project currently uses PyTorch ResNet18 checkpoints, so using Alibi directly would require an additional model wrapper or model conversion step.

`e-delaney/cfe_images_how_people_differ_from_machines` is relevant as a counterfactual image reference, but it is not a direct plug-and-play implementation for our BUSI and Pneumonia PyTorch models.

For this project phase, a local PyTorch implementation was created:

```text
scripts/run_cfproto_pytorch.py
```

This implementation follows the main idea of prototype-guided counterfactuals:

```text
1. Load the trained ResNet18 model.
2. Use the penultimate ResNet feature space.
3. Compute one feature prototype per class from the training split.
4. Select correctly classified test images.
5. Optimize the input image toward a target class.
6. Penalize large pixel changes and noisy changes.
7. Keep the counterfactual grayscale by default, because BUSI and Pneumonia are grayscale medical images.
8. Save original image, counterfactual image, difference image, predictions, probabilities, runtime, validity, and change metrics.
```

## First Smoke Tests

BUSI:

```text
Model: models/busi_resnet18_pretrained.pth
Output: results/cfproto/busi_first
Samples: 1
Steps: 250
Validity: 1/1
Example: benign -> malignant
```

Pneumonia:

```text
Model: models/pneumonia_resnet18_pretrained.pth
Output: results/cfproto/pneumonia_first
Samples: 1
Steps: 250
Validity: 1/1
Example: NORMAL -> PNEUMONIA
```

## Output Files

Each run writes:

```text
metadata.json
sample_XX.png
sample_XX.summary.png
```

The `metadata.json` file stores the original prediction, target class, counterfactual prediction, probabilities, runtime, whether the counterfactual is valid, and simple change metrics.

The current change metrics are:

```text
l1_mean
l2_mean
linf
changed_pixel_fraction
```

The `.summary.png` image shows:

```text
Original image | Counterfactual image | Absolute difference
```

## Current Status

The first prototype-guided counterfactual pipeline works for both datasets with the pretrained ResNet18 baselines.

This is a first working implementation, not yet the final evaluation setup. The generated examples are valid model counterfactuals, but they still contain visible noise-like perturbations. The next step is to run it on more samples and tune the regularization parameters to improve plausibility.

The next evaluation metrics are:

```text
Validity
Proximity
Sparsity
Runtime
```
