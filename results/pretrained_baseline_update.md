# Pretrained Baseline Update

## What Was Done

The ResNet18 ImageNet pretrained weights were fixed manually because the automatic `torchvision` download did not complete correctly. It created empty `.partial` files in the Torch cache and caused the pretrained model loading step to hang.

The required weight file was downloaded manually:

```text
~/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth
```

After that, `torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)` could load the pretrained weights directly from the local cache.

## Training Setup

Both datasets were retrained with:

```text
Model: ResNet18
Pretrained: Yes
Data augmentation: Yes
Class weights: Yes
Optimizer: Adam
Learning rate: 0.0001
Batch size: 16
```

BUSI was trained for 15 epochs:

```text
models/busi_resnet18_pretrained.pth
results/busi_pretrained_training_history.json
results/busi_pretrained_test_evaluation.json
```

Pneumonia was trained for 10 epochs:

```text
models/pneumonia_resnet18_pretrained.pth
results/pneumonia_pretrained_training_history.json
results/pneumonia_pretrained_test_evaluation.json
```

## Results

| Dataset | Model | Accuracy | Weighted F1 |
| --- | --- | ---: | ---: |
| BUSI | ResNet18 baseline | 0.7288 | 0.7043 |
| BUSI | ResNet18 improved without pretraining | 0.7203 | 0.7221 |
| BUSI | ResNet18 pretrained | 0.8390 | 0.8365 |
| Pneumonia | ResNet18 baseline | 0.7885 | 0.7679 |
| Pneumonia | ResNet18 improved without pretraining | 0.7965 | 0.7752 |
| Pneumonia | ResNet18 pretrained | 0.8782 | 0.8732 |

## Interpretation

Pretraining significantly improved both baselines.

For BUSI, the weighted F1 score increased from 0.7043 in the original baseline to 0.8365 with pretrained weights. This passes the target range of 0.80+ and gives a much stronger model for counterfactual explanations.

For Pneumonia, the weighted F1 score increased from 0.7679 in the original baseline to 0.8732 with pretrained weights. This passes the target range of 0.85+. The main remaining issue is that some NORMAL images are still classified as PNEUMONIA, but the number of these false positives decreased clearly.

## Next Step

The pretrained models are now the best available baselines and should be used for the next project phase:

```text
1. Start with prototype-guided / prototype-based counterfactual baselines.
2. Use the pretrained BUSI and Pneumonia checkpoints.
3. Generate first counterfactual examples.
4. Save original image, counterfactual image, original prediction, new prediction, and runtime.
```
