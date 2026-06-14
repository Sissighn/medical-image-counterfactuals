# Medical Image Counterfactuals

Project seminar code for training medical image classifiers and generating prototype-guided counterfactual explanations.

## Datasets

This repository expects local dataset folders under:

```text
data/processed/BUSI
data/processed/Pneumonia
```

The data folders are intentionally not tracked in Git because they are large and should be downloaded/prepared locally.

## Models

Trained model checkpoints are stored locally under:

```text
models/
```

Model checkpoints are intentionally ignored by Git. The current best local baselines are:

```text
models/busi_resnet18_pretrained.pth
models/pneumonia_resnet18_pretrained.pth
```

## Main Scripts

```text
src/train_model.py
scripts/evaluate_model.py
scripts/run_cfproto_pytorch.py
```

## Current Best Results

See:

```text
results/baseline_comparison.md
results/pretrained_baseline_update.md
results/cfproto_start.md
```
