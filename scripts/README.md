# Script Overview

Run all commands from the repository root with `PYTHONPATH=.`.

## Data Preparation

| Script | Purpose |
| --- | --- |
| `prepare_busi.py` | Excludes BUSI mask images and creates class-wise 70/15/15 splits with seed 42. |
| `prepare_pneumonia.py` | Copies the existing train, validation, and test structure of the Pneumonia dataset. |
| `check_dataset.py` | Reports image counts by dataset, split, and class. |
| `test_data_utils.py` | Loads one test batch from each prepared dataset. |
| `prepare_diffusion_training_data.py` | Builds the image collection used for diffusion training. |
| `check_diffusion_training_setup.py` | Checks paths and parameters for the local DVCE training setup. |

## Model Training

| Script | Purpose |
| --- | --- |
| `src/train_model.py` | Trains ResNet18 with optional ImageNet initialization, augmentation, and class weights. |
| `evaluate_model.py` | Computes accuracy, weighted F1, precision, recall, and the confusion matrix on the test split. |
| `train_autoencoder.py` | Trains the autoencoder used by CFProto. |
| `check_autoencoder_plausibility.py` | Compares reconstruction losses for original and artificially perturbed images. |
| `train_robust_resnet18_pgd.py` | Trains the robust auxiliary classifier used for DVCE Cone Projection. |

## Counterfactual Methods

| Script | Purpose |
| --- | --- |
| `create_evaluation_manifest.py` | Selects correctly classified test images and fixed target classes. |
| `run_cfproto_pytorch.py` | Generates prototype-guided pixel-space counterfactuals. |
| `run_goyal_cve_pytorch.py` | Replaces spatial feature cells with cells from a target-class distractor. |
| `run_sedc_t_pytorch.py` | Searches for counterfactuals by replacing Quickshift segments. |
| `run_dvce_pytorch.py` | Generates diffusion-based counterfactuals with optional Cone Projection. |

## Result Processing

| Script | Purpose |
| --- | --- |
| `summarize_counterfactual_evaluation.py` | Builds the shared quantitative table from final metadata. |
| `select_interpretable_examples.py` | Selects representative valid cases, visually questionable cases, and failures. |
| `create_qualitative_comparison_figures.py` | Creates the method-specific qualitative figures. |

Authoritative parameters are documented in
[`results/final_configs/README.md`](../results/final_configs/README.md).
