# Medical Image Counterfactuals

This repository contains code and experiment summaries for comparing
counterfactual explanation methods for medical image classification.

The project focuses on two datasets:

- BUSI breast ultrasound classification with `benign`, `malignant`, and `normal`
  classes.
- Chest X-ray pneumonia classification with `NORMAL` and `PNEUMONIA` classes.

The workflow is:

```text
prepare datasets -> train ResNet18 classifiers -> generate counterfactuals -> evaluate methods
```

## Current Status

The classification models are pretrained ResNet18 baselines trained with data
augmentation and class-weighted cross entropy.

| Dataset | Model | Accuracy | Weighted F1 |
| --- | --- | ---: | ---: |
| BUSI | ResNet18 pretrained | 0.8390 | 0.8365 |
| Pneumonia | ResNet18 pretrained | 0.8782 | 0.8732 |

Three counterfactual directions are currently implemented:

| Method | Role |
| --- | --- |
| Prototype-guided optimization baseline | High-validity technical baseline |
| SEDC-T segment replacement | Region-based and more localized explanations |
| DVCE-style diffusion-guided generation | Generative feasibility method |

The fixed evaluation summary is stored in:

```text
results/fixed_evaluation_summary.md
```

Method variants and project-specific adaptations are documented separately:

```text
results/method_variant_rationale.md
results/method_implementation_audit.md
```

## Repository Structure

```text
.
|-- README.md
|-- requirements.txt
|-- requirements-dvce.txt
|-- src/
|   |-- data_utils.py
|   |-- evaluation_manifest.py
|   `-- train_model.py
|-- scripts/
|   |-- create_evaluation_manifest.py
|   |-- evaluate_model.py
|   |-- prepare_busi.py
|   |-- prepare_pneumonia.py
|   |-- prepare_diffusion_training_data.py
|   |-- run_cfproto_pytorch.py
|   |-- run_sedc_t_pytorch.py
|   |-- run_dvce_medical_prototype.py
|   `-- summarize_counterfactual_evaluation.py
`-- results/
    |-- baseline_comparison.md
    |-- method_comparison.md
    |-- fixed_evaluation_summary.md
    |-- final_method_summary.md
    |-- evaluation_manifests/
    `-- fixed_evaluation/
```

The following folders are intentionally not tracked by Git:

```text
data/
models/
external/
checkpoints/
.venv/
.venv-dvce/
```

This keeps datasets, model checkpoints, external repositories, and generated
visual artifacts out of the repository.

## Environment Setup

Create and activate a Python virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

DVCE-related experiments may require a separate environment and the additional
dependencies listed in:

```text
requirements-dvce.txt
```

## Data

The code expects processed datasets in ImageFolder format:

```text
data/processed/BUSI/
|-- train/
|-- val/
`-- test/

data/processed/Pneumonia/
|-- train/
|-- val/
`-- test/
```

Each split must contain one subfolder per class.

Expected BUSI classes:

```text
benign
malignant
normal
```

Expected Pneumonia classes:

```text
NORMAL
PNEUMONIA
```

The raw and processed datasets are not included in this repository.

## Training

The training script is:

```text
src/train_model.py
```

Example for BUSI:

```bash
PYTHONPATH=. python src/train_model.py \
  --dataset_name BUSI_pretrained \
  --dataset_path data/processed/BUSI \
  --output_model_path models/busi_resnet18_pretrained.pth \
  --epochs 15 \
  --batch_size 16 \
  --learning_rate 0.0001 \
  --pretrained
```

Example for Pneumonia:

```bash
PYTHONPATH=. python src/train_model.py \
  --dataset_name Pneumonia_pretrained \
  --dataset_path data/processed/Pneumonia \
  --output_model_path models/pneumonia_resnet18_pretrained.pth \
  --epochs 10 \
  --batch_size 16 \
  --learning_rate 0.0001 \
  --pretrained
```

The training pipeline supports:

- ResNet18 classification
- optional ImageNet pretrained weights
- training-time data augmentation
- class-weighted cross entropy
- automatic device selection for MPS, CUDA, or CPU
- best-checkpoint saving based on validation F1 score

## Model Evaluation

Use the evaluation script to compute test metrics:

```bash
PYTHONPATH=. python scripts/evaluate_model.py \
  --model_path models/busi_resnet18_pretrained.pth \
  --dataset_path data/processed/BUSI \
  --output_path results/busi_pretrained_test_evaluation.json
```

The evaluation output includes:

- accuracy
- weighted F1 score
- weighted precision
- weighted recall
- classification report
- confusion matrix

Baseline results are summarized in:

```text
results/baseline_comparison.md
```

## Fixed Counterfactual Evaluation

The final method comparison uses fixed manifests of correctly classified test
samples. This prevents each method from silently evaluating on different images.

Current fixed manifests:

```text
results/evaluation_manifests/busi_balanced_5_per_class_second_best.json
results/evaluation_manifests/pneumonia_balanced_10_per_class_second_best.json
```

The target class is selected with the `second_best` strategy: the target is the
most likely non-original class according to the classifier.

Create a manifest:

```bash
PYTHONPATH=. python scripts/create_evaluation_manifest.py \
  --model_path models/busi_resnet18_pretrained.pth \
  --dataset_path data/processed/BUSI \
  --output_path results/evaluation_manifests/busi_balanced_5_per_class_second_best.json \
  --samples_per_class 5 \
  --target_strategy second_best
```

## Counterfactual Methods

### Prototype-Guided Optimization Baseline

```bash
PYTHONPATH=. python scripts/run_cfproto_pytorch.py \
  --model_path models/busi_resnet18_pretrained.pth \
  --dataset_path data/processed/BUSI \
  --output_dir results/fixed_evaluation/prototype_busi_balanced_manifest \
  --manifest_path results/evaluation_manifests/busi_balanced_5_per_class_second_best.json
```

This method computes feature prototypes from the ResNet18 embedding space and
optimizes an input image toward the target class while penalizing large or noisy
changes.

### SEDC-T Segment Replacement

```bash
PYTHONPATH=. python scripts/run_sedc_t_pytorch.py \
  --model_path models/busi_resnet18_pretrained.pth \
  --dataset_path data/processed/BUSI \
  --output_dir results/fixed_evaluation/sedc_t_original_style_busi_balanced_manifest \
  --manifest_path results/evaluation_manifests/busi_balanced_5_per_class_second_best.json \
  --search_mode original_best_first \
  --roi_mode none
```

This method segments the image and searches for region replacements that change
the classifier output to the target class. The `original_best_first` mode follows
the target-score best-first search more closely. The faster `greedy_minimal`
mode is kept as a project variant and should be reported separately.

### DVCE-Style Diffusion-Guided Generation

```bash
PYTHONPATH=. python scripts/run_dvce_medical_prototype.py \
  --model_path models/busi_resnet18_pretrained.pth \
  --dataset_path data/processed/BUSI \
  --output_dir results/fixed_evaluation/dvce_busi_manifest_5_current_checkpoint \
  --manifest_path results/evaluation_manifests/busi_balanced_5_per_class_second_best.json \
  --manifest_max_samples 5 \
  --run_generation \
  --timestep_respacing 50 \
  --skip_timesteps 44
```

This method adapts a diffusion-guided counterfactual workflow to the medical
ResNet18 classifiers. It is currently treated as a feasibility-level generative
method because the diffusion prior is not medical-domain-specific.

## Current Fixed Evaluation Results

| Method | Dataset | Samples | Validity | Mean CF confidence | Mean change | Runtime |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Prototype-guided optimization | BUSI | 15 | 1.00 | 0.9978 | 0.0559 | 5.24s |
| Prototype-guided optimization | Pneumonia | 20 | 1.00 | 0.9928 | 0.1442 | 5.69s |
| SEDC-T original-style best-first | BUSI | 15 | 0.80 | 0.6674 | 0.1517 | 6.59s |
| SEDC-T original-style best-first | Pneumonia | 20 | 0.55 | 0.7343 | 0.1410 | 13.78s |
| SEDC-T project variant | BUSI | 15 | 0.80 | 0.6376 | 0.1471 | 0.56s |
| SEDC-T project variant with lung-field ROI | Pneumonia | 20 | 0.45 | 0.7639 | 0.1510 | 0.39s |
| DVCE-style diffusion-guided generation | BUSI | 5 | 1.00 | 0.7034 | 0.3569 | 8.86s |
| DVCE-style diffusion-guided generation | Pneumonia | 5 | 0.80 | 0.7219 | 0.1654 | 9.49s |
| DVCE-style with Pneumonia fine-tuned checkpoint | Pneumonia | 5 | 0.80 | 0.6937 | 0.2469 | 15.63s |

Important interpretation:

```text
Validity means that the model prediction changed to the target class.
It does not imply that the image change is medically plausible.
```

Additional SEDC-T tuning results are summarized in
`results/sedc_t_tuning_summary.md`. They are treated as an ablation, not as a
replacement for the original-style SEDC-T result.

## Result Files

Important public result summaries:

```text
results/baseline_comparison.md
results/method_comparison.md
results/method_variant_rationale.md
results/method_implementation_audit.md
results/fixed_evaluation_summary.md
results/final_method_summary.md
```

Generated PNG/JPG visualizations are ignored by Git. Compact JSON and Markdown
summaries are kept when they are useful for reproducing or documenting the
evaluation.
