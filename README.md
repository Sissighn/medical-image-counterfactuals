# Counterfactual Explanations for Medical Image Classification

This repository contains the implementation and evaluation artifacts for a
seminar project comparing counterfactual explanation methods for medical image
classifiers.

The study uses two image-classification tasks:

- **BUSI breast ultrasound:** `benign`, `malignant`, and `normal`
- **Chest X-ray pneumonia:** `NORMAL` and `PNEUMONIA`

The end-to-end workflow is:

```text
prepare data → train classifiers → create fixed manifests
             → generate counterfactuals → compare and interpret results
```

> **Scope:** A valid counterfactual is one that makes the classifier predict
> the requested target class. Model validity does not establish medical
> plausibility, clinical causality, or diagnostic relevance.

## Results at a Glance

### Baseline Classifiers

| Dataset | Model | Accuracy | Weighted F1 |
| --- | --- | ---: | ---: |
| BUSI | Pretrained ResNet18 | 0.8390 | 0.8365 |
| Pneumonia | Pretrained ResNet18 | 0.8782 | 0.8732 |

### Fixed Counterfactual Evaluation

| Method | Dataset | Samples | Validity | Mean CF confidence | Changed-pixel fraction | Mean runtime |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| CFProto (original-style) | BUSI | 15 | 0.87 | 0.6815 | 0.0529 | 46.10 s |
| CFProto (original-style) | Pneumonia | 20 | 1.00 | 0.5740 | 0.0180 | 46.34 s |
| Goyal et al. (2019) CVE | BUSI | 15 | 1.00 | 0.5279 | 0.2596 | 0.25 s |
| Goyal et al. (2019) CVE | Pneumonia | 20 | 1.00 | 0.5231 | 0.3072 | 0.17 s |
| SEDC-T original-style best-first | BUSI | 15 | 0.80 | 0.6343 | 0.2640 | 6.71 s |
| SEDC-T original-style best-first | Pneumonia | 20 | 0.55 | 0.6759 | 0.3270 | 13.92 s |
| SEDC-T lung-field ROI ablation | Pneumonia | 20 | 0.50 | 0.7770 | 0.1745 | 15.23 s |
| DVCE Cone, OpenAI checkpoint | BUSI | 15 | 0.93 | 0.944 | 0.116 | 1,173.4 s |
| DVCE Cone, OpenAI checkpoint | Pneumonia | 20 | 0.80 | 0.837 | 0.051 | 700.2 s |
| DVCE Cone, fine-tuned checkpoint | BUSI | 15 | 1.00 | 0.998 | 0.156 | 44.6 s |
| DVCE Cone, fine-tuned checkpoint | Pneumonia | 20 | 1.00 | 0.980 | 0.067 | 44.9 s |

The OpenAI DVCE runtimes were measured on a CPU-bound machine and should not be
compared directly with runtimes from different hardware.

For the complete evaluation, including ablations and interpretation, see the
[method comparison and results](results/docs/method_comparison_and_results.md)
and the automatically generated
[fixed evaluation summary](results/docs/fixed_evaluation_summary.md).

## Methods

| Method | Approach | Role in this project |
| --- | --- | --- |
| CFProto | Prototype-guided pixel-space optimization | FISTA with shrinkage-thresholding, an untargeted hinge loss, encoder-space prototypes, and binary search over the attack constant |
| Goyal et al. (2019) CVE | Instance-based feature-space editing | Greedy feature-cell swaps from a nearest-unlike target-class distractor |
| SEDC-T | Region-based segment replacement | Original-style best-first search; the lung-field ROI variant is reported separately as an ablation |
| DVCE | Diffusion-guided generation | Cone Projection with a robust PGD auxiliary classifier is the original-faithful configuration for the non-robust ResNet18 |

The detailed [method fidelity audit](results/docs/method_fidelity_comparison.md)
documents which components reproduce the original methods and which choices are
project-specific.

## Repository Structure

```text
.
├── src/                         reusable model and data utilities
├── scripts/                     data, training, generation, and evaluation CLIs
├── results/
│   ├── baseline_classifiers/    classifier metrics and comparisons
│   ├── docs/                    central result and fidelity documentation
│   ├── evaluation_manifests/    fixed evaluation samples and target classes
│   ├── final/                   compact metadata from final runs
│   ├── final_configs/           method documentation and run commands
│   └── qualitative_figures/     qualitative figure manifests and interpretation
├── requirements.txt             core Python dependencies
└── requirements-dvce.txt        optional DVCE-specific dependencies
```

Large or locally generated assets are intentionally excluded from version
control, including:

```text
data/
models/
checkpoints/
external/
generated PNG and JPG result files
debug, smoke-test, and ablation outputs
```

The repository therefore contains reproducibility code, fixed manifests,
metrics, and compact metadata, but not datasets, trained weights, external
method repositories, or generated image collections.

## Documentation

- [Scripts overview](scripts/README.md)
- [Method comparison and results](results/docs/method_comparison_and_results.md)
- [Method fidelity audit](results/docs/method_fidelity_comparison.md)
- [Fixed evaluation summary](results/docs/fixed_evaluation_summary.md)
- [Qualitative results](results/qualitative_figures/qualitative_results_interpretation.md)
- [Final method configurations](results/final_configs/)

## Installation

Create an isolated environment and install the core dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

DVCE has additional dependencies and is best run in a separate environment:

```bash
python -m pip install -r requirements.txt
python -m pip install -r requirements-dvce.txt
```

The original DVCE code targets an older CUDA/PyTorch stack. Consult the comments
in [`requirements-dvce.txt`](requirements-dvce.txt) before reproducing DVCE on a
new platform.

## Data Layout

Raw and processed datasets are not distributed with this repository. After
running the preparation scripts, each dataset must follow the ImageFolder
layout:

```text
data/processed/BUSI/{train,val,test}/{class_name}/...
data/processed/Pneumonia/{train,val,test}/{class_name}/...
```

Use [`prepare_busi.py`](scripts/prepare_busi.py) and
[`prepare_pneumonia.py`](scripts/prepare_pneumonia.py) to create the processed
splits. Dataset acquisition and use remain subject to the respective dataset
licenses and access conditions.

## Reproducing the Pipeline

Run commands from the repository root with `PYTHONPATH=.`.

### 1. Train the Classifiers

BUSI:

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

Pneumonia:

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

Train the robust auxiliary classifiers required for DVCE Cone Projection with
[`train_robust_resnet18_pgd.py`](scripts/train_robust_resnet18_pgd.py).

### 2. Create a Fixed Evaluation Manifest

The committed manifests contain correctly classified test samples and fixed
target classes:

- [BUSI manifest](results/evaluation_manifests/busi_balanced_5_per_class_second_best.json)
- [Pneumonia manifest](results/evaluation_manifests/pneumonia_balanced_10_per_class_second_best.json)

Example:

```bash
PYTHONPATH=. python scripts/create_evaluation_manifest.py \
  --model_path models/busi_resnet18_pretrained.pth \
  --dataset_path data/processed/BUSI \
  --output_path results/evaluation_manifests/busi_balanced_5_per_class_second_best.json \
  --samples_per_class 5 \
  --target_strategy second_best
```

### 3. Generate Counterfactuals

#### CFProto

```bash
PYTHONPATH=. python scripts/run_cfproto_pytorch.py \
  --model_path models/busi_resnet18_pretrained.pth \
  --dataset_path data/processed/BUSI \
  --output_dir results/final/cfproto_encoder_knn/busi \
  --manifest_path results/evaluation_manifests/busi_balanced_5_per_class_second_best.json \
  --autoencoder_path models/autoencoder_busi_bottleneck256.pth \
  --theta 0.5 \
  --gamma 1.0 \
  --c_steps 5 \
  --prototype_k 3
```

The method is a PyTorch reimplementation of Alibi's `CounterfactualProto`.
Black-box numerical gradients, categorical variables, k-d-tree prototypes, and
TrustScore filtering are not reproduced. See the
[CFProto documentation](results/final_configs/cfproto_encoder_method_documentation.md)
for the complete implementation-to-reference comparison.

#### Goyal et al. (2019) CVE

```bash
PYTHONPATH=. python scripts/run_goyal_cve_pytorch.py \
  --model_path models/busi_resnet18_pretrained.pth \
  --dataset_path data/processed/BUSI \
  --output_dir results/final/goyal_cve_busi \
  --manifest_path results/evaluation_manifests/busi_balanced_5_per_class_second_best.json
```

The implementation follows the greedy feature-cell replacement baseline from
[Goyal et al.](https://github.com/facebookresearch/visual-counterfactuals),
using a nearest correctly classified target-class training image as the
distractor.

#### SEDC-T

```bash
PYTHONPATH=. python scripts/run_sedc_t_pytorch.py \
  --model_path models/busi_resnet18_pretrained.pth \
  --dataset_path data/processed/BUSI \
  --output_dir results/final/sedc_t_busi_original_style \
  --manifest_path results/evaluation_manifests/busi_balanced_5_per_class_second_best.json \
  --search_timeout_seconds 30
```

The retained Pneumonia ROI configuration uses the same search and replacement
mechanism but restricts candidate segments to a geometric lung-field region:

```bash
PYTHONPATH=. python scripts/run_sedc_t_pytorch.py \
  --model_path models/pneumonia_resnet18_pretrained.pth \
  --dataset_path data/processed/Pneumonia \
  --output_dir results/final/sedc_t_pneumonia_lung_field_roi \
  --manifest_path results/evaluation_manifests/pneumonia_balanced_10_per_class_second_best.json \
  --roi_mode lung_fields \
  --search_timeout_seconds 30
```

This ROI configuration is a project-specific ablation, not part of the
original SEDC-T method and not a medical lung segmentation.

#### DVCE

The original-faithful configuration for the non-robust ResNet18 uses Cone
Projection with a robust PGD ResNet18 as the second classifier:

```bash
PYTHONPATH=. python scripts/run_dvce_pytorch.py \
  --model_path models/busi_resnet18_pretrained.pth \
  --second_model_path models/busi_resnet18_robust_pgd.pth \
  --dataset_path data/processed/BUSI \
  --output_dir results/final/dvce_original_style_cone/openai/busi \
  --manifest_path results/evaluation_manifests/busi_balanced_5_per_class_second_best.json \
  --run_generation \
  --device auto \
  --timestep_respacing 200 \
  --skip_timesteps 100 \
  --classifier_lambda 0.1 \
  --lp_custom 1.0 \
  --lp_custom_value 0.15 \
  --denoise_dist_input \
  --no-clip_denoised \
  --deg_cone_projection 30 \
  --aug_num 16 \
  --diffusion_checkpoint_path external/DVCEs/checkpoints/256x256_diffusion_uncond.pt
```

See the [DVCE method documentation](results/final_configs/dvce_method_documentation.md),
[Cone Projection notes](results/final_configs/dvce_cone_projection.md), and
[complete run commands](results/final_configs/dvce_original_style_commands.md).

### 4. Summarize Results

[`summarize_counterfactual_evaluation.py`](scripts/summarize_counterfactual_evaluation.py)
builds compact comparison tables from the final metadata files.
[`select_interpretable_examples.py`](scripts/select_interpretable_examples.py)
and
[`create_qualitative_comparison_figures.py`](scripts/create_qualitative_comparison_figures.py)
support the qualitative analysis.

## Reproducibility Notes

- Fixed manifests keep samples and target classes identical across methods.
- Final metadata and Markdown summaries are versioned; large image outputs are
  regenerated locally and excluded from Git.
- Scripts generally overwrite the path passed through `--output_path` or
  `--output_dir`. Use a new output directory for exploratory runs that must be
  preserved.
- Random seeds and method-specific settings are recorded in the generated
  metadata where applicable.
