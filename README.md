# Counterfactual Explanations for Medical Image Classification

This repository contains the implementation and evaluation artifacts for a
seminar project on counterfactual explanations for medical images. It covers
two classification tasks:

- BUSI breast ultrasound with the classes `benign`, `malignant`, and `normal`
- Chest X-ray pneumonia classification with `NORMAL` and `PNEUMONIA`

The project compares CFProto, Counterfactual Visual Explanations by Goyal et
al. (2019), SEDC-T, and DVCE. Every method uses the same fixed set of correctly
classified test images and the same target classes.

A counterfactual is valid when the classifier predicts the requested target
class after the change. Model validity does not establish medical plausibility
or clinical relevance.

## Main Results

### Baseline Classifiers

| Dataset | Model | Accuracy | Weighted F1 |
| --- | --- | ---: | ---: |
| BUSI | ImageNet-pretrained ResNet18 | 0.8390 | 0.8365 |
| Pneumonia | ImageNet-pretrained ResNet18 | 0.8782 | 0.8732 |

### Fixed Counterfactual Evaluation

| Method | Dataset | n | Validity | Mean CF confidence | Mean change fraction | Mean runtime |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| CFProto | BUSI | 15 | 0.8667 | 0.6815 | 0.0529 | 46.10 s |
| CFProto | Pneumonia | 20 | 1.0000 | 0.5740 | 0.0180 | 46.34 s |
| Goyal-CVE | BUSI | 15 | 1.0000 | 0.5279 | 0.2596 | 0.25 s |
| Goyal-CVE | Pneumonia | 20 | 1.0000 | 0.5231 | 0.3072 | 0.17 s |
| SEDC-T | BUSI | 15 | 0.8000 | 0.6343 | 0.2640 | 6.71 s |
| SEDC-T | Pneumonia | 20 | 0.5500 | 0.6759 | 0.3270 | 13.92 s |
| SEDC-T lung-field ROI | Pneumonia | 20 | 0.5000 | 0.7770 | 0.1745 | 15.23 s |
| DVCE Cone, OpenAI checkpoint | BUSI | 15 | 0.9333 | 0.9439 | 0.1157 | 1173.45 s |
| DVCE Cone, OpenAI checkpoint | Pneumonia | 20 | 0.8000 | 0.8369 | 0.0510 | 700.15 s |
| DVCE Cone, medically fine-tuned | BUSI | 15 | 1.0000 | 0.9984 | 0.1565 | 44.62 s |
| DVCE Cone, medically fine-tuned | Pneumonia | 20 | 1.0000 | 0.9800 | 0.0669 | 44.94 s |

The change fraction is method-dependent. CFProto and Goyal-CVE report the
pixel fraction above 0.03, DVCE reports the pixel fraction above 0.05, and
SEDC-T reports the image area covered by the selected segment masks.

DVCE runtimes are meaningful only under comparable execution conditions. The
OpenAI runs used MPS for BUSI and CUDA for Pneumonia without diffusion FP16.
The medically fine-tuned Cone runs used CUDA with diffusion FP16.

The complete table, including ablations, is available in
[results/docs/fixed_evaluation_summary.md](results/docs/fixed_evaluation_summary.md).

## Repository Layout

```text
.
├── src/                         reusable models and shared utilities
├── scripts/                     data preparation, training, and evaluation
├── results/
│   ├── baseline_classifiers/    classifier metrics
│   ├── docs/                    result and method documentation
│   ├── evaluation_manifests/    fixed samples and target classes
│   ├── final/                   metadata from final method runs
│   ├── final_configs/           authoritative method configurations
│   └── qualitative_figures/     qualitative selection and interpretation
├── requirements.txt
└── requirements-dvce.txt
```

Datasets, model weights, external repositories, and generated images are not
versioned because of size and licensing constraints. Compact metadata, fixed
evaluation manifests, and all project scripts remain in the repository.

## Installation

Create the core environment separately from the DVCE environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For DVCE:

```bash
python -m venv .venv-dvce
source .venv-dvce/bin/activate
python -m pip install -r requirements.txt
python -m pip install -r requirements-dvce.txt
```

The local DVCE port was run with a newer PyTorch environment. The original
DVCE repository targeted Python 3.8, PyTorch 1.9, and CUDA 10.2, so
compatibility depends on the operating system, CUDA version, and checkpoints.

## Data

The preparation scripts expect raw data at:

```text
data/raw/BUSI/Dataset_BUSI_with_GT/
data/raw/Pneumonia/chest_xray/
```

After preparation, both datasets use a torchvision `ImageFolder` layout:

```text
data/processed/BUSI/{train,val,test}/{class_name}/...
data/processed/Pneumonia/{train,val,test}/{class_name}/...
```

```bash
PYTHONPATH=. python scripts/prepare_busi.py
PYTHONPATH=. python scripts/prepare_pneumonia.py
PYTHONPATH=. python scripts/check_dataset.py
```

Dataset acquisition and use remain subject to the original license and access
conditions.

## Training and Evaluation

Run all commands from the repository root with `PYTHONPATH=.`.

### Classifiers

```bash
PYTHONPATH=. python src/train_model.py \
  --dataset_name BUSI_pretrained \
  --dataset_path data/processed/BUSI \
  --output_model_path models/busi_resnet18_pretrained.pth \
  --history_path results/baseline_classifiers/busi_pretrained_training_history.json \
  --epochs 15 \
  --batch_size 16 \
  --learning_rate 0.0001 \
  --pretrained
```

```bash
PYTHONPATH=. python src/train_model.py \
  --dataset_name Pneumonia_pretrained \
  --dataset_path data/processed/Pneumonia \
  --output_model_path models/pneumonia_resnet18_pretrained.pth \
  --history_path results/baseline_classifiers/pneumonia_pretrained_training_history.json \
  --epochs 10 \
  --batch_size 16 \
  --learning_rate 0.0001 \
  --pretrained
```

Use `scripts/evaluate_model.py` for test evaluation. The PGD-robust ResNet18
models used by DVCE Cone Projection are trained with
`scripts/train_robust_resnet18_pgd.py`.

### Fixed Evaluation Samples

The committed manifests contain 15 BUSI images, balanced across three classes,
and 20 Pneumonia images, balanced across two classes:

- [BUSI manifest](results/evaluation_manifests/busi_balanced_5_per_class_second_best.json)
- [Pneumonia manifest](results/evaluation_manifests/pneumonia_balanced_10_per_class_second_best.json)

The target is the class with the second-highest original probability.

### Counterfactual Methods

Final parameters and complete command examples are documented in
[results/final_configs/README.md](results/final_configs/README.md). The method
entry points are:

- `scripts/run_cfproto_pytorch.py`
- `scripts/run_goyal_cve_pytorch.py`
- `scripts/run_sedc_t_pytorch.py`
- `scripts/run_dvce_pytorch.py`

### Result Summaries and Qualitative Figures

`scripts/summarize_counterfactual_evaluation.py` builds the quantitative table
from final metadata. `scripts/select_interpretable_examples.py` applies fixed
selection rules, and `scripts/create_qualitative_comparison_figures.py`
creates the 14 method- and dataset-specific figures.

Generated PNG files are excluded from version control. The fixed selection is
stored in
[results/qualitative_figures/selected_examples.json](results/qualitative_figures/selected_examples.json),
and the sample-to-figure mapping is stored in
[results/qualitative_figures/qualitative_figure_manifest.json](results/qualitative_figures/qualitative_figure_manifest.json).

## Documentation

- [Script overview](scripts/README.md)
- [Method comparison and results](results/docs/method_comparison_and_results.md)
- [Implementation fidelity](results/docs/method_fidelity_comparison.md)
- [Baseline classifiers](results/baseline_classifiers/README.md)
- [Final configurations](results/final_configs/README.md)
- [Qualitative results](results/qualitative_figures/qualitative_results_interpretation.md)
