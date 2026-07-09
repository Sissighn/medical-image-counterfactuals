# Medical Image Counterfactuals

This repository contains code and result summaries for comparing
counterfactual explanation methods for medical image classification.

Datasets:

- BUSI breast ultrasound: `benign`, `malignant`, `normal`
- Chest X-ray pneumonia: `NORMAL`, `PNEUMONIA`

Workflow:

```text
prepare datasets -> train ResNet18 classifiers -> create fixed manifests -> generate counterfactuals -> evaluate methods
```

## Current Status

| Dataset | Model | Accuracy | Weighted F1 |
| --- | --- | ---: | ---: |
| BUSI | ResNet18 pretrained | 0.8390 | 0.8365 |
| Pneumonia | ResNet18 pretrained | 0.8782 | 0.8732 |

Retained counterfactual directions:

| Method | Role |
| --- | --- |
| CFProto (original-style) prototype-guided optimization | FISTA optimization with shrinkage-thresholding, untargeted hinge attack loss, encoder-space class prototypes, following alibi's `CounterfactualProto` |
| Goyal et al. 2019 counterfactual visual explanations | Instance-based greedy feature-cell swaps from a nearest-unlike distractor |
| SEDC-T-style segment replacement | Region-based/localized explanation |
| DVCE original-style diffusion-guided generation | Generative feasibility method; final rows need regeneration, including optional Cone Projection with robust second classifiers |

Central summaries:

```text
results/docs/fixed_evaluation_summary.md
results/docs/method_comparison.md
results/docs/final_method_summary.md
results/docs/method_variant_rationale.md
```

## Repository Structure

```text
src/        reusable data/model utilities
scripts/    preparation, training, counterfactual, and evaluation scripts
results/    compact JSON/Markdown summaries and fixed manifests
```

Large local assets are intentionally ignored by Git:

```text
data/
models/
external/
checkpoints/
results/debug/
results/ablations/
generated PNG/JPG result images
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

DVCE experiments may need:

```text
requirements-dvce.txt
```

## Data Format

Processed datasets are expected in ImageFolder format:

```text
data/processed/BUSI/{train,val,test}/...
data/processed/Pneumonia/{train,val,test}/...
```

Raw and processed datasets are not included in this repository.

## Training Classifiers

Example BUSI classifier:

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

Example Pneumonia classifier:

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

Robust ResNet18 checkpoints for DVCE Cone Projection can be trained with:

```text
scripts/train_robust_resnet18_pgd.py
```

## Fixed Evaluation Manifests

The comparison uses fixed correctly classified test samples and fixed target
classes:

```text
results/evaluation_manifests/busi_balanced_5_per_class_second_best.json
results/evaluation_manifests/pneumonia_balanced_10_per_class_second_best.json
```

Create a BUSI manifest:

```bash
PYTHONPATH=. python scripts/create_evaluation_manifest.py \
  --model_path models/busi_resnet18_pretrained.pth \
  --dataset_path data/processed/BUSI \
  --output_path results/evaluation_manifests/busi_balanced_5_per_class_second_best.json \
  --samples_per_class 5 \
  --target_strategy second_best
```

## Counterfactual Methods

### CFProto (Original-Style Prototype-Guided Optimization)

Final BUSI command:

```bash
PYTHONPATH=. python scripts/run_cfproto_pytorch.py \
  --model_path models/busi_resnet18_pretrained.pth \
  --dataset_path data/processed/BUSI \
  --output_dir results/final/cfproto_encoder_knn/busi \
  --manifest_path results/evaluation_manifests/busi_balanced_5_per_class_second_best.json \
  --autoencoder_path models/autoencoder_busi_bottleneck256.pth \
  --theta 0.5 --gamma 1.0 --c_steps 5 --prototype_k 3
```

Follows alibi's `CounterfactualProto` closely: FISTA optimization with
shrinkage-thresholding and Nesterov momentum, an untargeted hinge attack loss,
a sum-based loss `c*L_attack + L2 + beta*L1 + gamma*L_AE + theta*L_proto`,
binary search over the attack constant `c`, and encoder-space class prototypes
built from the classifier's own predictions on the training split. `theta` is
recalibrated per dataset/autoencoder since all loss terms are sums (see
`results/final_configs/cfproto_encoder_method_documentation.md`).

Not reproduced: the original TensorFlow graph itself (reimplemented in
PyTorch), black-box mode with numerical gradients, categorical variables and
k-d-tree prototypes, and TrustScore filtering (disabled by default in alibi
too). Full method documentation and Soll-Ist comparison:
`results/final_configs/cfproto_encoder_method_documentation.md`.

### Goyal et al. 2019 (Counterfactual Visual Explanations)

```bash
PYTHONPATH=. python scripts/run_goyal_cve_pytorch.py \
  --model_path models/busi_resnet18_pretrained.pth \
  --dataset_path data/processed/BUSI \
  --output_dir results/final/goyal_cve_busi \
  --manifest_path results/evaluation_manifests/busi_balanced_5_per_class_second_best.json
```

Instance-based method after Goyal et al. (ICML 2019, arXiv:1904.07451): a
nearest-unlike-neighbor distractor image from the target class is retrieved,
then spatial cells of the query's ResNet18 layer4 feature map are greedily
replaced by distractor cells (each cell at most once, permutation constraint)
until the prediction flips to the target class. Reference implementation of
the baseline: https://github.com/facebookresearch/visual-counterfactuals.

### SEDC-T

```bash
PYTHONPATH=. python scripts/run_sedc_t_pytorch.py \
  --model_path models/busi_resnet18_pretrained.pth \
  --dataset_path data/processed/BUSI \
  --output_dir results/final/sedc_t_busi_original_style \
  --manifest_path results/evaluation_manifests/busi_balanced_5_per_class_second_best.json \
  --search_timeout_seconds 30
```

For the retained Pneumonia ROI ablation, use the same SEDC-T search and
replacement mechanism but restrict candidate segments to the lung-field ROI:

```bash
PYTHONPATH=. python scripts/run_sedc_t_pytorch.py \
  --model_path models/pneumonia_resnet18_pretrained.pth \
  --dataset_path data/processed/Pneumonia \
  --output_dir results/final/sedc_t_pneumonia_lung_field_roi \
  --manifest_path results/evaluation_manifests/pneumonia_balanced_10_per_class_second_best.json \
  --roi_mode lung_fields \
  --search_timeout_seconds 30
```

Full run commands and method fidelity documentation are in:

```text
results/final_configs/sedc_t_run_commands.md
results/final_configs/sedc_t_method_documentation.md
```

### DVCE Original-Style Generation

```bash
PYTHONPATH=. python scripts/run_dvce_medical_prototype.py \
  --model_path models/busi_resnet18_pretrained.pth \
  --dataset_path data/processed/BUSI \
  --output_dir results/final/dvce_original_style/openai/busi \
  --manifest_path results/evaluation_manifests/busi_balanced_5_per_class_second_best.json \
  --run_generation \
  --device cuda \
  --timestep_respacing 200 \
  --skip_timesteps 100 \
  --classifier_lambda 0.1 \
  --lp_custom 1.0 \
  --lp_custom_value 0.15 \
  --denoise_dist_input \
  --no-clip_denoised \
  --diffusion_checkpoint_path external/DVCEs/checkpoints/256x256_diffusion_uncond.pt
```

Additional OpenAI/Pneumonia-medical/BUSI-medical checkpoint commands are stored in:

```text
results/final_configs/dvce_original_style_commands.md
```

Cone Projection setup, robust classifier training commands, smoke tests, and
full fixed-manifest DVCE commands for the Uni GPU are stored in:

```text
results/final_configs/dvce_cone_projection.md
```

## Current Fixed Evaluation Results

| Method | Dataset | Samples | Validity | Mean CF confidence | Mean change | Runtime |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| CFProto (original-style) | BUSI | 15 | 0.87 | 0.6815 | 0.0529 | 46.10s |
| CFProto (original-style) | Pneumonia | 20 | 1.00 | 0.5740 | 0.0180 | 46.34s |
| Goyal et al. 2019 CVE | BUSI | 15 | 1.00 | 0.5279 | 0.2596 | 0.25s |
| Goyal et al. 2019 CVE | Pneumonia | 20 | 1.00 | 0.5231 | 0.3072 | 0.17s |
| SEDC-T original-style best-first | BUSI | 15 | 0.80 | 0.6343 | 0.2640 | 6.71s |
| SEDC-T original-style best-first | Pneumonia | 20 | 0.55 | 0.6759 | 0.3270 | 13.92s |
| SEDC-T lung-field ROI ablation | Pneumonia | 20 | 0.50 | 0.7770 | 0.1745 | 15.23s |

DVCE rows from the earlier free-guidance prototype were removed. New DVCE
original-style fixed-manifest results should be generated with the commands in
`results/final_configs/dvce_original_style_commands.md` and, for Cone
Projection, `results/final_configs/dvce_cone_projection.md`.

Validity means target-class model prediction. It does not imply medical
plausibility or clinical causality.
