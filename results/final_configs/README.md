# Final Method Configurations

This document describes the configurations used for the shared evaluation.
The corresponding `metadata.json` files under `results/final/` remain
authoritative for individual measurements.

Run all commands from the repository root with `PYTHONPATH=.`.

## Detailed Method Documentation

- [CFProto](cfproto_encoder_method_documentation.md)
- [Goyal-CVE](goyal_cve_method_documentation.md)
- [SEDC-T](sedc_t_method_documentation.md)
- [DVCE](dvce_method_documentation.md)
- [DVCE Cone Projection and robust-classifier workflow](dvce_cone_projection.md)

## CFProto

CFProto optimizes the input in `[0, 1]` pixel space. Its objective combines
classification loss, L1/L2 proximity, autoencoder reconstruction, and distance
to a prototype in encoder space. Optimization uses FISTA,
shrinkage-thresholding, and a search over the attack constant `c`.

| Parameter | BUSI | Pneumonia |
| --- | ---: | ---: |
| Iterations per `c` step | 1000 | 1000 |
| Learning rate | 0.01 | 0.01 |
| `beta` | 0.1 | 0.1 |
| `gamma` | 1.0 | 1.0 |
| `theta` | 0.5 | 0.05 |
| `c_init` | 1.0 | 1.0 |
| `c_steps` | 5 | 5 |
| Prototype neighbors | 3 | 3 |
| Prototype mode | kNN mean | kNN mean |

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

```bash
PYTHONPATH=. python scripts/run_cfproto_pytorch.py \
  --model_path models/pneumonia_resnet18_pretrained.pth \
  --dataset_path data/processed/Pneumonia \
  --output_dir results/final/cfproto_encoder_knn/pneumonia \
  --manifest_path results/evaluation_manifests/pneumonia_balanced_10_per_class_second_best.json \
  --autoencoder_path models/autoencoder_pneumonia_bottleneck256.pth \
  --theta 0.05 \
  --gamma 1.0 \
  --c_steps 5 \
  --prototype_k 3
```

The TensorFlow graph, black-box numerical gradients, categorical variables,
k-d-tree prototypes, and TrustScore filtering are not reproduced.

## Goyal-CVE

ResNet18 is split at the output of `layer4` into a spatial `512 × 7 × 7`
feature grid and the remaining classification head. At each iteration, the
unused query/distractor cell pair that maximizes target-class probability is
selected. Each query cell and each distractor cell can be used at most once.

The distractor is the nearest correctly classified training image from the
target class. Distance is measured as cosine distance between L2-normalized,
globally pooled `layer4` features. This retrieval rule is project-specific;
the greedy feature-cell replacement is the core reference method.

```bash
PYTHONPATH=. python scripts/run_goyal_cve_pytorch.py \
  --model_path models/busi_resnet18_pretrained.pth \
  --dataset_path data/processed/BUSI \
  --manifest_path results/evaluation_manifests/busi_balanced_5_per_class_second_best.json \
  --output_dir results/final/goyal_cve_busi \
  --batch_size 32 \
  --change_threshold 0.03
```

For Pneumonia, replace the model, dataset, manifest, and output paths.

## SEDC-T

SEDC-T segments each image with Quickshift and searches segment combinations
in best-first order. Expansion is scored by target-class probability minus
original-class probability. Selected segments are replaced with a blurred
version of the image.

| Parameter | Value |
| --- | ---: |
| Quickshift `kernel_size` | 4 |
| Quickshift `max_dist` | 200 |
| Quickshift `ratio` | 0.2 |
| Blur kernel | 31 |
| Timeout per sample | 30 s |
| Search | original-style best-first |

```bash
PYTHONPATH=. python scripts/run_sedc_t_pytorch.py \
  --model_path models/busi_resnet18_pretrained.pth \
  --dataset_path data/processed/BUSI \
  --output_dir results/final/sedc_t_busi_original_style \
  --manifest_path results/evaluation_manifests/busi_balanced_5_per_class_second_best.json \
  --search_timeout_seconds 30
```

The main Pneumonia run uses the same parameters with `--roi_mode none`. The
additional ablation uses:

```bash
PYTHONPATH=. python scripts/run_sedc_t_pytorch.py \
  --model_path models/pneumonia_resnet18_pretrained.pth \
  --dataset_path data/processed/Pneumonia \
  --output_dir results/final/sedc_t_pneumonia_lung_field_roi \
  --manifest_path results/evaluation_manifests/pneumonia_balanced_10_per_class_second_best.json \
  --roi_mode lung_fields \
  --roi_min_overlap 0.5 \
  --search_timeout_seconds 30
```

The ROI is a fixed geometric approximation for frontal chest radiographs, not
a medical lung segmentation.

## DVCE

DVCE uses an unconditional diffusion model and guides denoising with the
medical classifier gradient and an Lp proximity term. The main configuration
adds Cone Projection with a PGD-robust ResNet18 as the second classifier. Runs
without Cone Projection are ablations.

### Shared Settings

| Parameter | Value |
| --- | ---: |
| Output size | 256 |
| `timestep_respacing` | 200 |
| Skipped timesteps | 100 |
| `classifier_lambda` | 0.1 |
| `lp_custom` | 1.0 |
| `lp_custom_value` | 0.15 |
| `denoise_dist_input` | enabled |
| `clip_denoised` | disabled |
| Sampling | `p_sample` |
| Seed | 1 |

### Cone Variant

| Parameter | Value |
| --- | ---: |
| Cone Projection angle | 30 degrees |
| Augmentations | 16 |
| Second classifier | PGD-robust ResNet18 |

```bash
PYTHONPATH=. python scripts/run_dvce_pytorch.py \
  --model_path models/busi_resnet18_pretrained.pth \
  --second_model_path models/busi_resnet18_robust_pgd.pth \
  --dataset_path data/processed/BUSI \
  --output_dir results/final/dvce_original_style_cone/openai/busi \
  --manifest_path results/evaluation_manifests/busi_balanced_5_per_class_second_best.json \
  --run_generation \
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

For medically fine-tuned runs, replace only the diffusion checkpoint:

```text
external/DVCEs/checkpoints/medical_diffusion_busi_ema.pt
external/DVCEs/checkpoints/medical_diffusion_pneumonia_ema.pt
```

The documented medically fine-tuned Cone runs used CUDA with
`--diffusion_fp16`. The OpenAI Cone runs used MPS for BUSI and CUDA for
Pneumonia, both without diffusion FP16.

### No-Cone Ablation

No-Cone runs omit `--second_model_path` and `--deg_cone_projection 30`;
`--aug_num` remains at its default value of 1. The medically fine-tuned runs
and the Pneumonia OpenAI run used CUDA with diffusion FP16.
