# DVCE Cone Projection — Local Run Commands

## Goal

Train one PGD-robust ResNet18 per dataset and use these robust classifiers as
second classifiers for DVCE Cone Projection. Everything below runs locally
(MPS); no jobs are routed to Paul's uni GPU for this method — see his
2026-07-09 reply: his offer only covers jobs that would take multiple DAYS
locally, not routine training/evaluation the team can run themselves.

Originality note (updated 2026-07-09): `src/dvce_core.py` now matches the
original `dff_attack.py` cone semantics — the **robust** classifier's gradient
is projected onto the cone (30°) centered at the **explained** classifier's
gradient, on flattened CPU tensors, exactly like the original
(`cone_projection(grad_robust, grad_explained, deg)`). The classifier input is
the unclamped `_map_img` mapping, the adapter resizes bicubic to 224 like the
original `ResizeAndMeanWrapper`, and augmentations run at classifier size 224.
This mirrors the original role split: original `classifier_type 3` =
robust Madry+FT ↔ our robust PGD ResNet18 (`--second_model_path`); original
`second_classifier_type 30/31` = explained non-robust Swin-T/ConvNeXt ↔ our
normal ResNet18 (`--model_path`).

The normal classifier remains the model being explained:

```text
models/pneumonia_resnet18_pretrained.pth
models/busi_resnet18_pretrained.pth
```

The robust classifiers are only used for Cone Projection:

```text
models/pneumonia_resnet18_robust_pgd.pth
models/busi_resnet18_robust_pgd.pth
```

The robust checkpoints must keep the existing checkpoint format:

```text
model_state_dict
num_classes
classes
class_to_idx
```

All required diffusion checkpoints already exist locally:

```text
external/DVCEs/checkpoints/256x256_diffusion_uncond.pt
external/DVCEs/checkpoints/medical_diffusion_pneumonia_ema.pt
external/DVCEs/checkpoints/medical_diffusion_busi_ema.pt
```

## Step 1 — Train the robust classifiers

Pneumonia first (smaller, 2 classes):

```bash
PYTHONPATH=. .venv/bin/python scripts/train_robust_resnet18_pgd.py \
  --dataset_name Pneumonia_robust_pgd \
  --dataset_path data/processed/Pneumonia \
  --output_model_path models/pneumonia_resnet18_robust_pgd.pth \
  --history_path results/baseline_classifiers/robust/pneumonia_resnet18_robust_pgd_history.json \
  --epochs 5 \
  --batch_size 16 \
  --learning_rate 1e-4 \
  --pretrained \
  --epsilon 0.03 \
  --step_size 0.007 \
  --pgd_steps 7 \
  --clean_loss_weight 0.5
```

BUSI:

```bash
PYTHONPATH=. .venv/bin/python scripts/train_robust_resnet18_pgd.py \
  --dataset_name BUSI_robust_pgd \
  --dataset_path data/processed/BUSI \
  --output_model_path models/busi_resnet18_robust_pgd.pth \
  --history_path results/baseline_classifiers/robust/busi_resnet18_robust_pgd_history.json \
  --epochs 5 \
  --batch_size 16 \
  --learning_rate 1e-4 \
  --pretrained \
  --epsilon 0.03 \
  --step_size 0.007 \
  --pgd_steps 7 \
  --clean_loss_weight 0.5
```

PGD adversarial training does 7 forward/backward passes per batch just for
the attack, so this is noticeably slower than the original clean-classifier
training — expect it to take a while on MPS. If either run looks like it will
run multiple days, that's the case to flag to Paul.

## Step 2 — Check robust classifier quality

Pneumonia:

```bash
PYTHONPATH=. .venv/bin/python scripts/evaluate_model.py \
  --model_path models/pneumonia_resnet18_robust_pgd.pth \
  --dataset_path data/processed/Pneumonia \
  --output_path results/baseline_classifiers/robust/pneumonia_resnet18_robust_pgd_test_eval.json
```

BUSI:

```bash
PYTHONPATH=. .venv/bin/python scripts/evaluate_model.py \
  --model_path models/busi_resnet18_robust_pgd.pth \
  --dataset_path data/processed/BUSI \
  --output_path results/baseline_classifiers/robust/busi_resnet18_robust_pgd_test_eval.json
```

Minimum requirement: the robust classifiers should still classify the medical
test set meaningfully. They do not need to outperform the normal ResNet18, but
they should not collapse.

## Step 3 — Cone projection smoke test (fast, 1 sample, few diffusion steps)

Pneumonia:

```bash
PYTHONPATH=. .venv/bin/python scripts/run_dvce_medical_prototype.py \
  --model_path models/pneumonia_resnet18_pretrained.pth \
  --second_model_path models/pneumonia_resnet18_robust_pgd.pth \
  --dataset_path data/processed/Pneumonia \
  --output_dir results/debug/dvce_cone_smoke/pneumonia \
  --manifest_path results/evaluation_manifests/pneumonia_balanced_10_per_class_second_best.json \
  --manifest_max_samples 1 \
  --run_generation \
  --device auto \
  --timestep_respacing 10 \
  --skip_timesteps 8 \
  --classifier_lambda 0.1 \
  --lp_custom 1.0 \
  --lp_custom_value 0.15 \
  --denoise_dist_input \
  --no-clip_denoised \
  --deg_cone_projection 30 \
  --aug_num 1
```

BUSI:

```bash
PYTHONPATH=. .venv/bin/python scripts/run_dvce_medical_prototype.py \
  --model_path models/busi_resnet18_pretrained.pth \
  --second_model_path models/busi_resnet18_robust_pgd.pth \
  --dataset_path data/processed/BUSI \
  --output_dir results/debug/dvce_cone_smoke/busi \
  --manifest_path results/evaluation_manifests/busi_balanced_5_per_class_second_best.json \
  --manifest_max_samples 1 \
  --run_generation \
  --device auto \
  --timestep_respacing 10 \
  --skip_timesteps 8 \
  --classifier_lambda 0.1 \
  --lp_custom 1.0 \
  --lp_custom_value 0.15 \
  --denoise_dist_input \
  --no-clip_denoised \
  --deg_cone_projection 30 \
  --aug_num 1
```

Expected metadata fields in `metadata.json`:

```text
cone_projection_enabled: true
deg_cone_projection: 30
second_classifier_adapter.model_path: models/..._robust_pgd.pth
guidance_space: pred_xstart
gen_type: p_sample
```

## Step 4 — Full fixed-manifest DVCE runs with Cone Projection

These use all fixed manifest samples: BUSI 15, Pneumonia 20.
`--aug_num 16` is the original readme's setting for cone runs; each
augmented forward/backward pass multiplies the per-step cost, so time this
on one sample first (Step 3 above, but with `--aug_num 16`) before committing
to the full manifest. If it is unrealistic on MPS, drop to `--aug_num 1` and
note the deviation in the results.

BUSI, OpenAI checkpoint:

```bash
PYTHONPATH=. .venv/bin/python scripts/run_dvce_medical_prototype.py \
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

BUSI, BUSI fine-tuned checkpoint:

```bash
PYTHONPATH=. .venv/bin/python scripts/run_dvce_medical_prototype.py \
  --model_path models/busi_resnet18_pretrained.pth \
  --second_model_path models/busi_resnet18_robust_pgd.pth \
  --dataset_path data/processed/BUSI \
  --output_dir results/final/dvce_original_style_cone/busi_medical_checkpoint/busi \
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
  --diffusion_checkpoint_path external/DVCEs/checkpoints/medical_diffusion_busi_ema.pt
```

Pneumonia, OpenAI checkpoint:

```bash
PYTHONPATH=. .venv/bin/python scripts/run_dvce_medical_prototype.py \
  --model_path models/pneumonia_resnet18_pretrained.pth \
  --second_model_path models/pneumonia_resnet18_robust_pgd.pth \
  --dataset_path data/processed/Pneumonia \
  --output_dir results/final/dvce_original_style_cone/openai/pneumonia \
  --manifest_path results/evaluation_manifests/pneumonia_balanced_10_per_class_second_best.json \
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

Pneumonia, Pneumonia fine-tuned checkpoint:

```bash
PYTHONPATH=. .venv/bin/python scripts/run_dvce_medical_prototype.py \
  --model_path models/pneumonia_resnet18_pretrained.pth \
  --second_model_path models/pneumonia_resnet18_robust_pgd.pth \
  --dataset_path data/processed/Pneumonia \
  --output_dir results/final/dvce_original_style_cone/pneumonia_medical_checkpoint/pneumonia \
  --manifest_path results/evaluation_manifests/pneumonia_balanced_10_per_class_second_best.json \
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
  --diffusion_checkpoint_path external/DVCEs/checkpoints/medical_diffusion_pneumonia_ema.pt
```

## After the strictly original no-cone runs

Once the robust checkpoints exist, also run the strictly original no-cone
variant (robust classifier as the explained model) — commands are in
`results/final_configs/dvce_original_style_commands.md`, section 4.
