# DVCE Original-Style Run Commands

These commands use the original-code-nearer DVCE core in `src/dvce_core.py`.
The method variant is the no-cone baseline: `original_style_medical_no_cone`.
Only the diffusion checkpoint changes.

Originality status (2026-07-09): the core now matches the original
`dff_attack.py` in guidance space (`pred_xstart`), unclamped `_map_img`
classifier input, eps-norm rebalancing, cone-projection argument order
(robust gradient projected onto the cone around the explained classifier's
gradient), bicubic (interpolation=3) classifier resize at 224, and
augmentations at classifier size 224. Framework/classifier differences
(PyTorch versions, medical ResNet18 instead of Madry+FT/Swin-T/ConvNeXt)
are intentional.

Common settings (from the original `configs/default.yml` plus the
`--denoise_dist_input` flag that both published readme commands set):

```text
gen_type: p_sample
timestep_respacing: 200
skip_timesteps: 100
classifier_lambda: 0.1
lp_custom: 1.0
lp_custom_value: 0.15
enforce_same_norms: true
denoise_dist_input: true
clip_denoised: false
aug_num: 1
deg_cone_projection: 0
```

Important method note: in the original repo, the no-cone DVCE variant is
defined for a **robust** classifier (Madry + FT). Running no-cone DVCE on the
normal (non-robust) ResnNet18 is a deviation; the original explains non-robust
models only via cone projection. Once the PGD-robust checkpoints exist
(`models/*_resnet18_robust_pgd.pth`), the strictly original no-cone runs use
the robust model as `--model_path` (see the last section below). Keep the
normal-classifier no-cone runs only if you want them as an explicitly marked
ablation.

All commands below use `--device auto` (resolves to MPS locally). The
original always runs the diffusion UNet in fp16 (`use_fp16: True`); add
`--diffusion_fp16` only if these commands are ever run on a CUDA machine —
leave it off on MPS.

## 1. OpenAI Checkpoint

BUSI:

```bash
PYTHONPATH=. .venv/bin/python scripts/run_dvce_pytorch.py \
  --model_path models/busi_resnet18_pretrained.pth \
  --dataset_path data/processed/BUSI \
  --output_dir results/final/dvce_original_style/openai/busi \
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
  --diffusion_checkpoint_path external/DVCEs/checkpoints/256x256_diffusion_uncond.pt
```

Pneumonia:

```bash
PYTHONPATH=. .venv/bin/python scripts/run_dvce_pytorch.py \
  --model_path models/pneumonia_resnet18_pretrained.pth \
  --dataset_path data/processed/Pneumonia \
  --output_dir results/final/dvce_original_style/openai/pneumonia \
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
  --diffusion_checkpoint_path external/DVCEs/checkpoints/256x256_diffusion_uncond.pt
```

## 2. Pneumonia Fine-Tuned Medical Checkpoint

```bash
PYTHONPATH=. .venv/bin/python scripts/run_dvce_pytorch.py \
  --model_path models/pneumonia_resnet18_pretrained.pth \
  --dataset_path data/processed/Pneumonia \
  --output_dir results/final/dvce_original_style/pneumonia_medical_checkpoint/pneumonia \
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
  --diffusion_checkpoint_path external/DVCEs/checkpoints/medical_diffusion_pneumonia_ema.pt
```

## 3. BUSI Fine-Tuned Medical Checkpoint

```bash
PYTHONPATH=. .venv/bin/python scripts/run_dvce_pytorch.py \
  --model_path models/busi_resnet18_pretrained.pth \
  --dataset_path data/processed/BUSI \
  --output_dir results/final/dvce_original_style/busi_medical_checkpoint/busi \
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
  --diffusion_checkpoint_path external/DVCEs/checkpoints/medical_diffusion_busi_ema.pt
```

Interpretation note: these are checkpoint states of the same DVCE original-style method, not separate methods. Cone Projection remains disabled unless a second robust medical classifier is explicitly added.

Checkpoint naming note (resolved 2026-07-09): both fine-tuned EMA checkpoints
now exist under the documented names. `medical_diffusion_pneumonia_ema.pt` is
the renamed `ema_0.9999_005000.pt` (fine-tuning run from 2026-07-02),
`medical_diffusion_busi_ema.pt` is the renamed `ema_busi_0.9999_005000.pt`
(run from 2026-07-08). If this dataset mapping is wrong, only the two file
names need to be swapped. `model005000.pt`/`opt005000.pt` are raw
model/optimizer states from the first fine-tuning run and are not used by the
commands.

## 4. Strictly Original No-Cone Runs (robust classifier as explained model)

Only possible after the PGD-robust checkpoints exist. This mirrors the
original readme command `python imagenet_VCEs.py ... --denoise_dist_input`,
where the explained classifier itself is robust:

```bash
PYTHONPATH=. .venv/bin/python scripts/run_dvce_pytorch.py \
  --model_path models/pneumonia_resnet18_robust_pgd.pth \
  --dataset_path data/processed/Pneumonia \
  --output_dir results/final/dvce_original_style_robust/openai/pneumonia \
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
  --diffusion_checkpoint_path external/DVCEs/checkpoints/256x256_diffusion_uncond.pt
```

(BUSI analogously with `models/busi_resnet18_robust_pgd.pth`; for the
fine-tuned diffusion checkpoints swap `--diffusion_checkpoint_path`.)

For Cone Projection runs with robust second classifiers, use:

```text
results/final_configs/dvce_cone_projection.md
```
