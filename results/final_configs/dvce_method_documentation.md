# DVCE Method Documentation

Implementation:
[`src/dvce_core.py`](../../src/dvce_core.py) (guidance core) and
[`scripts/run_dvce_pytorch.py`](../../scripts/run_dvce_pytorch.py)
(runner/adapter).

Reference (original code): <https://github.com/valentyn1boreiko/DVCEs>
Ground-truth guidance file in the vendored original:
`external/DVCEs/blended_diffusion/optimization/dff_attack.py`
(`DiffusionAttack.cond_fn_clean`).

Paper: Augustin, Boreiko, Croce, Hein — *Diffusion Visual Counterfactual
Explanations*, NeurIPS 2022.

Run commands:
[`dvce_cone_projection.md`](dvce_cone_projection.md) (Cone variant, original-faithful) and
[`dvce_original_style_commands.md`](dvce_original_style_commands.md) (no-cone states).

---

## 1. What DVCE is

DVCE (Diffusion Visual Counterfactual Explanations) is a **generative**
counterfactual method. Instead of editing pixels or optimizing a mask directly,
it runs a **guided reverse diffusion process**: a pretrained unconditional
diffusion model denoises a noised version of the input image, and at every
denoising step a **classifier-gradient guidance term** nudges the image toward a
chosen **target class**, while a **distance term** keeps it close to the
original. The result is a counterfactual that lies on the diffusion model's
image manifold, so it tends to look like a plausible image rather than an
adversarial perturbation.

The central idea of the paper is that meaningful diffusion guidance needs
**robust classifier gradients**. Non-robust classifiers produce adversarial,
noise-like gradients. DVCE therefore uses **Cone Projection**: the (non-robust)
explained classifier's gradient is projected into a cone around a **robust**
classifier's gradient, keeping the explained model's direction only where it
already agrees with the robust one.

This project ports the original guidance logic to PyTorch and applies it to
medical ResNet-18 classifiers (BUSI ultrasound, Pneumonia X-ray). The guidance
mathematics follow the original; the framework and the classifier are the
intended project-specific substitutions.

---

## 2. How the implementation works, step by step

For one image, one target class:

1. **Load the diffusion backbone.** The unconditional 256×256 guided-diffusion
   UNet (`create_model_and_diffusion`) is built with the original DVCE config
   (`attention_resolutions="32,16,8"`, `learn_sigma=True`, `num_channels=256`,
   `num_head_channels=64`, `resblock_updown=True`, `use_scale_shift_norm=True`,
   `diffusion_steps=1000`, `rescale_timesteps=True`). Weights are loaded, the
   model is frozen and set to eval, then — exactly as in the original — the
   `qkv`/`norm`/`proj` parameters are re-enabled for gradients and fp16 is
   applied last. Two diffusion checkpoints are used:
   the **OpenAI** `256x256_diffusion_uncond.pt` (the original's checkpoint) and
   the **medically fine-tuned** EMA checkpoints per dataset.

2. **Prepare the image.** The input is resized to 256×256 and mapped from
   `[0,1]` to `[-1,1]` (`init_image`). This is the diffusion model's value
   range.

3. **Noise + skip.** The sampler noises `init_image` to the level of the
   starting timestep and runs the reverse process from there
   (`timestep_respacing=200`, `skip_timesteps=100`, i.e. it denoises the last
   100 of 200 respaced steps). The vendored `p_sample_loop_progressive`
   re-seeds from `seed` and draws its own noise (no explicit noise is passed),
   matching the original.

4. **Guided denoising (`cond_fn_clean`).** At each step the guidance function:
   - recomputes `p_mean_variance` internally and takes `x_in = pred_xstart`
     (the model's current estimate of the clean image);
   - maps `x_in` to `[0,1]` with the original unclamped `_map_img`
     (`0.5*(x+1)`, **no clamp**, so gradients survive for out-of-range pixels);
   - **classifier term:** evaluates the explained classifier on `x_in`,
     computes `log_softmax` for the target class, and takes the gradient
     w.r.t. `x`;
   - **Cone Projection (if a second, robust classifier is given and
     `deg_cone_projection>0`):** evaluates the robust classifier the same way,
     takes its gradient, and calls
     `cone_projection(grad_robust, grad_explained, deg)` on flattened CPU
     tensors — the robust gradient is projected onto the cone (default 30°)
     around the explained classifier's gradient;
   - **distance term:** an `lp_custom` distance (default L1) between `x_in` and
     `init_image`, either as an analytic sub-gradient or, with
     `--denoise_dist_input`, as an autograd gradient through the denoiser;
   - **eps-norm rebalancing (`enforce_same_norms`, default on):** the
     classifier gradient and the distance gradient are each renormalized to the
     norm of the diffusion model output `eps` before being combined, so neither
     term dominates purely by scale;
   - returns `classifier_lambda * grad_class − lp_custom_value * lp_grad`.
   The vendored `condition_mean` adds this to the step mean, scaled by the
   diffusion variance.

5. **Augmentations (Cone runs).** With `aug_num>1` (original readme uses
   `aug_num=16` for cone runs), each classifier is evaluated on
   `ImageAugmentations` copies at the **classifier size (224)** and the target
   log-confidence is averaged, matching the original. A fresh set of
   augmentations is drawn per classifier.

6. **Output.** After the loop, the final `pred_xstart` is mapped to `[0,1]`,
   clamped once, and returned as the counterfactual. The runner records the
   counterfactual image, absolute-difference map, an overlay visualization, the
   classifier's prediction/confidence before and after, validity
   (`prediction == target`), change metrics, and the full settings in
   `metadata.json`.

The classifier adapter (`MedicalResNetAdapter`) mirrors the original
`ResizeAndMeanWrapper`: it bicubically resizes to 224×224 (interpolation=3),
applies ImageNet normalization, and does **not** clamp its input.

---

## 3. Soll-Ist comparison with the original

| Aspect | Original (`dff_attack.py`) | This implementation | Status |
|---|---|---|---|
| Diffusion backbone + config | 256×256 guided-diffusion, specific config | identical config | ✅ |
| Model prep order | freeze → eval → to(device) → re-enable qkv/norm/proj grads → fp16 | identical | ✅ |
| Guidance space | `x_in = pred_xstart` | identical | ✅ |
| Classifier input mapping | `_map_img = 0.5*(x+1)`, unclamped | identical (unclamped) | ✅ |
| Classifier term | `log_softmax` target, grad w.r.t. x | identical | ✅ |
| Cone Projection order | `cone_projection(grad_1, grad_2)` = robust gradient projected onto cone around explained-model gradient, flattened on CPU | identical argument order and device handling | ✅ |
| Distance term | `lp_custom` on `x_in − init_image`, analytic or denoise-autograd | identical | ✅ |
| eps-norm rebalancing | `_renormalize_gradient(..., eps)` per term when `enforce_same_norms` | identical | ✅ |
| Combination | `classifier_lambda*grad_class − lp_custom_value*lp_grad` | identical | ✅ |
| Augmentations | `ImageAugmentations(clip_size=classifier_size, aug_num)`, per-classifier | identical (size 224) | ✅ |
| Sampler | `p_sample_loop_progressive`, seeded, skip_timesteps | identical (vendored file) | ✅ |
| Default params | `timestep_respacing=200`, `skip_timesteps=100`, `classifier_lambda=0.1`, `lp_custom=1.0`, `lp_custom_value=0.15`, `deg=30`, `aug_num=16` (cone) | identical (from `configs/default.yml` + readme) | ✅ |

---

## 4. Deliberate differences from the original (not fidelity problems)

1. **Framework and classifier.** PyTorch + medical ResNet-18 instead of the
   original's Madry+FT / Swin-T / ConvNeXt ImageNet classifiers. This is the
   intended project substitution.

2. **Robust second classifier.** The original's robust helper is the
   ImageNet Madry+FT model. Here it is a **PGD-adversarially trained ResNet-18**
   per dataset (`models/*_resnet18_robust_pgd.pth`,
   [`scripts/train_robust_resnet18_pgd.py`](../../scripts/train_robust_resnet18_pgd.py)),
   playing the exact same role (the gradient the cone is built around).

3. **Fine-tuned diffusion checkpoints.** In addition to the original OpenAI
   checkpoint, medically fine-tuned EMA checkpoints are evaluated as a
   **checkpoint ablation**. The OpenAI checkpoint is the original-faithful
   backbone; the fine-tuned ones test whether a medical prior improves the
   counterfactuals.

4. **No-cone on the normal classifier.** In the original, the no-cone variant
   is defined for a **robust** classifier. Running no-cone on the normal,
   non-robust ResNet-18 is therefore a **deviation** and is kept only as an
   explicitly marked **ablation**, not as an original-faithful run. The
   original-faithful configuration for the (non-robust) explained model is
   **Cone Projection**.

5. **MPS device workaround.** On Apple MPS, `adaptive_avg_pool` with
   non-divisible sizes (256→224) is unsupported, so the augmentation pooling is
   done via a CPU roundtrip that keeps the autograd graph intact. On CUDA it
   runs on-device unchanged. This is a device workaround only and does not
   change the method.

6. **`ImageAugmentations` loaded by file path.** The original package
   `__init__` pulls in heavy, unused dependencies (pytorch_msssim, lpips,
   tensorboard); the class is loaded directly from its source file to avoid
   those imports. The class itself is byte-identical to the original.

---

## 5. Key parameters

| Argument | Default | Meaning |
|---|---|---|
| `--diffusion_checkpoint_path` | OpenAI ckpt | Diffusion backbone (OpenAI or medical-ft EMA) |
| `--second_model_path` | — | Robust PGD ResNet-18 for Cone Projection (enables cone) |
| `--deg_cone_projection` | `0` | Cone half-angle in degrees; `30` = original cone; `0` = no cone |
| `--aug_num` | `1` | Augmentation count; `16` = original cone setting |
| `--classifier_lambda` | `0.1` | Classifier guidance weight |
| `--lp_custom` | `1.0` | Distance norm p (1 = L1) |
| `--lp_custom_value` | `0.15` | Distance guidance weight |
| `--enforce_same_norms` | on | eps-norm rebalancing of the two guidance terms |
| `--denoise_dist_input` | off (readme: on) | Distance gradient through the denoiser (both original readme commands set it) |
| `--timestep_respacing` | `200` | Respaced diffusion steps |
| `--skip_timesteps` | `100` | Steps skipped (start level of the reverse process) |
| `--clip_denoised` | off | Clamp `pred_xstart` during sampling (original: off) |
| `--diffusion_fp16` | off | fp16 diffusion UNet (CUDA; the original always uses fp16) |
| `--manifest_path` | — | Fixed evaluation manifest; samples and targets come from it |

---

## 6. Evaluation matrix and results

Two axes: **guidance variant** (Cone = original-faithful for the non-robust
ResNet-18; No-Cone = ablation) × **diffusion checkpoint** (OpenAI vs medically
fine-tuned). Full fixed manifests: BUSI 15 samples, Pneumonia 20 samples.
Balanced target directions (BUSI across its classes; Pneumonia 10×
NORMAL→PNEUMONIA and 10× PNEUMONIA→NORMAL).

| Variant | Checkpoint | Dataset | n | Validity | Mean CF conf. | Mean abs. diff | Changed px (>0.05) |
|---|---|---|---|---|---|---|---|
| **Cone** | OpenAI | BUSI | 15 | 0.93 | 0.944 | 0.024 | 0.116 |
| **Cone** | OpenAI | Pneumonia | 20 | 0.80 | 0.837 | 0.017 | 0.051 |
| **Cone** | BUSI-ft | BUSI | 15 | 1.00 | 0.998 | 0.028 | 0.156 |
| **Cone** | Pneu-ft | Pneumonia | 20 | 1.00 | 0.980 | 0.019 | 0.067 |
| No-Cone (abl.) | BUSI-ft | BUSI | 15 | 1.00 | 0.998 | 0.026 | 0.136 |
| No-Cone (abl.) | Pneu-ft | Pneumonia | 20 | 1.00 | 0.995 | 0.017 | 0.052 |
| No-Cone (abl.) | OpenAI | Pneumonia | 20 | 1.00 | 0.997 | 0.018 | 0.060 |

Reading:
- **Checkpoint effect:** the medically fine-tuned checkpoints reach full
  validity (1.00) and high confidence, while the generic OpenAI checkpoint is
  lower (0.93 BUSI, 0.80 Pneumonia) — forcing target-class counterfactuals is
  harder with a diffusion prior trained on natural images.
- **Original-faithful core:** Cone + OpenAI is the configuration closest to the
  original DVCE setup (original diffusion backbone + cone for a non-robust
  classifier). The fine-tuned Cone runs are the medical adaptation.
- The no-cone rows are ablations (deviation from the original for a non-robust
  model) and are labelled as such.

Result directories:
```text
results/final/dvce_original_style_cone/openai/{busi,pneumonia}/
results/final/dvce_original_style_cone/{busi,pneumonia}_medical_checkpoint/...
results/final/dvce_original_style/{busi,pneumonia}_medical_checkpoint/...   (no-cone ablation)
results/final/dvce_original_style/openai/pneumonia/                          (no-cone, extra)
```
