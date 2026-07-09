# Counterfactual Method Comparison

## Compared Methods

This project compares four counterfactual explanation directions for medical
image classification:

1. CFProto (original-style) prototype-guided optimization
2. Goyal et al. 2019 counterfactual visual explanations (feature-cell swaps)
3. SEDC-T-style segment replacement
4. DVCE original-style diffusion-guided generation

All methods use the full fixed evaluation manifests (BUSI 15, Pneumonia 20).
DVCE is more expensive per sample (diffusion sampling), but is now run on the
full manifests with the original-code-nearer core; its original-faithful
variant for the non-robust ResNet18 is Cone Projection.

## Quantitative Results

| Method | Dataset | Samples | Validity | Mean CF confidence | Mean change | Mean runtime |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| CFProto (original-style) prototype-guided optimization | BUSI | 15 | 0.87 | 0.6815 | 0.0529 changed pixel fraction | 46.10s |
| CFProto (original-style) prototype-guided optimization | Pneumonia | 20 | 1.00 | 0.5740 | 0.0180 changed pixel fraction | 46.34s |
| Goyal et al. 2019 counterfactual visual explanations | BUSI | 15 | 1.00 | 0.5279 | 0.2596 changed pixel fraction, 14.0 edits | 0.25s |
| Goyal et al. 2019 counterfactual visual explanations | Pneumonia | 20 | 1.00 | 0.5231 | 0.3072 changed pixel fraction, 16.15 edits | 0.17s |
| SEDC-T original-style best-first | BUSI | 15 | 0.80 | 0.6343 | 0.2640 changed pixel fraction | 6.71s |
| SEDC-T original-style best-first | Pneumonia | 20 | 0.55 | 0.6759 | 0.3270 changed pixel fraction | 13.92s |
| SEDC-T lung-field ROI ablation | Pneumonia | 20 | 0.50 | 0.7770 | 0.1745 changed pixel fraction | 15.23s |
| DVCE Cone (OpenAI, original-faithful) | BUSI | 15 | 0.93 | 0.944 | 0.116 changed pixel fraction | 1173.4s |
| DVCE Cone (OpenAI, original-faithful) | Pneumonia | 20 | 0.80 | 0.837 | 0.051 changed pixel fraction | 700.2s |
| DVCE Cone (fine-tuned checkpoint) | BUSI | 15 | 1.00 | 0.998 | 0.156 changed pixel fraction | 44.6s |
| DVCE Cone (fine-tuned checkpoint) | Pneumonia | 20 | 1.00 | 0.980 | 0.067 changed pixel fraction | 44.9s |

DVCE no-cone ablation rows (deviation from the original for a non-robust model)
are documented in `results/final_configs/dvce_method_documentation.md`. The DVCE
OpenAI runtimes above reflect the (CPU-bound) machine they ran on and are not
comparable across machines.

The generated central summary is:

```text
results/docs/fixed_evaluation_summary.md
```

## Interpretation

CFProto (original-style) follows alibi's `CounterfactualProto` faithfully:
FISTA optimization with shrinkage-thresholding and Nesterov momentum, an
untargeted hinge attack loss, a sum-based loss combining attack, L2, autoencoder,
and prototype terms, binary c-search, and encoder-space class prototypes built
from the classifier's own predictions on the training split. `gamma`/`theta`
are recalibrated per dataset/autoencoder since all loss terms are sums and their
raw magnitude depends on input and latent dimensionality (0.87 validity on BUSI
with `theta=0.5`, 1.00 on Pneumonia with `theta=0.05`; the two BUSI misses stem
directly from the untargeted attack loss finding a valid flip to a class other
than the manifest's fixed target). Not reproduced: the TensorFlow graph itself
(reimplemented in PyTorch), black-box mode with numerical gradients, categorical
variables/k-d-tree prototypes, and TrustScore filtering (disabled by default in
alibi too). Full documentation:
`results/final_configs/cfproto_encoder_method_documentation.md`.

Goyal et al. 2019 CVE reaches 1.00 validity by construction: with the full
49-cell budget the pooled feature converges to the distractor's, so the
prediction is guaranteed to flip. The informative metric is the number of edited
feature cells (mean 14.0 on BUSI, 16.15 on Pneumonia of 49), i.e. how sparse the
swap is. Mean CF confidence sits near 0.5 because the greedy search stops at the
first flip, prioritizing sparsity over margin. The edits are grounded in a real
target-class distractor image and localized to a coarse 7x7 cell grid.

SEDC-T gives localized segment-level changes and is often easier to discuss
visually. Its validity is lower, especially on Pneumonia, where diffuse model
cues make segment replacement difficult.

DVCE covers the generative direction. The core matches the original
`dff_attack.py`: `p_sample`, classifier and distance guidance on `pred_xstart`
(unclamped `_map_img`), eps-norm rebalancing when `enforce_same_norms=True`, and
Cone Projection projecting the robust PGD classifier's gradient onto the cone
around the explained classifier's gradient. For the non-robust explained
ResNet18, Cone Projection is the original-faithful variant (no-cone is a marked
ablation). The medically fine-tuned diffusion checkpoints reach full validity
(1.00); the generic OpenAI checkpoint is lower (0.93 BUSI, 0.80 Pneumonia),
which is expected since it was trained on natural images, not medical scans.
Full method write-up: `results/final_configs/dvce_method_documentation.md`.

Validity means that the model prediction changed to the target class. It does
not imply medical plausibility or clinical causality.
