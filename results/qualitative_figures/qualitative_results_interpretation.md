# Qualitative Results Interpretation

This document explains the qualitative per-method figures in
`results/qualitative_figures/per_method/`.

## Retained Figures

| Dataset | Method | Figure |
| --- | --- | --- |
| BUSI | CFProto (original-style, bottleneck256) | `per_method/busi/cfproto_original_style_bottleneck256.png` |
| BUSI | Goyal 2019 counterfactual visual explanations | `per_method/busi/goyal_2019_counterfactual_visual_explanations.png` |
| BUSI | SEDC-T original-style best-first | `per_method/busi/sedc_t_original_style_best_first.png` |
| Pneumonia | CFProto (original-style, bottleneck256) | `per_method/pneumonia/cfproto_original_style_bottleneck256.png` |
| Pneumonia | Goyal 2019 counterfactual visual explanations | `per_method/pneumonia/goyal_2019_counterfactual_visual_explanations.png` |
| Pneumonia | SEDC-T original-style best-first | `per_method/pneumonia/sedc_t_original_style_best_first.png` |
| Pneumonia | SEDC-T lung-field ROI ablation | `per_method/pneumonia/sedc_t_lung_field_roi_ablation.png` |

## CFProto (Original-Style Prototype-Guided Optimization)

The method follows alibi's `CounterfactualProto` faithfully: FISTA optimization
with shrinkage-thresholding, an untargeted hinge attack loss, encoder-space
class prototypes from classifier predictions (bottleneck-256 autoencoder),
binary c-search, and elastic-net selection. Its qualitative figures should be
discussed as model-behavior explanations, not as medical causal edits.

On BUSI the method is valid for 13 of 15 fixed samples, so the BUSI figure can
include a failure case; the two misses are a direct consequence of the
untargeted attack loss finding a confident flip to a class other than the
manifest's fixed target. On Pneumonia it is valid for all 20 of 20 fixed
samples. The changes are small under the fixed difference scale on both
datasets (more so on Pneumonia, where `theta` was calibrated an order of
magnitude smaller to match the autoencoder's larger raw prototype distances).
This is important: low-contrast difference maps do not mean the plotting is
broken; they indicate that the method found small perturbations according to
the recorded pixel-scale metrics.

## Goyal 2019 CVE

Goyal et al. 2019 CVE composites a counterfactual by swapping discriminative
spatial feature cells of the query for cells of a real target-class distractor
image. In the figures the edited cells are boxed on the original and the source
cells boxed on the distractor, so the swapped regions are directly visible. The
method is grounded in real cases and localized, but the edits are coarse (7x7
cell grid) and carry distractor anatomy/acquisition content. Validity is 1.00 by
construction and the number of edited cells is the sparsity signal; confidence
sits near the decision boundary because the search stops at the first flip.

## SEDC-T

SEDC-T figures are the most localized because the method changes image
segments. The original-style best-first figures are the method-faithfulness
reference. The only retained ablation is the Pneumonia lung-field ROI variant,
which uses the same best-first, Quickshift, and Gaussian-blur mechanism but
restricts candidate segments to a simple geometric lung-field mask. This ROI is
not a medical lung segmentation and is not part of the original SEDC-T method.
Pneumonia remains difficult, which supports the interpretation that the
classifier may use broad or diffuse cues that are not easily flipped by
segment replacement.

## DVCE

DVCE figures from the earlier free-guidance prototype have been removed. The
retained implementation now uses the original-code-nearer DVCE core:
`p_sample`, `pred_xstart` guidance, `classifier_lambda=0.1`, `lp_custom=1.0`,
`lp_custom_value=0.15`, and `enforce_same_norms=True`. New qualitative figures
should only be regenerated after the OpenAI, Pneumonia medical-checkpoint, and
BUSI medical-checkpoint states have been rerun with this core.

## Overall Reading

The qualitative comparison should emphasize trade-offs:

| Method | Strength | Limitation |
| --- | --- | --- |
| CFProto (original-style) | compact, mostly model-valid changes | can be visually subtle and not medically causal |
| Goyal 2019 CVE | sparse localized edits grounded in real cases | coarse 7x7 cell grid; confidence near decision boundary |
| SEDC-T | localized segment changes | lower validity, especially on Pneumonia |
| DVCE | generative counterfactual direction | artifact- and checkpoint-sensitive |

Across all methods, model validity and medical plausibility must be discussed
separately.
