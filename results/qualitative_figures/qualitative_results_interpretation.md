# Qualitative Results Interpretation

This document explains the qualitative per-method figures in
`results/qualitative_figures/per_method/`.

## Retained Figures

| Dataset | Method | Figure |
| --- | --- | --- |
| BUSI | CFProto-nearer prototype-guided optimization baseline | `per_method/busi/cfproto_nearer_prototype_guided_optimization_baseline.png` |
| BUSI | Goyal 2019 counterfactual visual explanations | `per_method/busi/goyal_2019_counterfactual_visual_explanations.png` |
| BUSI | SEDC-T original-style best-first | `per_method/busi/sedc_t_original_style_best_first.png` |
| Pneumonia | CFProto-nearer prototype-guided optimization baseline | `per_method/pneumonia/cfproto_nearer_prototype_guided_optimization_baseline.png` |
| Pneumonia | Goyal 2019 counterfactual visual explanations | `per_method/pneumonia/goyal_2019_counterfactual_visual_explanations.png` |
| Pneumonia | SEDC-T original-style best-first | `per_method/pneumonia/sedc_t_original_style_best_first.png` |
| Pneumonia | SEDC-T lung-field ROI ablation | `per_method/pneumonia/sedc_t_lung_field_roi_ablation.png` |

## CFProto-Nearer Prototype-Guided Optimization

The retained prototype-guided method uses encoder-space target-class kNN
prototypes, adaptive c-search, elastic-net selection, polynomial learning-rate
decay, and a targeted margin loss. Its qualitative figures should be discussed
as model-behavior explanations, not as medical causal edits.

On BUSI the method is valid on all fixed samples. On Pneumonia it is valid for
18 of 20 fixed samples, so the Pneumonia figure can include a failure case. The
changes are small under the fixed difference scale. This is important:
low-contrast difference maps do not mean the plotting is broken; they indicate
that the method found small perturbations according to the recorded pixel-scale
metrics.

The bottleneck256 and bottleneck1024 CFProto variants are currently documented
as quantitative ablations. They are not included as retained qualitative
per-method figures because they are not the main prototype-guided configuration.

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
| CFProto-nearer optimization | compact model-valid changes | can be visually subtle and not medically causal |
| Goyal 2019 CVE | sparse localized edits grounded in real cases | coarse 7x7 cell grid; confidence near decision boundary |
| SEDC-T | localized segment changes | lower validity, especially on Pneumonia |
| DVCE | generative counterfactual direction | artifact- and checkpoint-sensitive |

Across all methods, model validity and medical plausibility must be discussed
separately.
