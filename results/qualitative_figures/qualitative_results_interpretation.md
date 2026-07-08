# Qualitative Results Interpretation

This document explains the qualitative per-method figures in
`results/qualitative_figures/per_method/`.

## Retained Figures

| Dataset | Method | Figure |
| --- | --- | --- |
| BUSI | CFProto-nearer prototype-guided optimization baseline | `per_method/busi/cfproto_nearer_prototype_guided_optimization_baseline.png` |
| BUSI | Retrieval-based nearest-unlike-neighbor baseline | `per_method/busi/retrieval_based_nearest_unlike_neighbor_baseline.png` |
| BUSI | SEDC-T original-style best-first | `per_method/busi/sedc_t_original_style_best_first.png` |
| BUSI | SEDC-T project variant | `per_method/busi/sedc_t_project_variant.png` |
| BUSI | SEDC-T tuned project variant | `per_method/busi/sedc_t_tuned_project_variant_none_max_10.png` |
| BUSI | DVCE-style OpenAI checkpoint | `per_method/busi/dvce_style_openai_checkpoint.png` |
| Pneumonia | CFProto-nearer prototype-guided optimization baseline | `per_method/pneumonia/cfproto_nearer_prototype_guided_optimization_baseline.png` |
| Pneumonia | Retrieval-based nearest-unlike-neighbor baseline | `per_method/pneumonia/retrieval_based_nearest_unlike_neighbor_baseline.png` |
| Pneumonia | SEDC-T original-style best-first | `per_method/pneumonia/sedc_t_original_style_best_first.png` |
| Pneumonia | SEDC-T project variant with lung-field ROI | `per_method/pneumonia/sedc_t_project_variant_lung_fields.png` |
| Pneumonia | SEDC-T tuned project variant with lung-field ROI | `per_method/pneumonia/sedc_t_tuned_project_variant_lung_fields_max_10.png` |
| Pneumonia | SEDC-T tuned project variant without ROI | `per_method/pneumonia/sedc_t_tuned_project_variant_none_max_8.png` |
| Pneumonia | DVCE-style OpenAI checkpoint | `per_method/pneumonia/dvce_style_openai_checkpoint.png` |
| Pneumonia | DVCE-style Pneumonia fine-tuned checkpoint | `per_method/pneumonia/dvce_style_pneumonia_fine_tuned_checkpoint.png` |

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

## Retrieval-NUN

Retrieval-NUN shows the nearest real target-class training case. The method is
easy to interpret as case-based comparison and avoids generated artifacts, but
the difference map is not a minimal edit. Large changed areas are expected
because two different patient images are compared.

## SEDC-T

SEDC-T figures are the most localized because the method changes image
segments. The original-style best-first figures are the method-faithfulness
reference. Project and tuned variants help discuss runtime, ROI constraints,
and whether Pneumonia failure cases are caused by parameter choice. Pneumonia
remains difficult, which supports the interpretation that the classifier may
use broad or diffuse cues that are not easily flipped by a small number of
segment replacements.

## DVCE

DVCE figures represent the generative direction. The OpenAI checkpoint and the
Pneumonia fine-tuned checkpoint should be interpreted as feasibility states.
They can produce model-valid target-class changes on the small fixed subset,
but outputs can contain visible artifacts and depend strongly on checkpoint and
guidance parameters.

## Overall Reading

The qualitative comparison should emphasize trade-offs:

| Method | Strength | Limitation |
| --- | --- | --- |
| CFProto-nearer optimization | compact model-valid changes | can be visually subtle and not medically causal |
| Retrieval-NUN | real target-class examples | not a minimal edit |
| SEDC-T | localized segment changes | lower validity, especially on Pneumonia |
| DVCE | generative counterfactual direction | artifact- and checkpoint-sensitive |

Across all methods, model validity and medical plausibility must be discussed
separately.
