# Qualitative Comparison Figures

This folder contains paper-friendly qualitative comparison figures for the counterfactual methods evaluated in this project.

The main figures are stored in dataset-specific folders under `per_method/`, for example `per_method/busi/` and `per_method/pneumonia/`. Each figure contains one method on one dataset. Rows correspond to the qualitative case types available for that method:

- Most balanced valid case
- Highest-confidence valid case
- Low-plausibility valid case
- Failure case

The current figure-by-figure interpretation, including the exact displayed sample metrics and the conclusions aligned with Chapter 5 of the seminar paper, is documented in:

- `results/qualitative_figures/qualitative_results_interpretation.md`

The figures are composed from the existing per-example visualizations referenced in:

- `results/qualitative_selection/selected_examples.json`

The selected sample indices are preserved exactly. Redundant source headers and the repeated method/dataset heading are removed. Each row keeps one compact annotation with the class transition and both the original and counterfactual confidence, target, validity, MAD (`l1_mean`), RMS pixel difference (`l2_mean`), changed-pixel percentage, and only method-specific details needed for interpretation (Goyal distractor/embedding distance/edit count or SEDC-T segment count). The unrelated maximum norm `L_inf` from the former annotations is not shown because the paper defines RMS/`l2_mean` as its second proximity metric.

The comparison script does not recompute, stretch, or per-image normalize the embedded difference maps. Source plots are displayed as saved, and image data is only converted to the standard display range `[0, 1]`. Long white bands inside source plots are compacted for readability, but the image panels and color values are not changed.

This is intentional: very dark difference maps indicate genuinely small absolute changes on a fixed scale, not a plotting error. Stronger colors would only be appropriate with an explicitly labelled alternate scale, because otherwise tiny differences could appear misleadingly large.

Generated per-method figures:

- `results/qualitative_figures/per_method/busi/cfproto_original_style_bottleneck256.png`
- `results/qualitative_figures/per_method/busi/goyal_2019_counterfactual_visual_explanations.png`
- `results/qualitative_figures/per_method/busi/sedc_t_original_style_best_first.png`
- `results/qualitative_figures/per_method/busi/dvce_original_style_with_cone_projection_openai_checkpoint.png`
- `results/qualitative_figures/per_method/busi/dvce_original_style_with_cone_projection_busi_fine_tuned_checkpoint.png`
- `results/qualitative_figures/per_method/busi/dvce_original_style_without_cone_projection_busi_fine_tuned_checkpoint.png`
- `results/qualitative_figures/per_method/pneumonia/cfproto_original_style_bottleneck256.png`
- `results/qualitative_figures/per_method/pneumonia/goyal_2019_counterfactual_visual_explanations.png`
- `results/qualitative_figures/per_method/pneumonia/sedc_t_original_style_best_first.png`
- `results/qualitative_figures/per_method/pneumonia/sedc_t_lung_field_roi_ablation.png`
- `results/qualitative_figures/per_method/pneumonia/dvce_original_style_without_cone_projection_openai_checkpoint.png`
- `results/qualitative_figures/per_method/pneumonia/dvce_original_style_with_cone_projection_openai_checkpoint.png`
- `results/qualitative_figures/per_method/pneumonia/dvce_original_style_with_cone_projection_pneumonia_fine_tuned_checkpoint.png`
- `results/qualitative_figures/per_method/pneumonia/dvce_original_style_without_cone_projection_pneumonia_fine_tuned_checkpoint.png`

Warnings emitted during generation: 0
