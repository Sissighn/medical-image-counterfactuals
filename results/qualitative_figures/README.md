# Qualitative Comparison Figures

This folder contains paper-friendly qualitative comparison figures for the counterfactual methods evaluated in this project.

The main figures are stored in `per_method/`. Each figure contains one method on one dataset. Rows correspond to qualitative case types:

- Most balanced valid case
- Highest-confidence valid case
- Low-plausibility valid case
- Failure case

The figures are composed from the existing per-example visualizations referenced in:

- `results/meeting_paul_tuesday/selected_examples.json`

The comparison script does not recompute, stretch, or per-image normalize the embedded difference maps. Source plots are displayed as saved, and image data is only converted to the standard display range `[0, 1]`.

Generated per-method figures:

- `results/qualitative_figures/per_method/busi__prototype_guided_optimization_baseline.png`
- `results/qualitative_figures/per_method/busi__prototype_guided_plausibility_ablation.png`
- `results/qualitative_figures/per_method/busi__sedc_t_original_style_best_first.png`
- `results/qualitative_figures/per_method/busi__dvce_style_openai_checkpoint.png`
- `results/qualitative_figures/per_method/pneumonia__prototype_guided_optimization_baseline.png`
- `results/qualitative_figures/per_method/pneumonia__prototype_guided_plausibility_ablation.png`
- `results/qualitative_figures/per_method/pneumonia__sedc_t_original_style_best_first.png`
- `results/qualitative_figures/per_method/pneumonia__sedc_t_tuned_project_variant_none_max_8.png`
- `results/qualitative_figures/per_method/pneumonia__dvce_style_openai_checkpoint.png`
- `results/qualitative_figures/per_method/pneumonia__dvce_style_pneumonia_fine_tuned_checkpoint.png`

Warnings emitted during generation: 5

## Warnings

- BUSI / Prototype-guided optimization baseline / Failure case: no selected example available.
- BUSI / Prototype-guided plausibility ablation / Failure case: no selected example available.
- BUSI / DVCE-style, OpenAI checkpoint / Failure case: no selected example available.
- Pneumonia / Prototype-guided optimization baseline / Failure case: no selected example available.
- Pneumonia / Prototype-guided plausibility ablation / Failure case: no selected example available.
