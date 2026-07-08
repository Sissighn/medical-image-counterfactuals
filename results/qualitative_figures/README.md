# Qualitative Comparison Figures

This folder contains paper-friendly qualitative comparison figures for the counterfactual methods evaluated in this project.

The main figures are stored in dataset-specific folders under `per_method/`, for example `per_method/busi/` and `per_method/pneumonia/`. Each figure contains one method on one dataset. Rows correspond to the qualitative case types available for that method:

- Most balanced valid case
- Highest-confidence valid case
- Low-plausibility valid case
- Failure case

The figures are composed from the existing per-example visualizations referenced in:

- `results/qualitative_selection/selected_examples.json`

The comparison script does not recompute, stretch, or per-image normalize the embedded difference maps. Source plots are displayed as saved, and image data is only converted to the standard display range `[0, 1]`. Long white bands inside source plots are compacted for readability, but the image panels and color values are not changed.

This is intentional: very dark difference maps indicate genuinely small absolute changes on a fixed scale, not a plotting error. Stronger colors would only be appropriate with an explicitly labelled alternate scale, because otherwise tiny differences could appear misleadingly large.

For method-level interpretation and the trade-offs between CFProto-nearer prototype-guided optimization, Retrieval-NUN, SEDC-T, and DVCE, see:

- `results/qualitative_figures/qualitative_results_interpretation.md`

Generated per-method figures:

- `results/qualitative_figures/per_method/busi/sedc_t_original_style_best_first.png`
- `results/qualitative_figures/per_method/pneumonia/sedc_t_original_style_best_first.png`
- `results/qualitative_figures/per_method/pneumonia/sedc_t_lung_field_roi_ablation.png`

Warnings emitted during generation: 0
