# Qualitative Result Figures

The 14 final figures are organized by dataset and method under `per_method/`.
Generated PNG files are excluded from version control because of their size.

Where available, each method includes:

- a valid case with a small change fraction
- the valid case with the highest counterfactual confidence
- a visually questionable valid case
- a failure case

The selection and displayed measurements are stored in
[`selected_examples.json`](selected_examples.json).
[`qualitative_figure_manifest.json`](qualitative_figure_manifest.json) maps
sample indices to the 14 figures.

Each row reports the class transition, original and counterfactual confidence,
target class, validity, MAD (`l1_mean`), RMS (`l2_mean`), and the
method-dependent change fraction. Goyal-CVE also reports the distractor,
embedding distance, and number of feature-cell edits. SEDC-T reports the
number of selected segments.

Difference maps are not normalized per sample. A dark map therefore indicates
a small absolute change on the fixed display scale.

The case-by-case analysis is available in
[`qualitative_results_interpretation.md`](qualitative_results_interpretation.md).
