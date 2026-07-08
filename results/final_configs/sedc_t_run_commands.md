# SEDC-T Final Run Commands

Original-code-nearer SEDC-T (`scripts/run_sedc_t_pytorch.py`, ported from
`isedc/sedc_t2_fast.py`). Defaults match the reference: quickshift
segmentation (kernel_size=4, max_dist=200, ratio=0.2), Gaussian blur 31x31
replacement, best-first expansion by target-vs-original class score, stop at the
first valid level. Candidate predictions for each search level run in one
batched forward pass (equivalent to the reference `classifier.predict` on the
full candidate array).

Search timeout: the reference `max_time` is 600 s, but on these 224x224 medical
images every valid counterfactual is found within ~3 s (measured on the full
BUSI set). A sample that runs to the timeout is one where blur replacement
cannot flip the class at all, so a long timeout only wastes time without
recovering any counterfactual. The commands below therefore use
`--search_timeout_seconds 30` (a >10x margin over the observed ~3 s). Keep this
value identical across both datasets for a fair comparison.

Samples and targets come from the fixed evaluation manifests, which are bound to
the `*_resnet18_pretrained.pth` classifiers (the runner asserts the current
prediction matches the manifest).

## 1. BUSI — original-style reference (`roi_mode none`)

```bash
.venv/bin/python scripts/run_sedc_t_pytorch.py \
  --model_path models/busi_resnet18_pretrained.pth \
  --dataset_path data/processed/BUSI \
  --output_dir results/fixed_evaluation/sedc_t_busi_original_style \
  --manifest_path results/evaluation_manifests/busi_balanced_5_per_class_second_best.json \
  --search_timeout_seconds 30
```

## 2. Pneumonia — original-style reference (`roi_mode none`)

```bash
.venv/bin/python scripts/run_sedc_t_pytorch.py \
  --model_path models/pneumonia_resnet18_pretrained.pth \
  --dataset_path data/processed/Pneumonia \
  --output_dir results/fixed_evaluation/sedc_t_pneumonia_original_style \
  --manifest_path results/evaluation_manifests/pneumonia_balanced_10_per_class_second_best.json \
  --search_timeout_seconds 30
```

## 3. Pneumonia — lung-field ROI ablation

Project-specific ablation, not part of the original SEDC-T setup. Segments are
restricted to a coarse geometric lung-field prior (two rectangles with a
central mediastinum gap, tuned to include the apices and exclude the
sub-diaphragmatic region). The exact fractions are recorded per run under
`parameters.roi_lung_field_geometry` in `metadata.json`.

```bash
.venv/bin/python scripts/run_sedc_t_pytorch.py \
  --model_path models/pneumonia_resnet18_pretrained.pth \
  --dataset_path data/processed/Pneumonia \
  --output_dir results/fixed_evaluation/sedc_t_pneumonia_lung_field_roi \
  --manifest_path results/evaluation_manifests/pneumonia_balanced_10_per_class_second_best.json \
  --roi_mode lung_fields \
  --search_timeout_seconds 30
```

## 4. Regenerate the per-method qualitative figures

The runner writes `metadata.json` + sample PNGs; the figures under
`results/qualitative_figures/per_method/` are then rebuilt from that metadata.
Include the other methods' metadata files too so the full figure set stays
consistent (add their `metadata.json` paths to the `--metadata` list).

```bash
.venv/bin/python scripts/select_interpretable_examples.py \
  --copy_assets \
  --metadata \
    results/fixed_evaluation/sedc_t_busi_original_style/metadata.json \
    results/fixed_evaluation/sedc_t_pneumonia_original_style/metadata.json \
    results/fixed_evaluation/sedc_t_pneumonia_lung_field_roi/metadata.json

.venv/bin/python scripts/create_qualitative_comparison_figures.py
```

This produces:

- `results/qualitative_figures/per_method/busi/sedc_t_original_style_best_first.png`
- `results/qualitative_figures/per_method/pneumonia/sedc_t_original_style_best_first.png`
- `results/qualitative_figures/per_method/pneumonia/sedc_t_lung_field_roi_ablation.png`

Note: `roi_mode none` -> method "SEDC-T original-style best-first";
`roi_mode lung_fields` -> "SEDC-T lung-field ROI ablation". These names drive
the per-method figure slugs above.
