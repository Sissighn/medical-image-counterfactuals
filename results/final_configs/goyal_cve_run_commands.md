# Goyal et al. 2019 CVE Final Run Commands

Instance-based counterfactual visual explanations
(`scripts/run_goyal_cve_pytorch.py`) after Goyal, Wu, Ernst, Batra, Parikh &
Lee (2019), "Counterfactual Visual Explanations", ICML 2019, arXiv:1904.07451.
A reference implementation of the method (as baseline) is contained in the
official Meta repository
https://github.com/facebookresearch/visual-counterfactuals.

Method: the ResNet18 is decomposed into spatial extractor f (through layer4,
7x7x512 cells) and decision head g (global average pooling + fc). For each
manifest sample, a distractor image from the target class is retrieved as the
nearest correctly classified training image in pooled penultimate feature
space (cosine distance). Greedy exhaustive search then evaluates all
(query cell, distractor cell) swaps per iteration, commits the swap that
maximizes the target-class softmax probability, and stops at the first
prediction flip. Each query cell is edited at most once and each distractor
cell used at most once (permutation constraint, paper Section 3). With the
full 49-cell budget the pooled feature equals the distractor's, so validity
is guaranteed by construction; the reported metric of interest is the number
of edits (sparsity).

The decision-flipping edit lives in feature space. The composite
counterfactual image (aligned 32x32 image patches pasted from the distractor)
is the paper's standard visualization; pixel change metrics are computed on
this composite.

Samples and targets come from the fixed evaluation manifests, which are bound
to the `*_resnet18_pretrained.pth` classifiers (the runner asserts the current
prediction matches the manifest).

## 1. BUSI

```bash
.venv/bin/python scripts/run_goyal_cve_pytorch.py \
  --model_path models/busi_resnet18_pretrained.pth \
  --dataset_path data/processed/BUSI \
  --output_dir results/fixed_evaluation/goyal_cve_busi \
  --manifest_path results/evaluation_manifests/busi_balanced_5_per_class_second_best.json
```

## 2. Pneumonia

```bash
.venv/bin/python scripts/run_goyal_cve_pytorch.py \
  --model_path models/pneumonia_resnet18_pretrained.pth \
  --dataset_path data/processed/Pneumonia \
  --output_dir results/fixed_evaluation/goyal_cve_pneumonia \
  --manifest_path results/evaluation_manifests/pneumonia_balanced_10_per_class_second_best.json
```

## 3. Regenerate the qualitative figures

```bash
.venv/bin/python scripts/select_interpretable_examples.py \
  --copy_assets \
  --metadata \
    results/fixed_evaluation/goyal_cve_busi/metadata.json \
    results/fixed_evaluation/goyal_cve_pneumonia/metadata.json

.venv/bin/python scripts/create_qualitative_comparison_figures.py
```

Include the other methods' `metadata.json` paths in the `--metadata` list when
rebuilding the full comparison figure set.

## Notes

- This runner replaces the former `run_retrieval_nun_pytorch.py` baseline,
  which only retrieved the nearest unlike neighbor without editing the query
  and was not based on an original published method.
- `--max_edits` defaults to the full grid (49). Keep the default for both
  datasets; the paper's stopping criterion is the first prediction flip.
- Per-sample edit lists (cell coordinates and target-probability trajectory)
  are stored in `metadata.json` under `records[*].edits`.
