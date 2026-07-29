# Qualitative Result Interpretation

## Evidence Base

This analysis covers the 14 final method-specific result figures. Sample
selection and displayed values come from
[`selected_examples.json`](selected_examples.json), while
[`qualitative_figure_manifest.json`](qualitative_figure_manifest.json) maps
samples to figures. Raw measurements are stored in the corresponding
`metadata.json` files under `results/final/`.

A valid result reaches the requested target class. This is a statement about
the model, not evidence of medical plausibility.

MAD (`l1_mean`) is the mean absolute pixel difference. RMS (`l2_mean`) is the
root mean squared pixel difference in `[0, 1]` image space. For CFProto and
Goyal-CVE, the change fraction is the pixel fraction above 0.03; for DVCE, it
is the pixel fraction above 0.05. For SEDC-T, it is the image area covered by
the selected segment masks.

## BUSI

### Figure 5-1: CFProto

- Sample 14 changes from `normal` to `benign` with only 0.10% of pixels above
  the threshold.
- Sample 3 reaches `normal` but changes 31.84% of pixels and introduces a broad
  grainy pattern.
- Sample 13 misses `malignant` and ends at `benign` despite a small change.

The figure separates three independent properties: small pixel distance,
target achievement, and visual interpretability.

### Figure 5-2: Goyal-CVE

- Sample 6 uses seven feature-cell edits and changes 13.42% of pixels.
- Sample 13 has the highest confidence among the displayed BUSI cases and uses
  nine edits.
- Sample 9 uses 26 of 49 cells and changes 47.26% of pixels.

Few edits remain spatially localized. Many edits create block-like transitions
and broad transfers from the distractor image. Embedding distance describes
distractor similarity but does not establish medical plausibility.

### Figure 5-3: SEDC-T

- Sample 14 reaches `benign` with two segments and 4.43% mask area.
- Sample 0 reaches `malignant` with three segments and 6.24% mask area.
- Sample 9 uses 16 segments and changes 53.83% of the image.
- Sample 3 misses `normal` even though 16 segments and 60.23% of the image are
  replaced.

SEDC-T exposes the selected regions directly. Broad segment sets still
guarantee neither target achievement nor medical specificity.

### Figure 5-4: DVCE Cone with the OpenAI Checkpoint

Valid sample 3 changes 7.04% of pixels above the threshold. Sample 0 changes
18.45% and contains conspicuous artificial structures. Sample 1 remains
`benign` despite a visible global texture change.

Changes affect speckle patterns and tissue texture across larger regions. This
is consistent with a domain mismatch of the generic diffusion prior and does
not provide a clearly localized finding change.

### Figure 5-5: DVCE Cone with the BUSI Fine-Tuned Checkpoint

Samples 5, 8, and 0 all reach the target with confidence 1.00. Their change
fractions are 10.73%, 13.52%, and 22.35%. The ultrasound structure remains more
recognizable than in the OpenAI variant, but new ring-shaped, round, or broad
structures appear.

The high model validity of this run does not show that the generated structures
are realistic tumor changes.

### Figure 5-6: DVCE without Cone with the BUSI Fine-Tuned Checkpoint

The displayed No-Cone cases also reach the target with confidence 1.00. Their
change fractions are 9.99%, 12.66%, and 20.06%. Artifact patterns differ from
the Cone variant, but diffuse or artificial changes remain.

This figure is a guidance ablation, not a clinically realistic
transformation.

## Pneumonia

### Figure 5-7: CFProto

- Sample 3 narrowly changes from `NORMAL` to `PNEUMONIA`; the unrounded change
  fraction is 0.002657%.
- Sample 16 reaches `NORMAL` with 1.72% changed pixels.
- Sample 8 changes 10.52% and introduces a broad grid- and texture-like
  pattern.

Full validity on the Pneumonia set only shows that the decision boundary was
crossed. It does not identify a concrete pneumonia-specific image change.

### Figure 5-8: Goyal-CVE

- Sample 3 uses one cell, has embedding distance 0.0684, and changes 1.97% of
  pixels.
- Sample 1 uses five cells and changes 9.95%.
- Sample 10 uses 32 of 49 cells, has embedding distance 0.4402, and changes
  57.81%.

Few swaps produce a more localized model explanation. Many swaps transfer
large portions of anatomy and acquisition differences from the distractor.

### Figure 5-9: SEDC-T

- Sample 3 reaches `PNEUMONIA` with one segment and 6.40% mask area. The
  segment lies near the outer image or shoulder region.
- Sample 19 uses ten segments and 28.22% mask area.
- Sample 17 remains `PNEUMONIA` despite 15 segments and 44.21% mask area.

The border region in sample 3 is model-relevant but not interpretable as a
clear lung finding. The figure motivates the ROI ablation while documenting
the limited validity of unrestricted search.

### Figure 5-10: SEDC-T with Lung-Field ROI

Sample 3 concentrates the change in one upper-lung segment and reduces mask
area to 3.38%. Other valid cases still require 20.06% and 25.22%. Sample 19
remains `PNEUMONIA` even after three ROI-constrained segments are selected.

The geometric ROI can focus individual cases. It is not a true lung
segmentation and does not improve aggregate validity.

### Figure 5-11: DVCE Cone with the OpenAI Checkpoint

Sample 17 reaches `NORMAL` at a 3.07% change fraction. Sample 10 reaches
`NORMAL` at 7.28%. Sample 13 misses the target despite visible global
reconstruction.

Ribs, lung markings, and the heart silhouette are smoothed or reshaped across
larger areas. Small threshold fractions do not rule out global changes.

### Figure 5-12: DVCE Cone with the Pneumonia Fine-Tuned Checkpoint

Samples 19, 3, and 10 reach their targets with confidence between 0.99 and
1.00. Their change fractions are 4.88%, 6.38%, and 12.08%. Radiographic
structures remain partly sharper, while point-, line-, or color-like artifacts
still appear.

The figure documents high validity for this configuration, not an isolated
medical or causal effect of fine-tuning.

### Figure 5-13: DVCE without Cone with the OpenAI Checkpoint

All three displayed samples reach the target with confidence 1.00 at change
fractions of 3.47%, 6.93%, and 8.65%. Global texture changes, anatomical
shifts, and artificial structures remain visible.

Higher target confidence without Cone Projection is not evidence of better
medical plausibility.

### Figure 5-14: DVCE without Cone with the Pneumonia Fine-Tuned Checkpoint

Samples 17, 1, and 10 are valid, with change fractions of 3.00%, 5.23%, and
7.59%. Despite these values, contrast, spine, lung markings, or texture change
across larger regions.

Cone and No-Cone produce different artifact patterns. Guidance choice therefore
affects visual plausibility as well as validity.

## Cross-Method Conclusions

1. Model validity and medical plausibility require separate evaluation.
2. Small MAD, RMS, or change fractions are insufficient quality evidence.
3. Goyal-CVE and SEDC-T expose spatial changes but may include broad anatomical
   or acquisition-related regions.
4. The lung-field ROI is a locality ablation, not a medical segmentation.
5. DVCE remains sensitive to checkpoint and guidance; high confidence does not
   exclude global reconstruction or artifacts.
