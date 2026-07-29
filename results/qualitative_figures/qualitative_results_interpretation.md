# Qualitative Results Interpretation

Status: 2026-07-29

## Purpose and Data Basis

This document records the interpretation of the 14 final qualitative result figures. It is built on the current project result files and the current state of Chapter 5 of the seminar paper. It replaces the earlier, outdated interpretation file.

Authoritative sources:

- Seminar paper: `Seminararbeit.docx`, state 2026-07-29, 17:11:36, SHA-256 `d4cc372059875dc7735da13cbbd5352948d0ea92699d6ca895a45a607bba2ce9`
- selected samples and measured values: `results/qualitative_selection/selected_examples.json`
- mapping of samples to figures: `results/qualitative_figures/qualitative_figure_manifest.json`
- quantitative method overview: `results/qualitative_selection/method_tradeoff_table.md`
- raw run metadata: `results/final/**/metadata.json`
- final figures: `results/qualitative_figures/per_method/`

The seminar paper remains authoritative for the scientific argument. This document serves as technical evidence and interpretation support. It contains no additional medical assessment and does not extend the statements of the paper with unsupported causal claims.

## How to Read the Figures

- `prediction` denotes the model's predicted source class. The adjacent value is the model's confidence for that class.
- The class and confidence to the right of the arrow refer to the counterfactual.
- `target` is the requested target class.
- `valid: yes` only means that the model prediction after the change matches the target class. Neither visual nor medical plausibility follows from it.
- `MAD (l1_mean)` is the mean absolute pixel difference between the original and the counterfactual.
- `RMS (l2_mean)` is `sqrt(mean((CF - Original)^2))` in the image value range `[0, 1]`. It is not the unnormalized full L2 norm.
- For CFProto, Goyal-CVE, and DVCE, the `changed pixel fraction` is the pixel share above the stated threshold. For CFProto and Goyal-CVE this threshold is `0.03`, for DVCE `0.05`.
- For SEDC-T, the `changed pixel fraction` is the image share covered by the selected segment masks. No intensity threshold is used for it.
- For Goyal-CVE, the `embedding distance` is the cosine distance between the original and the selected distractor in the normalized penultimate ResNet18 feature space. The `feature-cell edits` give the number of exchanged cells of the `7 × 7` grid.
- For SEDC-T, `overlay (selected segments)` gives the number of selected segments.
- The difference maps use a fixed scale. A near-black difference map therefore stands for genuinely very small absolute changes and is not a rendering error.
- The roles "balanced valid case", "highest-confidence valid case", "valid case with low visual plausibility", and "failure case" structure the qualitative selection. They are not additional aggregate quantitative statistics.

## BUSI

### Figure 5-1: CFProto original-style

Figure: [`per_method/busi/cfproto_original_style_bottleneck256.png`](per_method/busi/cfproto_original_style_bottleneck256.png)

| Role | Sample | Prediction | Target | Valid | MAD | RMS | Changed pixels |
| --- | ---: | --- | --- | --- | ---: | ---: | ---: |
| balanced | 14 | normal `0.98` → benign `0.79` | benign | yes | `0.0003` | `0.0115` | `0.10 %` |
| highest confidence | 11 | normal `0.96` → benign `0.93` | benign | yes | `0.0007` | `0.0159` | `0.23 %` |
| low visual plausibility | 3 | benign `1.00` → normal `0.61` | normal | yes | `0.0782` | `0.1752` | `31.84 %` |
| failure case | 13 | normal `1.00` → benign `0.90` | malignant | no | `0.0013` | `0.0222` | `0.43 %` |

For the balanced sample 14, a change that is barely visible in the difference image is enough to switch from `normal` to `benign`; a clearly delineated sonographic marker cannot be derived from it. Sample 3 shows the counterexample: despite reaching the target, `31.84 %` of the pixels above the threshold are changed and a large-area grainy texture change appears. In the failure case 13 the change stays small, but the model switches to `benign` instead of the target class `malignant`. The figure thus separates three distinct properties: small pixel difference, target achievement, and medical interpretability.

### Figure 5-2: Goyal-CVE

Figure: [`per_method/busi/goyal_2019_counterfactual_visual_explanations.png`](per_method/busi/goyal_2019_counterfactual_visual_explanations.png)

| Role | Sample | Prediction | Cells | Embedding distance | Changed pixels |
| --- | ---: | --- | ---: | ---: | ---: |
| balanced | 6 | malignant `0.98` → benign `0.52` | 7 | `0.2184` | `13.42 %` |
| highest confidence | 13 | normal `1.00` → malignant `0.63` | 9 | `0.3311` | `16.44 %` |
| low visual plausibility | 9 | malignant `1.00` → benign `0.49` | 26 | `0.2873` | `47.26 %` |

All three counterfactuals reach their respective target class; the distractors used each belong to the target class. In the balanced sample 6, seven cells in the central lesion region are enough for the class change. The counterfactual confidence of `0.52` shows that the result lies just behind the decision boundary. In the highest-confidence case, nine cells are already spread across several image regions. Sample 9 requires 26 of 49 cells and changes `47.26 %` of the pixels. The block-shaped transitions and the large-area transfer from another patient image show that a real distractor source does not automatically produce a medically isolated lesion change. The embedding distance documents the similarity to the selected distractor but is not a proof of plausibility.

### Figure 5-3: SEDC-T original-style

Figure: [`per_method/busi/sedc_t_original_style_best_first.png`](per_method/busi/sedc_t_original_style_best_first.png)

| Role | Sample | Prediction | Target | Valid | Segments | Mask fraction |
| --- | ---: | --- | --- | --- | ---: | ---: |
| balanced | 14 | normal `0.98` → benign `0.73` | benign | yes | 2 | `4.43 %` |
| highest confidence | 0 | benign `1.00` → malignant `0.63` | malignant | yes | 3 | `6.24 %` |
| low visual plausibility | 9 | malignant `1.00` → benign `0.47` | benign | yes | 16 | `53.83 %` |
| failure case | 3 | benign `1.00` → malignant `0.98` | normal | no | 16 | `60.23 %` |

For the balanced sample 14, two small, spatially separate segments and a mask fraction of `4.43 %` are enough for the class change. Sample 9, by contrast, affects 16 segments and more than half of the image. The failure case 3 is particularly informative: although 16 segments, or `60.23 %` of the image area, are replaced, the prediction is not `normal` but `malignant` with `0.98`. SEDC-T enables a spatial attribution of the changed regions; however, a large or anatomically broad segment selection guarantees neither target achievement nor medical specificity.

### Figure 5-4: DVCE with Cone Projection and OpenAI checkpoint

Figure: [`per_method/busi/dvce_original_style_with_cone_projection_openai_checkpoint.png`](per_method/busi/dvce_original_style_with_cone_projection_openai_checkpoint.png)

| Role | Sample | Prediction | Target | Valid | MAD | RMS | Changed pixels |
| --- | ---: | --- | --- | --- | ---: | ---: | ---: |
| balanced | 3 | benign `1.00` → normal `0.67` | normal | yes | `0.0195` | `0.0284` | `7.04 %` |
| highest confidence | 14 | normal `0.98` → benign `1.00` | benign | yes | `0.0238` | `0.0377` | `11.47 %` |
| low visual plausibility | 0 | benign `1.00` → malignant `0.96` | malignant | yes | `0.0307` | `0.0452` | `18.45 %` |
| failure case | 1 | benign `1.00` → benign `0.60` | normal | no | `0.0203` | `0.0298` | `8.26 %` |

The valid examples show a smoothing of speckle patterns and tissue layers distributed across larger image regions. In sample 0, sharply delineated colored or rectangular structures additionally appear that cannot be assigned to any clear sonographic finding. In the failure case 1 the model stays at `benign` despite a visible global change. The figure thus documents the domain mismatch of the generic diffusion prior: the changes affect global image statistics and texture rather than a clearly localized lesion property.

### Figure 5-5: DVCE with Cone Projection and BUSI fine-tuned checkpoint

Figure: [`per_method/busi/dvce_original_style_with_cone_projection_busi_fine_tuned_checkpoint.png`](per_method/busi/dvce_original_style_with_cone_projection_busi_fine_tuned_checkpoint.png)

| Role | Sample | Prediction | Target | Valid | MAD | RMS | Changed pixels |
| --- | ---: | --- | --- | --- | ---: | ---: | ---: |
| balanced | 5 | malignant `0.98` → benign `1.00` | benign | yes | `0.0224` | `0.0337` | `10.73 %` |
| highest confidence | 8 | malignant `0.94` → benign `1.00` | benign | yes | `0.0262` | `0.0392` | `13.52 %` |
| low visual plausibility | 0 | benign `1.00` → malignant `1.00` | malignant | yes | `0.0349` | `0.0523` | `22.35 %` |

All shown cases reach the target class with `1.00`. Compared with the generic configuration in Figure 5-4, the basic ultrasound structure remains more clearly recognizable. Nevertheless, conspicuous new structures appear: a ring-shaped dark structure in the balanced case, additional round hypoechoic regions in the second case, and a large-area changed tissue texture in the low-plausibility case. The specific fine-tuned configuration shows a higher model validity across the full run than the OpenAI Cone configuration. Because of further differences between the runs, no isolated causal fine-tuning effect can be derived from this. Here too, high model validity does not establish medical plausibility.

### Figure 5-6: DVCE without Cone Projection and with BUSI fine-tuned checkpoint

Figure: [`per_method/busi/dvce_original_style_without_cone_projection_busi_fine_tuned_checkpoint.png`](per_method/busi/dvce_original_style_without_cone_projection_busi_fine_tuned_checkpoint.png)

| Role | Sample | Prediction | Target | Valid | MAD | RMS | Changed pixels |
| --- | ---: | --- | --- | --- | ---: | ---: | ---: |
| balanced | 5 | malignant `0.98` → benign `1.00` | benign | yes | `0.0215` | `0.0307` | `9.99 %` |
| highest confidence | 14 | normal `0.98` → benign `1.00` | benign | yes | `0.0245` | `0.0355` | `12.66 %` |
| low visual plausibility | 0 | benign `1.00` → malignant `1.00` | malignant | yes | `0.0325` | `0.0481` | `20.06 %` |

The figure is to be read as an ablation comparison to Figure 5-5. All three counterfactuals are model-valid and reach `1.00` confidence. The changes, however, are not clearly finding-related: in the balanced case the lesion surroundings are diffusely reshaped, in the second case a localized colored structure appears, and in the low-plausibility case a star-shaped dark structure emerges together with a changed image annotation. The ablation examines the guidance variant; it does not represent a clinically realistic tumor transformation.

## Pneumonia

### Figure 5-7: CFProto original-style

Figure: [`per_method/pneumonia/cfproto_original_style_bottleneck256.png`](per_method/pneumonia/cfproto_original_style_bottleneck256.png)

| Role | Sample | Prediction | Target | Valid | MAD | RMS | Changed pixels |
| --- | ---: | --- | --- | --- | ---: | ---: | ---: |
| balanced | 3 | NORMAL `0.57` → PNEUMONIA `0.51` | PNEUMONIA | yes | `0.0000` | `0.0017` | `0.00 %`* |
| highest confidence | 16 | PNEUMONIA `1.00` → NORMAL `0.79` | NORMAL | yes | `0.0028` | `0.0272` | `1.72 %` |
| low visual plausibility | 8 | NORMAL `0.89` → PNEUMONIA `0.76` | PNEUMONIA | yes | `0.0153` | `0.0604` | `10.52 %` |

\* The unrounded fraction at sample 3 is `0.002657 %`; the display `0.00 %` is the correct rounding to two decimal places.

For the balanced sample 3, the prediction flips only narrowly to `PNEUMONIA`, even though the difference map stays almost black. A concrete pneumonia-typical pattern cannot be derived from it. In sample 16, point-like changes are spread across the thorax and border regions. Sample 8 shows a large-area grid- and texture-like noise pattern. The full model validity of the method on the examined Pneumonia manifest is therefore no proof that medically plausible pneumonia evidence was specifically created or removed.

### Figure 5-8: Goyal-CVE

Figure: [`per_method/pneumonia/goyal_2019_counterfactual_visual_explanations.png`](per_method/pneumonia/goyal_2019_counterfactual_visual_explanations.png)

| Role | Sample | Prediction | Cells | Embedding distance | Changed pixels |
| --- | ---: | --- | ---: | ---: | ---: |
| balanced | 3 | NORMAL `0.57` → PNEUMONIA `0.54` | 1 | `0.0684` | `1.97 %` |
| highest confidence | 1 | NORMAL `0.88` → PNEUMONIA `0.58` | 5 | `0.0872` | `9.95 %` |
| low visual plausibility | 10 | PNEUMONIA `1.00` → NORMAL `0.51` | 32 | `0.4402` | `57.81 %` |

For the balanced sample 3, a single cell in the lower thorax region is enough for a narrow class change. The highest-confidence case requires five cells spread across the lung base, diaphragm, and border regions. Sample 10, by contrast, replaces 32 of 49 cells, or `57.81 %` of the pixels. The resulting mosaic takes over large-area anatomy and acquisition differences of the distractor image. Few cell swaps can provide a spatially well-localized model explanation; with many swaps there is no longer a medically isolated change. The clearly larger embedding distance of sample 10 is consistent with this selection but does not replace a visual or medical plausibility check.

### Figure 5-9: SEDC-T original-style

Figure: [`per_method/pneumonia/sedc_t_original_style_best_first.png`](per_method/pneumonia/sedc_t_original_style_best_first.png)

| Role | Sample | Prediction | Target | Valid | Segments | Mask fraction |
| --- | ---: | --- | --- | --- | ---: | ---: |
| balanced | 3 | NORMAL `0.57` → PNEUMONIA `0.64` | PNEUMONIA | yes | 1 | `6.40 %` |
| highest confidence | 4 | NORMAL `0.84` → PNEUMONIA `0.63` | PNEUMONIA | yes | 4 | `17.21 %` |
| low visual plausibility | 19 | PNEUMONIA `0.99` → NORMAL `0.54` | NORMAL | yes | 10 | `28.22 %` |
| failure case | 17 | PNEUMONIA `1.00` → PNEUMONIA `0.82` | NORMAL | no | 15 | `44.21 %` |

For the balanced sample 3, the decisive segment lies at the outer right image border or in the shoulder region and not within a clear lung finding. The fact that the prediction still switches to `PNEUMONIA` is an indication of a model-relevant but not clearly pathological image feature. The other valid examples require broader changes. In the failure case 17, 15 segments and `44.21 %` of the image area are changed, but the model stays at `PNEUMONIA` with `0.82`. The figure thus justifies both the investigation of an ROI restriction and the limited validity of the unrestricted search.

### Figure 5-10: SEDC-T with lung-field ROI

Figure: [`per_method/pneumonia/sedc_t_lung_field_roi_ablation.png`](per_method/pneumonia/sedc_t_lung_field_roi_ablation.png)

| Role | Sample | Prediction | Target | Valid | Segments | Mask fraction |
| --- | ---: | --- | --- | --- | ---: | ---: |
| balanced | 3 | NORMAL `0.57` → PNEUMONIA `0.62` | PNEUMONIA | yes | 1 | `3.38 %` |
| highest confidence | 6 | NORMAL `0.95` → PNEUMONIA `0.69` | PNEUMONIA | yes | 9 | `25.22 %` |
| low visual plausibility | 9 | NORMAL `0.84` → PNEUMONIA `0.55` | PNEUMONIA | yes | 6 | `20.06 %` |
| failure case | 19 | PNEUMONIA `0.99` → PNEUMONIA `0.97` | NORMAL | no | 3 | `5.88 %` |

For the balanced sample 3, the change concentrates on a single segment in the upper lung field. Compared with Figure 5-9, it is spatially more focused with a `3.38 %` mask fraction. The other examples, however, show the limit of the geometric ROI: for valid counterfactuals, large parts of both lung fields or adjacent regions are sometimes replaced. In the failure case, the model stays at `PNEUMONIA` with `0.97` despite three selected segments. The ROI can improve the locality of individual examples but is not a true lung segmentation and guarantees neither target achievement nor medical specificity.

### Figure 5-11: DVCE with Cone Projection and OpenAI checkpoint

Figure: [`per_method/pneumonia/dvce_original_style_with_cone_projection_openai_checkpoint.png`](per_method/pneumonia/dvce_original_style_with_cone_projection_openai_checkpoint.png)

| Role | Sample | Prediction | Target | Valid | MAD | RMS | Changed pixels |
| --- | ---: | --- | --- | --- | ---: | ---: | ---: |
| balanced | 17 | PNEUMONIA `1.00` → NORMAL `0.52` | NORMAL | yes | `0.0133` | `0.0204` | `3.07 %` |
| highest confidence | 7 | NORMAL `0.69` → PNEUMONIA `1.00` | PNEUMONIA | yes | `0.0187` | `0.0267` | `6.23 %` |
| low visual plausibility | 10 | PNEUMONIA `1.00` → NORMAL `0.91` | NORMAL | yes | `0.0191` | `0.0310` | `7.28 %` |
| failure case | 13 | PNEUMONIA `1.00` → PNEUMONIA `0.81` | NORMAL | no | `0.0146` | `0.0266` | `3.48 %` |

The balanced sample 17 reaches `NORMAL` at only `3.07 %` changed pixels above the threshold. The counterfactual nonetheless shows a global smoothing of ribs, lung markings, and heart silhouette. The same reconstruction character shapes the other valid cases. In the failure case 13 the prediction stays at `PNEUMONIA` despite the global reshaping. The figure thus shows why the generic diffusion prior is only of limited medical interpretability despite partly small measured values and high confidences.

### Figure 5-12: DVCE with Cone Projection and Pneumonia fine-tuned checkpoint

Figure: [`per_method/pneumonia/dvce_original_style_with_cone_projection_pneumonia_fine_tuned_checkpoint.png`](per_method/pneumonia/dvce_original_style_with_cone_projection_pneumonia_fine_tuned_checkpoint.png)

| Role | Sample | Prediction | Target | Valid | MAD | RMS | Changed pixels |
| --- | ---: | --- | --- | --- | ---: | ---: | ---: |
| balanced | 19 | PNEUMONIA `0.99` → NORMAL `0.99` | NORMAL | yes | `0.0171` | `0.0331` | `4.88 %` |
| highest confidence | 3 | NORMAL `0.57` → PNEUMONIA `1.00` | PNEUMONIA | yes | `0.0195` | `0.0267` | `6.38 %` |
| low visual plausibility | 10 | PNEUMONIA `1.00` → NORMAL `1.00` | NORMAL | yes | `0.0240` | `0.0391` | `12.08 %` |

All three cases are model-valid and reach confidences between `0.99` and `1.00`. Compared with Figure 5-11, radiographic base structures remain partly sharper. At the same time, point- and line-shaped light, dark, or colored structures as well as deformations at the rib and spine regions appear. These artifacts cannot be clearly assigned to a pneumonia-specific finding. The figure documents the high target-class achievement of this specific configuration, not a proven medical or isolated causal fine-tuning effect.

### Figure 5-13: DVCE without Cone Projection and with OpenAI checkpoint

Figure: [`per_method/pneumonia/dvce_original_style_without_cone_projection_openai_checkpoint.png`](per_method/pneumonia/dvce_original_style_without_cone_projection_openai_checkpoint.png)

| Role | Sample | Prediction | Target | Valid | MAD | RMS | Changed pixels |
| --- | ---: | --- | --- | --- | ---: | ---: | ---: |
| balanced | 19 | PNEUMONIA `0.99` → NORMAL `1.00` | NORMAL | yes | `0.0148` | `0.0275` | `3.47 %` |
| highest confidence | 2 | NORMAL `0.74` → PNEUMONIA `1.00` | PNEUMONIA | yes | `0.0194` | `0.0276` | `6.93 %` |
| low visual plausibility | 10 | PNEUMONIA `1.00` → NORMAL `1.00` | NORMAL | yes | `0.0204` | `0.0328` | `8.65 %` |

All shown counterfactuals reach the target class with `1.00`. In the narrow sense of model validity, this ablation is therefore more successful than the Cone configuration with the OpenAI checkpoint in Figure 5-11. Visually, however, global texture changes, anatomical shifts, and artificial point- or line-shaped structures remain recognizable. Particularly in the `PNEUMONIA` to `NORMAL` changes, the entire image is reshaped without the decision being traceable to the removal of a clear focal opacity. Higher target-class confidence without Cone Projection is therefore no evidence of better medical plausibility.

### Figure 5-14: DVCE without Cone Projection and with Pneumonia fine-tuned checkpoint

Figure: [`per_method/pneumonia/dvce_original_style_without_cone_projection_pneumonia_fine_tuned_checkpoint.png`](per_method/pneumonia/dvce_original_style_without_cone_projection_pneumonia_fine_tuned_checkpoint.png)

| Role | Sample | Prediction | Target | Valid | MAD | RMS | Changed pixels |
| --- | ---: | --- | --- | --- | ---: | ---: | ---: |
| balanced | 17 | PNEUMONIA `1.00` → NORMAL `0.99` | NORMAL | yes | `0.0137` | `0.0232` | `3.00 %` |
| highest confidence | 1 | NORMAL `0.89` → PNEUMONIA `1.00` | PNEUMONIA | yes | `0.0168` | `0.0258` | `5.23 %` |
| low visual plausibility | 10 | PNEUMONIA `1.00` → NORMAL `0.99` | NORMAL | yes | `0.0195` | `0.0305` | `7.59 %` |

The three counterfactuals are model-valid with confidences from `0.99` to `1.00`. For the balanced sample 17, the changed pixel fraction above the threshold is `3.00 %`; nonetheless, contrast, spine, and lung markings change across larger regions. The remaining cases show diffuse texture intensifications and anatomical smoothing that cannot be read as an unambiguous adding or removing of a pneumonia. Together with Figure 5-12, the figure shows that Cone and No-Cone produce different artifact patterns. The choice of guidance is therefore a plausibility problem and not only a validity problem.

## Cross-Method Conclusions

1. **Model validity and medical plausibility remain strictly separate.** A successful target-class change only shows that the examined model reacts to the change. It proves neither a realistic disease progression nor a causal medical transformation.
2. **Small distance values are not sufficient as proof of quality.** CFProto in particular, and individual DVCE cases, show that small MAD, RMS, or changed-pixel values can go together with barely interpretable, distributed, or globally acting changes.
3. **Locality is method-dependent.** Goyal-CVE localizes changes on a coarse `7 × 7` grid and binds them to real distractor images. SEDC-T localizes changes over segments. Both representations are spatially traceable but can contain broad anatomy or acquisition differences.
4. **The Pneumonia ROI is an ablation, not a medical lung segmentation.** It can focus individual changes but, in the present results, does not automatically increase validity and guarantees no pneumonia-specific evidence.
5. **DVCE remains strongly dependent on checkpoint and guidance.** The fine-tuned and the No-Cone configurations frequently reach a higher target-class achievement in the examined runs. The figures, however, still show global reconstructions and artificial structures. Because of not fully controlled execution conditions, no isolated causal fine-tuning effect is claimed.
6. **The figures are part of the results, not decoration.** Each figure explains either a method, a validity–locality–plausibility trade-off, a failure case, or a clearly marked ablation.
