# Fixed Counterfactual Evaluation Summary

This table is generated from method `metadata.json` files.

| Method | Dataset | Samples | Validity | Mean CF confidence | Mean change | Mean runtime (s) | Metadata |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| CFProto original-style prototype-guided counterfactuals | BUSI | 15 | 0.8667 | 0.6815 | 0.0529 | 46.10 | `results/final/cfproto_encoder_knn/busi/metadata.json` |
| CFProto original-style prototype-guided counterfactuals | Pneumonia | 20 | 1.0000 | 0.5740 | 0.0180 | 46.34 | `results/final/cfproto_encoder_knn/pneumonia/metadata.json` |
| Goyal 2019 counterfactual visual explanations | BUSI | 15 | 1.0000 | 0.5279 | 0.2596 | 0.25 | `results/fixed_evaluation/goyal_cve_busi/metadata.json` |
| Goyal 2019 counterfactual visual explanations | Pneumonia | 20 | 1.0000 | 0.5231 | 0.3072 | 0.17 | `results/fixed_evaluation/goyal_cve_pneumonia/metadata.json` |
| SEDC-T original-style best-first | BUSI | 15 | 0.8000 | 0.6343 | 0.2640 | 6.71 | `results/fixed_evaluation/sedc_t_busi_original_style/metadata.json` |
| SEDC-T original-style best-first | Pneumonia | 20 | 0.5500 | 0.6759 | 0.3270 | 13.92 | `results/fixed_evaluation/sedc_t_pneumonia_original_style/metadata.json` |
| SEDC-T lung-field ROI ablation | Pneumonia | 20 | 0.5000 | 0.7770 | 0.1745 | 15.23 | `results/fixed_evaluation/sedc_t_pneumonia_lung_field_roi/metadata.json` |

## Interpretation Notes

- Validity only checks whether the model prediction changed to the target class.
- Mean change is method-dependent and should be interpreted together with the qualitative images.
- Medical plausibility must be discussed separately from model validity.
- DVCE rows should use the original-code-nearer pred_xstart guidance core without Cone Projection unless explicitly stated otherwise.
- For DVCE rows, Mean change prioritizes the existing project metric and falls back to original-style L1 norm only when needed.
- CFProto follows `alibi.explainers.cfproto.CounterfactualProto` faithfully (FISTA with shrinkage-thresholding, untargeted hinge attack loss, binary c-search, encoder-space class prototypes); see `results/final_configs/cfproto_encoder_method_documentation.md` for the full Soll-Ist comparison. Not reproduced: the original TensorFlow graph itself (reimplemented in PyTorch), black-box/numerical-gradient mode, categorical variables and k-d-tree prototypes, and TrustScore filtering (disabled by default in alibi too).
