# Fixed Counterfactual Evaluation Summary

This table is generated from method `metadata.json` files.

| Method | Dataset | Samples | Validity | Mean CF confidence | Mean change | Mean runtime (s) | Metadata |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| CFProto original-style prototype-guided counterfactuals | BUSI | 15 | 0.8667 | 0.6815 | 0.0529 | 46.10 | `results/final/cfproto_encoder_knn/busi/metadata.json` |
| CFProto original-style prototype-guided counterfactuals | Pneumonia | 20 | 1.0000 | 0.5740 | 0.0180 | 46.34 | `results/final/cfproto_encoder_knn/pneumonia/metadata.json` |
| Goyal 2019 counterfactual visual explanations | BUSI | 15 | 1.0000 | 0.5279 | 0.2596 | 0.25 | `results/final/goyal_cve_busi/metadata.json` |
| Goyal 2019 counterfactual visual explanations | Pneumonia | 20 | 1.0000 | 0.5231 | 0.3072 | 0.17 | `results/final/goyal_cve_pneumonia/metadata.json` |
| SEDC-T original-style best-first | BUSI | 15 | 0.8000 | 0.6343 | 0.2640 | 6.71 | `results/final/sedc_t_busi_original_style/metadata.json` |
| SEDC-T original-style best-first | Pneumonia | 20 | 0.5500 | 0.6759 | 0.3270 | 13.92 | `results/final/sedc_t_pneumonia_original_style/metadata.json` |
| SEDC-T lung-field ROI ablation | Pneumonia | 20 | 0.5000 | 0.7770 | 0.1745 | 15.23 | `results/final/sedc_t_pneumonia_lung_field_roi/metadata.json` |
| DVCE original-style medical generation with Cone Projection with OpenAI checkpoint | BUSI | 15 | 0.9333 | 0.9439 | 0.1157 | 1173.45 | `results/final/dvce_original_style_cone/openai/busi/metadata.json` |
| DVCE original-style medical generation with Cone Projection with OpenAI checkpoint | Pneumonia | 20 | 0.8000 | 0.8369 | 0.0510 | 700.15 | `results/final/dvce_original_style_cone/openai/pneumonia/metadata.json` |
| DVCE original-style medical generation with Cone Projection with BUSI fine-tuned checkpoint | BUSI | 15 | 1.0000 | 0.9984 | 0.1565 | 44.62 | `results/final/dvce_original_style_cone/busi_medical_checkpoint/busi/metadata.json` |
| DVCE original-style medical generation with Cone Projection with Pneumonia fine-tuned checkpoint | Pneumonia | 20 | 1.0000 | 0.9800 | 0.0669 | 44.94 | `results/final/dvce_original_style_cone/pneumonia_medical_checkpoint/pneumonia/metadata.json` |
| DVCE original-style medical generation without Cone Projection with BUSI fine-tuned checkpoint | BUSI | 15 | 1.0000 | 0.9976 | 0.1359 | 33.24 | `results/final/dvce_original_style/busi_medical_checkpoint/busi/metadata.json` |
| DVCE original-style medical generation without Cone Projection with Pneumonia fine-tuned checkpoint | Pneumonia | 20 | 1.0000 | 0.9951 | 0.0520 | 33.21 | `results/final/dvce_original_style/pneumonia_medical_checkpoint/pneumonia/metadata.json` |
| DVCE original-style medical generation without Cone Projection with OpenAI checkpoint | Pneumonia | 20 | 1.0000 | 0.9972 | 0.0596 | 49.95 | `results/final/dvce_original_style/openai/pneumonia/metadata.json` |

## Interpretation Notes

- Validity only checks whether the model prediction changed to the target class.
- Mean change is method-dependent and should be interpreted together with the qualitative images.
- Medical plausibility must be discussed separately from model validity.
- DVCE uses the implementation closest to the original code, with `pred_xstart` guidance. Cone Projection with a robust PGD ResNet18 as the second classifier is the original-faithful variant for the non-robust explained ResNet18; rows without Cone Projection are explicit ablations. See `results/final_configs/dvce_method_documentation.md`.
- The DVCE OpenAI-checkpoint runs used MPS for BUSI and CUDA for Pneumonia without diffusion FP16, whereas the medical fine-tuned Cone runs used CUDA with diffusion FP16. Because device and precision conditions differ, these runtimes are not suitable for an isolated method or checkpoint comparison.
- For DVCE rows, mean change is the changed-pixel fraction at threshold 0.05, consistent with the SEDC-T and Goyal rows.
- CFProto follows `alibi.explainers.cfproto.CounterfactualProto` faithfully (FISTA with shrinkage-thresholding, untargeted hinge attack loss, binary search over `c`, and encoder-space class prototypes). See `results/final_configs/cfproto_encoder_method_documentation.md` for the full implementation-to-reference comparison. The original TensorFlow graph, black-box numerical-gradient mode, categorical variables, k-d-tree prototypes, and TrustScore filtering are not reproduced; TrustScore filtering is disabled by default in Alibi.
