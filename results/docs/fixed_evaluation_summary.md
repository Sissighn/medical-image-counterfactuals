# Fixed Counterfactual Evaluation

This table is generated directly from the final run metadata by `scripts/summarize_counterfactual_evaluation.py`.

| Method | Dataset | Samples | Validity | Mean CF confidence | Mean change fraction | Mean runtime (s) | Metadata |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| CFProto (original style) | BUSI | 15 | 0.8667 | 0.6815 | 0.0529 | 46.10 | `results/final/cfproto_encoder_knn/busi/metadata.json` |
| CFProto (original style) | Pneumonia | 20 | 1.0000 | 0.5740 | 0.0180 | 46.34 | `results/final/cfproto_encoder_knn/pneumonia/metadata.json` |
| Goyal et al. (2019) CVE | BUSI | 15 | 1.0000 | 0.5279 | 0.2596 | 0.25 | `results/final/goyal_cve_busi/metadata.json` |
| Goyal et al. (2019) CVE | Pneumonia | 20 | 1.0000 | 0.5231 | 0.3072 | 0.17 | `results/final/goyal_cve_pneumonia/metadata.json` |
| SEDC-T (original-style best-first) | BUSI | 15 | 0.8000 | 0.6343 | 0.2640 | 6.71 | `results/final/sedc_t_busi_original_style/metadata.json` |
| SEDC-T (original-style best-first) | Pneumonia | 20 | 0.5500 | 0.6759 | 0.3270 | 13.92 | `results/final/sedc_t_pneumonia_original_style/metadata.json` |
| SEDC-T lung-field ROI ablation | Pneumonia | 20 | 0.5000 | 0.7770 | 0.1745 | 15.23 | `results/final/sedc_t_pneumonia_lung_field_roi/metadata.json` |
| DVCE with Cone Projection and OpenAI checkpoint | BUSI | 15 | 0.9333 | 0.9439 | 0.1157 | 1173.45 | `results/final/dvce_original_style_cone/openai/busi/metadata.json` |
| DVCE with Cone Projection and OpenAI checkpoint | Pneumonia | 20 | 0.8000 | 0.8369 | 0.0510 | 700.15 | `results/final/dvce_original_style_cone/openai/pneumonia/metadata.json` |
| DVCE with Cone Projection and BUSI fine-tuned checkpoint | BUSI | 15 | 1.0000 | 0.9984 | 0.1565 | 44.62 | `results/final/dvce_original_style_cone/busi_medical_checkpoint/busi/metadata.json` |
| DVCE with Cone Projection and Pneumonia fine-tuned checkpoint | Pneumonia | 20 | 1.0000 | 0.9800 | 0.0669 | 44.94 | `results/final/dvce_original_style_cone/pneumonia_medical_checkpoint/pneumonia/metadata.json` |
| DVCE without Cone Projection and BUSI fine-tuned checkpoint | BUSI | 15 | 1.0000 | 0.9976 | 0.1359 | 33.24 | `results/final/dvce_original_style/busi_medical_checkpoint/busi/metadata.json` |
| DVCE without Cone Projection and Pneumonia fine-tuned checkpoint | Pneumonia | 20 | 1.0000 | 0.9951 | 0.0520 | 33.21 | `results/final/dvce_original_style/pneumonia_medical_checkpoint/pneumonia/metadata.json` |
| DVCE without Cone Projection and OpenAI checkpoint | Pneumonia | 20 | 1.0000 | 0.9972 | 0.0596 | 49.95 | `results/final/dvce_original_style/openai/pneumonia/metadata.json` |

## Interpretation Notes

- Validity only indicates that the model predicts the requested target class after the change.
- The change fraction is method-dependent and must be interpreted together with the qualitative figures.
- Neither validity nor high model confidence establishes medical plausibility.
- DVCE uses `pred_xstart` guidance. For the non-robust ResNet18, Cone Projection with a PGD-robust ResNet18 is the original-faithful main variant; runs without Cone Projection are ablations.
- The OpenAI DVCE runs used MPS for BUSI and CUDA for Pneumonia without diffusion FP16. The medically fine-tuned Cone runs used CUDA with diffusion FP16. These runtimes therefore do not isolate a checkpoint effect.
- For DVCE, the change fraction is the pixel fraction above 0.05. Goyal-CVE uses 0.03; SEDC-T reports the selected segment-mask area.
- CFProto ports the core of Alibi's `CounterfactualProto` to PyTorch. The TensorFlow graph, black-box numerical gradients, categorical variables, k-d trees, and TrustScore filtering are not reproduced.
