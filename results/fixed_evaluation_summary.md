# Fixed Counterfactual Evaluation Summary

This table is generated from method `metadata.json` files.

| Method | Dataset | Samples | Validity | Mean CF confidence | Mean change | Mean runtime (s) | Metadata |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| CFProto-nearer prototype-guided optimization baseline (encoder feature map) | BUSI | 15 | 1.0000 | 0.5471 | 0.0084 | 7.43 | `results/final/cfproto_encoder_knn/busi/metadata.json` |
| CFProto-nearer prototype-guided optimization baseline (encoder feature map) | Pneumonia | 20 | 0.9000 | 0.5767 | 0.0108 | 8.58 | `results/final/cfproto_encoder_knn/pneumonia/metadata.json` |
| CFProto-nearer prototype-guided optimization baseline (bottleneck256) | BUSI | 15 | 0.6667 | 0.6845 | 0.6168 | 8.65 | `results/final/cfproto_encoder_knn_bottleneck256/busi/metadata.json` |
| CFProto-nearer prototype-guided optimization baseline (bottleneck256) | Pneumonia | 20 | 0.5000 | 0.7537 | 0.6666 | 8.73 | `results/final/cfproto_encoder_knn_bottleneck256/pneumonia/metadata.json` |
| CFProto-nearer prototype-guided optimization baseline (bottleneck1024) | BUSI | 15 | 0.6667 | 0.6590 | 0.7053 | 14.60 | `results/final/cfproto_encoder_knn_bottleneck1024/busi/metadata.json` |
| CFProto-nearer prototype-guided optimization baseline (bottleneck1024) | Pneumonia | 20 | 0.5500 | 0.7292 | 0.6312 | 13.78 | `results/final/cfproto_encoder_knn_bottleneck1024/pneumonia/metadata.json` |
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
- The CFProto-nearer encoder feature-map row is the main prototype-guided result; bottleneck rows are retained only as ablations.
- The CFProto-nearer implementation still is not a full Alibi CFProto reproduction; FISTA/shrinkage, TrustScore, the original TensorFlow graph, and original Alibi k-d-tree machinery are not fully reproduced.
