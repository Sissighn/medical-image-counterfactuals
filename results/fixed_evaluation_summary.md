# Fixed Counterfactual Evaluation Summary

This table is generated from method `metadata.json` files.

| Method | Dataset | Samples | Validity | Mean CF confidence | Mean change | Mean runtime (s) | Metadata |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| CFProto-nearer prototype-guided optimization baseline | BUSI | 15 | 1.0000 | 0.5471 | 0.0084 | 7.43 | `results/final/cfproto_encoder_knn/busi/metadata.json` |
| CFProto-nearer prototype-guided optimization baseline | Pneumonia | 20 | 0.9000 | 0.5767 | 0.0108 | 8.58 | `results/final/cfproto_encoder_knn/pneumonia/metadata.json` |
| Retrieval-based nearest-unlike-neighbor baseline | BUSI | 15 | 1.0000 | 0.8191 | 0.8516 | 0.01 | `results/fixed_evaluation/retrieval_nun_busi_balanced_manifest/metadata.json` |
| Retrieval-based nearest-unlike-neighbor baseline | Pneumonia | 20 | 1.0000 | 0.6496 | 0.8741 | 0.01 | `results/fixed_evaluation/retrieval_nun_pneumonia_balanced_manifest/metadata.json` |
| SEDC-T original-style best-first segment replacement | BUSI | 15 | 0.8000 | 0.6674 | 0.1517 | 6.59 | `results/fixed_evaluation/sedc_t_original_style_busi_balanced_manifest/metadata.json` |
| SEDC-T original-style best-first segment replacement | Pneumonia | 20 | 0.5500 | 0.7343 | 0.1410 | 13.78 | `results/fixed_evaluation/sedc_t_original_style_pneumonia_balanced_manifest/metadata.json` |
| SEDC-T project variant | BUSI | 15 | 0.8000 | 0.6376 | 0.1471 | 0.56 | `results/fixed_evaluation/sedc_t_busi_balanced_manifest/metadata.json` |
| SEDC-T project variant (lung_fields) | Pneumonia | 20 | 0.4500 | 0.7639 | 0.1510 | 0.39 | `results/fixed_evaluation/sedc_t_pneumonia_balanced_manifest/metadata.json` |
| SEDC-T tuned project variant (none, max 10) | BUSI | 15 | 0.8000 | 0.6050 | 0.1698 | 1.00 | `results/fixed_evaluation/sedc_t_tuned_busi_none_blur_max10/metadata.json` |
| SEDC-T tuned project variant (lung_fields, max 10) | Pneumonia | 20 | 0.5000 | 0.7653 | 0.2131 | 0.71 | `results/fixed_evaluation/sedc_t_tuned_pneumonia_lung_roi_blur_max10/metadata.json` |
| SEDC-T tuned project variant (none, max 8) | Pneumonia | 20 | 0.6000 | 0.6800 | 0.1552 | 1.21 | `results/fixed_evaluation/sedc_t_tuned_pneumonia_none_blur_max8/metadata.json` |
| DVCE medical multi-sample generation evaluation with OpenAI checkpoint | BUSI | 5 | 1.0000 | 0.7034 | 0.3569 | 8.86 | `results/fixed_evaluation/dvce_busi_manifest_5_current_checkpoint/metadata.json` |
| DVCE medical multi-sample generation evaluation with OpenAI checkpoint | Pneumonia | 5 | 0.8000 | 0.7219 | 0.1654 | 9.49 | `results/fixed_evaluation/dvce_pneumonia_manifest_5_current_checkpoint/metadata.json` |
| DVCE medical multi-sample generation evaluation with Pneumonia fine-tuned checkpoint | Pneumonia | 5 | 0.8000 | 0.6937 | 0.2469 | 15.63 | `results/fixed_evaluation/dvce_pneumonia_manifest_5_medical_checkpoint_guidance200_sim10_skip40/metadata.json` |

## Interpretation Notes

- Validity only checks whether the model prediction changed to the target class.
- Mean change is method-dependent and should be interpreted together with the qualitative images.
- Medical plausibility must be discussed separately from model validity.
- The CFProto-nearer row is the only retained prototype-guided result.
- The CFProto-nearer implementation still is not a full Alibi CFProto reproduction; FISTA/shrinkage, TrustScore, the original TensorFlow graph, and original Alibi k-d-tree machinery are not fully reproduced.
