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
| Retrieval-based nearest-unlike-neighbor baseline | BUSI | 15 | 1.0000 | 0.8191 | 0.8516 | 0.01 | `results/fixed_evaluation/retrieval_nun_busi_balanced_manifest/metadata.json` |
| Retrieval-based nearest-unlike-neighbor baseline | Pneumonia | 20 | 1.0000 | 0.6496 | 0.8741 | 0.01 | `results/fixed_evaluation/retrieval_nun_pneumonia_balanced_manifest/metadata.json` |
| SEDC-T original-style best-first | BUSI | 15 | 0.8000 | 0.6343 | 0.2640 | 7.51 | `results/final/sedc_t_original_style_quickshift_gaussian/busi/metadata.json` |
| SEDC-T original-style best-first | Pneumonia | 20 | 0.5500 | 0.6702 | 0.3377 | 14.41 | `results/final/sedc_t_original_style_quickshift_gaussian/pneumonia/metadata.json` |
| SEDC-T lung-field ROI ablation | Pneumonia | 20 | 0.5000 | 0.7775 | 0.1843 | 15.48 | `results/final/sedc_t_lung_field_roi_quickshift_gaussian/pneumonia/metadata.json` |
| DVCE medical multi-sample generation evaluation with OpenAI checkpoint | BUSI | 5 | 1.0000 | 0.7034 | 0.3569 | 8.86 | `results/fixed_evaluation/dvce_busi_manifest_5_current_checkpoint/metadata.json` |
| DVCE medical multi-sample generation evaluation with OpenAI checkpoint | Pneumonia | 5 | 0.8000 | 0.7219 | 0.1654 | 9.49 | `results/fixed_evaluation/dvce_pneumonia_manifest_5_current_checkpoint/metadata.json` |
| DVCE medical multi-sample generation evaluation with Pneumonia fine-tuned checkpoint | Pneumonia | 5 | 0.8000 | 0.6937 | 0.2469 | 15.63 | `results/fixed_evaluation/dvce_pneumonia_manifest_5_medical_checkpoint_guidance200_sim10_skip40/metadata.json` |

## Interpretation Notes

- Validity only checks whether the model prediction changed to the target class.
- Mean change is method-dependent and should be interpreted together with the qualitative images.
- Medical plausibility must be discussed separately from model validity.
- The CFProto-nearer encoder feature-map row is the retained main
  prototype-guided result; bottleneck256 and bottleneck1024 are retained only as
  autoencoder/prototype-space ablations.
- The CFProto-nearer implementation still is not a full Alibi CFProto reproduction; FISTA/shrinkage, TrustScore, the original TensorFlow graph, and original Alibi k-d-tree machinery are not fully reproduced.
