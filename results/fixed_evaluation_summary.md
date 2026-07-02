# Fixed Counterfactual Evaluation Summary

This table is generated from method `metadata.json` files.

| Method | Dataset | Samples | Validity | Mean CF confidence | Mean change | Mean runtime (s) | Metadata |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| PyTorch prototype-guided optimization baseline | BUSI | 15 | 1.0000 | 0.9978 | 0.0559 | 5.24 | `results/fixed_evaluation/prototype_busi_balanced_manifest/metadata.json` |
| PyTorch prototype-guided optimization baseline | Pneumonia | 20 | 1.0000 | 0.9928 | 0.1442 | 5.69 | `results/fixed_evaluation/prototype_pneumonia_balanced_manifest/metadata.json` |
| SEDC-T-style targeted segment replacement | BUSI | 15 | 0.8000 | 0.6376 | 0.1471 | 0.56 | `results/fixed_evaluation/sedc_t_busi_balanced_manifest/metadata.json` |
| SEDC-T-style targeted segment replacement | Pneumonia | 20 | 0.4500 | 0.7639 | 0.1510 | 0.39 | `results/fixed_evaluation/sedc_t_pneumonia_balanced_manifest/metadata.json` |
| DVCE medical multi-sample generation evaluation | BUSI | 5 | 1.0000 | 0.7034 | 0.3569 | 8.86 | `results/fixed_evaluation/dvce_busi_manifest_5_current_checkpoint/metadata.json` |
| DVCE medical multi-sample generation evaluation | Pneumonia | 5 | 0.8000 | 0.7219 | 0.1654 | 9.49 | `results/fixed_evaluation/dvce_pneumonia_manifest_5_current_checkpoint/metadata.json` |
| DVCE medical multi-sample generation evaluation with Pneumonia fine-tuned checkpoint | Pneumonia | 5 | 0.8000 | 0.6937 | 0.2469 | 15.63 | `results/fixed_evaluation/dvce_pneumonia_manifest_5_medical_checkpoint_guidance200_sim10_skip40/metadata.json` |

## Interpretation Notes

- Validity only checks whether the model prediction changed to the target class.
- Mean change is method-dependent and should be interpreted together with the qualitative images.
- Medical plausibility must be discussed separately from model validity.
