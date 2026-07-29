# Baseline Classifiers

Three ResNet18 variants were evaluated for each dataset. All final
counterfactual runs use the ImageNet-pretrained model.

| Dataset | Variant | Pretrained | Augmentation | Class weights | Epochs | Accuracy | Weighted F1 |
| --- | --- | --- | --- | --- | ---: | ---: | ---: |
| BUSI | initial | no | no | no | 5 | 0.7288 | 0.7043 |
| BUSI | improved | no | yes | yes | 15 | 0.7203 | 0.7221 |
| BUSI | final | yes | yes | yes | 15 | 0.8390 | 0.8365 |
| Pneumonia | initial | no | no | no | 5 | 0.7885 | 0.7679 |
| Pneumonia | improved | no | yes | yes | 10 | 0.7965 | 0.7752 |
| Pneumonia | final | yes | yes | yes | 10 | 0.8782 | 0.8732 |

Augmentation consists of horizontal flips with probability 0.5, rotations up
to ±10 degrees, and brightness and contrast jitter of 0.1. Training used Adam,
a learning rate of 0.0001, and batch size 16. The checkpoint with the highest
weighted validation F1 was retained.

## Final Test Results

### BUSI

| Class | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| benign | 0.87 | 0.88 | 0.87 | 66 |
| malignant | 0.79 | 0.69 | 0.73 | 32 |
| normal | 0.83 | 0.95 | 0.88 | 20 |

### Pneumonia

| Class | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| NORMAL | 0.96 | 0.70 | 0.81 | 234 |
| PNEUMONIA | 0.85 | 0.98 | 0.91 | 390 |

Exact metrics and confusion matrices are stored in
`busi_pretrained_test_evaluation.json` and
`pneumonia_pretrained_test_evaluation.json`.

## Robust Auxiliary Classifiers

Additional PGD-trained ResNet18 models provide the second gradient direction
for DVCE Cone Projection. They are auxiliary models and are not the classifiers
being explained.

| Dataset | Clean-test accuracy | Weighted F1 |
| --- | ---: | ---: |
| BUSI | 0.5932 | 0.5360 |
| Pneumonia | 0.6250 | 0.4808 |

Training used five epochs, `epsilon=0.03`, step size `0.007`, seven PGD steps,
and equal weights for the clean and adversarial losses.
