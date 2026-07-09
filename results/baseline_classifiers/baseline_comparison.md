# Baseline Comparison

| Dataset | Model | Pretrained | Augmentation | Class weights | Epochs | Accuracy | Weighted F1 | Main issue |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | --- |
| BUSI | ResNet18 baseline | No | No | No | 5 | 0.7288 | 0.7043 | malignant and normal often predicted as benign |
| BUSI | ResNet18 improved | No | Yes | Yes | 15 | 0.7203 | 0.7221 | benign is less dominant, but classes are still confused |
| BUSI | ResNet18 pretrained | Yes | Yes | Yes | 15 | 0.8390 | 0.8365 | strongest BUSI baseline so far; malignant remains the hardest class |
| Pneumonia | ResNet18 baseline | No | No | No | 5 | 0.7885 | 0.7679 | many NORMAL images predicted as PNEUMONIA |
| Pneumonia | ResNet18 improved | No | Yes | Yes | 10 | 0.7965 | 0.7752 | many NORMAL images still predicted as PNEUMONIA |
| Pneumonia | ResNet18 pretrained | Yes | Yes | Yes | 10 | 0.8782 | 0.8732 | strongest Pneumonia baseline so far; NORMAL recall improved but false positives remain |

## Notes

- The improved training added random horizontal flips, small rotations, mild brightness/contrast jitter, and class-weighted cross entropy.
- Pretrained ImageNet weights were downloaded manually to `~/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth` because the automatic torchvision download created empty `.partial` files and hung.
- The training CLI still enables pretrained weights only when `--pretrained` is passed explicitly.
- BUSI improved mainly in per-class balance: malignant recall increased from 0.50 to 0.78 and normal recall increased from 0.35 to 0.80.
- Pneumonia improved only slightly on the test set. The validation split is very small, so its high validation F1 should not be treated as a reliable final signal.
- With pretrained weights, BUSI reached weighted F1 0.8365 and Pneumonia reached weighted F1 0.8732 on the test sets. These are the best baselines so far and are suitable starting points for the counterfactual methods.
