# CFProto Final Fixed-Manifest Summary

This summary reports the final fixed-manifest runs for the CFProto
original-style prototype-guided method (FISTA with shrinkage-thresholding,
untargeted hinge attack loss, encoder-space class prototypes from classifier
predictions, binary c-search, elastic-net best-counterfactual selection),
following `alibi.explainers.cfproto.CounterfactualProto`. See
[`cfproto_encoder_method_documentation.md`](../../final_configs/cfproto_encoder_method_documentation.md)
for the full method documentation and Soll-Ist comparison with the original.

These runs were executed on a CUDA machine (university GPU) rather than the
project's usual Apple Silicon/MPS machine, using the exact commands below.

## Exact Commands

### BUSI

```bash
PYTHONPATH=. python scripts/run_cfproto_pytorch.py \
  --model_path models/busi_resnet18_pretrained.pth \
  --dataset_path data/processed/BUSI \
  --output_dir results/final/cfproto_encoder_knn/busi \
  --manifest_path results/evaluation_manifests/busi_balanced_5_per_class_second_best.json \
  --max_iterations 1000 \
  --learning_rate_init 0.01 \
  --kappa 0.0 \
  --beta 0.1 \
  --gamma 1.0 \
  --theta 0.5 \
  --c_init 1.0 \
  --c_steps 5 \
  --autoencoder_path models/autoencoder_busi_bottleneck256.pth \
  --prototype_k 3 \
  --k_type mean \
  --batch_size 16
```

### Pneumonia

```bash
PYTHONPATH=. python scripts/run_cfproto_pytorch.py \
  --model_path models/pneumonia_resnet18_pretrained.pth \
  --dataset_path data/processed/Pneumonia \
  --output_dir results/final/cfproto_encoder_knn/pneumonia \
  --manifest_path results/evaluation_manifests/pneumonia_balanced_10_per_class_second_best.json \
  --max_iterations 1000 \
  --learning_rate_init 0.01 \
  --kappa 0.0 \
  --beta 0.1 \
  --gamma 1.0 \
  --theta 0.05 \
  --c_init 1.0 \
  --c_steps 5 \
  --autoencoder_path models/autoencoder_pneumonia_bottleneck256.pth \
  --prototype_k 3 \
  --k_type mean \
  --batch_size 16
```

## Final Parameters

| Parameter | BUSI | Pneumonia |
| --- | --- | --- |
| `autoencoder_path` | `autoencoder_busi_bottleneck256.pth` | `autoencoder_pneumonia_bottleneck256.pth` |
| `prototype_space` | encoder (bottleneck, latent_dim=256) | encoder (bottleneck, latent_dim=256) |
| `prototype_k` / `k_type` | 3 / mean | 3 / mean |
| `max_iterations` | 1000 | 1000 |
| `c_init` / `c_steps` | 1.0 / 5 | 1.0 / 5 |
| `kappa` | 0.0 | 0.0 |
| `beta` | 0.1 | 0.1 |
| `gamma` | 1.0 | 1.0 |
| `theta` | 0.5 | 0.05 (recalibrated, see method doc §4) |
| `learning_rate_init` | 0.01 | 0.01 |
| `target_class` | manifest target (`record["target_class_index"]`) | manifest target (`record["target_class_index"]`) |

## Quantitative Summary

| dataset | num_samples | valid_count | validity | mean_cf_confidence (valid) | mean_l1 | mean_l2 | mean_linf | mean_changed_pixel_fraction | mean_runtime_seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BUSI | 15 | 13 | 0.8667 | 0.6463 | 0.0121 | 0.0436 | 0.8506 | 0.0529 | 46.10 |
| Pneumonia | 20 | 20 | 1.0000 | 0.5740 | 0.0027 | 0.0210 | 0.8059 | 0.0180 | 46.34 |

## Metadata Files

- BUSI: `results/final/cfproto_encoder_knn/busi/metadata.json`
- Pneumonia: `results/final/cfproto_encoder_knn/pneumonia/metadata.json`

## Manifest Fairness Check

- BUSI: manifest samples unchanged = True, manifest targets unchanged = True, invalid filtered = False
- Pneumonia: manifest samples unchanged = True, manifest targets unchanged = True, invalid filtered = False

## Invalid Counterfactuals

| Dataset | Manifest sample | True label | Original prediction | Target | CF prediction | CF confidence |
| --- | ---: | --- | --- | --- | --- | ---: |
| BUSI | 0 | benign | benign | malignant | normal | 0.9178 |
| BUSI | 13 | normal | normal | malignant | benign | 0.9029 |

Both BUSI invalids are a direct, expected consequence of the untargeted
attack loss (see method doc §1/§4): the optimization found a confident flip
away from the original class, just not to the manifest's specific target
class.

## Cautious Interpretation

- Validity means the trained classifier predicts the manifest target class
  for the generated counterfactual. It does not imply medical plausibility.
- Because the attack loss is untargeted, a "failure" here is not evidence of
  an implementation bug — it can be the correct behavior of the original
  algorithm when the nearest confident flip differs from the fixed manifest
  target.
- Sparsity (changed pixel fraction) differs substantially between datasets
  (5.3% BUSI vs. 1.8% Pneumonia) because `theta` was calibrated separately per
  dataset/autoencoder to compensate for very different raw encoder-space
  prototype distances (see method doc §4); this is a calibration artifact,
  not a difference in method fidelity.

## Output Check

- BUSI: 30 PNG files in `results/final/cfproto_encoder_knn/busi`
- Pneumonia: 40 PNG files in `results/final/cfproto_encoder_knn/pneumonia`
- Total PNG files: 70
- No command errors were reported during the two final runs.
