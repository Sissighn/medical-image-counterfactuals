# CFProto Encoder-KNN Final Fixed-Manifest Summary

This summary reports the final fixed-manifest runs for the CFProto-nearer
prototype-guided optimization baseline. The runs use encoder-space local
KNN prototypes, adaptive binary-style c-search, elastic-net selection, and
polynomial learning-rate decay. They do not constitute a full Alibi
`CounterfactualProto` reproduction.

## Exact Commands

### BUSI

```bash
PYTHONPATH=. python scripts/run_cfproto_pytorch.py \
  --model_path models/busi_resnet18_pretrained.pth \
  --dataset_path data/processed/BUSI \
  --output_dir results/final/cfproto_encoder_knn/busi \
  --manifest_path results/evaluation_manifests/busi_balanced_5_per_class_second_best.json \
  --steps 300 \
  --learning_rate 0.01 \
  --attack_loss cw_hinge \
  --attack_const 1.0 \
  --c_init 1.0 \
  --c_steps 3 \
  --c_search_mode adaptive_binary \
  --kappa 0.0 \
  --lambda_l1 0.01 \
  --lambda_l2 5.0 \
  --lambda_tv 0.2 \
  --lambda_proto 0.05 \
  --autoencoder_path models/autoencoder_busi.pth \
  --gamma 0.0 \
  --prototype_space encoder \
  --prototype_mode knn_mean \
  --prototype_k 3 \
  --selection_metric elastic_net \
  --beta 0.1 \
  --lr_schedule polynomial \
  --max_delta 0.12 \
  --perturbation_resolution 28 \
  --batch_size 16
```

### Pneumonia

```bash
PYTHONPATH=. python scripts/run_cfproto_pytorch.py \
  --model_path models/pneumonia_resnet18_pretrained.pth \
  --dataset_path data/processed/Pneumonia \
  --output_dir results/final/cfproto_encoder_knn/pneumonia \
  --manifest_path results/evaluation_manifests/pneumonia_balanced_10_per_class_second_best.json \
  --steps 300 \
  --learning_rate 0.01 \
  --attack_loss cw_hinge \
  --attack_const 1.0 \
  --c_init 1.0 \
  --c_steps 3 \
  --c_search_mode adaptive_binary \
  --kappa 0.0 \
  --lambda_l1 0.01 \
  --lambda_l2 5.0 \
  --lambda_tv 0.2 \
  --lambda_proto 0.05 \
  --autoencoder_path models/autoencoder_pneumonia.pth \
  --gamma 0.0 \
  --prototype_space encoder \
  --prototype_mode knn_mean \
  --prototype_k 3 \
  --selection_metric elastic_net \
  --beta 0.1 \
  --lr_schedule polynomial \
  --max_delta 0.12 \
  --perturbation_resolution 28 \
  --batch_size 16
```

## Final Parameters

| Parameter | Value |
| --- | --- |
| `prototype_space` | `encoder` |
| `prototype_mode` | `knn_mean` |
| `prototype_k` | `3` |
| `c_search_mode` | `adaptive_binary` |
| `selection_metric` | `elastic_net` |
| `beta` | `0.1` |
| `lr_schedule` | `polynomial` |
| `attack_loss` | `cw_hinge` |
| `attack_const` | `1.0` |
| `c_init` | `1.0` |
| `c_steps` | `3` |
| `kappa` | `0.0` |
| `lambda_l1` | `0.01` |
| `lambda_l2` | `5.0` |
| `lambda_tv` | `0.2` |
| `lambda_proto` | `0.05` |
| `gamma` | `0.0` |
| `max_delta` | `0.12` |
| `perturbation_resolution` | `28` |
| `steps` | `300` |
| `learning_rate` | `0.01` |
| `force_grayscale` | `True` |
| `target_class` | manifest target (`record["target_class_index"]`) |

## Quantitative Summary

| dataset | num_samples | valid_count | validity | mean_cf_confidence | mean_target_confidence | mean_l1_mad | mean_l2 | mean_linf | mean_changed_pixel_fraction | mean_runtime_seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BUSI | 15 | 15 | 1.000000 | 0.547051 | 0.547051 | 0.006331 | 0.007600 | 0.018075 | 0.008384 | 7.429689 |
| Pneumonia | 20 | 18 | 0.900000 | 0.576668 | 0.476669 | 0.006500 | 0.007939 | 0.023833 | 0.010842 | 8.581645 |

## Metadata Files

- BUSI: `results/final/cfproto_encoder_knn/busi/metadata.json`
- Pneumonia: `results/final/cfproto_encoder_knn/pneumonia/metadata.json`

## Manifest Fairness Check

- BUSI: manifest samples unchanged = True, manifest targets unchanged = True, invalid filtered = False
- Pneumonia: manifest samples unchanged = True, manifest targets unchanged = True, invalid filtered = False

## Invalid Counterfactuals

| Dataset | Manifest sample | True label | Original prediction | Target | CF prediction | Target confidence | CF confidence |
| --- | ---: | --- | --- | --- | --- | ---: | ---: |
| Pneumonia | 11 | PNEUMONIA | PNEUMONIA | NORMAL | PNEUMONIA | 0.000014 | 0.999986 |
| Pneumonia | 16 | PNEUMONIA | PNEUMONIA | NORMAL | PNEUMONIA | 0.000001 | 0.999999 |

## Cautious Interpretation

- This method is CFProto-nearer because it uses autoencoder encoder-space local target-class KNN prototypes, adaptive binary-style c-search, elastic-net selection, and polynomial learning-rate decay.
- It is still not a full Alibi `CounterfactualProto` reproduction. FISTA/shrinkage, TrustScore, the original TensorFlow graph structure, and the original Alibi k-d-tree machinery were not fully reproduced.
- Validity means that the trained classifier predicts the manifest target class for the generated counterfactual. It does not imply medical plausibility.
- The changes can remain diffuse or visually subtle. They should be interpreted as model-behavior counterfactuals, not as clinically causal image edits.

## Output Check

- BUSI: 30 PNG files in `results/final/cfproto_encoder_knn/busi`
- Pneumonia: 40 PNG files in `results/final/cfproto_encoder_knn/pneumonia`
- Total PNG files: 70
- No command errors were reported during the two final runs.
