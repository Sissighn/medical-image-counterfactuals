# CFProto-Nearer Prototype-Guided Final Configuration

This file documents the final fixed-manifest configuration used for the
CFProto-nearer prototype-guided method.

## Method Role

- Method name: `cfproto_encoder_knn`
- Description: CFProto-nearer prototype-guided optimization baseline
- Main variant: `prototype_space=encoder`, `prototype_mode=knn_mean`
- Legacy variant: earlier ResNet/class-mean prototype-guided runs, retained only
  as legacy baselines or ablations

The final variant uses local target-class prototypes in the frozen
ConvAutoencoder encoder space. This is closer to the CFProto idea than the
earlier ResNet18 classifier-feature prototype space. It is still a
PyTorch-based CFProto-nearer implementation, not a full Alibi
`CounterfactualProto` reproduction.

## Dataset-Specific Paths

| Dataset | Classifier checkpoint | Autoencoder checkpoint | Manifest | Final output |
| --- | --- | --- | --- | --- |
| BUSI | `models/busi_resnet18_pretrained.pth` | `models/autoencoder_busi.pth` | `results/evaluation_manifests/busi_balanced_5_per_class_second_best.json` | `results/final/cfproto_encoder_knn/busi/` |
| Pneumonia | `models/pneumonia_resnet18_pretrained.pth` | `models/autoencoder_pneumonia.pth` | `results/evaluation_manifests/pneumonia_balanced_10_per_class_second_best.json` | `results/final/cfproto_encoder_knn/pneumonia/` |

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
| `force_grayscale` | `true` |
| `target_strategy` | manifest targets only |
| `batch_size` | `16` |

## Rationale

`prototype_space=encoder` and `prototype_mode=knn_mean` are used because local
target-class prototypes in an autoencoder representation are methodically
closer to CFProto than classifier-feature class means. The adaptive c-search,
elastic-net selection score, and polynomial learning-rate schedule are included
as CFProto-nearer mechanisms after small method checks showed no obvious
technical instability.

`gamma=0.0` is kept because the autoencoder reconstruction term was evaluated
as an ablation and did not show robust additional benefit as a main
configuration. This keeps the final method focused on encoder-space prototype
guidance rather than adding another tuned plausibility term.

The fixed evaluation manifests are not changed by this method. In manifest
mode, samples and target classes come directly from the manifest, and invalid
counterfactuals remain part of the metadata.
