# Final Method Summary

## Scope

This file summarizes the current method set used for the seminar project. The
prototype-guided track has been cleaned up: the retained main prototype-guided
method is the CFProto-nearer encoder feature-map configuration, with
bottleneck256 and bottleneck1024 kept only as ablations.

## Baseline Classifiers

| Dataset | Classes | Model | Accuracy | Weighted F1 |
| --- | --- | --- | ---: | ---: |
| BUSI | benign, malignant, normal | ResNet18 pretrained | 0.8390 | 0.8365 |
| Pneumonia | NORMAL, PNEUMONIA | ResNet18 pretrained | 0.8782 | 0.8732 |

## Method 1: CFProto-Nearer Prototype-Guided Optimization Baseline

The method optimizes the input image toward the fixed manifest target class
using:

- autoencoder encoder-space target-class kNN mean prototypes,
- adaptive attack-constant search,
- elastic-net best-counterfactual selection,
- polynomial learning-rate decay,
- targeted margin-style attack loss.

It remains a PyTorch-based implementation in this medical image pipeline, not a
full Alibi CFProto reproduction. FISTA/shrinkage, TrustScore, the original
TensorFlow graph, and the original Alibi k-d-tree machinery are not fully
reproduced.

| Dataset | Samples | Validity | Mean CF confidence | Mean changed pixel fraction | Mean runtime |
| --- | ---: | ---: | ---: | ---: | ---: |
| BUSI | 15 | 1.00 | 0.5471 | 0.0084 | 7.43s |
| Pneumonia | 20 | 0.90 | 0.5767 | 0.0108 | 8.58s |

### CFProto Autoencoder Bottleneck Ablations

The bottleneck256 and bottleneck1024 variants are retained as ablations, not as
main methods. They test whether compact autoencoder latent prototypes improve
the prototype-guided optimization. In the current fixed-manifest runs they are
less stable than the encoder feature-map configuration and change much larger
image areas.

| Variant | Dataset | Samples | Validity | Mean CF confidence | Mean changed pixel fraction | Mean runtime |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Bottleneck256 | BUSI | 15 | 0.67 | 0.6845 | 0.6168 | 8.65s |
| Bottleneck256 | Pneumonia | 20 | 0.50 | 0.7537 | 0.6666 | 8.73s |
| Bottleneck1024 | BUSI | 15 | 0.67 | 0.6590 | 0.7053 | 14.60s |
| Bottleneck1024 | Pneumonia | 20 | 0.55 | 0.7292 | 0.6312 | 13.78s |

## Method 2: Retrieval-Based Nearest-Unlike-Neighbor Baseline

Retrieval-NUN retrieves the nearest real training image from the manifest target
class in ResNet18 penultimate feature space. It is a case-based baseline, not a
minimal image edit.

| Dataset | Samples | Validity | Mean CF confidence | Mean changed pixel fraction | Mean embedding distance | Mean runtime |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BUSI | 15 | 1.00 | 0.8191 | 0.8516 | 0.2953 | 0.01s |
| Pneumonia | 20 | 1.00 | 0.6496 | 0.8741 | 0.2493 | 0.01s |

## Method 3: SEDC-T-Style Segment Replacement

SEDC-T changes image segments and queries the classifier for a target-class
flip. The original-style best-first run is the method-faithfulness reference.
The only retained ablation is a Pneumonia-specific lung-field ROI constraint.

| Variant | Dataset | Samples | Validity | Mean CF confidence | Mean changed pixel fraction | Mean runtime |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Original-style best-first | BUSI | 15 | 0.80 | 0.6343 | 0.2640 | 7.51s |
| Original-style best-first | Pneumonia | 20 | 0.55 | 0.6702 | 0.3377 | 14.41s |
| Lung-field ROI ablation | Pneumonia | 20 | 0.50 | 0.7775 | 0.1843 | 15.48s |

## Method 4: DVCE-Style Diffusion-Guided Generation

DVCE-style generation covers the generative direction. The OpenAI checkpoint and
the Pneumonia fine-tuned checkpoint are reported as checkpoint/guidance states
of the same feasibility method.

| Variant | Dataset | Samples | Validity | Mean CF confidence | Mean changed pixels above threshold | Mean runtime |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| OpenAI checkpoint | BUSI | 5 | 1.00 | 0.7034 | 0.3569 | 8.86s |
| OpenAI checkpoint | Pneumonia | 5 | 0.80 | 0.7219 | 0.1654 | 9.49s |
| Pneumonia fine-tuned checkpoint | Pneumonia | 5 | 0.80 | 0.6937 | 0.2469 | 15.63s |

## Main Takeaway

The methods expose different trade-offs. CFProto-nearer optimization is compact
and model-valid, Retrieval-NUN is intuitive but not minimal, SEDC-T is more
localized but less consistently valid, and DVCE is generative but sensitive to
checkpoint and guidance settings. Model validity must not be equated with
medical plausibility.
