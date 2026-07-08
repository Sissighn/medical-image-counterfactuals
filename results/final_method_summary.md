# Final Method Summary

## Scope

This file summarizes the current method set used for the seminar project. The
prototype-guided track has been cleaned up: the only retained prototype-guided
method is the CFProto-nearer prototype-guided optimization baseline.

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
flip. The original-style best-first run is the method-faithfulness reference;
project/tuned variants are reported separately because they change search speed,
ROI constraints, or segment budgets.

| Variant | Dataset | Samples | Validity | Mean CF confidence | Mean changed pixel fraction | Mean runtime |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Original-style best-first | BUSI | 15 | 0.80 | 0.6674 | 0.1517 | 6.59s |
| Original-style best-first | Pneumonia | 20 | 0.55 | 0.7343 | 0.1410 | 13.78s |
| Project variant | BUSI | 15 | 0.80 | 0.6376 | 0.1471 | 0.56s |
| Project variant with lung-field ROI | Pneumonia | 20 | 0.45 | 0.7639 | 0.1510 | 0.39s |
| Tuned project variant | BUSI | 15 | 0.80 | 0.6050 | 0.1698 | 1.00s |
| Tuned project variant | Pneumonia | 20 | 0.60 | 0.6800 | 0.1552 | 1.21s |

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
