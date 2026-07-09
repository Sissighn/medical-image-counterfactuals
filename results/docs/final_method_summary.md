# Final Method Summary

## Scope

This file summarizes the current method set used for the seminar project. The
prototype-guided method is CFProto (original-style), a faithful PyTorch port of
alibi's `CounterfactualProto` using a bottleneck-256 autoencoder; the earlier
feature-map and bottleneck1024 experiments have been replaced by this single
calibrated configuration.

## Baseline Classifiers

| Dataset | Classes | Model | Accuracy | Weighted F1 |
| --- | --- | --- | ---: | ---: |
| BUSI | benign, malignant, normal | ResNet18 pretrained | 0.8390 | 0.8365 |
| Pneumonia | NORMAL, PNEUMONIA | ResNet18 pretrained | 0.8782 | 0.8732 |

## Method 1: CFProto (Original-Style Prototype-Guided Optimization)

The method follows alibi's `CounterfactualProto` faithfully:

- FISTA optimization with shrinkage-thresholding and Nesterov momentum,
- an untargeted hinge attack loss on the original class,
- a sum-based loss `c*L_attack + L2 + beta*L1 + gamma*L_AE + theta*L_proto`,
- binary search over the attack constant `c` (x10 escalation),
- encoder-space class prototypes built from the classifier's own predictions
  on the training split (kNN mean),
- elastic-net (L2 + beta*L1) best-counterfactual selection.

Deliberate differences from the original are the framework (PyTorch instead of
the TensorFlow 1.x graph) and per-dataset/autoencoder recalibrated `gamma`/
`theta` weights, since all loss terms are sums and their raw magnitude depends
on input and latent dimensionality. Not reproduced: the TensorFlow graph
itself, black-box mode with numerical gradients, categorical variables/
k-d-tree prototypes, and TrustScore filtering (disabled by default in alibi
too). Full Soll-Ist comparison:
`results/final_configs/cfproto_encoder_method_documentation.md`.

| Dataset | Samples | Validity | Mean CF confidence | Mean changed pixel fraction | Mean runtime |
| --- | ---: | ---: | ---: | ---: | ---: |
| BUSI | 15 | 0.87 | 0.6815 | 0.0529 | 46.10s |
| Pneumonia | 20 | 1.00 | 0.5740 | 0.0180 | 46.34s |

The two BUSI failures (of 15) are an expected consequence of the untargeted
attack loss: the optimization found a confident flip away from the original
class that did not land on the manifest's specific fixed target class.

## Method 2: Goyal et al. 2019 Counterfactual Visual Explanations

Instance-based feature-space edit after Goyal et al. (ICML 2019,
arXiv:1904.07451). The ResNet18 is split into a spatial extractor (`layer4`,
7x7x512 cells) and a decision head (GAP + FC). A distractor image from the
target class is retrieved as the nearest correctly classified training image in
pooled feature space, then spatial cells of the query feature map are greedily
swapped for distractor cells (each cell at most once) until the prediction flips
to the target class. A reference implementation is the Goyal baseline in the
Meta repo `facebookresearch/visual-counterfactuals`.

Validity is 1.00 by construction (the full 49-cell budget converges the pooled
feature to the distractor's), so the reported quantity of interest is the number
of edited cells (sparsity). Mean CF confidence sits near 0.5 because the greedy
search stops at the first flip. See
`results/final_configs/goyal_cve_method_documentation.md`.

| Dataset | Samples | Validity | Mean CF confidence | Mean edits (of 49) | Mean changed pixel fraction | Mean runtime |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BUSI | 15 | 1.00 | 0.5279 | 14.0 | 0.2596 | 0.25s |
| Pneumonia | 20 | 1.00 | 0.5231 | 16.15 | 0.3072 | 0.17s |

## Method 3: SEDC-T-Style Segment Replacement

SEDC-T changes image segments and queries the classifier for a target-class
flip. The original-style best-first run is the method-faithfulness reference.
The only retained ablation is a Pneumonia-specific lung-field ROI constraint.

| Variant | Dataset | Samples | Validity | Mean CF confidence | Mean changed pixel fraction | Mean runtime |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Original-style best-first | BUSI | 15 | 0.80 | 0.6343 | 0.2640 | 6.71s |
| Original-style best-first | Pneumonia | 20 | 0.55 | 0.6759 | 0.3270 | 13.92s |
| Lung-field ROI ablation | Pneumonia | 20 | 0.50 | 0.7770 | 0.1745 | 15.23s |

## Method 4: DVCE Original-Style Diffusion-Guided Generation

DVCE generation covers the generative direction. The old free-guidance
prototype runs have been removed. The retained implementation now uses the
original-code-nearer DVCE core:

- `gen_type=p_sample`,
- `timestep_respacing=200`,
- `skip_timesteps=100`,
- `classifier_lambda=0.1`,
- `lp_custom=1.0`,
- `lp_custom_value=0.15`,
- `enforce_same_norms=True`,
- `clip_denoised=False`,
- Cone Projection optional via `--second_model_path` and `--deg_cone_projection`.

The planned checkpoint states are:

| State | Dataset | Status |
| --- | --- | --- |
| OpenAI checkpoint | BUSI and Pneumonia | rerun needed with original-style core |
| Pneumonia fine-tuned checkpoint | Pneumonia | rerun needed with original-style core |
| BUSI fine-tuned checkpoint | BUSI | run after BUSI checkpoint is available |
| Cone Projection | BUSI and Pneumonia | requires PGD-robust second ResNet18 classifiers |

Commands are documented in:

```text
results/final_configs/dvce_original_style_commands.md
results/final_configs/dvce_cone_projection_for_paul.md
```

## Main Takeaway

The methods expose different trade-offs. CFProto (original-style) optimization
is compact and mostly model-valid, Goyal et al. 2019 CVE gives sparse localized edits grounded in
real target-class images (validity guaranteed by construction, confidence near
the decision boundary), SEDC-T is more localized but less consistently valid,
and DVCE is generative but still pending fresh fixed-manifest results after the
original-style core update. Model validity must not be equated with medical
plausibility.
