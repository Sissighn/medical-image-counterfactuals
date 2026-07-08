# Counterfactual Method Comparison

## Compared Methods

This project compares four counterfactual explanation directions for medical
image classification:

1. CFProto-nearer prototype-guided optimization baseline
2. Retrieval-based nearest-unlike-neighbor baseline
3. SEDC-T-style segment replacement
4. DVCE-style diffusion-guided generation

All methods use fixed evaluation manifests where feasible. DVCE is evaluated on
the first five fixed manifest samples because diffusion sampling is substantially
more expensive.

## Quantitative Results

| Method | Dataset | Samples | Validity | Mean CF confidence | Mean change | Mean runtime |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| CFProto-nearer prototype-guided optimization baseline | BUSI | 15 | 1.00 | 0.5471 | 0.0084 changed pixel fraction | 7.43s |
| CFProto-nearer prototype-guided optimization baseline | Pneumonia | 20 | 0.90 | 0.5767 | 0.0108 changed pixel fraction | 8.58s |
| Retrieval-based nearest-unlike-neighbor baseline | BUSI | 15 | 1.00 | 0.8191 | 0.8516 changed pixel fraction | 0.01s |
| Retrieval-based nearest-unlike-neighbor baseline | Pneumonia | 20 | 1.00 | 0.6496 | 0.8741 changed pixel fraction | 0.01s |
| SEDC-T original-style best-first | BUSI | 15 | 0.80 | 0.6674 | 0.1517 changed pixel fraction | 6.59s |
| SEDC-T original-style best-first | Pneumonia | 20 | 0.55 | 0.7343 | 0.1410 changed pixel fraction | 13.78s |
| SEDC-T project variant | BUSI | 15 | 0.80 | 0.6376 | 0.1471 changed pixel fraction | 0.56s |
| SEDC-T project variant with lung-field ROI | Pneumonia | 20 | 0.45 | 0.7639 | 0.1510 changed pixel fraction | 0.39s |
| SEDC-T tuned project variant | BUSI | 15 | 0.80 | 0.6050 | 0.1698 changed pixel fraction | 1.00s |
| SEDC-T tuned project variant | Pneumonia | 20 | 0.60 | 0.6800 | 0.1552 changed pixel fraction | 1.21s |
| DVCE-style diffusion-guided generation | BUSI | 5 | 1.00 | 0.7034 | 0.3569 changed pixels above threshold | 8.86s |
| DVCE-style diffusion-guided generation | Pneumonia | 5 | 0.80 | 0.7219 | 0.1654 changed pixels above threshold | 9.49s |
| DVCE-style with Pneumonia fine-tuned checkpoint | Pneumonia | 5 | 0.80 | 0.6937 | 0.2469 changed pixels above threshold | 15.63s |

The generated central summary is:

```text
results/fixed_evaluation_summary.md
```

## Interpretation

The CFProto-nearer prototype-guided optimization baseline is the only retained
prototype-guided method. It uses encoder-space target-class kNN prototypes,
adaptive c-search, elastic-net selection, polynomial learning-rate decay, and a
targeted margin-style attack loss. It is methodically aligned with CFProto, but
it is not a full Alibi `CounterfactualProto` reproduction. FISTA/shrinkage,
TrustScore, the original TensorFlow graph, and the original Alibi k-d-tree
machinery are not fully reproduced.

Retrieval-NUN reaches 1.00 validity because it retrieves real target-class
training images that are correctly classified by the model. It is intuitive as
a nearest unlike case, but it is not a minimal edit of the original image.

SEDC-T gives localized segment-level changes and is often easier to discuss
visually. Its validity is lower, especially on Pneumonia, where diffuse model
cues make segment replacement difficult.

DVCE covers the generative direction. It can create model-valid samples on the
small fixed subset, but outputs remain sensitive to checkpoint and guidance
settings and must be interpreted as feasibility results.

Validity means that the model prediction changed to the target class. It does
not imply medical plausibility or clinical causality.
