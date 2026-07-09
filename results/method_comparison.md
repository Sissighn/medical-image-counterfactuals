# Counterfactual Method Comparison

## Compared Methods

This project compares four counterfactual explanation directions for medical
image classification:

1. CFProto-nearer prototype-guided optimization baseline
2. Goyal et al. 2019 counterfactual visual explanations (feature-cell swaps)
3. SEDC-T-style segment replacement
4. DVCE original-style diffusion-guided generation

All methods use fixed evaluation manifests where feasible. DVCE is evaluated on
the first five fixed manifest samples because diffusion sampling is substantially
more expensive. The earlier free-guidance DVCE prototype rows have been removed;
DVCE should be rerun with the original-style core before reporting final
quantitative values.

## Quantitative Results

| Method | Dataset | Samples | Validity | Mean CF confidence | Mean change | Mean runtime |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| CFProto-nearer prototype-guided optimization baseline | BUSI | 15 | 1.00 | 0.5471 | 0.0084 changed pixel fraction | 7.43s |
| CFProto-nearer prototype-guided optimization baseline | Pneumonia | 20 | 0.90 | 0.5767 | 0.0108 changed pixel fraction | 8.58s |
| CFProto bottleneck256 ablation | BUSI | 15 | 0.67 | 0.6845 | 0.6168 changed pixel fraction | 8.65s |
| CFProto bottleneck256 ablation | Pneumonia | 20 | 0.50 | 0.7537 | 0.6666 changed pixel fraction | 8.73s |
| CFProto bottleneck1024 ablation | BUSI | 15 | 0.67 | 0.6590 | 0.7053 changed pixel fraction | 14.60s |
| CFProto bottleneck1024 ablation | Pneumonia | 20 | 0.55 | 0.7292 | 0.6312 changed pixel fraction | 13.78s |
| Goyal et al. 2019 counterfactual visual explanations | BUSI | 15 | 1.00 | 0.5279 | 0.2596 changed pixel fraction, 14.0 edits | 0.25s |
| Goyal et al. 2019 counterfactual visual explanations | Pneumonia | 20 | 1.00 | 0.5231 | 0.3072 changed pixel fraction, 16.15 edits | 0.17s |
| SEDC-T original-style best-first | BUSI | 15 | 0.80 | 0.6343 | 0.2640 changed pixel fraction | 6.71s |
| SEDC-T original-style best-first | Pneumonia | 20 | 0.55 | 0.6759 | 0.3270 changed pixel fraction | 13.92s |
| SEDC-T lung-field ROI ablation | Pneumonia | 20 | 0.50 | 0.7770 | 0.1745 changed pixel fraction | 15.23s |

The generated central summary is:

```text
results/fixed_evaluation_summary.md
```

## Interpretation

The CFProto-nearer encoder feature-map configuration is the retained main
prototype-guided method. It uses encoder-space target-class kNN prototypes,
adaptive c-search, elastic-net selection, polynomial learning-rate decay, and a
targeted margin-style attack loss. The bottleneck256 and bottleneck1024 rows are
autoencoder/prototype-space ablations. They are methodically interesting, but
they are not treated as improvements because they reduce validity and increase
the changed pixel fraction in the current fixed-manifest runs. The implementation
is aligned with CFProto, but it is not a full Alibi `CounterfactualProto`
reproduction. FISTA/shrinkage, TrustScore, the original TensorFlow graph, and
the original Alibi k-d-tree machinery are not fully reproduced.

Goyal et al. 2019 CVE reaches 1.00 validity by construction: with the full
49-cell budget the pooled feature converges to the distractor's, so the
prediction is guaranteed to flip. The informative metric is the number of edited
feature cells (mean 14.0 on BUSI, 16.15 on Pneumonia of 49), i.e. how sparse the
swap is. Mean CF confidence sits near 0.5 because the greedy search stops at the
first flip, prioritizing sparsity over margin. The edits are grounded in a real
target-class distractor image and localized to a coarse 7x7 cell grid.

SEDC-T gives localized segment-level changes and is often easier to discuss
visually. Its validity is lower, especially on Pneumonia, where diffuse model
cues make segment replacement difficult.

DVCE covers the generative direction. The retained code path is now closer to
the local original DVCE implementation: it uses `p_sample`, evaluates classifier
and distance guidance on `pred_xstart`, and normalizes guidance terms against
`eps=model_output` when `enforce_same_norms=True`. Final DVCE rows should be
reported after rerunning the OpenAI checkpoint and the medical checkpoint
states with this implementation.

Validity means that the model prediction changed to the target class. It does
not imply medical plausibility or clinical causality.
