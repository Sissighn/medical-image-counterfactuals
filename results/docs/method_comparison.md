# Counterfactual Method Comparison

## Compared Methods

This project compares four counterfactual explanation directions for medical
image classification:

1. CFProto (original-style) prototype-guided optimization
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
| CFProto (original-style) prototype-guided optimization | BUSI | 15 | 0.87 | 0.6815 | 0.0529 changed pixel fraction | 46.10s |
| CFProto (original-style) prototype-guided optimization | Pneumonia | 20 | 1.00 | 0.5740 | 0.0180 changed pixel fraction | 46.34s |
| Goyal et al. 2019 counterfactual visual explanations | BUSI | 15 | 1.00 | 0.5279 | 0.2596 changed pixel fraction, 14.0 edits | 0.25s |
| Goyal et al. 2019 counterfactual visual explanations | Pneumonia | 20 | 1.00 | 0.5231 | 0.3072 changed pixel fraction, 16.15 edits | 0.17s |
| SEDC-T original-style best-first | BUSI | 15 | 0.80 | 0.6343 | 0.2640 changed pixel fraction | 6.71s |
| SEDC-T original-style best-first | Pneumonia | 20 | 0.55 | 0.6759 | 0.3270 changed pixel fraction | 13.92s |
| SEDC-T lung-field ROI ablation | Pneumonia | 20 | 0.50 | 0.7770 | 0.1745 changed pixel fraction | 15.23s |

The generated central summary is:

```text
results/docs/fixed_evaluation_summary.md
```

## Interpretation

CFProto (original-style) follows alibi's `CounterfactualProto` faithfully:
FISTA optimization with shrinkage-thresholding and Nesterov momentum, an
untargeted hinge attack loss, a sum-based loss combining attack, L2, autoencoder,
and prototype terms, binary c-search, and encoder-space class prototypes built
from the classifier's own predictions on the training split. `gamma`/`theta`
are recalibrated per dataset/autoencoder since all loss terms are sums and their
raw magnitude depends on input and latent dimensionality (0.87 validity on BUSI
with `theta=0.5`, 1.00 on Pneumonia with `theta=0.05`; the two BUSI misses stem
directly from the untargeted attack loss finding a valid flip to a class other
than the manifest's fixed target). Not reproduced: the TensorFlow graph itself
(reimplemented in PyTorch), black-box mode with numerical gradients, categorical
variables/k-d-tree prototypes, and TrustScore filtering (disabled by default in
alibi too). Full documentation:
`results/final_configs/cfproto_encoder_method_documentation.md`.

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
