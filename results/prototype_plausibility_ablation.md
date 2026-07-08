# Prototype-Guided Plausibility Ablation

This ablation tests whether the prototype-guided optimization baseline can
produce less diffuse image changes without changing the method core.

Note: these results were generated with the earlier cross-entropy based
prototype-guided objective. The current script now also supports CFProto-aligned
options such as `cw_hinge`, optional `lambda_l1`, prototype-distance target
selection, a symmetric geometric attack-constant search, encoder-space
prototypes via `--prototype_space encoder`, and an optional autoencoder
reconstruction term via `--autoencoder_path` and `--gamma`. Those newer options
are not reflected in the numbers below unless separate fixed-manifest runs are
generated.

## Setup

The baseline method remains the same:

- target-class optimization with the earlier cross-entropy attack loss,
- feature-prototype loss,
- L2 image regularization,
- total variation regularization,
- low-resolution grayscale perturbation.

Only conservative regularization parameters were changed:

| Parameter | Baseline | Plausibility ablation |
| --- | ---: | ---: |
| `lambda_l2` | 5.0 | 20.0 |
| `lambda_tv` | 0.2 | 0.5 |
| `max_delta` | 0.12 | 0.08 |
| `perturbation_resolution` | 28 | 28 |
| `steps` | 300 | 300 |

The goal is not to create a new method, but to test whether stronger smoothness
and smaller perturbation bounds improve visual plausibility.

## Results

| Dataset | Variant | Samples | Validity | Mean CF confidence | Mean L1/MAD | Mean L2 | Changed pixel fraction | Mean runtime |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BUSI | baseline | 15 | 1.00 | 0.9978 | 0.0126 | 0.0153 | 0.0559 | 5.24s |
| BUSI | plausibility ablation | 15 | 1.00 | 0.9953 | 0.0105 | 0.0127 | 0.0315 | 4.78s |
| Pneumonia | baseline | 20 | 1.00 | 0.9928 | 0.0167 | 0.0203 | 0.1442 | 5.69s |
| Pneumonia | plausibility ablation | 20 | 1.00 | 0.9814 | 0.0140 | 0.0172 | 0.0934 | 4.76s |

## Interpretation

The plausibility ablation keeps validity at 1.00 on both fixed manifests while
reducing the average amount of changed pixels. This is useful for presentation
because it provides less aggressive prototype-guided examples.

For individual examples, the thresholded changed-pixel fraction can be reported
as `0.0000` even though the image is not identical. This happens when all pixel
differences remain below the sparsity threshold (`0.03` in the `[0, 1]` image
range). Therefore, Prototype examples should be interpreted together with
`L1/MAD` and `L_inf`, not with the thresholded changed-pixel fraction alone.
Such cases are best understood as very small adversarial-style perturbations:
model-valid, but not visually or medically strong counterfactual evidence.

However, this does not solve the main interpretability limitation of the
prototype-guided baseline. The changes can still be diffuse and should be
reported as model-valid optimization results rather than clinically meaningful
image transformations.

Recommended wording:

```text
The plausibility-focused ablation shows that stronger regularization can reduce
the amount of visible change while keeping validity high. Nevertheless, the
method remains a technical prototype-guided optimization baseline: its outputs
are model-valid but not necessarily localized or medically causal.
```
