# DVCE Medical Checkpoint Comparison

This file compares the original diffusion checkpoint with the Pneumonia
fine-tuned checkpoint for the DVCE-style feasibility prototype.

The comparison uses the first five samples from the fixed Pneumonia evaluation
manifest:

```text
results/evaluation_manifests/pneumonia_balanced_10_per_class_second_best.json
```

All runs use the same medical ResNet18 classifier and the same target strategy.
Only the diffusion checkpoint and selected generation parameters differ.

## Checkpoints

| Checkpoint | Description |
| --- | --- |
| `256x256_diffusion_uncond.pt` | Original OpenAI unconditional 256x256 diffusion checkpoint used by the DVCE workflow |
| `ema_0.9999_005000.pt` | Pneumonia fine-tuned EMA checkpoint after 5000 training steps |

The EMA checkpoint is used for sampling because it is the standard smoothed
model state for diffusion generation.

## Quantitative Comparison

| Variant | Checkpoint | Guidance settings | Skip timesteps | Validity | Mean CF confidence | Mean absolute difference | Changed pixels > 0.05 | Runtime |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Current baseline | OpenAI checkpoint | classifier 80, similarity 20 | 44 | 0.80 | 0.7219 | 0.0332 | 0.1654 | 9.49s |
| Medical checkpoint, original settings | Pneumonia fine-tuned EMA | classifier 80, similarity 20 | 44 | 0.20 | 0.7673 | 0.0324 | 0.1615 | 9.50s |
| Medical checkpoint, stronger guidance | Pneumonia fine-tuned EMA | classifier 200, similarity 10 | 44 | 0.40 | 0.7236 | 0.0324 | 0.1615 | 9.30s |
| Medical checkpoint, compromise setting | Pneumonia fine-tuned EMA | classifier 200, similarity 10 | 40 | 0.80 | 0.6937 | 0.0504 | 0.2469 | 15.63s |
| Medical checkpoint, strongest tested setting | Pneumonia fine-tuned EMA | classifier 200, similarity 10 | 30 | 1.00 | 0.8527 | 0.0883 | 0.4453 | 25.96s |

## Interpretation

The fine-tuned checkpoint loads correctly and can be used inside the existing
DVCE-style pipeline. However, the original DVCE generation parameters do not
transfer directly to the fine-tuned checkpoint. With the original settings, the
Pneumonia fine-tuned checkpoint reaches only 1/5 valid counterfactuals.

Increasing classifier guidance improves validity, and allowing more diffusion
steps from the noisy state improves it further. The best quantitative validity
in this small test is reached by the `skip_timesteps=30` setting, but this
variant produces visibly stronger noise and therefore weaker visual
plausibility. The `skip_timesteps=40` setting is currently the best compromise:
it reaches the same 4/5 validity as the original checkpoint while using the
medical fine-tuned checkpoint, with a moderate increase in changed pixels and
runtime.

Qualitatively, the fine-tuned checkpoint does not yet fully solve the
interpretability problem. The generated counterfactuals can still contain
substantial noise-like texture. Therefore, the result should be reported as an
important technical improvement and parameter study, not as a fully
clinically-plausible DVCE solution.

## Recommended Reporting

For the seminar paper, the safest wording is:

```text
The Pneumonia fine-tuned diffusion checkpoint could be integrated successfully.
Compared with the original checkpoint, it requires adjusted guidance settings.
With a compromise setting, it reaches the same validity on the fixed
five-sample subset, but the visual plausibility remains limited by noise-like
artifacts. This confirms that diffusion fine-tuning is technically feasible,
while also showing that checkpoint adaptation alone is not sufficient for
clinically robust counterfactual generation.
```

The strongest visual example should be selected from the `skip_timesteps=40`
run, because it balances validity and image preservation better than the
`skip_timesteps=30` run.
