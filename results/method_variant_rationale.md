# Method Variant Rationale

This document explains why the project contains several states or variants of
the counterfactual methods. It is intended to make the experimental design
defensible: each variant has a specific purpose, and project-specific changes
are not presented as original method mechanisms.

## Guiding Principle

The comparison separates three levels:

1. **Main method result**: the variant used for the primary method comparison.
2. **Method-faithfulness check**: a variant closer to the original algorithmic
   mechanism, used to show that the implementation is not only an arbitrary
   project heuristic.
3. **Ablation / tuning result**: a controlled project-specific change used to
   test whether a failure is caused by parameter choice or by a method/data
   limitation.

This distinction is important because counterfactual methods can be model-valid
without being visually or medically plausible.

## Current Method Roles

| Method family | Main role in this project | Primary result | Additional states | Why additional states exist |
| --- | --- | --- | --- | --- |
| Prototype-guided optimization | Technical high-validity baseline | Fixed BUSI/Pneumonia runs | Method audit and plausibility ablation | To avoid claiming full Alibi CFProto reproduction and to test stronger regularization |
| Retrieval-NUN | Case-based nearest unlike baseline | Fixed BUSI/Pneumonia runs | none | To add an interpretable non-generative baseline using real target-class images |
| SEDC-T | Region-based/localized counterfactuals | Original-style best-first and project variant | Tuning ablation | To separate method fidelity from practical constraints |
| DVCE | Generative counterfactual feasibility | OpenAI checkpoint fixed 5-sample runs | Pneumonia fine-tuned checkpoint and guidance study | To test whether a medical diffusion prior improves outputs |

## Prototype-Guided Optimization Baseline

### Why This Is A Baseline

The implementation uses the idea of class prototypes, but it does not reproduce
all mechanisms of Alibi CFProto. In particular, it does not use the Alibi
`CounterfactualProto` explainer, an autoencoder/encoder, k-nearest encoded
prototypes, k-d trees, trust-score filtering, or the original `c_steps` search.

### Why It Is Still Useful

It is useful as a baseline because it:

- works with the existing PyTorch ResNet18 classifiers,
- supports multiclass BUSI and binary Pneumonia,
- produces valid target-class counterfactuals on the fixed manifests,
- provides a strong contrast to more localized or generative methods.

### Plausibility Ablation

A conservative plausibility-focused ablation was added without changing the
method core. It keeps the prototype-guided objective, but increases image and
smoothness regularization and lowers the maximum perturbation:

```text
lambda_l2=20.0, lambda_tv=0.5, max_delta=0.08
```

This keeps validity at 1.00 on both fixed manifests while reducing changed
pixel fraction:

| Dataset | Baseline changed fraction | Ablation changed fraction | Baseline L1/MAD | Ablation L1/MAD |
| --- | ---: | ---: | ---: | ---: |
| BUSI | 0.0559 | 0.0315 | 0.0126 | 0.0105 |
| Pneumonia | 0.1442 | 0.0934 | 0.0167 | 0.0140 |

The ablation is useful for presentation because it produces less aggressive
prototype-guided examples. It should still be reported as a technical baseline:
the changes can remain diffuse and are not necessarily medically causal.

### How To Report It

Use:

```text
prototype-guided optimization baseline
```

Do not use it as the main empirical representative of full Alibi CFProto. The
related CFProto literature can be discussed as motivation, but the implemented
method should be described as a project-specific baseline.

Use the stronger-regularized run as:

```text
prototype-guided plausibility ablation
```

## Retrieval-Based Nearest-Unlike-Neighbor Baseline

### Why This Is A Separate Baseline

Retrieval-NUN does not optimize, replace segments, or generate a new image. For
each fixed evaluation sample it uses the manifest target class and retrieves the
nearest real training image from that class in the trained ResNet18 penultimate
embedding space. Candidate images are filtered so that their true label and
model prediction both equal the target class.

This makes the method useful as a simple, stable, case-based baseline. It is
closest to the "retrieval-based prototype counterfactual" direction discussed
for the project, but it should be described precisely as nearest unlike
neighbor retrieval rather than as an edited counterfactual.

### Why It Is Useful

It is useful because it:

- uses real medical images instead of generated artifacts,
- supports BUSI and Pneumonia without additional training,
- reaches 1.00 validity by construction on the fixed manifests,
- gives a visually intuitive comparison case for the target class.

### Main Limitation

It is not a minimal edit of the original image. The difference map compares two
different real images and therefore includes patient anatomy, acquisition,
positioning, and dataset variation. High changed pixel fraction is expected and
does not mean the method failed; it means the method answers a different
question than edit-based approaches.

### How To Report It

Use:

```text
retrieval-based nearest-unlike-neighbor baseline
```

Report it as a case-based baseline, not as a generated or minimally edited
counterfactual.

## SEDC-T

### Why There Are Multiple SEDC-T States

SEDC-T is the method where implementation fidelity matters most, because the
original method is relatively concrete: segment the image, replace/remove
segments, query the classifier, and search for a targeted class change.

The project therefore reports two important variants:

- **Original-style best-first**: closer to the reference search, no ROI,
  target-score best-first expansion.
- **Project variant**: faster greedy search, optional smaller changed-area
  selection, optional simple Pneumonia lung-field ROI.

### Original-Style Best-First

This is the safer result when discussing the implemented SEDC-T method. It is
not a byte-for-byte copy of the original repository because the project uses a
PyTorch medical classifier, SLIC segmentation, fixed manifests, and a segment
budget/timeout. However, it follows the core SEDC-T mechanism more closely than
the project variant.

Result:

| Dataset | Samples | Validity | Mean changed pixel fraction | Runtime |
| --- | ---: | ---: | ---: | ---: |
| BUSI | 15 | 0.80 | 0.1517 | 6.59s |
| Pneumonia | 20 | 0.55 | 0.1410 | 13.78s |

### Project Variant

This variant is kept because it is much faster and allows controlled
constraints, but it must be reported as a project-specific implementation
choice.

Result:

| Dataset | Constraint | Samples | Validity | Mean changed pixel fraction | Runtime |
| --- | --- | ---: | ---: | ---: | ---: |
| BUSI | none | 15 | 0.80 | 0.1471 | 0.56s |
| Pneumonia | lung-field ROI | 20 | 0.45 | 0.1510 | 0.39s |

### Tuning Ablation

The SEDC-T tuning ablation tested whether low Pneumonia validity was simply a
parameter issue. The best tuned Pneumonia setting reached 0.60 validity:

```text
greedy_minimal, roi_mode=none, replacement_mode=blur, max_segments=8
```

This only improves from 11/20 to 12/20 compared with the original-style run.
Increasing the segment budget further increases changed area without increasing
validity. This supports the interpretation that Pneumonia is difficult for
segment replacement because the classifier may rely on broad or diffuse cues.

### How To Report It

Use the original-style best-first run as the main SEDC-T result when discussing
method fidelity. Use the project variant and tuning results as ablations:

```text
The original-style SEDC-T run is the method-faithfulness result. The
greedy/ROI/tuned variants are project-specific ablations used to test runtime,
locality, and anatomical constraints.
```

## DVCE

### Why DVCE Has Multiple Checkpoints

DVCE is the generative method category. The original available checkpoint is an
OpenAI unconditional 256x256 diffusion checkpoint. Although it is
unconditional, it still carries a natural-image prior and is not medical-domain
specific.

A Pneumonia fine-tuned checkpoint was therefore tested to see whether a more
medical diffusion prior improves the generated counterfactuals.

### Why Guidance Was Tuned

The original OpenAI checkpoint and the Pneumonia fine-tuned checkpoint did not
behave identically under the same guidance parameters. The fine-tuned checkpoint
needed adjusted classifier guidance and skip timesteps to recover useful
validity. This is reported as a checkpoint/guidance ablation, not as a separate
method.

### How To Report It

Use:

```text
DVCE-style diffusion-guided feasibility prototype
```

Treat the OpenAI checkpoint as the current baseline DVCE-style result and the
Pneumonia fine-tuned checkpoint as an ablation. Report clearly that DVCE is only
evaluated on five fixed samples per dataset because sampling is substantially
more expensive.

## Recommended Paper Framing

The paper should not argue that one method is universally best. The stronger
argument is that the methods reveal a trade-off:

- Prototype-guided optimization has the highest model validity but weak
  locality and limited medical plausibility.
- Retrieval-NUN provides real target-class cases and avoids generated artifacts,
  but it is not minimal and cannot isolate causal image evidence.
- SEDC-T has stronger locality and clearer region-level explanations, but lower
  validity, especially on Pneumonia.
- DVCE covers the generative method category, but current outputs remain
  sensitive to checkpoint and guidance settings.

The variants support this conclusion rather than weakening it. They show that
the results were checked for implementation fidelity and that project-specific
changes were evaluated as ablations instead of being silently mixed into the
main method claims.
