# Qualitative Results Interpretation

This document explains the qualitative counterfactual figures in
`results/qualitative_figures/per_method/`. The figures are intended for the
seminar paper and for discussion of model validity versus visual/medical
plausibility.

## How To Read The Figures

Each figure contains one method on one dataset. Rows show selected qualitative
cases:

- **Most balanced valid case**: a valid counterfactual with comparatively small
  change.
- **Highest-confidence valid case**: a valid counterfactual with high target
  confidence.
- **Low-plausibility valid case**: a model-valid counterfactual with visually
  questionable or large changes.
- **Failure case**: an invalid counterfactual, if available for that method.

Not every method has a failure-case row. If all evaluated examples for a method
were valid, the failure row is omitted.

The figure labels report:

- **MAD / L1**: mean absolute difference between original and counterfactual.
- **L_inf**: maximum absolute pixel difference.
- **changed>threshold**: fraction of pixels whose absolute difference exceeds
  the listed threshold.

The difference maps use a fixed scale from `0` to `1`. Very dark maps therefore
mean that the absolute pixel changes are small. They are not contrast-stretched
per image.

## Generated Figures

| Dataset | Method | Figure | Rows |
| --- | --- | --- | --- |
| BUSI | Prototype-guided optimization baseline | `per_method/busi__prototype_guided_optimization_baseline.png` | balanced, high-confidence, low-plausibility |
| BUSI | Prototype-guided plausibility ablation | `per_method/busi__prototype_guided_plausibility_ablation.png` | balanced, high-confidence, low-plausibility |
| BUSI | Retrieval-based nearest-unlike-neighbor baseline | `per_method/busi__retrieval_based_nearest_unlike_neighbor_baseline.png` | balanced, high-confidence, low-plausibility |
| BUSI | SEDC-T original-style best-first | `per_method/busi__sedc_t_original_style_best_first.png` | balanced, high-confidence, low-plausibility, failure |
| BUSI | DVCE-style, OpenAI checkpoint | `per_method/busi__dvce_style_openai_checkpoint.png` | balanced, high-confidence, low-plausibility |
| Pneumonia | Prototype-guided optimization baseline | `per_method/pneumonia__prototype_guided_optimization_baseline.png` | balanced, high-confidence, low-plausibility |
| Pneumonia | Prototype-guided plausibility ablation | `per_method/pneumonia__prototype_guided_plausibility_ablation.png` | balanced, high-confidence, low-plausibility |
| Pneumonia | Retrieval-based nearest-unlike-neighbor baseline | `per_method/pneumonia__retrieval_based_nearest_unlike_neighbor_baseline.png` | balanced, high-confidence, low-plausibility |
| Pneumonia | SEDC-T original-style best-first | `per_method/pneumonia__sedc_t_original_style_best_first.png` | balanced, high-confidence, low-plausibility, failure |
| Pneumonia | SEDC-T tuned project variant | `per_method/pneumonia__sedc_t_tuned_project_variant_none_max_8.png` | balanced, high-confidence, low-plausibility, failure |
| Pneumonia | DVCE-style, OpenAI checkpoint | `per_method/pneumonia__dvce_style_openai_checkpoint.png` | balanced, high-confidence, low-plausibility, failure |
| Pneumonia | DVCE-style, Pneumonia fine-tuned checkpoint | `per_method/pneumonia__dvce_style_pneumonia_fine_tuned_checkpoint.png` | balanced, high-confidence, low-plausibility, failure |

## Method-Level Interpretation

### Prototype-Guided Optimization Baseline

The prototype-guided baseline reaches very high model validity on both datasets
and often flips the classifier with only small absolute pixel changes. In the
figures, this is visible through very dark difference maps and low MAD values.
In some examples the thresholded changed-pixel fraction is close to zero, while
MAD and `L_inf` are still non-zero.

This means the images are not identical. Rather, the model decision can be
changed by very small, diffuse perturbations. This is useful as a technical
high-validity baseline, but weak as a visually interpretable medical
counterfactual.

Strengths:

- highest validity in the fixed evaluation,
- works consistently on BUSI and Pneumonia,
- shows how sensitive the classifier can be to small image changes.

Weaknesses:

- changes are diffuse and difficult to localize,
- visually almost unchanged images can receive a different class label,
- results are closer to adversarial-style perturbations than to medically
  meaningful image edits.

Interpretation:

```text
Prototype-guided optimization is model-valid but visually weak. It is best used
as a technical baseline rather than as the main clinically interpretable method.
```

### Prototype-Guided Plausibility Ablation

The plausibility ablation uses stronger regularization and a smaller allowed
perturbation. Quantitatively, it keeps validity at `1.00` while reducing changed
pixel fraction and MAD on both datasets. Qualitatively, the changes are even
less visible than in the baseline.

This improves preservation of the original image, but it does not fully solve
the interpretability issue. If the classifier flips with even smaller changes,
the result becomes more plausible as an image-preserving perturbation but less
convincing as a human-interpretable medical counterfactual.

Strengths:

- reduces image changes compared with the baseline,
- keeps validity high,
- useful as a conservative ablation.

Weaknesses:

- still diffuse,
- even more adversarial-style in some examples,
- low visual change does not automatically mean medical plausibility.

Interpretation:

```text
The ablation makes the baseline less visually aggressive, but the method remains
a model-optimization baseline rather than a localized medical explanation.
```

### Retrieval-Based Nearest-Unlike-Neighbor Baseline

Retrieval-NUN retrieves a real training image from the target class instead of
generating or editing the original image. The selected neighbor is nearest in
the trained ResNet18 embedding space and is required to be correctly classified
as the target class. This makes the method easy to explain visually: the
counterfactual is an actual target-class case.

The qualitative figures therefore look different from the edit-based methods.
The retrieved image can have different anatomy, positioning, image quality, or
acquisition properties. The difference map is consequently much brighter and
the changed-pixel fraction is high. This is expected and should not be
interpreted as a plotting error. The method is not trying to find the minimal
pixel change; it is showing the nearest unlike real case.

Strengths:

- uses real images and therefore avoids generated artifacts,
- provides an intuitive case-based comparison,
- works on both datasets without additional training,
- reaches 1.00 model validity because candidates are filtered by target class
  and target prediction.

Weaknesses:

- not a minimal edit of the original image,
- differences can reflect patient, anatomy, acquisition, and dataset variation,
- does not isolate causal image evidence for the class change,
- high changed-pixel fraction is expected and limits direct comparison with
  edit-based methods.

Interpretation:

```text
Retrieval-NUN is useful as an interpretable case-based baseline. It should be
compared to the other methods as nearest target-class evidence, not as a
minimal visual transformation.
```

### SEDC-T Original-Style Best-First

SEDC-T is the clearest localized method in the qualitative figures. It modifies
segments instead of applying diffuse pixel-level perturbations. The selected
segments make it easier to see where the model-relevant change was applied.

On BUSI, SEDC-T is comparatively interpretable: valid examples often involve
localized areas, and failure cases can be discussed directly. On Pneumonia, the
method is more difficult. Some valid examples require changes near borders,
shoulders, or broad image regions. This suggests that the classifier may rely on
non-local or non-anatomical cues.

Strengths:

- produces localized region-level explanations,
- easier to discuss visually than Prototype-guided optimization,
- failure cases are informative because one can inspect which segments were
  insufficient.

Weaknesses:

- lower validity than Prototype-guided optimization,
- sensitive to segmentation quality and segment budget,
- on Pneumonia, selected regions are not always medically intuitive.

Interpretation:

```text
SEDC-T provides the best locality, but not the best validity. It is useful for
explaining model behavior, especially when the selected segments are plausible.
```

### SEDC-T Tuned Project Variant

The tuned SEDC-T project variant was evaluated mainly for Pneumonia. It uses a
controlled parameter setting with no ROI, blur replacement, and a maximum of
eight segments. This improves Pneumonia validity only slightly compared with the
original-style run.

The qualitative figures show that valid cases can become more compact in some
examples, but difficult cases remain. The failure row is important: even with a
larger or tuned segment budget, some samples do not reach the target class.

Strengths:

- faster than original-style best-first search,
- slightly higher Pneumonia validity in the fixed evaluation,
- useful as a parameter ablation.

Weaknesses:

- not the main method-faithful SEDC-T result,
- validity improvement is small,
- changed area can increase, so better validity does not necessarily mean
  better interpretability.

Interpretation:

```text
The tuned SEDC-T variant shows that Pneumonia is not solved by simple parameter
tuning. The limitation appears to be method/data-related rather than only a bad
parameter choice.
```

### DVCE-Style With OpenAI Checkpoint

DVCE is the generative method category. The OpenAI checkpoint can produce valid
counterfactuals, but qualitative results show strong sensitivity to the
generation setting. Some examples preserve the image reasonably well, while
others contain visible noise or texture artifacts.

On BUSI, all five evaluated examples were valid, but the low-plausibility row
shows that validity can come with large visual changes. On Pneumonia, there are
valid examples and a failure case, making the trade-off clearer.

Strengths:

- represents a generative counterfactual approach,
- can produce target-class-valid outputs,
- changes can be more image-like than pure pixel optimization in some cases.

Weaknesses:

- evaluated only on five fixed samples because generation is expensive,
- visible artifacts remain,
- OpenAI checkpoint carries a natural-image prior, not a medical-domain prior.

Interpretation:

```text
DVCE with the OpenAI checkpoint is technically feasible, but visual plausibility
is unstable and must be discussed case by case.
```

### DVCE-Style With Pneumonia Fine-Tuned Checkpoint

The Pneumonia fine-tuned checkpoint was tested to reduce the mismatch between
natural-image diffusion priors and medical X-ray images. The checkpoint loads
and works technically, but the qualitative figures show that fine-tuning alone
does not remove all artifacts.

Compared with the OpenAI checkpoint, the fine-tuned checkpoint reaches the same
validity on the five-sample Pneumonia subset in the selected compromise setting,
but the changed-pixel fraction and runtime are higher. Some valid examples are
still noisy, and the failure case remains useful for discussing limitations.

Strengths:

- demonstrates successful integration of a medical fine-tuned diffusion model,
- covers the requested generative method direction,
- enables direct comparison against the OpenAI checkpoint.

Weaknesses:

- only five samples,
- higher runtime,
- artifacts and noise-like changes remain,
- improved domain prior does not automatically imply clinically plausible
  counterfactuals.

Interpretation:

```text
The fine-tuned checkpoint is a meaningful feasibility step, but not a complete
solution for visually or medically robust generative counterfactuals.
```

## Dataset-Level Comparison

### BUSI

BUSI results are generally easier to discuss qualitatively than Pneumonia
results. Ultrasound images contain localized structures, and SEDC-T can often
highlight specific image regions. Prototype-guided results are highly valid but
visually subtle or diffuse. DVCE can flip predictions, but the low-plausibility
examples show that generative changes may become too broad or noisy.

Best qualitative role by method:

- Prototype-guided: high-validity technical baseline.
- Retrieval-NUN: real target-class comparison case.
- SEDC-T: most localized visual explanation.
- DVCE: generative feasibility example with visible plausibility limitations.

### Pneumonia

Pneumonia is more difficult. The classifier may rely on broad intensity,
contrast, border, or acquisition-related cues. This makes localized segment
replacement less reliable and makes visual interpretation harder. SEDC-T
validity is lower than on BUSI, and tuning improves it only slightly. Prototype
still reaches high validity, but mainly through subtle perturbations. DVCE
produces valid samples in some cases, but artifacts remain visible.

Best qualitative role by method:

- Prototype-guided: demonstrates model sensitivity to small perturbations.
- Retrieval-NUN: shows nearest real X-ray cases in target-class embedding space.
- SEDC-T: shows the difficulty of region-based explanations on chest X-rays.
- DVCE: demonstrates feasibility and limitations of generative counterfactuals.

## Quantitative And Qualitative Trade-Offs

| Method | Validity | Locality | Visual plausibility | Main strength | Main limitation |
| --- | --- | --- | --- | --- | --- |
| Prototype-guided baseline | very high | weak | limited | strong technical validity | diffuse/adversarial-style changes |
| Prototype-guided ablation | very high | weak | slightly more conservative | smaller perturbations | still not localized |
| Retrieval-NUN | very high | not an edit | intuitive but not minimal | real target-class cases | patient/acquisition differences |
| SEDC-T original-style | medium | strong | mixed | localized segment explanations | lower validity, especially Pneumonia |
| SEDC-T tuned variant | medium | medium/strong | mixed | tests whether tuning helps | small gain, more changed area |
| DVCE OpenAI | medium/high on 5 samples | weak/medium | unstable | generative feasibility | artifacts, natural-image prior |
| DVCE fine-tuned | medium/high on 5 samples | weak/medium | unstable | medical checkpoint integration | artifacts remain, runtime higher |

## Main Conclusion

The qualitative figures support the central conclusion that there is no single
best method. Instead, the methods reveal different trade-offs:

- **Prototype-guided optimization** is strongest in model validity, but weakest
  in visual interpretability.
- **Retrieval-NUN** is strongest as a real-case comparison, but it is not a
  minimal counterfactual edit.
- **SEDC-T** is strongest in locality, but does not always achieve the target
  class, especially on Pneumonia.
- **DVCE** covers the generative direction, but current results remain sensitive
  to checkpoint and guidance settings and still show artifacts.

For the seminar paper, the safest argument is:

```text
Counterfactual validity and human/medical plausibility diverge. A method can
produce model-valid counterfactuals without producing visually convincing
medical explanations. Therefore, the methods should be compared using both
quantitative metrics and qualitative case analysis.
```
