# Implementation Fidelity to the Reference Methods

The project transfers four counterfactual methods to medical ResNet18
classifiers in PyTorch. Framework, dataset, and evaluation changes are
project-specific adaptations. This document separates them from the retained
algorithmic components.

## References

| Method | Project implementation | Reference |
| --- | --- | --- |
| CFProto | `scripts/run_cfproto_pytorch.py` | Alibi `CounterfactualProto` |
| Goyal-CVE | `scripts/run_goyal_cve_pytorch.py` | Goyal et al. (2019), `facebookresearch/visual-counterfactuals` |
| SEDC-T | `scripts/run_sedc_t_pytorch.py` | `ADMAntwerp/ImageCounterfactualExplanations` |
| DVCE | `src/dvce_core.py`, `scripts/run_dvce_pytorch.py` | `valentyn1boreiko/DVCEs` |

## CFProto

### Retained Algorithmic Components

- untargeted hinge loss between the original class and strongest alternative
- objective combining attack loss, squared L2 distance, autoencoder
  reconstruction, and prototype distance
- L1 regularization through FISTA shrinkage-thresholding
- polynomial learning-rate decay and gradient clipping
- binary search over the attack constant `c`
- class prototypes built from encoder representations of training images,
  grouped by model prediction
- selection of the valid result with the smallest Elastic Net distance

In manifest mode, the requested target class is the only prototype candidate.
Outside manifest mode, the nearest alternative class in prototype space can be
selected.

### Differences

- PyTorch instead of TensorFlow 1.x
- medical ResNet18 classifiers instead of the models used in the reference
  examples
- no numerical black-box gradients
- no categorical distances, k-d trees, or TrustScore filtering
- on failure, the final candidate is retained with `valid=false` instead of
  returning a zero array

The `gamma` and `theta` values were scaled for the image and encoder
dimensions. Final values are listed in
[`results/final_configs/README.md`](../final_configs/README.md).

## Goyal-CVE

### Retained Algorithmic Components

- split between spatial features and the classification head
- greedy exhaustive search over query/distractor cell pairs
- target-class probability maximization at every iteration
- each query cell and each distractor cell used at most once
- termination when the target class is first reached
- `7 × 7` feature grid at the output of ResNet18 `layer4`

For the linear ResNet18 head, incrementally updating the globally pooled
feature vector is mathematically equivalent to recomputing the entire edited
feature grid.

### Differences

The reference assumes a target-class distractor or samples distractors from a
pool. This project uses the nearest correctly classified training image from
the manifest target class. Distance is cosine distance between L2-normalized,
globally pooled `layer4` features.

Classification changes are produced in feature space. The saved pixel
composite is used for visualization and image-based change metrics.

## SEDC-T

### Retained Algorithmic Components

- Quickshift segmentation
- best-first search over segment sets
- expansion score `p(target) - p(original class)`
- termination after the first search level that contains a valid result
- selection of the valid candidate with the largest target-probability gain
- mean, blur, random, and inpainting replacement modes

Final runs use Quickshift with `kernel_size=4`, `max_dist=200`, and
`ratio=0.2`, plus a blur kernel of 31.

### Differences

- PyTorch and batched candidate evaluation instead of Keras
- 30-second rather than 600-second timeout; the value remains configurable
- on failure, the best invalid candidate is retained with `valid=false`
- the optional geometric lung-field ROI is a labeled project ablation, not a
  component of the reference method

## DVCE

### Retained Algorithmic Components

- unconditional OpenAI diffusion model with 256-pixel output
- classifier guidance on `pred_xstart`
- Lp proximity guidance toward the original image
- gradient-norm matching
- optional multiple image augmentations per step
- Cone Projection between the robust auxiliary classifier and the classifier
  being explained
- stochastic `p_sample` sampling

For the non-robust explained ResNet18, the Cone variant with a PGD-robust
ResNet18 as the second classifier is the original-faithful main
configuration. No-Cone runs are ablations.

### Differences

- medical ResNet18 classifiers instead of the ImageNet classifiers in the
  reference
- medically fine-tuned diffusion checkpoints as additional project variants
- compatibility adaptations for current NumPy, PyTorch, and torchvision
  versions
- device-dependent execution on MPS or CUDA

The OpenAI and medically fine-tuned runs did not use fully identical device
and precision conditions. Runtime or result differences must not be
interpreted as isolated checkpoint effects.

## Conclusion

The programs retain the core optimization, search, or guidance mechanism of
each method. Medical datasets, ResNet18, fixed targets, distractor selection,
timeouts, and checkpoints still change the experimental context. Fidelity
claims therefore refer to the algorithmic core, not to an identical
reproduction of the original experiments.
