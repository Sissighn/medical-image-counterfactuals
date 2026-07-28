# Method Fidelity: Detailed Comparison of Four Counterfactual Methods

This document records, for each implemented counterfactual method, which parts
faithfully reproduce the reference, where the implementation differs, and
which deliberate design decisions were made. The assessment is based on a
line-by-line comparison between the project implementations and their original
sources.

Framework and classifier changes—TensorFlow to PyTorch and ImageNet CNNs to a
medical ResNet18—are intentional project-wide adaptations and are not treated
as implementation errors.

## References

| Method | Project implementation | Original source |
| --- | --- | --- |
| CFProto | [`scripts/run_cfproto_pytorch.py`](../../scripts/run_cfproto_pytorch.py) | Alibi `CounterfactualProto` in [`alibi/explainers/cfproto.py`](https://github.com/SeldonIO/alibi) |
| Goyal et al. CVE | [`scripts/run_goyal_cve_pytorch.py`](../../scripts/run_goyal_cve_pytorch.py) | Goyal et al. (2019), arXiv:1904.07451; [reference code](https://github.com/facebookresearch/visual-counterfactuals) |
| SEDC-T | [`scripts/run_sedc_t_pytorch.py`](../../scripts/run_sedc_t_pytorch.py) | [`sedc_t2_fast.py`](https://github.com/ADMAntwerp/ImageCounterfactualExplanations) |
| DVCE | [`src/dvce_core.py`](../../src/dvce_core.py), [`scripts/run_dvce_pytorch.py`](../../scripts/run_dvce_pytorch.py) | [DVCEs](https://github.com/valentyn1boreiko/DVCEs), vendored locally under `external/DVCEs` for experiments |

**Overall result:** All four core algorithms were ported faithfully, and the
audit found no methodological implementation error. Every deviation is an
intentional design decision documented in the code, such as the framework,
dataset, or evaluation protocol, and does not alter the core algorithm.

## 1. CFProto: Prototype-Guided Counterfactuals

The implementation ports the TensorFlow 1.x graph from Alibi's
`CounterfactualProto.attack` to PyTorch. It applies the method to a medical
ResNet18 and uses the encoder of a convolutional autoencoder.

### 1.1 Faithful Components

- **Hinge attack loss:**
  `f(x,d) = max(0, p_orig − max_{i≠orig} p_i + kappa)`, including the `−10000`
  masking trick that excludes the original class from the non-target maximum.
  This matches Alibi's `loss_attack`
  ([implementation](../../scripts/run_cfproto_pytorch.py#L232-L240)).
- **Optimization objective:**
  `c·L_attack + L2 + gamma·L_AE + theta·L_proto`. The `beta·L1` term is
  introduced through shrinkage-thresholding rather than through the gradient.
  Every term is a sum, not a mean, as in the TensorFlow graph's `loss_opt`.
- **FISTA optimization:** element-wise shrinkage-thresholding with the same
  three-case structure: upper branch for `delta > beta`, return to the original
  for `|delta| ≤ beta`, and the lower branch otherwise. The update includes
  projection to the feature range, Nesterov momentum
  `zt = (i+1)/(i+4)`—equivalent to Alibi's
  `global_step/(global_step+3)` after incrementing—and a final clamp to `(0, 1)`
  ([implementation](../../scripts/run_cfproto_pytorch.py#L243-L253)).
- **Learning-rate decay:** polynomial decay with power 0.5 and a final value of
  zero, matching `tf.train.polynomial_decay(..., power=0.5)`.
- **Gradient clipping:** clipping to `(−1000, 1000)`, Alibi's default `clip`
  value.
- **Binary search over `c`:** `ub = min(ub, c)`, bisection once `ub < 1e9`, and
  tenfold escalation otherwise. This matches Alibi's constant update.
  `adv` and `adv_s` are reinitialized to the original input for every `c` step.
- **Prototype construction (`fit`):** class membership is determined from the
  classifier's predictions on the training data, not from ground-truth labels:
  `preds = argmax(predict(train_data))`. Encoder-space prototypes are class
  means or k-nearest-neighbor prototypes.
- **kNN prototype semantics:** when `k` is set, `k_type='mean'` returns the mean
  distance and `k_type='point'` returns the distance of the kth neighbor. In
  both cases, the prototype position is the mean of the k nearest encodings.
  This reproduces Alibi's slightly counterintuitive implementation
  ([implementation](../../scripts/run_cfproto_pytorch.py#L188-L205), compared
  with `cfproto.py:1043-1051`).
- **Nearest-prototype target selection:** among all candidate classes, the
  method selects the class with the minimum distance from `ENC(x)` to its
  prototype: `id_proto = min(dist_proto)`.
- **Best counterfactual selection:** the final choice combines Elastic Net
  distance `L2 + beta·L1`, the kappa condition in `compare()`, and target-class
  membership.

### 1.2 Differences from the Original

- **Framework:** the TensorFlow 1.x graph is reimplemented in PyTorch. The
  classifier is a ResNet18 whose ImageNet normalization is wrapped into the
  prediction function. Optimization runs in `[0,1]` pixel space, consistent
  with `feature_range=(0,1)`.
- **No black-box mode:** the original numerical-gradient path for
  non-differentiable models is omitted because the classifier used here is
  differentiable.
- **No categorical variables or k-d trees:** ABDM/MVDM distances and k-d-tree
  prototypes are omitted because they apply to tabular or categorical data
  rather than images.
- **No TrustScore filter:** the optional TrustScore threshold is not
  implemented. Alibi's default `threshold=0` disables it, so the default
  behavior remains equivalent.

### 1.3 Intentional Design Decisions

- **Hyperparameters based on the Alibi MNIST example rather than class
  defaults:** `c_init=1, c_steps=2`, while the Alibi class defaults are 10 and
  10. This keeps runtime manageable for medical images without changing the
  bisection logic.
- **Rescaled `gamma` and `theta`:** the Alibi MNIST examples use values of 100
  for 28×28 inputs and a small latent space. Because each objective term sums
  over its dimensions, the weights must be rescaled for 224×224 inputs and the
  encoder dimension. The defaults here are `gamma=1.0` and `theta=0.5`; their
  rationale appears in the command-line help, and `metadata.json` records the
  resulting `loss_terms`.
- **Behavior when no counterfactual is found:** instead of Alibi's zero array,
  the implementation returns the final `adv` with `found=False` and
  `valid=False`. This improves reporting while the validity metric still counts
  the result as invalid.
- **Encoder-space prototypes:** the method consistently uses an autoencoder
  encoder without a k-d-tree fallback, corresponding to Alibi's `enc_model`
  path.

## 2. Goyal et al. (2019): Counterfactual Visual Explanations

This instance-based method copies discriminative spatial cells from a real
target-class distractor image into the query's feature map until the prediction
changes.

### 2.1 Faithful Components

- **Greedy exhaustive search:** the implementation matches the reference
  `compute_counterfactual` with `lambd=0` and `topk=None`. It therefore
  reproduces the pure Goyal baseline without the semantic auxiliary term from
  Vandenhende et al. At each iteration, every remaining pair of query and
  distractor cells is evaluated, and the swap with the highest target-class
  probability is committed.
- **Selection criterion:** the reference maximizes
  `log(softmax)[:, distractor_class]`; this project maximizes
  `softmax[:, target]`. Because the logarithm is strictly monotonic, the
  resulting `argmax` is identical.
- **Stopping criterion:** the method stops at the first iteration in which the
  argmax prediction equals the target class
  ([implementation](../../scripts/run_goyal_cve_pytorch.py#L304)).
- **Permutation constraint:** each query cell is edited at most once and each
  distractor cell is used at most once, matching the reference filter
  `i != query_cell and j != distractor_cell`.
- **Feature split `f` / `g`:** ResNet18 is divided into a spatial extractor,
  from the convolutional stem through `layer4` with shape `[B,512,7,7]`, and a
  decision head consisting of global average pooling and a fully connected
  layer. This is the split used in the paper.
- **Incremental pooling**
  ([implementation](../../scripts/run_goyal_cve_pytorch.py#L307-L316)):
  `pooled + (f'(j) − f(i))/N`. This is mathematically equivalent to evaluating
  the reference head `g` on each fully edited feature map because
  `g = FC ∘ AveragePool` is linear in the mean across cells. It is faster but
  does not change the result.

### 2.2 Differences from the Original

- **Distractor selection, the only substantive protocol difference:** the
  paper defines the method for a supplied pair `(I, I')`, with `I'` from the
  target class. The reference code determines the distractor class from the
  confusion matrix and randomly samples up to `max_num_distractors` images to
  form a shared cell pool. This project instead uses the nearest correctly
  classified training neighbor from the manifest target class, measured by
  cosine distance between L2-normalized pooled penultimate features. The edit
  search remains faithful; only the distractor source is project-specific, as
  documented in the code and `metadata.json`.
- **Framework and backbone:** the project uses ResNet18 instead of the
  ImageNet CNNs from the paper and a single distractor image instead of a pool
  containing multiple images.

### 2.3 Intentional Design Decisions

- **`lambd=0` and no auxiliary model:** the implementation deliberately
  reproduces the Goyal baseline, not the ECCV 2022 extension by Vandenhende et
  al. that adds semantic consistency through a self-supervised auxiliary model.
- **`max_edits` cap:** the default is the complete 7×7 grid, or 49 edits. This
  guards against an infinite loop. Once every cell is replaced, the pooled
  feature equals that of the distractor and the prediction must change because
  pooling is permutation-invariant. The reference may instead fail on an empty
  edit set.
- **Fixed manifest target and cosine NUN retrieval:** these choices provide a
  fair and reproducible comparison with the other three methods on the same
  evaluation manifest.
- **Pixel composite for visualization:** the decision-changing edit occurs in
  feature space; the composite image inserts patches corresponding to the
  exchanged 7×7 cells. This follows the paper's standard visualization, and
  pixel-change metrics are calculated on that composite.

## 3. SEDC-T: Search for Explanations by Directed Contrast, Targeted

This is a best-first search over superpixel segments. The perturbed image is
expanded with the segment that most increases the difference between target-
and original-class scores, continuing until the prediction reaches the target
class. The implementation is a near one-to-one port of `sedc_t2_fast.py`.

### 3.1 Faithful Components

- **Best-first search:** the initial level contains every single segment.
  Thereafter, the pending candidate with the highest expansion score is
  repeatedly expanded until a valid level is found.
- **Expansion score:** `p_target − p_original_class` on the perturbed candidate,
  where the original class is the prediction for the unmodified image. This
  matches the reference expression `p_new_list − results[:, c]`.
- **Termination logic:** search stops after finding a valid counterfactual, when
  no expansion remains, or at the timeout. It preserves the reference
  behavior, including no deduplication of segment sets, checking the timeout
  only once per expansion level, and removing the expanded node from the
  pending set
  ([implementation](../../scripts/run_sedc_t_pytorch.py#L398-L511)).
- **All four replacement modes:** per-channel mean; `GaussianBlur (31,31), 0`;
  uniform random replacement; and per-segment Navier–Stokes inpainting with
  `cv2.INPAINT_NS` and radius 3.
- **Best explanation selection:** `argmax(P − p)` selects the valid candidate
  with the largest target-class score increase, using the first result on a
  tie. This matches line 115 of the reference.
- **Explanation image:** selected segments are shown on a black background, as
  in the original implementation.
- **Segmentation:** Quickshift with
  `kernel_size=4, max_dist=200, ratio=0.2`, matching the values in the original
  repository's experiment scripts (`experiment_*.py`, `gen_t2_mnv2.py`).
- **Batched prediction:** evaluating all candidates at one level in a single
  forward pass is numerically equivalent to the reference call
  `classifier.predict(cf_candidates)`. Candidate ordering is preserved so that
  `max` and `argmax` break ties identically.

### 3.2 Differences from the Original

- **Framework and input:** the project uses PyTorch and ResNet18 instead of
  Keras. Search runs in `[0,1]` pixel space, with ImageNet normalization applied
  during the forward pass.
- **Failure behavior:** the reference returns `None`. For reporting purposes,
  this project also retains the best invalid attempt while emitting the same
  “No CF found on the requested parameters” message. The result is still
  counted as invalid.

### 3.3 Intentional Design Decisions

- **Default timeout of 30 rather than 600 seconds:** the reference uses
  `max_time=600`. Every valid counterfactual observed on the 224×224 medical
  images is found within a few seconds, so 30 seconds limits waits for failures
  without discarding a valid counterfactual. The value is configurable: `600`
  matches the reference and `0` disables the timeout. It should remain
  consistent across datasets for a fair comparison.
- **Lung-field ROI ablation (`--roi_mode lung_fields`):** this is a
  project-specific extension, not part of the original SEDC-T method. A coarse,
  content-independent geometric prior for frontal chest radiographs restricts
  which segments may be selected. It is disabled by default:
  `--roi_mode none` is the original-style configuration. The ablation is
  identified in the README and in `metadata.json` under
  `method_fidelity_note`.
- **Configurable Quickshift parameters:** command-line arguments expose the
  parameters while retaining the original values as defaults.
- **`blur` as the default replacement:** this is the most commonly used mode in
  the original experiments; all other modes remain available.

## 4. DVCE: Diffusion Visual Counterfactual Explanations

DVCE generates counterfactuals using an unconditional OpenAI 256×256 diffusion
process guided by the medical classifier's gradient, an Lp distance term, and
optionally a robust second classifier through Cone Projection. The
implementation ports the core DVCE logic from `dff_attack.py`
(`DiffusionAttack`).

### 4.1 Faithful Components

- **`cond_fn_clean`**
  ([implementation](../../src/dvce_core.py#L315-L396)): internally reconstructs
  `p_mean_variance` with `clip_denoised=False`, evaluates the classifier and Lp
  distance on `pred_xstart`, and passes unclamped values through `_map_img`
  (`0.5·(x+1)` without clamping) so gradients also flow for out-of-range
  pixels. It returns
  `grad_out = lambda·grad_class − lp_value·lp_grad`, matching the original
  structure.
- **Classifier gradient:** target log-softmax averaged across augmentations,
  with `y.view(-1).repeat(aug_num)` indexing, as in the original.
- **Lp distance term:** both branches are reproduced: the closed-form Lp
  gradient (`compute_lp_gradient`) by default and autograd over
  `compute_lp_dist` when `--denoise_dist_input` is set. Distance is always
  measured against `init_image`.
- **`enforce_same_norms`:** `_renormalize_gradient` separately normalizes the
  classifier and distance gradients to the norm of `eps = model_output`, as in
  the original. `condition_mean` passes `p_mean_var['model_output']` as `eps`
  to `cond_fn`.
- **Cone Projection**
  ([implementation](../../src/dvce_core.py#L188-L246)): a functionally
  equivalent copy, including computation on flattened CPU tensors. Argument
  order matches `dff_attack.py`: the robust auxiliary classifier's gradient is
  projected onto a cone around the explained classifier's gradient.
- **`ImageAugmentations`:** the original class is loaded unchanged by file
  path, avoiding only heavyweight package imports such as `lpips` and
  TensorBoard. With `aug_num=1`, it is the identity, as in the original.
- **Classifier adapter (`MedicalResNetAdapter`):** mirrors
  `ResizeAndMeanWrapper`: bicubic resize to 224 with `interpolation=3`, ImageNet
  normalization, and no input clamping.
- **Sampling:** uses the vendored `p_sample_loop_progressive`, passing `eps`
  through `condition_mean` and reseeding random number generators. The final
  image is `pred_xstart` from the last step, followed by `_map_img` and clamping
  to `(0,1)`.
- **Backbone configuration and gradient rule:** uses
  `attention_resolutions="32,16,8"`, `learn_sigma`, `num_channels=256`, and the
  remaining reference configuration. After loading, only `qkv`, `norm`, and
  `proj` parameters are set to `requires_grad=True`, matching the
  `DiffusionAttack` sequence: freeze, evaluate, move to the device, reactivate
  selected gradients, and optionally enable fp16.
- **Hyperparameter defaults:** match the original `default.yml`:
  `timestep_respacing=200`, `skip_timesteps=100`, `lp_custom=1.0`,
  `lp_custom_value=0.15`, `classifier_lambda=0.1`,
  `enforce_same_norms=true`, `gen_type=p_sample`, `seed=1`, and
  `model_output_size=256`.

### 4.2 Differences from the Original

- **Reversed `classifier` and `second_classifier` roles:** in this project,
  `classifier` is the medical model under explanation and `second_classifier`
  is the robust auxiliary model. The original terminology assigns these names
  in the opposite order. The cone geometry is unchanged: the robust gradient is
  projected onto a cone around the explained model's gradient. This mapping is
  documented and used consistently.
- **Omitted paths disabled by the DVCE configuration:** `layer_reg`,
  LPIPS/L2 similarity, and the blended-diffusion branch are not ported. Their
  weights are zero in `default.yml`.
- **Reporting additions:** the project records additional metrics, such as mean
  absolute difference and changed-pixel fractions, and uses manifest-driven
  sample selection that is not part of the original implementation.

### 4.3 Intentional Design Decisions

- **`denoise_dist_input` enabled in original-faithful commands:** the original
  argument default is `False`, but both published DVCE commands in the original
  README set `--denoise_dist_input`. The original-faithful project runs
  therefore enable it and record the choice in the `debug` field.
- **Cone Projection as the primary faithful variant for a non-robust
  ResNet18:** `--deg_cone_projection 30 --aug_num 16` with a robust PGD ResNet18
  as the second classifier mirrors the second command in the original README.
- **MPS-to-CPU round trip during augmentation:** this is a device workaround
  because MPS does not support non-divisible adaptive pooling. It preserves the
  autograd graph and does not change the method; CUDA execution remains on the
  device.
- **Per-sample seed `seed + sample_index`:** this provides reproducibility for
  single-image generation rather than using one fixed batch seed. It is an
  evaluation detail.
- **Random-number generation for the second augmentation:** the second
  classifier receives a newly drawn augmentation. The order of random draws may
  therefore differ from the original; this is stochastic rather than
  methodological.
- **Optional medical diffusion checkpoint:** in addition to the OpenAI
  checkpoint, the command accepts a fine-tuned diffusion model through
  `--diffusion_checkpoint_path`.

## 5. Summary of Intentional Deviations

The following three points should be identified explicitly as intentional
deviations in a publication. They are already documented in the metadata and
code comments:

| # | Method | Deviation | Effect |
| --- | --- | --- | --- |
| 1 | Goyal CVE | Nearest-unlike-neighbor distractor selected by cosine distance in pooled feature space instead of random confusion-matrix sampling | The edit search is identical; only the distractor source is project-specific. |
| 2 | SEDC-T | Search timeout of 30 rather than 600 seconds | No observed valid counterfactual is lost; only the wait for failure cases is capped. |
| 3 | CFProto | Hyperparameters based on the Alibi MNIST example (`c_init=1, c_steps=2`) with dimension-dependent rescaling of `gamma` and `theta` | The algorithm is unchanged; computational effort and weighting are adapted to 224×224 inputs. |

All other differences arise from the framework (TensorFlow to PyTorch),
classifier domain (ImageNet to medical imaging), or reporting protocol and are
intentional across the project.

> **Reporting note:** Model validity means that the classifier predicts the
> target class. It does not establish medical plausibility or clinical
> causality.

## 6. Project-Specific Extensions

The following components are not part of the original methods:

- **SEDC-T lung-field ROI:** a geometric lung-field prior used as an ablation
  and disabled by default.
- **Fixed evaluation manifests:** identical samples and target classes across
  all four methods for a fair comparison.
- **Unified change and validity metrics:** consistent metrics across all
  methods.
- **DVCE Cone Projection with a robust PGD ResNet18:** a second classifier for
  explaining the non-robust medical classifier.
