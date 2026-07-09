# CFProto Method Documentation

Implementation: [`scripts/run_cfproto_pytorch.py`](../../scripts/run_cfproto_pytorch.py)

Reference (paper):
Van Looveren & Klaise (2019/2021), "Interpretable Counterfactual Explanations
Guided by Prototypes", ECML-PKDD 2021, <https://arxiv.org/abs/1907.02584>

Reference (original code): the official Seldon implementation
`alibi.explainers.cfproto.CounterfactualProto` in
<https://github.com/SeldonIO/alibi>.

Run commands: [`cfproto_encoder_run_commands.md`](cfproto_encoder_run_commands.md)

---

## 1. What CFProto is

CFProto finds a counterfactual by **optimizing the input image directly**
(unlike Goyal et al., which swaps feature cells from a real distractor, and
SEDC-T, which removes/blurs segments). Starting from the original image, it
perturbs pixels under a combined loss that (a) pushes the classifier away
from the original class, (b) keeps the perturbation small in L1/L2, and (c)
pulls the encoding of the perturbed image toward a **prototype** of a
plausible target class in an autoencoder's latent space. The prototype term
is what makes CFProto's counterfactuals more interpretable/in-distribution
than a plain adversarial perturbation.

This project ports the method to PyTorch / ResNet-18 on 224×224 medical
images. The optimizer, loss structure, prototype definition, and c-search are
kept faithful to the original; framework and classifier are the intended
project-specific substitutions.

---

## 2. How the implementation works, step by step

1. **Class prototypes (`fit`).** All training images are encoded once with a
   frozen convolutional autoencoder. Class membership is defined by the
   **classifier's own predictions** on the training split (not the true
   labels), matching alibi's `fit()`. The prototype of a class is the mean
   encoding of the training images predicted as that class (`k=None`), or the
   mean of the `k` nearest encodings to the query (`--prototype_k`,
   `k_type='mean'`/`'point'`).

2. **Target selection.** Among the candidate classes (all classes but the
   original, or the manifest's fixed target), the class whose prototype is
   closest to the query's own encoding is used — the standard alibi rule.

3. **FISTA optimization.** The counterfactual is found with the Fast
   Iterative Shrinkage-Thresholding Algorithm: a gradient step on an
   auxiliary variable `adv_s`, followed by element-wise shrinkage-
   thresholding toward the original image (the L1 term `beta`) and a
   Nesterov-style momentum update `adv_s = adv + t/(t+3) * (adv - adv_prev)`,
   projected back onto the pixel range `[0, 1]`.

4. **Loss.** The optimized loss is a **sum** (not a mean) of
   `c · L_attack + L2 + gamma · L_AE + theta · L_proto`, with the L1 term
   applied only through shrinkage-thresholding, exactly as in the original
   TensorFlow graph. `L_attack` is the **untargeted** hinge loss
   `max(0, p_orig − max_{i≠orig} p_i + kappa)` — it only demands the original
   class stop being the argmax; the target class is reached via the
   prototype term and confirmed after the fact.

5. **Binary search over `c`.** The attack-loss weight `c` is adjusted between
   `c_steps` outer iterations: doubled (×10, matching the original) when no
   valid counterfactual was found yet, halved toward the midpoint of the
   known bounds once one is found — searching for the smallest `c` that still
   yields a valid flip.

6. **Best-counterfactual selection.** Across all iterations and c-steps, the
   candidate with the smallest elastic-net distance (`L2 + beta · L1` to the
   original) among those satisfying the counterfactual condition
   (kappa-adjusted argmax differs from the original class *and* lands in a
   candidate target class) is kept.

7. **Outputs per sample.** Original/counterfactual/difference/overlay
   visualization, full `c_history` (bounds and outcome per step), raw
   (unweighted) loss terms at the selected result, elastic-net distance,
   change metrics, and prototype details — all written to `metadata.json`.

---

## 3. Soll-Ist comparison with the original

| Aspect | Original (alibi `CounterfactualProto`) | This implementation | Status |
|---|---|---|---|
| Optimizer | FISTA: gradient step + shrinkage-thresholding + Nesterov momentum | identical | ✅ |
| Attack loss | untargeted hinge `max(0, p_orig − max_{i≠orig} p_i + κ)` | identical | ✅ |
| Loss terms | `c·L_attack + L2 + β·L1 + γ·L_AE + θ·L_proto` (sums) | identical | ✅ |
| L1 handling | via shrinkage-thresholding, not the gradient | identical | ✅ |
| `c` search | binary search, ×10 escalation, no upper bound until a valid CF is found | identical | ✅ |
| Class prototypes | mean/kNN-mean of training encodings, membership via classifier predictions | identical | ✅ |
| Target selection | nearest prototype among candidate classes | identical | ✅ |
| Best-CF selection | smallest `L2 + β·L1` among valid candidates | identical | ✅ |
| Learning-rate schedule | polynomial decay, power 0.5 | identical | ✅ |
| Feature range / gradient clip | `(0, 1)` / `(-1000, 1000)` | identical | ✅ |

---

## 4. Deliberate differences from the original (not fidelity problems)

1. **Framework and classifier.** PyTorch / ResNet-18 on medical images
   instead of the original TensorFlow 1.x graph and Keras/CNN models. This is
   the intended project substitution; optimization still happens in raw
   `[0, 1]` pixel space with the classifier's normalization wrapped into the
   predict function (`feature_range=(0, 1)` in alibi's terms).

2. **Autograd instead of a static TF1 graph.** Gradients are computed with
   `torch.autograd.grad` per iteration rather than a precompiled graph. This
   is numerically equivalent, just built differently.

3. **`gamma`/`theta` recalibrated per dataset/autoencoder.** All loss terms
   are sums, so their raw magnitude depends on the input and latent
   dimensionality. The alibi MNIST example's `gamma=theta=100` (28×28 inputs,
   small latent space) diverges at 224×224 with this project's autoencoders.
   Recalibrated so the weighted prototype/AE terms stay comparable to the L2
   sum instead of dominating it (see `cfproto_encoder_run_commands.md` for
   the exact values and the calibration diagnostics behind them).

4. **Not implemented (also unused in the paper's image experiments or
   disabled by default in alibi):** the TensorFlow computation graph itself;
   black-box mode with numerical gradients; categorical variables and
   k-d-tree prototypes (an encoder is used instead, as in the paper's image
   experiments); TrustScore threshold filtering (alibi's default
   `threshold=0` also disables it).

---

## 5. Key parameters

| Argument | Final value (BUSI / Pneumonia) | Meaning |
|---|---|---|
| `--model_path` | dataset-specific | ResNet18 classifier checkpoint |
| `--dataset_path` | dataset-specific | Processed dataset root (`train`/`val`/`test`) |
| `--manifest_path` | dataset-specific | Fixed evaluation manifest; samples and targets come from it |
| `--autoencoder_path` | `autoencoder_{busi,pneumonia}_bottleneck256.pth` | Frozen encoder used for `ae_model`/`enc_model` |
| `--max_iterations` | `1000` | FISTA iterations per `c` step (alibi default) |
| `--learning_rate_init` | `0.01` | Initial learning rate for the polynomial decay (alibi default) |
| `--kappa` | `0.0` | Confidence margin of the hinge attack loss (alibi default) |
| `--beta` | `0.1` | L1 weight, applied via shrinkage-thresholding (alibi default) |
| `--gamma` | `1.0` | Autoencoder reconstruction loss weight (recalibrated, see §4) |
| `--theta` | `0.5` (BUSI) / `0.05` (Pneumonia) | Prototype loss weight (recalibrated per dataset, see §4) |
| `--c_init` | `1.0` | Initial attack-loss constant (alibi MNIST example) |
| `--c_steps` | `5` | Number of binary-search updates for `c` |
| `--prototype_k` | `3` | Nearest training encodings defining a class prototype |
| `--k_type` | `mean` | Mean of the `k` nearest encodings |
| `--batch_size` | `16` | Batch size for the one-time prototype fit over the training split |

---

## 6. Results on the fixed manifests

| Dataset | Samples | Validity | Mean CF confidence (valid) | Mean L1 | Mean L2 | Mean changed pixel fraction | Mean runtime |
|---|---:|---:|---:|---:|---:|---:|---:|
| BUSI | 15 | 0.87 (13/15) | 0.646 | 0.0121 | 0.0436 | 0.0529 | 46.1s |
| Pneumonia | 20 | 1.00 (20/20) | 0.574 | 0.0027 | 0.0210 | 0.0180 | 46.3s |

Reading the numbers:

- **Validity below 1.0 on BUSI (13/15)** is expected and not a bug: the
  attack loss is untargeted, so on 2 samples the optimization found a valid
  flip away from the original class that did not match the manifest's fixed
  target class, which counts as invalid for this evaluation.
- **Sparsity is dataset-dependent by construction.** Pneumonia counterfactuals
  change far fewer pixels (1.8%) than BUSI (5.3%) at their respective
  calibrated `theta`; this reflects the ~34× difference in raw encoder-space
  class separation between the two autoencoders (see §4), not a difference
  in method fidelity.
- **Runtime (~46s/sample)** is dominated by `max_iterations=1000 × c_steps=5`
  sequential single-image FISTA steps, each a forward+backward pass through
  both the classifier and the autoencoder; these runs used a CUDA GPU rather
  than the project's usual Apple Silicon/MPS machine.
