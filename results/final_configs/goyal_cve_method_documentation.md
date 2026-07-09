# Goyal et al. 2019 CVE Method Documentation

Implementation: [`scripts/run_goyal_cve_pytorch.py`](../../scripts/run_goyal_cve_pytorch.py)

Reference (paper):
Goyal, Wu, Ernst, Batra, Parikh & Lee (2019), "Counterfactual Visual
Explanations", ICML 2019, <https://arxiv.org/abs/1904.07451>

Reference (original code): the official Meta repository
<https://github.com/facebookresearch/visual-counterfactuals> (Vandenhende et
al., ECCV 2022) contains a faithful implementation of the Goyal et al. method
as a baseline.

Run commands: [`goyal_cve_run_commands.md`](goyal_cve_run_commands.md)

---

## 1. What the Goyal et al. method is

The Goyal et al. counterfactual visual explanation answers the question: *which
region of a query image `I` would have to be replaced by content from a
distractor image `I'` of a different class so that the classifier changes its
prediction to that other class?* It is an **instance-based, feature-space edit**
method. It does not optimize pixels (unlike CFProto) and does not blur or remove
segments (unlike SEDC-T). Instead it copies discriminative spatial cells of a
real distractor image into the query, in the classifier's own feature map.

The method decomposes a CNN into a spatial feature extractor `f` and a decision
head `g`. For a query image `I` (predicted class `c`) and a distractor image
`I'` from the target class `c'`, the counterfactual feature map is

```text
f*(I) = (1 - a) ∘ f(I) + a ∘ (P f(I'))
```

where `a` is a binary gate vector over spatial cells and `P` is a permutation
matrix that aligns distractor cells to query cells. The number of edits
`||a||_1` is minimized subject to `argmax g(f*(I)) = c'`. The output is sparse:
only a few feature cells are swapped to flip the decision.

This project ports the method to PyTorch / ResNet-18. The search logic follows
the paper's greedy exhaustive-search variant; the framework and classifier are
the intended project-specific substitutions.

---

## 2. How the implementation works, step by step

1. **Network decomposition.** ResNet-18 is split into `f` (conv stem through
   `layer4`, output `[512, 7, 7]` for 224×224 inputs, i.e. 49 spatial cells) and
   `g` (global average pooling followed by the fully connected classifier).

2. **Distractor database.** All correctly classified training images are encoded
   once. Their L2-normalized pooled penultimate features are stored per class.
   A candidate is kept only if its true label equals the target class *and* the
   model prediction equals the target class.

3. **Distractor selection.** For each manifest sample, the distractor `I'` is
   the nearest correctly classified training image of the manifest target class
   in pooled feature space (cosine distance). The paper defines the method for a
   given `(I, I')` pair; the nearest-neighbor rule is the standard instantiation
   and makes the distractor a real *nearest-unlike* case.

4. **Greedy exhaustive search.** Starting from `f(I)`, each iteration evaluates
   *all* remaining `(query cell i, distractor cell j)` swaps, computes the
   target-class softmax probability of every resulting edited feature map, and
   commits the single swap that maximizes it. Because `g` for ResNet is the FC
   layer applied to the spatial mean, each candidate's pooled vector is obtained
   with the exact incremental mean update
   `pooled + (f'(j) − f_current(i)) / N`, which is identical to evaluating `g` on
   the fully edited feature map but far cheaper.

5. **Constraints.** Each query cell is edited at most once and each distractor
   cell is used at most once (the permutation constraint `P` from the paper).

6. **Termination.** The search stops at the first iteration where the argmax
   prediction equals the target class. With the full 49-cell budget the pooled
   feature converges to the distractor's, so a flip is guaranteed and validity is
   1.00 by construction; the reported quantity of interest is therefore the
   **number of edits** (sparsity), not validity.

7. **Composite visualization.** The decision-flipping edit lives in feature
   space. For the figure, the image patches aligned with the swapped 7×7 cells
   (each a 32×32 pixel block) are pasted from the distractor into the query. This
   composite is the paper's standard visualization; pixel change metrics
   (`l1_mean`, `changed_pixel_fraction`, …) are computed on it.

8. **Outputs per sample.** Original with edited cells boxed, distractor with
   source cells boxed, composite counterfactual, difference map, overlay; plus
   the full per-edit list (cell coordinates and target-probability trajectory),
   change metrics, embedding distance, and diagnostics — all written to
   `metadata.json`.

---

## 3. Soll-Ist comparison with the original

| Aspect | Original (Goyal et al. 2019 / Meta baseline) | This implementation | Status |
|---|---|---|---|
| Network split | spatial extractor `f` + decision head `g` | `f` = ResNet18 through `layer4`, `g` = GAP + FC | ✅ |
| Edit space | swap spatial cells of the last conv feature map | identical (7×7×512 `layer4` cells) | ✅ |
| Search | greedy exhaustive over all `(i, j)` cell pairs | identical | ✅ |
| Selection criterion | maximize target-class score per step | maximize target-class softmax probability | ✅ mathematically equivalent |
| Permutation constraint `P` | each query/distractor cell used at most once | identical (open-cell masks) | ✅ |
| Stopping | first prediction flip to the target class | identical | ✅ |
| Sparsity metric | number of edited cells `||a||_1` | `num_edits` in `metadata.json` | ✅ |
| Distractor | an image of the target ("distractor") class | nearest correctly classified target-class training image | ✅ standard instantiation |

---

## 4. Deliberate differences from the original (not fidelity problems)

1. **Framework and classifier.** PyTorch / ResNet-18 on medical images instead
   of the original VGG-16/ResNet-50 on CUB-200. This is the intended project
   substitution. The feature grid is 7×7 (49 cells) at 224×224, versus the
   larger grids in the original bird-classification setup.

2. **Distractor selection made explicit.** The paper assumes a given `(I, I')`
   pair. This runner fixes `I'` as the nearest-unlike training neighbor of the
   target class, so the choice is deterministic and reproducible. The retrieval
   reuses the same feature space as the former retrieval-NUN baseline, which this
   method replaces.

3. **Incremental pooled-mean update.** Candidate feature maps are scored via the
   exact incremental mean instead of materializing 49×49 full `[512, 7, 7]`
   tensors. The result is bit-for-bit identical to evaluating `g` on each edited
   map; it only reduces runtime and memory.

4. **`--max_edits` cap.** Defaults to the full grid (49). The paper's stopping
   rule is the first flip, so this cap only bounds pathological cases and does
   not change any reported counterfactual.

---

## 5. Key parameters

| Argument | Default | Meaning |
|---|---|---|
| `--model_path` | — | ResNet18 classifier checkpoint |
| `--dataset_path` | — | Processed dataset root (`train`/`val`/`test`) |
| `--manifest_path` | — | Fixed evaluation manifest; samples and targets come from it |
| `--max_edits` | full grid (49) | Maximum number of cell swaps before stopping |
| `--batch_size` | 32 | Batch size for building the distractor database |
| `--change_threshold` | 0.03 | Pixel threshold for `changed_pixel_fraction` |

---

## 6. Results on the fixed manifests

| Dataset | Samples | Validity | Mean CF confidence | Mean edits | Mean changed pixel fraction | Mean runtime |
|---|---:|---:|---:|---:|---:|---:|
| BUSI | 15 | 1.00 | 0.5279 | 14.0 | 0.2596 | 0.25s |
| Pneumonia | 20 | 1.00 | 0.5231 | 16.15 | 0.3072 | 0.17s |

Reading the numbers:

- **Validity is 1.00 by construction** because the full-grid budget converges the
  pooled feature to the distractor's. The informative sparsity signal is the mean
  number of edits (14.0 / 16.15 of 49 cells), i.e. roughly a third of the feature
  grid suffices to flip the decision.
- **Mean CF confidence sits near 0.5** because the greedy search stops at the
  *first* flip; it prioritizes sparsity over margin, so the counterfactual lands
  just past the decision boundary rather than deep inside the target region.
- **Runtime is sub-second** because the search is a small number of batched
  linear evaluations; the one-time cost is building the distractor database over
  the training split.
