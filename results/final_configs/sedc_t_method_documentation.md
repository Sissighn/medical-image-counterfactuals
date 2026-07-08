# SEDC-T Method Documentation

Implementation: [`scripts/run_sedc_t_pytorch.py`](../../scripts/run_sedc_t_pytorch.py)

Reference (original code):
<https://github.com/ADMAntwerp/ImageCounterfactualExplanations/blob/main/isedc/sedc_t2_fast.py>

Run commands: [`sedc_t_run_commands.md`](sedc_t_run_commands.md)

---

## 1. What SEDC-T is

SEDC-T (Search for Explanations for Document/image Classification — Targeted) is
a model-agnostic, perturbation-based counterfactual method. Given an image, a
classifier, and a desired **target class**, it searches for the smallest set of
image segments whose perturbation flips the classifier's prediction to that
target class. It is a *targeted* best-first search: at each step it prefers the
partial solution that most increases the target-vs-current class score margin.

This project ports the reference `sedc_t2_fast` (best-first variant) to
PyTorch / ResNet-18. The search logic follows the original; the framework and
classifier are the intended project-specific substitutions.

---

## 2. How the implementation works, step by step

1. **Segmentation.** The image is over-segmented with quickshift
   (`kernel_size=4`, `max_dist=200`, `ratio=0.2`, Lab color space). These are
   the exact parameters used in the reference repository's experiment scripts.

2. **Perturbation reference image.** A fully perturbed version of the image is
   built once, using one of four modes from the original:
   - `mean` — each channel filled with its mean value
   - `blur` — Gaussian blur, kernel `31×31` (project default)
   - `random` — uniform random pixels
   - `inpaint` — segment-wise Navier–Stokes inpainting (`cv2.inpaint`)

   Perturbing a segment means copying that segment's pixels from this reference
   image into the working image.

3. **Initial level.** Every single allowed segment is perturbed on its own,
   producing one candidate per segment. All candidates are classified in a
   single batched forward pass.
   - Any candidate already predicted as the target class becomes a **valid
     counterfactual**.
   - All others are kept as **pending nodes** to expand, each scored by the
     expansion score `p_target − p_current_class`.

4. **Best-first expansion.** While no valid counterfactual exists yet and
   pending nodes remain: take the single pending node with the highest
   expansion score, remove it, and create children by adding each not-yet-used
   segment to it. All children of that node are classified in one batch. Valid
   children are recorded; the rest become new pending nodes. The loop repeats
   one node per iteration.

5. **Termination.** The search stops as soon as a level produces a valid
   counterfactual, or when no pending nodes remain, or when an expansion
   produces no children, or when the per-target timeout is reached (checked once
   per expansion level, matching the reference `max_time`).

6. **Selection.** Among all valid counterfactuals, the one with the highest
   target-score increase is chosen (the reference `np.argmax(P − p)`).

7. **Outputs per sample.** The counterfactual image, the "explanation" image
   (original pixels of the selected segments on a black background, as in the
   reference), a side-by-side/summary visualization, change metrics, the full
   search history, and diagnostics — all written to `metadata.json`.

---

## 3. Soll-Ist comparison with the original

| Aspect | Original (`sedc_t2_fast.py`) | This implementation | Status |
|---|---|---|---|
| Segmentation | quickshift(4, 200, 0.2), passed in externally | identical | ✅ verified against repo experiment scripts |
| Replacement modes | mean / blur / random / inpaint | all four implemented | ✅ |
| Blur perturbation | applied once to the whole image, then copied in per segment | identical | ✅ |
| Initial candidates | each segment alone, one batched predict | identical (batched) | ✅ |
| Expansion | **one** pending node (highest expansion score) expanded per step; its children classified in one batch | identical — exactly one parent per while-iteration | ✅ |
| Stop condition | `len(R)==0 and len(combo_set)>0 and max_time>elapsed` | `not valid_candidates and pending and expansion_produced_children and not timed_out()` | ✅ |
| Selection on success | `argmax(P − p)` over valid candidates of the successful level | `max(..., key=target_score_increase)` | ✅ mathematically identical |
| Explanation image | original pixels of selected segments, rest set to 0 | identical | ✅ |
| Timeout check frequency | once per node expansion, not per forward pass | identical | ✅ |
| Expansion score | `p_target − p_current_class` | identical | ✅ |

---

## 4. Deliberate differences from the original (not fidelity problems)

1. **Framework and classifier.** PyTorch / ResNet-18 instead of the original
   TensorFlow/Keras classifier. This is the intended project substitution and
   is expected to differ.

2. **No counterfactual found.** The original returns `None`. This
   implementation instead keeps the best non-valid attempt with
   `valid_counterfactual = false`, so that failure cases still produce a record
   for the evaluation. The reference "No CF found on the requested parameters"
   message is still printed.

3. **Search timeout default 30 s (reference: 600 s).** On these 224×224 medical
   images every valid counterfactual is found within ~3 s (measured on the full
   BUSI set), so a 30 s cap only bounds the wait on samples that have no
   counterfactual at all and drops no counterfactual. Use
   `--search_timeout_seconds 600` to match the reference exactly; the set of
   samples that find a counterfactual is identical either way.

4. **Batched prediction.** Each search level is classified in a single forward
   pass (like the reference `classifier.predict(cf_candidates)`), rather than
   one image at a time. This was verified to produce bit-for-bit identical
   segment selections versus the sequential version; it only reduces runtime.

5. **Shape-agnostic perturbations.** The original hardcodes `(224, 224, 3)` for
   the mean/random/inpaint reference images because its loader always resizes to
   224×224. This implementation derives the shape from the actual image —
   identical result at 224×224, but more robust.

6. **Lung-field ROI ablation (`--roi_mode lung_fields`).** This does **not**
   exist in the original SEDC-T. It is a project-specific ablation that
   restricts perturbable segments to a coarse geometric lung-field prior (two
   rectangles with a central mediastinum gap; the exact fractions are recorded
   per run under `parameters.roi_lung_field_geometry` in `metadata.json`). It is
   explicitly not part of the fidelity comparison. The default `--roi_mode none`
   is the original-style reference.

---

## 5. Key parameters

| Argument | Default | Meaning |
|---|---|---|
| `--replacement_mode` | `blur` | Perturbation mode (mean/blur/random/inpaint) |
| `--blur_kernel` | `31` | Gaussian blur kernel size (blur mode) |
| `--quickshift_kernel_size` | `4` | quickshift kernel size |
| `--quickshift_max_dist` | `200` | quickshift max distance |
| `--quickshift_ratio` | `0.2` | quickshift color-vs-space ratio |
| `--search_timeout_seconds` | `30` | Per-target search timeout (600 = reference) |
| `--roi_mode` | `none` | `none` = original-style reference; `lung_fields` = ablation |
| `--roi_min_overlap` | `0.50` | Min. fraction of a segment inside the ROI to be selectable |
| `--manifest_path` | — | Fixed evaluation manifest; samples and targets come from it |
