import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.segmentation import quickshift
from torchvision import models, utils

try:
    import cv2
except ImportError:
    cv2 = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_utils import create_dataloaders
from src.evaluation_manifest import (
    load_image_from_manifest_record,
    load_manifest_records,
    manifest_record_metadata,
)
from src.train_model import get_device

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def create_model(num_classes):
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def denormalize(images):
    mean = IMAGENET_MEAN.to(images.device)
    std = IMAGENET_STD.to(images.device)
    return (images * std + mean).clamp(0.0, 1.0)


def normalize(images):
    mean = IMAGENET_MEAN.to(images.device)
    std = IMAGENET_STD.to(images.device)
    return (images - mean) / std


def image_to_grayscale(image):
    return image.mean(dim=0).detach().cpu()


def load_checkpoint_model(model_path, device):
    checkpoint = torch.load(model_path, map_location=device)
    model = create_model(checkpoint["num_classes"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model, checkpoint


def predict(model, pixels):
    with torch.no_grad():
        logits = model(normalize(pixels))
        probabilities = F.softmax(logits, dim=1)
        prediction = int(torch.argmax(probabilities, dim=1).item())
    return prediction, probabilities[0].detach().cpu().tolist()


def choose_samples(model, test_loader, device, max_samples):
    samples = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            probabilities = F.softmax(logits, dim=1)
            predictions = torch.argmax(probabilities, dim=1)

            for idx in range(images.shape[0]):
                if predictions[idx] == labels[idx]:
                    samples.append(
                        {
                            "image": images[idx : idx + 1].detach(),
                            "label": int(labels[idx].item()),
                            "prediction": int(predictions[idx].item()),
                            "probabilities": probabilities[idx].detach().cpu().tolist(),
                        }
                    )
                    if len(samples) >= max_samples:
                        return samples

    return samples


def choose_manifest_samples(
    model, test_dataset, device, manifest_path, max_records=None
):
    manifest, records = load_manifest_records(manifest_path, max_records=max_records)
    samples = []

    with torch.no_grad():
        for record in records:
            image, label, image_path = load_image_from_manifest_record(
                test_dataset, record
            )
            image = image.to(device)
            logits = model(image)
            probabilities = F.softmax(logits, dim=1)
            prediction = int(torch.argmax(probabilities, dim=1).item())
            expected_prediction = int(record["original_prediction_index"])

            if prediction != expected_prediction:
                raise ValueError(
                    "Current model prediction does not match manifest for "
                    f"manifest sample {record['manifest_sample_index']}: "
                    f"manifest={expected_prediction}, current={prediction}"
                )

            samples.append(
                {
                    **manifest_record_metadata(record),
                    "image": image.detach(),
                    "label": label,
                    "prediction": prediction,
                    "probabilities": probabilities[0].detach().cpu().tolist(),
                    "target_class_index": int(record["target_class_index"]),
                    "target_class": record["target_class"],
                    "image_source_path": image_path,
                }
            )

    return manifest, samples


def select_target_classes(probabilities, original_class, strategy):
    ranked = torch.argsort(torch.tensor(probabilities), descending=True).tolist()
    candidates = [class_idx for class_idx in ranked if class_idx != original_class]

    if strategy == "all":
        return candidates
    if not candidates:
        return [int((original_class + 1) % len(probabilities))]
    return [candidates[0]]


def create_segments(
    image_pixels,
    quickshift_kernel_size,
    quickshift_max_dist,
    quickshift_ratio,
):
    image_np = image_pixels[0].detach().cpu().permute(1, 2, 0).numpy()
    return quickshift(
        image_np,
        kernel_size=quickshift_kernel_size,
        max_dist=quickshift_max_dist,
        ratio=quickshift_ratio,
        convert2lab=True,
        channel_axis=-1,
    )


def normalize_blur_kernel(blur_kernel):
    if blur_kernel < 1:
        raise ValueError("blur_kernel must be >= 1.")
    if blur_kernel % 2 == 0:
        corrected = blur_kernel + 1
        print(
            f"WARNING: blur_kernel={blur_kernel} is even; using {corrected} "
            "for Gaussian/average blur."
        )
        return corrected
    return blur_kernel


def create_replacement_image(image_pixels, blur_kernel):
    if cv2 is None:
        raise ImportError(
            "SEDC-T Gaussian-blur replacement requires OpenCV. "
            "Install opencv-python with `pip install -r requirements.txt`."
        )

    blur_kernel = normalize_blur_kernel(blur_kernel)
    image_np = image_pixels[0].detach().cpu().permute(1, 2, 0).numpy()
    blurred_np = cv2.GaussianBlur(image_np, (blur_kernel, blur_kernel), 0)
    blurred_np = np.clip(blurred_np, 0.0, 1.0)
    return (
        torch.from_numpy(blurred_np)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device=image_pixels.device, dtype=image_pixels.dtype)
    )


def replace_segments(image_pixels, replacement_pixels, segments, selected_segments):
    if not selected_segments:
        return image_pixels.clone()

    mask = torch.zeros(
        1, 1, image_pixels.shape[2], image_pixels.shape[3], device=image_pixels.device
    )
    segments_tensor = torch.as_tensor(segments, device=image_pixels.device)
    for segment_id in selected_segments:
        mask[0, 0] = torch.logical_or(
            mask[0, 0].bool(), segments_tensor == segment_id
        ).float()

    return image_pixels * (1.0 - mask) + replacement_pixels * mask


def compute_segment_pixel_fraction(segments, selected_segments):
    if not selected_segments:
        return 0.0

    segments_tensor = torch.as_tensor(segments)
    segment_mask = torch.zeros(segments.shape, dtype=torch.bool)
    for segment_id in selected_segments:
        segment_mask |= segments_tensor == segment_id

    return float(segment_mask.float().mean().item())


def create_roi_mask(shape, roi_mode):
    height, width = shape
    mask = torch.zeros(height, width, dtype=torch.bool)

    if roi_mode == "none":
        mask[:, :] = True
        return mask

    if roi_mode == "lung_fields":
        y_min = int(height * 0.15)
        y_max = int(height * 0.88)
        left_x_min = int(width * 0.08)
        left_x_max = int(width * 0.47)
        right_x_min = int(width * 0.53)
        right_x_max = int(width * 0.92)

        mask[y_min:y_max, left_x_min:left_x_max] = True
        mask[y_min:y_max, right_x_min:right_x_max] = True
        return mask

    raise ValueError(f"Unsupported ROI mode: {roi_mode}")


def get_allowed_segments(segments, roi_mode, roi_min_overlap):
    all_segments = set(
        int(segment_id) for segment_id in torch.unique(torch.as_tensor(segments))
    )
    segments_tensor = torch.as_tensor(segments)
    roi_mask = create_roi_mask(segments.shape, roi_mode)

    allowed = set()
    for segment_id in all_segments:
        segment_mask = segments_tensor == segment_id
        roi_overlap = float((segment_mask & roi_mask).float().sum().item())
        roi_overlap /= float(segment_mask.float().sum().item())
        if roi_overlap >= roi_min_overlap:
            allowed.add(segment_id)

    return allowed, roi_mask


def evaluate_segment_set(
    model,
    image_pixels,
    original_class,
    target_class,
    original_original_class_probability,
    original_target_probability,
    segments,
    replacement_pixels,
    selected_segments,
    added_segment=None,
):
    candidate_pixels = replace_segments(
        image_pixels, replacement_pixels, segments, selected_segments
    )
    prediction, probabilities = predict(model, candidate_pixels)
    target_probability = float(probabilities[target_class])
    original_class_probability = float(probabilities[original_class])
    target_score_increase = target_probability - original_target_probability
    original_class_score_decrease = (
        original_original_class_probability - original_class_probability
    )
    expansion_score = target_probability - original_class_probability

    return {
        "segment_id": added_segment,
        "segments": sorted(int(segment_id) for segment_id in selected_segments),
        "prediction": prediction,
        "probabilities": probabilities,
        "target_probability": target_probability,
        "original_class_probability": original_class_probability,
        "target_score_increase": target_score_increase,
        "original_class_score_decrease": original_class_score_decrease,
        "expansion_score": expansion_score,
        "pixels": candidate_pixels,
        "changed_pixel_fraction": compute_segment_pixel_fraction(
            segments, selected_segments
        ),
    }


def generate_sedc_t_original_best_first_counterfactual(
    model,
    image_pixels,
    target_class,
    segments,
    replacement_pixels,
    allowed_segments,
    timeout_seconds,
):
    """Original-style SEDC-T best-first search by class-margin expansion.

    This mirrors the reference implementation's core search more closely than
    the project-specific greedy-minimal variant: evaluate all one-segment
    removals/replacements, repeatedly expand the pending segment set with the
    largest target-vs-original class probability margin, and choose valid
    results by minimal segment evidence first.
    """

    start_time = time.time()
    allowed_segments = sorted(allowed_segments)
    if not allowed_segments:
        raise RuntimeError("No segments available for SEDC-T search.")

    original_class, original_probabilities = predict(model, image_pixels)
    original_original_class_probability = float(original_probabilities[original_class])
    original_target_probability = float(original_probabilities[target_class])
    pending = []
    valid_candidates = []
    evaluated_candidates = []
    seen_sets = set()
    search_history = []

    def timed_out():
        return (
            timeout_seconds is not None and time.time() - start_time > timeout_seconds
        )

    def add_candidate(candidate, step_label):
        evaluated_candidates.append(candidate)
        search_history.append(
            {
                "step": len(search_history) + 1,
                "phase": step_label,
                "selected_segments": candidate["segments"],
                "prediction": candidate["prediction"],
                "target_probability": candidate["target_probability"],
                "original_class_probability": candidate["original_class_probability"],
                "target_score_increase": candidate["target_score_increase"],
                "original_class_score_decrease": candidate[
                    "original_class_score_decrease"
                ],
                "expansion_score": candidate["expansion_score"],
            }
        )

        if candidate["prediction"] == target_class:
            valid_candidates.append(candidate)
        else:
            pending.append(candidate)

    for segment_id in allowed_segments:
        segment_set = (segment_id,)
        seen_sets.add(segment_set)
        candidate = evaluate_segment_set(
            model=model,
            image_pixels=image_pixels,
            original_class=original_class,
            target_class=target_class,
            original_original_class_probability=original_original_class_probability,
            original_target_probability=original_target_probability,
            segments=segments,
            replacement_pixels=replacement_pixels,
            selected_segments=list(segment_set),
            added_segment=segment_id,
        )
        add_candidate(candidate, "initial")
        if timed_out():
            break

    while not valid_candidates and pending and not timed_out():
        expandable_indices = list(range(len(pending)))
        if not expandable_indices:
            break

        best_pending_idx = max(
            expandable_indices,
            key=lambda idx: pending[idx]["expansion_score"],
        )
        base_candidate = pending.pop(best_pending_idx)
        base_segments = set(base_candidate["segments"])

        for segment_id in allowed_segments:
            if segment_id in base_segments:
                continue

            segment_set = tuple(sorted([*base_segments, segment_id]))
            if segment_set in seen_sets:
                continue

            seen_sets.add(segment_set)
            candidate = evaluate_segment_set(
                model=model,
                image_pixels=image_pixels,
                original_class=original_class,
                target_class=target_class,
                original_original_class_probability=original_original_class_probability,
                original_target_probability=original_target_probability,
                segments=segments,
                replacement_pixels=replacement_pixels,
                selected_segments=list(segment_set),
                added_segment=segment_id,
            )
            add_candidate(candidate, "expand")
            if timed_out():
                break

    runtime = time.time() - start_time
    timed_out_flag = timed_out()

    if not evaluated_candidates:
        raise RuntimeError("SEDC-T best-first search did not evaluate any candidates.")

    if valid_candidates:
        # The reference SEDC-T implementation stops as soon as a valid
        # expansion level is found and then selects the candidate with the
        # highest target-score increase within that level.
        best_result = max(
            valid_candidates,
            key=lambda candidate: (
                candidate["target_score_increase"],
                candidate["target_probability"],
                candidate["expansion_score"],
            ),
        )
    else:
        best_result = max(
            evaluated_candidates,
            key=lambda candidate: (
                candidate["expansion_score"],
                candidate["target_probability"],
            ),
        )

    return {
        "image": best_result["pixels"],
        "selected_segments": best_result["segments"],
        "prediction": best_result["prediction"],
        "probabilities": best_result["probabilities"],
        "valid": best_result["prediction"] == target_class,
        "runtime_seconds": runtime,
        "search_history": search_history,
        "search_diagnostics": {
            "original_class": original_class,
            "original_original_class_probability": original_original_class_probability,
            "original_target_probability": original_target_probability,
            "evaluated_candidates": len(evaluated_candidates),
            "valid_candidates": len(valid_candidates),
            "pending_candidates": len(pending),
            "timed_out": timed_out_flag,
        },
    }


def compute_change_metrics(original_pixels, cf_pixels, selected_segments, segments):
    diff = torch.abs(original_pixels - cf_pixels)
    segment_mask = torch.zeros(segments.shape, dtype=torch.bool)
    for segment_id in selected_segments:
        segment_mask |= torch.as_tensor(segments) == segment_id

    return {
        "l1_mean": float(diff.mean().item()),
        "l2_mean": float(torch.sqrt(torch.mean(diff**2)).item()),
        "linf": float(diff.max().item()),
        "num_changed_segments": len(selected_segments),
        "num_total_segments": int(torch.unique(torch.as_tensor(segments)).numel()),
        "changed_segment_fraction": len(selected_segments)
        / int(torch.unique(torch.as_tensor(segments)).numel()),
        "changed_pixel_fraction": float(segment_mask.float().mean().item()),
    }


def save_sedc_t_visualization(
    original_pixels,
    cf_pixels,
    segments,
    selected_segments,
    output_path,
    true_label,
    original_prediction,
    original_confidence,
    target_class,
    counterfactual_prediction,
    counterfactual_confidence,
    valid_counterfactual,
):
    utils.save_image(
        torch.cat([original_pixels, cf_pixels], dim=0), output_path, nrow=2
    )

    figure_path = output_path.with_suffix(".summary.png")
    fig, axes = plt.subplots(1, 4, figsize=(13, 4.8))

    original_gray = image_to_grayscale(original_pixels[0])
    cf_gray = image_to_grayscale(cf_pixels[0])
    diff_image = image_to_grayscale(torch.abs(original_pixels[0] - cf_pixels[0]))

    selected_mask = torch.zeros(segments.shape, dtype=torch.float32)
    for segment_id in selected_segments:
        selected_mask[torch.as_tensor(segments) == segment_id] = 1.0

    axes[0].imshow(original_gray, cmap="gray", vmin=0.0, vmax=1.0)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(cf_gray, cmap="gray", vmin=0.0, vmax=1.0)
    axes[1].set_title("Counterfactual")
    axes[1].axis("off")

    diff_plot = axes[2].imshow(diff_image, cmap="gray", vmin=0.0, vmax=1.0)
    axes[2].set_title("Difference")
    axes[2].axis("off")
    fig.colorbar(diff_plot, ax=axes[2], fraction=0.046, pad=0.04)

    axes[3].imshow(original_gray, cmap="gray", vmin=0.0, vmax=1.0)
    selected_overlay = np.ma.masked_where(
        selected_mask.numpy() == 0, selected_mask.numpy()
    )
    overlay_plot = axes[3].imshow(
        selected_overlay,
        cmap="autumn",
        vmin=0.0,
        vmax=1.0,
        alpha=0.45,
    )
    axes[3].set_title("Selected segments")
    axes[3].axis("off")
    fig.colorbar(overlay_plot, ax=axes[3], fraction=0.046, pad=0.04)

    valid_text = "yes" if valid_counterfactual else "no"
    title = (
        f"True label: {true_label}\n"
        f"Target: {original_prediction} -> {target_class}\n"
        f"Prediction: {original_prediction} ({original_confidence:.2f}) -> "
        f"{counterfactual_prediction} ({counterfactual_confidence:.2f})\n"
        f"Valid CF: {valid_text}"
    )
    fig.suptitle(title, y=0.98, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.78])
    fig.savefig(figure_path, dpi=150)
    plt.close(fig)


def compute_aggregate_metrics(records):
    valid_count = sum(record["valid_counterfactual"] for record in records)
    aggregate = {
        "num_samples": len(records),
        "valid_count": valid_count,
        "validity": valid_count / len(records) if records else 0.0,
    }

    if not records:
        return aggregate

    for metric_name in [
        "l1_mean",
        "l2_mean",
        "linf",
        "num_changed_segments",
        "changed_segment_fraction",
        "changed_pixel_fraction",
    ]:
        values = [record["change_metrics"][metric_name] for record in records]
        aggregate[metric_name] = {
            "mean": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
        }

    runtimes = [record["runtime_seconds"] for record in records]
    aggregate["runtime_seconds"] = {
        "mean": sum(runtimes) / len(runtimes),
        "min": min(runtimes),
        "max": max(runtimes),
    }

    return aggregate


def is_original_style_reference(args):
    return args.roi_mode == "none"


def method_fidelity_note(args):
    if is_original_style_reference(args):
        return (
            "Original-style SEDC-T reference: best-first search, quickshift "
            "segmentation, gaussian blur replacement, no ROI restriction, "
            "no explicit segment-count cap."
        )
    return (
        "Project-specific SEDC-T ROI ablation: same best-first search, "
        "quickshift segmentation, and gaussian blur replacement as the "
        "original-style reference, but candidate segments are restricted to "
        "a geometric lung-field ROI. This is not part of the original SEDC-T "
        "setup."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_samples", type=int, default=3)
    parser.add_argument(
        "--manifest_path",
        type=str,
        default=None,
        help="Optional fixed evaluation manifest. If set, samples and targets come from this JSON.",
    )
    parser.add_argument(
        "--manifest_max_samples",
        type=int,
        default=None,
        help="Optional cap for manifest mode. By default all manifest records are used.",
    )
    parser.add_argument(
        "--target_strategy", choices=["second_best", "all"], default="all"
    )
    parser.add_argument("--quickshift_kernel_size", type=int, default=4)
    parser.add_argument("--quickshift_max_dist", type=float, default=200.0)
    parser.add_argument("--quickshift_ratio", type=float, default=0.2)
    parser.add_argument(
        "--search_timeout_seconds",
        type=float,
        default=30.0,
        help=(
            "Per-image timeout for original_best_first search. "
            "Use 0 or a negative value to disable the timeout."
        ),
    )
    parser.add_argument("--blur_kernel", type=int, default=31)
    parser.add_argument(
        "--roi_mode",
        choices=["none", "lung_fields"],
        default="none",
        help=(
            "none is the original-style reference. lung_fields is the retained "
            "Pneumonia ROI ablation."
        ),
    )
    parser.add_argument(
        "--roi_min_overlap",
        type=float,
        default=0.50,
        help="Minimum fraction of a segment that must overlap the ROI to be selectable.",
    )
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    device = get_device()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, checkpoint = load_checkpoint_model(args.model_path, device)
    classes = checkpoint["classes"]

    data = create_dataloaders(
        args.dataset_path, batch_size=args.batch_size, use_augmentation=False
    )

    print(f"Device: {device}")
    print(f"Classes: {classes}")
    if is_original_style_reference(args):
        print("Running original-style SEDC-T reference.")
    else:
        print(f"Running project-specific SEDC-T ablation: {method_fidelity_note(args)}")
    manifest = None
    if args.manifest_path:
        print(f"Loading fixed evaluation manifest: {args.manifest_path}")
        manifest, samples = choose_manifest_samples(
            model=model,
            test_dataset=data["test_dataset"],
            device=device,
            manifest_path=args.manifest_path,
            max_records=args.manifest_max_samples,
        )
    else:
        print("Selecting correctly classified test samples...")
        samples = choose_samples(model, data["test_loader"], device, args.max_samples)
    if not samples:
        raise RuntimeError("No correctly classified test samples found.")

    timeout_seconds = (
        args.search_timeout_seconds
        if args.search_timeout_seconds and args.search_timeout_seconds > 0
        else None
    )
    records = []
    for sample_idx, sample in enumerate(samples):
        image = sample["image"]
        original_pixels = denormalize(image).detach()
        replacement_pixels = create_replacement_image(original_pixels, args.blur_kernel)
        segments = create_segments(
            image_pixels=original_pixels,
            quickshift_kernel_size=args.quickshift_kernel_size,
            quickshift_max_dist=args.quickshift_max_dist,
            quickshift_ratio=args.quickshift_ratio,
        )
        allowed_segments, roi_mask = get_allowed_segments(
            segments,
            args.roi_mode,
            args.roi_min_overlap,
        )
        if not allowed_segments:
            raise RuntimeError(
                "No allowed SEDC-T segments remain after ROI filtering. "
                "Use --roi_mode none or lower --roi_min_overlap."
            )

        original_class = sample["prediction"]
        original_confidence = float(sample["probabilities"][original_class])
        if "target_class_index" in sample:
            target_candidates = [sample["target_class_index"]]
        else:
            target_candidates = select_target_classes(
                sample["probabilities"], original_class, args.target_strategy
            )

        best_attempt = None
        attempted_targets = []
        for target_class in target_candidates:
            print(
                f"Sample {sample_idx}: {classes[original_class]} -> {classes[target_class]}"
            )
            result = generate_sedc_t_original_best_first_counterfactual(
                model=model,
                image_pixels=original_pixels,
                target_class=target_class,
                segments=segments,
                replacement_pixels=replacement_pixels,
                allowed_segments=allowed_segments,
                timeout_seconds=timeout_seconds,
            )
            attempt = {
                "target_class_index": target_class,
                "target_class": classes[target_class],
                "result": result,
            }
            attempted_targets.append(attempt)

            if result["valid"]:
                best_attempt = attempt
                break

            if (
                best_attempt is None
                or result["probabilities"][target_class]
                > best_attempt["result"]["probabilities"][
                    best_attempt["target_class_index"]
                ]
            ):
                best_attempt = attempt

        target_class = best_attempt["target_class_index"]
        result = best_attempt["result"]
        counterfactual_confidence = float(result["probabilities"][result["prediction"]])
        valid_counterfactual = result["prediction"] == target_class

        output_path = output_dir / f"sample_{sample_idx:02d}.png"
        save_sedc_t_visualization(
            original_pixels=original_pixels,
            cf_pixels=result["image"],
            segments=segments,
            selected_segments=result["selected_segments"],
            output_path=output_path,
            true_label=classes[sample["label"]],
            original_prediction=classes[original_class],
            original_confidence=original_confidence,
            target_class=classes[target_class],
            counterfactual_prediction=classes[result["prediction"]],
            counterfactual_confidence=counterfactual_confidence,
            valid_counterfactual=valid_counterfactual,
        )

        change_metrics = compute_change_metrics(
            original_pixels, result["image"], result["selected_segments"], segments
        )
        record = {
            "sample_index": sample_idx,
            **{
                key: sample[key]
                for key in [
                    "manifest_sample_index",
                    "dataset_index",
                    "source_image_path",
                ]
                if key in sample
            },
            "true_label_index": sample["label"],
            "true_label": classes[sample["label"]],
            "original_prediction_index": original_class,
            "original_prediction": classes[original_class],
            "original_confidence": original_confidence,
            "target_class_index": target_class,
            "target_class": classes[target_class],
            "counterfactual_prediction_index": result["prediction"],
            "counterfactual_prediction": classes[result["prediction"]],
            "counterfactual_confidence": counterfactual_confidence,
            "valid_counterfactual": valid_counterfactual,
            "selected_segments": result["selected_segments"],
            "attempted_targets": [
                {
                    "target_class": attempt["target_class"],
                    "valid": attempt["result"]["valid"],
                    "counterfactual_prediction": classes[
                        attempt["result"]["prediction"]
                    ],
                    "selected_segments": attempt["result"]["selected_segments"],
                }
                for attempt in attempted_targets
            ],
            "original_probabilities": sample["probabilities"],
            "counterfactual_probabilities": result["probabilities"],
            "runtime_seconds": result["runtime_seconds"],
            "search_history": result["search_history"],
            "search_diagnostics": result.get("search_diagnostics"),
            "change_metrics": change_metrics,
            "candidate_segments": {
                "allowed": len(allowed_segments),
                "total": int(torch.unique(torch.as_tensor(segments)).numel()),
                "roi_pixel_fraction": float(roi_mask.float().mean().item()),
            },
            "image_path": str(output_path),
            "summary_path": str(output_path.with_suffix(".summary.png")),
        }
        records.append(record)

    metadata = {
        "method": (
            "SEDC-T original-style best-first segment replacement"
            if is_original_style_reference(args)
            else "SEDC-T-style lung-field ROI ablation"
        ),
        "model_path": args.model_path,
        "dataset_path": args.dataset_path,
        "classes": classes,
        "method_fidelity_note": method_fidelity_note(args),
        "parameters": {
            "segmentation_method": "quickshift",
            "quickshift_kernel_size": args.quickshift_kernel_size,
            "quickshift_max_dist": args.quickshift_max_dist,
            "quickshift_ratio": args.quickshift_ratio,
            "search_mode": "original_best_first",
            "search_timeout_seconds": timeout_seconds,
            "replacement_mode": "gaussian_blur",
            "blur_kernel": args.blur_kernel,
            "roi_mode": args.roi_mode,
            "roi_min_overlap": args.roi_min_overlap,
            "target_strategy": args.target_strategy,
            "manifest_path": args.manifest_path,
            "manifest_max_samples": args.manifest_max_samples,
        },
        "evaluation_manifest": {
            "path": args.manifest_path,
            "num_records_available": manifest.get("num_samples") if manifest else None,
            "num_records_used": len(samples) if manifest else None,
            "target_strategy": manifest.get("target_strategy") if manifest else None,
        },
        "records": records,
        "aggregate_metrics": compute_aggregate_metrics(records),
    }

    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)

    valid_count = sum(record["valid_counterfactual"] for record in records)
    print(f"Saved {len(records)} SEDC-T attempts to {output_dir}")
    print(f"Valid counterfactuals: {valid_count}/{len(records)}")


if __name__ == "__main__":
    main()
