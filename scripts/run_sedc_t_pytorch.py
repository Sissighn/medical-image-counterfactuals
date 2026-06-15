import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.segmentation import slic
from torchvision import models, utils

from src.data_utils import create_dataloaders
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


def select_target_classes(probabilities, original_class, strategy):
    ranked = torch.argsort(torch.tensor(probabilities), descending=True).tolist()
    candidates = [class_idx for class_idx in ranked if class_idx != original_class]

    if strategy == "all":
        return candidates
    if not candidates:
        return [int((original_class + 1) % len(probabilities))]
    return [candidates[0]]


def create_segments(image_pixels, n_segments, compactness):
    image_np = image_pixels[0].detach().cpu().permute(1, 2, 0).numpy()
    return slic(
        image_np,
        n_segments=n_segments,
        compactness=compactness,
        start_label=0,
        channel_axis=-1,
    )


def create_replacement_image(image_pixels, mode, blur_kernel):
    if mode == "mean":
        mean_value = image_pixels.mean(dim=(2, 3), keepdim=True)
        return mean_value.expand_as(image_pixels)

    if mode == "blur":
        padding = blur_kernel // 2
        return F.avg_pool2d(
            image_pixels, kernel_size=blur_kernel, stride=1, padding=padding
        ).clamp(0.0, 1.0)

    raise ValueError(f"Unsupported replacement mode: {mode}")


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


def get_allowed_segments(segments, border_fraction):
    all_segments = set(int(segment_id) for segment_id in torch.unique(torch.as_tensor(segments)))
    if border_fraction <= 0:
        return all_segments

    height, width = segments.shape
    min_y = int(height * border_fraction)
    max_y = int(height * (1.0 - border_fraction))
    min_x = int(width * border_fraction)
    max_x = int(width * (1.0 - border_fraction))

    allowed = set()
    for segment_id in all_segments:
        ys, xs = torch.where(torch.as_tensor(segments) == segment_id)
        if (
            int(ys.min()) >= min_y
            and int(ys.max()) < max_y
            and int(xs.min()) >= min_x
            and int(xs.max()) < max_x
        ):
            allowed.add(segment_id)

    return allowed


def generate_sedc_t_counterfactual(
    model,
    image_pixels,
    target_class,
    segments,
    replacement_pixels,
    max_segments,
    allowed_segments,
):
    start_time = time.time()
    available_segments = set(allowed_segments)
    selected_segments = []
    search_history = []
    best_result = None

    if not available_segments:
        raise RuntimeError("No segments available for SEDC-T search.")

    for step in range(max_segments):
        best_candidate = None

        for segment_id in sorted(available_segments):
            candidate_segments = selected_segments + [segment_id]
            candidate_pixels = replace_segments(
                image_pixels, replacement_pixels, segments, candidate_segments
            )
            prediction, probabilities = predict(model, candidate_pixels)
            target_probability = float(probabilities[target_class])

            candidate = {
                "segment_id": segment_id,
                "segments": candidate_segments,
                "prediction": prediction,
                "probabilities": probabilities,
                "target_probability": target_probability,
                "pixels": candidate_pixels,
            }

            if (
                best_candidate is None
                or candidate["target_probability"] > best_candidate["target_probability"]
            ):
                best_candidate = candidate

        selected_segments = best_candidate["segments"]
        available_segments.remove(best_candidate["segment_id"])
        search_history.append(
            {
                "step": step + 1,
                "added_segment": best_candidate["segment_id"],
                "selected_segments": selected_segments,
                "prediction": best_candidate["prediction"],
                "target_probability": best_candidate["target_probability"],
            }
        )
        best_result = best_candidate

        if best_candidate["prediction"] == target_class:
            break

        if not available_segments:
            break

    runtime = time.time() - start_time

    return {
        "image": best_result["pixels"],
        "selected_segments": selected_segments,
        "prediction": best_result["prediction"],
        "probabilities": best_result["probabilities"],
        "valid": best_result["prediction"] == target_class,
        "runtime_seconds": runtime,
        "search_history": search_history,
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
    utils.save_image(torch.cat([original_pixels, cf_pixels], dim=0), output_path, nrow=2)

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
    selected_overlay = np.ma.masked_where(selected_mask.numpy() == 0, selected_mask.numpy())
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_samples", type=int, default=3)
    parser.add_argument("--target_strategy", choices=["second_best", "all"], default="all")
    parser.add_argument("--n_segments", type=int, default=40)
    parser.add_argument("--compactness", type=float, default=10.0)
    parser.add_argument("--max_segments", type=int, default=6)
    parser.add_argument("--replacement_mode", choices=["blur", "mean"], default="blur")
    parser.add_argument("--blur_kernel", type=int, default=31)
    parser.add_argument("--exclude_border_fraction", type=float, default=0.0)
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
    print("Selecting correctly classified test samples...")
    samples = choose_samples(model, data["test_loader"], device, args.max_samples)
    if not samples:
        raise RuntimeError("No correctly classified test samples found.")

    records = []
    for sample_idx, sample in enumerate(samples):
        image = sample["image"]
        original_pixels = denormalize(image).detach()
        replacement_pixels = create_replacement_image(
            original_pixels, args.replacement_mode, args.blur_kernel
        )
        segments = create_segments(original_pixels, args.n_segments, args.compactness)
        allowed_segments = get_allowed_segments(segments, args.exclude_border_fraction)

        original_class = sample["prediction"]
        original_confidence = float(sample["probabilities"][original_class])
        target_candidates = select_target_classes(
            sample["probabilities"], original_class, args.target_strategy
        )

        best_attempt = None
        attempted_targets = []
        for target_class in target_candidates:
            print(
                f"Sample {sample_idx}: {classes[original_class]} -> {classes[target_class]}"
            )
            result = generate_sedc_t_counterfactual(
                model=model,
                image_pixels=original_pixels,
                target_class=target_class,
                segments=segments,
                replacement_pixels=replacement_pixels,
                max_segments=args.max_segments,
                allowed_segments=allowed_segments,
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

            if best_attempt is None or result["probabilities"][target_class] > best_attempt[
                "result"
            ]["probabilities"][best_attempt["target_class_index"]]:
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
            "change_metrics": change_metrics,
            "image_path": str(output_path),
            "summary_path": str(output_path.with_suffix(".summary.png")),
        }
        records.append(record)

    metadata = {
        "method": "SEDC-T inspired targeted segment replacement",
        "model_path": args.model_path,
        "dataset_path": args.dataset_path,
        "classes": classes,
        "parameters": {
            "n_segments": args.n_segments,
            "compactness": args.compactness,
            "max_segments": args.max_segments,
            "replacement_mode": args.replacement_mode,
            "blur_kernel": args.blur_kernel,
            "exclude_border_fraction": args.exclude_border_fraction,
            "target_strategy": args.target_strategy,
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
