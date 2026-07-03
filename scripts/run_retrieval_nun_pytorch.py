"""Run a retrieval-based nearest-unlike-neighbor baseline.

For each fixed evaluation image, this baseline retrieves the nearest real
training image from the manifest target class in ResNet18 penultimate feature
space. It is a case-based counterfactual baseline: it does not edit or generate
the original image.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models, utils

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


def create_model(num_classes: int) -> nn.Module:
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


class ResNetWithFeatures(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.features = nn.Sequential(*list(model.children())[:-1])
        self.classifier = model.fc

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = torch.flatten(self.features(x), 1)
        logits = self.classifier(features)
        return logits, features


def load_checkpoint_model(model_path: str, device: torch.device) -> tuple[nn.Module, dict[str, Any]]:
    checkpoint = torch.load(model_path, map_location=device)
    model = create_model(checkpoint["num_classes"])
    model.load_state_dict(checkpoint["model_state_dict"])
    wrapped = ResNetWithFeatures(model).to(device)
    wrapped.eval()
    return wrapped, checkpoint


def denormalize(images: torch.Tensor) -> torch.Tensor:
    mean = IMAGENET_MEAN.to(images.device)
    std = IMAGENET_STD.to(images.device)
    return (images * std + mean).clamp(0.0, 1.0)


def image_to_grayscale(image: torch.Tensor) -> torch.Tensor:
    return image.mean(dim=0).detach().cpu()


def compute_change_metrics(
    original_pixels: torch.Tensor,
    retrieved_pixels: torch.Tensor,
    threshold: float = 0.03,
) -> dict[str, float]:
    diff = torch.abs(retrieved_pixels - original_pixels)
    return {
        "l1_mean": float(diff.mean().item()),
        "l2_mean": float(torch.sqrt(torch.mean(diff**2)).item()),
        "linf": float(diff.max().item()),
        "sparsity_threshold": threshold,
        "changed_pixel_fraction": float((diff > threshold).float().mean().item()),
    }


def compute_image_debug_stats(
    original_pixels: torch.Tensor,
    retrieved_pixels: torch.Tensor,
) -> dict[str, Any]:
    diff = torch.abs(original_pixels - retrieved_pixels)
    return {
        "value_range": "[0, 1]",
        "difference_formula": "abs(original_image - retrieved_neighbor_image)",
        "normalization_note": (
            "Images are denormalized from ImageNet mean/std before visualization "
            "and difference computation."
        ),
        "original": {
            "min": float(original_pixels.min().item()),
            "max": float(original_pixels.max().item()),
            "mean": float(original_pixels.mean().item()),
        },
        "counterfactual": {
            "min": float(retrieved_pixels.min().item()),
            "max": float(retrieved_pixels.max().item()),
            "mean": float(retrieved_pixels.mean().item()),
        },
        "diff": {
            "min": float(diff.min().item()),
            "max": float(diff.max().item()),
            "mean": float(diff.mean().item()),
            "std": float(diff.std().item()),
        },
    }


def save_retrieval_visualization(
    original_pixels: torch.Tensor,
    retrieved_pixels: torch.Tensor,
    output_path: Path,
    true_label: str,
    original_prediction: str,
    original_confidence: float,
    target_class: str,
    retrieved_prediction: str,
    retrieved_confidence: float,
    retrieved_true_label: str,
    embedding_distance: float,
    valid_counterfactual: bool,
) -> None:
    diff = torch.abs(retrieved_pixels - original_pixels)
    grid = torch.cat([original_pixels, retrieved_pixels, diff.clamp(0.0, 1.0)], dim=0)
    utils.save_image(grid, output_path, nrow=3)

    figure_path = output_path.with_suffix(".summary.png")
    fig, axes = plt.subplots(1, 4, figsize=(13.5, 4.9))

    original_gray = image_to_grayscale(original_pixels[0])
    retrieved_gray = image_to_grayscale(retrieved_pixels[0])
    diff_image = image_to_grayscale(diff[0])

    axes[0].imshow(original_gray, cmap="gray", vmin=0.0, vmax=1.0)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(retrieved_gray, cmap="gray", vmin=0.0, vmax=1.0)
    axes[1].set_title("Retrieved NUN")
    axes[1].axis("off")

    diff_plot = axes[2].imshow(diff_image, cmap="gray", vmin=0.0, vmax=1.0)
    axes[2].set_title("Difference")
    axes[2].axis("off")
    fig.colorbar(diff_plot, ax=axes[2], fraction=0.046, pad=0.04)

    axes[3].imshow(original_gray, cmap="gray", vmin=0.0, vmax=1.0)
    overlay_plot = axes[3].imshow(
        diff_image,
        cmap="hot",
        vmin=0.0,
        vmax=1.0,
        alpha=0.4,
    )
    axes[3].set_title("Overlay")
    axes[3].axis("off")
    fig.colorbar(overlay_plot, ax=axes[3], fraction=0.046, pad=0.04)

    valid_text = "yes" if valid_counterfactual else "no"
    title = (
        f"True label: {true_label}\n"
        f"Target: {original_prediction} -> {target_class}\n"
        f"Retrieved true label: {retrieved_true_label}\n"
        f"Prediction: {original_prediction} ({original_confidence:.2f}) -> "
        f"{retrieved_prediction} ({retrieved_confidence:.2f})\n"
        f"Embedding distance: {embedding_distance:.4f} | Valid CF: {valid_text}"
    )
    fig.suptitle(title, y=0.99, fontsize=10.5)
    fig.tight_layout(rect=[0, 0, 1, 0.72])
    fig.savefig(figure_path, dpi=150)
    plt.close(fig)


def build_retrieval_database(
    model: nn.Module,
    train_dataset: Any,
    classes: list[str],
    device: torch.device,
    batch_size: int,
) -> dict[int, list[dict[str, Any]]]:
    candidates_by_class: dict[int, list[dict[str, Any]]] = defaultdict(list)
    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    offset = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits, features = model(images)
            probabilities = F.softmax(logits, dim=1)
            predictions = torch.argmax(probabilities, dim=1)
            normalized_features = F.normalize(features, p=2, dim=1).cpu()

            for batch_idx in range(images.shape[0]):
                label = int(labels[batch_idx].item())
                prediction = int(predictions[batch_idx].item())
                if prediction != label:
                    continue

                dataset_index = offset + batch_idx
                image_path, dataset_label = train_dataset.samples[dataset_index]
                if int(dataset_label) != label:
                    raise ValueError(
                        "Training dataset index/label mismatch at "
                        f"{dataset_index}: batch={label}, sample={dataset_label}"
                    )

                confidence = float(probabilities[batch_idx, prediction].item())
                candidates_by_class[label].append(
                    {
                        "train_dataset_index": dataset_index,
                        "image_path": str(image_path),
                        "true_label_index": label,
                        "true_label": classes[label],
                        "prediction_index": prediction,
                        "prediction": classes[prediction],
                        "confidence": confidence,
                        "probabilities": probabilities[batch_idx].detach().cpu().tolist(),
                        "feature": normalized_features[batch_idx],
                    }
                )
            offset += images.shape[0]

    return candidates_by_class


def load_manifest_samples(
    model: nn.Module,
    test_dataset: Any,
    device: torch.device,
    manifest_path: str,
    max_records: int | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest, records = load_manifest_records(manifest_path, max_records=max_records)
    samples = []

    with torch.no_grad():
        for record in records:
            image, label, image_path = load_image_from_manifest_record(test_dataset, record)
            image = image.to(device)
            logits, features = model(image)
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
                    "label": int(label),
                    "prediction": prediction,
                    "probabilities": probabilities[0].detach().cpu().tolist(),
                    "target_class_index": int(record["target_class_index"]),
                    "target_class": str(record["target_class"]),
                    "image_source_path": image_path,
                    "feature": F.normalize(features, p=2, dim=1)[0].detach().cpu(),
                }
            )

    return manifest, samples


def retrieve_nearest_candidate(
    query_feature: torch.Tensor,
    candidates: list[dict[str, Any]],
) -> tuple[dict[str, Any], float]:
    if not candidates:
        raise ValueError("No retrieval candidates available for the requested class.")

    candidate_features = torch.stack([candidate["feature"] for candidate in candidates])
    similarities = torch.matmul(candidate_features, query_feature.cpu())
    distances = 1.0 - similarities
    best_idx = int(torch.argmin(distances).item())
    return candidates[best_idx], float(distances[best_idx].item())


def compute_aggregate_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    valid_count = sum(record["valid_counterfactual"] for record in records)
    aggregate: dict[str, Any] = {
        "num_samples": len(records),
        "valid_count": valid_count,
        "validity": valid_count / len(records) if records else 0.0,
    }
    if not records:
        return aggregate

    metric_names = ["l1_mean", "l2_mean", "linf", "changed_pixel_fraction"]
    for metric_name in metric_names:
        values = [record["change_metrics"][metric_name] for record in records]
        aggregate[metric_name] = {
            "mean": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
        }

    scalar_metrics = [
        ("runtime_seconds", "runtime_seconds"),
        ("embedding_distance", "embedding_distance"),
        ("counterfactual_confidence", "counterfactual_confidence"),
    ]
    for output_key, record_key in scalar_metrics:
        values = [float(record[record_key]) for record in records]
        aggregate[output_key] = {
            "mean": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
        }

    aggregate["mean_counterfactual_confidence"] = aggregate["counterfactual_confidence"][
        "mean"
    ]
    return aggregate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--manifest_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--manifest_max_samples", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--change_threshold", type=float, default=0.03)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, checkpoint = load_checkpoint_model(args.model_path, device)
    classes = checkpoint["classes"]

    data = create_dataloaders(
        args.dataset_path,
        batch_size=args.batch_size,
        use_augmentation=False,
    )

    print(f"Device: {device}")
    print(f"Classes: {classes}")
    print("Building training-split retrieval database...")
    candidates_by_class = build_retrieval_database(
        model=model,
        train_dataset=data["train_dataset"],
        classes=classes,
        device=device,
        batch_size=args.batch_size,
    )
    candidate_counts = {
        classes[class_idx]: len(candidates_by_class.get(class_idx, []))
        for class_idx in range(len(classes))
    }
    print(f"Correctly classified training candidates: {candidate_counts}")

    manifest, samples = load_manifest_samples(
        model=model,
        test_dataset=data["test_dataset"],
        device=device,
        manifest_path=args.manifest_path,
        max_records=args.manifest_max_samples,
    )
    if not samples:
        raise RuntimeError("No manifest samples found.")

    records = []
    for sample_index, sample in enumerate(samples):
        start_time = time.time()
        target_class_index = sample["target_class_index"]
        target_candidates = candidates_by_class.get(target_class_index, [])
        retrieved, embedding_distance = retrieve_nearest_candidate(
            sample["feature"],
            target_candidates,
        )

        retrieved_image, retrieved_label = data["train_dataset"][
            retrieved["train_dataset_index"]
        ]
        if int(retrieved_label) != target_class_index:
            raise ValueError(
                "Retrieved image label does not match target class: "
                f"{retrieved_label} vs {target_class_index}"
            )

        original_pixels = denormalize(sample["image"]).detach()
        retrieved_pixels = denormalize(
            retrieved_image.unsqueeze(0).to(device)
        ).detach()

        runtime_seconds = time.time() - start_time
        original_prediction = int(sample["prediction"])
        original_confidence = float(sample["probabilities"][original_prediction])
        retrieved_prediction = int(retrieved["prediction_index"])
        retrieved_confidence = float(retrieved["confidence"])
        valid_counterfactual = retrieved_prediction == target_class_index

        output_path = output_dir / f"sample_{sample_index:02d}.png"
        save_retrieval_visualization(
            original_pixels=original_pixels,
            retrieved_pixels=retrieved_pixels,
            output_path=output_path,
            true_label=classes[sample["label"]],
            original_prediction=classes[original_prediction],
            original_confidence=original_confidence,
            target_class=classes[target_class_index],
            retrieved_prediction=classes[retrieved_prediction],
            retrieved_confidence=retrieved_confidence,
            retrieved_true_label=retrieved["true_label"],
            embedding_distance=embedding_distance,
            valid_counterfactual=valid_counterfactual,
        )

        change_metrics = compute_change_metrics(
            original_pixels,
            retrieved_pixels,
            threshold=args.change_threshold,
        )
        image_debug_stats = compute_image_debug_stats(original_pixels, retrieved_pixels)

        print(
            f"Sample {sample_index}: {classes[original_prediction]} -> "
            f"{classes[target_class_index]} | distance={embedding_distance:.4f} | "
            f"valid={valid_counterfactual}"
        )

        records.append(
            {
                "sample_index": sample_index,
                **{
                    key: sample[key]
                    for key in [
                        "manifest_sample_index",
                        "dataset_index",
                        "source_image_path",
                    ]
                    if key in sample
                },
                "true_label_index": int(sample["label"]),
                "true_label": classes[sample["label"]],
                "original_prediction_index": original_prediction,
                "original_prediction": classes[original_prediction],
                "original_confidence": original_confidence,
                "target_class_index": target_class_index,
                "target_class": classes[target_class_index],
                "counterfactual_prediction_index": retrieved_prediction,
                "counterfactual_prediction": classes[retrieved_prediction],
                "counterfactual_confidence": retrieved_confidence,
                "valid_counterfactual": valid_counterfactual,
                "retrieved_split": "train",
                "retrieved_dataset_index": int(retrieved["train_dataset_index"]),
                "retrieved_image_path": retrieved["image_path"],
                "retrieved_true_label_index": int(retrieved["true_label_index"]),
                "retrieved_true_label": retrieved["true_label"],
                "embedding_distance": embedding_distance,
                "distance_metric": "cosine_distance_on_normalized_penultimate_resnet18_embeddings",
                "candidate_pool_size": len(target_candidates),
                "original_probabilities": sample["probabilities"],
                "counterfactual_probabilities": retrieved["probabilities"],
                "runtime_seconds": runtime_seconds,
                "change_metrics": change_metrics,
                "image_debug_stats": image_debug_stats,
                "image_path": str(output_path),
                "summary_path": str(output_path.with_suffix(".summary.png")),
            }
        )

    metadata = {
        "method": "Retrieval-based nearest-unlike-neighbor baseline",
        "model_path": args.model_path,
        "dataset_path": args.dataset_path,
        "classes": classes,
        "parameters": {
            "retrieval_split": "train",
            "candidate_filter": (
                "candidate true label equals target class and model prediction "
                "equals target class"
            ),
            "distance_metric": "cosine distance",
            "embedding": "ResNet18 penultimate layer, L2-normalized",
            "batch_size": args.batch_size,
            "change_threshold": args.change_threshold,
            "manifest_path": args.manifest_path,
            "manifest_max_samples": args.manifest_max_samples,
        },
        "evaluation_manifest": {
            "path": args.manifest_path,
            "purpose": manifest.get("purpose"),
            "num_manifest_records": len(manifest.get("records", [])),
            "num_used_records": len(samples),
            "target_strategy": manifest.get("target_strategy"),
        },
        "candidate_database": {
            "split": "train",
            "correctly_classified_candidates_by_class": candidate_counts,
            "total_correctly_classified_candidates": sum(candidate_counts.values()),
        },
        "records": records,
        "aggregate_metrics": compute_aggregate_metrics(records),
        "interpretation_note": (
            "This method retrieves a real training image from the target class. "
            "It is interpretable as a nearest unlike case, but it is not a "
            "minimal edit of the original image."
        ),
    }

    metadata_path = output_dir / "metadata.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
    print(f"Saved metadata to {metadata_path}")


if __name__ == "__main__":
    main()
