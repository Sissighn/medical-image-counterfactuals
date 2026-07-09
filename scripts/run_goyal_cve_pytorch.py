"""Run counterfactual visual explanations after Goyal et al. (ICML 2019).

Reference: Goyal, Wu, Ernst, Batra, Parikh & Lee (2019), "Counterfactual
Visual Explanations", ICML 2019, arXiv:1904.07451. A faithful reference
implementation of this method (as baseline) is contained in the official
Meta repository https://github.com/facebookresearch/visual-counterfactuals
(Vandenhende et al., ECCV 2022).

Method (paper Section 3): the CNN is decomposed into a spatial feature
extractor f and a decision head g. For a query image I (predicted class c)
and a distractor image I' from the target class c', the counterfactual
feature map is

    f*(I) = (1 - a) o f(I) + a o (P f(I'))

where a is a binary gate vector over spatial cells and P is a permutation
matrix aligning distractor cells to query cells. The number of edits ||a||_1
is minimized subject to argmax g(f*(I)) = c'. Following the paper's greedy
exhaustive-search variant, each iteration evaluates all remaining
(query cell i, distractor cell j) swaps, commits the swap that maximizes the
softmax probability of c', and stops as soon as the prediction flips to c'.
Each query cell is edited at most once and each distractor cell is used at
most once (permutation constraint).

Distractor selection: the paper defines the method for a given (I, I') pair
with I' taken from the target class. This runner instantiates I' as the
nearest correctly classified training image of the manifest target class in
the classifier's pooled penultimate feature space (cosine distance), i.e. a
nearest-unlike-neighbor distractor.
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
from matplotlib.patches import Rectangle
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


class ResNetSpatialSplit(nn.Module):
    """Decomposes ResNet18 into spatial extractor f and decision head g.

    f: conv stem through layer4, output [B, 512, 7, 7] for 224x224 inputs.
    g: global average pooling followed by the fully connected classifier.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.spatial_extractor = nn.Sequential(*list(model.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = model.fc

    def spatial_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.spatial_extractor(x)

    def head_from_pooled(self, pooled: torch.Tensor) -> torch.Tensor:
        return self.classifier(pooled)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        spatial = self.spatial_features(x)
        pooled = torch.flatten(self.pool(spatial), 1)
        logits = self.head_from_pooled(pooled)
        return logits, pooled, spatial


def load_checkpoint_model(model_path: str, device: torch.device) -> tuple[ResNetSpatialSplit, dict[str, Any]]:
    checkpoint = torch.load(model_path, map_location=device)
    model = create_model(checkpoint["num_classes"])
    model.load_state_dict(checkpoint["model_state_dict"])
    wrapped = ResNetSpatialSplit(model).to(device)
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
    counterfactual_pixels: torch.Tensor,
    threshold: float = 0.03,
) -> dict[str, float]:
    diff = torch.abs(counterfactual_pixels - original_pixels)
    return {
        "l1_mean": float(diff.mean().item()),
        "l2_mean": float(torch.sqrt(torch.mean(diff**2)).item()),
        "linf": float(diff.max().item()),
        "sparsity_threshold": threshold,
        "changed_pixel_fraction": float((diff > threshold).float().mean().item()),
    }


def compute_image_debug_stats(
    original_pixels: torch.Tensor,
    counterfactual_pixels: torch.Tensor,
) -> dict[str, Any]:
    diff = torch.abs(original_pixels - counterfactual_pixels)
    return {
        "value_range": "[0, 1]",
        "difference_formula": "abs(original_image - composite_counterfactual_image)",
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
            "min": float(counterfactual_pixels.min().item()),
            "max": float(counterfactual_pixels.max().item()),
            "mean": float(counterfactual_pixels.mean().item()),
        },
        "diff": {
            "min": float(diff.min().item()),
            "max": float(diff.max().item()),
            "mean": float(diff.mean().item()),
            "std": float(diff.std().item()),
        },
    }


def build_retrieval_database(
    model: ResNetSpatialSplit,
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
            logits, pooled, _ = model(images)
            probabilities = F.softmax(logits, dim=1)
            predictions = torch.argmax(probabilities, dim=1)
            normalized_features = F.normalize(pooled, p=2, dim=1).cpu()

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
                        "feature": normalized_features[batch_idx],
                    }
                )
            offset += images.shape[0]

    return candidates_by_class


def load_manifest_samples(
    model: ResNetSpatialSplit,
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
            logits, pooled, spatial = model(image)
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
                    "feature": F.normalize(pooled, p=2, dim=1)[0].detach().cpu(),
                    "spatial": spatial[0].detach(),
                }
            )

    return manifest, samples


def retrieve_nearest_distractor(
    query_feature: torch.Tensor,
    candidates: list[dict[str, Any]],
) -> tuple[dict[str, Any], float]:
    if not candidates:
        raise ValueError("No distractor candidates available for the requested class.")

    candidate_features = torch.stack([candidate["feature"] for candidate in candidates])
    similarities = torch.matmul(candidate_features, query_feature.cpu())
    distances = 1.0 - similarities
    best_idx = int(torch.argmin(distances).item())
    return candidates[best_idx], float(distances[best_idx].item())


def greedy_exhaustive_search(
    model: ResNetSpatialSplit,
    query_spatial: torch.Tensor,
    distractor_spatial: torch.Tensor,
    target_index: int,
    max_edits: int,
) -> dict[str, Any]:
    """Greedy exhaustive search over spatial cell swaps (paper Section 3.2).

    At each iteration all remaining (query cell, distractor cell) pairs are
    scored and the swap maximizing the target-class softmax probability is
    committed. For ResNet the head g is the fully connected layer applied to
    the spatial mean, so candidate pooled vectors are computed with the exact
    incremental mean update pooled + (f'(j) - f_current(i)) / N, which is
    identical to evaluating g on every edited feature map.
    """
    channels, grid_h, grid_w = query_spatial.shape
    num_cells = grid_h * grid_w
    current = query_spatial.reshape(channels, num_cells).clone()
    distractor_cells = distractor_spatial.reshape(channels, num_cells)
    pooled = current.mean(dim=1)

    query_open = torch.ones(num_cells, dtype=torch.bool, device=current.device)
    distractor_open = torch.ones(num_cells, dtype=torch.bool, device=current.device)
    edits: list[dict[str, Any]] = []

    with torch.no_grad():
        logits = model.head_from_pooled(pooled.unsqueeze(0))[0]
        prediction = int(torch.argmax(logits).item())

        while prediction != target_index and len(edits) < max_edits:
            # pooled_candidates[i, j]: pooled feature after replacing query
            # cell i with distractor cell j.
            delta = (
                distractor_cells.T.unsqueeze(0) - current.T.unsqueeze(1)
            ) / num_cells
            pooled_candidates = pooled.view(1, 1, channels) + delta
            logits_candidates = model.head_from_pooled(
                pooled_candidates.reshape(num_cells * num_cells, channels)
            )
            target_probabilities = F.softmax(logits_candidates, dim=1)[
                :, target_index
            ].reshape(num_cells, num_cells)
            target_probabilities[~query_open, :] = -1.0
            target_probabilities[:, ~distractor_open] = -1.0

            best_flat = int(torch.argmax(target_probabilities).item())
            best_query_cell = best_flat // num_cells
            best_distractor_cell = best_flat % num_cells

            current[:, best_query_cell] = distractor_cells[:, best_distractor_cell]
            pooled = current.mean(dim=1)
            query_open[best_query_cell] = False
            distractor_open[best_distractor_cell] = False

            logits = model.head_from_pooled(pooled.unsqueeze(0))[0]
            probabilities = F.softmax(logits, dim=0)
            prediction = int(torch.argmax(logits).item())

            edits.append(
                {
                    "edit_index": len(edits) + 1,
                    "query_cell": [
                        best_query_cell // grid_w,
                        best_query_cell % grid_w,
                    ],
                    "distractor_cell": [
                        best_distractor_cell // grid_w,
                        best_distractor_cell % grid_w,
                    ],
                    "target_probability_after": float(
                        probabilities[target_index].item()
                    ),
                    "prediction_after": prediction,
                }
            )

        final_probabilities = F.softmax(logits, dim=0)

    return {
        "edits": edits,
        "num_edits": len(edits),
        "grid_size": [grid_h, grid_w],
        "final_prediction": prediction,
        "final_probabilities": final_probabilities.detach().cpu().tolist(),
    }


def build_composite_image(
    original_pixels: torch.Tensor,
    distractor_pixels: torch.Tensor,
    edits: list[dict[str, Any]],
    grid_size: list[int],
) -> torch.Tensor:
    """Paste the image patches aligned with the swapped feature cells.

    The decision-flipping edit happens in feature space; this pixel-space
    composite is the standard visualization used by the paper and the
    reference implementation (each 7x7 feature cell maps to the aligned
    image patch).
    """
    composite = original_pixels.clone()
    grid_h, grid_w = grid_size
    patch_h = original_pixels.shape[-2] // grid_h
    patch_w = original_pixels.shape[-1] // grid_w

    for edit in edits:
        q_row, q_col = edit["query_cell"]
        d_row, d_col = edit["distractor_cell"]
        composite[
            ...,
            q_row * patch_h : (q_row + 1) * patch_h,
            q_col * patch_w : (q_col + 1) * patch_w,
        ] = distractor_pixels[
            ...,
            d_row * patch_h : (d_row + 1) * patch_h,
            d_col * patch_w : (d_col + 1) * patch_w,
        ]

    return composite


def draw_cell_boxes(
    axis: Any,
    cells: list[list[int]],
    grid_size: list[int],
    image_height: int,
    image_width: int,
    color: str,
) -> None:
    grid_h, grid_w = grid_size
    patch_h = image_height // grid_h
    patch_w = image_width // grid_w
    for row, col in cells:
        axis.add_patch(
            Rectangle(
                (col * patch_w, row * patch_h),
                patch_w,
                patch_h,
                linewidth=1.4,
                edgecolor=color,
                facecolor="none",
            )
        )


def save_counterfactual_visualization(
    original_pixels: torch.Tensor,
    distractor_pixels: torch.Tensor,
    composite_pixels: torch.Tensor,
    edits: list[dict[str, Any]],
    grid_size: list[int],
    output_path: Path,
    true_label: str,
    original_prediction: str,
    original_confidence: float,
    target_class: str,
    counterfactual_prediction: str,
    counterfactual_confidence: float,
    distractor_true_label: str,
    embedding_distance: float,
    num_edits: int,
    valid_counterfactual: bool,
) -> None:
    diff = torch.abs(composite_pixels - original_pixels)
    grid = torch.cat(
        [original_pixels, distractor_pixels, composite_pixels, diff.clamp(0.0, 1.0)],
        dim=0,
    )
    utils.save_image(grid, output_path, nrow=4)

    figure_path = output_path.with_suffix(".summary.png")
    fig, axes = plt.subplots(1, 5, figsize=(17, 4.9))

    original_gray = image_to_grayscale(original_pixels[0])
    distractor_gray = image_to_grayscale(distractor_pixels[0])
    composite_gray = image_to_grayscale(composite_pixels[0])
    diff_image = image_to_grayscale(diff[0])
    image_height, image_width = original_gray.shape

    query_cells = [edit["query_cell"] for edit in edits]
    distractor_cells = [edit["distractor_cell"] for edit in edits]

    axes[0].imshow(original_gray, cmap="gray", vmin=0.0, vmax=1.0)
    axes[0].set_title("Original (edited cells)")
    draw_cell_boxes(axes[0], query_cells, grid_size, image_height, image_width, "red")
    axes[0].axis("off")

    axes[1].imshow(distractor_gray, cmap="gray", vmin=0.0, vmax=1.0)
    axes[1].set_title("Distractor (source cells)")
    draw_cell_boxes(
        axes[1], distractor_cells, grid_size, image_height, image_width, "lime"
    )
    axes[1].axis("off")

    axes[2].imshow(composite_gray, cmap="gray", vmin=0.0, vmax=1.0)
    axes[2].set_title("Composite counterfactual")
    axes[2].axis("off")

    diff_plot = axes[3].imshow(diff_image, cmap="gray", vmin=0.0, vmax=1.0)
    axes[3].set_title("Difference")
    axes[3].axis("off")
    fig.colorbar(diff_plot, ax=axes[3], fraction=0.046, pad=0.04)

    axes[4].imshow(original_gray, cmap="gray", vmin=0.0, vmax=1.0)
    overlay_plot = axes[4].imshow(
        diff_image,
        cmap="hot",
        vmin=0.0,
        vmax=1.0,
        alpha=0.4,
    )
    axes[4].set_title("Overlay")
    axes[4].axis("off")
    fig.colorbar(overlay_plot, ax=axes[4], fraction=0.046, pad=0.04)

    valid_text = "yes" if valid_counterfactual else "no"
    title = (
        f"True label: {true_label}\n"
        f"Target: {original_prediction} -> {target_class}\n"
        f"Distractor true label: {distractor_true_label} "
        f"(embedding distance {embedding_distance:.4f})\n"
        f"Prediction: {original_prediction} ({original_confidence:.2f}) -> "
        f"{counterfactual_prediction} ({counterfactual_confidence:.2f})\n"
        f"Feature cell edits: {num_edits} | Valid CF: {valid_text}"
    )
    fig.suptitle(title, y=0.99, fontsize=10.5)
    fig.tight_layout(rect=[0, 0, 1, 0.72])
    fig.savefig(figure_path, dpi=150)
    plt.close(fig)


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
        ("num_edits", "num_edits"),
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
    parser.add_argument(
        "--max_edits",
        type=int,
        default=None,
        help=(
            "Maximum number of cell swaps. Defaults to the full grid (49 for "
            "224x224 ResNet18), at which point the pooled feature equals the "
            "distractor's and the prediction is guaranteed to flip."
        ),
    )
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
    print("Building training-split distractor database...")
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
        distractor, embedding_distance = retrieve_nearest_distractor(
            sample["feature"],
            target_candidates,
        )

        distractor_image, distractor_label = data["train_dataset"][
            distractor["train_dataset_index"]
        ]
        if int(distractor_label) != target_class_index:
            raise ValueError(
                "Distractor image label does not match target class: "
                f"{distractor_label} vs {target_class_index}"
            )

        distractor_batch = distractor_image.unsqueeze(0).to(device)
        with torch.no_grad():
            distractor_logits, _, distractor_spatial = model(distractor_batch)
            distractor_prediction = int(torch.argmax(distractor_logits, dim=1).item())
        if distractor_prediction != target_class_index:
            raise ValueError(
                "Distractor image is no longer classified as the target class: "
                f"{distractor_prediction} vs {target_class_index}"
            )

        grid_cells = sample["spatial"].shape[-2] * sample["spatial"].shape[-1]
        max_edits = args.max_edits if args.max_edits is not None else grid_cells
        search_result = greedy_exhaustive_search(
            model=model,
            query_spatial=sample["spatial"],
            distractor_spatial=distractor_spatial[0],
            target_index=target_class_index,
            max_edits=max_edits,
        )
        runtime_seconds = time.time() - start_time

        original_pixels = denormalize(sample["image"]).detach()
        distractor_pixels = denormalize(distractor_batch).detach()
        composite_pixels = build_composite_image(
            original_pixels,
            distractor_pixels,
            search_result["edits"],
            search_result["grid_size"],
        )

        original_prediction = int(sample["prediction"])
        original_confidence = float(sample["probabilities"][original_prediction])
        counterfactual_prediction = int(search_result["final_prediction"])
        counterfactual_confidence = float(
            search_result["final_probabilities"][counterfactual_prediction]
        )
        valid_counterfactual = counterfactual_prediction == target_class_index

        output_path = output_dir / f"sample_{sample_index:02d}.png"
        save_counterfactual_visualization(
            original_pixels=original_pixels,
            distractor_pixels=distractor_pixels,
            composite_pixels=composite_pixels,
            edits=search_result["edits"],
            grid_size=search_result["grid_size"],
            output_path=output_path,
            true_label=classes[sample["label"]],
            original_prediction=classes[original_prediction],
            original_confidence=original_confidence,
            target_class=classes[target_class_index],
            counterfactual_prediction=classes[counterfactual_prediction],
            counterfactual_confidence=counterfactual_confidence,
            distractor_true_label=distractor["true_label"],
            embedding_distance=embedding_distance,
            num_edits=search_result["num_edits"],
            valid_counterfactual=valid_counterfactual,
        )

        change_metrics = compute_change_metrics(
            original_pixels,
            composite_pixels,
            threshold=args.change_threshold,
        )
        image_debug_stats = compute_image_debug_stats(original_pixels, composite_pixels)

        print(
            f"Sample {sample_index}: {classes[original_prediction]} -> "
            f"{classes[target_class_index]} | edits={search_result['num_edits']} | "
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
                "counterfactual_prediction_index": counterfactual_prediction,
                "counterfactual_prediction": classes[counterfactual_prediction],
                "counterfactual_confidence": counterfactual_confidence,
                "valid_counterfactual": valid_counterfactual,
                "num_edits": search_result["num_edits"],
                "grid_size": search_result["grid_size"],
                "edits": search_result["edits"],
                "distractor_split": "train",
                "distractor_dataset_index": int(distractor["train_dataset_index"]),
                "distractor_image_path": distractor["image_path"],
                "distractor_true_label_index": int(distractor["true_label_index"]),
                "distractor_true_label": distractor["true_label"],
                "distractor_confidence": float(distractor["confidence"]),
                "embedding_distance": embedding_distance,
                "distance_metric": "cosine_distance_on_normalized_pooled_resnet18_embeddings",
                "candidate_pool_size": len(target_candidates),
                "original_probabilities": sample["probabilities"],
                "counterfactual_probabilities": search_result["final_probabilities"],
                "runtime_seconds": runtime_seconds,
                "change_metrics": change_metrics,
                "image_debug_stats": image_debug_stats,
                "image_path": str(output_path),
                "summary_path": str(output_path.with_suffix(".summary.png")),
            }
        )

    metadata = {
        "method": "Goyal et al. 2019 counterfactual visual explanations (greedy exhaustive search)",
        "references": {
            "paper": (
                "Goyal, Wu, Ernst, Batra, Parikh & Lee (2019), Counterfactual "
                "Visual Explanations, ICML 2019, arXiv:1904.07451"
            ),
            "reference_implementation": (
                "https://github.com/facebookresearch/visual-counterfactuals "
                "(official Meta repo for Vandenhende et al. 2022, contains the "
                "Goyal et al. baseline)"
            ),
        },
        "model_path": args.model_path,
        "dataset_path": args.dataset_path,
        "classes": classes,
        "parameters": {
            "feature_layer": "ResNet18 layer4 output (spatial cells before global average pooling)",
            "search": "greedy exhaustive search over (query cell, distractor cell) swaps",
            "selection_criterion": "maximize target-class softmax probability per iteration",
            "constraints": (
                "each query cell edited at most once; each distractor cell used "
                "at most once (permutation constraint)"
            ),
            "stopping": "first iteration where argmax prediction equals the target class",
            "max_edits": args.max_edits,
            "distractor_selection": (
                "nearest correctly classified training image of the manifest "
                "target class, cosine distance on L2-normalized pooled "
                "penultimate features"
            ),
            "distractor_split": "train",
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
            "The decision-flipping edit is performed in the classifier's "
            "spatial feature space (cell swaps from a real distractor image of "
            "the target class). The composite counterfactual image pastes the "
            "aligned image patches and is the standard visualization; pixel "
            "change metrics are computed on this composite."
        ),
    }

    metadata_path = output_dir / "metadata.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
    print(f"Saved metadata to {metadata_path}")


if __name__ == "__main__":
    main()
