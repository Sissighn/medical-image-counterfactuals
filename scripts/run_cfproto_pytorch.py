import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, utils

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.autoencoder import ARCHITECTURE_NAME, ConvAutoencoder
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


class ResNetWithFeatures(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.features = nn.Sequential(*list(model.children())[:-1])
        self.classifier = model.fc

    def forward(self, x):
        features = torch.flatten(self.features(x), 1)
        logits = self.classifier(features)
        return logits, features


def denormalize(images):
    mean = IMAGENET_MEAN.to(images.device)
    std = IMAGENET_STD.to(images.device)
    return (images * std + mean).clamp(0.0, 1.0)


def normalize(images):
    mean = IMAGENET_MEAN.to(images.device)
    std = IMAGENET_STD.to(images.device)
    return (images - mean) / std


def total_variation(images):
    vertical = torch.mean(torch.abs(images[:, :, 1:, :] - images[:, :, :-1, :]))
    horizontal = torch.mean(torch.abs(images[:, :, :, 1:] - images[:, :, :, :-1]))
    return vertical + horizontal


def load_checkpoint_model(model_path, device):
    checkpoint = torch.load(model_path, map_location=device)
    model = create_model(checkpoint["num_classes"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model = ResNetWithFeatures(model).to(device)
    model.eval()
    return model, checkpoint


def load_autoencoder_model(autoencoder_path, device):
    checkpoint = torch.load(autoencoder_path, map_location=device)
    architecture = checkpoint.get("architecture")
    if architecture != ARCHITECTURE_NAME:
        raise ValueError(
            f"Unsupported autoencoder architecture: {architecture}. "
            f"Expected {ARCHITECTURE_NAME}."
        )

    autoencoder = ConvAutoencoder(
        input_channels=checkpoint.get("input_channels", 3),
        base_channels=checkpoint.get("base_channels", 32),
    ).to(device)
    autoencoder.load_state_dict(checkpoint["model_state_dict"])
    autoencoder.eval()
    for parameter in autoencoder.parameters():
        parameter.requires_grad_(False)
    return autoencoder, checkpoint


def compute_feature_prototypes(model, train_loader, num_classes, device):
    feature_sums = None
    class_counts = torch.zeros(num_classes, device=device)

    with torch.no_grad():
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            _, features = model(images)

            if feature_sums is None:
                feature_sums = torch.zeros(num_classes, features.shape[1], device=device)

            for class_idx in range(num_classes):
                mask = labels == class_idx
                if mask.any():
                    feature_sums[class_idx] += features[mask].sum(dim=0)
                    class_counts[class_idx] += mask.sum()

    return feature_sums / class_counts.unsqueeze(1).clamp_min(1.0)


def compute_autoencoder_latent_prototypes(autoencoder, train_loader, num_classes, device):
    autoencoder.eval()
    latent_sums = None
    class_counts = torch.zeros(num_classes, device=device)

    with torch.no_grad():
        for images, labels in train_loader:
            pixels = denormalize(images.to(device))
            labels = labels.to(device)
            latents = torch.flatten(autoencoder.encode(pixels), start_dim=1)

            if latent_sums is None:
                latent_sums = torch.zeros(num_classes, latents.shape[1], device=device)

            for class_idx in range(num_classes):
                mask = labels == class_idx
                if mask.any():
                    latent_sums[class_idx] += latents[mask].sum(dim=0)
                    class_counts[class_idx] += mask.sum()

    missing_classes = [
        class_idx for class_idx, count in enumerate(class_counts.tolist()) if count == 0
    ]
    if missing_classes:
        raise ValueError(
            "Cannot compute autoencoder latent prototypes because no training "
            f"examples were found for class indices: {missing_classes}"
        )

    return latent_sums / class_counts.unsqueeze(1)


def compute_autoencoder_latent_bank(autoencoder, train_loader, device):
    autoencoder.eval()
    latent_batches = []
    label_batches = []

    with torch.no_grad():
        for images, labels in train_loader:
            pixels = denormalize(images.to(device))
            latents = torch.flatten(autoencoder.encode(pixels), start_dim=1)
            latent_batches.append(latents.detach())
            label_batches.append(labels.to(device).detach())

    if not latent_batches:
        raise ValueError("Cannot build encoder prototype bank from an empty train loader.")

    return torch.cat(latent_batches, dim=0), torch.cat(label_batches, dim=0)


def compute_class_mean_prototypes_from_bank(latent_bank, label_bank, num_classes):
    prototypes = []
    missing_classes = []
    for class_idx in range(num_classes):
        mask = label_bank == class_idx
        if not mask.any():
            missing_classes.append(class_idx)
            prototypes.append(torch.zeros(latent_bank.shape[1], device=latent_bank.device))
        else:
            prototypes.append(latent_bank[mask].mean(dim=0))

    if missing_classes:
        raise ValueError(
            "Cannot compute encoder class-mean prototypes because no training "
            f"examples were found for class indices: {missing_classes}"
        )

    return torch.stack(prototypes, dim=0)


def select_encoder_knn_prototype(
    autoencoder,
    original_pixels,
    latent_bank,
    label_bank,
    target_class,
    prototype_mode,
    prototype_k,
):
    if prototype_mode not in {"knn_mean", "knn_point"}:
        raise ValueError(f"Unsupported KNN prototype_mode: {prototype_mode}")

    with torch.no_grad():
        original_latent = torch.flatten(
            autoencoder.encode(original_pixels), start_dim=1
        )[0]
        target_mask = label_bank == target_class
        target_latents = latent_bank[target_mask]
        if target_latents.numel() == 0:
            raise ValueError(
                "Cannot compute KNN prototype because no training examples were "
                f"found for target class index {target_class}."
            )

        distances = torch.linalg.vector_norm(
            target_latents - original_latent.unsqueeze(0), dim=1
        )
        k = min(prototype_k, target_latents.shape[0])
        nearest_distances, nearest_indices = torch.topk(
            distances, k=k, largest=False
        )
        nearest_latents = target_latents[nearest_indices]
        if prototype_mode == "knn_point":
            prototype = nearest_latents[0]
        else:
            prototype = nearest_latents.mean(dim=0)

    return prototype, {
        "prototype_mode": prototype_mode,
        "prototype_k_requested": prototype_k,
        "prototype_k_used": int(k),
        "knn_distances": nearest_distances.detach().cpu().tolist(),
    }


def cw_attack_loss(probabilities, target_class, kappa):
    target_probability = probabilities[:, target_class]
    non_target_mask = torch.ones_like(probabilities, dtype=torch.bool)
    non_target_mask[:, target_class] = False
    non_target_probability = probabilities.masked_fill(
        ~non_target_mask, -1e9
    ).max(dim=1).values
    return torch.clamp(
        non_target_probability - target_probability + kappa,
        min=0.0,
    ).mean()


def compute_attack_loss(logits, probabilities, target, target_class, attack_loss, kappa):
    if attack_loss == "cross_entropy":
        return F.cross_entropy(logits, target)
    if attack_loss == "cw_hinge":
        return cw_attack_loss(probabilities, target_class, kappa)
    raise ValueError(f"Unsupported attack loss: {attack_loss}")


def choose_samples(model, test_loader, device, max_samples):
    samples = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            logits, features = model(images)
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
                            "features": features[idx].detach().cpu(),
                        }
                    )
                    if len(samples) >= max_samples:
                        return samples

    return samples


def choose_manifest_samples(model, test_dataset, device, manifest_path, max_records=None):
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
                    "label": label,
                    "prediction": prediction,
                    "probabilities": probabilities[0].detach().cpu().tolist(),
                    "target_class_index": int(record["target_class_index"]),
                    "target_class": record["target_class"],
                    "image_source_path": image_path,
                    "features": features[0].detach().cpu(),
                }
            )

    return manifest, samples


def select_target_classes(probabilities, original_class, strategy):
    ranked = torch.argsort(probabilities, descending=True).tolist()
    candidates = [class_idx for class_idx in ranked if class_idx != original_class]

    if strategy == "all":
        return candidates
    if not candidates:
        return [int((original_class + 1) % probabilities.shape[0])]
    return [candidates[0]]


def select_target_by_prototype_distance(features, prototypes, original_class):
    distances = torch.linalg.vector_norm(
        prototypes.detach().cpu() - features.detach().cpu(),
        dim=1,
    )
    distances[original_class] = float("inf")
    return int(torch.argmin(distances).item())


def generate_counterfactual(
    model,
    image,
    original_class,
    target_class,
    target_prototype,
    prototype_space,
    steps,
    learning_rate,
    attack_loss,
    attack_const,
    kappa,
    lambda_l1,
    lambda_l2,
    lambda_tv,
    lambda_proto,
    max_delta,
    force_grayscale,
    perturbation_resolution,
    autoencoder=None,
    gamma=0.0,
    selection_metric="current",
    beta=0.1,
    lr_schedule="constant",
):
    original_pixels = denormalize(image).detach()
    channels = 1 if force_grayscale else 3
    perturbation = torch.zeros(
        1,
        channels,
        perturbation_resolution,
        perturbation_resolution,
        device=image.device,
        requires_grad=True,
    )
    optimizer = torch.optim.Adam([perturbation], lr=learning_rate)
    target = torch.tensor([target_class], device=image.device)

    best = None
    last_loss_terms = None
    start_time = time.time()

    for step in range(steps):
        if lr_schedule == "polynomial":
            lr_multiplier = max(0.0, 1.0 - (step / max(steps, 1))) ** 0.5
            for group in optimizer.param_groups:
                group["lr"] = learning_rate * lr_multiplier
        elif lr_schedule != "constant":
            raise ValueError(f"Unsupported lr_schedule: {lr_schedule}")

        optimizer.zero_grad()

        smooth_delta = F.interpolate(
            torch.tanh(perturbation),
            size=original_pixels.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        if force_grayscale:
            smooth_delta = smooth_delta.repeat(1, 3, 1, 1)

        cf_pixels = (original_pixels + max_delta * smooth_delta).clamp(0.0, 1.0)
        cf_normalized = normalize(cf_pixels)
        logits, features = model(cf_normalized)
        probabilities = F.softmax(logits, dim=1)

        class_loss = compute_attack_loss(
            logits=logits,
            probabilities=probabilities,
            target=target,
            target_class=target_class,
            attack_loss=attack_loss,
            kappa=kappa,
        )
        l1_loss = torch.mean(torch.abs(cf_pixels - original_pixels))
        l2_loss = F.mse_loss(cf_pixels, original_pixels)
        tv_loss = total_variation(smooth_delta)
        if prototype_space == "encoder":
            if autoencoder is None:
                raise ValueError(
                    "prototype_space='encoder' requires an autoencoder checkpoint."
                )
            latent = torch.flatten(autoencoder.encode(cf_pixels), start_dim=1)
            proto_loss = F.mse_loss(latent[0], target_prototype)
        elif prototype_space == "resnet":
            proto_loss = F.mse_loss(features[0], target_prototype)
        else:
            raise ValueError(f"Unsupported prototype_space: {prototype_space}")
        ae_loss = torch.tensor(0.0, device=image.device)
        if autoencoder is not None:
            ae_loss = F.mse_loss(autoencoder(cf_pixels), cf_pixels)
        loss_terms = {
            "class_loss": float(class_loss.detach().item()),
            "l1_loss": float(l1_loss.detach().item()),
            "l2_loss": float(l2_loss.detach().item()),
            "tv_loss": float(tv_loss.detach().item()),
            "proto_loss": float(proto_loss.detach().item()),
            "ae_loss": float(ae_loss.detach().item()),
        }
        last_loss_terms = loss_terms

        loss = (
            attack_const * class_loss
            + lambda_l1 * l1_loss
            + lambda_l2 * l2_loss
            + lambda_tv * tv_loss
            + lambda_proto * proto_loss
            + gamma * ae_loss
        )
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            prediction = int(torch.argmax(probabilities, dim=1).item())
            target_probability = float(probabilities[0, target_class].item())

            if prediction == target_class:
                selection_distances = compute_elastic_net_distance(
                    original_pixels, cf_pixels, beta
                )
                if selection_metric == "elastic_net":
                    score = selection_distances["elastic_net_distance"]
                elif selection_metric == "current":
                    score = float(loss.item())
                else:
                    raise ValueError(f"Unsupported selection_metric: {selection_metric}")
                if best is None or score < best["score"]:
                    best = {
                        "image": cf_pixels.detach().clone(),
                        "step": step + 1,
                        "score": score,
                        "selection_metric": selection_metric,
                        "selection_distances": selection_distances,
                        "target_probability": target_probability,
                        "attack_const": attack_const,
                        "autoencoder_loss": float(ae_loss.item()),
                        "loss_terms": loss_terms,
                    }

    runtime = time.time() - start_time

    with torch.no_grad():
        if best is not None:
            final_pixels = best["image"]
        else:
            smooth_delta = F.interpolate(
                torch.tanh(perturbation),
                size=original_pixels.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            if force_grayscale:
                smooth_delta = smooth_delta.repeat(1, 3, 1, 1)
            final_pixels = (original_pixels + max_delta * smooth_delta).clamp(0.0, 1.0)

        final_normalized = normalize(final_pixels)
        final_logits, _ = model(final_normalized)
        final_probabilities = F.softmax(final_logits, dim=1)
        final_prediction = int(torch.argmax(final_probabilities, dim=1).item())

    return {
        "image": final_pixels,
        "valid": final_prediction == target_class,
        "prediction": final_prediction,
        "probabilities": final_probabilities[0].detach().cpu().tolist(),
        "runtime_seconds": runtime,
        "best_step": best["step"] if best is not None else None,
        "attack_const": best["attack_const"] if best is not None else attack_const,
        "autoencoder_loss": best["autoencoder_loss"] if best is not None else None,
        "loss_terms": best["loss_terms"] if best is not None else last_loss_terms,
        "selection_metric": best["selection_metric"] if best is not None else selection_metric,
        "selection_distances": (
            best["selection_distances"]
            if best is not None
            else compute_elastic_net_distance(original_pixels, final_pixels, beta)
        ),
    }


def image_to_grayscale(image):
    return image.mean(dim=0).detach().cpu()


def save_counterfactual_visualization(
    original_pixels,
    cf_pixels,
    output_path,
    true_label,
    original_prediction,
    original_confidence,
    target_class,
    counterfactual_prediction,
    counterfactual_confidence,
    valid_counterfactual,
):
    diff = torch.abs(cf_pixels - original_pixels)
    grid = torch.cat([original_pixels, cf_pixels, diff.clamp(0.0, 1.0)], dim=0)
    utils.save_image(grid, output_path, nrow=3)

    figure_path = output_path.with_suffix(".summary.png")
    fig, axes = plt.subplots(1, 4, figsize=(13, 4.8))

    original_gray = image_to_grayscale(original_pixels[0])
    cf_gray = image_to_grayscale(cf_pixels[0])
    diff_image = image_to_grayscale(diff[0])

    labels = ["Original", "Counterfactual"]
    for axis, image, label in zip(axes[:2], [original_gray, cf_gray], labels):
        axis.imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
        axis.set_title(label)
        axis.axis("off")

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
        f"Prediction: {original_prediction} ({original_confidence:.2f}) -> "
        f"{counterfactual_prediction} ({counterfactual_confidence:.2f})\n"
        f"Valid CF: {valid_text}"
    )
    fig.suptitle(title, y=0.98, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.78])
    fig.savefig(figure_path, dpi=150)
    plt.close(fig)


def compute_image_debug_stats(original_pixels, cf_pixels):
    diff = torch.abs(original_pixels - cf_pixels)
    return {
        "value_range": "[0, 1]",
        "difference_formula": "abs(original_image - counterfactual_image)",
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
            "min": float(cf_pixels.min().item()),
            "max": float(cf_pixels.max().item()),
            "mean": float(cf_pixels.mean().item()),
        },
        "diff": {
            "min": float(diff.min().item()),
            "max": float(diff.max().item()),
            "mean": float(diff.mean().item()),
            "std": float(diff.std().item()),
        },
    }


def compute_change_metrics(
    original_pixels,
    cf_pixels,
    threshold=0.03,
    thresholds=(0.03, 0.01, 0.005, 0.001),
):
    diff = torch.abs(cf_pixels - original_pixels)
    metrics = {
        "l1_mean": float(diff.mean().item()),
        "l2_mean": float(torch.sqrt(torch.mean(diff**2)).item()),
        "linf": float(diff.max().item()),
        "sparsity_threshold": threshold,
        "changed_pixel_fraction": float((diff > threshold).float().mean().item()),
    }
    for value in thresholds:
        suffix = str(value).replace(".", "_")
        metrics[f"changed_pixel_fraction_threshold_{suffix}"] = float(
            (diff > value).float().mean().item()
        )
    return metrics


def compute_elastic_net_distance(original_pixels, cf_pixels, beta):
    diff = cf_pixels - original_pixels
    l2_distance = float(torch.sum(diff**2).item())
    l1_distance = float(torch.sum(torch.abs(diff)).item())
    return {
        "l1_distance_sum": l1_distance,
        "l2_distance_sum": l2_distance,
        "beta": beta,
        "elastic_net_distance": l2_distance + beta * l1_distance,
    }


def get_attack_consts(attack_const, c_init, c_steps):
    """Return fixed geometric attack-constant candidates around c_init.

    This is a lightweight CFProto-inspired c-search over a symmetric geometric
    grid, not the adaptive binary search used by Alibi CFProto. For c_steps <= 1
    the previous behavior is kept exactly: only attack_const is used.
    With an even number of steps, the grid keeps exactly c_steps candidates and
    includes c_init, so one side necessarily receives one extra candidate.
    """
    if c_steps <= 1:
        return [attack_const]

    lower_steps = c_steps // 2
    upper_steps = c_steps - lower_steps
    return [c_init * (2**step) for step in range(-lower_steps, upper_steps)]


def score_attempt_for_selection(result, original_pixels, selection_metric, beta):
    diff = torch.abs(result["image"] - original_pixels)
    if selection_metric == "elastic_net":
        distances = compute_elastic_net_distance(original_pixels, result["image"], beta)
        return (
            not result["valid"],
            distances["elastic_net_distance"],
            -float(result["probabilities"][result["prediction"]]),
        )
    if selection_metric != "current":
        raise ValueError(f"Unsupported selection_metric: {selection_metric}")
    return (
        not result["valid"],
        float(diff.mean().item()),
        -float(result["probabilities"][result["prediction"]]),
    )


def compute_aggregate_metrics(records):
    valid_count = sum(record["valid_counterfactual"] for record in records)
    aggregate = {
        "num_samples": len(records),
        "valid_count": valid_count,
        "validity": valid_count / len(records) if records else 0.0,
    }

    if not records:
        return aggregate

    metric_names = [
        "l1_mean",
        "l2_mean",
        "linf",
        "changed_pixel_fraction",
        "changed_pixel_fraction_threshold_0_03",
        "changed_pixel_fraction_threshold_0_01",
        "changed_pixel_fraction_threshold_0_005",
        "changed_pixel_fraction_threshold_0_001",
    ]
    for metric_name in metric_names:
        values = [
            record["change_metrics"][metric_name]
            for record in records
            if metric_name in record["change_metrics"]
        ]
        if not values:
            continue
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
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument(
        "--attack_loss",
        choices=["cross_entropy", "cw_hinge"],
        default="cross_entropy",
        help=(
            "cross_entropy keeps the earlier project baseline. cw_hinge uses a "
            "targeted Carlini-Wagner-style margin loss closer to Alibi CFProto."
        ),
    )
    parser.add_argument(
        "--attack_const",
        type=float,
        default=1.0,
        help="Weight for the attack loss. Comparable to CFProto's c constant.",
    )
    parser.add_argument(
        "--c_init",
        type=float,
        default=1.0,
        help=(
            "Center value for the optional symmetric geometric c-search. "
            "Used only when c_steps > 1."
        ),
    )
    parser.add_argument(
        "--c_steps",
        type=int,
        default=1,
        help=(
            "Number of attack-constant candidates in a fixed geometric grid "
            "around c_init. This is a lightweight CFProto-inspired search, "
            "not the full Alibi binary search."
        ),
    )
    parser.add_argument(
        "--c_search_mode",
        choices=["fixed_grid", "adaptive_binary"],
        default="fixed_grid",
        help=(
            "Attack-constant search mode. fixed_grid preserves the earlier "
            "geometric candidates. adaptive_binary updates c according to "
            "whether the previous fixed-sample/fixed-target attempt was valid."
        ),
    )
    parser.add_argument(
        "--kappa",
        type=float,
        default=0.0,
        help="Target margin for cw_hinge attack loss.",
    )
    parser.add_argument(
        "--lambda_l1",
        type=float,
        default=0.0,
        help="Optional L1 image-distance penalty. This is not FISTA shrinkage.",
    )
    parser.add_argument("--lambda_l2", type=float, default=5.0)
    parser.add_argument("--lambda_tv", type=float, default=0.2)
    parser.add_argument("--lambda_proto", type=float, default=0.05)
    parser.add_argument(
        "--autoencoder_path",
        type=str,
        default=None,
        help=(
            "Optional ConvAutoencoder checkpoint. Together with gamma this enables "
            "a CFProto-like autoencoder plausibility term. Without this path, the "
            "script keeps the previous behavior and skips the autoencoder forward pass."
        ),
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.0,
        help=(
            "Weight for the optional autoencoder reconstruction loss "
            "MSE(AE(counterfactual), counterfactual). Active only when "
            "autoencoder_path is set."
        ),
    )
    parser.add_argument(
        "--prototype_space",
        choices=["resnet", "encoder"],
        default="resnet",
        help=(
            "Prototype representation space. 'encoder' uses frozen "
            "ConvAutoencoder encoder prototypes and is the CFProto-nearer "
            "configuration. 'resnet' keeps the legacy ResNet18 penultimate "
            "feature prototypes for backward compatibility."
        ),
    )
    parser.add_argument(
        "--prototype_mode",
        choices=["class_mean", "knn_mean", "knn_point"],
        default="class_mean",
        help=(
            "Prototype construction mode. class_mean uses one class prototype. "
            "knn_mean and knn_point use target-class nearest neighbours in the "
            "autoencoder encoder space and require prototype_space=encoder."
        ),
    )
    parser.add_argument(
        "--prototype_k",
        type=int,
        default=5,
        help="Number of target-class neighbours for knn_mean or knn_point.",
    )
    parser.add_argument(
        "--selection_metric",
        choices=["current", "elastic_net"],
        default="current",
        help=(
            "Best-counterfactual selection metric. current preserves the earlier "
            "loss-based choice. elastic_net chooses the smallest L2 + beta * L1 "
            "distance among valid attempts."
        ),
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.1,
        help="L1 weight for elastic-net best-counterfactual selection.",
    )
    parser.add_argument(
        "--lr_schedule",
        choices=["constant", "polynomial"],
        default="constant",
        help="Learning-rate schedule. polynomial applies a CFProto-like decay.",
    )
    parser.add_argument("--max_delta", type=float, default=0.12)
    parser.add_argument("--perturbation_resolution", type=int, default=28)
    parser.add_argument(
        "--target_strategy",
        choices=["second_best", "all", "prototype_distance"],
        default="all",
        help=(
            "Target selection for non-manifest runs. Manifest mode keeps the "
            "manifest target for fair fixed evaluation."
        ),
    )
    parser.add_argument("--allow_color_changes", action="store_true")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument(
        "--log_loss_terms",
        action="store_true",
        help=(
            "Print unweighted raw loss terms for the selected result per sample. "
            "Useful for calibrating lambda/gamma magnitudes."
        ),
    )
    args = parser.parse_args()

    if args.prototype_space == "encoder" and not args.autoencoder_path:
        parser.error("--prototype_space encoder requires --autoencoder_path.")
    if args.prototype_mode in {"knn_mean", "knn_point"} and args.prototype_space != "encoder":
        parser.error("knn_mean and knn_point require --prototype_space encoder.")
    if args.prototype_k < 1:
        parser.error("--prototype_k must be at least 1.")

    device = get_device()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, checkpoint = load_checkpoint_model(args.model_path, device)
    classes = checkpoint["classes"]
    num_classes = checkpoint["num_classes"]
    autoencoder = None
    autoencoder_checkpoint = None
    if args.autoencoder_path:
        print(f"Loading autoencoder checkpoint: {args.autoencoder_path}")
        autoencoder, autoencoder_checkpoint = load_autoencoder_model(
            args.autoencoder_path,
            device,
        )

    data = create_dataloaders(
        args.dataset_path, batch_size=args.batch_size, use_augmentation=False
    )

    print(f"Device: {device}")
    print(f"Classes: {classes}")
    latent_bank = None
    label_bank = None
    if args.prototype_space == "encoder":
        print("Computing autoencoder encoder-space prototype bank from training split...")
        latent_bank, label_bank = compute_autoencoder_latent_bank(
            autoencoder, data["train_loader"], device
        )
        if args.prototype_mode == "class_mean":
            prototypes = compute_class_mean_prototypes_from_bank(
                latent_bank, label_bank, num_classes
            )
        else:
            prototypes = None
        prototype_source = "autoencoder_encoder"
    else:
        print("Computing legacy ResNet18 feature prototypes from training split...")
        prototypes = compute_feature_prototypes(
            model, data["train_loader"], num_classes, device
        )
        prototype_source = "resnet18_penultimate_features"

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

    records = []
    for sample_idx, sample in enumerate(samples):
        image = sample["image"]
        original_class = sample["prediction"]
        original_probabilities = torch.tensor(sample["probabilities"])
        original_pixels = denormalize(image).detach()
        if "target_class_index" in sample:
            target_candidates = [sample["target_class_index"]]
        elif args.target_strategy == "prototype_distance":
            if args.prototype_space == "encoder":
                with torch.no_grad():
                    original_features = torch.flatten(
                        autoencoder.encode(original_pixels), start_dim=1
                    )[0].detach().cpu()
            else:
                original_features = sample["features"]
            target_candidates = [
                select_target_by_prototype_distance(
                    original_features, prototypes, original_class
                )
            ]
        else:
            target_candidates = select_target_classes(
                original_probabilities, original_class, args.target_strategy
            )
        attempted_targets = []
        all_c_history = []
        best_attempt = None
        attack_consts = get_attack_consts(
            attack_const=args.attack_const,
            c_init=args.c_init,
            c_steps=args.c_steps,
        )

        for target_class in target_candidates:
            print(
                f"Sample {sample_idx}: {classes[original_class]} -> {classes[target_class]}"
            )
            if args.prototype_mode == "class_mean":
                target_prototype = prototypes[target_class]
                prototype_details = {
                    "prototype_mode": args.prototype_mode,
                    "local_prototype_used": False,
                    "knn_distances": None,
                }
            else:
                target_prototype, prototype_details = select_encoder_knn_prototype(
                    autoencoder=autoencoder,
                    original_pixels=original_pixels,
                    latent_bank=latent_bank,
                    label_bank=label_bank,
                    target_class=target_class,
                    prototype_mode=args.prototype_mode,
                    prototype_k=args.prototype_k,
                )
                prototype_details["local_prototype_used"] = True

            c_history = []
            if args.c_search_mode == "fixed_grid":
                attack_const_trials = [
                    {
                        "c_step": c_step,
                        "attack_const": attack_const,
                        "lower_bound": None,
                        "upper_bound": None,
                    }
                    for c_step, attack_const in enumerate(attack_consts)
                ]
            else:
                current_c = args.c_init
                lower_bound = None
                upper_bound = None
                attack_const_trials = [None] * max(args.c_steps, 1)

            for trial_index, trial in enumerate(attack_const_trials):
                if args.c_search_mode == "adaptive_binary":
                    trial = {
                        "c_step": trial_index,
                        "attack_const": current_c,
                        "lower_bound": lower_bound,
                        "upper_bound": upper_bound,
                    }
                attack_const = trial["attack_const"]
                result = generate_counterfactual(
                    model=model,
                    image=image,
                    original_class=original_class,
                    target_class=target_class,
                    target_prototype=target_prototype,
                    prototype_space=args.prototype_space,
                    steps=args.steps,
                    learning_rate=args.learning_rate,
                    attack_loss=args.attack_loss,
                    attack_const=attack_const,
                    kappa=args.kappa,
                    lambda_l1=args.lambda_l1,
                    lambda_l2=args.lambda_l2,
                    lambda_tv=args.lambda_tv,
                    lambda_proto=args.lambda_proto,
                    max_delta=args.max_delta,
                    force_grayscale=not args.allow_color_changes,
                    perturbation_resolution=args.perturbation_resolution,
                    autoencoder=autoencoder,
                    gamma=args.gamma,
                    selection_metric=args.selection_metric,
                    beta=args.beta,
                    lr_schedule=args.lr_schedule,
                )
                if args.c_search_mode == "adaptive_binary":
                    if result["valid"]:
                        upper_bound = attack_const
                        if lower_bound is None:
                            current_c = max(attack_const / 2.0, 1e-12)
                        else:
                            current_c = (lower_bound + upper_bound) / 2.0
                    else:
                        lower_bound = attack_const
                        if upper_bound is None:
                            current_c = attack_const * 2.0
                        else:
                            current_c = (lower_bound + upper_bound) / 2.0

                c_history_entry = {
                    **trial,
                    "valid": result["valid"],
                    "counterfactual_prediction": classes[result["prediction"]],
                    "best_step": result["best_step"],
                    "selection_distances": result["selection_distances"],
                }
                if args.c_search_mode == "adaptive_binary":
                    c_history_entry["updated_lower_bound"] = lower_bound
                    c_history_entry["updated_upper_bound"] = upper_bound
                    c_history_entry["next_attack_const"] = current_c
                c_history.append(c_history_entry)
                all_c_history.append(c_history_entry)

                attempt = {
                    "target_class_index": target_class,
                    "target_class": classes[target_class],
                    "attack_const": attack_const,
                    "c_step": trial["c_step"],
                    "c_search_mode": args.c_search_mode,
                    "prototype_details": prototype_details,
                    "result": result,
                }
                attempted_targets.append(attempt)

                if best_attempt is None or score_attempt_for_selection(
                    result,
                    original_pixels,
                    selection_metric=args.selection_metric,
                    beta=args.beta,
                ) < score_attempt_for_selection(
                    best_attempt["result"],
                    original_pixels,
                    selection_metric=args.selection_metric,
                    beta=args.beta,
                ):
                    best_attempt = attempt

            if best_attempt is not None and best_attempt["result"]["valid"]:
                break

        target_class = best_attempt["target_class_index"]
        result = best_attempt["result"]

        output_path = output_dir / f"sample_{sample_idx:02d}.png"
        original_confidence = float(sample["probabilities"][original_class])
        counterfactual_confidence = float(result["probabilities"][result["prediction"]])
        valid_counterfactual = result["prediction"] == target_class
        save_counterfactual_visualization(
            original_pixels=original_pixels,
            cf_pixels=result["image"],
            output_path=output_path,
            true_label=classes[sample["label"]],
            original_prediction=classes[original_class],
            original_confidence=original_confidence,
            target_class=classes[target_class],
            counterfactual_prediction=classes[result["prediction"]],
            counterfactual_confidence=counterfactual_confidence,
            valid_counterfactual=valid_counterfactual,
        )
        change_metrics = compute_change_metrics(original_pixels, result["image"])
        image_debug_stats = compute_image_debug_stats(original_pixels, result["image"])

        print(
            "  original "
            f"min={image_debug_stats['original']['min']:.4f} "
            f"max={image_debug_stats['original']['max']:.4f} "
            f"mean={image_debug_stats['original']['mean']:.4f}"
        )
        print(
            "  counterfactual "
            f"min={image_debug_stats['counterfactual']['min']:.4f} "
            f"max={image_debug_stats['counterfactual']['max']:.4f} "
            f"mean={image_debug_stats['counterfactual']['mean']:.4f}"
        )
        print(
            "  diff "
            f"min={image_debug_stats['diff']['min']:.4f} "
            f"max={image_debug_stats['diff']['max']:.4f} "
            f"mean={image_debug_stats['diff']['mean']:.4f} "
            f"std={image_debug_stats['diff']['std']:.4f}"
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
            "selection_metric": result["selection_metric"],
            "selection_distances": result["selection_distances"],
            "c_history": all_c_history,
            "prototype_details": best_attempt.get("prototype_details"),
            "attempted_targets": [
                {
                    "target_class": attempt["target_class"],
                    "attack_const": attempt["attack_const"],
                    "c_step": attempt["c_step"],
                    "c_search_mode": attempt["c_search_mode"],
                    "prototype_details": attempt["prototype_details"],
                    "valid": attempt["result"]["valid"],
                    "counterfactual_prediction": classes[
                        attempt["result"]["prediction"]
                    ],
                    "best_step": attempt["result"]["best_step"],
                    "autoencoder_loss": attempt["result"]["autoencoder_loss"],
                    "loss_terms": attempt["result"]["loss_terms"],
                    "selection_distances": attempt["result"]["selection_distances"],
                }
                for attempt in attempted_targets
            ],
            "original_probabilities": sample["probabilities"],
            "counterfactual_probabilities": result["probabilities"],
            "runtime_seconds": result["runtime_seconds"],
            "best_step": result["best_step"],
            "attack_const": result["attack_const"],
            "autoencoder_loss": result["autoencoder_loss"],
            "loss_terms": result["loss_terms"],
            "change_metrics": change_metrics,
            "image_debug_stats": image_debug_stats,
            "image_path": str(output_path),
            "summary_path": str(output_path.with_suffix(".summary.png")),
        }
        records.append(record)

        if args.log_loss_terms and result["loss_terms"] is not None:
            loss_terms = result["loss_terms"]
            print("  raw loss terms at selected result")
            for loss_name in [
                "class_loss",
                "l1_loss",
                "l2_loss",
                "tv_loss",
                "proto_loss",
                "ae_loss",
            ]:
                print(f"    {loss_name}: {loss_terms[loss_name]:.6f}")

    cfproto_implemented = [
        "optional CW-style targeted hinge loss",
        "optional prototype-distance target selection for non-manifest runs",
        "optional L1 image-distance penalty",
        "optional symmetric geometric attack-constant c-search",
        "optional adaptive binary-style attack-constant c-search",
        "optional elastic-net best-counterfactual selection",
        "optional polynomial learning-rate decay",
    ]
    cfproto_legacy = []
    cfproto_not_implemented = [
        "Alibi TensorFlow CounterfactualProto graph",
        "FISTA shrinkage-thresholding optimizer",
        "full Alibi binary search over c",
        "k-d-tree prototype selection",
        "TrustScore filtering",
    ]
    if args.prototype_space == "encoder":
        cfproto_implemented.append("encoder-space class prototypes via autoencoder encoder")
        if args.prototype_mode in {"knn_mean", "knn_point"}:
            cfproto_implemented.append(
                "local target-class k-nearest encoder prototypes"
            )
        cfproto_legacy.append("ResNet18 penultimate feature prototypes")
    else:
        cfproto_legacy.append("ResNet18 penultimate feature prototypes")
        cfproto_not_implemented.append(
            "encoder-space class prototypes via autoencoder encoder"
        )
    if args.autoencoder_path:
        cfproto_implemented.append("optional autoencoder reconstruction loss")
    else:
        cfproto_not_implemented.append("autoencoder reconstruction loss")

    metadata = {
        "method": "PyTorch prototype-guided optimization baseline",
        "model_path": args.model_path,
        "dataset_path": args.dataset_path,
        "classes": classes,
        "parameters": {
            "steps": args.steps,
            "learning_rate": args.learning_rate,
            "attack_loss": args.attack_loss,
            "attack_const": args.attack_const,
            "c_init": args.c_init,
            "c_steps": args.c_steps,
            "c_search_mode": args.c_search_mode,
            "kappa": args.kappa,
            "lambda_l1": args.lambda_l1,
            "lambda_l2": args.lambda_l2,
            "lambda_tv": args.lambda_tv,
            "lambda_proto": args.lambda_proto,
            "autoencoder_path": args.autoencoder_path,
            "gamma": args.gamma,
            "prototype_space": args.prototype_space,
            "prototype_mode": args.prototype_mode,
            "prototype_k": args.prototype_k,
            "selection_metric": args.selection_metric,
            "beta": args.beta,
            "lr_schedule": args.lr_schedule,
            "max_delta": args.max_delta,
            "perturbation_resolution": args.perturbation_resolution,
            "target_strategy": args.target_strategy,
            "force_grayscale": not args.allow_color_changes,
            "manifest_path": args.manifest_path,
            "manifest_max_samples": args.manifest_max_samples,
            "log_loss_terms": args.log_loss_terms,
        },
        "autoencoder_checkpoint": (
            {
                "architecture": autoencoder_checkpoint.get("architecture"),
                "dataset_path": autoencoder_checkpoint.get("dataset_path"),
                "image_size": autoencoder_checkpoint.get("image_size"),
                "base_channels": autoencoder_checkpoint.get("base_channels"),
                "pixel_range": autoencoder_checkpoint.get("pixel_range"),
            }
            if autoencoder_checkpoint is not None
            else None
        ),
        "cfproto_alignment_note": {
            "implemented": cfproto_implemented,
            "legacy": cfproto_legacy,
            "not_implemented": cfproto_not_implemented,
            "prototype_space_note": (
                "Autoencoder encoder-space prototypes are closer to the CFProto "
                "idea than ResNet18 classifier-feature prototypes. This script "
                "still remains a PyTorch adaptation rather than a full Alibi "
                "CounterfactualProto reproduction."
            ),
        },
        "prototype_configuration": {
            "prototype_space": args.prototype_space,
            "prototype_mode": args.prototype_mode,
            "prototype_k": args.prototype_k,
            "prototype_source": prototype_source,
            "prototype_shape": list(prototypes.shape) if prototypes is not None else None,
            "autoencoder_prototypes_used": args.prototype_space == "encoder",
            "legacy_resnet_feature_prototypes": args.prototype_space == "resnet",
            "local_knn_prototypes_used": args.prototype_mode in {"knn_mean", "knn_point"},
            "preferred_cfproto_near_configuration": (
                "--prototype_space encoder with --autoencoder_path"
            ),
        },
        "manifest_fairness_note": {
            "manifest_samples_unchanged": args.manifest_path is not None,
            "manifest_targets_unchanged": args.manifest_path is not None,
            "target_selection_in_manifest_mode": (
                "record['target_class_index'] is used directly; target_strategy is ignored."
                if args.manifest_path
                else "non-manifest mode may use target_strategy"
            ),
            "invalid_counterfactuals_filtered": False,
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
    print(f"Saved {len(records)} counterfactual attempts to {output_dir}")
    print(f"Valid counterfactuals: {valid_count}/{len(records)}")


if __name__ == "__main__":
    main()
