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

from src.autoencoder import (
    ARCHITECTURE_NAME,
    BOTTLENECK_ARCHITECTURE_NAME,
    ConvAutoencoder,
    ConvAutoencoderBottleneck,
)
from src.data_utils import IMAGE_SIZE, create_dataloaders
from src.evaluation_manifest import (
    load_image_from_manifest_record,
    load_manifest_records,
    manifest_record_metadata,
)
from src.train_model import get_device


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

# Original alibi CounterfactualProto constants: perturbed instances live in
# feature_range and graph gradients are clipped to `clip`.
FEATURE_RANGE = (0.0, 1.0)
GRADIENT_CLIP = (-1000.0, 1000.0)


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


def predict_proba(model, pixels):
    """Class probabilities on [0, 1] pixel inputs.

    Plays the role of alibi's `predict` function returning probabilities;
    the ImageNet normalization is part of the wrapped predictor.
    """
    logits, _ = model(normalize(pixels))
    return F.softmax(logits, dim=1)


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
    if architecture == ARCHITECTURE_NAME:
        autoencoder = ConvAutoencoder(
            input_channels=checkpoint.get("input_channels", 3),
            base_channels=checkpoint.get("base_channels", 32),
        ).to(device)
    elif architecture == BOTTLENECK_ARCHITECTURE_NAME:
        autoencoder = ConvAutoencoderBottleneck(
            input_channels=checkpoint.get("input_channels", 3),
            base_channels=checkpoint.get("base_channels", 32),
            image_size=checkpoint.get("image_size", IMAGE_SIZE),
            latent_dim=checkpoint.get("latent_dim", 256),
        ).to(device)
    else:
        raise ValueError(
            f"Unsupported autoencoder architecture: {architecture}. "
            f"Expected {ARCHITECTURE_NAME} or {BOTTLENECK_ARCHITECTURE_NAME}."
        )

    autoencoder.load_state_dict(checkpoint["model_state_dict"])
    autoencoder.eval()
    for parameter in autoencoder.parameters():
        parameter.requires_grad_(False)
    return autoencoder, checkpoint


def encode_flat(autoencoder, pixels):
    return torch.flatten(autoencoder.encode(pixels), start_dim=1)


def fit_class_encodings(model, autoencoder, train_loader, device):
    """Encoder-space class encodings and class-mean prototypes.

    Mirrors alibi `CounterfactualProto.fit`: class membership is determined by
    the classifier predictions on the training data, not by the true labels.
    """
    encodings = []
    predictions = []

    with torch.no_grad():
        for images, _ in train_loader:
            pixels = denormalize(images.to(device))
            encodings.append(encode_flat(autoencoder, pixels).detach())
            probabilities = predict_proba(model, pixels)
            predictions.append(torch.argmax(probabilities, dim=1).detach())

    if not encodings:
        raise ValueError("Cannot build encoder prototypes from an empty train loader.")

    encodings = torch.cat(encodings, dim=0)
    predictions = torch.cat(predictions, dim=0)

    class_enc = {}
    class_proto = {}
    num_classes = model.classifier.out_features
    for class_idx in range(num_classes):
        class_encodings = encodings[predictions == class_idx]
        class_enc[class_idx] = class_encodings
        if class_encodings.shape[0] > 0:
            class_proto[class_idx] = class_encodings.mean(dim=0)

    return class_enc, class_proto


def select_class_prototype(
    autoencoder,
    original_pixels,
    class_enc,
    class_proto,
    target_classes,
    k,
    k_type,
):
    """Closest target-class prototype, following alibi `attack`.

    With `k is None` the prototype of a class is the mean encoding of all
    instances predicted as that class. With `k` set, the k nearest encodings
    (`k_type='mean'`: their mean and mean distance, `k_type='point'`: the
    k-th nearest point) define the prototype. Among all candidate classes the
    one with the smallest distance to ENC(x) is selected.
    """
    with torch.no_grad():
        original_encoding = encode_flat(autoencoder, original_pixels)[0]

        dist_proto = {}
        proto_by_class = {}
        for class_idx in target_classes:
            if k is None:
                if class_idx not in class_proto:
                    continue
                prototype = class_proto[class_idx]
                dist_proto[class_idx] = float(
                    torch.linalg.vector_norm(original_encoding - prototype).item()
                )
                proto_by_class[class_idx] = prototype
            else:
                class_encodings = class_enc[class_idx]
                if class_encodings.shape[0] == 0:
                    continue
                distances = torch.linalg.vector_norm(
                    class_encodings - original_encoding.unsqueeze(0), dim=1
                )
                k_used = min(k, class_encodings.shape[0])
                nearest_distances, nearest_indices = torch.topk(
                    distances, k=k_used, largest=False
                )
                if k_type == "mean":
                    dist_proto[class_idx] = float(nearest_distances.mean().item())
                elif k_type == "point":
                    dist_proto[class_idx] = float(nearest_distances[-1].item())
                else:
                    raise ValueError(f"Unsupported k_type: {k_type}")
                proto_by_class[class_idx] = class_encodings[nearest_indices].mean(dim=0)

        if not dist_proto:
            raise ValueError(
                "No prototype could be built: no training examples were predicted "
                f"as any of the target classes {target_classes}."
            )

        id_proto = min(dist_proto, key=dist_proto.get)

    return proto_by_class[id_proto], id_proto, {
        "prototype_mode": "class_mean" if k is None else f"knn_{k_type}",
        "prototype_k": k,
        "k_type": k_type,
        "candidate_distances": {str(c): d for c, d in dist_proto.items()},
        "id_proto": int(id_proto),
    }


def compare(probabilities, orig_class, kappa):
    """Counterfactual condition from alibi: argmax after adding kappa to the
    original class probability must differ from the original class."""
    adjusted = probabilities.clone()
    adjusted[orig_class] += kappa
    return int(torch.argmax(adjusted).item()) != orig_class


def attack_loss_terms(probabilities, orig_class, kappa):
    """Hinge attack loss f(x, d) = max(0, p_orig - max_{i != orig} p_i + kappa)."""
    one_hot = torch.zeros_like(probabilities)
    one_hot[:, orig_class] = 1.0
    target_proba = torch.sum(probabilities * one_hot, dim=1)
    nontarget_proba_max = torch.max(
        (1.0 - one_hot) * probabilities - one_hot * 10000.0, dim=1
    ).values
    return torch.clamp(target_proba - nontarget_proba_max + kappa, min=0.0)


def shrinkage_thresholding(adv_s, orig, beta):
    """Element-wise FISTA shrinkage-thresholding around the original instance,
    projected onto the feature range (alibi `shrinkage_thresholding` scope)."""
    delta = adv_s - orig
    upper = torch.clamp(adv_s - beta, max=FEATURE_RANGE[1])
    lower = torch.clamp(adv_s + beta, min=FEATURE_RANGE[0])
    return torch.where(
        delta > beta,
        upper,
        torch.where(delta.abs() <= beta, orig, lower),
    )


def compute_loss_terms(
    model, autoencoder, pixels, orig_pixels, orig_class, kappa, beta
):
    """Raw (unweighted) loss terms for a given counterfactual, as sums like in
    the original implementation."""
    with torch.no_grad():
        probabilities = predict_proba(model, pixels)
        attack = attack_loss_terms(probabilities, orig_class, kappa)
        delta = pixels - orig_pixels
        l1 = torch.sum(torch.abs(delta))
        l2 = torch.sum(delta**2)
        reconstruction = autoencoder(pixels)
        ae = torch.sum((reconstruction - pixels) ** 2)
    return {
        "attack_loss": float(attack.sum().item()),
        "l1_loss_sum": float(l1.item()),
        "l2_loss_sum": float(l2.item()),
        "elastic_net_distance": float((l2 + beta * l1).item()),
        "ae_loss_sum": float(ae.item()),
    }


def cfproto_attack(
    model,
    autoencoder,
    original_pixels,
    orig_class,
    target_classes,
    target_prototype,
    max_iterations,
    learning_rate_init,
    kappa,
    beta,
    gamma,
    theta,
    c_init,
    c_steps,
    verbose=False,
    print_every=100,
):
    """FISTA attack loop following alibi `CounterfactualProto.attack` for a
    single instance.

    Optimized loss (gradient step on the auxiliary variable adv_s):
        c * L_attack + L2 + gamma * L_AE + theta * L_proto
    The beta * L1 elastic-net term is handled by shrinkage-thresholding, not by
    the gradient. All loss terms are sums, as in the original TF graph.
    """
    orig = original_pixels.detach()
    proto = target_prototype.detach()

    const_lb = 0.0
    const = c_init
    const_ub = 1e10

    overall_best_dist = 1e10
    overall_best_adv = None
    overall_best = None
    c_history = []
    last_adv = orig.clone()

    start_time = time.time()

    for c_step in range(c_steps):
        # variables are re-initialized to the original instance for each c step
        adv = orig.clone()
        adv_s = orig.clone()

        current_best_dist = 1e10
        current_best_class = -1

        for iteration in range(max_iterations):
            # polynomial learning-rate decay (power 0.5, end learning rate 0)
            learning_rate = (
                learning_rate_init * (1.0 - iteration / max_iterations) ** 0.5
            )

            # gradient of the optimized loss w.r.t. adv_s
            adv_s_var = adv_s.clone().requires_grad_(True)
            probabilities_s = predict_proba(model, adv_s_var)
            loss_attack_s = const * attack_loss_terms(
                probabilities_s, orig_class, kappa
            ).sum()
            loss_l2_s = torch.sum((adv_s_var - orig) ** 2)
            loss_ae_s = gamma * torch.sum((autoencoder(adv_s_var) - adv_s_var) ** 2)
            loss_proto_s = theta * torch.sum(
                (encode_flat(autoencoder, adv_s_var)[0] - proto) ** 2
            )
            loss_opt = loss_attack_s + loss_l2_s + loss_ae_s + loss_proto_s
            gradient = torch.autograd.grad(loss_opt, adv_s_var)[0]
            gradient = gradient.clamp(GRADIENT_CLIP[0], GRADIENT_CLIP[1])

            with torch.no_grad():
                # gradient descent step on adv_s
                adv_s_step = adv_s - learning_rate * gradient

                # FISTA: shrinkage-thresholding then momentum update
                adv_new = shrinkage_thresholding(adv_s_step, orig, beta)
                zt = (iteration + 1) / (iteration + 4)
                adv_s = (adv_new + zt * (adv_new - adv)).clamp(
                    FEATURE_RANGE[0], FEATURE_RANGE[1]
                )
                adv = adv_new

                # evaluate the counterfactual candidate adv
                probabilities = predict_proba(model, adv)[0]
                adv_class = int(torch.argmax(probabilities).item())
                delta = adv - orig
                l1 = torch.sum(torch.abs(delta))
                l2 = torch.sum(delta**2)
                dist = float((l2 + beta * l1).item())
                condition = (
                    compare(probabilities, orig_class, kappa)
                    and adv_class in target_classes
                )

                if condition and dist < current_best_dist:
                    current_best_dist = dist
                    current_best_class = adv_class

                if condition and dist < overall_best_dist:
                    overall_best_dist = dist
                    overall_best_adv = adv.detach().clone()
                    overall_best = {
                        "c_step": c_step,
                        "iteration": iteration + 1,
                        "attack_const": const,
                        "elastic_net_distance": dist,
                    }

            if verbose and iteration % print_every == 0:
                print(
                    f"    c_step {c_step} (c={const:.4f}) iteration {iteration}: "
                    f"loss_opt={float(loss_opt.item()):.4f} "
                    f"attack={float(loss_attack_s.item()):.4f} "
                    f"l2={float(loss_l2_s.item()):.4f} "
                    f"ae={float(loss_ae_s.item()):.4f} "
                    f"proto={float(loss_proto_s.item()):.4f}"
                )

        last_adv = adv.detach().clone()

        # adjust the constant c like the original binary search
        entry = {
            "c_step": c_step,
            "attack_const": const,
            "lower_bound": const_lb,
            "upper_bound": const_ub,
            "found_valid": current_best_class != -1,
            "current_best_distance": (
                current_best_dist if current_best_class != -1 else None
            ),
        }
        if current_best_class != -1:
            const_ub = min(const_ub, const)
            if const_ub < 1e9:
                const = (const_lb + const_ub) / 2.0
        else:
            const_lb = max(const_lb, const)
            if const_ub < 1e9:
                const = (const_lb + const_ub) / 2.0
            else:
                const *= 10.0
        entry["updated_lower_bound"] = const_lb
        entry["updated_upper_bound"] = const_ub
        entry["next_attack_const"] = const
        c_history.append(entry)

    runtime = time.time() - start_time

    found = overall_best_adv is not None
    final_pixels = overall_best_adv if found else last_adv

    with torch.no_grad():
        final_probabilities = predict_proba(model, final_pixels)
        final_prediction = int(torch.argmax(final_probabilities, dim=1).item())

    return {
        "image": final_pixels,
        "found": found,
        "valid": found,
        "prediction": final_prediction,
        "probabilities": final_probabilities[0].detach().cpu().tolist(),
        "runtime_seconds": runtime,
        "best_c_step": overall_best["c_step"] if found else None,
        "best_iteration": overall_best["iteration"] if found else None,
        "attack_const": overall_best["attack_const"] if found else None,
        "c_history": c_history,
    }


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
    parser.add_argument(
        "--max_iterations",
        type=int,
        default=1000,
        help="FISTA iterations per c step (alibi default: 1000).",
    )
    parser.add_argument(
        "--learning_rate_init",
        type=float,
        default=0.01,
        help="Initial learning rate for the polynomial decay (alibi default: 1e-2).",
    )
    parser.add_argument(
        "--kappa",
        type=float,
        default=0.0,
        help="Confidence margin of the hinge attack loss (alibi default: 0).",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.1,
        help="L1 weight, applied via FISTA shrinkage-thresholding (alibi default: 0.1).",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=1.0,
        help=(
            "Weight of the autoencoder reconstruction loss "
            "gamma * ||AE(cf) - cf||_2^2. The alibi MNIST example uses 100 on "
            "28x28 inputs; since all loss terms are sums, the weight must be "
            "rescaled to the input/encoder dimensionality (224x224 here)."
        ),
    )
    parser.add_argument(
        "--theta",
        type=float,
        default=0.5,
        help=(
            "Weight of the prototype loss theta * ||ENC(cf) - proto||_2^2. The "
            "alibi MNIST example uses 100 on a small latent space; rescale to "
            "the encoder dimensionality so the term is comparable to the L2 sum "
            "(check loss_terms in metadata.json)."
        ),
    )
    parser.add_argument(
        "--c_init",
        type=float,
        default=1.0,
        help="Initial attack-loss constant c (alibi MNIST example: 1).",
    )
    parser.add_argument(
        "--c_steps",
        type=int,
        default=2,
        help="Number of binary-search updates for c (alibi MNIST example: 2).",
    )
    parser.add_argument(
        "--autoencoder_path",
        type=str,
        required=True,
        help="ConvAutoencoder checkpoint used as ae_model and enc_model.",
    )
    parser.add_argument(
        "--prototype_k",
        type=int,
        default=None,
        help=(
            "Number of nearest instances defining a class prototype. Defaults "
            "to using all instances of the class (alibi default: k=None)."
        ),
    )
    parser.add_argument(
        "--k_type",
        choices=["mean", "point"],
        default="mean",
        help="Prototype from mean of k nearest encodings or the k-th nearest point.",
    )
    parser.add_argument(
        "--target_strategy",
        choices=["second_best", "all"],
        default="all",
        help=(
            "Target candidates for non-manifest runs. 'all' matches the alibi "
            "default (all classes except the original); the prototype picks the "
            "nearest candidate class. Manifest mode always uses the manifest target."
        ),
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print intermediate optimization losses every print_every iterations.",
    )
    parser.add_argument("--print_every", type=int, default=100)
    args = parser.parse_args()

    if args.prototype_k is not None and args.prototype_k < 1:
        parser.error("--prototype_k must be at least 1 when set.")

    device = get_device()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, checkpoint = load_checkpoint_model(args.model_path, device)
    classes = checkpoint["classes"]
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
    print("Fitting encoder-space class prototypes from training split (like alibi fit)...")
    class_enc, class_proto = fit_class_encodings(
        model, autoencoder, data["train_loader"], device
    )

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
        else:
            target_candidates = select_target_classes(
                original_probabilities, original_class, args.target_strategy
            )

        target_prototype, id_proto, prototype_details = select_class_prototype(
            autoencoder=autoencoder,
            original_pixels=original_pixels,
            class_enc=class_enc,
            class_proto=class_proto,
            target_classes=target_candidates,
            k=args.prototype_k,
            k_type=args.k_type,
        )

        print(
            f"Sample {sample_idx}: {classes[original_class]} -> "
            f"prototype class {classes[id_proto]} "
            f"(candidates: {[classes[c] for c in target_candidates]})"
        )

        result = cfproto_attack(
            model=model,
            autoencoder=autoencoder,
            original_pixels=original_pixels,
            orig_class=original_class,
            target_classes=target_candidates,
            target_prototype=target_prototype,
            max_iterations=args.max_iterations,
            learning_rate_init=args.learning_rate_init,
            kappa=args.kappa,
            beta=args.beta,
            gamma=args.gamma,
            theta=args.theta,
            c_init=args.c_init,
            c_steps=args.c_steps,
            verbose=args.verbose,
            print_every=args.print_every,
        )

        target_class = id_proto
        valid_counterfactual = result["valid"] and result["prediction"] in target_candidates
        loss_terms = compute_loss_terms(
            model=model,
            autoencoder=autoencoder,
            pixels=result["image"],
            orig_pixels=original_pixels,
            orig_class=original_class,
            kappa=args.kappa,
            beta=args.beta,
        )
        selection_distances = compute_elastic_net_distance(
            original_pixels, result["image"], args.beta
        )

        output_path = output_dir / f"sample_{sample_idx:02d}.png"
        original_confidence = float(sample["probabilities"][original_class])
        counterfactual_confidence = float(result["probabilities"][result["prediction"]])
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
        print(f"  valid counterfactual: {valid_counterfactual}")

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
            "target_candidates": [classes[c] for c in target_candidates],
            "counterfactual_prediction_index": result["prediction"],
            "counterfactual_prediction": classes[result["prediction"]],
            "counterfactual_confidence": counterfactual_confidence,
            "valid_counterfactual": valid_counterfactual,
            "counterfactual_found": result["found"],
            "selection_metric": "elastic_net",
            "selection_distances": selection_distances,
            "c_history": result["c_history"],
            "prototype_details": prototype_details,
            "original_probabilities": sample["probabilities"],
            "counterfactual_probabilities": result["probabilities"],
            "runtime_seconds": result["runtime_seconds"],
            "best_c_step": result["best_c_step"],
            "best_iteration": result["best_iteration"],
            "attack_const": result["attack_const"],
            "loss_terms": loss_terms,
            "change_metrics": change_metrics,
            "image_debug_stats": image_debug_stats,
            "image_path": str(output_path),
            "summary_path": str(output_path.with_suffix(".summary.png")),
        }
        records.append(record)

    metadata = {
        "method": "CFProto original-style prototype-guided counterfactuals",
        "model_path": args.model_path,
        "dataset_path": args.dataset_path,
        "classes": classes,
        "parameters": {
            "max_iterations": args.max_iterations,
            "learning_rate_init": args.learning_rate_init,
            "kappa": args.kappa,
            "beta": args.beta,
            "gamma": args.gamma,
            "theta": args.theta,
            "c_init": args.c_init,
            "c_steps": args.c_steps,
            "autoencoder_path": args.autoencoder_path,
            "prototype_space": "encoder",
            "prototype_mode": "class_mean" if args.prototype_k is None else f"knn_{args.k_type}",
            "prototype_k": args.prototype_k,
            "k_type": args.k_type,
            "feature_range": list(FEATURE_RANGE),
            "gradient_clip": list(GRADIENT_CLIP),
            "target_strategy": args.target_strategy,
            "manifest_path": args.manifest_path,
            "manifest_max_samples": args.manifest_max_samples,
        },
        "autoencoder_checkpoint": (
            {
                "architecture": autoencoder_checkpoint.get("architecture"),
                "latent_dim": autoencoder_checkpoint.get("latent_dim"),
                "dataset_path": autoencoder_checkpoint.get("dataset_path"),
                "image_size": autoencoder_checkpoint.get("image_size"),
                "base_channels": autoencoder_checkpoint.get("base_channels"),
                "pixel_range": autoencoder_checkpoint.get("pixel_range"),
                "autoencoder_path": args.autoencoder_path,
            }
            if autoencoder_checkpoint is not None
            else None
        ),
        "cfproto_alignment_note": {
            "implemented": [
                "FISTA optimization with shrinkage-thresholding and Nesterov momentum",
                "hinge attack loss pushing the original class below the best other class",
                "loss c * L_attack + L2 + beta * L1 + gamma * L_AE + theta * L_proto (sums)",
                "binary search over the attack constant c with x10 escalation",
                "polynomial learning-rate decay (power 0.5)",
                "encoder-space class prototypes from classifier predictions (fit)",
                "nearest-prototype target selection over candidate classes",
                "elastic-net (L2 + beta * L1) best-counterfactual selection",
                "counterfactual condition: kappa-adjusted argmax differs from original class",
            ],
            "not_implemented": [
                "TensorFlow 1.x graph implementation (reimplemented in PyTorch)",
                "black-box mode with numerical gradients",
                "categorical variables and k-d-tree prototypes",
                "TrustScore threshold filtering (alibi default threshold=0 also disables it)",
            ],
            "framework_note": (
                "The classifier is a PyTorch ResNet-18 whose ImageNet normalization "
                "is wrapped into the predict function; optimization happens in "
                "[0, 1] pixel space, matching feature_range=(0, 1) in alibi."
            ),
        },
        "prototype_configuration": {
            "prototype_space": "encoder",
            "prototype_mode": "class_mean" if args.prototype_k is None else f"knn_{args.k_type}",
            "prototype_k": args.prototype_k,
            "k_type": args.k_type,
            "class_membership": "classifier predictions on the training split",
            "autoencoder_architecture": autoencoder_checkpoint.get("architecture"),
            "autoencoder_latent_dim": autoencoder_checkpoint.get("latent_dim"),
            "autoencoder_path": args.autoencoder_path,
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
