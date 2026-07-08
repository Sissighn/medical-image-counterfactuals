import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torchvision.transforms.functional as TF

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.autoencoder import ARCHITECTURE_NAME, ConvAutoencoder
from src.data_utils import create_dataloaders
from src.train_model import get_device


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def denormalize(images):
    mean = IMAGENET_MEAN.to(images.device)
    std = IMAGENET_STD.to(images.device)
    return (images * std + mean).clamp(0.0, 1.0)


def load_autoencoder(autoencoder_path, device):
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
    return autoencoder, checkpoint


def make_brightness_contrast_perturbation(images):
    return torch.clamp(images * 1.05 + 0.02, 0.0, 1.0)


def make_blur_perturbation(images):
    return TF.gaussian_blur(images, kernel_size=[5, 5], sigma=[0.6, 0.6])


def make_patch_perturbation(images, patch_fraction=0.2, low_res=10, strength=0.15):
    perturbed = images.clone()
    _, _, height, width = images.shape
    patch_size = max(1, int(min(height, width) * patch_fraction))

    for idx in range(images.shape[0]):
        top = int(torch.randint(0, height - patch_size + 1, (1,), device=images.device).item())
        left = int(torch.randint(0, width - patch_size + 1, (1,), device=images.device).item())
        low_res_noise = torch.randn(
            1,
            images.shape[1],
            low_res,
            low_res,
            device=images.device,
        )
        patch_offset = torch.nn.functional.interpolate(
            low_res_noise,
            size=(patch_size, patch_size),
            mode="bilinear",
            align_corners=False,
        )[0]
        patch_offset = patch_offset / patch_offset.abs().amax().clamp_min(1e-6)
        patch_offset = patch_offset * strength
        perturbed[
            idx,
            :,
            top : top + patch_size,
            left : left + patch_size,
        ] = torch.clamp(
            perturbed[idx, :, top : top + patch_size, left : left + patch_size]
            + patch_offset,
            0.0,
            1.0,
        )

    return perturbed


def make_noisy_perturbation(images, noise_std):
    noise = torch.randn_like(images) * noise_std
    return torch.clamp(images + noise, 0.0, 1.0)


def reconstruction_loss(autoencoder, images):
    reconstructions = autoencoder(images).clamp(0.0, 1.0)
    per_image = torch.mean((reconstructions - images) ** 2, dim=(1, 2, 3))
    return per_image, reconstructions


def image_to_grayscale(image):
    return image.mean(dim=0).detach().cpu()


def save_example_grid(records, output_path):
    selected = records[: min(5, len(records))]
    if not selected:
        return

    fig, axes = plt.subplots(
        len(selected),
        5,
        figsize=(13, 2.4 * len(selected)),
        squeeze=False,
    )
    for row, record in enumerate(selected):
        panels = [
            (record["original_image"], "Original"),
            (record["brightness_contrast_image"], "Brightness/Contrast"),
            (record["blur_image"], "Blur"),
            (record["patch_image"], "Patch"),
            (record["noisy_image"], "Strong noise"),
        ]
        for col, (image, title) in enumerate(panels):
            axes[row, col].imshow(image_to_grayscale(image), cmap="gray", vmin=0.0, vmax=1.0)
            axes[row, col].set_title(title)
            axes[row, col].axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def check_plausibility(
    dataset_path,
    autoencoder_path,
    output_csv,
    max_samples,
    batch_size,
    noise_std,
    patch_fraction,
    patch_low_res,
    patch_strength,
):
    device = get_device()
    autoencoder, checkpoint = load_autoencoder(autoencoder_path, device)
    data = create_dataloaders(
        dataset_path,
        batch_size=batch_size,
        use_augmentation=False,
    )

    rows = []
    visual_records = []
    sample_counter = 0

    with torch.no_grad():
        for images, labels in data["test_loader"]:
            images = denormalize(images.to(device))
            labels = labels.to(device)
            brightness_contrast = make_brightness_contrast_perturbation(images)
            blur = make_blur_perturbation(images)
            patch = make_patch_perturbation(
                images,
                patch_fraction=patch_fraction,
                low_res=patch_low_res,
                strength=patch_strength,
            )
            noisy = make_noisy_perturbation(images, noise_std=noise_std)

            original_loss, _ = reconstruction_loss(autoencoder, images)
            brightness_contrast_loss, _ = reconstruction_loss(
                autoencoder,
                brightness_contrast,
            )
            blur_loss, _ = reconstruction_loss(autoencoder, blur)
            patch_loss, _ = reconstruction_loss(autoencoder, patch)
            noisy_loss, _ = reconstruction_loss(autoencoder, noisy)

            for idx in range(images.shape[0]):
                rows.append(
                    {
                        "sample_index": sample_counter,
                        "label_index": int(labels[idx].item()),
                        "label": data["classes"][int(labels[idx].item())],
                        "loss_original": float(original_loss[idx].item()),
                        "loss_brightness_contrast": float(
                            brightness_contrast_loss[idx].item()
                        ),
                        "loss_blur": float(blur_loss[idx].item()),
                        "loss_patch_perturbation": float(patch_loss[idx].item()),
                        "loss_strong_noise": float(noisy_loss[idx].item()),
                    }
                )
                if len(visual_records) < 5:
                    visual_records.append(
                        {
                            "original_image": images[idx].detach().cpu(),
                            "brightness_contrast_image": brightness_contrast[
                                idx
                            ].detach().cpu(),
                            "blur_image": blur[idx].detach().cpu(),
                            "patch_image": patch[idx].detach().cpu(),
                            "noisy_image": noisy[idx].detach().cpu(),
                        }
                    )
                sample_counter += 1
                if sample_counter >= max_samples:
                    break

            if sample_counter >= max_samples:
                break

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sample_index",
                "label_index",
                "label",
                "loss_original",
                "loss_brightness_contrast",
                "loss_blur",
                "loss_patch_perturbation",
                "loss_strong_noise",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    means = {
        "loss_original": sum(row["loss_original"] for row in rows) / len(rows),
        "loss_brightness_contrast": sum(
            row["loss_brightness_contrast"] for row in rows
        )
        / len(rows),
        "loss_blur": sum(row["loss_blur"] for row in rows) / len(rows),
        "loss_patch_perturbation": sum(
            row["loss_patch_perturbation"] for row in rows
        )
        / len(rows),
        "loss_strong_noise": sum(row["loss_strong_noise"] for row in rows) / len(rows),
    }
    perturbation_checks = {}
    for key in [
        "loss_brightness_contrast",
        "loss_blur",
        "loss_patch_perturbation",
        "loss_strong_noise",
    ]:
        perturbation_checks[f"{key}_above_original"] = (
            means[key] > means["loss_original"]
        )
    interpretation = {
        "brightness_contrast": (
            "above original"
            if perturbation_checks["loss_brightness_contrast_above_original"]
            else "not above original"
        ),
        "blur": (
            "above original"
            if perturbation_checks["loss_blur_above_original"]
            else (
                "below or equal to original; this is compatible with the smoothing "
                "bias of pixel-MSE autoencoders"
            )
        ),
        "patch_perturbation": (
            "above original; relevant for CFProto-like localized perturbations"
            if perturbation_checks["loss_patch_perturbation_above_original"]
            else (
                "not above original; this would weaken the AE term for CFProto-like "
                "patch perturbations"
            )
        ),
        "strong_noise": (
            "above original"
            if perturbation_checks["loss_strong_noise_above_original"]
            else "not above original; this would indicate a serious AE-prior issue"
        ),
    }

    summary = {
        "dataset_path": dataset_path,
        "autoencoder_path": autoencoder_path,
        "autoencoder_checkpoint": {
            "architecture": checkpoint.get("architecture"),
            "dataset_path": checkpoint.get("dataset_path"),
            "image_size": checkpoint.get("image_size"),
            "base_channels": checkpoint.get("base_channels"),
            "pixel_range": checkpoint.get("pixel_range"),
        },
        "num_samples": len(rows),
        "noise_std": noise_std,
        "patch_fraction": patch_fraction,
        "patch_low_res": patch_low_res,
        "patch_strength": patch_strength,
        "mean_losses": means,
        "perturbation_checks": perturbation_checks,
        "interpretation": interpretation,
        "warning": None
        if perturbation_checks["loss_patch_perturbation_above_original"]
        else (
            "Patch perturbation loss is not above the original loss. This weakens "
            "the autoencoder term for the CFProto-like perturbation type."
        ),
    }

    summary_path = output_csv.with_suffix(".summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)

    figure_path = output_csv.with_suffix(".examples.png")
    save_example_grid(visual_records, figure_path)

    print(f"Saved plausibility CSV to: {output_csv}")
    print(f"Saved plausibility summary to: {summary_path}")
    print(f"Saved example grid to: {figure_path}")
    print("Mean AE reconstruction losses:")
    print(f"  original:             {means['loss_original']:.6f}")
    print(f"  brightness/contrast:  {means['loss_brightness_contrast']:.6f}")
    print(f"  blur:                 {means['loss_blur']:.6f}")
    print(f"  patch perturbation:   {means['loss_patch_perturbation']:.6f}")
    print(f"  strong noise:         {means['loss_strong_noise']:.6f}")
    print("Perturbation losses above original:")
    for key, value in perturbation_checks.items():
        print(f"  {key}: {'yes' if value else 'no'}")
    print("Interpretation:")
    for key, value in interpretation.items():
        print(f"  {key}: {value}")
    if summary["warning"]:
        print(f"WARNING: {summary['warning']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--autoencoder_path", type=str, required=True)
    parser.add_argument("--output_csv", type=str, required=True)
    parser.add_argument("--max_samples", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--noise_std", type=float, default=0.18)
    parser.add_argument("--patch_fraction", type=float, default=0.2)
    parser.add_argument("--patch_low_res", type=int, default=10)
    parser.add_argument("--patch_strength", type=float, default=0.15)
    args = parser.parse_args()

    check_plausibility(
        dataset_path=args.dataset_path,
        autoencoder_path=args.autoencoder_path,
        output_csv=args.output_csv,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        noise_std=args.noise_std,
        patch_fraction=args.patch_fraction,
        patch_low_res=args.patch_low_res,
        patch_strength=args.patch_strength,
    )


if __name__ == "__main__":
    main()
