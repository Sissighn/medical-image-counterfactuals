import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.autoencoder import (  # noqa: E402
    ARCHITECTURE_NAME,
    BOTTLENECK_ARCHITECTURE_NAME,
    ConvAutoencoder,
    ConvAutoencoderBottleneck,
)
from src.data_utils import IMAGE_SIZE, create_dataloaders  # noqa: E402
from src.train_model import get_device  # noqa: E402


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def denormalize(images):
    mean = IMAGENET_MEAN.to(images.device)
    std = IMAGENET_STD.to(images.device)
    return (images * std + mean).clamp(0.0, 1.0)


def load_autoencoder(autoencoder_path, device):
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
    return autoencoder, checkpoint


def perturb_brightness_contrast(images):
    return torch.clamp(images * 1.12 + 0.04, 0.0, 1.0)


def perturb_blur(images):
    return TF.gaussian_blur(images, kernel_size=[7, 7], sigma=[1.1, 1.1])


def perturb_patch(images, patch_fraction=0.22):
    patched = images.clone()
    _, _, height, width = patched.shape
    patch_h = max(1, int(height * patch_fraction))
    patch_w = max(1, int(width * patch_fraction))
    y0 = (height - patch_h) // 2
    x0 = (width - patch_w) // 2
    patched[:, :, y0 : y0 + patch_h, x0 : x0 + patch_w] = 0.0
    return patched


def perturb_noise(images, noise_std):
    return torch.clamp(images + torch.randn_like(images) * noise_std, 0.0, 1.0)


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

    panel_keys = [
        ("original", "Original"),
        ("brightness_contrast", "Brightness/Contrast"),
        ("blur", "Blur"),
        ("patch", "Patch"),
        ("strong_noise", "Strong noise"),
    ]
    fig, axes = plt.subplots(
        len(selected),
        len(panel_keys),
        figsize=(13, 2.4 * len(selected)),
        squeeze=False,
    )
    for row, record in enumerate(selected):
        for col, (key, title) in enumerate(panel_keys):
            axes[row, col].imshow(
                image_to_grayscale(record[key]),
                cmap="gray",
                vmin=0.0,
                vmax=1.0,
            )
            axes[row, col].set_title(title)
            axes[row, col].axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def check_plausibility(
    dataset_path,
    autoencoder_path,
    output_dir,
    max_samples,
    batch_size,
    noise_std,
    num_workers,
):
    device = get_device()
    autoencoder, checkpoint = load_autoencoder(autoencoder_path, device)
    data = create_dataloaders(
        dataset_path,
        batch_size=batch_size,
        use_augmentation=False,
        num_workers=num_workers,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    visual_records = []
    sample_counter = 0

    with torch.no_grad():
        for images, labels in data["test_loader"]:
            images = denormalize(images.to(device))
            labels = labels.to(device)
            variants = {
                "original": images,
                "brightness_contrast": perturb_brightness_contrast(images),
                "blur": perturb_blur(images),
                "patch": perturb_patch(images),
                "strong_noise": perturb_noise(images, noise_std=noise_std),
            }
            losses = {
                name: reconstruction_loss(autoencoder, variant)[0]
                for name, variant in variants.items()
            }

            for idx in range(images.shape[0]):
                row = {
                    "sample_index": sample_counter,
                    "label_index": int(labels[idx].item()),
                    "label": data["classes"][int(labels[idx].item())],
                }
                for name in variants:
                    row[f"loss_{name}"] = float(losses[name][idx].item())
                rows.append(row)

                if len(visual_records) < 5:
                    visual_records.append(
                        {name: variant[idx].detach().cpu() for name, variant in variants.items()}
                    )

                sample_counter += 1
                if sample_counter >= max_samples:
                    break

            if sample_counter >= max_samples:
                break

    csv_path = output_dir / "plausibility_losses.csv"
    fieldnames = [
        "sample_index",
        "label_index",
        "label",
        "loss_original",
        "loss_brightness_contrast",
        "loss_blur",
        "loss_patch",
        "loss_strong_noise",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    means = {
        key: sum(row[key] for row in rows) / len(rows)
        for key in fieldnames
        if key.startswith("loss_")
    }
    summary = {
        "dataset_path": dataset_path,
        "autoencoder_path": autoencoder_path,
        "num_samples": len(rows),
        "noise_std": noise_std,
        "autoencoder_checkpoint": {
            "architecture": checkpoint.get("architecture"),
            "latent_dim": checkpoint.get("latent_dim"),
            "dataset_path": checkpoint.get("dataset_path"),
            "image_size": checkpoint.get("image_size"),
            "base_channels": checkpoint.get("base_channels"),
            "pixel_range": checkpoint.get("pixel_range"),
        },
        "mean_losses": means,
        "patch_above_original": means["loss_patch"] > means["loss_original"],
        "strong_noise_above_original": means["loss_strong_noise"] > means["loss_original"],
    }
    summary_path = output_dir / "plausibility_summary.json"
    summary_path.write_text(json.dumps(summary, indent=4) + "\n")

    figure_path = output_dir / "plausibility_examples.png"
    save_example_grid(visual_records, figure_path)

    print(f"Saved plausibility CSV to: {csv_path}")
    print(f"Saved plausibility summary to: {summary_path}")
    print(f"Saved example grid to: {figure_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--autoencoder_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_samples", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--noise_std", type=float, default=0.18)
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()

    check_plausibility(
        dataset_path=args.dataset_path,
        autoencoder_path=args.autoencoder_path,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        noise_std=args.noise_std,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
