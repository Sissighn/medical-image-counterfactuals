import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.autoencoder import ARCHITECTURE_NAME, ConvAutoencoder
from src.data_utils import IMAGE_SIZE, create_dataloaders
from src.train_model import get_device


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def denormalize(images):
    mean = IMAGENET_MEAN.to(images.device)
    std = IMAGENET_STD.to(images.device)
    return (images * std + mean).clamp(0.0, 1.0)


def image_to_grayscale(image):
    return image.mean(dim=0).detach().cpu()


def save_loss_curve(history, output_path):
    figure_path = output_path.with_name(f"{output_path.stem}_loss_curve.png")
    epochs = [record["epoch"] for record in history]
    losses = [record["train_loss"] for record in history]

    fig, axis = plt.subplots(figsize=(7, 4.5))
    axis.plot(epochs, losses, marker="o", linewidth=1.5)
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Training reconstruction MSE")
    axis.set_title("Autoencoder Training Loss")
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)
    print(f"Loss curve saved to: {figure_path}")


def save_reconstruction_examples(autoencoder, data_loader, device, output_path, max_examples):
    figure_path = output_path.with_name(f"{output_path.stem}_reconstructions.png")
    autoencoder.eval()

    with torch.no_grad():
        images, _ = next(iter(data_loader))
        images = denormalize(images.to(device))[:max_examples]
        reconstructions = autoencoder(images).clamp(0.0, 1.0)
        diffs = torch.abs(reconstructions - images)

    rows = images.shape[0]
    fig, axes = plt.subplots(rows, 3, figsize=(8, 2.4 * rows), squeeze=False)
    for row in range(rows):
        panels = [
            (image_to_grayscale(images[row]), "Original"),
            (image_to_grayscale(reconstructions[row]), "Reconstruction"),
            (image_to_grayscale(diffs[row]), "Absolute difference"),
        ]
        for col, (panel, title) in enumerate(panels):
            cmap = "gray" if col < 2 else "magma"
            axes[row, col].imshow(panel, cmap=cmap, vmin=0.0, vmax=1.0)
            axes[row, col].set_title(title)
            axes[row, col].axis("off")

    fig.tight_layout()
    fig.savefig(figure_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Reconstruction examples saved to: {figure_path}")


def train_autoencoder(
    dataset_path,
    output_path,
    epochs,
    batch_size,
    learning_rate,
    base_channels,
    max_batches=None,
    num_reconstruction_examples=5,
):
    device = get_device()
    start_time = time.time()
    print(f"Device: {device}")
    print(f"Dataset path: {dataset_path}")
    print(f"Architecture: {ARCHITECTURE_NAME}")

    data = create_dataloaders(
        dataset_path,
        batch_size=batch_size,
        use_augmentation=False,
    )
    train_loader = data["train_loader"]

    autoencoder = ConvAutoencoder(base_channels=base_channels).to(device)
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=learning_rate)
    history = []

    for epoch in range(epochs):
        autoencoder.train()
        running_loss = 0.0
        seen_images = 0
        num_batches = 0

        for batch_idx, (images, _) in enumerate(train_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            images = denormalize(images.to(device))

            optimizer.zero_grad()
            reconstructions = autoencoder(images)
            loss = F.mse_loss(reconstructions, images)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            seen_images += images.size(0)
            num_batches += 1

        epoch_loss = running_loss / seen_images if seen_images else 0.0
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": epoch_loss,
                "num_images": seen_images,
                "num_batches": num_batches,
            }
        )
        print(f"Epoch {epoch + 1}/{epochs} | Train reconstruction MSE: {epoch_loss:.6f}")

    training_seconds = time.time() - start_time

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": autoencoder.state_dict(),
        "dataset_path": str(dataset_path),
        "image_size": IMAGE_SIZE,
        "input_channels": 3,
        "base_channels": base_channels,
        "architecture": ARCHITECTURE_NAME,
        "loss": "mse_reconstruction",
        "pixel_range": "[0, 1]",
        "normalization_note": (
            "Training batches are loaded with the project ImageNet-normalized "
            "DataLoader and denormalized before reconstruction loss."
        ),
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "max_batches": max_batches,
        "training_seconds": training_seconds,
        "classes": data["classes"],
        "history": history,
    }
    torch.save(checkpoint, output_path)

    history_path = output_path.with_suffix(".history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=4)

    print(f"Autoencoder checkpoint saved to: {output_path}")
    print(f"Training history saved to: {history_path}")
    print(f"Training time: {training_seconds:.1f} seconds")

    save_loss_curve(history, output_path)
    save_reconstruction_examples(
        autoencoder=autoencoder,
        data_loader=data["test_loader"],
        device=device,
        output_path=output_path,
        max_examples=num_reconstruction_examples,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--base_channels", type=int, default=32)
    parser.add_argument(
        "--max_batches",
        type=int,
        default=None,
        help="Optional cap for smoke tests. By default the full train split is used.",
    )
    parser.add_argument("--num_reconstruction_examples", type=int, default=5)
    args = parser.parse_args()

    train_autoencoder(
        dataset_path=args.dataset_path,
        output_path=args.output_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        base_channels=args.base_channels,
        max_batches=args.max_batches,
        num_reconstruction_examples=args.num_reconstruction_examples,
    )


if __name__ == "__main__":
    main()
