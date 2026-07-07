import argparse
import json
import sys
from pathlib import Path

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


def train_autoencoder(
    dataset_path,
    output_path,
    epochs,
    batch_size,
    learning_rate,
    base_channels,
    max_batches=None,
):
    device = get_device()
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
        "classes": data["classes"],
        "history": history,
    }
    torch.save(checkpoint, output_path)

    history_path = output_path.with_suffix(".history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=4)

    print(f"Autoencoder checkpoint saved to: {output_path}")
    print(f"Training history saved to: {history_path}")


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
    args = parser.parse_args()

    train_autoencoder(
        dataset_path=args.dataset_path,
        output_path=args.output_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        base_channels=args.base_channels,
        max_batches=args.max_batches,
    )


if __name__ == "__main__":
    main()
