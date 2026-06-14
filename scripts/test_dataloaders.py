from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Pfade zu den vorbereiteten Datensätzen
BUSI_DIR = Path("data/processed/BUSI")
PNEUMONIA_DIR = Path("data/processed/Pneumonia")


# Diese Bildgröße ist Standard für viele CNN-Modelle, z. B. ResNet18
IMAGE_SIZE = 224

# Anzahl Bilder, die gleichzeitig geladen werden
BATCH_SIZE = 16


def get_transforms():
    """
    Transformations = Vorbereitungsschritte für jedes Bild.

    Resize:
    Bild wird auf 224x224 Pixel gebracht.

    ToTensor:
    Bild wird in ein Zahlenformat umgewandelt, mit dem PyTorch arbeiten kann.

    Normalize:
    Pixelwerte werden standardisiert.
    Diese Werte sind Standardwerte, die oft bei vortrainierten CNNs wie ResNet genutzt werden.
    """
    train_transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    eval_transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    return train_transform, eval_transform


def create_dataloaders(dataset_dir):
    train_transform, eval_transform = get_transforms()

    train_dataset = datasets.ImageFolder(
        root=dataset_dir / "train", transform=train_transform
    )

    val_dataset = datasets.ImageFolder(
        root=dataset_dir / "val", transform=eval_transform
    )

    test_dataset = datasets.ImageFolder(
        root=dataset_dir / "test", transform=eval_transform
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return (
        train_dataset,
        val_dataset,
        test_dataset,
        train_loader,
        val_loader,
        test_loader,
    )


def check_dataset(name, dataset_dir):
    print("=" * 50)
    print(name)
    print("=" * 50)

    train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader = (
        create_dataloaders(dataset_dir)
    )

    print("Klassen:")
    print(train_dataset.classes)
    print()

    print("Class-to-index:")
    print(train_dataset.class_to_idx)
    print()

    print("Anzahl Bilder:")
    print(f"Train: {len(train_dataset)}")
    print(f"Val:   {len(val_dataset)}")
    print(f"Test:  {len(test_dataset)}")
    print()

    images, labels = next(iter(train_loader))

    print("Ein Batch wurde erfolgreich geladen.")
    print(f"Bild-Batch-Shape:  {images.shape}")
    print(f"Label-Batch-Shape: {labels.shape}")
    print(f"Labels: {labels}")
    print()


def main():
    check_dataset("BUSI", BUSI_DIR)
    check_dataset("Pneumonia", PNEUMONIA_DIR)


if __name__ == "__main__":
    main()
