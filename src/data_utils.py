from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

IMAGE_SIZE = 224
BATCH_SIZE = 16


def get_transforms(use_augmentation=True):
    train_transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            *(
                [
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomRotation(degrees=10),
                    transforms.ColorJitter(brightness=0.1, contrast=0.1),
                ]
                if use_augmentation
                else []
            ),
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


def create_dataloaders(dataset_dir, batch_size=BATCH_SIZE, use_augmentation=True):
    dataset_dir = Path(dataset_dir)

    train_transform, eval_transform = get_transforms(use_augmentation=use_augmentation)

    train_dataset = datasets.ImageFolder(
        root=dataset_dir / "train", transform=train_transform
    )

    val_dataset = datasets.ImageFolder(
        root=dataset_dir / "val", transform=eval_transform
    )

    test_dataset = datasets.ImageFolder(
        root=dataset_dir / "test", transform=eval_transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return {
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "test_dataset": test_dataset,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "classes": train_dataset.classes,
        "class_to_idx": train_dataset.class_to_idx,
        "num_classes": len(train_dataset.classes),
    }
