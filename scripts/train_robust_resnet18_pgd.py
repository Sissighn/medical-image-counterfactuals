import argparse
import json
import socket
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score
from torchvision import models
from torchvision.models import ResNet18_Weights

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_utils import create_dataloaders
from src.train_model import compute_class_weights, get_device


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def create_model(num_classes, pretrained=False, download_timeout=30):
    weights = None
    if pretrained:
        weights = ResNet18_Weights.DEFAULT

    previous_timeout = socket.getdefaulttimeout()
    socket.setdefaulttimeout(download_timeout)
    try:
        model = models.resnet18(weights=weights)
    except Exception as error:
        print(f"Could not load pretrained weights: {error}")
        model = models.resnet18(weights=None)
    finally:
        socket.setdefaulttimeout(previous_timeout)

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def denormalize(images):
    mean = IMAGENET_MEAN.to(images.device)
    std = IMAGENET_STD.to(images.device)
    return (images * std + mean).clamp(0.0, 1.0)


def normalize(images):
    mean = IMAGENET_MEAN.to(images.device)
    std = IMAGENET_STD.to(images.device)
    return (images - mean) / std


def pgd_attack(
    model,
    images_normalized,
    labels,
    criterion,
    epsilon,
    step_size,
    steps,
):
    was_training = model.training
    model.eval()

    original_pixels = denormalize(images_normalized).detach()
    adv_pixels = original_pixels.clone().detach()
    adv_pixels = adv_pixels + torch.empty_like(adv_pixels).uniform_(-epsilon, epsilon)
    adv_pixels = adv_pixels.clamp(0.0, 1.0)

    for _ in range(steps):
        adv_pixels.requires_grad_(True)
        logits = model(normalize(adv_pixels))
        loss = criterion(logits, labels)
        grad = torch.autograd.grad(loss, adv_pixels)[0]
        adv_pixels = adv_pixels.detach() + step_size * grad.sign()
        adv_pixels = torch.max(
            torch.min(adv_pixels, original_pixels + epsilon),
            original_pixels - epsilon,
        )
        adv_pixels = adv_pixels.clamp(0.0, 1.0).detach()

    if was_training:
        model.train()
    return normalize(adv_pixels)


def train_one_epoch_robust(
    model,
    train_loader,
    criterion,
    optimizer,
    device,
    epsilon,
    step_size,
    pgd_steps,
    clean_loss_weight,
):
    model.train()
    running_loss = 0.0
    all_labels = []
    all_predictions = []

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        adv_images = pgd_attack(
            model=model,
            images_normalized=images,
            labels=labels,
            criterion=criterion,
            epsilon=epsilon,
            step_size=step_size,
            steps=pgd_steps,
        )

        optimizer.zero_grad()
        clean_logits = model(images)
        adv_logits = model(adv_images)
        clean_loss = criterion(clean_logits, labels)
        adv_loss = criterion(adv_logits, labels)
        loss = clean_loss_weight * clean_loss + (1.0 - clean_loss_weight) * adv_loss
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        predictions = torch.argmax(clean_logits, dim=1)
        all_labels.extend(labels.detach().cpu().tolist())
        all_predictions.extend(predictions.detach().cpu().tolist())

    return (
        running_loss / len(train_loader.dataset),
        accuracy_score(all_labels, all_predictions),
        f1_score(all_labels, all_predictions, average="weighted"),
    )


def evaluate(model, data_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            running_loss += loss.item() * images.size(0)
            predictions = torch.argmax(logits, dim=1)
            all_labels.extend(labels.detach().cpu().tolist())
            all_predictions.extend(predictions.detach().cpu().tolist())

    return (
        running_loss / len(data_loader.dataset),
        accuracy_score(all_labels, all_predictions),
        f1_score(all_labels, all_predictions, average="weighted"),
    )


def train_robust_model(args):
    device = get_device()
    output_model_path = Path(args.output_model_path)
    output_model_path.parent.mkdir(parents=True, exist_ok=True)

    data = create_dataloaders(
        args.dataset_path,
        batch_size=args.batch_size,
        use_augmentation=not args.no_augmentation,
    )
    classes = data["classes"]
    num_classes = data["num_classes"]

    model = create_model(num_classes, pretrained=args.pretrained).to(device)
    class_weights = None
    if not args.no_class_weights:
        class_weights = compute_class_weights(
            data["train_dataset"], num_classes, device
        )
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    best_val_f1 = 0.0
    history = []

    print(f"Device: {device}")
    print(f"Dataset: {args.dataset_name}")
    print(f"Classes: {classes}")
    print(
        "PGD settings: "
        f"epsilon={args.epsilon}, step_size={args.step_size}, "
        f"steps={args.pgd_steps}, clean_loss_weight={args.clean_loss_weight}"
    )

    for epoch in range(args.epochs):
        train_loss, train_acc, train_f1 = train_one_epoch_robust(
            model=model,
            train_loader=data["train_loader"],
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epsilon=args.epsilon,
            step_size=args.step_size,
            pgd_steps=args.pgd_steps,
            clean_loss_weight=args.clean_loss_weight,
        )
        val_loss, val_acc, val_f1 = evaluate(
            model, data["val_loader"], criterion, device
        )
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "train_f1": train_f1,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "val_f1": val_f1,
            }
        )

        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"train loss={train_loss:.4f} acc={train_acc:.4f} f1={train_f1:.4f} | "
            f"val loss={val_loss:.4f} acc={val_acc:.4f} f1={val_f1:.4f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            checkpoint = {
                "dataset_name": args.dataset_name,
                "model_name": "resnet18",
                "training_type": "pgd_adversarial_training",
                "model_state_dict": model.state_dict(),
                "classes": classes,
                "class_to_idx": data["class_to_idx"],
                "num_classes": num_classes,
                "image_size": 224,
                "pretrained": args.pretrained,
                "use_augmentation": not args.no_augmentation,
                "use_class_weights": not args.no_class_weights,
                "class_weights": (
                    class_weights.detach().cpu().tolist()
                    if class_weights is not None
                    else None
                ),
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "best_val_f1": best_val_f1,
                "pgd_epsilon_pixel_space": args.epsilon,
                "pgd_step_size_pixel_space": args.step_size,
                "pgd_steps": args.pgd_steps,
                "clean_loss_weight": args.clean_loss_weight,
            }
            torch.save(checkpoint, output_model_path)
            print(f"Saved best robust checkpoint: {output_model_path}")

    history_path = Path(args.history_path)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=4)
    print(f"Saved training history: {history_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--output_model_path", required=True)
    parser.add_argument("--history_path", required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--no_augmentation", action="store_true")
    parser.add_argument("--no_class_weights", action="store_true")
    parser.add_argument("--epsilon", type=float, default=0.03)
    parser.add_argument("--step_size", type=float, default=0.007)
    parser.add_argument("--pgd_steps", type=int, default=7)
    parser.add_argument("--clean_loss_weight", type=float, default=0.5)
    args = parser.parse_args()

    train_robust_model(args)


if __name__ == "__main__":
    main()
