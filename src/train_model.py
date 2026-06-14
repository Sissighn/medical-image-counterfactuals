from pathlib import Path
import argparse
import json
import socket

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torchvision.models import ResNet18_Weights
from sklearn.metrics import accuracy_score, f1_score, classification_report

from src.data_utils import create_dataloaders


def get_device():
    """
    Wählt automatisch aus, ob GPU oder CPU genutzt wird.

    Auf Mac mit Apple Silicon kann 'mps' verwendet werden.
    Auf Windows/Linux mit NVIDIA-GPU kann 'cuda' verwendet werden.
    Sonst wird CPU verwendet.
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def create_model(num_classes, pretrained=False, download_timeout=30):
    """
    Erstellt ein ResNet18-Modell.

    ResNet18 ist ein CNN, das oft für Bildklassifikation genutzt wird.
    Optional kann ein vortrainiertes Modell verwendet werden. Die letzte Schicht
    wird ersetzt, damit sie zu unserer Anzahl an Klassen passt.
    """
    weights = None
    if pretrained:
        try:
            weights = ResNet18_Weights.DEFAULT
        except Exception as error:
            print(f"Pretrained weights konnten nicht vorbereitet werden: {error}")
            print("Training wird mit zufälliger Initialisierung fortgesetzt.")

    previous_timeout = socket.getdefaulttimeout()
    socket.setdefaulttimeout(download_timeout)
    try:
        try:
            model = models.resnet18(weights=weights)
        except Exception as error:
            print(f"Pretrained weights konnten nicht geladen werden: {error}")
            print("Training wird mit zufälliger Initialisierung fortgesetzt.")
            model = models.resnet18(weights=None)
    finally:
        socket.setdefaulttimeout(previous_timeout)

    # Anzahl der Eingabefeatures der letzten Schicht
    in_features = model.fc.in_features

    # Letzte Schicht ersetzen:
    # BUSI: 3 Klassen
    # Pneumonia: 2 Klassen
    model.fc = nn.Linear(in_features, num_classes)

    return model


def compute_class_weights(train_dataset, num_classes, device):
    targets = torch.tensor(train_dataset.targets, dtype=torch.long)
    class_counts = torch.bincount(targets, minlength=num_classes).float()
    class_weights = class_counts.sum() / (num_classes * class_counts)
    return class_weights.to(device)


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()

    running_loss = 0.0
    all_labels = []
    all_predictions = []

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        # Alte Gradienten löschen
        optimizer.zero_grad()

        # Vorhersage berechnen
        outputs = model(images)

        # Fehler berechnen
        loss = criterion(outputs, labels)

        # Backpropagation
        loss.backward()

        # Modellgewichte aktualisieren
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        predictions = torch.argmax(outputs, dim=1)

        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predictions.cpu().numpy())

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_accuracy = accuracy_score(all_labels, all_predictions)
    epoch_f1 = f1_score(all_labels, all_predictions, average="weighted")

    return epoch_loss, epoch_accuracy, epoch_f1


def evaluate(model, data_loader, criterion, device):
    model.eval()

    running_loss = 0.0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            predictions = torch.argmax(outputs, dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())

    epoch_loss = running_loss / len(data_loader.dataset)
    epoch_accuracy = accuracy_score(all_labels, all_predictions)
    epoch_f1 = f1_score(all_labels, all_predictions, average="weighted")

    return epoch_loss, epoch_accuracy, epoch_f1, all_labels, all_predictions


def train_model(
    dataset_name,
    dataset_path,
    output_model_path,
    epochs,
    batch_size,
    learning_rate,
    pretrained,
    use_augmentation,
    use_class_weights,
):
    device = get_device()
    print(f"Device: {device}")

    data = create_dataloaders(
        dataset_path, batch_size=batch_size, use_augmentation=use_augmentation
    )

    train_loader = data["train_loader"]
    val_loader = data["val_loader"]
    train_dataset = data["train_dataset"]

    classes = data["classes"]
    class_to_idx = data["class_to_idx"]
    num_classes = data["num_classes"]

    print(f"Dataset: {dataset_name}")
    print(f"Classes: {classes}")
    print(f"Class to index: {class_to_idx}")
    print(f"Number of classes: {num_classes}")
    print(f"Pretrained ResNet18: {pretrained}")
    print(f"Data Augmentation: {use_augmentation}")
    print(f"Class Weights: {use_class_weights}")
    print()

    model = create_model(num_classes, pretrained=pretrained)
    model = model.to(device)

    class_weights = None
    if use_class_weights:
        class_weights = compute_class_weights(train_dataset, num_classes, device)
        print("Berechnete Class Weights:")
        for class_name, weight in zip(classes, class_weights.detach().cpu().tolist()):
            print(f"  {class_name}: {weight:.4f}")
        print()

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_val_f1 = 0.0
    history = []

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        print("-" * 30)

        train_loss, train_acc, train_f1 = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        val_loss, val_acc, val_f1, val_labels, val_predictions = evaluate(
            model, val_loader, criterion, device
        )

        print(
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Train F1: {train_f1:.4f}"
        )
        print(
            f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f} | Val F1:   {val_f1:.4f}"
        )
        print()

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

        # Bestes Modell speichern
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1

            checkpoint = {
                "dataset_name": dataset_name,
                "model_name": "resnet18",
                "model_state_dict": model.state_dict(),
                "classes": classes,
                "class_to_idx": class_to_idx,
                "num_classes": num_classes,
                "image_size": 224,
                "pretrained": pretrained,
                "use_augmentation": use_augmentation,
                "use_class_weights": use_class_weights,
                "class_weights": (
                    class_weights.detach().cpu().tolist()
                    if class_weights is not None
                    else None
                ),
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "best_val_f1": best_val_f1,
            }

            torch.save(checkpoint, output_model_path)
            print(f"Neues bestes Modell gespeichert: {output_model_path}")
            print()

    # Trainingshistorie speichern
    results_path = Path("results") / f"{dataset_name.lower()}_training_history.json"

    with open(results_path, "w") as f:
        json.dump(history, f, indent=4)

    print("Training abgeschlossen.")
    print(f"Bestes Val F1: {best_val_f1:.4f}")
    print(f"Training History gespeichert unter: {results_path}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--output_model_path", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--no_augmentation", action="store_true")
    parser.add_argument("--no_class_weights", action="store_true")

    args = parser.parse_args()

    train_model(
        dataset_name=args.dataset_name,
        dataset_path=args.dataset_path,
        output_model_path=args.output_model_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        pretrained=args.pretrained,
        use_augmentation=not args.no_augmentation,
        use_class_weights=not args.no_class_weights,
    )


if __name__ == "__main__":
    main()
