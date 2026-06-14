import argparse
from pathlib import Path
import json

import torch
import torch.nn as nn
from torchvision import models

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
)

from src.data_utils import create_dataloaders


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def create_model(num_classes):
    model = models.resnet18(weights=None)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model


def evaluate_model(model, test_loader, device):
    model.eval()

    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())

    return all_labels, all_predictions


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)

    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    checkpoint = torch.load(args.model_path, map_location=device)

    num_classes = checkpoint["num_classes"]
    classes = checkpoint["classes"]
    class_to_idx = checkpoint["class_to_idx"]

    print("Modellinformationen:")
    print(f"  Klassen: {classes}")
    print(f"  Class-to-index: {class_to_idx}")
    print(f"  Anzahl Klassen: {num_classes}")
    print()

    model = create_model(num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    data = create_dataloaders(args.dataset_path, batch_size=16)
    test_loader = data["test_loader"]

    true_labels, predicted_labels = evaluate_model(model, test_loader, device)

    accuracy = accuracy_score(true_labels, predicted_labels)
    f1_weighted = f1_score(
        true_labels, predicted_labels, average="weighted", zero_division=0
    )
    precision_weighted = precision_score(
        true_labels, predicted_labels, average="weighted", zero_division=0
    )
    recall_weighted = recall_score(
        true_labels, predicted_labels, average="weighted", zero_division=0
    )

    report = classification_report(
        true_labels, predicted_labels, target_names=classes, zero_division=0
    )

    matrix = confusion_matrix(true_labels, predicted_labels)

    print("=" * 60)
    print("TEST-EVALUATION")
    print("=" * 60)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"F1 Score:  {f1_weighted:.4f}")
    print(f"Precision: {precision_weighted:.4f}")
    print(f"Recall:    {recall_weighted:.4f}")
    print()

    print("Classification Report:")
    print(report)

    print("Confusion Matrix:")
    print(matrix)
    print()

    results = {
        "model_path": args.model_path,
        "dataset_path": args.dataset_path,
        "classes": classes,
        "class_to_idx": class_to_idx,
        "accuracy": accuracy,
        "f1_weighted": f1_weighted,
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
    }

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Evaluation gespeichert unter: {output_path}")


if __name__ == "__main__":
    main()
