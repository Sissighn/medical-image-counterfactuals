import argparse

import torch
import torch.nn as nn
from torchvision import models

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


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)

    args = parser.parse_args()

    device = get_device()

    checkpoint = torch.load(args.model_path, map_location=device)

    num_classes = checkpoint["num_classes"]
    classes = checkpoint["classes"]

    model = create_model(num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    data = create_dataloaders(args.dataset_path, batch_size=8)
    test_loader = data["test_loader"]

    images, labels = next(iter(test_loader))
    images = images.to(device)

    with torch.no_grad():
        outputs = model(images)
        predictions = torch.argmax(outputs, dim=1)

    print("Modell wurde erfolgreich geladen.")
    print("Klassen:", classes)
    print("Echte Labels:      ", labels.tolist())
    print("Vorhergesagte IDs: ", predictions.cpu().tolist())
    print("Vorhergesagte Klassen:")

    for prediction in predictions.cpu().tolist():
        print("  ", classes[prediction])


if __name__ == "__main__":
    main()
