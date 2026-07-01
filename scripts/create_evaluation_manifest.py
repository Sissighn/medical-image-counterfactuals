import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_utils import create_dataloaders


def get_device(device_name):
    if device_name != "auto":
        return torch.device(device_name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def create_resnet18(num_classes):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def load_model(model_path, device):
    checkpoint = torch.load(model_path, map_location=device)
    model = create_resnet18(checkpoint["num_classes"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint


def select_target_class(probabilities, original_class, strategy):
    ranked = torch.argsort(torch.tensor(probabilities), descending=True).tolist()
    if strategy == "second_best":
        for class_index in ranked:
            if class_index != original_class:
                return int(class_index)
    if strategy == "next_class":
        return int((original_class + 1) % len(probabilities))
    raise ValueError(f"Unknown target strategy: {strategy}")


def build_manifest(model, data, classes, device, max_samples, target_strategy):
    test_loader = data["test_loader"]
    test_dataset = data["test_dataset"]
    records = []
    dataset_index = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            probabilities = F.softmax(logits, dim=1)
            predictions = torch.argmax(probabilities, dim=1)

            for batch_index in range(images.shape[0]):
                current_dataset_index = dataset_index + batch_index
                true_label_index = int(labels[batch_index].item())
                prediction_index = int(predictions[batch_index].item())
                sample_probabilities = probabilities[batch_index].detach().cpu().tolist()

                if prediction_index != true_label_index:
                    continue

                target_class_index = select_target_class(
                    sample_probabilities, prediction_index, target_strategy
                )
                image_path, _ = test_dataset.samples[current_dataset_index]
                records.append(
                    {
                        "manifest_sample_index": len(records),
                        "dataset_index": current_dataset_index,
                        "image_path": image_path,
                        "true_label_index": true_label_index,
                        "true_label": classes[true_label_index],
                        "original_prediction_index": prediction_index,
                        "original_prediction": classes[prediction_index],
                        "original_confidence": float(
                            sample_probabilities[prediction_index]
                        ),
                        "target_class_index": target_class_index,
                        "target_class": classes[target_class_index],
                        "target_initial_confidence": float(
                            sample_probabilities[target_class_index]
                        ),
                        "probabilities": sample_probabilities,
                    }
                )
                if len(records) >= max_samples:
                    return records

            dataset_index += images.shape[0]

    return records


def build_balanced_manifest(
    model,
    data,
    classes,
    device,
    samples_per_class,
    target_strategy,
):
    test_loader = data["test_loader"]
    test_dataset = data["test_dataset"]
    records = []
    counts_by_class = {class_index: 0 for class_index in range(len(classes))}
    dataset_index = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            probabilities = F.softmax(logits, dim=1)
            predictions = torch.argmax(probabilities, dim=1)

            for batch_index in range(images.shape[0]):
                current_dataset_index = dataset_index + batch_index
                true_label_index = int(labels[batch_index].item())
                prediction_index = int(predictions[batch_index].item())
                sample_probabilities = probabilities[batch_index].detach().cpu().tolist()

                if prediction_index != true_label_index:
                    continue
                if counts_by_class[true_label_index] >= samples_per_class:
                    continue

                target_class_index = select_target_class(
                    sample_probabilities, prediction_index, target_strategy
                )
                image_path, _ = test_dataset.samples[current_dataset_index]
                records.append(
                    {
                        "manifest_sample_index": len(records),
                        "dataset_index": current_dataset_index,
                        "image_path": image_path,
                        "true_label_index": true_label_index,
                        "true_label": classes[true_label_index],
                        "original_prediction_index": prediction_index,
                        "original_prediction": classes[prediction_index],
                        "original_confidence": float(
                            sample_probabilities[prediction_index]
                        ),
                        "target_class_index": target_class_index,
                        "target_class": classes[target_class_index],
                        "target_initial_confidence": float(
                            sample_probabilities[target_class_index]
                        ),
                        "probabilities": sample_probabilities,
                    }
                )
                counts_by_class[true_label_index] += 1

                if all(count >= samples_per_class for count in counts_by_class.values()):
                    return records, counts_by_class

            dataset_index += images.shape[0]

    return records, counts_by_class


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--max_samples", type=int, default=20)
    parser.add_argument(
        "--samples_per_class",
        type=int,
        default=None,
        help="If set, selects up to this many correctly classified samples per class.",
    )
    parser.add_argument(
        "--target_strategy",
        choices=["second_best", "next_class"],
        default="second_best",
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    device = get_device(args.device)
    model, checkpoint = load_model(args.model_path, device)
    classes = checkpoint["classes"]
    data = create_dataloaders(
        args.dataset_path, batch_size=args.batch_size, use_augmentation=False
    )

    counts_by_class = None
    if args.samples_per_class is not None:
        records, counts_by_class = build_balanced_manifest(
            model=model,
            data=data,
            classes=classes,
            device=device,
            samples_per_class=args.samples_per_class,
            target_strategy=args.target_strategy,
        )
    else:
        records = build_manifest(
            model=model,
            data=data,
            classes=classes,
            device=device,
            max_samples=args.max_samples,
            target_strategy=args.target_strategy,
        )

    manifest = {
        "purpose": "fixed correctly classified test sample set for counterfactual evaluation",
        "model_path": args.model_path,
        "dataset_path": args.dataset_path,
        "classes": classes,
        "class_to_idx": checkpoint["class_to_idx"],
        "max_samples": args.max_samples,
        "num_samples": len(records),
        "target_strategy": args.target_strategy,
        "samples_per_class": args.samples_per_class,
        "counts_by_class": {
            classes[class_index]: count
            for class_index, count in (counts_by_class or {}).items()
        },
        "records": records,
    }

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(manifest, f, indent=4)

    print(f"Saved evaluation manifest with {len(records)} samples to {output_path}")


if __name__ == "__main__":
    main()
