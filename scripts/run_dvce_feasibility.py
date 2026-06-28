import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, utils

from src.data_utils import create_dataloaders


DVCE_REPO_URL = "https://github.com/valentyn1boreiko/DVCEs.git"
DVCE_REQUIRED_FILES = [
    "imagenet_VCEs.py",
    "blended_diffusion/main.py",
    "configs/default.yml",
    "configs/blended.yml",
    "python_38_dvces.yml",
]
DVCE_DIFFUSION_CHECKPOINT = "checkpoints/256x256_diffusion_uncond.pt"

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def create_model(num_classes):
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def load_checkpoint_model(model_path, device):
    checkpoint = torch.load(model_path, map_location=device)
    model = create_model(checkpoint["num_classes"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model, checkpoint


def denormalize(images):
    mean = IMAGENET_MEAN.to(images.device)
    std = IMAGENET_STD.to(images.device)
    return (images * std + mean).clamp(0.0, 1.0)


def clone_dvce_repo(repo_path):
    repo_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--depth", "1", DVCE_REPO_URL, str(repo_path)],
        check=True,
    )


def check_dvce_repo(repo_path):
    repo_path = Path(repo_path)
    missing_files = [
        required_file
        for required_file in DVCE_REQUIRED_FILES
        if not (repo_path / required_file).exists()
    ]
    return {
        "repo_path": str(repo_path),
        "exists": repo_path.exists(),
        "required_files_present": not missing_files,
        "missing_files": missing_files,
        "diffusion_checkpoint_exists": (repo_path / DVCE_DIFFUSION_CHECKPOINT).exists(),
        "diffusion_checkpoint_path": str(repo_path / DVCE_DIFFUSION_CHECKPOINT),
    }


def check_python_modules():
    module_names = [
        "torch",
        "torchvision",
        "numpy",
        "yaml",
        "skimage",
        "lpips",
        "blobfile",
        "robustbench",
        "timm",
        "kornia",
    ]
    status = {}
    for module_name in module_names:
        try:
            __import__(module_name)
            status[module_name] = True
        except Exception:
            status[module_name] = False
    return status


def choose_correct_sample(model, test_loader, device):
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            probabilities = F.softmax(logits, dim=1)
            predictions = torch.argmax(probabilities, dim=1)

            for idx in range(images.shape[0]):
                if predictions[idx] == labels[idx]:
                    return {
                        "image": images[idx : idx + 1].detach(),
                        "true_label_index": int(labels[idx].item()),
                        "prediction_index": int(predictions[idx].item()),
                        "probabilities": probabilities[idx].detach().cpu().tolist(),
                    }

    raise RuntimeError("No correctly classified test sample found.")


def select_target(probabilities, original_class):
    ranked = torch.argsort(torch.tensor(probabilities), descending=True).tolist()
    for class_idx in ranked:
        if class_idx != original_class:
            return int(class_idx)
    return int((original_class + 1) % len(probabilities))


def save_sample_preview(sample_pixels, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    resized = F.interpolate(sample_pixels, size=(256, 256), mode="bilinear", align_corners=False)
    utils.save_image(resized, output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dvce_repo", type=str, default="external/DVCEs")
    parser.add_argument("--clone_if_missing", action="store_true")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="results/dvce_feasibility")
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    repo_path = Path(args.dvce_repo)
    if args.clone_if_missing and not repo_path.exists():
        print(f"Cloning DVCE repository into {repo_path}...")
        clone_dvce_repo(repo_path)

    device = get_device()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, checkpoint = load_checkpoint_model(args.model_path, device)
    classes = checkpoint["classes"]
    data = create_dataloaders(
        args.dataset_path, batch_size=args.batch_size, use_augmentation=False
    )
    sample = choose_correct_sample(model, data["test_loader"], device)

    original_class = sample["prediction_index"]
    target_class = select_target(sample["probabilities"], original_class)
    original_confidence = float(sample["probabilities"][original_class])
    target_initial_confidence = float(sample["probabilities"][target_class])

    sample_pixels = denormalize(sample["image"])
    sample_path = output_dir / "dvce_input_sample_256.png"
    save_sample_preview(sample_pixels, sample_path)

    repo_status = check_dvce_repo(repo_path)
    module_status = check_python_modules()

    metadata = {
        "purpose": "DVCE integration feasibility smoke test",
        "python": sys.version,
        "device": str(device),
        "dvce_repo": repo_status,
        "python_modules": module_status,
        "classifier_adapter": {
            "model_path": args.model_path,
            "dataset_path": args.dataset_path,
            "classes": classes,
            "true_label_index": sample["true_label_index"],
            "true_label": classes[sample["true_label_index"]],
            "original_prediction_index": original_class,
            "original_prediction": classes[original_class],
            "original_confidence": original_confidence,
            "target_class_index": target_class,
            "target_class": classes[target_class],
            "target_initial_confidence": target_initial_confidence,
            "sample_export_path": str(sample_path),
        },
        "next_required_steps": [
            "Create a separate DVCE-compatible environment if missing modules are required.",
            "Download the OpenAI 256x256 diffusion checkpoint into the DVCE repo checkpoints folder.",
            "Replace ImageNet dataset and classifier loading in the DVCE code with this project adapter.",
            "Run a one-image targeted generation attempt and save comparable metadata/visualizations.",
        ],
    }

    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)

    missing_modules = [name for name, available in module_status.items() if not available]

    print(f"Saved feasibility metadata to {metadata_path}")
    print(f"Saved 256x256 input sample to {sample_path}")
    print(f"DVCE repo present: {repo_status['exists']}")
    print(f"DVCE required files present: {repo_status['required_files_present']}")
    print(f"DVCE diffusion checkpoint present: {repo_status['diffusion_checkpoint_exists']}")
    print(f"Missing optional/required DVCE modules in current env: {missing_modules}")
    print(
        "Classifier adapter sample: "
        f"{classes[original_class]} ({original_confidence:.3f}) -> "
        f"{classes[target_class]} initial confidence {target_initial_confidence:.3f}"
    )


if __name__ == "__main__":
    main()
