import argparse
import gc
import importlib
import json
import sys
import time
import traceback
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.hub
import matplotlib.pyplot as plt
from torchvision import models, utils

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_utils import create_dataloaders
from src.evaluation_manifest import (
    load_image_from_manifest_record,
    load_manifest_records,
    manifest_record_metadata,
)
from src.dvce_core import (
    DVCE_CHECKPOINT_RELATIVE_PATH,
    add_dvce_to_python_path,
    build_dvce_model_config,
    cast_diffusion_numpy_arrays_to_float32,
    generate_dvce_counterfactual,
    resolve_diffusion_checkpoint_path,
)

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def apply_dvce_compatibility_patches():
    if not hasattr(np, "float"):
        np.float = float
    if not hasattr(np, "int"):
        np.int = int

    torchvision_utils = types.ModuleType("torchvision.models.utils")
    torchvision_utils.load_state_dict_from_url = torch.hub.load_state_dict_from_url
    sys.modules["torchvision.models.utils"] = torchvision_utils

    try:
        import clip as openai_clip

        clip_package = types.ModuleType("CLIP")
        clip_package.clip = openai_clip
        sys.modules["CLIP"] = clip_package
    except Exception:
        pass


def resolve_device(device_name):
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


def load_medical_model(model_path, device):
    checkpoint = torch.load(model_path, map_location=device)
    model = create_resnet18(checkpoint["num_classes"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, checkpoint


def validate_second_classifier_checkpoint(primary_checkpoint, second_checkpoint):
    primary_classes = list(primary_checkpoint["classes"])
    second_classes = list(second_checkpoint["classes"])
    if primary_classes != second_classes:
        raise ValueError(
            "Second classifier classes must match the explained classifier. "
            f"Primary classes={primary_classes}, second classes={second_classes}"
        )
    if int(primary_checkpoint["num_classes"]) != int(second_checkpoint["num_classes"]):
        raise ValueError(
            "Second classifier num_classes must match the explained classifier. "
            f"Primary={primary_checkpoint['num_classes']}, "
            f"second={second_checkpoint['num_classes']}"
        )


class MedicalResNetAdapter(nn.Module):
    """DVCE-facing classifier adapter for project ResNet18 checkpoints.

    Mirrors the original ResizeAndMeanWrapper: bicubic resize (interpolation=3)
    to the classifier input size, normalization, no input clamping. Unclamped
    inputs keep guidance gradients alive for out-of-range pixels, exactly as in
    the original DVCE cond_fn.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.register_buffer("mean", IMAGENET_MEAN.clone())
        self.register_buffer("std", IMAGENET_STD.clone())

    def forward(self, images):
        if images.shape[-2:] != (224, 224):
            images = F.interpolate(
                images, size=(224, 224), mode="bicubic", align_corners=False
            )
        normalized = (images - self.mean.to(images.device)) / self.std.to(images.device)
        return self.model(normalized)


def denormalize(images):
    mean = IMAGENET_MEAN.to(images.device)
    std = IMAGENET_STD.to(images.device)
    return (images * std + mean).clamp(0.0, 1.0)


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
                        "image_normalized": images[idx : idx + 1].detach(),
                        "true_label_index": int(labels[idx].item()),
                        "prediction_index": int(predictions[idx].item()),
                        "probabilities": probabilities[idx].detach().cpu().tolist(),
                    }

    raise RuntimeError("No correctly classified test sample found.")


def choose_correct_samples(model, test_loader, device, num_samples):
    samples = []
    dataset_index = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            probabilities = F.softmax(logits, dim=1)
            predictions = torch.argmax(probabilities, dim=1)

            for idx in range(images.shape[0]):
                if predictions[idx] == labels[idx]:
                    samples.append(
                        {
                            "sample_index": len(samples),
                            "dataset_index": dataset_index + idx,
                            "image_normalized": images[idx : idx + 1].detach(),
                            "true_label_index": int(labels[idx].item()),
                            "prediction_index": int(predictions[idx].item()),
                            "probabilities": probabilities[idx].detach().cpu().tolist(),
                        }
                    )
                    if len(samples) >= num_samples:
                        return samples
            dataset_index += images.shape[0]

    if not samples:
        raise RuntimeError("No correctly classified test sample found.")
    return samples


def choose_manifest_samples(
    model, test_dataset, device, manifest_path, max_records=None
):
    manifest, records = load_manifest_records(manifest_path, max_records=max_records)
    samples = []

    with torch.no_grad():
        for record in records:
            image, label, image_path = load_image_from_manifest_record(
                test_dataset, record
            )
            image = image.to(device)
            logits = model(image)
            probabilities = F.softmax(logits, dim=1)
            prediction = int(torch.argmax(probabilities, dim=1).item())
            expected_prediction = int(record["original_prediction_index"])

            if prediction != expected_prediction:
                raise ValueError(
                    "Current model prediction does not match manifest for "
                    f"manifest sample {record['manifest_sample_index']}: "
                    f"manifest={expected_prediction}, current={prediction}"
                )

            samples.append(
                {
                    **manifest_record_metadata(record),
                    "sample_index": len(samples),
                    "image_normalized": image.detach(),
                    "true_label_index": int(label),
                    "prediction_index": prediction,
                    "probabilities": probabilities[0].detach().cpu().tolist(),
                    "target_class_index": int(record["target_class_index"]),
                    "target_class": record["target_class"],
                    "image_source_path": image_path,
                }
            )

    if not samples:
        raise RuntimeError("No manifest samples loaded.")
    return manifest, samples


def select_target_class(probabilities, original_class):
    ranked_classes = torch.argsort(
        torch.tensor(probabilities), descending=True
    ).tolist()
    for class_index in ranked_classes:
        if class_index != original_class:
            return int(class_index)
    return int((original_class + 1) % len(probabilities))


def tensor_stats(tensor):
    detached = tensor.detach().cpu()
    return {
        "shape": list(detached.shape),
        "min": float(detached.min().item()),
        "max": float(detached.max().item()),
        "mean": float(detached.mean().item()),
        "std": float(detached.std().item()),
    }


def run_adapter_forward(adapter, image_01):
    with torch.no_grad():
        logits = adapter(image_01)
        probabilities = F.softmax(logits, dim=1)
        prediction = int(torch.argmax(probabilities, dim=1).item())
    return {
        "logits_shape": list(logits.shape),
        "prediction_index": prediction,
        "confidence": float(probabilities[0, prediction].item()),
        "probabilities": probabilities[0].detach().cpu().tolist(),
    }


def run_classifier_guidance_gradient_check(adapter, image_01, target_class):
    candidate = image_01.clone().detach().requires_grad_(True)
    target = torch.tensor([target_class], dtype=torch.long, device=candidate.device)
    logits = adapter(candidate)
    loss = F.cross_entropy(logits, target)
    loss.backward()
    gradient = candidate.grad.detach()
    return {
        "target_loss": float(loss.detach().cpu().item()),
        "gradient_available": candidate.grad is not None,
        "gradient_stats": tensor_stats(gradient),
        "has_nonzero_gradient": bool(torch.any(gradient.abs() > 0).item()),
    }


def run_dvce_core_initialization(
    repo_path, timestep_respacing, model_output_size, checkpoint_path=None
):
    start_time = time.time()
    checkpoint_path = resolve_diffusion_checkpoint_path(repo_path, checkpoint_path)

    script_util = importlib.import_module(
        "blended_diffusion.guided_diffusion.guided_diffusion.script_util"
    )
    model_config = build_dvce_model_config(
        timestep_respacing, model_output_size, use_fp16=True
    )
    model, diffusion = script_util.create_model_and_diffusion(**model_config)
    cast_diffusion_numpy_arrays_to_float32(diffusion)
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.requires_grad_(False)
    model.eval()

    metadata = {
        "ok": True,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_exists": checkpoint_path.exists(),
        "model_output_size": model_output_size,
        "timestep_respacing": timestep_respacing,
        "diffusion_num_timesteps": int(getattr(diffusion, "num_timesteps", -1)),
        "model_parameter_count": int(sum(p.numel() for p in model.parameters())),
        "loaded_on_device": "cpu",
        "elapsed_seconds": round(time.time() - start_time, 3),
        "note": (
            "This initializes the DVCE/OpenAI diffusion backbone only. "
            "No denoising loop or counterfactual image is generated in this step."
        ),
    }

    del state_dict
    del model
    del diffusion
    gc.collect()
    return metadata


def predict_with_adapter(adapter, image_01, classes):
    with torch.no_grad():
        logits = adapter(image_01)
        probabilities = F.softmax(logits, dim=1)
        prediction_index = int(torch.argmax(probabilities, dim=1).item())
        confidence = float(probabilities[0, prediction_index].item())
    return {
        "prediction_index": prediction_index,
        "prediction": classes[prediction_index],
        "confidence": confidence,
        "probabilities": probabilities[0].detach().cpu().tolist(),
    }


def compute_difference_stats(original_01, counterfactual_01):
    diff = torch.abs(original_01 - counterfactual_01)
    flat_diff = diff.view(diff.shape[0], -1)
    changed_pixels = torch.mean((diff > 0.05).float()).item()

    # Existing project metrics are kept for compatibility with earlier DVCE runs
    # and the other counterfactual baselines.
    mean_absolute_difference = diff.mean().detach().cpu().item()
    mean_l2_distance = torch.sqrt(torch.mean(diff.pow(2))).detach().cpu().item()
    linf_distance = diff.max().detach().cpu().item()

    # DVCE-paper-style per-image vector norms. The original DVCE code reports
    # Lp norms over the flattened image difference, not mean-normalized values.
    l1_norm = flat_diff.norm(p=1, dim=1).detach().cpu()
    l1_5_norm = flat_diff.norm(p=1.5, dim=1).detach().cpu()
    l2_norm = flat_diff.norm(p=2, dim=1).detach().cpu()

    return diff, {
        "original_stats": tensor_stats(original_01),
        "counterfactual_stats": tensor_stats(counterfactual_01),
        "diff_stats": tensor_stats(diff),
        "changed_pixels_threshold_0_05": float(changed_pixels),
        "mean_absolute_difference": float(mean_absolute_difference),
        "mean_l2_distance": float(mean_l2_distance),
        "linf_distance": float(linf_distance),
        "l1_norm": float(l1_norm.mean().item()),
        "l1_5_norm": float(l1_5_norm.mean().item()),
        "l2_norm": float(l2_norm.mean().item()),
    }


def image_to_numpy(image_01):
    image = image_01.detach().cpu().squeeze(0)
    if image.shape[0] == 1:
        return image.squeeze(0).numpy()
    return image.permute(1, 2, 0).numpy()


def save_generation_visualization(
    original_01,
    counterfactual_01,
    diff,
    output_path,
    title,
):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    diff_gray = diff.detach().cpu().mean(dim=1, keepdim=True)

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.suptitle(title, fontsize=12)

    axes[0].imshow(image_to_numpy(original_01), cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(image_to_numpy(counterfactual_01), cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("DVCE candidate")
    axes[1].axis("off")

    diff_plot = axes[2].imshow(
        diff_gray.squeeze(0).squeeze(0).numpy(), cmap="gray", vmin=0, vmax=1
    )
    axes[2].set_title("Absolute difference")
    axes[2].axis("off")
    fig.colorbar(diff_plot, ax=axes[2], fraction=0.046, pad=0.04)

    axes[3].imshow(image_to_numpy(original_01), cmap="gray", vmin=0, vmax=1)
    overlay = axes[3].imshow(
        diff_gray.squeeze(0).squeeze(0).numpy(), cmap="magma", vmin=0, vmax=1, alpha=0.4
    )
    axes[3].set_title("Overlay")
    axes[3].axis("off")
    fig.colorbar(overlay, ax=axes[3], fraction=0.046, pad=0.04)

    plt.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_preview(image_01, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    resized = F.interpolate(
        image_01, size=(256, 256), mode="bilinear", align_corners=False
    )
    utils.save_image(resized, output_path)


def mean_or_none(values):
    values = [value for value in values if value is not None]
    if not values:
        return None
    return float(sum(values) / len(values))


def summarize_generation_records(records):
    successful_records = [record for record in records if record.get("ok")]
    valid_records = [
        record
        for record in successful_records
        if record.get("valid_counterfactual") is True
    ]
    return {
        "num_samples": len(records),
        "successful_generations": len(successful_records),
        "failed_generations": len(records) - len(successful_records),
        "valid_count": len(valid_records),
        "generation_success_rate": (
            len(successful_records) / len(records) if records else 0.0
        ),
        "validity": len(valid_records) / len(records) if records else 0.0,
        "validity_among_successful_generations": (
            len(valid_records) / len(successful_records) if successful_records else 0.0
        ),
        "mean_counterfactual_confidence": mean_or_none(
            [record.get("counterfactual_confidence") for record in successful_records]
        ),
        "mean_runtime_seconds": mean_or_none(
            [record.get("runtime_seconds") for record in successful_records]
        ),
        "mean_absolute_difference": mean_or_none(
            [record.get("mean_absolute_difference") for record in successful_records]
        ),
        "mean_l2_distance": mean_or_none(
            [record.get("mean_l2_distance") for record in successful_records]
        ),
        "mean_linf_distance": mean_or_none(
            [record.get("linf_distance") for record in successful_records]
        ),
        "mean_l1_norm": mean_or_none(
            [record.get("l1_norm") for record in successful_records]
        ),
        "mean_l1_5_norm": mean_or_none(
            [record.get("l1_5_norm") for record in successful_records]
        ),
        "mean_l2_norm": mean_or_none(
            [record.get("l2_norm") for record in successful_records]
        ),
        "mean_changed_pixels_threshold_0_05": mean_or_none(
            [
                record.get("changed_pixels_threshold_0_05")
                for record in successful_records
            ]
        ),
    }


def run_generation_for_sample(
    sample,
    sample_output_dir,
    repo_path,
    adapter,
    second_adapter,
    classes,
    device,
    args,
):
    original_class = sample["prediction_index"]
    target_class = sample.get(
        "target_class_index",
        select_target_class(sample["probabilities"], original_class),
    )
    image_01 = denormalize(sample["image_normalized"]).to(device)
    image_256 = F.interpolate(
        image_01,
        size=(args.model_output_size, args.model_output_size),
        mode="bilinear",
        align_corners=False,
    )

    sample_output_dir.mkdir(parents=True, exist_ok=True)
    preview_path = sample_output_dir / "dvce_medical_input_256.png"
    save_preview(image_01.detach().cpu(), preview_path)

    adapter_forward = run_adapter_forward(adapter, image_256)
    gradient_check = run_classifier_guidance_gradient_check(
        adapter, image_256, target_class
    )

    record_base = {
        "sample_index": sample["sample_index"],
        "dataset_index": sample["dataset_index"],
        **{
            key: sample[key]
            for key in [
                "manifest_sample_index",
                "source_image_path",
            ]
            if key in sample
        },
        "true_label_index": sample["true_label_index"],
        "true_label": classes[sample["true_label_index"]],
        "original_prediction_index": original_class,
        "original_prediction": classes[original_class],
        "original_confidence_before_resize": float(
            sample["probabilities"][original_class]
        ),
        "target_class_index": target_class,
        "target_class": classes[target_class],
        "target_initial_confidence_before_resize": float(
            sample["probabilities"][target_class]
        ),
        "adapter_forward_check": adapter_forward,
        "classifier_guidance_gradient_check": gradient_check,
        "second_classifier_path": args.second_model_path,
        "cone_projection_requested": args.deg_cone_projection > 0,
        "input_01_stats": tensor_stats(image_01),
        "input_256_stats": tensor_stats(image_256),
        "sample_preview_path": str(preview_path),
    }

    try:
        counterfactual_01, generation_settings = generate_dvce_counterfactual(
            repo_path=repo_path,
            classifier=adapter,
            original_image_01=image_01,
            target_class=target_class,
            device=device,
            model_output_size=args.model_output_size,
            timestep_respacing=args.timestep_respacing,
            skip_timesteps=args.skip_timesteps,
            use_ddim=args.use_ddim,
            use_fp16=args.diffusion_fp16,
            seed=args.seed + sample["sample_index"],
            diffusion_checkpoint_path=args.diffusion_checkpoint_path,
            classifier_lambda=args.classifier_lambda,
            lp_custom=args.lp_custom,
            lp_custom_value=args.lp_custom_value,
            enforce_same_norms=args.enforce_same_norms,
            denoise_dist_input=args.denoise_dist_input,
            aug_num=args.aug_num,
            clip_denoised=args.clip_denoised,
            deg_cone_projection=args.deg_cone_projection,
            second_classifier=second_adapter,
        )
        original_256 = F.interpolate(
            image_01,
            size=(args.model_output_size, args.model_output_size),
            mode="bilinear",
            align_corners=False,
        )
        counterfactual_prediction = predict_with_adapter(
            adapter, counterfactual_01.to(device), classes
        )
        original_prediction = predict_with_adapter(
            adapter, original_256.to(device), classes
        )
        diff, difference_metadata = compute_difference_stats(
            original_256.detach().cpu(), counterfactual_01.detach().cpu()
        )

        counterfactual_path = sample_output_dir / "dvce_counterfactual_candidate.png"
        diff_path = sample_output_dir / "dvce_absolute_difference.png"
        visualization_path = sample_output_dir / "dvce_counterfactual_visualization.png"
        utils.save_image(counterfactual_01.detach().cpu(), counterfactual_path)
        utils.save_image(diff.detach().cpu(), diff_path)

        valid_counterfactual = (
            counterfactual_prediction["prediction_index"] == target_class
        )
        visualization_title = (
            f"Target: {classes[original_class]} -> {classes[target_class]}\n"
            f"Prediction: {original_prediction['prediction']} "
            f"({original_prediction['confidence']:.2f}) -> "
            f"{counterfactual_prediction['prediction']} "
            f"({counterfactual_prediction['confidence']:.2f})\n"
            f"Valid CF: {'yes' if valid_counterfactual else 'no'}"
        )
        save_generation_visualization(
            original_256.detach().cpu(),
            counterfactual_01.detach().cpu(),
            diff,
            visualization_path,
            visualization_title,
        )

        record = {
            **record_base,
            "ok": True,
            "error": None,
            **generation_settings,
            "original_prediction_index": original_prediction["prediction_index"],
            "original_prediction": original_prediction["prediction"],
            "original_confidence": original_prediction["confidence"],
            "target_class_index": target_class,
            "target_class": classes[target_class],
            "counterfactual_prediction_index": counterfactual_prediction[
                "prediction_index"
            ],
            "counterfactual_prediction": counterfactual_prediction["prediction"],
            "counterfactual_confidence": counterfactual_prediction["confidence"],
            "valid_counterfactual": valid_counterfactual,
            "original_probabilities": original_prediction["probabilities"],
            "counterfactual_probabilities": counterfactual_prediction["probabilities"],
            "counterfactual_path": str(counterfactual_path),
            "difference_path": str(diff_path),
            "visualization_path": str(visualization_path),
            **difference_metadata,
        }
    except Exception as error:
        record = {
            **record_base,
            "ok": False,
            "error": f"{type(error).__name__}: {error}",
            "error_traceback": traceback.format_exc(),
            "valid_counterfactual": False,
        }

    with open(sample_output_dir / "metadata.json", "w") as f:
        json.dump(record, f, indent=4)

    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dvce_repo", type=str, default="external/DVCEs")
    parser.add_argument(
        "--diffusion_checkpoint_path",
        type=str,
        default=None,
        help="Optional diffusion checkpoint override. Defaults to the DVCE OpenAI checkpoint.",
    )
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument(
        "--second_model_path",
        "--second_classifier_path",
        dest="second_model_path",
        type=str,
        default=None,
        help=(
            "Optional robust second classifier checkpoint for DVCE Cone "
            "Projection. Must use the same classes/checkpoint format as "
            "--model_path."
        ),
    )
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--timestep_respacing", type=str, default="200")
    parser.add_argument("--skip_timesteps", type=int, default=100)
    parser.add_argument("--model_output_size", type=int, default=256)
    parser.add_argument(
        "--skip_diffusion_core",
        action="store_true",
        help=(
            "Skip only the preliminary diffusion initialization check. "
            "Actual generation still uses src.dvce_core when --run_generation is set."
        ),
    )
    parser.add_argument("--run_generation", action="store_true")
    parser.add_argument("--classifier_lambda", type=float, default=0.1)
    parser.add_argument("--lp_custom", type=float, default=1.0)
    parser.add_argument("--lp_custom_value", type=float, default=0.15)
    parser.add_argument(
        "--enforce_same_norms",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--denoise_dist_input", action="store_true")
    parser.add_argument("--deg_cone_projection", type=float, default=0.0)
    parser.add_argument("--aug_num", type=int, default=1)
    parser.add_argument(
        "--clip_denoised",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--use_ddim", action="store_true")
    parser.add_argument("--diffusion_fp16", action="store_true")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num_generation_samples", type=int, default=1)
    parser.add_argument(
        "--manifest_path",
        type=str,
        default=None,
        help="Optional fixed evaluation manifest. If set, samples and targets come from this JSON.",
    )
    parser.add_argument(
        "--manifest_max_samples",
        type=int,
        default=None,
        help="Optional cap for manifest mode. By default all manifest records are used.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    repo_path = Path(args.dvce_repo).resolve()
    diffusion_checkpoint_path = resolve_diffusion_checkpoint_path(
        repo_path, args.diffusion_checkpoint_path
    )
    add_dvce_to_python_path(repo_path)
    apply_dvce_compatibility_patches()

    device = resolve_device(args.device)
    model, checkpoint = load_medical_model(args.model_path, device)
    classes = checkpoint["classes"]
    second_model = None
    second_checkpoint = None
    second_adapter = None
    if args.second_model_path:
        second_model, second_checkpoint = load_medical_model(
            args.second_model_path, device
        )
        validate_second_classifier_checkpoint(checkpoint, second_checkpoint)
        second_adapter = MedicalResNetAdapter(second_model).to(device)
        second_adapter.eval()
    if args.deg_cone_projection > 0 and second_adapter is None:
        raise ValueError(
            "--deg_cone_projection > 0 requires --second_model_path/"
            "--second_classifier_path so Cone Projection is actually defined."
        )
    cone_projection_enabled = second_adapter is not None and args.deg_cone_projection > 0
    dvce_variant = (
        "original_style_medical_cone_projection"
        if cone_projection_enabled
        else "original_style_medical_no_cone"
    )
    data = create_dataloaders(
        args.dataset_path, batch_size=args.batch_size, use_augmentation=False
    )
    manifest = None
    manifest_samples = None
    if args.manifest_path:
        print(f"Loading fixed evaluation manifest: {args.manifest_path}")
        manifest, manifest_samples = choose_manifest_samples(
            model=model,
            test_dataset=data["test_dataset"],
            device=device,
            manifest_path=args.manifest_path,
            max_records=args.manifest_max_samples,
        )
        sample = manifest_samples[0]
    else:
        sample = choose_correct_sample(model, data["test_loader"], device)

    original_class = sample["prediction_index"]
    target_class = sample.get(
        "target_class_index",
        select_target_class(sample["probabilities"], original_class),
    )
    image_01 = denormalize(sample["image_normalized"]).to(device)
    image_256 = F.interpolate(
        image_01,
        size=(args.model_output_size, args.model_output_size),
        mode="bilinear",
        align_corners=False,
    )

    adapter = MedicalResNetAdapter(model).to(device)
    adapter.eval()

    preview_path = output_dir / "dvce_medical_input_256.png"
    save_preview(image_01.detach().cpu(), preview_path)

    adapter_forward = run_adapter_forward(adapter, image_256)
    gradient_check = run_classifier_guidance_gradient_check(
        adapter, image_256, target_class
    )

    diffusion_core = {
        "ok": False,
        "skipped": True,
        "reason": "Skipped by --skip_diffusion_core.",
    }
    if not args.skip_diffusion_core:
        diffusion_core = run_dvce_core_initialization(
            repo_path=repo_path,
            timestep_respacing=args.timestep_respacing,
            model_output_size=args.model_output_size,
            checkpoint_path=args.diffusion_checkpoint_path,
        )

    if args.run_generation and (args.num_generation_samples > 1 or args.manifest_path):
        if manifest_samples is not None:
            samples = manifest_samples
        else:
            samples = choose_correct_samples(
                model=model,
                test_loader=data["test_loader"],
                device=device,
                num_samples=args.num_generation_samples,
            )
        records = []
        for current_sample in samples:
            sample_output_dir = (
                output_dir / f"sample_{current_sample['sample_index']:02d}"
            )
            print(
                "Running DVCE sample "
                f"{current_sample['sample_index'] + 1}/{len(samples)} "
                f"(dataset index {current_sample['dataset_index']})..."
            )
            record = run_generation_for_sample(
                sample=current_sample,
                sample_output_dir=sample_output_dir,
                repo_path=repo_path,
                adapter=adapter,
                second_adapter=second_adapter,
                classes=classes,
                device=device,
                args=args,
            )
            records.append(record)
            status_text = "valid" if record.get("valid_counterfactual") else "not valid"
            print(
                "Sample result: "
                f"{status_text}; "
                f"{record.get('original_prediction')} -> "
                f"{record.get('counterfactual_prediction', 'generation failed')}"
            )

        metadata = {
            "purpose": "DVCE medical multi-sample generation evaluation",
            "status": "multi_sample_generation_completed",
            "dvce_variant": dvce_variant,
            "guidance_core": "pred_xstart_eps_norm_lp_custom",
            "cone_projection_requested": args.deg_cone_projection > 0,
            "cone_projection_enabled": cone_projection_enabled,
            "python": sys.version,
            "device": str(device),
            "dvce_repo": str(repo_path),
            "dvce_default_checkpoint": str(repo_path / DVCE_CHECKPOINT_RELATIVE_PATH),
            "diffusion_checkpoint_path": str(diffusion_checkpoint_path),
            "runner_settings": {
                "timestep_respacing": args.timestep_respacing,
                "skip_timesteps": args.skip_timesteps,
                "model_output_size": args.model_output_size,
                "batch_size": args.batch_size,
                "num_generation_samples": args.num_generation_samples,
                "manifest_path": args.manifest_path,
                "manifest_max_samples": args.manifest_max_samples,
                "classifier_lambda": args.classifier_lambda,
                "lp_custom": args.lp_custom,
                "lp_custom_value": args.lp_custom_value,
                "enforce_same_norms": args.enforce_same_norms,
                "gen_type": "ddim" if args.use_ddim else "p_sample",
                "clip_denoised": args.clip_denoised,
                "denoise_dist_input": args.denoise_dist_input,
                "deg_cone_projection": args.deg_cone_projection,
                "aug_num": args.aug_num,
                "use_ddim": args.use_ddim,
                "diffusion_fp16": args.diffusion_fp16,
                "seed": args.seed,
                "diffusion_checkpoint_path": str(diffusion_checkpoint_path),
                "diffusion_checkpoint_arg": args.diffusion_checkpoint_path,
                "second_model_path": args.second_model_path,
            },
            "classifier_adapter": {
                "model_path": args.model_path,
                "dataset_path": args.dataset_path,
                "classes": classes,
            },
            "second_classifier_adapter": {
                "model_path": args.second_model_path,
                "classes": (
                    second_checkpoint["classes"] if second_checkpoint else None
                ),
                "num_classes": (
                    second_checkpoint["num_classes"] if second_checkpoint else None
                ),
                "role": (
                    "robust classifier for Cone Projection"
                    if cone_projection_enabled
                    else None
                ),
            },
            "evaluation_manifest": {
                "path": args.manifest_path,
                "num_records_available": (
                    manifest.get("num_samples") if manifest else None
                ),
                "num_records_used": len(samples) if manifest else None,
                "target_strategy": (
                    manifest.get("target_strategy") if manifest else None
                ),
            },
            "dvce_core_initialization": diffusion_core,
            "aggregate_metrics": summarize_generation_records(records),
            "records": records,
        }
        metadata_path = output_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)

        print(f"Saved multi-sample DVCE metadata to {metadata_path}")
        print(json.dumps(metadata["aggregate_metrics"], indent=4))
        return

    generation_result = {
        "attempted": False,
        "ok": False,
        "error": None,
    }
    if args.run_generation:
        try:
            counterfactual_01, generation_settings = generate_dvce_counterfactual(
                repo_path=repo_path,
                classifier=adapter,
                original_image_01=image_01,
                target_class=target_class,
                device=device,
                model_output_size=args.model_output_size,
                timestep_respacing=args.timestep_respacing,
                skip_timesteps=args.skip_timesteps,
                use_ddim=args.use_ddim,
                use_fp16=args.diffusion_fp16,
                seed=args.seed,
                diffusion_checkpoint_path=args.diffusion_checkpoint_path,
                classifier_lambda=args.classifier_lambda,
                lp_custom=args.lp_custom,
                lp_custom_value=args.lp_custom_value,
                enforce_same_norms=args.enforce_same_norms,
                denoise_dist_input=args.denoise_dist_input,
                aug_num=args.aug_num,
                clip_denoised=args.clip_denoised,
                deg_cone_projection=args.deg_cone_projection,
                second_classifier=second_adapter,
            )
            original_256 = F.interpolate(
                image_01,
                size=(args.model_output_size, args.model_output_size),
                mode="bilinear",
                align_corners=False,
            )
            counterfactual_prediction = predict_with_adapter(
                adapter, counterfactual_01.to(device), classes
            )
            original_prediction = predict_with_adapter(
                adapter, original_256.to(device), classes
            )
            diff, difference_metadata = compute_difference_stats(
                original_256.detach().cpu(), counterfactual_01.detach().cpu()
            )

            counterfactual_path = output_dir / "dvce_counterfactual_candidate.png"
            diff_path = output_dir / "dvce_absolute_difference.png"
            visualization_path = output_dir / "dvce_counterfactual_visualization.png"
            utils.save_image(counterfactual_01.detach().cpu(), counterfactual_path)
            utils.save_image(diff.detach().cpu(), diff_path)

            valid_counterfactual = (
                counterfactual_prediction["prediction_index"] == target_class
            )
            visualization_title = (
                f"Target: {classes[original_class]} -> {classes[target_class]}\n"
                f"Prediction: {original_prediction['prediction']} "
                f"({original_prediction['confidence']:.2f}) -> "
                f"{counterfactual_prediction['prediction']} "
                f"({counterfactual_prediction['confidence']:.2f})\n"
                f"Valid CF: {'yes' if valid_counterfactual else 'no'}"
            )
            save_generation_visualization(
                original_256.detach().cpu(),
                counterfactual_01.detach().cpu(),
                diff,
                visualization_path,
                visualization_title,
            )

            generation_result = {
                "attempted": True,
                "ok": True,
                "error": None,
                **generation_settings,
                "true_label_index": sample["true_label_index"],
                "true_label": classes[sample["true_label_index"]],
                "original_prediction_index": original_prediction["prediction_index"],
                "original_prediction": original_prediction["prediction"],
                "original_confidence": original_prediction["confidence"],
                "target_class_index": target_class,
                "target_class": classes[target_class],
                "counterfactual_prediction_index": counterfactual_prediction[
                    "prediction_index"
                ],
                "counterfactual_prediction": counterfactual_prediction["prediction"],
                "counterfactual_confidence": counterfactual_prediction["confidence"],
                "valid_counterfactual": valid_counterfactual,
                "original_probabilities": original_prediction["probabilities"],
                "counterfactual_probabilities": counterfactual_prediction[
                    "probabilities"
                ],
                "counterfactual_path": str(counterfactual_path),
                "difference_path": str(diff_path),
                "visualization_path": str(visualization_path),
                **difference_metadata,
            }
        except Exception as error:
            generation_result = {
                "attempted": True,
                "ok": False,
                "error": f"{type(error).__name__}: {error}",
            }

    if generation_result.get("ok"):
        status = "single_image_generation_completed"
        next_step = (
            "Tune guidance and noise settings on a small fixed sample set, then compare "
            "validity, visual plausibility, changed pixels, and runtime against the "
            "prototype-guided baseline and SEDC-T."
        )
    elif generation_result.get("attempted"):
        status = "single_image_generation_failed"
        next_step = (
            "Inspect the generation error and retry with safer device/noise settings."
        )
    elif diffusion_core.get("ok"):
        status = "steps_1_to_3_completed"
        next_step = (
            "Run a single targeted DVCE generation attempt and save original, generated "
            "image, difference map, prediction metadata, and validity."
        )
    else:
        status = "partial"
        next_step = (
            "Fix the failed adapter, gradient, or diffusion initialization check."
        )

    metadata = {
        "purpose": "DVCE original-style medical single-image smoke test",
        "status": status,
        "dvce_variant": dvce_variant,
        "guidance_core": "pred_xstart_eps_norm_lp_custom",
        "cone_projection_requested": args.deg_cone_projection > 0,
        "cone_projection_enabled": cone_projection_enabled,
        "python": sys.version,
        "device": str(device),
        "dvce_repo": str(repo_path),
        "dvce_default_checkpoint": str(repo_path / DVCE_CHECKPOINT_RELATIVE_PATH),
        "diffusion_checkpoint_path": str(diffusion_checkpoint_path),
        "runner_settings": {
            "timestep_respacing": args.timestep_respacing,
            "skip_timesteps": args.skip_timesteps,
            "model_output_size": args.model_output_size,
            "batch_size": args.batch_size,
            "classifier_lambda": args.classifier_lambda,
            "lp_custom": args.lp_custom,
            "lp_custom_value": args.lp_custom_value,
            "enforce_same_norms": args.enforce_same_norms,
            "gen_type": "ddim" if args.use_ddim else "p_sample",
            "clip_denoised": args.clip_denoised,
            "denoise_dist_input": args.denoise_dist_input,
            "deg_cone_projection": args.deg_cone_projection,
            "aug_num": args.aug_num,
            "diffusion_checkpoint_path": str(diffusion_checkpoint_path),
            "diffusion_checkpoint_arg": args.diffusion_checkpoint_path,
            "second_model_path": args.second_model_path,
        },
        "classifier_adapter": {
            "model_path": args.model_path,
            "dataset_path": args.dataset_path,
            "classes": classes,
            "true_label_index": sample["true_label_index"],
            "true_label": classes[sample["true_label_index"]],
            "original_prediction_index": original_class,
            "original_prediction": classes[original_class],
            "original_confidence": float(sample["probabilities"][original_class]),
            "target_class_index": target_class,
            "target_class": classes[target_class],
            "target_initial_confidence": float(sample["probabilities"][target_class]),
            "input_01_stats": tensor_stats(image_01),
            "input_256_stats": tensor_stats(image_256),
            "sample_preview_path": str(preview_path),
        },
        "second_classifier_adapter": {
            "model_path": args.second_model_path,
            "classes": second_checkpoint["classes"] if second_checkpoint else None,
            "num_classes": (
                second_checkpoint["num_classes"] if second_checkpoint else None
            ),
            "role": (
                "robust classifier for Cone Projection"
                if cone_projection_enabled
                else None
            ),
        },
        "adapter_forward_check": adapter_forward,
        "classifier_guidance_gradient_check": gradient_check,
        "dvce_core_initialization": diffusion_core,
        "single_image_generation": generation_result,
        "next_step_after_success": next_step,
    }

    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"Saved DVCE original-style medical metadata to {metadata_path}")
    print(
        "Adapter sample: "
        f"{classes[original_class]} ({sample['probabilities'][original_class]:.3f}) -> "
        f"target {classes[target_class]} "
        f"(initial {sample['probabilities'][target_class]:.3f})"
    )
    print(
        "Gradient check: "
        f"{'ok' if gradient_check['has_nonzero_gradient'] else 'failed'}; "
        f"DVCE core init: {'ok' if diffusion_core.get('ok') else 'not ok'}"
    )
    if generation_result["attempted"]:
        if generation_result["ok"]:
            print(
                "Generation: ok; "
                f"valid={generation_result['valid_counterfactual']}; "
                f"prediction={generation_result['counterfactual_prediction']} "
                f"({generation_result['counterfactual_confidence']:.3f})"
            )
        else:
            print(f"Generation failed: {generation_result['error']}")


if __name__ == "__main__":
    main()
