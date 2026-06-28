import argparse
import gc
import importlib
import json
import sys
import time
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


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
DVCE_CHECKPOINT_RELATIVE_PATH = Path("checkpoints") / "256x256_diffusion_uncond.pt"


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


def add_dvce_to_python_path(repo_path):
    repo_path = Path(repo_path).resolve()
    paths = [
        repo_path,
        repo_path / "blended_diffusion",
        repo_path / "blended_diffusion" / "guided_diffusion",
    ]
    for path in paths:
        path_string = str(path)
        if path_string not in sys.path:
            sys.path.insert(0, path_string)


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


class MedicalResNetAdapter(nn.Module):
    """DVCE-facing classifier adapter for project ResNet18 checkpoints.

    The adapter accepts image tensors in [0, 1], resizes them to the medical
    classifier input size, applies ImageNet normalization, and returns logits.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.register_buffer("mean", IMAGENET_MEAN.clone())
        self.register_buffer("std", IMAGENET_STD.clone())

    def forward(self, images):
        images = images.clamp(0.0, 1.0)
        if images.shape[-2:] != (224, 224):
            images = F.interpolate(
                images, size=(224, 224), mode="bilinear", align_corners=False
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


def select_target_class(probabilities, original_class):
    ranked_classes = torch.argsort(torch.tensor(probabilities), descending=True).tolist()
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


def build_dvce_model_config(timestep_respacing, model_output_size, use_fp16=True):
    script_util = importlib.import_module(
        "blended_diffusion.guided_diffusion.guided_diffusion.script_util"
    )
    model_config = script_util.model_and_diffusion_defaults()
    model_config.update(
        {
            "attention_resolutions": "32, 16, 8",
            "class_cond": model_output_size == 512,
            "diffusion_steps": 1000,
            "rescale_timesteps": True,
            "timestep_respacing": timestep_respacing,
            "image_size": model_output_size,
            "learn_sigma": True,
            "noise_schedule": "linear",
            "num_channels": 256,
            "num_head_channels": 64,
            "num_res_blocks": 2,
            "resblock_updown": True,
            "use_fp16": use_fp16,
            "use_scale_shift_norm": True,
        }
    )
    return model_config


def run_dvce_core_initialization(repo_path, timestep_respacing, model_output_size):
    start_time = time.time()
    checkpoint_path = Path(repo_path).resolve() / DVCE_CHECKPOINT_RELATIVE_PATH

    script_util = importlib.import_module(
        "blended_diffusion.guided_diffusion.guided_diffusion.script_util"
    )
    model_config = build_dvce_model_config(
        timestep_respacing, model_output_size, use_fp16=True
    )
    model, diffusion = script_util.create_model_and_diffusion(**model_config)
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


def load_dvce_diffusion_backbone(
    repo_path, device, timestep_respacing, model_output_size, use_fp16=False
):
    checkpoint_path = Path(repo_path).resolve() / DVCE_CHECKPOINT_RELATIVE_PATH
    script_util = importlib.import_module(
        "blended_diffusion.guided_diffusion.guided_diffusion.script_util"
    )
    model_config = build_dvce_model_config(
        timestep_respacing, model_output_size, use_fp16=use_fp16
    )
    model, diffusion = script_util.create_model_and_diffusion(**model_config)
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.requires_grad_(False)
    model.eval()
    if use_fp16:
        model.convert_to_fp16()
    model = model.to(device)
    return model, diffusion


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


def create_medical_guidance_fn(
    adapter,
    target_class,
    original_image_01,
    classifier_guidance_scale,
    similarity_guidance_scale,
):
    target = torch.tensor([target_class], dtype=torch.long, device=original_image_01.device)

    def cond_fn(x, t, y=None, eps=None, **kwargs):
        with torch.enable_grad():
            x = x.detach().requires_grad_(True)
            image_01 = x.clamp(-1.0, 1.0).add(1.0).div(2.0)
            logits = adapter(image_01)
            log_probs = F.log_softmax(logits, dim=1)
            target_log_probability = log_probs[:, target].mean()
            similarity_loss = F.mse_loss(image_01, original_image_01)
            objective = (
                classifier_guidance_scale * target_log_probability
                - similarity_guidance_scale * similarity_loss
            )
            gradient = torch.autograd.grad(objective, x)[0]
        return gradient

    return cond_fn


def run_single_dvce_generation(
    repo_path,
    adapter,
    original_image_01,
    target_class,
    device,
    timestep_respacing,
    skip_timesteps,
    model_output_size,
    classifier_guidance_scale,
    similarity_guidance_scale,
    use_ddim,
    use_fp16,
    seed,
):
    start_time = time.time()
    diffusion_model, diffusion = load_dvce_diffusion_backbone(
        repo_path=repo_path,
        device=device,
        timestep_respacing=timestep_respacing,
        model_output_size=model_output_size,
        use_fp16=use_fp16,
    )

    original_256 = F.interpolate(
        original_image_01,
        size=(model_output_size, model_output_size),
        mode="bilinear",
        align_corners=False,
    ).to(device)
    init_image = original_256.mul(2.0).sub(1.0)

    torch.manual_seed(seed)
    noise = torch.randn_like(init_image)
    cond_fn = create_medical_guidance_fn(
        adapter=adapter,
        target_class=target_class,
        original_image_01=original_256,
        classifier_guidance_scale=classifier_guidance_scale,
        similarity_guidance_scale=similarity_guidance_scale,
    )

    loop = (
        diffusion.ddim_sample_loop_progressive
        if use_ddim
        else diffusion.p_sample_loop_progressive
    )
    final = None
    steps_seen = 0
    for sample in loop(
        diffusion_model,
        shape=tuple(init_image.shape),
        noise=noise,
        clip_denoised=True,
        cond_fn=cond_fn,
        model_kwargs={},
        device=device,
        progress=False,
        skip_timesteps=skip_timesteps,
        init_image=init_image,
    ):
        final = sample["sample"]
        steps_seen += 1

    if final is None:
        raise RuntimeError("DVCE sampling loop did not return a sample.")

    counterfactual_01 = final.detach().clamp(-1.0, 1.0).add(1.0).div(2.0)
    runtime_seconds = time.time() - start_time

    del diffusion_model
    del diffusion
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return counterfactual_01, {
        "runtime_seconds": round(runtime_seconds, 3),
        "steps_seen": steps_seen,
        "use_ddim": use_ddim,
        "use_fp16": use_fp16,
        "classifier_guidance_scale": classifier_guidance_scale,
        "similarity_guidance_scale": similarity_guidance_scale,
        "seed": seed,
    }


def compute_difference_stats(original_01, counterfactual_01):
    diff = torch.abs(original_01 - counterfactual_01)
    changed_pixels = torch.mean((diff > 0.05).float()).item()
    return diff, {
        "original_stats": tensor_stats(original_01),
        "counterfactual_stats": tensor_stats(counterfactual_01),
        "diff_stats": tensor_stats(diff),
        "changed_pixels_threshold_0_05": float(changed_pixels),
        "mean_absolute_difference": float(diff.mean().detach().cpu().item()),
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
    resized = F.interpolate(image_01, size=(256, 256), mode="bilinear", align_corners=False)
    utils.save_image(resized, output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dvce_repo", type=str, default="external/DVCEs")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--timestep_respacing", type=str, default="10")
    parser.add_argument("--skip_timesteps", type=int, default=8)
    parser.add_argument("--model_output_size", type=int, default=256)
    parser.add_argument("--skip_diffusion_core", action="store_true")
    parser.add_argument("--run_generation", action="store_true")
    parser.add_argument("--classifier_guidance_scale", type=float, default=80.0)
    parser.add_argument("--similarity_guidance_scale", type=float, default=20.0)
    parser.add_argument("--use_ddim", action="store_true")
    parser.add_argument("--diffusion_fp16", action="store_true")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    repo_path = Path(args.dvce_repo).resolve()
    add_dvce_to_python_path(repo_path)
    apply_dvce_compatibility_patches()

    device = resolve_device(args.device)
    model, checkpoint = load_medical_model(args.model_path, device)
    classes = checkpoint["classes"]
    data = create_dataloaders(
        args.dataset_path, batch_size=args.batch_size, use_augmentation=False
    )
    sample = choose_correct_sample(model, data["test_loader"], device)

    original_class = sample["prediction_index"]
    target_class = select_target_class(sample["probabilities"], original_class)
    image_01 = denormalize(sample["image_normalized"]).to(device)
    image_256 = F.interpolate(
        image_01, size=(args.model_output_size, args.model_output_size), mode="bilinear", align_corners=False
    )

    adapter = MedicalResNetAdapter(model).to(device)
    adapter.eval()

    preview_path = output_dir / "dvce_medical_input_256.png"
    save_preview(image_01.detach().cpu(), preview_path)

    adapter_forward = run_adapter_forward(adapter, image_256)
    gradient_check = run_classifier_guidance_gradient_check(adapter, image_256, target_class)

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
        )

    generation_result = {
        "attempted": False,
        "ok": False,
        "error": None,
    }
    if args.run_generation:
        try:
            counterfactual_01, generation_settings = run_single_dvce_generation(
                repo_path=repo_path,
                adapter=adapter,
                original_image_01=image_01,
                target_class=target_class,
                device=device,
                timestep_respacing=args.timestep_respacing,
                skip_timesteps=args.skip_timesteps,
                model_output_size=args.model_output_size,
                classifier_guidance_scale=args.classifier_guidance_scale,
                similarity_guidance_scale=args.similarity_guidance_scale,
                use_ddim=args.use_ddim,
                use_fp16=args.diffusion_fp16,
                seed=args.seed,
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
            original_prediction = predict_with_adapter(adapter, original_256.to(device), classes)
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
            "validity, visual plausibility, changed pixels, and runtime against CFProto "
            "and SEDC-T."
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
        next_step = "Fix the failed adapter, gradient, or diffusion initialization check."

    metadata = {
        "purpose": "DVCE medical adapter prototype smoke test",
        "status": status,
        "python": sys.version,
        "device": str(device),
        "dvce_repo": str(repo_path),
        "dvce_checkpoint": str(repo_path / DVCE_CHECKPOINT_RELATIVE_PATH),
        "runner_settings": {
            "timestep_respacing": args.timestep_respacing,
            "skip_timesteps": args.skip_timesteps,
            "model_output_size": args.model_output_size,
            "batch_size": args.batch_size,
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
        "adapter_forward_check": adapter_forward,
        "classifier_guidance_gradient_check": gradient_check,
        "dvce_core_initialization": diffusion_core,
        "single_image_generation": generation_result,
        "next_step_after_success": next_step,
    }

    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"Saved DVCE medical prototype metadata to {metadata_path}")
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
