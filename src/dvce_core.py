from __future__ import annotations

import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

DVCE_CHECKPOINT_RELATIVE_PATH = Path("checkpoints") / "256x256_diffusion_uncond.pt"

if not hasattr(np, "float"):
    np.float = float
if not hasattr(np, "int"):
    np.int = int


def add_dvce_to_python_path(repo_path: str | Path) -> None:
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


def resolve_diffusion_checkpoint_path(
    repo_path: str | Path,
    checkpoint_path: str | Path | None = None,
) -> Path:
    if checkpoint_path:
        return Path(checkpoint_path).resolve()
    return Path(repo_path).resolve() / DVCE_CHECKPOINT_RELATIVE_PATH


def build_dvce_model_config(
    timestep_respacing: str,
    model_output_size: int,
    use_fp16: bool = False,
) -> dict[str, Any]:
    from blended_diffusion.guided_diffusion.guided_diffusion import script_util

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


def cast_diffusion_numpy_arrays_to_float32(diffusion: Any) -> None:
    for name, value in vars(diffusion).items():
        if isinstance(value, np.ndarray) and value.dtype == np.float64:
            setattr(diffusion, name, value.astype(np.float32))


def load_dvce_diffusion_backbone(
    repo_path: str | Path,
    device: torch.device,
    timestep_respacing: str,
    model_output_size: int,
    use_fp16: bool = False,
    checkpoint_path: str | Path | None = None,
) -> tuple[torch.nn.Module, Any, dict[str, Any]]:
    add_dvce_to_python_path(repo_path)
    from blended_diffusion.guided_diffusion.guided_diffusion import script_util

    checkpoint_path = resolve_diffusion_checkpoint_path(repo_path, checkpoint_path)
    model_config = build_dvce_model_config(
        timestep_respacing=timestep_respacing,
        model_output_size=model_output_size,
        use_fp16=use_fp16,
    )
    model, diffusion = script_util.create_model_and_diffusion(**model_config)
    cast_diffusion_numpy_arrays_to_float32(diffusion)

    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.requires_grad_(False)
    model.eval()
    model = model.to(device)
    gradient_parameter_names = []
    for name, parameter in model.named_parameters():
        if "qkv" in name or "norm" in name or "proj" in name:
            parameter.requires_grad_(True)
            gradient_parameter_names.append(name)
    if use_fp16:
        model.convert_to_fp16()

    return (
        model,
        diffusion,
        {
            "checkpoint_path": str(checkpoint_path),
            "model_config": model_config,
            "diffusion_num_timesteps": int(getattr(diffusion, "num_timesteps", -1)),
            "gradient_parameter_count": len(gradient_parameter_names),
            "gradient_parameter_rule": (
                "qkv/norm/proj parameters require gradients, matching local "
                "DiffusionAttack."
            ),
        },
    )


def load_image_augmentations_class() -> Any:
    import importlib.util

    import blended_diffusion

    module_path = (
        Path(blended_diffusion.__file__).parent / "optimization" / "augmentations.py"
    )
    spec = importlib.util.spec_from_file_location(
        "dvce_original_augmentations", module_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ImageAugmentations


def map_minus1_1_to_0_1(images: torch.Tensor) -> torch.Tensor:
    return images.add(1.0).div(2.0)


def renormalize_gradient(
    grad: torch.Tensor,
    eps: torch.Tensor,
    small_const: float = 1e-22,
) -> tuple[torch.Tensor, torch.Tensor]:
    grad_norm = grad.view(grad.shape[0], -1).norm(p=2, dim=1).view(
        grad.shape[0], 1, 1, 1
    )
    grad_norm = torch.where(grad_norm < small_const, grad_norm + small_const, grad_norm)
    grad /= grad_norm
    grad *= eps.view(grad.shape[0], -1).norm(p=2, dim=1).view(
        grad.shape[0], 1, 1, 1
    )
    return grad, grad_norm


def cone_projection(
    grad_temp_1: torch.Tensor,
    grad_temp_2: torch.Tensor,
    deg: float,
    subspace_projection: bool = False,
) -> torch.Tensor:
    original_shape = grad_temp_2.shape
    flattened_inputs = not subspace_projection and grad_temp_1.dim() > 2
    if flattened_inputs:
        grad_temp_1 = grad_temp_1.view(grad_temp_1.shape[0], -1)
        grad_temp_2 = grad_temp_2.view(grad_temp_2.shape[0], -1)

    if subspace_projection:
        grad_temp = []
        for i_image, grad_temp_1_image in enumerate(grad_temp_1):
            grad_temp.append(
                sum(
                    [
                        grad_temp_1_image_dimension
                        * (
                            (
                                grad_temp_2[i_image].unsqueeze(0)
                                * grad_temp_1_image_dimension
                            ).sum()
                            / (
                                grad_temp_1_image_dimension
                                * grad_temp_1_image_dimension
                            ).sum()
                        )
                        for grad_temp_1_image_dimension in grad_temp_1_image
                    ]
                )
            )
        del grad_temp_1
        grad_temp = torch.stack(grad_temp, 0)
    else:
        angles_before = torch.acos(
            (grad_temp_1 * grad_temp_2).sum(1)
            / (grad_temp_1.norm(p=2, dim=1) * grad_temp_2.norm(p=2, dim=1))
        )

        grad_temp_2 /= grad_temp_2.norm(p=2, dim=1).view(grad_temp_1.shape[0], -1)
        grad_temp_1 = grad_temp_1 - (
            (grad_temp_1 * grad_temp_2).sum(1)
            / (grad_temp_2.norm(p=2, dim=1) ** 2)
        ).view(grad_temp_1.shape[0], -1) * grad_temp_2
        grad_temp_1 /= grad_temp_1.norm(p=2, dim=1).view(grad_temp_1.shape[0], -1)
        radians = torch.tensor([deg], device=grad_temp_1.device).deg2rad()

        cone_projection_result = grad_temp_1 * torch.tan(radians) + grad_temp_2

        grad_temp = grad_temp_2.clone()
        grad_temp[angles_before > radians] = cone_projection_result[
            angles_before > radians
        ]

    if flattened_inputs:
        return grad_temp.view(original_shape)
    return grad_temp


def compute_lp_dist(x: torch.Tensor, y: torch.Tensor, p: float) -> torch.Tensor:
    if int(p) == 1:
        return F.l1_loss(x, y)
    if int(p) == 2:
        return F.mse_loss(x, y)
    return torch.mean(torch.abs(x - y) ** p)


def compute_lp_gradient(
    diff: torch.Tensor,
    p: float,
    small_const: float = 1e-12,
) -> torch.Tensor:
    if p < 1:
        return (p * (diff.abs() + small_const) ** (p - 1)) * diff.sign()
    return (p * diff.abs() ** (p - 1)) * diff.sign()


def make_original_style_cond_fn(
    diffusion: Any,
    diffusion_model: torch.nn.Module,
    classifier: torch.nn.Module,
    second_classifier: torch.nn.Module | None,
    init_image: torch.Tensor,
    target_class: int,
    classifier_lambda: float,
    lp_custom: float,
    lp_custom_value: float,
    enforce_same_norms: bool,
    denoise_dist_input: bool,
    aug_num: int,
    model_output_size: int,
    deg_cone_projection: float,
    classifier_size: int = 224,
) -> Any:
    target = torch.tensor([target_class], dtype=torch.long, device=init_image.device)
    augmentations = None
    if aug_num > 1:
        augmentations = load_image_augmentations_class()(classifier_size, aug_num).to(
            init_image.device
        )

    def unscale_timestep(t: torch.Tensor) -> torch.Tensor:
        return (t * (diffusion.num_timesteps / 1000)).long()

    def apply_augmentations(images: torch.Tensor) -> torch.Tensor:
        if augmentations is None:
            return images
        if images.device.type == "mps":
            return augmentations.cpu()(images.cpu()).to(images.device)
        return augmentations(images)

    def cond_fn_clean(x: torch.Tensor, t: torch.Tensor, y=None, eps=None, **kwargs):
        if eps is None:
            raise ValueError("Original-style DVCE cond_fn requires eps=model_output.")

        grad_out = torch.zeros_like(x)
        x = x.detach().requires_grad_()
        t = unscale_timestep(t)
        y = target if y is None else y

        with torch.enable_grad():
            out = diffusion.p_mean_variance(
                diffusion_model,
                x,
                t,
                clip_denoised=False,
                model_kwargs={"y": y},
            )
            x_in = out["pred_xstart"]

            if classifier_lambda != 0:
                x_classifier = apply_augmentations(x_in)
                logits = classifier(map_minus1_1_to_0_1(x_classifier))
                log_probs = F.log_softmax(logits, dim=-1)
                repeated_target = y.view(-1).repeat(aug_num)
                target_log_confs = log_probs[
                    range(x_classifier.shape[0]),
                    repeated_target,
                ]
                grad_class = torch.autograd.grad(
                    target_log_confs.mean(),
                    x,
                    retain_graph=denoise_dist_input or second_classifier is not None,
                )[0]

                if second_classifier is not None and deg_cone_projection > 0:
                    x_classifier_2 = apply_augmentations(x_in)
                    logits_2 = second_classifier(map_minus1_1_to_0_1(x_classifier_2))
                    log_probs_2 = F.log_softmax(logits_2, dim=-1)
                    target_log_confs_2 = log_probs_2[
                        range(x_classifier_2.shape[0]),
                        repeated_target,
                    ]
                    grad_2 = torch.autograd.grad(
                        target_log_confs_2.mean(),
                        x,
                        retain_graph=denoise_dist_input,
                    )[0]
                    grad_class = (
                        cone_projection(
                            grad_2.view(x.shape[0], -1).cpu(),
                            grad_class.view(x.shape[0], -1).cpu(),
                            deg=deg_cone_projection,
                        )
                        .view_as(grad_class)
                        .to(x.device)
                    )

                if enforce_same_norms:
                    grad_class, _ = renormalize_gradient(grad_class, eps)
                grad_out = grad_out + classifier_lambda * grad_class

            if lp_custom:
                if denoise_dist_input:
                    lp_dist = compute_lp_dist(x_in, init_image, lp_custom)
                    lp_grad = torch.autograd.grad(lp_dist, x)[0]
                else:
                    diff = x_in - init_image
                    lp_grad = compute_lp_gradient(diff, lp_custom)

                if enforce_same_norms:
                    lp_grad, _ = renormalize_gradient(lp_grad, eps)
                grad_out = grad_out - lp_custom_value * lp_grad

        return grad_out

    return cond_fn_clean


def generate_dvce_counterfactual(
    repo_path: str | Path,
    classifier: torch.nn.Module,
    original_image_01: torch.Tensor,
    target_class: int,
    device: torch.device,
    diffusion_checkpoint_path: str | Path | None = None,
    model_output_size: int = 256,
    timestep_respacing: str = "200",
    skip_timesteps: int = 100,
    use_ddim: bool = False,
    use_fp16: bool = False,
    seed: int = 1,
    classifier_lambda: float = 0.1,
    lp_custom: float = 1.0,
    lp_custom_value: float = 0.15,
    enforce_same_norms: bool = True,
    denoise_dist_input: bool = False,
    aug_num: int = 1,
    clip_denoised: bool = False,
    deg_cone_projection: float = 0.0,
    second_classifier: torch.nn.Module | None = None,
    classifier_size: int = 224,
) -> tuple[torch.Tensor, dict[str, Any]]:

    start_time = time.time()
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    diffusion_model, diffusion, backbone_info = load_dvce_diffusion_backbone(
        repo_path=repo_path,
        device=device,
        timestep_respacing=timestep_respacing,
        model_output_size=model_output_size,
        use_fp16=use_fp16,
        checkpoint_path=diffusion_checkpoint_path,
    )

    original_resized = F.interpolate(
        original_image_01.to(device),
        size=(model_output_size, model_output_size),
        mode="bilinear",
        align_corners=False,
    ).clamp(0.0, 1.0)
    init_image = original_resized.mul(2.0).sub(1.0)
    if use_fp16:
        init_image = init_image.half()

    cond_fn = make_original_style_cond_fn(
        diffusion=diffusion,
        diffusion_model=diffusion_model,
        classifier=classifier,
        second_classifier=second_classifier,
        init_image=init_image,
        target_class=target_class,
        classifier_lambda=classifier_lambda,
        lp_custom=lp_custom,
        lp_custom_value=lp_custom_value,
        enforce_same_norms=enforce_same_norms,
        denoise_dist_input=denoise_dist_input,
        aug_num=aug_num,
        model_output_size=model_output_size,
        deg_cone_projection=deg_cone_projection,
        classifier_size=classifier_size,
    )

    gen_type = "ddim" if use_ddim else "p_sample"
    loop = (
        diffusion.ddim_sample_loop_progressive
        if use_ddim
        else diffusion.p_sample_loop_progressive
    )
    model_kwargs = {
        "y": torch.tensor([target_class], dtype=torch.long, device=device),
    }

    final_pred_xstart = None
    final_sample = None
    steps_seen = 0
    loop_kwargs = {
        "clip_denoised": clip_denoised,
        "cond_fn": cond_fn,
        "model_kwargs": model_kwargs,
        "device": device,
        "progress": False,
        "skip_timesteps": skip_timesteps,
        "init_image": init_image,
        "randomize_class": False,
    }
    if not use_ddim:
        loop_kwargs["seed"] = seed

    for sample in loop(
        diffusion_model,
        shape=tuple(init_image.shape),
        **loop_kwargs,
    ):
        final_sample = sample.get("sample")
        final_pred_xstart = sample.get("pred_xstart", final_sample)
        steps_seen += 1

    if final_pred_xstart is None:
        raise RuntimeError("DVCE sampling loop did not return a pred_xstart sample.")

    counterfactual_01 = (
        map_minus1_1_to_0_1(final_pred_xstart.float()).clamp(0.0, 1.0).detach().cpu()
    )
    runtime_seconds = time.time() - start_time

    debug = {
        "runtime_seconds": round(runtime_seconds, 3),
        "steps_seen": steps_seen,
        "seed": seed,
        "model_output_size": model_output_size,
        "timestep_respacing": timestep_respacing,
        "skip_timesteps": skip_timesteps,
        "use_ddim": use_ddim,
        "gen_type": gen_type,
        "use_fp16": use_fp16,
        "clip_denoised": clip_denoised,
        "classifier_lambda": classifier_lambda,
        "lp_custom": lp_custom,
        "lp_custom_value": lp_custom_value,
        "enforce_same_norms": enforce_same_norms,
        "denoise_dist_input": denoise_dist_input,
        "denoise_dist_input_default_note": (
            "The original argparse default is False, but both published DVCE "
            "commands in the original readme pass --denoise_dist_input, so "
            "original-faithful runs should enable it."
        ),
        "deg_cone_projection": deg_cone_projection,
        "classifier_size": classifier_size,
        "cone_projection_enabled": second_classifier is not None
        and deg_cone_projection > 0,
        "aug_num": aug_num,
        "guidance_space": "pred_xstart",
        "distance_target": "init_image",
        "sampling_backbone": backbone_info,
        "note": (
            "Original-style medical DVCE core: cond_fn recomputes "
            "p_mean_variance, evaluates classifier and LP distance on "
            "pred_xstart (unclamped _map_img input), separately normalizes "
            "classifier/distance gradients to eps=model_output when "
            "enforce_same_norms is enabled, and returns grad_class - lp_grad. "
            "Cone projection (active with second classifier and "
            "deg_cone_projection > 0) projects the robust second classifier's "
            "gradient onto the cone centered at the explained classifier's "
            "gradient, matching the original dff_attack.py argument order."
        ),
    }

    del diffusion_model
    del diffusion
    if device.type == "mps":
        torch.mps.empty_cache()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return counterfactual_01, debug
