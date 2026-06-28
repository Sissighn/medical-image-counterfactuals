import argparse
import importlib
import json
import sys
import types
from pathlib import Path

import numpy as np
import torch.hub


def apply_compatibility_patches():
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


def check_import(module_name):
    try:
        module = importlib.import_module(module_name)
        return {
            "module": module_name,
            "ok": True,
            "version": str(getattr(module, "__version__", "")),
            "error": None,
        }
    except Exception as error:
        return {
            "module": module_name,
            "ok": False,
            "version": "",
            "error": f"{type(error).__name__}: {error}",
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dvce_repo", type=str, default="external/DVCEs")
    parser.add_argument(
        "--output_path",
        type=str,
        default="results/dvce_feasibility/dvce_environment_check.json",
    )
    args = parser.parse_args()

    repo_path = Path(args.dvce_repo).resolve()
    sys.path.insert(0, str(repo_path))
    sys.path.insert(0, str(repo_path / "blended_diffusion"))
    sys.path.insert(0, str(repo_path / "blended_diffusion" / "guided_diffusion"))

    apply_compatibility_patches()

    modules = [
        "torch",
        "torchvision",
        "numpy",
        "scipy",
        "yaml",
        "lpips",
        "blobfile",
        "timm",
        "kornia",
        "robustness",
        "robustbench",
        "perceptual_advex",
        "advex_uar",
        "clip",
        "configs",
        "utils_svces.config",
        "blended_diffusion.guided_diffusion.guided_diffusion.script_util",
        "blended_diffusion.optimization.arguments",
        "blended_diffusion.optimization.dff_attack",
        "blended_diffusion.optimization",
        "blended_diffusion.optimization.image_editor",
    ]

    results = [check_import(module_name) for module_name in modules]
    checkpoint_path = repo_path / "checkpoints" / "256x256_diffusion_uncond.pt"
    metadata = {
        "dvce_repo": str(repo_path),
        "python": sys.version,
        "compatibility_patches": [
            "np.float -> float",
            "np.int -> int",
            "torchvision.models.utils.load_state_dict_from_url -> torch.hub.load_state_dict_from_url",
            "CLIP.clip -> installed openai clip package when available",
        ],
        "diffusion_checkpoint": {
            "path": str(checkpoint_path),
            "exists": checkpoint_path.exists(),
        },
        "imports": results,
        "all_core_imports_ok": all(
            result["ok"]
            for result in results
            if result["module"] != "blended_diffusion.optimization.image_editor"
        ),
        "notes": [
            "image_editor is not required for the first DiffusionAttack import path.",
            "Full generation still requires the OpenAI 256x256 diffusion checkpoint.",
        ],
    }

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metadata, f, indent=4)

    for result in results:
        status = "OK" if result["ok"] else "FAIL"
        print(f"{status} {result['module']} {result['error'] or result['version']}")
    print(f"Diffusion checkpoint exists: {checkpoint_path.exists()}")
    print(f"Saved environment check to {output_path}")


if __name__ == "__main__":
    main()
