import argparse
import json
from pathlib import Path


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def count_images(path):
    path = Path(path)
    if not path.exists():
        return 0
    return sum(
        1
        for item in path.rglob("*")
        if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS
    )


def build_training_command(
    dvce_repo,
    data_dir,
    checkpoint_path,
    output_dir,
    batch_size,
    lr,
    lr_anneal_steps,
    save_interval,
    diffusion_training_test,
):
    guided_root = (Path(dvce_repo) / "blended_diffusion" / "guided_diffusion").resolve()
    image_train = guided_root / "scripts" / "image_train.py"

    model_flags = (
        "--image_size 256 --num_channels 256 --num_res_blocks 2 "
        "--learn_sigma True --class_cond False "
        "--attention_resolutions 32,16,8 --num_head_channels 64 "
        "--resblock_updown True --use_scale_shift_norm True "
        "--diffusion_steps 1000 --noise_schedule linear --rescale_timesteps True"
    )
    train_flags = (
        f"--data_dir {data_dir} --batch_size {batch_size} --lr {lr} "
        f"--lr_anneal_steps {lr_anneal_steps} --save_interval {save_interval} "
        f"--resume_checkpoint {checkpoint_path}"
    )
    test_prefix = "DIFFUSION_TRAINING_TEST=1 " if diffusion_training_test else ""

    return (
        f"cd {guided_root}\n"
        f"{test_prefix}PYTHONPATH={guided_root} OPENAI_LOGDIR={output_dir} "
        f"python scripts/image_train.py {model_flags} {train_flags}"
    ), str(image_train)


def write_markdown_report(report, output_path):
    lines = [
        "# Diffusion Training Setup",
        "",
        "This report checks whether the local project is ready for medical diffusion fine-tuning.",
        "",
        "## Status",
        "",
        f"- DVCE repo exists: `{report['dvce_repo_exists']}`",
        f"- guided-diffusion training script exists: `{report['image_train_exists']}`",
        f"- resume checkpoint exists: `{report['resume_checkpoint_exists']}`",
        f"- training data images: `{report['num_training_images']}`",
        "",
        "## Recommended Command",
        "",
        "```bash",
        report["recommended_command"],
        "```",
        "",
        "## Notes",
        "",
        "- Use Pneumonia first because it has the largest available medical image set.",
        "- The command resumes from the OpenAI 256x256 unconditional checkpoint.",
        "- Run a short smoke test first with `DIFFUSION_TRAINING_TEST=1` and a small `--lr_anneal_steps`, then use a stronger GPU for a real run.",
        "- After training, point the DVCE runner to the new checkpoint and rerun the fixed evaluation manifests.",
    ]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dvce_repo", type=str, default="external/DVCEs")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/diffusion_training/pneumonia",
    )
    parser.add_argument(
        "--resume_checkpoint",
        type=str,
        default="external/DVCEs/checkpoints/256x256_diffusion_uncond.pt",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/diffusion_training_setup",
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--lr_anneal_steps", type=int, default=1000)
    parser.add_argument("--save_interval", type=int, default=250)
    parser.add_argument("--diffusion_training_test", action="store_true")
    args = parser.parse_args()

    command, image_train = build_training_command(
        dvce_repo=args.dvce_repo,
        data_dir=Path(args.data_dir).resolve(),
        checkpoint_path=Path(args.resume_checkpoint).resolve(),
        output_dir=Path(args.output_dir).resolve() / "openai_logdir",
        batch_size=args.batch_size,
        lr=args.lr,
        lr_anneal_steps=args.lr_anneal_steps,
        save_interval=args.save_interval,
        diffusion_training_test=args.diffusion_training_test,
    )

    report = {
        "purpose": "medical diffusion fine-tuning setup check",
        "dvce_repo": args.dvce_repo,
        "dvce_repo_exists": Path(args.dvce_repo).exists(),
        "image_train_path": image_train,
        "image_train_exists": Path(image_train).exists(),
        "data_dir": args.data_dir,
        "num_training_images": count_images(args.data_dir),
        "resume_checkpoint": args.resume_checkpoint,
        "resume_checkpoint_exists": Path(args.resume_checkpoint).exists(),
        "batch_size": args.batch_size,
        "lr": args.lr,
        "lr_anneal_steps": args.lr_anneal_steps,
        "save_interval": args.save_interval,
        "diffusion_training_test": args.diffusion_training_test,
        "recommended_command": command,
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "setup_check.json"
    md_path = output_dir / "README.md"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=4)
    write_markdown_report(report, md_path)

    print(f"Saved diffusion training setup check to {json_path}")
    print(f"Saved diffusion training command README to {md_path}")


if __name__ == "__main__":
    main()
