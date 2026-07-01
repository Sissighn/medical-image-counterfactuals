import argparse
import json
from pathlib import Path

from PIL import Image


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def iter_image_paths(dataset_dir, splits):
    dataset_dir = Path(dataset_dir)
    for split in splits:
        split_dir = dataset_dir / split
        if not split_dir.exists():
            continue
        for path in sorted(split_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                yield split, path


def prepare_dataset(
    dataset_name,
    dataset_dir,
    output_dir,
    splits,
    image_size,
    max_images=None,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for split, source_path in iter_image_paths(dataset_dir, splits):
        if max_images is not None and len(records) >= max_images:
            break

        class_name = source_path.parent.name
        output_name = (
            f"{dataset_name}_{len(records):06d}_{split}_{class_name}"
            f"{source_path.suffix.lower()}"
        )
        output_path = output_dir / output_name

        with Image.open(source_path) as image:
            image = image.convert("RGB")
            image = image.resize((image_size, image_size), Image.BICUBIC)
            image.save(output_path)

        records.append(
            {
                "dataset": dataset_name,
                "split": split,
                "class_name": class_name,
                "source_path": str(source_path),
                "output_path": str(output_path),
            }
        )

    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--busi_dir", type=str, default="data/processed/BUSI")
    parser.add_argument("--pneumonia_dir", type=str, default="data/processed/Pneumonia")
    parser.add_argument("--output_dir", type=str, default="data/diffusion_training")
    parser.add_argument(
        "--dataset",
        choices=["busi", "pneumonia", "combined"],
        default="combined",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train"],
        help="Dataset splits to export. Use train only by default to avoid test leakage.",
    )
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--max_images_per_dataset", type=int, default=None)
    args = parser.parse_args()

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    all_records = []
    dataset_configs = []
    if args.dataset in {"busi", "combined"}:
        dataset_configs.append(("busi", args.busi_dir))
    if args.dataset in {"pneumonia", "combined"}:
        dataset_configs.append(("pneumonia", args.pneumonia_dir))

    for dataset_name, dataset_dir in dataset_configs:
        dataset_output_dir = output_root / dataset_name
        records = prepare_dataset(
            dataset_name=dataset_name,
            dataset_dir=dataset_dir,
            output_dir=dataset_output_dir,
            splits=args.splits,
            image_size=args.image_size,
            max_images=args.max_images_per_dataset,
        )
        all_records.extend(records)

    if args.dataset == "combined":
        combined_dir = output_root / "combined"
        combined_dir.mkdir(parents=True, exist_ok=True)
        combined_records = []
        for index, record in enumerate(all_records):
            source_path = Path(record["output_path"])
            output_path = combined_dir / f"combined_{index:06d}_{source_path.name}"
            with Image.open(source_path) as image:
                image.save(output_path)
            combined_records.append({**record, "combined_output_path": str(output_path)})
    else:
        combined_records = []

    metadata = {
        "purpose": "prepare flat 256x256 image folders for diffusion training",
        "dataset": args.dataset,
        "splits": args.splits,
        "image_size": args.image_size,
        "output_dir": str(output_root),
        "num_images": len(all_records),
        "num_combined_images": len(combined_records),
        "counts_by_dataset": {
            dataset_name: sum(1 for record in all_records if record["dataset"] == dataset_name)
            for dataset_name, _ in dataset_configs
        },
        "records": all_records,
    }

    metadata_path = output_root / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"Prepared {len(all_records)} images for diffusion training.")
    if combined_records:
        print(f"Prepared {len(combined_records)} combined images.")
    print(f"Saved metadata to {metadata_path}")


if __name__ == "__main__":
    main()
