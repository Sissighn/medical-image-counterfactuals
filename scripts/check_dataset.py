from pathlib import Path

DATASETS = {
    "BUSI": {
        "path": Path("data/processed/BUSI"),
        "classes": ["normal", "benign", "malignant"],
    },
    "Pneumonia": {
        "path": Path("data/processed/Pneumonia"),
        "classes": ["NORMAL", "PNEUMONIA"],
    },
}

SPLITS = ["train", "val", "test"]


def count_images(folder):
    image_extensions = [".png", ".jpg", ".jpeg"]
    return sum(
        1
        for file_path in folder.iterdir()
        if file_path.suffix.lower() in image_extensions
    )


def main():
    for dataset_name, dataset_info in DATASETS.items():
        print("=" * 40)
        print(dataset_name)
        print("=" * 40)

        dataset_path = dataset_info["path"]
        classes = dataset_info["classes"]

        for split in SPLITS:
            print(split)

            for class_name in classes:
                folder = dataset_path / split / class_name

                if not folder.exists():
                    print(f"  {class_name}: ORDNER FEHLT")
                    continue

                count = count_images(folder)
                print(f"  {class_name}: {count}")

            print()


if __name__ == "__main__":
    main()
