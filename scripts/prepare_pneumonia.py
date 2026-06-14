from pathlib import Path
import shutil

RAW_DIR = Path("data/raw/Pneumonia/chest_xray")
OUT_DIR = Path("data/processed/Pneumonia")

SPLITS = ["train", "val", "test"]
CLASSES = ["NORMAL", "PNEUMONIA"]


def create_folder_structure():
    for split in SPLITS:
        for class_name in CLASSES:
            folder = OUT_DIR / split / class_name
            folder.mkdir(parents=True, exist_ok=True)


def copy_images(split, class_name):
    source_folder = RAW_DIR / split / class_name
    target_folder = OUT_DIR / split / class_name

    count = 0

    for file_path in source_folder.iterdir():
        if file_path.suffix.lower() not in [".png", ".jpg", ".jpeg"]:
            continue

        target_path = target_folder / file_path.name
        shutil.copy2(file_path, target_path)
        count += 1

    return count


def main():
    create_folder_structure()

    for split in SPLITS:
        print(split)

        for class_name in CLASSES:
            count = copy_images(split, class_name)
            print(f"  {class_name}: {count}")

        print()

    print("Pneumonia wurde sauber vorbereitet.")


if __name__ == "__main__":
    main()
