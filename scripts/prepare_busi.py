from pathlib import Path
import shutil
import random

# Damit die Aufteilung jedes Mal gleich bleibt
random.seed(42)

# Pfade
RAW_DIR = Path("data/raw/BUSI/Dataset_BUSI_with_GT")
OUT_DIR = Path("data/processed/BUSI")

CLASSES = ["normal", "benign", "malignant"]

# Verhältnis:
# 70% Training, 15% Validation, 15% Test
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


def create_folder_structure():
    for split in ["train", "val", "test"]:
        for class_name in CLASSES:
            folder = OUT_DIR / split / class_name
            folder.mkdir(parents=True, exist_ok=True)


def get_image_files(class_name):
    class_folder = RAW_DIR / class_name

    image_files = []

    for file_path in class_folder.iterdir():
        # Nur Bilddateien nehmen
        if file_path.suffix.lower() not in [".png", ".jpg", ".jpeg"]:
            continue

        # Masken ignorieren
        if "_mask" in file_path.name:
            continue

        image_files.append(file_path)

    return image_files


def split_files(files):
    random.shuffle(files)

    n = len(files)
    train_end = int(n * TRAIN_RATIO)
    val_end = int(n * (TRAIN_RATIO + VAL_RATIO))

    train_files = files[:train_end]
    val_files = files[train_end:val_end]
    test_files = files[val_end:]

    return train_files, val_files, test_files


def copy_files(files, split, class_name):
    target_folder = OUT_DIR / split / class_name

    for file_path in files:
        target_path = target_folder / file_path.name
        shutil.copy2(file_path, target_path)


def main():
    create_folder_structure()

    for class_name in CLASSES:
        files = get_image_files(class_name)
        train_files, val_files, test_files = split_files(files)

        copy_files(train_files, "train", class_name)
        copy_files(val_files, "val", class_name)
        copy_files(test_files, "test", class_name)

        print(f"{class_name}:")
        print(f"  train: {len(train_files)}")
        print(f"  val:   {len(val_files)}")
        print(f"  test:  {len(test_files)}")
        print()

    print("BUSI wurde sauber vorbereitet.")


if __name__ == "__main__":
    main()
