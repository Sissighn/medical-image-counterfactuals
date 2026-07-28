from src.data_utils import create_dataloaders


def test_dataset(name, path):
    print("=" * 50)
    print(name)
    print("=" * 50)

    data = create_dataloaders(path)

    print("Classes:")
    print(data["classes"])
    print()

    print("Class-to-index:")
    print(data["class_to_idx"])
    print()

    print("Number of classes:")
    print(data["num_classes"])
    print()

    images, labels = next(iter(data["train_loader"]))

    print("Batch loaded successfully.")
    print("Images shape:", images.shape)
    print("Labels shape:", labels.shape)
    print()


def main():
    test_dataset("BUSI", "data/processed/BUSI")
    test_dataset("Pneumonia", "data/processed/Pneumonia")


if __name__ == "__main__":
    main()
