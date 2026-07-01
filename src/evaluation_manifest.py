import json
from pathlib import Path


def load_manifest_records(manifest_path, max_records=None):
    manifest_path = Path(manifest_path)
    with open(manifest_path) as f:
        manifest = json.load(f)

    records = manifest.get("records", [])
    if max_records is not None:
        records = records[:max_records]

    return manifest, records


def load_image_from_manifest_record(test_dataset, record):
    dataset_index = int(record["dataset_index"])
    image, label = test_dataset[dataset_index]
    image_path, dataset_label = test_dataset.samples[dataset_index]

    expected_label = int(record["true_label_index"])
    if int(label) != expected_label or int(dataset_label) != expected_label:
        raise ValueError(
            "Manifest label mismatch for dataset index "
            f"{dataset_index}: manifest={expected_label}, dataset={int(label)}"
        )

    expected_path = Path(record["image_path"]).resolve()
    actual_path = Path(image_path).resolve()
    if expected_path != actual_path:
        raise ValueError(
            "Manifest image path mismatch for dataset index "
            f"{dataset_index}: manifest={expected_path}, dataset={actual_path}"
        )

    return image.unsqueeze(0), int(label), str(image_path)


def manifest_record_metadata(record):
    return {
        "manifest_sample_index": int(record["manifest_sample_index"]),
        "dataset_index": int(record["dataset_index"]),
        "source_image_path": record["image_path"],
    }
