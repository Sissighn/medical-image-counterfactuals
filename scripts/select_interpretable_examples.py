import argparse
import json
import shutil
from pathlib import Path


def load_json(path):
    with open(path) as f:
        return json.load(f)


def infer_dataset(metadata):
    dataset_path = str(metadata.get("dataset_path") or "")
    classifier_adapter = metadata.get("classifier_adapter", {})
    dataset_path = dataset_path or str(classifier_adapter.get("dataset_path") or "")
    lowered = dataset_path.lower()
    if "busi" in lowered:
        return "BUSI"
    if "pneumonia" in lowered:
        return "Pneumonia"
    return "unknown"


def infer_method(metadata):
    method = metadata.get("method") or metadata.get("purpose") or "unknown"
    checkpoint = str(metadata.get("diffusion_checkpoint_path") or "")
    parameters = metadata.get("parameters", {})

    if method == "DVCE medical multi-sample generation evaluation":
        if "ema_0.9999_005000" in checkpoint:
            return "DVCE-style, Pneumonia fine-tuned checkpoint"
        if "256x256_diffusion_uncond" in checkpoint:
            return "DVCE-style, OpenAI checkpoint"
        return "DVCE-style"

    if method == "SEDC-T-style targeted segment replacement":
        search_mode = parameters.get("search_mode") or "greedy_minimal"
        roi_mode = parameters.get("roi_mode") or "none"
        max_segments = parameters.get("max_segments")
        if search_mode == "greedy_minimal" and max_segments and max_segments > 6:
            return f"SEDC-T tuned project variant ({roi_mode}, max {max_segments})"
        if roi_mode != "none":
            return f"SEDC-T project variant ({roi_mode})"
        return "SEDC-T project variant"

    if method == "SEDC-T original-style best-first segment replacement":
        return "SEDC-T original-style best-first"

    if method == "Retrieval-based nearest-unlike-neighbor baseline":
        return "Retrieval-based nearest-unlike-neighbor baseline"

    if "prototype-guided" in method.lower():
        parameters = metadata.get("parameters", {})
        if (
            parameters.get("prototype_space") == "encoder"
            and parameters.get("prototype_mode") == "knn_mean"
            and parameters.get("c_search_mode") == "adaptive_binary"
            and parameters.get("selection_metric") == "elastic_net"
        ):
            return "CFProto-nearer prototype-guided optimization baseline"
        return "Removed non-final prototype-guided result"

    return method


def aggregate_value(aggregate, key):
    value = aggregate.get(key)
    if isinstance(value, dict):
        return value.get("mean")
    return value


def record_change(record):
    change_metrics = record.get("change_metrics") or {}
    if "changed_pixel_fraction" in change_metrics:
        return float(change_metrics["changed_pixel_fraction"])
    if "changed_pixels_threshold_0_05" in record:
        return float(record["changed_pixels_threshold_0_05"])
    if "mean_absolute_difference" in record:
        return float(record["mean_absolute_difference"])
    if "l1_mean" in change_metrics:
        return float(change_metrics["l1_mean"])
    return None


def record_primary_change(record, method):
    change_metrics = record.get("change_metrics") or {}
    if method.startswith("Retrieval-based"):
        if "embedding_distance" in record:
            return float(record["embedding_distance"])
    if method.startswith("CFProto-nearer"):
        if "l1_mean" in change_metrics:
            return float(change_metrics["l1_mean"])
        if "mean_absolute_difference" in record:
            return float(record["mean_absolute_difference"])
    return record_change(record)


def record_l1(record):
    change_metrics = record.get("change_metrics") or {}
    if "l1_mean" in change_metrics:
        return float(change_metrics["l1_mean"])
    if "mean_absolute_difference" in record:
        return float(record["mean_absolute_difference"])
    return None


def record_l2(record):
    change_metrics = record.get("change_metrics") or {}
    if "l2_mean" in change_metrics:
        return float(change_metrics["l2_mean"])
    return None


def record_linf(record):
    change_metrics = record.get("change_metrics") or {}
    if "linf" in change_metrics:
        return float(change_metrics["linf"])
    diff_stats = record.get("diff_stats") or {}
    if "max" in diff_stats:
        return float(diff_stats["max"])
    image_debug_stats = record.get("image_debug_stats") or {}
    diff_debug = image_debug_stats.get("diff") or {}
    if "max" in diff_debug:
        return float(diff_debug["max"])
    return None


def record_threshold(record):
    change_metrics = record.get("change_metrics") or {}
    if "sparsity_threshold" in change_metrics:
        return float(change_metrics["sparsity_threshold"])
    if "changed_pixels_threshold_0_05" in record:
        return 0.05
    return None


def record_segments(record):
    change_metrics = record.get("change_metrics") or {}
    if "num_changed_segments" in change_metrics:
        return int(change_metrics["num_changed_segments"])
    selected_segments = record.get("selected_segments")
    if isinstance(selected_segments, list):
        return len(selected_segments)
    return None


def record_image_path(record):
    for key in ["summary_path", "visualization_path", "image_path", "counterfactual_path"]:
        value = record.get(key)
        if value:
            return value
    return ""


def safe_float(value):
    if isinstance(value, (int, float)):
        return float(value)
    return None


def fmt(value, digits=4):
    if value is None:
        return ""
    return f"{float(value):.{digits}f}"


def normalize_record(metadata_path, metadata, record):
    method = infer_method(metadata)
    confidence = safe_float(record.get("counterfactual_confidence"))
    changed_pixel_fraction = record_change(record)
    change = record_primary_change(record, method)
    l1 = record_l1(record)
    l2 = record_l2(record)
    linf = record_linf(record)
    threshold = record_threshold(record)
    num_segments = record_segments(record)
    image_path = record_image_path(record)
    embedding_distance = safe_float(record.get("embedding_distance"))

    return {
        "metadata_path": str(metadata_path),
        "method": method,
        "raw_method": metadata.get("method") or metadata.get("purpose") or "unknown",
        "dataset": infer_dataset(metadata),
        "sample_index": record.get("sample_index"),
        "manifest_sample_index": record.get("manifest_sample_index"),
        "true_label": record.get("true_label"),
        "original_prediction": record.get("original_prediction"),
        "target_class": record.get("target_class"),
        "counterfactual_prediction": record.get("counterfactual_prediction"),
        "valid_counterfactual": bool(record.get("valid_counterfactual")),
        "counterfactual_confidence": confidence,
        "change": change,
        "change_metric": (
            "embedding_distance"
            if method.startswith("Retrieval-based")
            else "l1_mean"
            if method.startswith("CFProto-nearer")
            else "changed_pixel_fraction"
        ),
        "embedding_distance": embedding_distance,
        "changed_pixel_fraction": changed_pixel_fraction,
        "sparsity_threshold": threshold,
        "l1": l1,
        "l2": l2,
        "linf": linf,
        "num_changed_segments": num_segments,
        "runtime_seconds": safe_float(record.get("runtime_seconds")),
        "image_path": image_path,
        "source_image_path": record.get("source_image_path"),
    }


def row_sort_value(record, key, default):
    value = record.get(key)
    return default if value is None else value


def record_identity(record):
    return (record.get("metadata_path"), record.get("sample_index"))


def choose_prefer_new(candidates, key, used_identities, prefer="min"):
    ordered = sorted(candidates, key=key, reverse=(prefer == "max"))
    for record in ordered:
        if record_identity(record) not in used_identities:
            return record
    return ordered[0] if ordered else None


def select_examples(records):
    method = records[0]["method"] if records else ""
    valid = [record for record in records if record["valid_counterfactual"]]
    invalid = [record for record in records if not record["valid_counterfactual"]]
    selected = []
    used_identities = set()

    if valid:
        best_balanced = choose_prefer_new(
            valid,
            key=lambda record: (
                row_sort_value(record, "change", 1e9),
                -row_sort_value(record, "counterfactual_confidence", -1e9),
            ),
            used_identities=used_identities,
            prefer="min",
        )
        selected.append(
            (
                "best_valid_balanced",
                best_balanced,
                (
                    "Valid retrieval with the smallest embedding distance."
                    if method.startswith("Retrieval-based")
                    else "Valid CF with the smallest available changed area."
                ),
            )
        )
        used_identities.add(record_identity(best_balanced))

        highest_confidence = choose_prefer_new(
            valid,
            key=lambda record: (
                row_sort_value(record, "counterfactual_confidence", -1e9),
                -row_sort_value(record, "change", 1e9),
            ),
            used_identities=used_identities,
            prefer="max",
        )
        selected.append(
            (
                "highest_confidence_valid",
                highest_confidence,
                "Valid CF with the strongest target-class confidence.",
            )
        )
        used_identities.add(record_identity(highest_confidence))

        questionable = choose_prefer_new(
            valid,
            key=lambda record: (
                row_sort_value(record, "change", -1e9),
                row_sort_value(record, "l1", -1e9),
            ),
            used_identities=used_identities,
            prefer="max",
        )
        selected.append(
            (
                "visually_questionable_valid",
                questionable,
                (
                    "Valid retrieval with a large embedding distance; useful to discuss nearest-case limitations."
                    if method.startswith("Retrieval-based")
                    else "Valid CF with a large change; useful to discuss model validity versus plausibility."
                ),
            )
        )
        used_identities.add(record_identity(questionable))

    if invalid:
        selected.append(
            (
                "failure_case",
                min(
                    invalid,
                    key=lambda record: (
                        row_sort_value(record, "change", 1e9),
                        -row_sort_value(record, "counterfactual_confidence", -1e9),
                    ),
                ),
                "Invalid CF; useful to discuss method limitations.",
            )
        )

    deduplicated = []
    seen = set()
    for category, record, rationale in selected:
        key = (category, record["metadata_path"], record["sample_index"])
        if key not in seen:
            seen.add(key)
            deduplicated.append((category, record, rationale))
    return deduplicated


def method_summary(metadata_path, metadata):
    aggregate = metadata.get("aggregate_metrics") or {}
    records = metadata.get("records") or []
    confidence_values = [
        record.get("counterfactual_confidence")
        for record in records
        if isinstance(record.get("counterfactual_confidence"), (int, float))
    ]
    mean_conf = (
        sum(confidence_values) / len(confidence_values)
        if confidence_values
        else aggregate.get("mean_counterfactual_confidence")
    )
    return {
        "metadata_path": str(metadata_path),
        "method": infer_method(metadata),
        "dataset": infer_dataset(metadata),
        "samples": aggregate.get("num_samples") or len(records),
        "validity": aggregate.get("validity"),
        "valid_count": aggregate.get("valid_count"),
        "mean_confidence": mean_conf,
        "mean_change": (
            aggregate_value(aggregate, "changed_pixel_fraction")
            or aggregate.get("mean_changed_pixels_threshold_0_05")
            or aggregate.get("mean_absolute_difference")
            or aggregate_value(aggregate, "l1_mean")
        ),
        "mean_runtime": (
            aggregate_value(aggregate, "runtime_seconds")
            or aggregate.get("mean_runtime_seconds")
        ),
    }


def markdown_link(path):
    return f"`{path}`" if path else ""


def copy_image(record, category, output_dir):
    source = Path(record["image_path"])
    if not source.exists():
        return ""
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    stem = (
        f"{record['dataset']}_{record['method']}_sample_{record['sample_index']}_{category}"
        .lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("(", "")
        .replace(")", "")
        .replace(",", "")
    )
    target = images_dir / f"{stem}{source.suffix}"
    shutil.copy2(source, target)
    return str(target)


def write_selected_examples(selections, output_dir, copy_assets):
    lines = [
        "# Selected Counterfactual Examples",
        "",
        "Examples are selected from existing fixed-evaluation metadata. They are not new methods.",
        "",
        "| Method | Dataset | Category | Sample | Valid | Prediction | Target | CF prediction | CF confidence | Primary change | Embedding dist | Changed pixels | L1/MAD | L∞ | Segments | Plot | Rationale |",
        "| --- | --- | --- | ---: | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    manifest = []
    for group_key in sorted(selections):
        for category, record, rationale in selections[group_key]:
            copied = copy_image(record, category, output_dir) if copy_assets else ""
            plot_path = copied or record["image_path"]
            lines.append(
                "| "
                f"{record['method']} | "
                f"{record['dataset']} | "
                f"{category} | "
                f"{record['sample_index']} | "
                f"{'yes' if record['valid_counterfactual'] else 'no'} | "
                f"{record.get('original_prediction') or ''} | "
                f"{record.get('target_class') or ''} | "
                f"{record.get('counterfactual_prediction') or ''} | "
                f"{fmt(record.get('counterfactual_confidence'))} | "
                f"{fmt(record.get('change'))} | "
                f"{fmt(record.get('embedding_distance'))} | "
                f"{fmt(record.get('changed_pixel_fraction'))} | "
                f"{fmt(record.get('l1'))} | "
                f"{fmt(record.get('linf'))} | "
                f"{record.get('num_changed_segments') if record.get('num_changed_segments') is not None else ''} | "
                f"{markdown_link(plot_path)} | "
                f"{rationale} |"
            )
            item = dict(record)
            item["category"] = category
            item["rationale"] = rationale
            item["copied_plot_path"] = copied
            manifest.append(item)

    (output_dir / "selected_examples.md").write_text("\n".join(lines) + "\n")
    with open(output_dir / "selected_examples.json", "w") as f:
        json.dump({"selected_examples": manifest}, f, indent=4)


def write_tradeoff_table(summaries, output_dir):
    lines = [
        "# Method Trade-off Table",
        "",
        "| Method | Dataset | Samples | Validity | Mean CF confidence | Mean change | Mean runtime | Plausibility note |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for summary in summaries:
        method = summary["method"]
        if method.startswith("CFProto-nearer"):
            note = "Final CFProto-nearer prototype-guided run; model-valid but not full Alibi CFProto."
        elif "Retrieval-based" in method:
            note = "Real target-class examples; intuitive case baseline but not a minimal edit."
        elif "SEDC-T original" in method:
            note = "Localized and method-faithful; moderate validity and slower runtime."
        elif "SEDC-T tuned" in method:
            note = "Ablation; slightly higher Pneumonia validity but more changed area."
        elif "SEDC-T project" in method:
            note = "Fast/local; includes project-specific constraints when ROI is used."
        elif "DVCE" in method:
            note = "Generative feasibility; validity and plausibility depend on checkpoint/guidance."
        else:
            note = "Interpret together with selected examples."

        lines.append(
            "| "
            f"{method} | {summary['dataset']} | {summary['samples']} | "
            f"{fmt(summary.get('validity'))} | {fmt(summary.get('mean_confidence'))} | "
            f"{fmt(summary.get('mean_change'))} | {fmt(summary.get('mean_runtime'), digits=2)} | "
            f"{note} |"
        )
    (output_dir / "method_tradeoff_table.md").write_text("\n".join(lines) + "\n")


def write_readme(output_dir):
    lines = [
        "# Meeting Package: Counterfactual Interpretability",
        "",
        "This folder collects the most useful material for discussing the current counterfactual results.",
        "",
        "## Main Story",
        "",
        "- The CFProto-nearer prototype-guided run is the only retained prototype-guided result.",
        "- Its changes can be model-valid without being medically plausible.",
        "- Retrieval-NUN retrieves real target-class cases and is visually intuitive, but it is not a minimal image edit.",
        "- SEDC-T provides localized region changes and is the clearest region-based method, but validity is lower, especially on Pneumonia.",
        "- SEDC-T tuning improves Pneumonia only slightly, suggesting a method/data limitation rather than a simple parameter issue.",
        "- DVCE covers the generative method category, but outputs remain sensitive to checkpoint and guidance settings.",
        "- Validity means target-class model prediction, not medical plausibility.",
        "",
        "## Files",
        "",
        "- `selected_examples.md`: concrete good, difficult, and failure examples.",
        "- `method_tradeoff_table.md`: compact quantitative method trade-off table.",
        "- `selected_examples.json`: machine-readable selected example metadata.",
        "- `images/`: copied plot images, if `--copy_assets` was used.",
        "",
        "## Suggested Discussion Questions",
        "",
        "1. Which examples are acceptable as visually interpretable for the report/presentation?",
        "2. Should the main conclusion emphasize validity/locality/plausibility trade-offs instead of a single best method?",
        "3. Is the SEDC-T Pneumonia limitation acceptable as a reported result, or should it be framed mainly as a failure case?",
        "4. Should DVCE remain a feasibility result with five samples, or should it be expanded despite runtime and artifacts?",
    ]
    (output_dir / "README.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", nargs="+", required=True)
    parser.add_argument("--output_dir", default="results/meeting_paul_tuesday")
    parser.add_argument("--copy_assets", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    grouped = {}
    summaries = []
    missing_images = []

    for metadata_path in args.metadata:
        metadata_path = Path(metadata_path)
        metadata = load_json(metadata_path)
        summaries.append(method_summary(metadata_path, metadata))
        normalized_records = [
            normalize_record(metadata_path, metadata, record)
            for record in metadata.get("records", [])
        ]
        key = (infer_method(metadata), infer_dataset(metadata), str(metadata_path))
        grouped[key] = select_examples(normalized_records)
        for record in normalized_records:
            image_path = record.get("image_path")
            if image_path and not Path(image_path).exists():
                missing_images.append(image_path)

    write_readme(output_dir)
    write_selected_examples(grouped, output_dir, args.copy_assets)
    write_tradeoff_table(summaries, output_dir)

    with open(output_dir / "selection_report.json", "w") as f:
        json.dump(
            {
                "metadata_files": args.metadata,
                "missing_images": sorted(set(missing_images)),
            },
            f,
            indent=4,
        )

    print(f"Saved meeting package to {output_dir}")
    if missing_images:
        print(f"Warning: {len(set(missing_images))} referenced images are missing.")


if __name__ == "__main__":
    main()
