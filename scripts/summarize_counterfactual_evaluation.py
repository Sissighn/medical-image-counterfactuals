import argparse
import json
from pathlib import Path


def load_metadata(path):
    with open(path) as f:
        metadata = json.load(f)
    return metadata


def infer_dataset_name(metadata):
    dataset_path = str(metadata.get("dataset_path") or "")
    lowered = dataset_path.lower()
    if "busi" in lowered:
        return "BUSI"
    if "pneumonia" in lowered:
        return "Pneumonia"
    classifier_adapter = metadata.get("classifier_adapter", {})
    dataset_path = str(classifier_adapter.get("dataset_path") or "")
    lowered = dataset_path.lower()
    if "busi" in lowered:
        return "BUSI"
    if "pneumonia" in lowered:
        return "Pneumonia"
    return "unknown"


def get_aggregate(metadata):
    return metadata.get("aggregate_metrics", {})


def format_float(value, digits=4):
    if value is None:
        return ""
    return f"{float(value):.{digits}f}"


def metric_mean(aggregate, key):
    value = aggregate.get(key)
    if isinstance(value, dict):
        return value.get("mean")
    return value


def record_mean(records, key):
    values = [
        record.get(key)
        for record in records
        if isinstance(record.get(key), (int, float))
    ]
    if not values:
        return None
    return float(sum(values) / len(values))


def summarize_metadata(path):
    metadata = load_metadata(path)
    aggregate = get_aggregate(metadata)
    records = metadata.get("records", [])

    method = metadata.get("method") or metadata.get("purpose") or "unknown"
    parameters = metadata.get("parameters", {})
    if method == "PyTorch prototype-guided optimization baseline":
        if (
            parameters.get("prototype_space") == "encoder"
            and parameters.get("prototype_mode") == "knn_mean"
            and parameters.get("c_search_mode") == "adaptive_binary"
            and parameters.get("selection_metric") == "elastic_net"
        ):
            method = "cfproto_encoder_knn final CFProto-nearer prototype-guided method"
        elif parameters.get("prototype_space") == "encoder":
            method = "Prototype-guided encoder-space CFProto-near ablation"
        elif (
            parameters.get("lambda_l2", 0) > 5.0
            or parameters.get("lambda_tv", 0) > 0.2
            or parameters.get("max_delta", 1.0) < 0.12
        ):
            method = "Prototype-guided plausibility ablation"
        if (
            parameters.get("attack_loss") == "cw_hinge"
            or parameters.get("lambda_l1", 0) > 0
            or parameters.get("c_steps", 1) > 1
            or parameters.get("target_strategy") == "prototype_distance"
        ):
            if method == "PyTorch prototype-guided optimization baseline":
                method = "Prototype-guided legacy CFProto-aligned ablation"
        elif method == "PyTorch prototype-guided optimization baseline":
            method = "Prototype-guided legacy ResNet/class-mean baseline"
    if method == "SEDC-T-style targeted segment replacement":
        search_mode = parameters.get("search_mode")
        roi_mode = parameters.get("roi_mode") or "none"
        max_segments = parameters.get("max_segments")
        if search_mode == "greedy_minimal" and max_segments and max_segments > 6:
            method = f"SEDC-T tuned project variant ({roi_mode}, max {max_segments})"
        elif roi_mode != "none":
            method = f"SEDC-T project variant ({roi_mode})"
        else:
            method = "SEDC-T project variant"
    if method == "Retrieval-based nearest-unlike-neighbor baseline":
        method = "Retrieval-based nearest-unlike-neighbor baseline"
    diffusion_checkpoint_path = str(metadata.get("diffusion_checkpoint_path") or "")
    if method == "DVCE medical multi-sample generation evaluation":
        if "ema_0.9999_005000" in diffusion_checkpoint_path:
            method = (
                "DVCE medical multi-sample generation evaluation "
                "with Pneumonia fine-tuned checkpoint"
            )
        elif "256x256_diffusion_uncond" in diffusion_checkpoint_path:
            method = (
                "DVCE medical multi-sample generation evaluation "
                "with OpenAI checkpoint"
            )
    dataset = infer_dataset_name(metadata)
    samples = aggregate.get("num_samples") or len(records)
    validity = aggregate.get("validity")
    confidence = aggregate.get("mean_counterfactual_confidence") or record_mean(
        records, "counterfactual_confidence"
    )
    changed = (
        metric_mean(aggregate, "changed_pixel_fraction")
        or aggregate.get("mean_changed_pixels_threshold_0_05")
        or aggregate.get("mean_absolute_difference")
        or metric_mean(aggregate, "l1_mean")
    )
    runtime = metric_mean(aggregate, "runtime_seconds") or aggregate.get(
        "mean_runtime_seconds"
    )

    return {
        "metadata_path": str(path),
        "method": method,
        "dataset": dataset,
        "samples": samples,
        "validity": validity,
        "mean_counterfactual_confidence": confidence,
        "mean_change": changed,
        "mean_runtime_seconds": runtime,
    }


def write_markdown(rows, output_path):
    lines = [
        "# Fixed Counterfactual Evaluation Summary",
        "",
        "This table is generated from method `metadata.json` files.",
        "",
        "| Method | Dataset | Samples | Validity | Mean CF confidence | Mean change | Mean runtime (s) | Metadata |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]

    for row in rows:
        lines.append(
            "| "
            f"{row['method']} | "
            f"{row['dataset']} | "
            f"{row['samples']} | "
            f"{format_float(row['validity'])} | "
            f"{format_float(row['mean_counterfactual_confidence'])} | "
            f"{format_float(row['mean_change'])} | "
            f"{format_float(row['mean_runtime_seconds'], digits=2)} | "
            f"`{row['metadata_path']}` |"
        )

    lines.extend(
        [
            "",
            "## Interpretation Notes",
            "",
            "- Validity only checks whether the model prediction changed to the target class.",
            "- Mean change is method-dependent and should be interpreted together with the qualitative images.",
            "- Medical plausibility must be discussed separately from model validity.",
            "- The `cfproto_encoder_knn` rows are the final CFProto-nearer prototype-guided runs. Earlier prototype-guided rows are legacy baselines or ablations and should not be treated as replacements for the final method.",
            "- The CFProto-nearer implementation still is not a full Alibi CFProto reproduction; FISTA/shrinkage, TrustScore, the original TensorFlow graph, and original Alibi k-d-tree machinery are not fully reproduced.",
        ]
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", nargs="+", required=True)
    parser.add_argument(
        "--output_path",
        type=str,
        default="results/fixed_evaluation_summary.md",
    )
    args = parser.parse_args()

    rows = [summarize_metadata(Path(path)) for path in args.metadata]
    write_markdown(rows, args.output_path)

    json_path = Path(args.output_path).with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump({"rows": rows}, f, indent=4)

    print(f"Saved summary table to {args.output_path}")
    print(f"Saved summary JSON to {json_path}")


if __name__ == "__main__":
    main()
