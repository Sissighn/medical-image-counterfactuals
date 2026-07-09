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
    if method == "SEDC-T-style targeted segment replacement":
        method = "Removed non-final SEDC-T result"
    if method == "SEDC-T original-style best-first segment replacement":
        method = "SEDC-T original-style best-first"
    if method == "SEDC-T-style lung-field ROI ablation":
        method = "SEDC-T lung-field ROI ablation"
    if method == "Retrieval-based nearest-unlike-neighbor baseline":
        method = "Removed retrieval-NUN baseline"
    if method == (
        "Goyal et al. 2019 counterfactual visual explanations "
        "(greedy exhaustive search)"
    ):
        method = "Goyal 2019 counterfactual visual explanations"
    diffusion_checkpoint_path = str(metadata.get("diffusion_checkpoint_path") or "")
    if method == "DVCE medical multi-sample generation evaluation":
        checkpoint_lower = diffusion_checkpoint_path.lower()
        runner_settings = metadata.get("runner_settings", {}) or {}
        guidance_space = runner_settings.get("guidance_space") or metadata.get(
            "guidance_space"
        )
        original_style = (
            guidance_space == "pred_xstart"
            or runner_settings.get("gen_type") == "p_sample"
        )
        prefix = (
            "DVCE original-style medical generation"
            if original_style
            else "Removed legacy DVCE free-guidance result"
        )
        if original_style:
            prefix = (
                f"{prefix} with Cone Projection"
                if metadata.get("cone_projection_enabled")
                else f"{prefix} without Cone Projection"
            )

        if "ema_0.9999_005000" in checkpoint_lower or "pneumonia" in checkpoint_lower:
            method = f"{prefix} with Pneumonia fine-tuned checkpoint"
        elif "busi" in checkpoint_lower:
            method = f"{prefix} with BUSI fine-tuned checkpoint"
        elif "256x256_diffusion_uncond" in checkpoint_lower:
            method = f"{prefix} with OpenAI checkpoint"
        else:
            method = prefix
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
        or aggregate.get("mean_l1_norm")
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
            "- DVCE rows should use the original-code-nearer pred_xstart guidance core without Cone Projection unless explicitly stated otherwise.",
            "- For DVCE rows, Mean change prioritizes the existing project metric and falls back to original-style L1 norm only when needed.",
            "- CFProto follows `alibi.explainers.cfproto.CounterfactualProto` faithfully (FISTA with shrinkage-thresholding, untargeted hinge attack loss, binary c-search, encoder-space class prototypes); see `results/final_configs/cfproto_encoder_method_documentation.md` for the full Soll-Ist comparison. Not reproduced: the original TensorFlow graph itself (reimplemented in PyTorch), black-box/numerical-gradient mode, categorical variables and k-d-tree prototypes, and TrustScore filtering (disabled by default in alibi too).",
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
