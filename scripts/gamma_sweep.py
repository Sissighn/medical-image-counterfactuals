import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_gamma_values(values):
    return [float(value) for value in values.split(",") if value.strip()]


def mean(values):
    return sum(values) / len(values) if values else 0.0


def std(values):
    if len(values) <= 1:
        return 0.0
    value_mean = mean(values)
    return math.sqrt(sum((value - value_mean) ** 2 for value in values) / len(values))


def get_change_metric(metrics, name):
    if name in metrics:
        return metrics[name]
    if name == "changed_pixel_fraction_threshold_0_03":
        return metrics.get("changed_pixel_fraction", 0.0)
    return 0.0


def summarize_metadata(metadata_path, gamma):
    with open(metadata_path) as f:
        metadata = json.load(f)

    records = metadata["records"]
    valid_count = sum(record["valid_counterfactual"] for record in records)
    loss_terms = [record.get("loss_terms") or {} for record in records]
    change_metrics = [record["change_metrics"] for record in records]

    mean_ae_loss = mean([terms.get("ae_loss", 0.0) for terms in loss_terms])
    l1_changes = [metrics["l1_mean"] for metrics in change_metrics]
    l2_changes = [metrics["l2_mean"] for metrics in change_metrics]
    linf_changes = [metrics["linf"] for metrics in change_metrics]
    target_confidences = [
        record["counterfactual_probabilities"][record["target_class_index"]]
        for record in records
    ]
    row = {
        "gamma": gamma,
        "num_samples": len(records),
        "valid_count": valid_count,
        "validity": valid_count / len(records) if records else 0.0,
        "mean_class_loss": mean([terms.get("class_loss", 0.0) for terms in loss_terms]),
        "mean_l1_loss": mean([terms.get("l1_loss", 0.0) for terms in loss_terms]),
        "mean_l2_loss": mean([terms.get("l2_loss", 0.0) for terms in loss_terms]),
        "mean_tv_loss": mean([terms.get("tv_loss", 0.0) for terms in loss_terms]),
        "mean_proto_loss": mean([terms.get("proto_loss", 0.0) for terms in loss_terms]),
        "mean_ae_loss": mean_ae_loss,
        "mean_autoencoder_loss": mean_ae_loss,
        "mean_weighted_ae_contribution": gamma * mean_ae_loss,
        "mean_weighted_autoencoder_loss": gamma * mean_ae_loss,
        "mean_l1_change": mean(l1_changes),
        "mean_l2_change": mean(l2_changes),
        "mean_linf_change": mean(linf_changes),
        "std_l1_change": std(l1_changes),
        "std_l2_change": std(l2_changes),
        "std_linf_change": std(linf_changes),
        "mean_changed_pixel_fraction": mean(
            [metrics["changed_pixel_fraction"] for metrics in change_metrics]
        ),
        "mean_changed_pixel_fraction_0_03": mean(
            [
                get_change_metric(metrics, "changed_pixel_fraction_threshold_0_03")
                for metrics in change_metrics
            ]
        ),
        "mean_changed_pixel_fraction_0_01": mean(
            [
                get_change_metric(metrics, "changed_pixel_fraction_threshold_0_01")
                for metrics in change_metrics
            ]
        ),
        "mean_changed_pixel_fraction_0_005": mean(
            [
                get_change_metric(metrics, "changed_pixel_fraction_threshold_0_005")
                for metrics in change_metrics
            ]
        ),
        "mean_changed_pixel_fraction_0_001": mean(
            [
                get_change_metric(metrics, "changed_pixel_fraction_threshold_0_001")
                for metrics in change_metrics
            ]
        ),
        "mean_cf_confidence": mean(
            [record["counterfactual_confidence"] for record in records]
        ),
        "mean_target_confidence": mean(target_confidences),
        "mean_runtime_seconds": mean(
            [record["runtime_seconds"] for record in records]
        ),
        "output_dir": str(metadata_path.parent),
    }
    return row


def has_stable_improvement(rows, metric_name):
    baseline = rows[0][metric_name]
    improved = [row[metric_name] < baseline for row in rows[1:]]
    return sum(improved) >= max(2, len(improved) // 2 + 1)


def choose_recommendation(rows):
    if not rows:
        return "No recommendation possible because no sweep rows were generated."

    stable_validity = all(row["validity"] >= rows[0]["validity"] for row in rows)
    l1_improves = has_stable_improvement(rows, "mean_l1_change")
    l2_improves = has_stable_improvement(rows, "mean_l2_change")
    ae_decreases = has_stable_improvement(rows, "mean_autoencoder_loss")

    if stable_validity and l1_improves and l2_improves and ae_decreases:
        return (
            "Use as a cautious ablation candidate: validity remains stable and "
            "several proximity/AE metrics improve relative to gamma=0."
        )
    if stable_validity and (l1_improves or l2_improves):
        return (
            "Use only as an ablation: validity remains stable, but the improvement "
            "trend is partial rather than robust across all CFProto-relevant metrics."
        )
    return (
        "Do not use as a main configuration: the autoencoder term is technically "
        "implemented, but this sweep does not show a robust additional benefit."
    )


def write_interpretation(output_dir, rows, config):
    output_dir = Path(output_dir)
    path = output_dir / "gamma_sweep_interpretation.md"
    recommendation = choose_recommendation(rows)

    columns = [
        "gamma",
        "validity",
        "mean_cf_confidence",
        "mean_target_confidence",
        "mean_l1_change",
        "mean_l2_change",
        "mean_linf_change",
        "mean_changed_pixel_fraction_0_03",
        "mean_changed_pixel_fraction_0_01",
        "mean_changed_pixel_fraction_0_005",
        "mean_changed_pixel_fraction_0_001",
        "mean_autoencoder_loss",
        "mean_weighted_autoencoder_loss",
    ]

    lines = [
        f"# Gamma Sweep Interpretation: {config['dataset_name']}",
        "",
        "## Setup",
        "",
        f"- Dataset: {config['dataset_name']}",
        f"- Samples: {config['manifest_max_samples']}",
        f"- Steps: {config['steps']}",
        f"- Gamma values: {', '.join(str(value) for value in config['gamma_values'])}",
        f"- Attack loss: `{config['attack_loss']}`",
        f"- Lambda L1/L2/TV/Prototype: {config['lambda_l1']}, {config['lambda_l2']}, {config['lambda_tv']}, {config['lambda_proto']}",
        f"- Autoencoder path: `{config['autoencoder_path']}`",
        "",
        "## Summary Table",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                values.append(f"{value:.6f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")

    baseline = rows[0] if rows else None
    lines.extend(
        [
            "",
            "## CFProto-Oriented Interpretation",
            "",
            "The sweep tests the CFProto-inspired autoencoder term "
            "`gamma * MSE(AE(counterfactual), counterfactual)`. In CFProto, this "
            "type of term is intended to regularize counterfactuals toward more "
            "data-like regions. In this project, the autoencoder is frozen and "
            "used as an additional PyTorch loss term, but the implementation is "
            "still not a full Alibi `CounterfactualProto` reproduction.",
            "",
        ]
    )
    if baseline is not None:
        lines.extend(
            [
                f"Compared with gamma=0, validity at the tested gamma values is "
                f"{'stable' if all(row['validity'] >= baseline['validity'] for row in rows) else 'not uniformly stable'}.",
                "The smaller changed-pixel thresholds are especially important for "
                "Pneumonia, because threshold 0.03 can hide small but non-zero changes.",
                "",
                f"Recommendation: {recommendation}",
            ]
        )

    path.write_text("\n".join(lines) + "\n")
    return path


def write_alignment_note(output_root):
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / "cfproto_autoencoder_alignment.md"
    path.write_text(
        """# CFProto Autoencoder Alignment

## Tested CFProto Idea

This experiment tests the optional autoencoder reconstruction loss used in
CFProto-style counterfactual optimization. The intended role is to penalize
counterfactual images that are hard for an autoencoder trained on the data
distribution to reconstruct:

```text
ae_loss = MSE(AE(counterfactual), counterfactual)
total_loss += gamma * ae_loss
```

`gamma` controls how strongly this plausibility term affects the optimization.

## Covered By This Project

- A ConvAutoencoder checkpoint is loaded from disk.
- The autoencoder is frozen and used in `eval()` mode.
- The loss is computed on denormalized `[0, 1]` counterfactual pixels.
- The total PyTorch optimization loss includes `gamma * ae_loss`.
- Raw loss terms are logged for gamma calibration.

## Still Not Full Alibi CFProto

- No TensorFlow Alibi `CounterfactualProto` graph is used.
- No FISTA shrinkage optimizer is implemented.
- No full adaptive binary search over `c` is implemented.
- Prototypes are computed in the medical ResNet18 feature space, not in an
  independent encoder/k-d-tree prototype space.

## Scientific Wording

If the gamma sweep does not show a robust metric trend, the autoencoder term
should not be presented as an improvement. It should be described as a
CFProto-inspired regularization ablation. If a robust trend appears, it should
still be phrased cautiously: the term may stabilize proximity or
autoencoder-loss metrics without reducing validity, but it is not a clinical
plausibility guarantee.
""",
    )
    return path


def run_gamma_sweep(
    model_path,
    dataset_path,
    autoencoder_path,
    manifest_path,
    output_dir,
    gamma_values,
    manifest_max_samples,
    steps,
    batch_size,
    attack_loss,
    lambda_l1,
    lambda_l2,
    lambda_tv,
    lambda_proto,
    max_delta,
    c_init,
    c_steps,
    kappa,
    dataset_name,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for gamma in gamma_values:
        gamma_label = str(gamma).replace(".", "p").replace("-", "m")
        gamma_output_dir = output_dir / f"gamma_{gamma_label}"
        command = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "run_cfproto_pytorch.py"),
            "--model_path",
            model_path,
            "--dataset_path",
            dataset_path,
            "--output_dir",
            str(gamma_output_dir),
            "--manifest_path",
            manifest_path,
            "--manifest_max_samples",
            str(manifest_max_samples),
            "--steps",
            str(steps),
            "--batch_size",
            str(batch_size),
            "--attack_loss",
            attack_loss,
            "--lambda_l1",
            str(lambda_l1),
            "--lambda_l2",
            str(lambda_l2),
            "--lambda_tv",
            str(lambda_tv),
            "--lambda_proto",
            str(lambda_proto),
            "--max_delta",
            str(max_delta),
            "--c_init",
            str(c_init),
            "--c_steps",
            str(c_steps),
            "--kappa",
            str(kappa),
            "--autoencoder_path",
            autoencoder_path,
            "--gamma",
            str(gamma),
            "--log_loss_terms",
        ]

        print(f"\nRunning gamma={gamma}")
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)
        rows.append(summarize_metadata(gamma_output_dir / "metadata.json", gamma))

    csv_path = output_dir / "gamma_sweep_summary.csv"
    json_path = output_dir / "gamma_sweep_summary.json"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with open(json_path, "w") as f:
        json.dump(rows, f, indent=4)
    interpretation_path = write_interpretation(
        output_dir=output_dir,
        rows=rows,
        config={
            "dataset_name": dataset_name,
            "manifest_max_samples": manifest_max_samples,
            "steps": steps,
            "gamma_values": gamma_values,
            "attack_loss": attack_loss,
            "lambda_l1": lambda_l1,
            "lambda_l2": lambda_l2,
            "lambda_tv": lambda_tv,
            "lambda_proto": lambda_proto,
            "autoencoder_path": autoencoder_path,
        },
    )
    alignment_path = write_alignment_note(output_dir.parent)

    print(f"\nSaved gamma sweep CSV to: {csv_path}")
    print(f"Saved gamma sweep JSON to: {json_path}")
    print(f"Saved gamma sweep interpretation to: {interpretation_path}")
    print(f"Saved CFProto alignment note to: {alignment_path}")
    print("\nGamma sweep summary:")
    for row in rows:
        print(
            f"gamma={row['gamma']:g} | "
            f"validity={row['validity']:.2f} | "
            f"weighted_AE={row['mean_weighted_ae_contribution']:.6f} | "
            f"L1={row['mean_l1_change']:.6f} | "
            f"L2={row['mean_l2_change']:.6f} | "
            f"changed@0.03={row['mean_changed_pixel_fraction_0_03']:.4f} | "
            f"changed@0.005={row['mean_changed_pixel_fraction_0_005']:.4f}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--autoencoder_path", type=str, required=True)
    parser.add_argument("--manifest_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--gamma_values", type=str, default="0.0,0.5,2.0,5.0,10.0")
    parser.add_argument("--manifest_max_samples", type=int, default=5)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--attack_loss", choices=["cross_entropy", "cw_hinge"], default="cw_hinge")
    parser.add_argument("--lambda_l1", type=float, default=0.01)
    parser.add_argument("--lambda_l2", type=float, default=5.0)
    parser.add_argument("--lambda_tv", type=float, default=0.2)
    parser.add_argument("--lambda_proto", type=float, default=0.05)
    parser.add_argument("--max_delta", type=float, default=0.12)
    parser.add_argument("--c_init", type=float, default=1.0)
    parser.add_argument("--c_steps", type=int, default=1)
    parser.add_argument("--kappa", type=float, default=0.0)
    args = parser.parse_args()

    run_gamma_sweep(
        model_path=args.model_path,
        dataset_path=args.dataset_path,
        autoencoder_path=args.autoencoder_path,
        manifest_path=args.manifest_path,
        output_dir=args.output_dir,
        gamma_values=parse_gamma_values(args.gamma_values),
        manifest_max_samples=args.manifest_max_samples,
        steps=args.steps,
        batch_size=args.batch_size,
        attack_loss=args.attack_loss,
        lambda_l1=args.lambda_l1,
        lambda_l2=args.lambda_l2,
        lambda_tv=args.lambda_tv,
        lambda_proto=args.lambda_proto,
        max_delta=args.max_delta,
        c_init=args.c_init,
        c_steps=args.c_steps,
        kappa=args.kappa,
        dataset_name=args.dataset_name or Path(args.dataset_path).name,
    )


if __name__ == "__main__":
    main()
