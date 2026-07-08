import argparse
import csv
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_cfproto_pytorch import (
    compute_attack_loss,
    compute_autoencoder_latent_prototypes,
    compute_feature_prototypes,
    denormalize,
    get_attack_consts,
    load_autoencoder_model,
    load_checkpoint_model,
    normalize,
    total_variation,
)
from src.data_utils import create_dataloaders
from src.evaluation_manifest import load_image_from_manifest_record, load_manifest_records
from src.train_model import get_device


DATASET_CONFIGS = {
    "busi": {
        "model_path": "models/busi_resnet18_pretrained.pth",
        "dataset_path": "data/processed/BUSI",
        "manifest_path": "results/evaluation_manifests/busi_balanced_5_per_class_second_best.json",
        "autoencoder_path": "models/autoencoder_busi.pth",
    },
    "pneumonia": {
        "model_path": "models/pneumonia_resnet18_pretrained.pth",
        "dataset_path": "data/processed/Pneumonia",
        "manifest_path": "results/evaluation_manifests/pneumonia_balanced_10_per_class_second_best.json",
        "autoencoder_path": "models/autoencoder_pneumonia.pth",
    },
}


def existing_dataset_configs(dataset_names):
    configs = {}
    for name in dataset_names:
        config = DATASET_CONFIGS[name]
        missing = [key for key, value in config.items() if not Path(value).exists()]
        if missing:
            print(
                f"Skipping {name}: missing files for keys {missing}",
                file=sys.stderr,
            )
            continue
        configs[name] = config
    return configs


def load_manifest_samples(model, test_dataset, manifest_path, max_samples, device):
    manifest, records = load_manifest_records(manifest_path, max_records=max_samples)
    samples = []

    with torch.no_grad():
        for record in records:
            image, label, image_path = load_image_from_manifest_record(
                test_dataset, record
            )
            image = image.to(device)
            logits, features = model(image)
            probabilities = F.softmax(logits, dim=1)
            prediction = int(torch.argmax(probabilities, dim=1).item())
            expected_prediction = int(record["original_prediction_index"])
            if prediction != expected_prediction:
                raise ValueError(
                    "Current model prediction does not match manifest for "
                    f"sample {record['manifest_sample_index']}: "
                    f"manifest={expected_prediction}, current={prediction}"
                )

            samples.append(
                {
                    "manifest_sample_index": int(record["manifest_sample_index"]),
                    "dataset_index": int(record["dataset_index"]),
                    "image_path": image_path,
                    "image": image.detach(),
                    "label": int(label),
                    "original_prediction": prediction,
                    "original_probabilities": probabilities[0].detach().cpu().tolist(),
                    "target_class": int(record["target_class_index"]),
                    "features": features[0].detach().cpu(),
                }
            )

    return manifest, samples


def compute_weighted_terms(loss_terms, parameters):
    return {
        "weighted_class_loss": parameters["attack_const"] * loss_terms["class_loss"],
        "weighted_l1_loss": parameters["lambda_l1"] * loss_terms["l1_loss"],
        "weighted_l2_loss": parameters["lambda_l2"] * loss_terms["l2_loss"],
        "weighted_tv_loss": parameters["lambda_tv"] * loss_terms["tv_loss"],
        "weighted_proto_loss": parameters["lambda_proto"] * loss_terms["proto_loss"],
        "weighted_ae_loss": parameters["gamma"] * loss_terms["ae_loss"],
    }


def run_loss_probe(
    model,
    autoencoder,
    image,
    target_class,
    target_prototype,
    prototype_space,
    parameters,
):
    original_pixels = denormalize(image).detach()
    perturbation = torch.zeros(
        1,
        1,
        parameters["perturbation_resolution"],
        parameters["perturbation_resolution"],
        device=image.device,
        requires_grad=True,
    )
    optimizer = torch.optim.Adam([perturbation], lr=parameters["learning_rate"])
    target = torch.tensor([target_class], device=image.device)
    step_rows = []

    for step in range(1, parameters["steps"] + 1):
        optimizer.zero_grad()

        smooth_delta = F.interpolate(
            torch.tanh(perturbation),
            size=original_pixels.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).repeat(1, 3, 1, 1)
        cf_pixels = (
            original_pixels + parameters["max_delta"] * smooth_delta
        ).clamp(0.0, 1.0)
        cf_normalized = normalize(cf_pixels)
        logits, features = model(cf_normalized)
        probabilities = F.softmax(logits, dim=1)

        class_loss = compute_attack_loss(
            logits=logits,
            probabilities=probabilities,
            target=target,
            target_class=target_class,
            attack_loss=parameters["attack_loss"],
            kappa=parameters["kappa"],
        )
        l1_loss = torch.mean(torch.abs(cf_pixels - original_pixels))
        l2_loss = F.mse_loss(cf_pixels, original_pixels)
        tv_loss = total_variation(smooth_delta)
        if prototype_space == "encoder":
            latent = torch.flatten(autoencoder.encode(cf_pixels), start_dim=1)
            proto_loss = F.mse_loss(latent[0], target_prototype)
        else:
            proto_loss = F.mse_loss(features[0], target_prototype)
        ae_loss = (
            F.mse_loss(autoencoder(cf_pixels), cf_pixels)
            if autoencoder is not None
            else torch.tensor(0.0, device=image.device)
        )

        loss_terms = {
            "class_loss": float(class_loss.detach().item()),
            "l1_loss": float(l1_loss.detach().item()),
            "l2_loss": float(l2_loss.detach().item()),
            "tv_loss": float(tv_loss.detach().item()),
            "proto_loss": float(proto_loss.detach().item()),
            "ae_loss": float(ae_loss.detach().item()),
        }
        weighted_terms = compute_weighted_terms(loss_terms, parameters)
        total_loss = sum(weighted_terms.values())

        loss = (
            parameters["attack_const"] * class_loss
            + parameters["lambda_l1"] * l1_loss
            + parameters["lambda_l2"] * l2_loss
            + parameters["lambda_tv"] * tv_loss
            + parameters["lambda_proto"] * proto_loss
            + parameters["gamma"] * ae_loss
        )
        loss.backward()
        optimizer.step()

        step_rows.append(
            {
                "step": step,
                **loss_terms,
                **weighted_terms,
                "total_loss": float(total_loss),
                "prediction": int(torch.argmax(probabilities, dim=1).item()),
                "target_confidence": float(probabilities[0, target_class].item()),
            }
        )

    with torch.no_grad():
        final_delta = F.interpolate(
            torch.tanh(perturbation),
            size=original_pixels.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).repeat(1, 3, 1, 1)
        final_pixels = (
            original_pixels + parameters["max_delta"] * final_delta
        ).clamp(0.0, 1.0)
        final_logits, _ = model(normalize(final_pixels))
        final_probabilities = F.softmax(final_logits, dim=1)
        final_prediction = int(torch.argmax(final_probabilities, dim=1).item())

    return {
        "final_prediction": final_prediction,
        "final_target_confidence": float(final_probabilities[0, target_class].item()),
        "valid_counterfactual": final_prediction == target_class,
        "step_rows": step_rows,
        "final_loss_row": step_rows[-1],
    }


def summarize_ratios(rows):
    summary = {}
    for space in ["resnet", "encoder"]:
        matching = [row for row in rows if row["prototype_space"] == space]
        if not matching:
            continue
        for field in ["proto_loss", "weighted_proto_loss", "total_loss"]:
            values = [float(row[field]) for row in matching]
            summary[f"{space}_{field}_mean"] = sum(values) / len(values)
    if "resnet_weighted_proto_loss_mean" in summary and "encoder_weighted_proto_loss_mean" in summary:
        denominator = max(summary["resnet_weighted_proto_loss_mean"], 1e-12)
        summary["encoder_to_resnet_weighted_proto_loss_ratio"] = (
            summary["encoder_weighted_proto_loss_mean"] / denominator
        )
    return summary


def write_interpretation(output_dir, rows, aggregate, parameters):
    path = Path(output_dir) / "loss_scale_interpretation.md"
    ratio = aggregate.get("encoder_to_resnet_weighted_proto_loss_ratio")
    lines = [
        "# CFProto Prototype-Space Loss Scale Check",
        "",
        "This debug run checks loss magnitudes for the prototype-guided PyTorch",
        "baseline before any larger fixed-manifest result runs. It is not a final",
        "counterfactual quality evaluation.",
        "",
        "## Configuration",
        "",
        f"- Samples per dataset: {parameters['max_samples']}",
        f"- Optimization steps: {parameters['steps']}",
        f"- Attack loss: `{parameters['attack_loss']}`",
        f"- `lambda_proto`: {parameters['lambda_proto']}",
        f"- `gamma`: {parameters['gamma']}",
        "",
        "## Prototype Loss Scale",
        "",
    ]

    for space in ["resnet", "encoder"]:
        proto_key = f"{space}_proto_loss_mean"
        weighted_key = f"{space}_weighted_proto_loss_mean"
        total_key = f"{space}_total_loss_mean"
        if proto_key in aggregate:
            lines.extend(
                [
                    f"- `{space}` mean raw prototype loss: {aggregate[proto_key]:.6f}",
                    f"- `{space}` mean weighted prototype loss: {aggregate[weighted_key]:.6f}",
                    f"- `{space}` mean total loss: {aggregate[total_key]:.6f}",
                ]
            )

    if ratio is not None:
        lines.extend(
            [
                "",
                f"The encoder/resnet weighted prototype-loss ratio is {ratio:.3f}.",
            ]
        )

    lines.extend(
        [
            "",
            "## Cautious Interpretation",
            "",
            "- `prototype_space=encoder` runs technically and produces finite loss values.",
            "- Validity is logged only as a debugging signal because the run uses too few",
            "  steps for counterfactual quality assessment.",
            "- If the weighted encoder prototype loss is much larger or much smaller than",
            "  the legacy ResNet value, `lambda_proto` should be adjusted before larger",
            "  result runs.",
            "- A small fair 300-step comparison is only sensible after this scale check",
            "  suggests that the weighted prototype loss is not dominating or vanishing.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def write_outputs(output_dir, final_rows, step_histories, aggregate, parameters):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "loss_scale_summary.csv"
    fieldnames = list(final_rows[0].keys()) if final_rows else []
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(final_rows)

    json_path = output_dir / "loss_scale_summary.json"
    with open(json_path, "w") as f:
        json.dump(
            {
                "parameters": parameters,
                "aggregate": aggregate,
                "final_rows": final_rows,
                "step_histories": step_histories,
            },
            f,
            indent=4,
        )

    write_interpretation(output_dir, final_rows, aggregate, parameters)
    return csv_path, json_path, output_dir / "loss_scale_interpretation.md"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["busi", "pneumonia"],
        default=["busi"],
        help="Datasets to check. Defaults to BUSI to keep the smoke test small.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/debug/cfproto_loss_scales",
    )
    parser.add_argument("--max_samples", type=int, default=2)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--attack_loss", choices=["cross_entropy", "cw_hinge"], default="cw_hinge")
    parser.add_argument("--attack_const", type=float, default=1.0)
    parser.add_argument("--c_init", type=float, default=1.0)
    parser.add_argument("--c_steps", type=int, default=1)
    parser.add_argument("--kappa", type=float, default=0.0)
    parser.add_argument("--lambda_l1", type=float, default=0.01)
    parser.add_argument("--lambda_l2", type=float, default=5.0)
    parser.add_argument("--lambda_tv", type=float, default=0.2)
    parser.add_argument("--lambda_proto", type=float, default=0.05)
    parser.add_argument("--gamma", type=float, default=0.0)
    parser.add_argument("--max_delta", type=float, default=0.12)
    parser.add_argument("--perturbation_resolution", type=int, default=28)
    args = parser.parse_args()

    if args.max_samples > 2:
        raise ValueError("This debug script is intentionally capped at max_samples <= 2.")
    if args.steps > 20:
        raise ValueError("This debug script is intentionally capped at steps <= 20.")

    device = get_device()
    configs = existing_dataset_configs(args.datasets)
    if not configs:
        raise RuntimeError("No complete dataset configuration is available.")

    parameters = {
        "max_samples": args.max_samples,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "attack_loss": args.attack_loss,
        "attack_const": args.attack_const,
        "c_init": args.c_init,
        "c_steps": args.c_steps,
        "attack_consts": get_attack_consts(args.attack_const, args.c_init, args.c_steps),
        "kappa": args.kappa,
        "lambda_l1": args.lambda_l1,
        "lambda_l2": args.lambda_l2,
        "lambda_tv": args.lambda_tv,
        "lambda_proto": args.lambda_proto,
        "gamma": args.gamma,
        "max_delta": args.max_delta,
        "perturbation_resolution": args.perturbation_resolution,
    }

    final_rows = []
    step_histories = []
    for dataset_name, config in configs.items():
        print(f"Checking {dataset_name} on {device}...")
        model, checkpoint = load_checkpoint_model(config["model_path"], device)
        classes = checkpoint["classes"]
        data = create_dataloaders(
            config["dataset_path"],
            batch_size=args.batch_size,
            use_augmentation=False,
        )
        autoencoder, _ = load_autoencoder_model(config["autoencoder_path"], device)
        _, samples = load_manifest_samples(
            model,
            data["test_dataset"],
            config["manifest_path"],
            args.max_samples,
            device,
        )

        prototype_sets = {
            "resnet": compute_feature_prototypes(
                model, data["train_loader"], len(classes), device
            ),
            "encoder": compute_autoencoder_latent_prototypes(
                autoencoder, data["train_loader"], len(classes), device
            ),
        }

        for prototype_space, prototypes in prototype_sets.items():
            for sample in samples:
                target_class = sample["target_class"]
                result = run_loss_probe(
                    model=model,
                    autoencoder=autoencoder,
                    image=sample["image"],
                    target_class=target_class,
                    target_prototype=prototypes[target_class],
                    prototype_space=prototype_space,
                    parameters=parameters,
                )
                final_loss_row = result["final_loss_row"]
                row = {
                    "dataset": dataset_name,
                    "prototype_space": prototype_space,
                    "prototype_source": (
                        "autoencoder_encoder"
                        if prototype_space == "encoder"
                        else "resnet18_penultimate_features"
                    ),
                    "prototype_shape": "x".join(str(value) for value in prototypes.shape),
                    "manifest_sample_index": sample["manifest_sample_index"],
                    "dataset_index": sample["dataset_index"],
                    "true_label": classes[sample["label"]],
                    "original_prediction": classes[sample["original_prediction"]],
                    "target_class": classes[target_class],
                    "final_prediction": classes[result["final_prediction"]],
                    "target_confidence": result["final_target_confidence"],
                    "valid_counterfactual": result["valid_counterfactual"],
                    **{
                        key: final_loss_row[key]
                        for key in [
                            "class_loss",
                            "l1_loss",
                            "l2_loss",
                            "tv_loss",
                            "proto_loss",
                            "ae_loss",
                            "weighted_class_loss",
                            "weighted_l1_loss",
                            "weighted_l2_loss",
                            "weighted_tv_loss",
                            "weighted_proto_loss",
                            "weighted_ae_loss",
                            "total_loss",
                        ]
                    },
                }
                final_rows.append(row)
                step_histories.append(
                    {
                        "dataset": dataset_name,
                        "prototype_space": prototype_space,
                        "manifest_sample_index": sample["manifest_sample_index"],
                        "target_class": classes[target_class],
                        "steps": result["step_rows"],
                    }
                )

    aggregate = summarize_ratios(final_rows)
    csv_path, json_path, md_path = write_outputs(
        args.output_dir, final_rows, step_histories, aggregate, parameters
    )
    print(f"Saved CSV: {csv_path}")
    print(f"Saved JSON: {json_path}")
    print(f"Saved interpretation: {md_path}")


if __name__ == "__main__":
    main()
