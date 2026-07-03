"""Create multi-row qualitative comparison figures from selected CF examples.

The script composes existing per-example visualizations into dataset-level
comparison figures. It does not recompute or re-normalize difference maps.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import textwrap
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np


CATEGORY_ORDER = [
    ("best_valid_balanced", "Most balanced valid case"),
    ("highest_confidence_valid", "Highest-confidence valid case"),
    ("visually_questionable_valid", "Low-plausibility valid case"),
    ("failure_case", "Failure case"),
]

METHOD_ORDER = [
    "Prototype-guided optimization baseline",
    "Prototype-guided plausibility ablation",
    "SEDC-T original-style best-first",
    "SEDC-T tuned project variant",
    "DVCE-style, OpenAI checkpoint",
    "DVCE-style, Pneumonia fine-tuned checkpoint",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create qualitative comparison figures from selected counterfactual "
            "examples."
        )
    )
    parser.add_argument(
        "--selected_examples",
        type=Path,
        default=Path("results/meeting_paul_tuesday/selected_examples.json"),
        help="JSON file created by select_interpretable_examples.py.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("results/qualitative_figures"),
        help="Directory for generated qualitative comparison figures.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for saved PNG figures.",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Optional dataset names to include, for example BUSI Pneumonia.",
    )
    parser.add_argument(
        "--include_dataset_overview",
        action="store_true",
        help=(
            "Also create broad dataset-level figures with methods as columns. "
            "By default, only paper-friendly per-method figures are created."
        ),
    )
    return parser.parse_args()


def load_selected_examples(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Selected examples file does not exist: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    examples = data.get("selected_examples", [])
    if not isinstance(examples, list):
        raise ValueError(f"Expected a list at key 'selected_examples' in {path}")
    return examples


def method_sort_key(method: str) -> tuple[int, str]:
    for index, prefix in enumerate(METHOD_ORDER):
        if method.startswith(prefix):
            return index, method
    return len(METHOD_ORDER), method


def slugify(value: str) -> str:
    value = value.lower().strip()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")


def choose_example_path(example: dict[str, Any]) -> tuple[Path | None, str | None]:
    candidates = [
        ("copied_plot_path", example.get("copied_plot_path")),
        ("image_path", example.get("image_path")),
    ]
    for field, value in candidates:
        if not value:
            continue
        path = Path(value)
        if path.exists():
            return path, field
    return None, None


def normalize_image_for_display(image: np.ndarray) -> np.ndarray:
    """Return image data in [0, 1] for display without contrast stretching."""
    if image.dtype.kind in {"u", "i"}:
        max_value = np.iinfo(image.dtype).max
        return image.astype(np.float32) / float(max_value)
    image = image.astype(np.float32)
    if image.size and math.isfinite(float(np.nanmax(image))) and np.nanmax(image) > 1.0:
        image = image / 255.0
    return np.clip(image, 0.0, 1.0)


def format_float(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "n/a"


def format_cell_note(example: dict[str, Any]) -> str:
    valid = "yes" if example.get("valid_counterfactual") else "no"
    sample = example.get("manifest_sample_index", example.get("sample_index", "n/a"))
    pred = example.get("counterfactual_prediction", "n/a")
    target = example.get("target_class", "n/a")
    confidence = format_float(example.get("counterfactual_confidence"), 2)
    change = format_float(example.get("change"), 3)
    return (
        f"sample {sample} | target {target} | CF {pred} ({confidence})\n"
        f"valid {valid} | change {change}"
    )


def draw_missing_cell(ax: plt.Axes, message: str) -> None:
    ax.set_facecolor("#f3f3f3")
    ax.text(
        0.5,
        0.5,
        message,
        ha="center",
        va="center",
        fontsize=9,
        color="#555555",
        wrap=True,
        transform=ax.transAxes,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#cccccc")


def create_dataset_figure(
    dataset: str,
    examples: list[dict[str, Any]],
    output_dir: Path,
    dpi: int,
) -> dict[str, Any]:
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    warnings: list[str] = []

    for example in examples:
        method = str(example.get("method", "Unknown method"))
        category = str(example.get("category", ""))
        grouped[method][category] = example

    methods = sorted(grouped.keys(), key=method_sort_key)
    if not methods:
        warnings.append(f"No methods found for dataset {dataset}.")
        return {"dataset": dataset, "figure_path": None, "methods": [], "warnings": warnings}

    n_rows = len(CATEGORY_ORDER)
    n_cols = len(methods)
    fig_width = max(7.0, 4.8 * n_cols)
    fig_height = max(7.0, 3.0 * n_rows)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(fig_width, fig_height),
        squeeze=False,
        constrained_layout=True,
    )
    fig.suptitle(
        f"Qualitative counterfactual comparison on {dataset}",
        fontsize=16,
        fontweight="bold",
    )

    for col, method in enumerate(methods):
        axes[0, col].set_title(textwrap.fill(method, width=28), fontsize=11, pad=12)

    for row, (category, label) in enumerate(CATEGORY_ORDER):
        for col, method in enumerate(methods):
            ax = axes[row, col]
            ax.set_xticks([])
            ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(
                    textwrap.fill(label, width=22),
                    fontsize=11,
                    fontweight="bold",
                    rotation=0,
                    labelpad=78,
                    va="center",
                )

            example = grouped[method].get(category)
            if example is None:
                message = f"No selected example\nfor {label}"
                draw_missing_cell(ax, message)
                warnings.append(
                    f"{dataset} / {method} / {label}: no selected example available."
                )
                continue

            image_path, _ = choose_example_path(example)
            if image_path is None:
                message = "Image missing"
                draw_missing_cell(ax, message)
                warnings.append(
                    f"{dataset} / {method} / {label}: missing image path. "
                    f"Tried copied_plot_path={example.get('copied_plot_path')} and "
                    f"image_path={example.get('image_path')}."
                )
                continue

            try:
                image = normalize_image_for_display(mpimg.imread(image_path))
            except Exception as exc:  # pragma: no cover - defensive for broken image files.
                draw_missing_cell(ax, "Could not read image")
                warnings.append(f"{image_path}: could not be read ({exc}).")
                continue

            ax.imshow(image)
            ax.text(
                0.5,
                -0.08,
                format_cell_note(example),
                ha="center",
                va="top",
                fontsize=8,
                transform=ax.transAxes,
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    figure_path = output_dir / f"qualitative_comparison_{dataset.lower()}.png"
    fig.savefig(figure_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return {
        "type": "dataset_overview",
        "dataset": dataset,
        "figure_path": str(figure_path),
        "methods": methods,
        "categories": [label for _, label in CATEGORY_ORDER],
        "warnings": warnings,
    }


def create_method_figure(
    dataset: str,
    method: str,
    examples_by_category: dict[str, dict[str, Any]],
    output_dir: Path,
    dpi: int,
) -> dict[str, Any]:
    warnings: list[str] = []
    n_rows = len(CATEGORY_ORDER)
    fig, axes = plt.subplots(
        n_rows,
        1,
        figsize=(12.5, 3.25 * n_rows),
        squeeze=False,
        constrained_layout=True,
    )
    fig.suptitle(
        f"{method}\n{dataset}",
        fontsize=15,
        fontweight="bold",
    )

    for row, (category, label) in enumerate(CATEGORY_ORDER):
        ax = axes[row, 0]
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylabel(
            textwrap.fill(label, width=22),
            fontsize=11,
            fontweight="bold",
            rotation=0,
            labelpad=82,
            va="center",
        )

        example = examples_by_category.get(category)
        if example is None:
            draw_missing_cell(ax, f"No selected example\nfor {label}")
            warnings.append(f"{dataset} / {method} / {label}: no selected example available.")
            continue

        image_path, _ = choose_example_path(example)
        if image_path is None:
            draw_missing_cell(ax, "Image missing")
            warnings.append(
                f"{dataset} / {method} / {label}: missing image path. "
                f"Tried copied_plot_path={example.get('copied_plot_path')} and "
                f"image_path={example.get('image_path')}."
            )
            continue

        try:
            image = normalize_image_for_display(mpimg.imread(image_path))
        except Exception as exc:  # pragma: no cover - defensive for broken image files.
            draw_missing_cell(ax, "Could not read image")
            warnings.append(f"{image_path}: could not be read ({exc}).")
            continue

        ax.imshow(image)
        ax.text(
            0.5,
            -0.08,
            format_cell_note(example),
            ha="center",
            va="top",
            fontsize=9,
            transform=ax.transAxes,
        )

    method_output_dir = output_dir / "per_method"
    method_output_dir.mkdir(parents=True, exist_ok=True)
    figure_path = method_output_dir / f"{slugify(dataset)}__{slugify(method)}.png"
    fig.savefig(figure_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return {
        "type": "per_method",
        "dataset": dataset,
        "method": method,
        "figure_path": str(figure_path),
        "categories": [label for _, label in CATEGORY_ORDER],
        "warnings": warnings,
    }


def create_method_figures(
    by_dataset: dict[str, list[dict[str, Any]]],
    output_dir: Path,
    dpi: int,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for dataset in sorted(by_dataset):
        grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
        for example in by_dataset[dataset]:
            method = str(example.get("method", "Unknown method"))
            category = str(example.get("category", ""))
            grouped[method][category] = example

        for method in sorted(grouped, key=method_sort_key):
            results.append(
                create_method_figure(
                    dataset=dataset,
                    method=method,
                    examples_by_category=grouped[method],
                    output_dir=output_dir,
                    dpi=dpi,
                )
            )
    return results


def write_readme(
    output_dir: Path,
    selected_examples_path: Path,
    method_results: list[dict[str, Any]],
    overview_results: list[dict[str, Any]],
) -> None:
    all_results = method_results + overview_results
    warning_count = sum(len(result.get("warnings", [])) for result in all_results)
    lines = [
        "# Qualitative Comparison Figures",
        "",
        "This folder contains paper-friendly qualitative comparison figures for "
        "the counterfactual methods evaluated in this project.",
        "",
        "The main figures are stored in `per_method/`. Each figure contains one "
        "method on one dataset. Rows correspond to qualitative case types:",
        "",
    ]
    for _, label in CATEGORY_ORDER:
        lines.append(f"- {label}")
    lines.extend(
        [
            "",
            "The figures are composed from the existing per-example visualizations "
            "referenced in:",
            "",
            f"- `{selected_examples_path}`",
            "",
            "The comparison script does not recompute, stretch, or per-image normalize "
            "the embedded difference maps. Source plots are displayed as saved, and "
            "image data is only converted to the standard display range `[0, 1]`.",
            "",
            "Generated per-method figures:",
            "",
        ]
    )
    for result in method_results:
        figure_path = result.get("figure_path")
        if figure_path:
            lines.append(f"- `{figure_path}`")
    if overview_results:
        lines.extend(["", "Optional dataset overview figures:", ""])
        for result in overview_results:
            figure_path = result.get("figure_path")
            if figure_path:
                lines.append(f"- `{figure_path}`")
    lines.extend(
        [
            "",
            f"Warnings emitted during generation: {warning_count}",
            "",
        ]
    )
    if warning_count:
        lines.append("## Warnings")
        lines.append("")
        for result in all_results:
            for warning in result.get("warnings", []):
                lines.append(f"- {warning}")
        lines.append("")

    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    examples = load_selected_examples(args.selected_examples)
    requested_datasets = set(args.datasets) if args.datasets else None

    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for example in examples:
        dataset = str(example.get("dataset", "Unknown dataset"))
        if requested_datasets and dataset not in requested_datasets:
            continue
        by_dataset[dataset].append(example)

    if not by_dataset:
        raise ValueError("No selected examples matched the requested datasets.")

    method_results = create_method_figures(by_dataset, args.output_dir, args.dpi)
    for result in method_results:
        for warning in result.get("warnings", []):
            print(f"WARNING: {warning}")
        if result.get("figure_path"):
            print(f"Wrote {result['figure_path']}")

    overview_results = []
    if args.include_dataset_overview:
        for dataset in sorted(by_dataset):
            result = create_dataset_figure(dataset, by_dataset[dataset], args.output_dir, args.dpi)
            overview_results.append(result)
            for warning in result.get("warnings", []):
                print(f"WARNING: {warning}")
            if result.get("figure_path"):
                print(f"Wrote {result['figure_path']}")

    manifest_path = args.output_dir / "qualitative_figure_manifest.json"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "selected_examples_path": str(args.selected_examples),
                "dpi": args.dpi,
                "per_method_figures": method_results,
                "dataset_overview_figures": overview_results,
            },
            f,
            indent=2,
        )
    write_readme(args.output_dir, args.selected_examples, method_results, overview_results)
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
