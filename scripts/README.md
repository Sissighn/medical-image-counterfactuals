# Scripts Overview

This folder contains pipeline scripts, evaluation scripts, and a few older
debug/smoke-test utilities. The main project workflow should use the core
scripts below.

## Core Pipeline Scripts

| Script | Role |
| --- | --- |
| `prepare_busi.py` | Prepare the BUSI dataset in the processed `train/val/test` folder structure. |
| `prepare_pneumonia.py` | Prepare the Pneumonia dataset in the processed `train/val/test` folder structure. |
| `evaluate_model.py` | Evaluate a saved classifier on the full test split and write metrics to JSON. |
| `train_robust_resnet18_pgd.py` | Train a PGD-adversarially robust ResNet18 in the same checkpoint format as the normal classifiers, for DVCE Cone Projection. |
| `train_autoencoder.py` | Train an unsupervised ConvAutoencoder, including the compact bottleneck variant, on denormalized `[0, 1]` images for the CFProto (original-style) encoder-prototype method. |
| `check_autoencoder_plausibility.py` | Check whether a trained autoencoder assigns higher reconstruction loss to brightness/contrast, blur, patch, and noise perturbations than to original images. |
| `create_evaluation_manifest.py` | Create fixed correctly classified evaluation samples for counterfactual comparison. |
| `run_cfproto_pytorch.py` | Run the CFProto original-style prototype-guided counterfactual method (FISTA with shrinkage-thresholding, hinge attack loss, binary c-search, encoder-space class prototypes) following alibi's `CounterfactualProto`. |
| `run_goyal_cve_pytorch.py` | Run Goyal et al. 2019 counterfactual visual explanations (greedy exhaustive feature-cell swaps from a nearest-unlike-neighbor distractor). |
| `run_sedc_t_pytorch.py` | Run SEDC-T original-style best-first, plus the retained Pneumonia lung-field ROI ablation via `--roi_mode lung_fields`. |
| `run_dvce_pytorch.py` | Run the original-style DVCE diffusion-guided counterfactual generation with OpenAI or medical fine-tuned diffusion checkpoints. |
| `summarize_counterfactual_evaluation.py` | Generate compact summary tables from method metadata files. |
| `select_interpretable_examples.py` | Select good, difficult, and failure examples from existing metadata for qualitative inspection. |
| `create_qualitative_comparison_figures.py` | Compose selected per-example plots into dataset-level qualitative comparison figures. |

## Diffusion Fine-Tuning Utilities

| Script | Role |
| --- | --- |
| `prepare_diffusion_training_data.py` | Export processed medical images as flat 256x256 RGB folders for diffusion fine-tuning. |
| `check_diffusion_training_setup.py` | Check training data/checkpoints and generate a guided-diffusion training command. |

These scripts support the DVCE method but are not themselves a counterfactual
method. They document how the medical fine-tuned diffusion checkpoints were
prepared and are kept for reproducibility. If rerunning
`prepare_diffusion_training_data.py` with a smaller subset, use a clean output
folder to avoid leaving stale exported images.

## Lightweight Checks

| Script | Role |
| --- | --- |
| `check_dataset.py` | Count processed images per split and class. |
| `test_data_utils.py` | Smoke-test the central `src.data_utils.create_dataloaders` function. |

These scripts are intentionally lightweight. They should not be used as final
evaluation scripts for the seminar results.

## Output Behavior

Most scripts overwrite the file or folder passed via `--output_path` or
`--output_dir`. This is intentional for reproducible reruns, but important when
preserving old results. Use a new output folder name for parameter studies or
exploratory runs.
