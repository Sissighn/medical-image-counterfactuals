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
| `create_evaluation_manifest.py` | Create fixed correctly classified evaluation samples for counterfactual comparison. |
| `run_cfproto_pytorch.py` | Run the prototype-guided optimization baseline. |
| `run_sedc_t_pytorch.py` | Run SEDC-T original-style or the faster project variant via `--search_mode`. |
| `run_dvce_medical_prototype.py` | Run the DVCE-style diffusion-guided counterfactual prototype. |
| `summarize_counterfactual_evaluation.py` | Generate compact summary tables from method metadata files. |
| `select_interpretable_examples.py` | Select good, difficult, and failure examples from existing metadata for qualitative inspection. |
| `create_qualitative_comparison_figures.py` | Compose selected per-example plots into dataset-level qualitative comparison figures. |

## Diffusion Fine-Tuning Utilities

| Script | Role |
| --- | --- |
| `prepare_diffusion_training_data.py` | Export processed medical images as flat 256x256 RGB folders for diffusion fine-tuning. |
| `check_diffusion_training_setup.py` | Check training data/checkpoints and generate a guided-diffusion training command. |

These scripts support the DVCE-style method but are not themselves a
counterfactual method. If rerunning `prepare_diffusion_training_data.py` with a
smaller subset, use a clean output folder to avoid leaving stale exported images.

## DVCE Smoke-Test Utilities

| Script | Role |
| --- | --- |
| `check_dvce_environment.py` | Check whether the DVCE repository and required imports are available. |
| `run_dvce_feasibility.py` | Run an early integration smoke test with the medical classifier and one sample. |

These scripts are useful for setup/debugging. The actual DVCE-style evaluation
uses `run_dvce_medical_prototype.py`.

## Debug And Legacy Checks

| Script | Role |
| --- | --- |
| `check_dataset.py` | Count processed images per split and class. |
| `test_data_utils.py` | Smoke-test the central `src.data_utils.create_dataloaders` function. |
| `test_dataloaders.py` | Older standalone ImageFolder/DataLoader check. Kept for reference, but not part of the main workflow. |
| `test_saved_model.py` | Quick one-batch checkpoint loading test. For full metrics use `evaluate_model.py`. |

These scripts are intentionally lightweight and may overlap. They should not be
used as final evaluation scripts for the seminar results.

## Output Behavior

Most scripts overwrite the file or folder passed via `--output_path` or
`--output_dir`. This is intentional for reproducible reruns, but important when
preserving old results. Use a new output folder name for parameter studies or
exploratory runs.
