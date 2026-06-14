# Medical Image Counterfactuals

This repository contains project seminar code for training medical image classifiers and generating prototype-guided counterfactual explanations for medical imaging datasets.

The current project focuses on two classification tasks:

- BUSI breast ultrasound classification
- Chest X-ray pneumonia classification

The main goal is to build reliable baseline image classifiers and then use them as the basis for counterfactual explanation methods. The first implemented counterfactual approach is a PyTorch prototype-guided method inspired by CFProto.

## Project Status

The current best baseline models are pretrained ResNet18 classifiers trained with data augmentation and class-weighted cross entropy.

| Dataset | Model | Accuracy | Weighted F1 |
| --- | --- | ---: | ---: |
| BUSI | ResNet18 pretrained | 0.8390 | 0.8365 |
| Pneumonia | ResNet18 pretrained | 0.8782 | 0.8732 |

The first prototype-guided counterfactual pipeline works for both datasets and produces valid model counterfactuals. The generated examples are an initial working version and still require further tuning for visual plausibility.

Detailed result files are stored under `results/`.

## Repository Structure

```text
.
|-- README.md
|-- requirements.txt
|-- src/
|   |-- data_utils.py
|   `-- train_model.py
|-- scripts/
|   |-- check_dataset.py
|   |-- evaluate_model.py
|   |-- prepare_busi.py
|   |-- prepare_pneumonia.py
|   |-- run_cfproto_pytorch.py
|   |-- test_data_utils.py
|   |-- test_dataloaders.py
|   `-- test_saved_model.py
`-- results/
    |-- baseline_comparison.md
    |-- pretrained_baseline_update.md
    |-- cfproto_start.md
    `-- *.json
```

The following folders are intentionally not tracked by Git:

```text
data/
models/
.venv/
```

This keeps the repository lightweight and avoids committing large datasets, model checkpoints, or environment files.

## Environment Setup

Create and activate a Python virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

The project has been developed locally with PyTorch, torchvision, scikit-learn, Pillow, and matplotlib.

## Data

The code expects processed datasets in ImageFolder format:

```text
data/processed/BUSI/
|-- train/
|-- val/
`-- test/

data/processed/Pneumonia/
|-- train/
|-- val/
`-- test/
```

Each split must contain one subfolder per class.

Expected BUSI classes:

```text
benign
malignant
normal
```

Expected Pneumonia classes:

```text
NORMAL
PNEUMONIA
```

The raw and processed data are not included in this repository.

## Training

The training script is:

```text
src/train_model.py
```

Example for BUSI:

```bash
PYTHONPATH=. python src/train_model.py \
  --dataset_name BUSI_pretrained \
  --dataset_path data/processed/BUSI \
  --output_model_path models/busi_resnet18_pretrained.pth \
  --epochs 15 \
  --batch_size 16 \
  --learning_rate 0.0001 \
  --pretrained
```

Example for Pneumonia:

```bash
PYTHONPATH=. python src/train_model.py \
  --dataset_name Pneumonia_pretrained \
  --dataset_path data/processed/Pneumonia \
  --output_model_path models/pneumonia_resnet18_pretrained.pth \
  --epochs 10 \
  --batch_size 16 \
  --learning_rate 0.0001 \
  --pretrained
```

The training pipeline supports:

- ResNet18 classification
- Optional ImageNet pretrained weights
- Training-time data augmentation
- Class-weighted cross entropy
- Automatic device selection for MPS, CUDA, or CPU
- Best-checkpoint saving based on validation F1 score

## Pretrained Weights

The pretrained ResNet18 weights may need to be available in the local Torch cache:

```text
~/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth
```

In this project, the automatic torchvision download created empty `.partial` files and hung. The weights were therefore downloaded manually once and then loaded from the local cache.

## Evaluation

Use the evaluation script to compute test metrics:

```bash
PYTHONPATH=. python scripts/evaluate_model.py \
  --model_path models/busi_resnet18_pretrained.pth \
  --dataset_path data/processed/BUSI \
  --output_path results/busi_pretrained_test_evaluation.json
```

The evaluation output includes:

- Accuracy
- Weighted F1 score
- Weighted precision
- Weighted recall
- Classification report
- Confusion matrix

The main baseline comparison is documented in:

```text
results/baseline_comparison.md
```

## Counterfactual Explanations

The first implemented counterfactual method is:

```text
scripts/run_cfproto_pytorch.py
```

This is a PyTorch implementation inspired by CFProto. It is designed to work directly with the trained PyTorch ResNet18 checkpoints.

The method works as follows:

1. Load a trained ResNet18 checkpoint.
2. Extract penultimate-layer feature vectors.
3. Compute one feature prototype per class from the training split.
4. Select correctly classified test samples.
5. Optimize the input image toward a target class.
6. Penalize large pixel changes and noisy changes.
7. Keep counterfactual images grayscale by default.
8. Save original image, counterfactual image, difference image, predictions, probabilities, runtime, and change metrics.

Example for BUSI:

```bash
PYTHONPATH=. python scripts/run_cfproto_pytorch.py \
  --model_path models/busi_resnet18_pretrained.pth \
  --dataset_path data/processed/BUSI \
  --output_dir results/cfproto/busi_first \
  --max_samples 1 \
  --steps 250
```

Example for Pneumonia:

```bash
PYTHONPATH=. python scripts/run_cfproto_pytorch.py \
  --model_path models/pneumonia_resnet18_pretrained.pth \
  --dataset_path data/processed/Pneumonia \
  --output_dir results/cfproto/pneumonia_first \
  --max_samples 1 \
  --steps 250
```

Each run writes:

```text
metadata.json
sample_XX.png
sample_XX.summary.png
```

The PNG files are ignored by Git because they are generated artifacts. The metadata JSON files are tracked because they are small and useful for documenting experiments.

## External Methods Considered

Two external repositories were considered for the first counterfactual method:

- `SeldonIO/alibi`
- `e-delaney/cfe_images_how_people_differ_from_machines`

Alibi contains a CFProto-style method, but its image examples and model integration are mainly oriented toward TensorFlow/Keras workflows. Since this project currently uses PyTorch ResNet18 checkpoints, a local PyTorch implementation was created first.

The Delaney repository is relevant as a reference for counterfactual image explanations, but it is not directly plug-and-play for the current BUSI and Pneumonia PyTorch pipeline.

## Current Documentation

Important project notes are stored in:

```text
results/baseline_comparison.md
results/pretrained_baseline_update.md
results/cfproto_start.md
```

These files document the baseline progression, the pretrained model update, and the first counterfactual experiments.

## Next Steps

Planned next steps:

1. Run the prototype-guided counterfactual method on more samples.
2. Tune regularization parameters to improve visual plausibility.
3. Compute aggregate counterfactual metrics:
   - Validity
   - Proximity
   - Sparsity
   - Runtime
4. Implement a second method such as SEDC-T.
5. Compare methods across BUSI and Pneumonia.
