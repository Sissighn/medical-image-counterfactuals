# DVCE — Anleitung für dich (Fine-Tuned-Checkpoint-Runs)

Du übernimmst 4 DVCE-Runs mit den fine-getunten Diffusion-Checkpoints
(2x No-Cone, 2x Cone Projection). Die OpenAI-Checkpoint-Runs mache ich
selbst. Bitte orientiere dich inhaltlich an meiner E-Mail an Paul von heute morgen
5 Uhr (!!) — der Hintergrund/die Originalitäts-Anpassungen sind dort erklärt,
hier bekommst du nur die für dich relevanten Schritte und Befehle.

## 1. Was du woher bekommst

**Von Paul (aus seinen alten E-Mails):**

```text
external/DVCEs/checkpoints/medical_diffusion_pneumonia_ema.pt
external/DVCEs/checkpoints/medical_diffusion_busi_ema.pt
```

**Von mir (Google Drive + mein Repo):**

```text
1. GitHub-Repo (Code): https://github.com/Sissighn/medical-image-counterfactuals
   Branch main, Commit cb82d38

2. external/DVCEs Code-Zip (ohne Checkpoints, ohne .git) — liegt bei mir im
   Google Drive Ordner, ca. 5 MB.
3. data/processed/Pneumonia/ (Google Drive, ~1,2 GB)
4. data/processed/BUSI/ (Google Drive, ~255 MB)
5. models/pneumonia_resnet18_pretrained.pth (Google Drive, 43 MB)
6. models/busi_resnet18_pretrained.pth (Google Drive, 43 MB)
7. models/pneumonia_resnet18_robust_pgd.pth (Google Drive) — PGD-robuster
   Klassifikator, nur für die Cone-Projection-Runs nötig
8. models/busi_resnet18_robust_pgd.pth (Google Drive) — PGD-robuster

requirements.txt und requirements-dvce.txt sind schon im Repo.
```

## 2. Setup

```bash
git clone https://github.com/Sissighn/medical-image-counterfactuals
cd medical-image-counterfactuals
git checkout main
# Commit-Hash von mir bestätigen lassen, dann ggf.:
# git checkout <hash>

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dvce.txt
```

Dann die Ordner an die richtigen Stellen legen:

```text
external/DVCEs/           <- Inhalt des Zips von mir entpacken
external/DVCEs/checkpoints/medical_diffusion_pneumonia_ema.pt   <- von Paul
external/DVCEs/checkpoints/medical_diffusion_busi_ema.pt        <- von Paul
data/processed/Pneumonia/
data/processed/BUSI/
models/pneumonia_resnet18_pretrained.pth
models/busi_resnet18_pretrained.pth
models/pneumonia_resnet18_robust_pgd.pth
models/busi_resnet18_robust_pgd.pth
```

Kurzer Check, dass alles am richtigen Platz liegt:

```bash
ls external/DVCEs/blended_diffusion
ls external/DVCEs/checkpoints
ls data/processed/Pneumonia data/processed/BUSI
ls models
```

## 3. Deine 4 Befehle

Reihenfolge: erst die beiden No-Cone-Runs (schneller, kein `aug_num`), dann
die beiden Cone-Runs. Für die Cone-Runs vorher unbedingt einmal mit
`--manifest_max_samples 1` die Laufzeit testen (Beispiel unten), bevor du das
volle Manifest laufen lässt — `aug_num 16` macht jeden Diffusionsschritt
deutlich teurer.

### 3.1 No-Cone, Pneumonia (fine-getunter Checkpoint)

```bash
PYTHONPATH=. .venv/bin/python scripts/run_dvce_medical_prototype.py \
  --model_path models/pneumonia_resnet18_pretrained.pth \
  --dataset_path data/processed/Pneumonia \
  --output_dir results/final/dvce_original_style/pneumonia_medical_checkpoint/pneumonia \
  --manifest_path results/evaluation_manifests/pneumonia_balanced_10_per_class_second_best.json \
  --run_generation --device auto \
  --timestep_respacing 200 --skip_timesteps 100 \
  --classifier_lambda 0.1 --lp_custom 1.0 --lp_custom_value 0.15 \
  --denoise_dist_input --no-clip_denoised \
  --diffusion_checkpoint_path external/DVCEs/checkpoints/medical_diffusion_pneumonia_ema.pt
```

### 3.2 No-Cone, BUSI (fine-getunter Checkpoint)

```bash
PYTHONPATH=. .venv/bin/python scripts/run_dvce_medical_prototype.py \
  --model_path models/busi_resnet18_pretrained.pth \
  --dataset_path data/processed/BUSI \
  --output_dir results/final/dvce_original_style/busi_medical_checkpoint/busi \
  --manifest_path results/evaluation_manifests/busi_balanced_5_per_class_second_best.json \
  --run_generation --device auto \
  --timestep_respacing 200 --skip_timesteps 100 \
  --classifier_lambda 0.1 --lp_custom 1.0 --lp_custom_value 0.15 \
  --denoise_dist_input --no-clip_denoised \
  --diffusion_checkpoint_path external/DVCEs/checkpoints/medical_diffusion_busi_ema.pt
```

### 3.3 Laufzeit-**Test** vor dem vollen Cone-Run (1 Sample, empfohlen)

```bash
PYTHONPATH=. .venv/bin/python scripts/run_dvce_medical_prototype.py \
  --model_path models/pneumonia_resnet18_pretrained.pth \
  --second_model_path models/pneumonia_resnet18_robust_pgd.pth \
  --dataset_path data/processed/Pneumonia \
  --output_dir results/debug/dvce_cone_smoke/pneumonia \
  --manifest_path results/evaluation_manifests/pneumonia_balanced_10_per_class_second_best.json \
  --manifest_max_samples 1 --run_generation --device auto \
  --timestep_respacing 200 --skip_timesteps 100 \
  --classifier_lambda 0.1 --lp_custom 1.0 --lp_custom_value 0.15 \
  --denoise_dist_input --no-clip_denoised --deg_cone_projection 30 --aug_num 16 \
  --diffusion_checkpoint_path external/DVCEs/checkpoints/medical_diffusion_pneumonia_ema.pt
```

Wenn die Laufzeit für 1 Sample realistisch aussieht (× 20 Pneumonia- bzw.
× 15 BUSI-Samples), weiter mit den vollen Runs. Falls nicht: kurz bei mir
melden, dann reduzieren wir `--aug_num` auf 1 und vermerken das als
Abweichung.

### 3.4 Cone Projection, Pneumonia (fine-getunter Checkpoint, volles Manifest)

```bash
PYTHONPATH=. .venv/bin/python scripts/run_dvce_medical_prototype.py \
  --model_path models/pneumonia_resnet18_pretrained.pth \
  --second_model_path models/pneumonia_resnet18_robust_pgd.pth \
  --dataset_path data/processed/Pneumonia \
  --output_dir results/final/dvce_original_style_cone/pneumonia_medical_checkpoint/pneumonia \
  --manifest_path results/evaluation_manifests/pneumonia_balanced_10_per_class_second_best.json \
  --run_generation --device auto \
  --timestep_respacing 200 --skip_timesteps 100 \
  --classifier_lambda 0.1 --lp_custom 1.0 --lp_custom_value 0.15 \
  --denoise_dist_input --no-clip_denoised --deg_cone_projection 30 --aug_num 16 \
  --diffusion_checkpoint_path external/DVCEs/checkpoints/medical_diffusion_pneumonia_ema.pt
```

### 3.5 Cone Projection, BUSI (fine-getunter Checkpoint, volles Manifest)

```bash
PYTHONPATH=. .venv/bin/python scripts/run_dvce_medical_prototype.py \
  --model_path models/busi_resnet18_pretrained.pth \
  --second_model_path models/busi_resnet18_robust_pgd.pth \
  --dataset_path data/processed/BUSI \
  --output_dir results/final/dvce_original_style_cone/busi_medical_checkpoint/busi \
  --manifest_path results/evaluation_manifests/busi_balanced_5_per_class_second_best.json \
  --run_generation --device auto \
  --timestep_respacing 200 --skip_timesteps 100 \
  --classifier_lambda 0.1 --lp_custom 1.0 --lp_custom_value 0.15 \
  --denoise_dist_input --no-clip_denoised --deg_cone_projection 30 --aug_num 16 \
  --diffusion_checkpoint_path external/DVCEs/checkpoints/medical_diffusion_busi_ema.pt
```

## 4. Wichtig

- Immer ein Kommando nach dem anderen laufen lassen, nicht parallel — sonst
  teilen sich zwei Runs Speicher/GPU und werden beide langsamer oder stürzen ab.
- `--device auto` erkennt automatisch MPS (Mac) oder CUDA. Falls du eine
  NVIDIA-GPU hast, kannst du zusätzlich `--diffusion_fp16` anhängen (das
  Original läuft immer in fp16); auf Mac/MPS weglassen.
- Jeder Lauf schreibt `metadata.json` in den jeweiligen `--output_dir`.
  Bitte am Ende `results/final/dvce_original_style/*` und
  `results/final/dvce_original_style_cone/*` (die kompletten Ordner inkl.
  PNGs) zurückschicken.
- Bei Fehlern: Traceback + welches der 4 Kommandos schicken, nicht selbst
  reparieren.
