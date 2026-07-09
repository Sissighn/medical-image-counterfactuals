# CFProto Original-Style Fixed-Manifest Run Commands

These are the commands for the fixed-manifest CFProto original-style
evaluation. The script follows alibi's `CounterfactualProto` (FISTA with
shrinkage-thresholding, untargeted hinge attack loss on the original class,
binary c-search with x10 escalation, encoder-space class prototypes from
classifier predictions, elastic-net best-counterfactual selection).

Structural defaults follow the original library and its MNIST image example
(`c_init=1`, `c_steps` from binary search, `max_iterations=1000`, `beta=0.1`,
`kappa=0`, `learning_rate_init=1e-2`, polynomial decay power 0.5).

`gamma` and `theta` are the only values re-calibrated for this project: all
loss terms are sums (as in the original), so the MNIST example values
(`gamma=100`, `theta=100`) do not transfer to 224x224 inputs and this encoder;
they make the prototype term larger than every other term by orders of
magnitude and the optimization diverges. Calibration rule: pick `gamma`/`theta`
so the weighted AE and prototype terms are comparable to the L2 sum
(inspect `loss_terms` in `metadata.json` or use `--verbose`).

`--prototype_k 3 --k_type mean` corresponds to `explain(k=3, k_type='mean')`
in alibi. Omitting `--prototype_k` uses the class-mean prototype
(alibi default `k=None`).

## BUSI

```bash
PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 PYTHONPATH=. python scripts/run_cfproto_pytorch.py \
  --model_path models/busi_resnet18_pretrained.pth \
  --dataset_path data/processed/BUSI \
  --output_dir results/final/cfproto_encoder_knn/busi \
  --manifest_path results/evaluation_manifests/busi_balanced_5_per_class_second_best.json \
  --max_iterations 1000 \
  --learning_rate_init 0.01 \
  --kappa 0.0 \
  --beta 0.1 \
  --gamma 1.0 \
  --theta 0.5 \
  --c_init 1.0 \
  --c_steps 5 \
  --autoencoder_path models/autoencoder_busi_bottleneck256.pth \
  --prototype_k 3 \
  --k_type mean \
  --batch_size 16
```

## Pneumonia

```bash
PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 PYTHONPATH=. python scripts/run_cfproto_pytorch.py \
  --model_path models/pneumonia_resnet18_pretrained.pth \
  --dataset_path data/processed/Pneumonia \
  --output_dir results/final/cfproto_encoder_knn/pneumonia \
  --manifest_path results/evaluation_manifests/pneumonia_balanced_10_per_class_second_best.json \
  --max_iterations 1000 \
  --learning_rate_init 0.01 \
  --kappa 0.0 \
  --beta 0.1 \
  --gamma 1.0 \
  --theta 0.05 \
  --c_init 1.0 \
  --c_steps 5 \
  --autoencoder_path models/autoencoder_pneumonia_bottleneck256.pth \
  --prototype_k 3 \
  --k_type mean \
  --batch_size 16
```

## Calibration of `theta`/`gamma`

Because all loss terms are sums (as in the original), the raw prototype
distance depends on the dataset and the trained autoencoder. With the
bottleneck-256 autoencoder the raw `||ENC(cf) - proto||^2` at the starting
point is ~32k on BUSI but ~1.1M on Pneumonia (~34x larger), so the same
`theta` weights the prototype term very differently:

- **BUSI**: `theta=0.5` → sparse, prototype-guided CFs (diff mean ~0.002,
  3/4 valid in a 4-sample `max_iterations=300` smoke test).
- **Pneumonia**: `theta=0.5` over-weights the prototype term (it explodes from
  0.55M to 38M during optimization and the image changes globally, diff mean
  ~0.31). `theta=0.05` keeps the prototype term stable (~2.7M plateau, vs L2
  ~31k) and yields sparse CFs: **4/4 valid, l1 mean 0.0006, 0.5% changed
  pixels**, all flipping to PNEUMONIA. This 10x reduction matches the ~34x
  larger raw distances (same order of magnitude).

For any other autoencoder (e.g. the feature-map `autoencoder_*.pth` without
bottleneck, whose encoding has ~50k dimensions) re-check `loss_terms` in
`metadata.json` or run with `--verbose` and pick `theta` so the weighted
prototype term stays comparable to / does not explode past the L2 sum.
