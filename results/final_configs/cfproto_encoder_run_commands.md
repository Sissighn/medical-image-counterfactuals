# CFProto-Encoder-KNN Final Fixed-Manifest Run Commands

These are the final commands used for the fixed-manifest CFProto-nearer
prototype-guided evaluation.

## BUSI

```bash
PYTHONPATH=. python scripts/run_cfproto_pytorch.py \
  --model_path models/busi_resnet18_pretrained.pth \
  --dataset_path data/processed/BUSI \
  --output_dir results/final/cfproto_encoder_knn/busi \
  --manifest_path results/evaluation_manifests/busi_balanced_5_per_class_second_best.json \
  --steps 300 \
  --learning_rate 0.01 \
  --attack_loss cw_hinge \
  --attack_const 1.0 \
  --c_init 1.0 \
  --c_steps 3 \
  --c_search_mode adaptive_binary \
  --kappa 0.0 \
  --lambda_l1 0.01 \
  --lambda_l2 5.0 \
  --lambda_tv 0.2 \
  --lambda_proto 0.05 \
  --autoencoder_path models/autoencoder_busi.pth \
  --gamma 0.0 \
  --prototype_space encoder \
  --prototype_mode knn_mean \
  --prototype_k 3 \
  --selection_metric elastic_net \
  --beta 0.1 \
  --lr_schedule polynomial \
  --max_delta 0.12 \
  --perturbation_resolution 28 \
  --batch_size 16
```

## Pneumonia

```bash
PYTHONPATH=. python scripts/run_cfproto_pytorch.py \
  --model_path models/pneumonia_resnet18_pretrained.pth \
  --dataset_path data/processed/Pneumonia \
  --output_dir results/final/cfproto_encoder_knn/pneumonia \
  --manifest_path results/evaluation_manifests/pneumonia_balanced_10_per_class_second_best.json \
  --steps 300 \
  --learning_rate 0.01 \
  --attack_loss cw_hinge \
  --attack_const 1.0 \
  --c_init 1.0 \
  --c_steps 3 \
  --c_search_mode adaptive_binary \
  --kappa 0.0 \
  --lambda_l1 0.01 \
  --lambda_l2 5.0 \
  --lambda_tv 0.2 \
  --lambda_proto 0.05 \
  --autoencoder_path models/autoencoder_pneumonia.pth \
  --gamma 0.0 \
  --prototype_space encoder \
  --prototype_mode knn_mean \
  --prototype_k 3 \
  --selection_metric elastic_net \
  --beta 0.1 \
  --lr_schedule polynomial \
  --max_delta 0.12 \
  --perturbation_resolution 28 \
  --batch_size 16
```
