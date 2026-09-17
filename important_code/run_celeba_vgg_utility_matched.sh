#!/usr/bin/env bash
set -euo pipefail

# Adjust only these paths if the server layout differs.
CODE_DIR="${CODE_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"
BIDO_ROOT="${BIDO_ROOT:-$HOME/pytorch_xllll/paper/6-Unknown-Aware Bilateral Dependency Optimization/BiDO_plus-main/low_resolution}"
CELEBA_IMAGES="${CELEBA_IMAGES:-$BIDO_ROOT/attack_datasets/CelebA/Img}"
BASE_MANIFEST="${BASE_MANIFEST:-$CODE_DIR/experiment_manifests/celeba_bido_clean.json}"
PRETRAINED_VGG="${PRETRAINED_VGG:-$BIDO_ROOT/results/pretrained/vgg16_bn-6c64b313.pth}"
EXP_ROOT="${EXP_ROOT:-$CODE_DIR/results_utility_matched/celeba_vgg}"
ARCFACE_CTX_ID="${ARCFACE_CTX_ID:--1}"
BIDO_EPOCHS="${BIDO_EPOCHS:-20}"
CONTROL_EPOCHS="${CONTROL_EPOCHS:-30}"
DECODER_EPOCHS="${DECODER_EPOCHS:-120}"
MATCH_TOLERANCE="${MATCH_TOLERANCE:-0.003}"
PROTOCOL_MANIFEST="$EXP_ROOT/protocol_manifest.json"
SEEDS=(2027 2028 2029)

mkdir -p "$EXP_ROOT/logs" "$EXP_ROOT/attacks" "$EXP_ROOT/decoder_output"
cd "$CODE_DIR"

python celeba_vgg_utility_matched.py \
  --stage all \
  --base_manifest "$BASE_MANIFEST" \
  --protocol_manifest "$PROTOCOL_MANIFEST" \
  --data_root "$CELEBA_IMAGES" \
  --pretrained_vgg "$PRETRAINED_VGG" \
  --output_root "$EXP_ROOT" \
  --validation_fraction 0.10 \
  --split_seed 314159 \
  --seed 2027 \
  --bido_epochs "$BIDO_EPOCHS" \
  --control_epochs "$CONTROL_EPOCHS" \
  --batch_size 8 \
  --eval_batch_size 64 \
  --num_workers 8 \
  --lr 5e-5 \
  --weight_decay 1e-4 \
  --lr_milestones 40 \
  --lr_gamma 0.2 \
  --alpha 0.001 \
  --beta 0.005 \
  --match_tolerance "$MATCH_TOLERANCE" \
  2>&1 | tee "$EXP_ROOT/logs/target_pair.log"

BIDO_TARGET="$EXP_ROOT/targets/bido/best_validation.pth"
CONTROL_TARGET="$EXP_ROOT/targets/control/matched_validation.pth"

for variant in bido control; do
  if [[ "$variant" == "bido" ]]; then
    target="$BIDO_TARGET"
  else
    target="$CONTROL_TARGET"
  fi
  for seed in "${SEEDS[@]}"; do
    run_name="utility_matched_${variant}_seed${seed}"
    decoder="$EXP_ROOT/decoder_output/models/decoder_${run_name}.pth"
    attack_dir="$EXP_ROOT/attacks/${variant}_seed${seed}"

    python train_decoder_multidataset.py \
      --dataset celeba \
      --data_root "$CELEBA_IMAGES" \
      --target_weight "$target" \
      --output_root "$EXP_ROOT/decoder_output" \
      --run_name "$run_name" \
      --arch vgg \
      --img_size 64 \
      --num_classes 0 \
      --feature_dim 2048 \
      --epochs "$DECODER_EPOCHS" \
      --batch_size 64 \
      --lr 2e-4 \
      --max_samples 30000 \
      --l1_weight 10 \
      --mse_weight 1 \
      --manifest_path "$BASE_MANIFEST" \
      --manifest_partition auxiliary \
      --seed "$seed" \
      --num_workers 8 \
      2>&1 | tee "$EXP_ROOT/logs/decoder_${variant}_seed${seed}.log"

    python attack_decoder_multidataset.py \
      --dataset celeba \
      --data_root "$CELEBA_IMAGES" \
      --target_weight "$target" \
      --decoder_path "$decoder" \
      --output_dir "$attack_dir" \
      --arch vgg \
      --img_size 64 \
      --num_classes 0 \
      --feature_dim 2048 \
      --test_samples 500 \
      --max_eval_scan 500 \
      --label_source dataset \
      --manifest_path "$BASE_MANIFEST" \
      --manifest_partition test \
      --save_pairs 40 \
      --save_raw_pairs \
      --seed "$seed" \
      --num_workers 4 \
      2>&1 | tee "$EXP_ROOT/logs/attack_${variant}_seed${seed}.log"

    python evaluate_lpips.py \
      --pairs_dir "$attack_dir/raw_pairs" \
      --output "$attack_dir/lpips.json" \
      --net alex \
      --max_samples 500 \
      --seed "$seed" \
      2>&1 | tee "$EXP_ROOT/logs/lpips_${variant}_seed${seed}.log"

    python evaluate_arcface.py \
      --pairs_dir "$attack_dir/raw_pairs" \
      --output "$attack_dir/arcface.json" \
      --model buffalo_l \
      --ctx_id "$ARCFACE_CTX_ID" \
      --embedding_size 112 \
      --threshold 0.30 \
      --max_samples 500 \
      --seed "$seed" \
      2>&1 | tee "$EXP_ROOT/logs/arcface_${variant}_seed${seed}.log"
  done
done

for seed in "${SEEDS[@]}"; do
  python compare_defense_results.py \
    --bido_summary "$EXP_ROOT/attacks/bido_seed${seed}/summary.json" \
    --undefended_summary "$EXP_ROOT/attacks/control_seed${seed}/summary.json" \
    --bido_lpips "$EXP_ROOT/attacks/bido_seed${seed}/lpips.json" \
    --undefended_lpips "$EXP_ROOT/attacks/control_seed${seed}/lpips.json" \
    --bido_arcface "$EXP_ROOT/attacks/bido_seed${seed}/arcface.json" \
    --undefended_arcface "$EXP_ROOT/attacks/control_seed${seed}/arcface.json" \
    --output "$EXP_ROOT/paired_seed${seed}.json" \
    --bootstrap_samples 10000 \
    --seed "$seed" \
    2>&1 | tee "$EXP_ROOT/logs/paired_seed${seed}.log"
done

python summarize_utility_matched.py \
  --experiment_root "$EXP_ROOT" \
  --target_summary "$EXP_ROOT/target_pair_summary.json" \
  --seeds "${SEEDS[@]}" \
  --bootstrap_samples 10000 \
  --bootstrap_seed 2027 \
  --output_json "$EXP_ROOT/utility_matched_summary.json" \
  --output_csv "$EXP_ROOT/utility_matched_summary.csv" \
  2>&1 | tee "$EXP_ROOT/logs/summary.log"

echo "Completed. Results: $EXP_ROOT/utility_matched_summary.json"
