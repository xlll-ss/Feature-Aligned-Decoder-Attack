#!/usr/bin/env bash
set -euo pipefail

CODE_DIR="${CODE_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"
UTILITY_ROOT="${UTILITY_ROOT:-$CODE_DIR/results_utility_matched/celeba_vgg}"
BIDO_ROOT="${BIDO_ROOT:-$HOME/pytorch_xllll/paper/6-Unknown-Aware Bilateral Dependency Optimization/BiDO_plus-main/low_resolution}"
CELEBA_IMAGES="${CELEBA_IMAGES:-$BIDO_ROOT/attack_datasets/CelebA/Img}"
MANIFEST="${MANIFEST:-$CODE_DIR/experiment_manifests/celeba_bido_clean.json}"
TARGET="${TARGET:-$UTILITY_ROOT/targets/bido/best_validation.pth}"
OUT="${OUT:-$CODE_DIR/results_interface_ablation/celeba_vgg_bido}"
CALIBRATION="$OUT/auxiliary_feature_calibration.pth"
ARCFACE_CTX_ID="${ARCFACE_CTX_ID:--1}"
SEEDS=(2027 2028 2029)
CONDITIONS=(
  known_wb_ce
  unknown_wb_ce
  unknown_wb_label_smoothing_01
  unknown_wb_temperature_2
  known_weight_only
  unknown_weight_only
  bias_only
)

mkdir -p "$OUT/logs"
cd "$CODE_DIR"

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "Missing required file: $1" >&2
    exit 1
  fi
}

require_file "$MANIFEST"
require_file "$TARGET"
require_file "$CODE_DIR/attack_interface_ablation.py"
require_file "$CODE_DIR/evaluate_interface_perceptual.py"
require_file "$CODE_DIR/summarize_interface_ablation.py"

for seed in "${SEEDS[@]}"; do
  decoder="$UTILITY_ROOT/decoder_output/models/decoder_utility_matched_bido_seed${seed}.pth"
  seed_dir="$OUT/seed${seed}"
  require_file "$decoder"
  python attack_interface_ablation.py \
    --data_root "$CELEBA_IMAGES" \
    --target_weight "$TARGET" \
    --decoder_path "$decoder" \
    --manifest_path "$MANIFEST" \
    --manifest_partition test \
    --output_dir "$seed_dir" \
    --calibration_path "$CALIBRATION" \
    --calibration_partition auxiliary \
    --calibration_samples 30000 \
    --calibration_batch_size 64 \
    --conditions "${CONDITIONS[@]}" \
    --test_samples 500 \
    --img_size 64 \
    --feature_dim 2048 \
    --num_classes 0 \
    --num_workers 4 \
    --epsilon 1e-12 \
    --seed "$seed" \
    --save_raw_pairs \
    2>&1 | tee "$OUT/logs/attack_seed${seed}.log"

  python evaluate_interface_perceptual.py \
    --seed_dir "$seed_dir" \
    --conditions "${CONDITIONS[@]}" \
    --lpips_net alex \
    --arcface_model buffalo_l \
    --arcface_ctx_id "$ARCFACE_CTX_ID" \
    --embedding_size 112 \
    --threshold 0.30 \
    --seed "$seed" \
    2>&1 | tee "$OUT/logs/perceptual_seed${seed}.log"
done

python summarize_interface_ablation.py \
  --root "$OUT" \
  --seeds "${SEEDS[@]}" \
  --reference known_wb_ce \
  --bootstrap_samples 10000 \
  --bootstrap_seed 2027 \
  --output_json "$OUT/interface_ablation_summary.json" \
  --output_csv "$OUT/interface_ablation_summary.csv" \
  2>&1 | tee "$OUT/logs/summary.log"

echo "Completed: $OUT/interface_ablation_summary.json"
