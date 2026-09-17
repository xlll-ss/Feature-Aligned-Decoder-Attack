#!/usr/bin/env bash
set -euo pipefail

# Override these variables only when the server layout differs.
CODE_DIR="${CODE_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"
PYTHON="${PYTHON:-python}"
BIDO_ROOT="${BIDO_ROOT:-$HOME/pytorch_xllll/paper/6-Unknown-Aware Bilateral Dependency Optimization/BiDO_plus-main/low_resolution}"
CELEBA_IMAGES="${CELEBA_IMAGES:-$BIDO_ROOT/attack_datasets/CelebA/Img}"
PRETRAINED_VGG="${PRETRAINED_VGG:-$BIDO_ROOT/results/pretrained/vgg16_bn-6c64b313.pth}"
BASE_MANIFEST="${BASE_MANIFEST:-$CODE_DIR/experiment_manifests/celeba_bido_clean.json}"
UTILITY_ROOT="${UTILITY_ROOT:-$CODE_DIR/results_utility_matched/celeba_vgg}"
PROTOCOL_MANIFEST="${PROTOCOL_MANIFEST:-$UTILITY_ROOT/protocol_manifest.json}"
SOURCE_TARGET_2027="${SOURCE_TARGET_2027:-$UTILITY_ROOT/targets/bido/best_validation.pth}"
OUT="${OUT:-$CODE_DIR/results_target_model_seeds/celeba_vgg_bido}"
ARCFACE_CTX_ID="${ARCFACE_CTX_ID:--1}"

TARGET_SEEDS_TEXT="${TARGET_SEEDS:-2027 2028 2029}"
DECODER_SEED="${DECODER_SEED:-2027}"
TARGET_EPOCHS="${TARGET_EPOCHS:-20}"
DECODER_EPOCHS="${DECODER_EPOCHS:-120}"
TARGET_BATCH_SIZE="${TARGET_BATCH_SIZE:-8}"
DECODER_BATCH_SIZE="${DECODER_BATCH_SIZE:-64}"
TARGET_MAX_TRAIN="${TARGET_MAX_TRAIN:-0}"
TARGET_MAX_VALIDATION="${TARGET_MAX_VALIDATION:-0}"
TARGET_MAX_TEST="${TARGET_MAX_TEST:-0}"
DECODER_MAX_SAMPLES="${DECODER_MAX_SAMPLES:-30000}"
TEST_SAMPLES="${TEST_SAMPLES:-500}"
TARGET_WORKERS="${TARGET_WORKERS:-8}"
DECODER_WORKERS="${DECODER_WORKERS:-8}"
EVAL_WORKERS="${EVAL_WORKERS:-4}"
BOOTSTRAP_SAMPLES="${BOOTSTRAP_SAMPLES:-10000}"
SMOKE="${SMOKE:-0}"
RESTART="${RESTART:-0}"

if [[ "$SMOKE" == "1" ]]; then
  TARGET_SEEDS_TEXT="2027 2028"
  TARGET_EPOCHS=1
  DECODER_EPOCHS=1
  TARGET_MAX_TRAIN=128
  TARGET_MAX_VALIDATION=64
  TARGET_MAX_TEST=64
  DECODER_MAX_SAMPLES=128
  TEST_SAMPLES=8
  TARGET_WORKERS=2
  DECODER_WORKERS=2
  EVAL_WORKERS=0
  BOOTSTRAP_SAMPLES=200
  OUT="${OUT}_smoke"
  echo "[SMOKE] This run checks the pipeline only; its values are not for the paper."
fi

read -r -a TARGET_SEEDS_ARRAY <<< "$TARGET_SEEDS_TEXT"

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "Missing required file: $1" >&2
    exit 1
  fi
}

require_file "$PRETRAINED_VGG"
require_file "$BASE_MANIFEST"
require_file "$PROTOCOL_MANIFEST"
require_file "$CODE_DIR/train_celeba_vgg_target_seed.py"
require_file "$CODE_DIR/train_decoder_core_ablation.py"
require_file "$CODE_DIR/evaluate_decoder_core_ablation.py"
require_file "$CODE_DIR/evaluate_fair_baseline_perceptual.py"
require_file "$CODE_DIR/summarize_target_model_seeds.py"
if [[ "$SMOKE" != "1" ]]; then
  require_file "$SOURCE_TARGET_2027"
fi

mkdir -p "$OUT/logs"
cd "$CODE_DIR"

restart_args=()
if [[ "$RESTART" == "1" ]]; then
  restart_args+=(--restart)
fi

for target_seed in "${TARGET_SEEDS_ARRAY[@]}"; do
  seed_root="$OUT/target_seed${target_seed}"
  mkdir -p "$seed_root"

  if [[ "$RESTART" != "1" && -f "$seed_root/target_summary.json" ]]; then
    echo "[skip] target seed evaluation already complete: $seed_root/target_summary.json"
  elif [[ "$SMOKE" != "1" && "$target_seed" == "2027" ]]; then
    "$PYTHON" train_celeba_vgg_target_seed.py \
      --stage evaluate \
      --protocol_manifest "$PROTOCOL_MANIFEST" \
      --data_root "$CELEBA_IMAGES" \
      --output_root "$seed_root" \
      --target_checkpoint "$SOURCE_TARGET_2027" \
      --target_seed "$target_seed" \
      --epochs "$TARGET_EPOCHS" \
      --batch_size "$TARGET_BATCH_SIZE" \
      --eval_batch_size 64 \
      --num_workers "$TARGET_WORKERS" \
      --max_train_samples "$TARGET_MAX_TRAIN" \
      --max_validation_samples "$TARGET_MAX_VALIDATION" \
      --max_test_samples "$TARGET_MAX_TEST" \
      2>&1 | tee "$OUT/logs/target_seed${target_seed}.log"
  else
    "$PYTHON" train_celeba_vgg_target_seed.py \
      --stage all \
      --protocol_manifest "$PROTOCOL_MANIFEST" \
      --data_root "$CELEBA_IMAGES" \
      --pretrained_vgg "$PRETRAINED_VGG" \
      --output_root "$seed_root" \
      --target_seed "$target_seed" \
      --epochs "$TARGET_EPOCHS" \
      --batch_size "$TARGET_BATCH_SIZE" \
      --eval_batch_size 64 \
      --num_workers "$TARGET_WORKERS" \
      --lr 5e-5 \
      --weight_decay 1e-4 \
      --lr_milestones 40 \
      --lr_gamma 0.2 \
      --alpha 0.001 \
      --beta 0.005 \
      --max_train_samples "$TARGET_MAX_TRAIN" \
      --max_validation_samples "$TARGET_MAX_VALIDATION" \
      --max_test_samples "$TARGET_MAX_TEST" \
      "${restart_args[@]}" \
      2>&1 | tee "$OUT/logs/target_seed${target_seed}.log"
  fi

  if [[ "$SMOKE" != "1" && "$target_seed" == "2027" ]]; then
    target="$SOURCE_TARGET_2027"
  else
    target="$seed_root/target/best_validation.pth"
  fi
  require_file "$target"

  decoder_root="$seed_root/decoder"
  "$PYTHON" train_decoder_core_ablation.py \
    --variant full \
    --data_root "$CELEBA_IMAGES" \
    --target_weight "$target" \
    --manifest_path "$BASE_MANIFEST" \
    --output_root "$decoder_root" \
    --epochs "$DECODER_EPOCHS" \
    --batch_size "$DECODER_BATCH_SIZE" \
    --lr 2e-4 \
    --max_samples "$DECODER_MAX_SAMPLES" \
    --num_workers "$DECODER_WORKERS" \
    --save_every 10 \
    --seed "$DECODER_SEED" \
    "${restart_args[@]}" \
    2>&1 | tee "$OUT/logs/decoder_target_seed${target_seed}.log"

  decoder="$decoder_root/models/decoder_full_seed${DECODER_SEED}.pth"
  require_file "$decoder"
  attack_dir="$seed_root/attack"
  mkdir -p "$attack_dir"
  if [[ "$RESTART" != "1" && -f "$attack_dir/summary.json" ]]; then
    echo "[skip] attack already complete: $attack_dir/summary.json"
  else
    "$PYTHON" evaluate_decoder_core_ablation.py \
      --variant full \
      --data_root "$CELEBA_IMAGES" \
      --target_weight "$target" \
      --decoder_path "$decoder" \
      --manifest_path "$BASE_MANIFEST" \
      --output_dir "$attack_dir" \
      --test_samples "$TEST_SAMPLES" \
      --num_workers "$EVAL_WORKERS" \
      --warmup 10 \
      --seed "$target_seed" \
      2>&1 | tee "$OUT/logs/attack_target_seed${target_seed}.log"
  fi

  if [[ "$RESTART" != "1" && -f "$attack_dir/lpips.json" && -f "$attack_dir/arcface.json" ]]; then
    echo "[skip] perceptual metrics already complete: $attack_dir"
  else
    "$PYTHON" evaluate_fair_baseline_perceptual.py \
      --run_dir "$attack_dir" \
      --lpips_net alex \
      --arcface_model buffalo_l \
      --arcface_ctx_id "$ARCFACE_CTX_ID" \
      --embedding_size 112 \
      --threshold 0.30 \
      --seed "$target_seed" \
      2>&1 | tee "$OUT/logs/perceptual_target_seed${target_seed}.log"
  fi
done

"$PYTHON" summarize_target_model_seeds.py \
  --root "$OUT" \
  --target_seeds "${TARGET_SEEDS_ARRAY[@]}" \
  --bootstrap_samples "$BOOTSTRAP_SAMPLES" \
  --bootstrap_seed 2027 \
  --output_json "$OUT/target_model_seed_summary.json" \
  --output_csv "$OUT/target_model_seed_summary.csv" \
  2>&1 | tee "$OUT/logs/summary.log"

echo "Completed: $OUT/target_model_seed_summary.json"
