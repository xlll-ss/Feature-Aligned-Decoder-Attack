#!/usr/bin/env bash
set -euo pipefail

# Override these paths only when the server layout differs.
CODE_DIR="${CODE_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"
PYTHON="${PYTHON:-python}"
BIDO_ROOT="${BIDO_ROOT:-$HOME/pytorch_xllll/paper/6-Unknown-Aware Bilateral Dependency Optimization/BiDO_plus-main/low_resolution}"
CELEBA_IMAGES="${CELEBA_IMAGES:-$BIDO_ROOT/attack_datasets/CelebA/Img}"
PRETRAINED_VGG="${PRETRAINED_VGG:-$BIDO_ROOT/results/pretrained/vgg16_bn-6c64b313.pth}"
BASE_MANIFEST="${BASE_MANIFEST:-$CODE_DIR/experiment_manifests/celeba_bido_clean.json}"
UTILITY_ROOT="${UTILITY_ROOT:-$CODE_DIR/results_utility_matched/celeba_vgg}"
PROTOCOL_MANIFEST="${PROTOCOL_MANIFEST:-$UTILITY_ROOT/protocol_manifest.json}"
SOURCE_TARGET_STRENGTH1="${SOURCE_TARGET_STRENGTH1:-$UTILITY_ROOT/targets/bido/best_validation.pth}"
MULTI_TARGET_ROOT="${MULTI_TARGET_ROOT:-$CODE_DIR/results_target_model_seeds/celeba_vgg_bido}"
SOURCE_DECODER_STRENGTH1="${SOURCE_DECODER_STRENGTH1:-$MULTI_TARGET_ROOT/target_seed2027/decoder/models/decoder_full_seed2027.pth}"
OUT="${OUT:-$CODE_DIR/results_defense_tradeoff/celeba_vgg_seed2027}"
ARCFACE_CTX_ID="${ARCFACE_CTX_ID:--1}"

TARGET_SEED="${TARGET_SEED:-2027}"
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

STRENGTHS=(0 0.25 0.5 1 2)
TAGS=(0 0p25 0p5 1 2)
ALPHAS=(0 0.00025 0.0005 0.001 0.002)
BETAS=(0 0.00125 0.0025 0.005 0.01)

if [[ "$SMOKE" == "1" ]]; then
  STRENGTHS=(0 1)
  TAGS=(0 1)
  ALPHAS=(0 0.001)
  BETAS=(0 0.005)
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
  echo "[SMOKE] This run checks the pipeline only; its curve is not for the paper."
fi

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
require_file "$CODE_DIR/summarize_defense_tradeoff.py"
require_file "$CODE_DIR/plot_defense_tradeoff.py"
if [[ "$SMOKE" != "1" ]]; then
  require_file "$SOURCE_TARGET_STRENGTH1"
  require_file "$SOURCE_DECODER_STRENGTH1"
fi
"$PYTHON" -c "import matplotlib, numpy" >/dev/null

mkdir -p "$OUT/logs"
cd "$CODE_DIR"

restart_args=()
if [[ "$RESTART" == "1" ]]; then
  restart_args+=(--restart)
fi

for index in "${!STRENGTHS[@]}"; do
  strength="${STRENGTHS[$index]}"
  tag="${TAGS[$index]}"
  alpha="${ALPHAS[$index]}"
  beta="${BETAS[$index]}"
  point_root="$OUT/strength_${tag}"
  mkdir -p "$point_root"

  echo "======================================================================"
  echo "strength=$strength alpha=$alpha beta=$beta"
  echo "======================================================================"

  if [[ "$RESTART" != "1" && -f "$point_root/target_summary.json" ]]; then
    echo "[skip] target evaluation already complete: $point_root/target_summary.json"
  elif [[ "$SMOKE" != "1" && "$strength" == "1" ]]; then
    "$PYTHON" train_celeba_vgg_target_seed.py \
      --stage evaluate \
      --protocol_manifest "$PROTOCOL_MANIFEST" \
      --data_root "$CELEBA_IMAGES" \
      --output_root "$point_root" \
      --target_checkpoint "$SOURCE_TARGET_STRENGTH1" \
      --target_seed "$TARGET_SEED" \
      --epochs "$TARGET_EPOCHS" \
      --batch_size "$TARGET_BATCH_SIZE" \
      --eval_batch_size 64 \
      --num_workers "$TARGET_WORKERS" \
      --max_train_samples "$TARGET_MAX_TRAIN" \
      --max_validation_samples "$TARGET_MAX_VALIDATION" \
      --max_test_samples "$TARGET_MAX_TEST" \
      2>&1 | tee "$OUT/logs/target_strength_${tag}.log"
  else
    "$PYTHON" train_celeba_vgg_target_seed.py \
      --stage all \
      --protocol_manifest "$PROTOCOL_MANIFEST" \
      --data_root "$CELEBA_IMAGES" \
      --pretrained_vgg "$PRETRAINED_VGG" \
      --output_root "$point_root" \
      --target_seed "$TARGET_SEED" \
      --epochs "$TARGET_EPOCHS" \
      --batch_size "$TARGET_BATCH_SIZE" \
      --eval_batch_size 64 \
      --num_workers "$TARGET_WORKERS" \
      --lr 5e-5 \
      --weight_decay 1e-4 \
      --lr_milestones 40 \
      --lr_gamma 0.2 \
      --alpha "$alpha" \
      --beta "$beta" \
      --max_train_samples "$TARGET_MAX_TRAIN" \
      --max_validation_samples "$TARGET_MAX_VALIDATION" \
      --max_test_samples "$TARGET_MAX_TEST" \
      "${restart_args[@]}" \
      2>&1 | tee "$OUT/logs/target_strength_${tag}.log"
  fi

  if [[ "$SMOKE" != "1" && "$strength" == "1" ]]; then
    target="$SOURCE_TARGET_STRENGTH1"
    decoder="$SOURCE_DECODER_STRENGTH1"
  else
    target="$point_root/target/best_validation.pth"
    decoder_root="$point_root/decoder"
    require_file "$target"
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
      2>&1 | tee "$OUT/logs/decoder_strength_${tag}.log"
    decoder="$decoder_root/models/decoder_full_seed${DECODER_SEED}.pth"
  fi
  require_file "$target"
  require_file "$decoder"

  attack_dir="$point_root/attack"
  mkdir -p "$attack_dir"
  if [[ "$RESTART" != "1" && -f "$attack_dir/summary.json" ]]; then
    echo "[skip] reconstruction evaluation already complete: $attack_dir/summary.json"
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
      --seed "$TARGET_SEED" \
      2>&1 | tee "$OUT/logs/attack_strength_${tag}.log"
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
      --seed "$TARGET_SEED" \
      2>&1 | tee "$OUT/logs/perceptual_strength_${tag}.log"
  fi
done

"$PYTHON" summarize_defense_tradeoff.py \
  --root "$OUT" \
  --strengths "${STRENGTHS[@]}" \
  --base_alpha 0.001 \
  --base_beta 0.005 \
  --target_seed "$TARGET_SEED" \
  --bootstrap_samples "$BOOTSTRAP_SAMPLES" \
  --bootstrap_seed 2027 \
  --output_json "$OUT/defense_tradeoff_summary.json" \
  --output_csv "$OUT/defense_tradeoff_summary.csv" \
  2>&1 | tee "$OUT/logs/summary.log"

"$PYTHON" plot_defense_tradeoff.py \
  --summary "$OUT/defense_tradeoff_summary.json" \
  --output_png "$OUT/defense_privacy_utility_curve.png" \
  --output_pdf "$OUT/defense_privacy_utility_curve.pdf" \
  2>&1 | tee "$OUT/logs/plot.log"

echo "Completed: $OUT/defense_tradeoff_summary.json"
echo "Figure:    $OUT/defense_privacy_utility_curve.png"
