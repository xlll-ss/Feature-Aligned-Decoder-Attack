#!/usr/bin/env bash
set -euo pipefail

# Override these variables only when the server layout differs.
CODE_DIR="${CODE_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"
PYTHON="${PYTHON:-python}"
BIDO_ROOT="${BIDO_ROOT:-$HOME/pytorch_xllll/paper/6-Unknown-Aware Bilateral Dependency Optimization/BiDO_plus-main/low_resolution}"
CELEBA_IMAGES="${CELEBA_IMAGES:-$BIDO_ROOT/attack_datasets/CelebA/Img}"
UTILITY_ROOT="${UTILITY_ROOT:-$CODE_DIR/results_utility_matched/celeba_vgg}"
TARGET="${TARGET:-$UTILITY_ROOT/targets/bido/best_validation.pth}"
MANIFEST="${MANIFEST:-$CODE_DIR/experiment_manifests/celeba_bido_clean.json}"
OUT="${OUT:-$CODE_DIR/results_decoder_core_ablation/celeba_vgg_bido}"
ARCFACE_CTX_ID="${ARCFACE_CTX_ID:--1}"

EPOCHS="${EPOCHS:-120}"
BATCH_SIZE="${BATCH_SIZE:-64}"
MAX_SAMPLES="${MAX_SAMPLES:-30000}"
TEST_SAMPLES="${TEST_SAMPLES:-500}"
TRAIN_WORKERS="${TRAIN_WORKERS:-8}"
EVAL_WORKERS="${EVAL_WORKERS:-4}"
BOOTSTRAP_SAMPLES="${BOOTSTRAP_SAMPLES:-10000}"
SEEDS_TEXT="${SEEDS:-2027 2028 2029}"
VARIANTS_TEXT="${VARIANTS:-full l1_only mse_only pixel_feature no_feature_norm compact residual}"
SMOKE="${SMOKE:-0}"
RESTART="${RESTART:-0}"

if [[ "$SMOKE" == "1" ]]; then
  EPOCHS=1
  BATCH_SIZE=16
  MAX_SAMPLES=128
  TEST_SAMPLES=8
  TRAIN_WORKERS=2
  EVAL_WORKERS=0
  BOOTSTRAP_SAMPLES=200
  OUT="${OUT}_smoke"
  echo "[SMOKE] This run checks the pipeline only; its ablation values are not for the paper."
fi

read -r -a SEEDS_ARRAY <<< "$SEEDS_TEXT"
read -r -a VARIANTS_ARRAY <<< "$VARIANTS_TEXT"

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "Missing required file: $1" >&2
    exit 1
  fi
}

require_file "$TARGET"
require_file "$MANIFEST"
require_file "$CODE_DIR/train_decoder_core_ablation.py"
require_file "$CODE_DIR/evaluate_decoder_core_ablation.py"
require_file "$CODE_DIR/evaluate_fair_baseline_perceptual.py"
require_file "$CODE_DIR/summarize_decoder_core_ablation.py"

mkdir -p "$OUT/logs"
cd "$CODE_DIR"

restart_args=()
if [[ "$RESTART" == "1" ]]; then
  restart_args+=(--restart)
fi

for variant in "${VARIANTS_ARRAY[@]}"; do
  case "$variant" in
    full|l1_only|mse_only|pixel_feature|no_feature_norm|compact|residual) ;;
    *)
      echo "Unknown decoder variant: $variant" >&2
      exit 1
      ;;
  esac

  for seed in "${SEEDS_ARRAY[@]}"; do
    if [[ "$variant" == "full" ]]; then
      decoder="$UTILITY_ROOT/decoder_output/models/decoder_utility_matched_bido_seed${seed}.pth"
      require_file "$decoder"
    else
      "$PYTHON" train_decoder_core_ablation.py \
        --variant "$variant" \
        --data_root "$CELEBA_IMAGES" \
        --target_weight "$TARGET" \
        --manifest_path "$MANIFEST" \
        --output_root "$OUT" \
        --epochs "$EPOCHS" \
        --batch_size "$BATCH_SIZE" \
        --lr 2e-4 \
        --max_samples "$MAX_SAMPLES" \
        --num_workers "$TRAIN_WORKERS" \
        --save_every 10 \
        --seed "$seed" \
        "${restart_args[@]}" \
        2>&1 | tee "$OUT/logs/train_${variant}_seed${seed}.log"
      decoder="$OUT/models/decoder_${variant}_seed${seed}.pth"
      require_file "$decoder"
    fi

    run_dir="$OUT/evaluation/$variant/seed$seed"
    mkdir -p "$run_dir"
    if [[ "$RESTART" != "1" && -f "$run_dir/summary.json" ]]; then
      echo "[skip] reconstruction evaluation already complete: $run_dir/summary.json"
    else
      "$PYTHON" evaluate_decoder_core_ablation.py \
        --variant "$variant" \
        --data_root "$CELEBA_IMAGES" \
        --target_weight "$TARGET" \
        --decoder_path "$decoder" \
        --manifest_path "$MANIFEST" \
        --output_dir "$run_dir" \
        --test_samples "$TEST_SAMPLES" \
        --num_workers "$EVAL_WORKERS" \
        --warmup 10 \
        --seed "$seed" \
        2>&1 | tee "$OUT/logs/evaluate_${variant}_seed${seed}.log"
    fi

    if [[ "$RESTART" != "1" && -f "$run_dir/lpips.json" && -f "$run_dir/arcface.json" ]]; then
      echo "[skip] perceptual evaluation already complete: $run_dir"
    else
      "$PYTHON" evaluate_fair_baseline_perceptual.py \
        --run_dir "$run_dir" \
        --lpips_net alex \
        --arcface_model buffalo_l \
        --arcface_ctx_id "$ARCFACE_CTX_ID" \
        --embedding_size 112 \
        --threshold 0.30 \
        --seed "$seed" \
        2>&1 | tee "$OUT/logs/perceptual_${variant}_seed${seed}.log"
    fi
  done
done

"$PYTHON" summarize_decoder_core_ablation.py \
  --root "$OUT" \
  --variants "${VARIANTS_ARRAY[@]}" \
  --seeds "${SEEDS_ARRAY[@]}" \
  --reference full \
  --bootstrap_samples "$BOOTSTRAP_SAMPLES" \
  --bootstrap_seed 2027 \
  --output_json "$OUT/decoder_core_ablation_summary.json" \
  --output_csv "$OUT/decoder_core_ablation_summary.csv" \
  2>&1 | tee "$OUT/logs/summary.log"

echo "Completed: $OUT/decoder_core_ablation_summary.json"
