#!/usr/bin/env bash
set -euo pipefail

CODE_DIR="${CODE_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"
UTILITY_ROOT="${UTILITY_ROOT:-$CODE_DIR/results_utility_matched/celeba_vgg}"
BIDO_ROOT="${BIDO_ROOT:-$HOME/pytorch_xllll/paper/6-Unknown-Aware Bilateral Dependency Optimization/BiDO_plus-main/low_resolution}"
CELEBA_IMAGES="${CELEBA_IMAGES:-$BIDO_ROOT/attack_datasets/CelebA/Img}"
MANIFEST="${MANIFEST:-$CODE_DIR/experiment_manifests/celeba_bido_clean.json}"
TARGET="${TARGET:-$UTILITY_ROOT/targets/bido/best_validation.pth}"
IG_REPO="${IG_REPO:-$CODE_DIR/third_party/invertinggradients}"
OUT="${OUT:-$CODE_DIR/results_fair_baselines/celeba_vgg_bido}"
ARCFACE_CTX_ID="${ARCFACE_CTX_ID:--1}"
TEST_SAMPLES="${TEST_SAMPLES:-50}"
DLG_STEPS="${DLG_STEPS:-300}"
DLG_RESTARTS="${DLG_RESTARTS:-1}"
IG_STEPS="${IG_STEPS:-4800}"
IG_RESTARTS="${IG_RESTARTS:-3}"
METHODS_TEXT="${METHODS:-fad dlg_joint idlg ig}"
SEEDS_TEXT="${SEEDS:-2027 2028 2029}"
SMOKE="${SMOKE:-0}"
RESTART="${RESTART:-0}"

if [[ "$SMOKE" == "1" ]]; then
  TEST_SAMPLES=1
  DLG_STEPS=2
  DLG_RESTARTS=1
  IG_STEPS=2
  IG_RESTARTS=1
  OUT="${OUT}_smoke"
fi

read -r -a METHODS_ARRAY <<< "$METHODS_TEXT"
read -r -a SEEDS_ARRAY <<< "$SEEDS_TEXT"

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "Missing required file: $1" >&2
    exit 1
  fi
}

require_file "$TARGET"
require_file "$MANIFEST"
require_file "$CODE_DIR/fair_baseline_comparison.py"
require_file "$CODE_DIR/summarize_fair_baselines.py"
require_file "$CODE_DIR/evaluate_fair_baseline_perceptual.py"
require_file "$IG_REPO/inversefed/reconstruction_algorithms.py"

mkdir -p "$OUT/logs"
cd "$CODE_DIR"

for method in "${METHODS_ARRAY[@]}"; do
  case "$method" in
    fad)
      steps=0
      restarts=0
      ;;
    dlg_joint|idlg|dlg_known)
      steps="$DLG_STEPS"
      restarts="$DLG_RESTARTS"
      ;;
    ig)
      steps="$IG_STEPS"
      restarts="$IG_RESTARTS"
      ;;
    *)
      echo "Unknown method: $method" >&2
      exit 1
      ;;
  esac

  for seed in "${SEEDS_ARRAY[@]}"; do
    run_dir="$OUT/$method/seed$seed"
    decoder="$UTILITY_ROOT/decoder_output/models/decoder_utility_matched_bido_seed${seed}.pth"
    restart_args=()
    decoder_args=()
    if [[ "$RESTART" == "1" ]]; then
      restart_args+=(--restart)
    fi
    if [[ "$method" == "fad" ]]; then
      require_file "$decoder"
      decoder_args+=(--decoder_path "$decoder")
    fi

    python fair_baseline_comparison.py \
      --method "$method" \
      --ig_repo "$IG_REPO" \
      --data_root "$CELEBA_IMAGES" \
      --manifest_path "$MANIFEST" \
      --target_weight "$TARGET" \
      "${decoder_args[@]}" \
      --output_dir "$run_dir" \
      --test_samples "$TEST_SAMPLES" \
      --steps "$steps" \
      --restarts "$restarts" \
      --seed "$seed" \
      --num_workers 0 \
      "${restart_args[@]}" \
      2>&1 | tee "$OUT/logs/${method}_seed${seed}.log"

    python evaluate_fair_baseline_perceptual.py \
      --run_dir "$run_dir" \
      --lpips_net alex \
      --arcface_model buffalo_l \
      --arcface_ctx_id "$ARCFACE_CTX_ID" \
      --embedding_size 112 \
      --threshold 0.30 \
      --seed "$seed" \
      2>&1 | tee "$OUT/logs/perceptual_${method}_seed${seed}.log"
  done
done

python summarize_fair_baselines.py \
  --root "$OUT" \
  --methods "${METHODS_ARRAY[@]}" \
  --seeds "${SEEDS_ARRAY[@]}" \
  --reference fad \
  --bootstrap_samples 10000 \
  --bootstrap_seed 2027 \
  --output_json "$OUT/fair_baseline_summary.json" \
  --output_csv "$OUT/fair_baseline_summary.csv" \
  2>&1 | tee "$OUT/logs/summary.log"

echo "Completed: $OUT/fair_baseline_summary.json"
