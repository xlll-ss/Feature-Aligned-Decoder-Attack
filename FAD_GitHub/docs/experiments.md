# Experiment map

Run from `important_code/` after setting the variables in the main README.
Launchers default `CODE_DIR` to their own directory in this copy. Original
source files remain unchanged.

| Question | Launcher | Summary entry point |
| --- | --- | --- |
| Does BiDO+ change reconstruction at matched validation utility? | `run_celeba_vgg_utility_matched.sh` | `summarize_utility_matched.py` |
| Are labels necessary, and which gradient tensors matter? | `run_interface_ablation.sh` | `summarize_interface_ablation.py` |
| How does FAD compare with iterative attacks under the same interface? | `run_fair_baseline_comparison.sh` | `summarize_fair_baselines.py` |
| Which decoder components affect image quality? | `run_decoder_core_ablation.sh` | `summarize_decoder_core_ablation.py` |
| Does the result persist across target initializations? | `run_multi_target_seed_experiment.sh` | `summarize_target_model_seeds.py` |
| How do defense strength, privacy, and utility interact? | `run_defense_tradeoff_sweep.sh` | `summarize_defense_tradeoff.py` |

The utility-matched outputs are inputs to subsequent experiments. Existing
launchers retain their tested target/decoder schedules, seed lists, losses,
metrics, and output naming. Data/model path overrides are documented in the
README; inspect each launcher's first lines for additional controls.

## ResNet18 domain experiments

These independent runs use official dataset test splits. Choose a distinct
output root for every dataset and smoke run; sharing roots can incorrectly
reuse checkpoints from another dataset.

```bash
python cifar10_resnet18_generalization.py --help
python fashionmnist_resnet18_generalization.py --help
python celeba_resnet18_ablation.py --help

python cifar10_resnet18_generalization.py \
  --stage all --data_root /absolute/path/to/cifar10 \
  --output_root "$CODE_DIR/results_generalization/cifar10_resnet18" \
  --seeds 2027 2028 2029

python fashionmnist_resnet18_generalization.py \
  --stage all --data_root /absolute/path/to/fashionmnist \
  --output_root "$CODE_DIR/results_generalization/fashionmnist_resnet18" \
  --seeds 2027 2028 2029
```

Inspect the entry-point defaults and archived server configuration when
reproducing published metrics. The CelebA ResNet18 run is a qualitative
demonstration and is not a utility-matched defense comparison.

## Diagnostics and visualizations

- `attack_feature_baselines.py`: mean-image and nearest-auxiliary diagnostics.
- `attack_fad_gradient_diagnostic.py`: observed-interface diagnostics.
- `attack_batch_candidates.py`: conditional candidate recovery from batches
  and deltas; report coverage and precision with conditional image quality.
- `run_fedavg_sweep.sh`: archived local-update diagnostic. Set
  `CONTROL_CHECKPOINT`, `DEFENDED_CHECKPOINT`, `CELEBA_IMAGES`, and `ROOT`
  explicitly before using it. These earlier targets are not utility matched.
- `plot_defense_tradeoff.py`: defense sweep plot from a valid summary JSON.
- `make_three_dataset_batch64.py`: full sample grids from saved raw pairs.
- `evaluate_lpips.py`, `evaluate_arcface.py`: additional metrics from raw pairs.
- `validate_results.py`: saved-result checks.

Use `python SCRIPT.py --help` for the required input/output arguments. A
summary or checkpoint that exists but has zero bytes is not a completed run;
restore it from the server or use a fresh output root before restarting.
