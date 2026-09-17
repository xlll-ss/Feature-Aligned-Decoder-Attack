# Feature-Aligned Decoder Attack (FAD)

Research code for evaluating final-layer gradient leakage from
BiDO+-protected image classifiers. FAD recovers a classifier-input feature
analytically and reconstructs an image with a decoder trained on disjoint
auxiliary data.

## Scope

For a single sample and a biased linear head, a usable weight-gradient row
equals its bias-gradient entry times the input feature. Their ratio recovers
that feature without knowing the private label. The relation is classical;
this project evaluates its interaction with representation defense. Exact
feature recovery does not imply exact image recovery. Aggregated gradients
and local model deltas are separate boundary diagnostics.

The principal controlled target is CelebA + VGG. CIFAR-10 and Fashion-MNIST
ResNet18 runs demonstrate transfer across image domains. The low-utility
CelebA ResNet18 run is retained as a qualitative demonstration, not a
quantitative architecture comparison. No Scale-MIA experiment is included.

## Repository layout

- `important_code/`: target/decoder training, attacks, evaluations,
  summary generation, plotting, and experiment launchers.
- `important_code/experiment_manifests/`: existing CelebA partition metadata;
  machine-specific paths are replaced, with assignments, labels, and
  path-list hashes preserved.
- `tests/`: existing checks for partitions, recovery, and checkpoint mapping.
- `docs/`: experiment map, source provenance, and local result inventory.

Datasets, model checkpoints, virtual environments, cached files, and private
experiment outputs are not bundled. Obtain datasets and pretrained models
separately under their original distribution terms.

## Environment

Use Linux/Bash for the experiment launchers. Install a compatible PyTorch
and torchvision pair for your CUDA environment first, then install the
remaining dependencies:

```bash
python -m pip install -r requirements.txt
python -m pip install -r requirements-arcface.txt
python tests/run_tests.py
```

The ArcFace dependency file uses CPU ONNX Runtime. For GPU inference,
replace `onnxruntime` with a compatible `onnxruntime-gpu` package; do not
install both. The supplied CelebA launchers invoke perceptual/identity
evaluation and therefore need these dependencies and the InsightFace model.
LPIPS and InsightFace may obtain pretrained weights on first use. Exact
server package versions were not recovered; record your environment before
new experiments. No local GPU experiment was rerun during repository cleanup.

## Configure data and launch experiments

Run from `important_code/`. Use absolute dataset/model paths. These variables
override legacy server defaults in the launchers:

```bash
cd important_code
export CODE_DIR="$PWD"
export BIDO_ROOT="/absolute/path/to/BiDO_plus-main/low_resolution"
export CELEBA_IMAGES="/absolute/path/to/CelebA/Img"
export PRETRAINED_VGG="/absolute/path/to/vgg16_bn-6c64b313.pth"
export BASE_MANIFEST="$CODE_DIR/experiment_manifests/celeba_bido_clean.json"
export MANIFEST="$BASE_MANIFEST"
export ARCFACE_CTX_ID=-1
```

The bundled manifest uses relative image filenames. Dataset loaders receive
`CELEBA_IMAGES` as the root override. Path-only normalization does not alter
experimental partitions or class IDs. Keep the manifests fixed across runs.

Run each experiment individually, preserving the preceding outputs:

```bash
bash run_celeba_vgg_utility_matched.sh
bash run_interface_ablation.sh
bash setup_ig_baseline.sh
bash run_fair_baseline_comparison.sh
bash run_decoder_core_ablation.sh
bash run_multi_target_seed_experiment.sh
bash run_defense_tradeoff_sweep.sh
```

These are full experiments and can take substantial GPU time. Several
launchers support `SMOKE=1`; inspect their settings before using that mode.
The multi-target run provides the strength-one decoder reused by the
defense sweep. Fair comparison uses the official Inverting Gradients
implementation obtained by `setup_ig_baseline.sh`; its original license
remains in the downloaded checkout. To pin a checkout, set `IG_COMMIT` to
the commit used on your server. That commit is not recoverable from the
empty local third-party copy, so an exact historical match is not promised.

Dataset and boundary commands are documented in [the experiment map](docs/experiments.md).
Each Python entry point also exposes its complete options with `--help`.

## Experimental records

The local source contains many zero-byte result files, including final
summary JSON files. This repository does not present them as valid evidence
and does not fabricate replacement measurements from the paper. The
inventory in `docs/local_result_inventory.csv` records their state before
cleanup. Restore original server summaries and per-image measurements before
publishing a reproducibility bundle.

The repository has not been uploaded to GitHub. No project license has been
chosen by the author; select an appropriate license before public release.
