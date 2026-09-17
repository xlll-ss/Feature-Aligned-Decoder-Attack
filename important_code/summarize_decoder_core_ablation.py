"""Aggregate FAD decoder-core ablations across decoder seeds."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def indexed(records, metric):
    return {int(row["idx"]): float(row[metric]) for row in records}


def perceptual_indexed(path, metric):
    records = load(path)["records"]
    return {int(row["sample_id"]): float(row[metric]) for row in records}


def mean_std(values):
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean_over_decoder_seeds": float(array.mean()),
        "std_over_decoder_seeds": (
            float(array.std(ddof=1)) if array.size > 1 else 0.0
        ),
    }


def hierarchical_paired_ci(candidate_maps, reference_maps, resamples, seed):
    deltas = []
    for candidate, reference in zip(candidate_maps, reference_maps):
        shared = sorted(set(candidate) & set(reference))
        if not shared:
            raise RuntimeError("No shared sample IDs for paired decoder comparison")
        deltas.append(
            np.asarray(
                [candidate[index] - reference[index] for index in shared],
                dtype=np.float64,
            )
        )
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(resamples, dtype=np.float64)
    for bootstrap_index in range(resamples):
        selected_seeds = rng.integers(0, len(deltas), size=len(deltas))
        seed_means = []
        for selected_seed in selected_seeds:
            values = deltas[selected_seed]
            selected_images = rng.integers(0, len(values), size=len(values))
            seed_means.append(values[selected_images].mean())
        bootstrap[bootstrap_index] = np.mean(seed_means)
    return [float(value) for value in np.percentile(bootstrap, [2.5, 97.5])]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--variants", nargs="+", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[2027, 2028, 2029])
    parser.add_argument("--reference", default="full")
    parser.add_argument("--bootstrap_samples", type=int, default=10000)
    parser.add_argument("--bootstrap_seed", type=int, default=2027)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_csv", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.reference not in args.variants:
        raise ValueError("--reference must be included in --variants")
    root = Path(args.root)
    runs = {
        variant: {
            seed: load(root / "evaluation" / variant / f"seed{seed}" / "summary.json")
            for seed in args.seeds
        }
        for variant in args.variants
    }
    reference_run = runs[args.reference][args.seeds[0]]
    for variant in args.variants:
        for seed in args.seeds:
            run = runs[variant][seed]
            if run["variant"] != variant or run["seed"] != seed:
                raise RuntimeError(f"Unexpected metadata for variant={variant}, seed={seed}")
            if run["target_weight"] != reference_run["target_weight"]:
                raise RuntimeError(f"Target checkpoint differs for variant={variant}, seed={seed}")
            if (
                run["protocol"]["sample_set_sha256"]
                != reference_run["protocol"]["sample_set_sha256"]
            ):
                raise RuntimeError(f"Test sample set differs for variant={variant}, seed={seed}")
            if run["requested_samples"] != reference_run["requested_samples"]:
                raise RuntimeError(f"Requested sample count differs for variant={variant}, seed={seed}")

    metric_names = (
        "relative_feature_error",
        "clean_leaked_output_mse",
        "psnr",
        "ssim",
        "id_loss",
        "lpips",
        "arcface_cosine",
    )
    aggregated = {}
    for variant_index, variant in enumerate(args.variants):
        first = runs[variant][args.seeds[0]]
        result = {
            "variant_spec": first["decoder"]["variant_spec"],
            "model_config": first["decoder"]["model_config"],
            "parameter_count": first["decoder"]["parameter_count"],
            "decoder_seeds": args.seeds,
            "requested_samples_per_seed": first["requested_samples"],
            "valid_rate": mean_std([runs[variant][seed]["valid_rate"] for seed in args.seeds]),
            "label_inference_accuracy": mean_std(
                [runs[variant][seed]["label_inference_accuracy"]["mean"] for seed in args.seeds]
            ),
        }
        maps_by_metric = {}
        for metric in metric_names:
            seed_maps, seed_means = [], []
            for seed in args.seeds:
                run_dir = root / "evaluation" / variant / f"seed{seed}"
                if metric in {
                    "relative_feature_error",
                    "clean_leaked_output_mse",
                    "psnr",
                    "ssim",
                    "id_loss",
                }:
                    values = indexed(runs[variant][seed]["per_sample"], metric)
                elif metric == "lpips":
                    values = perceptual_indexed(run_dir / "lpips.json", "lpips")
                else:
                    values = perceptual_indexed(
                        run_dir / "arcface.json", "arcface_cosine"
                    )
                if not values:
                    raise RuntimeError(f"No {metric} values for variant={variant}, seed={seed}")
                seed_maps.append(values)
                seed_means.append(float(np.mean(list(values.values()))))
            result[metric] = mean_std(seed_means)
            maps_by_metric[metric] = seed_maps
        result["decoder_seconds"] = mean_std(
            [runs[variant][seed]["metrics"]["decoder_seconds"]["mean"] for seed in args.seeds]
        )
        result["paired_delta_vs_full"] = {}
        for metric_index, metric in enumerate(metric_names):
            reference_maps = []
            for seed in args.seeds:
                reference_dir = root / "evaluation" / args.reference / f"seed{seed}"
                if metric in {
                    "relative_feature_error",
                    "clean_leaked_output_mse",
                    "psnr",
                    "ssim",
                    "id_loss",
                }:
                    values = indexed(runs[args.reference][seed]["per_sample"], metric)
                elif metric == "lpips":
                    values = perceptual_indexed(reference_dir / "lpips.json", "lpips")
                else:
                    values = perceptual_indexed(
                        reference_dir / "arcface.json", "arcface_cosine"
                    )
                reference_maps.append(values)
            candidate_maps = maps_by_metric[metric]
            seed_deltas = []
            for candidate, reference in zip(candidate_maps, reference_maps):
                shared = sorted(set(candidate) & set(reference))
                seed_deltas.append(
                    float(np.mean([candidate[index] - reference[index] for index in shared]))
                )
            result["paired_delta_vs_full"][metric] = {
                "mean": float(np.mean(seed_deltas)),
                "hierarchical_bootstrap_ci95": hierarchical_paired_ci(
                    candidate_maps,
                    reference_maps,
                    args.bootstrap_samples,
                    args.bootstrap_seed + variant_index * 10 + metric_index,
                ),
            }
        aggregated[variant] = result

    output = {
        "experiment": "FAD decoder core ablation",
        "reference_variant": args.reference,
        "controls": {
            "same_utility_matched_bido_target": True,
            "same_auxiliary_partition": len(
                {
                    runs[variant][seed]["protocol"].get("auxiliary_partition")
                    for variant in args.variants
                    for seed in args.seeds
                }
            ) == 1,
            "auxiliary_partitions_by_variant": {
                variant: sorted(
                    {
                        runs[variant][seed]["protocol"].get("auxiliary_partition")
                        for seed in args.seeds
                    },
                    key=lambda value: (value is None, value),
                )
                for variant in args.variants
            },
            "same_auxiliary_sample_count": len(
                {
                    runs[variant][seed]["protocol"].get("auxiliary_samples")
                    for variant in args.variants
                    for seed in args.seeds
                }
            ) == 1,
            "auxiliary_samples_by_variant": {
                variant: sorted(
                    {
                        runs[variant][seed]["protocol"].get("auxiliary_samples")
                        for seed in args.seeds
                    },
                    key=lambda value: (value is None, value),
                )
                for variant in args.variants
            },
            "same_training_epochs": len(
                {
                    runs[variant][seed]["decoder"].get("training_epoch")
                    for variant in args.variants
                    for seed in args.seeds
                }
            ) == 1,
            "training_epochs_by_variant": {
                variant: sorted(
                    {
                        runs[variant][seed]["decoder"].get("training_epoch")
                        for seed in args.seeds
                    },
                    key=lambda value: (value is None, value),
                )
                for variant in args.variants
            },
            "same_ordered_test_images": True,
            "same_unknown_label_weight_bias_interface": True,
            "decoder_seeds": args.seeds,
            "test_samples_per_seed": reference_run["requested_samples"],
        },
        "bootstrap": {
            "method": "hierarchical paired resampling over decoder seeds and image IDs",
            "samples": args.bootstrap_samples,
            "seed": args.bootstrap_seed,
        },
        "variants": aggregated,
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    fields = [
        "variant", "family", "description", "l1_weight", "mse_weight",
        "feat_cycle_weight", "parameter_count", "N_per_seed", "decoder_seeds",
        "valid_rate", "psnr", "psnr_std", "ssim", "ssim_std", "id_loss",
        "id_loss_std", "lpips", "lpips_std", "arcface_cosine",
        "arcface_cosine_std", "decoder_ms", "relative_feature_error",
        "clean_leaked_output_mse",
    ]
    csv_path = Path(args.output_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for variant, result in aggregated.items():
            spec = result["variant_spec"]
            writer.writerow(
                {
                    "variant": variant,
                    "family": spec["family"],
                    "description": spec["description"],
                    "l1_weight": spec["l1_weight"],
                    "mse_weight": spec["mse_weight"],
                    "feat_cycle_weight": spec["feat_cycle_weight"],
                    "parameter_count": result["parameter_count"],
                    "N_per_seed": result["requested_samples_per_seed"],
                    "decoder_seeds": len(result["decoder_seeds"]),
                    "valid_rate": result["valid_rate"]["mean_over_decoder_seeds"],
                    "psnr": result["psnr"]["mean_over_decoder_seeds"],
                    "psnr_std": result["psnr"]["std_over_decoder_seeds"],
                    "ssim": result["ssim"]["mean_over_decoder_seeds"],
                    "ssim_std": result["ssim"]["std_over_decoder_seeds"],
                    "id_loss": result["id_loss"]["mean_over_decoder_seeds"],
                    "id_loss_std": result["id_loss"]["std_over_decoder_seeds"],
                    "lpips": result["lpips"]["mean_over_decoder_seeds"],
                    "lpips_std": result["lpips"]["std_over_decoder_seeds"],
                    "arcface_cosine": result["arcface_cosine"]["mean_over_decoder_seeds"],
                    "arcface_cosine_std": result["arcface_cosine"]["std_over_decoder_seeds"],
                    "decoder_ms": 1000 * result["decoder_seconds"]["mean_over_decoder_seeds"],
                    "relative_feature_error": result["relative_feature_error"]["mean_over_decoder_seeds"],
                    "clean_leaked_output_mse": result["clean_leaked_output_mse"]["mean_over_decoder_seeds"],
                }
            )
    for variant, result in aggregated.items():
        print(
            f"{variant:18s} params={result['parameter_count']:10d} "
            f"PSNR={result['psnr']['mean_over_decoder_seeds']:.3f} "
            f"SSIM={result['ssim']['mean_over_decoder_seeds']:.4f} "
            f"LPIPS={result['lpips']['mean_over_decoder_seeds']:.4f}"
        )
    print(f"summary_json: {output_path}")
    print(f"summary_csv: {csv_path}")


if __name__ == "__main__":
    main()
