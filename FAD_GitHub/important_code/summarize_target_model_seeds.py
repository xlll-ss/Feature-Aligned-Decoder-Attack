"""Aggregate FAD robustness across independently trained target-model seeds."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


ATTACK_METRICS = (
    "psnr",
    "ssim",
    "id_loss",
    "lpips",
    "arcface_cosine",
    "relative_feature_error",
    "clean_leaked_output_mse",
    "attack_target_accuracy",
)


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def indexed(records, metric):
    if metric == "attack_target_accuracy":
        return {int(row["idx"]): float(row["target_correct"]) for row in records}
    return {int(row["idx"]): float(row[metric]) for row in records}


def perceptual_indexed(path, metric):
    return {
        int(row["sample_id"]): float(row[metric])
        for row in load(path)["records"]
    }


def mean_std(values):
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean_over_target_seeds": float(array.mean()),
        "std_over_target_seeds": (
            float(array.std(ddof=1)) if array.size > 1 else 0.0
        ),
        "min_over_target_seeds": float(array.min()),
        "max_over_target_seeds": float(array.max()),
    }


def hierarchical_mean_ci(seed_maps, resamples, seed):
    arrays = [
        np.asarray([values[index] for index in sorted(values)], dtype=np.float64)
        for values in seed_maps
    ]
    if any(array.size == 0 for array in arrays):
        raise RuntimeError("Cannot bootstrap an empty target-seed result")
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(resamples, dtype=np.float64)
    for bootstrap_index in range(resamples):
        selected_targets = rng.integers(0, len(arrays), size=len(arrays))
        target_means = []
        for target_index in selected_targets:
            values = arrays[target_index]
            selected_images = rng.integers(0, len(values), size=len(values))
            target_means.append(values[selected_images].mean())
        bootstrap[bootstrap_index] = np.mean(target_means)
    return [float(value) for value in np.percentile(bootstrap, [2.5, 97.5])]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument(
        "--target_seeds", type=int, nargs="+", default=[2027, 2028, 2029]
    )
    parser.add_argument("--bootstrap_samples", type=int, default=10000)
    parser.add_argument("--bootstrap_seed", type=int, default=2027)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_csv", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.bootstrap_samples <= 0:
        raise ValueError("--bootstrap_samples must be positive")
    if len(set(args.target_seeds)) != len(args.target_seeds):
        raise ValueError("--target_seeds contains duplicates")
    root = Path(args.root).expanduser()
    runs = {}
    reference_hash = None
    reference_samples = None
    for target_seed in args.target_seeds:
        seed_root = root / f"target_seed{target_seed}"
        target = load(seed_root / "target_summary.json")
        attack_dir = seed_root / "attack"
        attack = load(attack_dir / "summary.json")
        if int(target["target_seed"]) != target_seed:
            raise RuntimeError(f"Unexpected target summary seed: {target_seed}")
        if int(attack["seed"]) != target_seed or attack["variant"] != "full":
            raise RuntimeError(f"Unexpected attack metadata for target seed {target_seed}")
        if attack["target_weight"] != target["target_checkpoint"]:
            raise RuntimeError(f"Target checkpoint differs for seed {target_seed}")
        sample_hash = attack["protocol"]["sample_set_sha256"]
        requested_samples = int(attack["requested_samples"])
        if reference_hash is None:
            reference_hash = sample_hash
            reference_samples = requested_samples
        if sample_hash != reference_hash or requested_samples != reference_samples:
            raise RuntimeError(f"Attack test set differs for target seed {target_seed}")

        maps = {}
        for metric in ATTACK_METRICS:
            if metric == "lpips":
                values = perceptual_indexed(attack_dir / "lpips.json", metric)
            elif metric == "arcface_cosine":
                values = perceptual_indexed(attack_dir / "arcface.json", metric)
            else:
                values = indexed(attack["per_sample"], metric)
            if not values:
                raise RuntimeError(f"No {metric} values for target seed {target_seed}")
            maps[metric] = values

        training_protocol = attack["decoder"].get("training_protocol") or {}
        decoder_seed = training_protocol.get("seed")
        runs[target_seed] = {
            "target": target,
            "attack": attack,
            "metric_maps": maps,
            "decoder_seed": decoder_seed,
            "decoder_epochs": attack["decoder"].get("training_epoch"),
            "decoder_auxiliary_samples": attack["protocol"].get(
                "auxiliary_samples"
            ),
        }

    initialization_hashes = [
        runs[seed]["target"].get("initialization_sha256")
        for seed in args.target_seeds
    ]
    if any(value is None for value in initialization_hashes):
        raise RuntimeError("A target checkpoint is missing its initialization hash")
    if len(set(initialization_hashes)) != len(initialization_hashes):
        raise RuntimeError("Target seeds do not have distinct initializations")
    decoder_seeds = {runs[seed]["decoder_seed"] for seed in args.target_seeds}
    if len(decoder_seeds) != 1 or None in decoder_seeds:
        raise RuntimeError("Decoder seed is not fixed across target models")

    rows = []
    for target_seed in args.target_seeds:
        run = runs[target_seed]
        target = run["target"]
        attack = run["attack"]
        row = {
            "target_seed": target_seed,
            "initialization_sha256": target["initialization_sha256"],
            "selected_epoch": target["selected_epoch"],
            "selection_validation_accuracy": target["selection_validation"][
                "accuracy"
            ]["mean"],
            "sealed_test_accuracy": target["sealed_test"]["accuracy"]["mean"],
            "decoder_seed": run["decoder_seed"],
            "decoder_epochs": run["decoder_epochs"],
            "decoder_auxiliary_samples": run["decoder_auxiliary_samples"],
            "decoder_parameter_count": attack["decoder"]["parameter_count"],
            "requested_attack_samples": attack["requested_samples"],
            "valid_rate": attack["valid_rate"],
            "label_inference_accuracy": attack["label_inference_accuracy"]["mean"],
            "decoder_ms": 1000.0 * attack["metrics"]["decoder_seconds"]["mean"],
        }
        for metric in ATTACK_METRICS:
            row[metric] = float(np.mean(list(run["metric_maps"][metric].values())))
        rows.append(row)

    aggregate = {
        "selection_validation_accuracy": mean_std(
            [row["selection_validation_accuracy"] for row in rows]
        ),
        "sealed_test_accuracy": mean_std(
            [row["sealed_test_accuracy"] for row in rows]
        ),
        "valid_rate": mean_std([row["valid_rate"] for row in rows]),
        "label_inference_accuracy": mean_std(
            [row["label_inference_accuracy"] for row in rows]
        ),
        "decoder_ms": mean_std([row["decoder_ms"] for row in rows]),
    }
    for metric_index, metric in enumerate(ATTACK_METRICS):
        result = mean_std([row[metric] for row in rows])
        result["hierarchical_bootstrap_ci95"] = hierarchical_mean_ci(
            [runs[seed]["metric_maps"][metric] for seed in args.target_seeds],
            args.bootstrap_samples,
            args.bootstrap_seed + metric_index,
        )
        aggregate[metric] = result

    training_configs = {
        json.dumps(
            runs[seed]["target"]["training_hyperparameters"], sort_keys=True
        )
        for seed in args.target_seeds
    }
    partition_configs = {
        json.dumps(runs[seed]["target"].get("partition_sha256"), sort_keys=True)
        for seed in args.target_seeds
    }
    output = {
        "experiment": "FAD robustness across independently trained target-model seeds",
        "target_seeds": args.target_seeds,
        "controls": {
            "distinct_target_initializations": True,
            "same_target_training_hyperparameters": len(training_configs) == 1,
            "same_target_data_partitions": len(partition_configs) == 1,
            "test_never_used_for_target_selection": all(
                not runs[seed]["target"]["test_used_for_selection"]
                for seed in args.target_seeds
            ),
            "fixed_decoder_seed": next(iter(decoder_seeds)),
            "same_decoder_training_epochs": len(
                {runs[seed]["decoder_epochs"] for seed in args.target_seeds}
            )
            == 1,
            "same_decoder_auxiliary_sample_count": len(
                {
                    runs[seed]["decoder_auxiliary_samples"]
                    for seed in args.target_seeds
                }
            )
            == 1,
            "same_ordered_attack_test_images": True,
            "same_unknown_label_weight_bias_interface": True,
            "attack_samples_per_target": reference_samples,
        },
        "bootstrap": {
            "method": "hierarchical resampling over target seeds and image IDs",
            "samples": args.bootstrap_samples,
            "seed": args.bootstrap_seed,
        },
        "per_target_seed": rows,
        "aggregate": aggregate,
    }
    output_path = Path(args.output_json).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    csv_path = Path(args.output_csv).expanduser()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(
            f"target_seed={row['target_seed']} utility={row['sealed_test_accuracy']:.4f} "
            f"PSNR={row['psnr']:.3f} SSIM={row['ssim']:.4f} "
            f"LPIPS={row['lpips']:.4f}"
        )
    print(f"summary_json: {output_path}")
    print(f"summary_csv: {csv_path}")


if __name__ == "__main__":
    main()
