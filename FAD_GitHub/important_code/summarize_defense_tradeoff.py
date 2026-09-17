"""Summarize the BiDO defense-strength/privacy/utility sweep."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


METRICS = (
    "psnr",
    "ssim",
    "id_loss",
    "lpips",
    "arcface_cosine",
    "attack_target_accuracy",
)


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def strength_tag(value):
    return f"{value:g}".replace(".", "p")


def indexed(records, metric):
    if metric == "attack_target_accuracy":
        return {int(row["idx"]): float(row["target_correct"]) for row in records}
    return {int(row["idx"]): float(row[metric]) for row in records}


def perceptual_indexed(path, metric):
    return {
        int(row["sample_id"]): float(row[metric])
        for row in load(path)["records"]
    }


def bootstrap_mean_ci(values, resamples, seed):
    array = np.asarray(list(values.values()), dtype=np.float64)
    if array.size == 0:
        raise RuntimeError("Cannot bootstrap an empty metric")
    rng = np.random.default_rng(seed)
    sampled = rng.choice(array, size=(resamples, array.size), replace=True).mean(axis=1)
    return [float(value) for value in np.percentile(sampled, [2.5, 97.5])]


def paired_delta(candidate, reference, resamples, seed):
    shared = sorted(set(candidate) & set(reference))
    if not shared:
        raise RuntimeError("No shared sample IDs for paired defense comparison")
    values = np.asarray(
        [candidate[index] - reference[index] for index in shared], dtype=np.float64
    )
    rng = np.random.default_rng(seed)
    sampled = rng.choice(values, size=(resamples, values.size), replace=True).mean(axis=1)
    return {
        "mean": float(values.mean()),
        "paired_bootstrap_ci95": [
            float(value) for value in np.percentile(sampled, [2.5, 97.5])
        ],
        "paired_samples": len(shared),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument(
        "--strengths", type=float, nargs="+", default=[0.0, 0.25, 0.5, 1.0, 2.0]
    )
    parser.add_argument("--base_alpha", type=float, default=0.001)
    parser.add_argument("--base_beta", type=float, default=0.005)
    parser.add_argument("--target_seed", type=int, default=2027)
    parser.add_argument("--bootstrap_samples", type=int, default=10000)
    parser.add_argument("--bootstrap_seed", type=int, default=2027)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_csv", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.bootstrap_samples <= 0:
        raise ValueError("--bootstrap_samples must be positive")
    if len(set(args.strengths)) != len(args.strengths):
        raise ValueError("--strengths contains duplicates")
    if any(value < 0 for value in args.strengths):
        raise ValueError("Defense strengths must be non-negative")
    if 0.0 not in args.strengths:
        raise ValueError("The sweep must contain strength 0 as the no-defense reference")

    root = Path(args.root).expanduser()
    runs = {}
    reference_sample_hash = None
    reference_samples = None
    for strength in args.strengths:
        tag = strength_tag(strength)
        point_root = root / f"strength_{tag}"
        target = load(point_root / "target_summary.json")
        attack_dir = point_root / "attack"
        attack = load(attack_dir / "summary.json")
        if int(target["target_seed"]) != args.target_seed:
            raise RuntimeError(f"Unexpected target seed at defense strength {strength:g}")
        if int(attack["seed"]) != args.target_seed or attack["variant"] != "full":
            raise RuntimeError(f"Unexpected attack metadata at strength {strength:g}")
        if attack["target_weight"] != target["target_checkpoint"]:
            raise RuntimeError(f"Target checkpoint differs at strength {strength:g}")

        hyperparameters = target["training_hyperparameters"]
        expected_alpha = args.base_alpha * strength
        expected_beta = args.base_beta * strength
        if not np.isclose(float(hyperparameters["alpha"]), expected_alpha):
            raise RuntimeError(f"Unexpected alpha at strength {strength:g}")
        if not np.isclose(float(hyperparameters["beta"]), expected_beta):
            raise RuntimeError(f"Unexpected beta at strength {strength:g}")

        sample_hash = attack["protocol"]["sample_set_sha256"]
        requested_samples = int(attack["requested_samples"])
        if reference_sample_hash is None:
            reference_sample_hash = sample_hash
            reference_samples = requested_samples
        if sample_hash != reference_sample_hash or requested_samples != reference_samples:
            raise RuntimeError(f"Attack test set differs at strength {strength:g}")

        metric_maps = {}
        for metric in METRICS:
            if metric == "lpips":
                values = perceptual_indexed(attack_dir / "lpips.json", metric)
            elif metric == "arcface_cosine":
                values = perceptual_indexed(attack_dir / "arcface.json", metric)
            else:
                values = indexed(attack["per_sample"], metric)
            if not values:
                raise RuntimeError(f"No {metric} values at strength {strength:g}")
            metric_maps[metric] = values

        decoder_protocol = attack["decoder"].get("training_protocol") or {}
        decoder_target = decoder_protocol.get("target_weight")
        if decoder_target is not None and decoder_target != attack["target_weight"]:
            raise RuntimeError(
                f"Decoder was trained for another target at strength {strength:g}"
            )
        runs[strength] = {
            "tag": tag,
            "target": target,
            "attack": attack,
            "metric_maps": metric_maps,
            "decoder_seed": decoder_protocol.get("seed"),
            "decoder_epochs": attack["decoder"].get("training_epoch"),
            "decoder_auxiliary_samples": attack["protocol"].get(
                "auxiliary_samples"
            ),
        }

    initialization_hashes = {
        runs[strength]["target"].get("initialization_sha256")
        for strength in args.strengths
    }
    if None in initialization_hashes or len(initialization_hashes) != 1:
        raise RuntimeError("Defense-strength targets did not share one initialization")
    decoder_seeds = {runs[value]["decoder_seed"] for value in args.strengths}
    if None in decoder_seeds or len(decoder_seeds) != 1:
        raise RuntimeError("Decoder seed is not fixed across defense strengths")

    target_configs = set()
    partition_configs = set()
    for strength in args.strengths:
        target = runs[strength]["target"]
        config = dict(target["training_hyperparameters"])
        config.pop("alpha", None)
        config.pop("beta", None)
        target_configs.add(json.dumps(config, sort_keys=True))
        partition_configs.add(json.dumps(target.get("partition_sha256"), sort_keys=True))

    reference_maps = runs[0.0]["metric_maps"]
    points = []
    for point_index, strength in enumerate(sorted(args.strengths)):
        run = runs[strength]
        target = run["target"]
        attack = run["attack"]
        utility = target["sealed_test"]["accuracy"]
        selection_utility = target["selection_validation"]["accuracy"]
        point = {
            "strength": strength,
            "strength_tag": run["tag"],
            "alpha": args.base_alpha * strength,
            "beta": args.base_beta * strength,
            "selected_epoch": target["selected_epoch"],
            "selection_validation_accuracy": selection_utility["mean"],
            "sealed_test_accuracy": utility["mean"],
            "sealed_test_accuracy_ci95": [
                utility["ci95"]["lower"],
                utility["ci95"]["upper"],
            ],
            "valid_rate": attack["valid_rate"],
            "label_inference_accuracy": attack["label_inference_accuracy"]["mean"],
            "decoder_seed": run["decoder_seed"],
            "decoder_epochs": run["decoder_epochs"],
            "decoder_auxiliary_samples": run["decoder_auxiliary_samples"],
            "decoder_ms": 1000.0 * attack["metrics"]["decoder_seconds"]["mean"],
            "metrics": {},
            "delta_vs_no_defense": {},
        }
        for metric_index, metric in enumerate(METRICS):
            values = run["metric_maps"][metric]
            point["metrics"][metric] = {
                "mean": float(np.mean(list(values.values()))),
                "image_bootstrap_ci95": bootstrap_mean_ci(
                    values,
                    args.bootstrap_samples,
                    args.bootstrap_seed + point_index * 100 + metric_index,
                ),
            }
            point["delta_vs_no_defense"][metric] = paired_delta(
                values,
                reference_maps[metric],
                args.bootstrap_samples,
                args.bootstrap_seed + point_index * 100 + metric_index + 50,
            )
        points.append(point)

    output = {
        "experiment": "BiDO defense-strength/privacy/utility sweep",
        "strength_definition": {
            "objective": "cross_entropy + alpha*I(X,Z) - beta*I(Y,Z)",
            "alpha": "0.001 * strength",
            "beta": "0.005 * strength",
            "zero_strength": "true no-dependency-defense endpoint",
            "unit_strength": "main-paper BiDO+ setting",
        },
        "metric_directions": {
            "utility_accuracy": "higher is better utility",
            "psnr": "lower means less visual leakage",
            "ssim": "lower means less structural leakage",
            "id_loss": "higher means less target-feature similarity",
            "lpips": "higher means less perceptual similarity",
            "arcface_cosine": "lower means less identity similarity",
        },
        "controls": {
            "target_seed": args.target_seed,
            "same_target_initialization": True,
            "same_target_training_hyperparameters_except_alpha_beta": len(
                target_configs
            )
            == 1,
            "same_target_data_partitions": len(partition_configs) == 1,
            "test_never_used_for_target_selection": all(
                not runs[value]["target"]["test_used_for_selection"]
                for value in args.strengths
            ),
            "fixed_decoder_seed": next(iter(decoder_seeds)),
            "same_decoder_training_epochs": len(
                {runs[value]["decoder_epochs"] for value in args.strengths}
            )
            == 1,
            "same_decoder_auxiliary_sample_count": len(
                {
                    runs[value]["decoder_auxiliary_samples"]
                    for value in args.strengths
                }
            )
            == 1,
            "same_ordered_attack_test_images": True,
            "same_unknown_label_weight_bias_interface": True,
            "attack_samples_per_strength": reference_samples,
        },
        "bootstrap": {
            "method": "image bootstrap per strength; paired image bootstrap versus strength 0",
            "samples": args.bootstrap_samples,
            "seed": args.bootstrap_seed,
        },
        "points": points,
    }
    output_path = Path(args.output_json).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    rows = []
    for point in points:
        row = {
            key: point[key]
            for key in (
                "strength",
                "alpha",
                "beta",
                "selected_epoch",
                "selection_validation_accuracy",
                "sealed_test_accuracy",
                "valid_rate",
                "label_inference_accuracy",
                "decoder_seed",
                "decoder_epochs",
                "decoder_auxiliary_samples",
                "decoder_ms",
            )
        }
        row["sealed_test_accuracy_ci95_lower"] = point[
            "sealed_test_accuracy_ci95"
        ][0]
        row["sealed_test_accuracy_ci95_upper"] = point[
            "sealed_test_accuracy_ci95"
        ][1]
        for metric in METRICS:
            row[metric] = point["metrics"][metric]["mean"]
            row[f"{metric}_ci95_lower"] = point["metrics"][metric][
                "image_bootstrap_ci95"
            ][0]
            row[f"{metric}_ci95_upper"] = point["metrics"][metric][
                "image_bootstrap_ci95"
            ][1]
            row[f"delta_{metric}_vs_zero"] = point["delta_vs_no_defense"][metric][
                "mean"
            ]
        rows.append(row)
    csv_path = Path(args.output_csv).expanduser()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(
            f"strength={row['strength']:g} utility={row['sealed_test_accuracy']:.4f} "
            f"PSNR={row['psnr']:.3f} SSIM={row['ssim']:.4f} "
            f"LPIPS={row['lpips']:.4f}"
        )
    print(f"summary_json: {output_path}")
    print(f"summary_csv: {csv_path}")


if __name__ == "__main__":
    main()
