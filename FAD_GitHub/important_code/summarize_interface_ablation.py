"""Aggregate interface-ablation results across decoder seeds."""

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
    return {int(row["sample_id"]): float(row[metric]) for row in load(path)["records"]}


def aggregate_seed_means(values):
    array = np.asarray(values, dtype=np.float64)
    return float(array.mean()), float(array.std(ddof=1)) if array.size > 1 else 0.0


def hierarchical_delta_ci(condition_values, reference_values, samples, seed):
    deltas = []
    for condition, reference in zip(condition_values, reference_values):
        ids = sorted(set(condition) & set(reference))
        if not ids:
            raise RuntimeError("No shared IDs for paired bootstrap")
        deltas.append(
            np.asarray([condition[idx] - reference[idx] for idx in ids], dtype=np.float64)
        )
    rng = np.random.default_rng(seed)
    bootstrap = []
    for _ in range(samples):
        selected_seeds = rng.integers(0, len(deltas), size=len(deltas))
        seed_means = []
        for seed_index in selected_seeds:
            values = deltas[seed_index]
            seed_means.append(values[rng.integers(0, len(values), size=len(values))].mean())
        bootstrap.append(np.mean(seed_means))
    return [float(value) for value in np.percentile(bootstrap, [2.5, 97.5])]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[2027, 2028, 2029])
    parser.add_argument("--reference", default="known_wb_ce")
    parser.add_argument("--bootstrap_samples", type=int, default=10000)
    parser.add_argument("--bootstrap_seed", type=int, default=2027)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_csv", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    seed_summaries = {seed: load(root / f"seed{seed}" / "summary.json") for seed in args.seeds}
    conditions = list(seed_summaries[args.seeds[0]]["conditions"])
    if args.reference not in conditions:
        raise RuntimeError(f"Reference condition is absent: {args.reference}")
    for seed, summary in seed_summaries.items():
        if list(summary["conditions"]) != conditions:
            raise RuntimeError(f"Condition order differs for seed={seed}")
    expected_ids = {}
    for seed, summary in seed_summaries.items():
        for condition in conditions:
            current = summary["conditions"][condition]
            ids = tuple(sorted(int(row["idx"]) for row in current["per_sample"]))
            if condition not in expected_ids:
                expected_ids[condition] = ids
            if ids != expected_ids[condition]:
                raise RuntimeError(
                    f"Evaluated sample IDs differ for seed={seed}, condition={condition}"
                )

    results = {}
    paired_metrics = (
        "relative_feature_error",
        "feature_cosine",
        "feature_norm_ratio",
        "psnr",
        "ssim",
        "id_loss",
        "lpips",
        "arcface_cosine",
    )
    for condition_index, condition in enumerate(conditions):
        first = seed_summaries[args.seeds[0]]["conditions"][condition]
        result = {
            "protocol": first["protocol"],
            "decoder_seeds": args.seeds,
            "valid_rate": first["valid_rate"],
            "label_inference_accuracy": (
                None
                if first["label_inference_accuracy"] is None
                else first["label_inference_accuracy"]["mean"]
            ),
        }
        for metric in (
            "relative_feature_error",
            "feature_cosine",
            "feature_norm_ratio",
            "psnr",
            "ssim",
            "id_loss",
        ):
            means = [
                seed_summaries[seed]["conditions"][condition]["metrics"][metric]["mean"]
                for seed in args.seeds
            ]
            mean, std = aggregate_seed_means(means)
            result[metric] = {"mean_over_decoder_seeds": mean, "std_over_decoder_seeds": std}
        lpips_means, arcface_means, pass_means = [], [], []
        for seed in args.seeds:
            condition_dir = root / f"seed{seed}" / condition
            lpips_data = load(condition_dir / "lpips.json")
            arcface_data = load(condition_dir / "arcface.json")
            lpips_means.append(lpips_data["lpips"]["mean"])
            arcface_means.append(arcface_data["arcface_cosine"]["mean"])
            pass_means.append(arcface_data["similarity_above_threshold"]["mean"])
        for metric, means in (
            ("lpips", lpips_means),
            ("arcface_cosine", arcface_means),
            ("arcface_pass_rate", pass_means),
        ):
            mean, std = aggregate_seed_means(means)
            result[metric] = {"mean_over_decoder_seeds": mean, "std_over_decoder_seeds": std}

        result["paired_delta_vs_reference"] = {}
        for metric_offset, metric in enumerate(paired_metrics):
            condition_values, reference_values = [], []
            for seed in args.seeds:
                if metric not in ("lpips", "arcface_cosine"):
                    condition_values.append(
                        indexed(seed_summaries[seed]["conditions"][condition]["per_sample"], metric)
                    )
                    reference_values.append(
                        indexed(seed_summaries[seed]["conditions"][args.reference]["per_sample"], metric)
                    )
                elif metric == "lpips":
                    condition_values.append(
                        perceptual_indexed(root / f"seed{seed}" / condition / "lpips.json", "lpips")
                    )
                    reference_values.append(
                        perceptual_indexed(root / f"seed{seed}" / args.reference / "lpips.json", "lpips")
                    )
                else:
                    condition_values.append(
                        perceptual_indexed(
                            root / f"seed{seed}" / condition / "arcface.json", "arcface_cosine"
                        )
                    )
                    reference_values.append(
                        perceptual_indexed(
                            root / f"seed{seed}" / args.reference / "arcface.json", "arcface_cosine"
                        )
                    )
            seed_deltas = []
            for candidate, reference in zip(condition_values, reference_values):
                ids = sorted(set(candidate) & set(reference))
                seed_deltas.append(np.mean([candidate[idx] - reference[idx] for idx in ids]))
            result["paired_delta_vs_reference"][metric] = {
                "mean": float(np.mean(seed_deltas)),
                "hierarchical_bootstrap_ci95": hierarchical_delta_ci(
                    condition_values,
                    reference_values,
                    args.bootstrap_samples,
                    args.bootstrap_seed + condition_index * 10 + metric_offset,
                ),
            }
        results[condition] = result

    output = {
        "experiment": "FAD label-knowledge and released-interface ablation",
        "reference_condition": args.reference,
        "decoder_seeds": args.seeds,
        "bootstrap": {
            "method": "hierarchical paired resampling over decoder seeds and image IDs",
            "samples": args.bootstrap_samples,
            "seed": args.bootstrap_seed,
        },
        "conditions": results,
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    fields = [
        "condition", "label_known", "observed", "loss", "valid_rate",
        "label_inference_accuracy", "relative_feature_error", "feature_cosine",
        "feature_norm_ratio", "psnr", "psnr_std", "ssim", "ssim_std",
        "lpips", "lpips_std", "arcface_cosine", "arcface_cosine_std",
        "arcface_pass_rate",
    ]
    csv_path = Path(args.output_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for condition, result in results.items():
            protocol = result["protocol"]
            writer.writerow(
                {
                    "condition": condition,
                    "label_known": protocol["label_known"],
                    "observed": protocol["observed"],
                    "loss": f"CE(ls={protocol['label_smoothing']},T={protocol['temperature']})",
                    "valid_rate": result["valid_rate"],
                    "label_inference_accuracy": result["label_inference_accuracy"],
                    "relative_feature_error": result["relative_feature_error"]["mean_over_decoder_seeds"],
                    "feature_cosine": result["feature_cosine"]["mean_over_decoder_seeds"],
                    "feature_norm_ratio": result["feature_norm_ratio"]["mean_over_decoder_seeds"],
                    "psnr": result["psnr"]["mean_over_decoder_seeds"],
                    "psnr_std": result["psnr"]["std_over_decoder_seeds"],
                    "ssim": result["ssim"]["mean_over_decoder_seeds"],
                    "ssim_std": result["ssim"]["std_over_decoder_seeds"],
                    "lpips": result["lpips"]["mean_over_decoder_seeds"],
                    "lpips_std": result["lpips"]["std_over_decoder_seeds"],
                    "arcface_cosine": result["arcface_cosine"]["mean_over_decoder_seeds"],
                    "arcface_cosine_std": result["arcface_cosine"]["std_over_decoder_seeds"],
                    "arcface_pass_rate": result["arcface_pass_rate"]["mean_over_decoder_seeds"],
                }
            )
    for condition, result in results.items():
        print(
            f"{condition:34s} feat_err={result['relative_feature_error']['mean_over_decoder_seeds']:.3e} "
            f"PSNR={result['psnr']['mean_over_decoder_seeds']:.2f} "
            f"LPIPS={result['lpips']['mean_over_decoder_seeds']:.4f}"
        )
    print(f"summary_json: {output_path}")
    print(f"summary_csv: {csv_path}")


if __name__ == "__main__":
    main()
