"""Aggregate fair-baseline runs over attack/decoder seeds."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def indexed(records, metric, id_key="source_idx"):
    return {int(row[id_key]): float(row[metric]) for row in records}


def perceptual_indexed(path, metric):
    records = load(path)["records"]
    result = {}
    for row in records:
        if "sample_id" in row:
            sample_id = int(row["sample_id"])
        else:
            sample_id = int(row["real"].split("_", 1)[0])
        result[sample_id] = float(row[metric])
    return result


def mean_std(values):
    values = np.asarray([value for value in values if value is not None], dtype=np.float64)
    if values.size == 0:
        return {"mean_over_seeds": None, "std_over_seeds": None, "available_seeds": 0}
    return {
        "mean_over_seeds": float(values.mean()),
        "std_over_seeds": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        "available_seeds": int(values.size),
    }


def paired_hierarchical_ci(candidate, reference, resamples, seed):
    paired_deltas = []
    for candidate_seed, reference_seed in zip(candidate, reference):
        shared = sorted(set(candidate_seed) & set(reference_seed))
        if not shared:
            continue
        paired_deltas.append(
            np.asarray(
                [candidate_seed[index] - reference_seed[index] for index in shared],
                dtype=np.float64,
            )
        )
    if not paired_deltas:
        return None
    rng = np.random.default_rng(seed)
    samples = np.empty(resamples, dtype=np.float64)
    for bootstrap_index in range(resamples):
        selected_seeds = rng.integers(0, len(paired_deltas), size=len(paired_deltas))
        seed_means = []
        for selected_seed in selected_seeds:
            values = paired_deltas[selected_seed]
            selected_images = rng.integers(0, len(values), size=len(values))
            seed_means.append(values[selected_images].mean())
        samples[bootstrap_index] = np.mean(seed_means)
    return [float(value) for value in np.percentile(samples, [2.5, 97.5])]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--methods", nargs="+", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[2027, 2028, 2029])
    parser.add_argument("--reference", default="fad")
    parser.add_argument("--bootstrap_samples", type=int, default=10000)
    parser.add_argument("--bootstrap_seed", type=int, default=2027)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_csv", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.reference not in args.methods:
        raise ValueError("--reference must be included in --methods")
    root = Path(args.root)
    runs = {
        method: {
            seed: load(root / method / f"seed{seed}" / "summary.json")
            for seed in args.seeds
        }
        for method in args.methods
    }
    reference_hashes = [
        runs[args.reference][seed]["protocol"]["sample_set_sha256"] for seed in args.seeds
    ]
    for method in args.methods:
        for seed in args.seeds:
            summary = runs[method][seed]
            if summary["method"] != method or summary["seed"] != seed:
                raise RuntimeError(f"Unexpected metadata for method={method}, seed={seed}")
            if summary["target_weight"] != runs[args.reference][args.seeds[0]]["target_weight"]:
                raise RuntimeError(f"Target checkpoint differs for method={method}, seed={seed}")
            if summary["protocol"]["sample_set_sha256"] != reference_hashes[0]:
                raise RuntimeError(f"Sample set differs for method={method}, seed={seed}")
            raw_ids = {int(row["source_idx"]) for row in summary["per_sample"]}
            raw_ids.update(int(row["source_idx"]) for row in summary["failures"])
            if len(raw_ids) != summary["requested_samples"]:
                raise RuntimeError(f"Incomplete sample accounting for method={method}, seed={seed}")

    metric_names = ("psnr", "ssim", "id_loss", "lpips", "arcface_cosine")
    aggregated = {}
    for method_index, method in enumerate(args.methods):
        first = runs[method][args.seeds[0]]
        result = {
            "display_name": first["display_name"],
            "protocol": first["protocol"],
            "steps": first["steps"],
            "restarts": first["restarts"],
            "seeds": args.seeds,
            "requested_samples_per_seed": first["requested_samples"],
            "valid_rate": mean_std([runs[method][seed]["valid_rate"] for seed in args.seeds]),
            "label_inference_accuracy": None,
        }
        label_values = [
            runs[method][seed]["label_inference_accuracy"] for seed in args.seeds
        ]
        if all(value is not None for value in label_values):
            result["label_inference_accuracy"] = mean_std(
                [value["mean"] for value in label_values]
            )

        per_metric_by_seed = {}
        for metric in metric_names:
            seed_means = []
            seed_maps = []
            for seed in args.seeds:
                run_dir = root / method / f"seed{seed}"
                if metric in ("psnr", "ssim", "id_loss"):
                    values = indexed(runs[method][seed]["per_sample"], metric)
                elif metric == "lpips":
                    values = perceptual_indexed(run_dir / "lpips.json", "lpips")
                else:
                    values = perceptual_indexed(run_dir / "arcface.json", "arcface_cosine")
                seed_maps.append(values)
                seed_means.append(
                    float(np.mean(list(values.values()))) if values else None
                )
            result[metric] = mean_std(seed_means)
            per_metric_by_seed[metric] = seed_maps

        result["elapsed_seconds"] = mean_std(
            [runs[method][seed]["metrics"]["elapsed_seconds"]["mean"] for seed in args.seeds]
        )
        result["paired_delta_vs_fad"] = {}
        for metric_index, metric in enumerate(metric_names):
            reference_maps = []
            for seed in args.seeds:
                reference_dir = root / args.reference / f"seed{seed}"
                if metric in ("psnr", "ssim", "id_loss"):
                    values = indexed(runs[args.reference][seed]["per_sample"], metric)
                elif metric == "lpips":
                    values = perceptual_indexed(reference_dir / "lpips.json", "lpips")
                else:
                    values = perceptual_indexed(
                        reference_dir / "arcface.json", "arcface_cosine"
                    )
                reference_maps.append(values)
            candidate_maps = per_metric_by_seed[metric]
            seed_deltas = []
            for candidate, reference in zip(candidate_maps, reference_maps):
                shared = sorted(set(candidate) & set(reference))
                if shared:
                    seed_deltas.append(
                        float(np.mean([candidate[index] - reference[index] for index in shared]))
                    )
            result["paired_delta_vs_fad"][metric] = {
                "mean": float(np.mean(seed_deltas)) if seed_deltas else None,
                "hierarchical_bootstrap_ci95": paired_hierarchical_ci(
                    candidate_maps,
                    reference_maps,
                    args.bootstrap_samples,
                    args.bootstrap_seed + method_index * 10 + metric_index,
                ),
            }
        aggregated[method] = result

    output = {
        "experiment": "fair same-interface reconstruction baseline comparison",
        "reference_method": args.reference,
        "fairness_constraints": {
            "same_target_checkpoint": True,
            "same_ordered_test_samples": True,
            "same_batch_size": 1,
            "same_observation": ["fc_layer.weight", "fc_layer.bias"],
            "unknown_label_for_main_methods": ["fad", "dlg_joint", "idlg", "ig"],
            "known_label_diagnostic": (
                "dlg_known" if "dlg_known" in args.methods else None
            ),
            "stochastic_runs": len(args.seeds),
            "metric_population": "successful reconstructions; valid rate shown separately",
        },
        "bootstrap": {
            "method": "hierarchical paired resampling over seeds and shared image IDs",
            "samples": args.bootstrap_samples,
            "seed": args.bootstrap_seed,
        },
        "methods": aggregated,
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    fields = [
        "method", "display_name", "label_access", "gradient_scope", "N_per_seed",
        "seeds", "steps", "restarts", "offline_auxiliary_images", "valid_rate",
        "label_inference_accuracy", "psnr", "psnr_std", "ssim", "ssim_std",
        "id_loss", "id_loss_std", "lpips", "lpips_std", "arcface_cosine",
        "arcface_cosine_std", "online_seconds_per_sample",
    ]
    csv_path = Path(args.output_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for method, result in aggregated.items():
            label_accuracy = result["label_inference_accuracy"]
            writer.writerow(
                {
                    "method": method,
                    "display_name": result["display_name"],
                    "label_access": result["protocol"]["label_access"],
                    "gradient_scope": "+".join(result["protocol"]["gradient_scope"]),
                    "N_per_seed": result["requested_samples_per_seed"],
                    "seeds": len(result["seeds"]),
                    "steps": result["steps"],
                    "restarts": result["restarts"],
                    "offline_auxiliary_images": result["protocol"]["offline_auxiliary_images"],
                    "valid_rate": result["valid_rate"]["mean_over_seeds"],
                    "label_inference_accuracy": (
                        label_accuracy["mean_over_seeds"] if label_accuracy else None
                    ),
                    "psnr": result["psnr"]["mean_over_seeds"],
                    "psnr_std": result["psnr"]["std_over_seeds"],
                    "ssim": result["ssim"]["mean_over_seeds"],
                    "ssim_std": result["ssim"]["std_over_seeds"],
                    "id_loss": result["id_loss"]["mean_over_seeds"],
                    "id_loss_std": result["id_loss"]["std_over_seeds"],
                    "lpips": result["lpips"]["mean_over_seeds"],
                    "lpips_std": result["lpips"]["std_over_seeds"],
                    "arcface_cosine": result["arcface_cosine"]["mean_over_seeds"],
                    "arcface_cosine_std": result["arcface_cosine"]["std_over_seeds"],
                    "online_seconds_per_sample": result["elapsed_seconds"]["mean_over_seeds"],
                }
            )
    for method, result in aggregated.items():
        psnr = result["psnr"]["mean_over_seeds"]
        ssim = result["ssim"]["mean_over_seeds"]
        lpips = result["lpips"]["mean_over_seeds"]
        print(
            f"{method:12s} valid={result['valid_rate']['mean_over_seeds']:.3f} "
            f"PSNR={'NA' if psnr is None else f'{psnr:.3f}'} "
            f"SSIM={'NA' if ssim is None else f'{ssim:.4f}'} "
            f"LPIPS={'NA' if lpips is None else f'{lpips:.4f}'}"
        )
    print(f"summary_json: {output_path}")
    print(f"summary_csv: {csv_path}")


if __name__ == "__main__":
    main()
