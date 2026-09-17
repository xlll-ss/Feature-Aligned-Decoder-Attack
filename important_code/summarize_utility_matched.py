"""Aggregate utility-matched target and reconstruction comparisons."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def unique(records, key, source):
    result = {}
    for row in records:
        sample_id = int(row[key])
        if sample_id in result:
            raise RuntimeError(f"Duplicate sample {sample_id} in {source}")
        result[sample_id] = row
    return result


def attack_metric(path, metric):
    data = load(path)
    if data.get("saturated_samples") != 0:
        raise RuntimeError(f"Saturated samples are not allowed in paired analysis: {path}")
    rows = unique(data.get("per_sample", []), "idx", path)
    return data, {key: float(value[metric]) for key, value in rows.items()}


def named_metric(path, records_key, id_key, metric):
    data = load(path)
    rows = unique(data.get(records_key, []), id_key, path)
    return data, {key: float(value[metric]) for key, value in rows.items()}


def lpips_metric(path):
    data = load(path)
    values = {}
    for row in data.get("records", []):
        sample_id = int(Path(row["real"]).name.split("_", 1)[0])
        values[sample_id] = float(row["lpips"])
    return data, values


def hierarchical_ci(seed_deltas, bootstrap_samples, seed):
    keys = sorted(seed_deltas)
    arrays = [np.asarray(seed_deltas[key], dtype=np.float64) for key in keys]
    if not arrays or any(array.size == 0 for array in arrays):
        raise RuntimeError("No paired deltas for hierarchical bootstrap")
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(bootstrap_samples):
        selected_seeds = rng.integers(0, len(arrays), size=len(arrays))
        replicate = []
        for seed_index in selected_seeds:
            array = arrays[seed_index]
            replicate.append(array[rng.integers(0, array.size, size=array.size)].mean())
        values.append(float(np.mean(replicate)))
    return [float(x) for x in np.percentile(values, [2.5, 97.5])]


def summarize_metric(metric, seed_pairs, bootstrap_samples, bootstrap_seed):
    seed_rows, seed_deltas = [], {}
    for seed, (bido_values, control_values) in sorted(seed_pairs.items()):
        if set(bido_values) != set(control_values):
            raise RuntimeError(f"Sample IDs differ for {metric}, seed={seed}")
        ids = sorted(bido_values)
        bido = np.asarray([bido_values[key] for key in ids], dtype=np.float64)
        control = np.asarray([control_values[key] for key in ids], dtype=np.float64)
        delta = bido - control
        seed_deltas[seed] = delta
        seed_rows.append(
            {
                "seed": seed,
                "n": len(ids),
                "bido_mean": float(bido.mean()),
                "control_mean": float(control.mean()),
                "delta_bido_minus_control": float(delta.mean()),
            }
        )
    deltas = np.asarray([row["delta_bido_minus_control"] for row in seed_rows])
    return {
        "metric": metric,
        "per_decoder_seed": seed_rows,
        "decoder_seed_count": len(seed_rows),
        "bido_mean_over_seeds": float(np.mean([row["bido_mean"] for row in seed_rows])),
        "control_mean_over_seeds": float(np.mean([row["control_mean"] for row in seed_rows])),
        "paired_delta_mean_over_seeds": float(deltas.mean()),
        "paired_delta_std_over_seeds": float(deltas.std(ddof=1)) if len(deltas) > 1 else 0.0,
        "hierarchical_bootstrap_ci95": hierarchical_ci(
            seed_deltas, bootstrap_samples, bootstrap_seed
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_root", required=True)
    parser.add_argument("--target_summary", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[2027, 2028, 2029])
    parser.add_argument("--bootstrap_samples", type=int, default=10000)
    parser.add_argument("--bootstrap_seed", type=int, default=2027)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_csv", required=True)
    args = parser.parse_args()
    root = Path(args.experiment_root)
    target_summary = load(args.target_summary)
    if not target_summary.get("match_achieved"):
        raise RuntimeError("Target pair did not satisfy the prespecified utility tolerance")

    metric_pairs = {
        name: {} for name in (
            "psnr", "ssim", "id_loss", "relative_feature_error", "lpips",
            "arcface_cosine", "arcface_similarity_above_threshold"
        )
    }
    for seed in args.seeds:
        attack = {}
        lpips = {}
        arcface = {}
        for variant in ("bido", "control"):
            result_dir = root / "attacks" / f"{variant}_seed{seed}"
            attack[variant] = {
                metric: attack_metric(result_dir / "summary.json", metric)[1]
                for metric in ("psnr", "ssim", "id_loss", "relative_feature_error")
            }
            lpips[variant] = lpips_metric(result_dir / "lpips.json")[1]
            arcface[variant] = {
                metric: named_metric(
                    result_dir / "arcface.json", "records", "sample_id", metric
                )[1]
                for metric in ("arcface_cosine", "similarity_above_threshold")
            }
        for metric in attack["bido"]:
            metric_pairs[metric][seed] = (attack["bido"][metric], attack["control"][metric])
        metric_pairs["lpips"][seed] = (lpips["bido"], lpips["control"])
        metric_pairs["arcface_cosine"][seed] = (
            arcface["bido"]["arcface_cosine"], arcface["control"]["arcface_cosine"]
        )
        metric_pairs["arcface_similarity_above_threshold"][seed] = (
            arcface["bido"]["similarity_above_threshold"],
            arcface["control"]["similarity_above_threshold"],
        )

    summaries = []
    for offset, (metric, pairs) in enumerate(metric_pairs.items()):
        summaries.append(
            summarize_metric(
                metric, pairs, args.bootstrap_samples, args.bootstrap_seed + offset
            )
        )
    output = {
        "comparison": "BiDO minus fully undefended, utility matched on independent validation",
        "target_pair": target_summary,
        "decoder_seeds": args.seeds,
        "bootstrap": {
            "method": "hierarchical resampling of decoder seeds and paired images",
            "samples": args.bootstrap_samples,
            "seed": args.bootstrap_seed,
        },
        "metrics": {row["metric"]: row for row in summaries},
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    csv_path = Path(args.output_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "metric", "bido_mean_over_seeds", "control_mean_over_seeds",
        "paired_delta_mean_over_seeds", "paired_delta_std_over_seeds",
        "ci95_lower", "ci95_upper", "decoder_seed_count",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in summaries:
            writer.writerow({
                **{key: row[key] for key in fields if key in row},
                "ci95_lower": row["hierarchical_bootstrap_ci95"][0],
                "ci95_upper": row["hierarchical_bootstrap_ci95"][1],
            })
    for row in summaries:
        ci = row["hierarchical_bootstrap_ci95"]
        print(
            f"{row['metric']}: BiDO={row['bido_mean_over_seeds']:.6f}, "
            f"control={row['control_mean_over_seeds']:.6f}, "
            f"delta={row['paired_delta_mean_over_seeds']:+.6f}, "
            f"95% CI=[{ci[0]:+.6f}, {ci[1]:+.6f}]"
        )
    print(f"summary_json: {output_path}")
    print(f"summary_csv: {csv_path}")


if __name__ == "__main__":
    main()
