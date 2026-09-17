"""Strict paired comparison of defended and undefended reconstruction results."""

import argparse
import json
from pathlib import Path

import numpy as np


ARCFACE_CONFIG_KEYS = (
    "metric",
    "backend",
    "model",
    "embedding_size",
    "threshold",
    "preprocessing",
)


def load_json(path):
    return json.loads(Path(path).expanduser().read_text(encoding="utf-8"))


def require_unique_records(records, id_key, source_name):
    result = {}
    for record in records:
        if id_key not in record:
            raise ValueError(f"Missing '{id_key}' in {source_name}: {record}")
        sample_id = int(record[id_key])
        if sample_id in result:
            raise ValueError(
                f"Duplicate sample_id={sample_id} in {source_name}"
            )
        result[sample_id] = record
    if not result:
        raise ValueError(f"No records in {source_name}")
    return result


def attack_records(path):
    data = load_json(path)
    records = data.get("per_sample", [])
    result = require_unique_records(records, "idx", path)

    for sample_id, record in result.items():
        for metric in ("psnr", "ssim"):
            if metric not in record:
                raise ValueError(
                    f"Missing '{metric}' for sample_id={sample_id} in {path}"
                )
            if not np.isfinite(float(record[metric])):
                raise ValueError(
                    f"Non-finite '{metric}' for sample_id={sample_id} in {path}"
                )

    return result


def lpips_records(path):
    data = load_json(path)
    records = data.get("records", [])
    if not records:
        raise ValueError(f"No LPIPS records in {path}")

    result = {}
    for record in records:
        if "real" not in record or "lpips" not in record:
            raise ValueError(f"Malformed LPIPS record in {path}: {record}")

        filename = Path(record["real"]).name
        prefix = filename.split("_", 1)[0]
        try:
            sample_id = int(prefix)
        except ValueError as exc:
            raise ValueError(
                f"Cannot parse sample ID from LPIPS filename '{filename}' in {path}"
            ) from exc

        if sample_id in result:
            raise ValueError(f"Duplicate sample_id={sample_id} in LPIPS file {path}")

        value = float(record["lpips"])
        if not np.isfinite(value):
            raise ValueError(
                f"Non-finite LPIPS for sample_id={sample_id} in {path}"
            )
        result[sample_id] = value

    return result


def arcface_records(path):
    data = load_json(path)
    records = data.get("records", [])
    result = require_unique_records(records, "sample_id", path)

    total_pairs = int(data.get("total_pairs", -1))
    valid_pairs = int(data.get("valid_pairs", -1))
    if total_pairs != len(result) or valid_pairs != len(result):
        raise ValueError(
            f"ArcFace file must contain only valid stored records: "
            f"total_pairs={total_pairs}, valid_pairs={valid_pairs}, "
            f"records={len(result)} in {path}"
        )

    for sample_id, record in result.items():
        for metric in ("arcface_cosine", "similarity_above_threshold"):
            if metric not in record:
                raise ValueError(
                    f"Missing '{metric}' for sample_id={sample_id} in {path}"
                )

        cosine = float(record["arcface_cosine"])
        if not np.isfinite(cosine):
            raise ValueError(
                f"Non-finite ArcFace cosine for sample_id={sample_id} in {path}"
            )

        if not isinstance(record["similarity_above_threshold"], bool):
            raise ValueError(
                "ArcFace similarity_above_threshold must be boolean for "
                f"sample_id={sample_id} in {path}"
            )

    return data, result


def validate_same_ids(reference, candidate, reference_name, candidate_name):
    reference_ids = set(reference)
    candidate_ids = set(candidate)

    if reference_ids != candidate_ids:
        missing = sorted(reference_ids - candidate_ids)
        extra = sorted(candidate_ids - reference_ids)
        raise RuntimeError(
            f"Sample IDs differ between {reference_name} and {candidate_name}: "
            f"missing_in_{candidate_name}={missing[:10]} "
            f"(total={len(missing)}), "
            f"extra_in_{candidate_name}={extra[:10]} "
            f"(total={len(extra)})"
        )


def validate_arcface_protocol(bido_data, undefended_data):
    mismatches = {}
    for key in ARCFACE_CONFIG_KEYS:
        bido_value = bido_data.get(key)
        undefended_value = undefended_data.get(key)
        if bido_value != undefended_value:
            mismatches[key] = {
                "bido": bido_value,
                "undefended": undefended_value,
            }

    if mismatches:
        raise RuntimeError(
            "BiDO+ and undefended ArcFace protocols differ: "
            f"{json.dumps(mismatches, ensure_ascii=True)}"
        )


def paired_summary(defended, undefended, seed, bootstrap_samples):
    defended = np.asarray(defended, dtype=np.float64)
    undefended = np.asarray(undefended, dtype=np.float64)

    if defended.shape != undefended.shape or defended.size == 0:
        raise ValueError(
            "Paired arrays must be non-empty and have equal shape"
        )
    if not np.all(np.isfinite(defended)) or not np.all(np.isfinite(undefended)):
        raise ValueError("Paired arrays must contain only finite values")

    delta = defended - undefended
    rng = np.random.default_rng(seed)
    sample_indices = rng.integers(
        0,
        delta.size,
        size=(bootstrap_samples, delta.size),
    )
    bootstrap_means = delta[sample_indices].mean(axis=1)
    lower, upper = np.percentile(bootstrap_means, [2.5, 97.5])

    return {
        "n": int(delta.size),
        "defended_mean": float(defended.mean()),
        "undefended_mean": float(undefended.mean()),
        "paired_delta_defended_minus_undefended": float(delta.mean()),
        "paired_delta_ci95": {
            "lower": float(lower),
            "upper": float(upper),
        },
    }


def values_from_attack(records, sample_ids, metric):
    return [float(records[sample_id][metric]) for sample_id in sample_ids]


def values_from_lpips(records, sample_ids):
    return [float(records[sample_id]) for sample_id in sample_ids]


def values_from_arcface(records, sample_ids, metric):
    if metric == "similarity_above_threshold":
        return [
            float(records[sample_id][metric])
            for sample_id in sample_ids
        ]
    return [float(records[sample_id][metric]) for sample_id in sample_ids]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bido_summary", required=True)
    parser.add_argument("--undefended_summary", required=True)
    parser.add_argument("--bido_lpips", default="")
    parser.add_argument("--undefended_lpips", default="")
    parser.add_argument("--bido_arcface", default="")
    parser.add_argument("--undefended_arcface", default="")
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap_samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args()

    if args.bootstrap_samples < 1000:
        raise ValueError("--bootstrap_samples must be at least 1000")

    bido_attack = attack_records(args.bido_summary)
    undefended_attack = attack_records(args.undefended_summary)
    validate_same_ids(
        bido_attack,
        undefended_attack,
        "bido_summary",
        "undefended_summary",
    )
    sample_ids = sorted(bido_attack)

    metrics = {}
    for offset, metric in enumerate(("psnr", "ssim")):
        metrics[metric] = paired_summary(
            values_from_attack(bido_attack, sample_ids, metric),
            values_from_attack(undefended_attack, sample_ids, metric),
            seed=args.seed + offset,
            bootstrap_samples=args.bootstrap_samples,
        )

    if bool(args.bido_lpips) != bool(args.undefended_lpips):
        raise ValueError(
            "Provide both --bido_lpips and --undefended_lpips, or neither"
        )

    if args.bido_lpips:
        bido_lpips = lpips_records(args.bido_lpips)
        undefended_lpips = lpips_records(args.undefended_lpips)
        validate_same_ids(
            bido_attack,
            bido_lpips,
            "bido_summary",
            "bido_lpips",
        )
        validate_same_ids(
            undefended_attack,
            undefended_lpips,
            "undefended_summary",
            "undefended_lpips",
        )
        metrics["lpips"] = paired_summary(
            values_from_lpips(bido_lpips, sample_ids),
            values_from_lpips(undefended_lpips, sample_ids),
            seed=args.seed + 10,
            bootstrap_samples=args.bootstrap_samples,
        )

    if bool(args.bido_arcface) != bool(args.undefended_arcface):
        raise ValueError(
            "Provide both --bido_arcface and --undefended_arcface, or neither"
        )

    arcface_protocol = None
    if args.bido_arcface:
        bido_arcface_data, bido_arcface = arcface_records(args.bido_arcface)
        undefended_arcface_data, undefended_arcface = arcface_records(
            args.undefended_arcface
        )

        validate_arcface_protocol(bido_arcface_data, undefended_arcface_data)
        validate_same_ids(
            bido_attack,
            bido_arcface,
            "bido_summary",
            "bido_arcface",
        )
        validate_same_ids(
            undefended_attack,
            undefended_arcface,
            "undefended_summary",
            "undefended_arcface",
        )

        arcface_protocol = {
            key: bido_arcface_data.get(key)
            for key in ARCFACE_CONFIG_KEYS
        }
        metrics["arcface_cosine"] = paired_summary(
            values_from_arcface(
                bido_arcface,
                sample_ids,
                "arcface_cosine",
            ),
            values_from_arcface(
                undefended_arcface,
                sample_ids,
                "arcface_cosine",
            ),
            seed=args.seed + 20,
            bootstrap_samples=args.bootstrap_samples,
        )
        metrics["arcface_similarity_above_threshold"] = paired_summary(
            values_from_arcface(
                bido_arcface,
                sample_ids,
                "similarity_above_threshold",
            ),
            values_from_arcface(
                undefended_arcface,
                sample_ids,
                "similarity_above_threshold",
            ),
            seed=args.seed + 21,
            bootstrap_samples=args.bootstrap_samples,
        )

    output_data = {
        "comparison": "BiDO+ minus undefended",
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.seed,
        "sample_ids": sample_ids,
        "metrics": metrics,
    }
    if arcface_protocol is not None:
        output_data["arcface_protocol"] = arcface_protocol

    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(output_data, indent=2),
        encoding="utf-8",
    )

    for metric, summary in metrics.items():
        ci = summary["paired_delta_ci95"]
        print(
            f"{metric}: "
            f"BiDO+={summary['defended_mean']:.6f}, "
            f"undefended={summary['undefended_mean']:.6f}, "
            f"paired delta="
            f"{summary['paired_delta_defended_minus_undefended']:+.6f}, "
            f"95% CI=[{ci['lower']:+.6f}, {ci['upper']:+.6f}], "
            f"n={summary['n']}"
        )
    print(f"summary_json: {output}")


if __name__ == "__main__":
    main()