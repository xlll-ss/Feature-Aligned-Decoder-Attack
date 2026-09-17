import argparse
import csv
import json
from pathlib import Path

import numpy as np


def close_enough(left, right, tolerance=1e-6):
    return left is not None and right is not None and abs(float(left) - float(right)) <= tolerance


def validate_single_sample(data):
    records = data.get("per_sample", [])
    issues = []
    if not records:
        issues.append("missing per_sample records")
        return issues
    for metric, legacy_key in (("psnr", "avg_psnr"), ("ssim", "avg_ssim"), ("id_loss", "avg_id_loss")):
        values = [record[metric] for record in records if metric in record]
        if len(values) != len(records):
            issues.append(f"missing {metric} in some records")
            continue
        computed = float(np.mean(values))
        reported = data.get(legacy_key)
        if reported is not None and not close_enough(computed, reported):
            issues.append(f"{legacy_key} mismatch: computed={computed:.8f}, reported={reported:.8f}")
    if data.get("test_samples") != len(records):
        issues.append(f"test_samples mismatch: records={len(records)}, reported={data.get('test_samples')}")
    return issues


def validate_batch(data):
    issues = []
    records = data.get("unique_records", [])
    reported = data.get("unique_candidates_recovered")
    if reported != len(records):
        issues.append(f"unique candidate count mismatch: records={len(records)}, reported={reported}")
    target_count = data.get("unique_label_targets", 0)
    expected_coverage = len(records) / max(target_count, 1)
    if not close_enough(expected_coverage, data.get("unique_candidate_coverage")):
        issues.append("unique_candidate_coverage mismatch")
    return issues


def validate_baselines(data):
    issues = []
    records = data.get("per_sample", [])
    if data.get("evaluated_samples") != len(records):
        issues.append("evaluated_samples mismatch")
    for method, summary in data.get("methods", {}).items():
        for metric in ("psnr", "ssim", "id_loss"):
            values = [record[method][metric] for record in records]
            if values and not close_enough(float(np.mean(values)), summary[metric]["mean"]):
                issues.append(f"{method}.{metric} mean mismatch")
    return issues


def main():
    parser = argparse.ArgumentParser(description="Validate FAD summary JSON files against their per-sample records")
    parser.add_argument("--root", required=True)
    parser.add_argument("--output_csv", default="")
    args = parser.parse_args()

    root = Path(args.root)
    rows = []
    for path in sorted(root.rglob("summary.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if "unique_records" in data:
                experiment_type = "batch_candidates"
                issues = validate_batch(data)
            elif "methods" in data and "per_sample" in data:
                experiment_type = "feature_baselines"
                issues = validate_baselines(data)
            else:
                experiment_type = "single_sample"
                issues = validate_single_sample(data)
        except Exception as exc:
            experiment_type = "unreadable"
            issues = [str(exc)]
        rows.append({
            "path": str(path),
            "experiment_type": experiment_type,
            "status": "PASS" if not issues else "FAIL",
            "issues": " | ".join(issues),
        })

    if not rows:
        raise SystemExit(f"No summary.json files found under {root}")
    for row in rows:
        print(f"{row['status']:4s} {row['experiment_type']:18s} {row['path']}")
        if row["issues"]:
            print(f"     {row['issues']}")
    if args.output_csv:
        output = Path(args.output_csv)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved validation report: {output}")
    if any(row["status"] == "FAIL" for row in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
