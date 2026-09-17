import argparse
import csv
import re
from pathlib import Path


SAMPLE_PATTERN = re.compile(r"^(?:Sample|Batch|Eval)\s+\d+", re.MULTILINE)
SUMMARY_PATTERN = re.compile(r"^Attack Summary", re.MULTILINE)
COMMAND_PATTERN = re.compile(r"(?:^|\s)python(?:3)?\s+[^\n]+\.py", re.MULTILINE)
JSON_PATTERN = re.compile(r"summary_json:\s*(\S+)")
METRIC_PATTERN = re.compile(r"(?:PSNR[: ]+)(-?\d+(?:\.\d+)?)", re.IGNORECASE)


def classify(sample_count, summary_count, command_count, has_error):
    if has_error:
        return "FAILED_RUN_PRESENT"
    if sample_count and command_count:
        return "TRACE_WITH_COMMAND"
    if sample_count:
        return "TRACE_NO_COMMAND"
    if summary_count:
        return "SUMMARY_ONLY"
    return "NO_MACHINE_READABLE_RESULT"


def main():
    parser = argparse.ArgumentParser(description="Inventory legacy text logs without trusting manually copied means")
    parser.add_argument("--root", required=True)
    parser.add_argument("--output_csv", required=True)
    args = parser.parse_args()

    root = Path(args.root)
    rows = []
    for path in sorted(root.rglob("*.txt")):
        text = path.read_text(encoding="utf-8", errors="replace")
        sample_count = len(SAMPLE_PATTERN.findall(text))
        summary_count = len(SUMMARY_PATTERN.findall(text))
        command_count = len(COMMAND_PATTERN.findall(text))
        psnr_values = [float(value) for value in METRIC_PATTERN.findall(text)]
        has_error = "Traceback (most recent call last)" in text or "RuntimeError:" in text
        rows.append({
            "path": str(path),
            "bytes": path.stat().st_size,
            "sample_lines": sample_count,
            "attack_summaries": summary_count,
            "command_lines": command_count,
            "summary_json_references": len(JSON_PATTERN.findall(text)),
            "psnr_values": len(psnr_values),
            "distinct_psnr_values": len(set(psnr_values)),
            "has_error": has_error,
            "audit_class": classify(sample_count, summary_count, command_count, has_error),
        })

    output = Path(args.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(
            f"{row['audit_class']:26s} samples={row['sample_lines']:4d} "
            f"summaries={row['attack_summaries']:2d} {row['path']}"
        )
    print(f"Saved audit: {output}")


if __name__ == "__main__":
    main()
