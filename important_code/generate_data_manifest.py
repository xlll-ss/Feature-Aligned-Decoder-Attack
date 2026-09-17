import argparse
from collections import Counter

from experiment_utils import build_split_manifest


def main():
    parser = argparse.ArgumentParser(description="Create deterministic, disjoint image partitions for FAD experiments")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--auxiliary_fraction", type=float, default=0.70)
    parser.add_argument("--validation_fraction", type=float, default=0.10)
    parser.add_argument("--label_file", default="", help="Optional whitespace-separated filename/label file")
    parser.add_argument("--label_offset", type=int, default=0, help="Use -1 for one-based identity labels")
    parser.add_argument("--group_by_label", action="store_true", help="Keep each identity in exactly one partition")
    args = parser.parse_args()

    manifest = build_split_manifest(
        args.data_root,
        args.output,
        seed=args.seed,
        auxiliary_fraction=args.auxiliary_fraction,
        validation_fraction=args.validation_fraction,
        label_file=args.label_file,
        label_offset=args.label_offset,
        group_by_label=args.group_by_label,
    )
    counts = Counter(manifest["assignments"].values())
    print(f"Saved manifest: {args.output}")
    print(f"Images: {manifest['image_count']}")
    for partition in ("auxiliary", "validation", "test"):
        print(f"{partition}: {counts[partition]}")
    print(f"path_list_sha256: {manifest['path_list_sha256']}")
    print(f"labeled images: {manifest['labeled_image_count']}")
    print(f"split unit: {manifest['split_unit']}")
    if manifest["labels"]:
        label_values = list(manifest["labels"].values())
        print(f"label range: {min(label_values)}..{max(label_values)}")
        print(f"unique labels: {len(set(label_values))}")
    if args.label_file and manifest["labeled_image_count"] != manifest["image_count"]:
        raise RuntimeError(
            f"Label coverage is incomplete: {manifest['labeled_image_count']} / {manifest['image_count']}"
        )


if __name__ == "__main__":
    main()
