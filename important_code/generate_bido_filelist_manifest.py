"""Build a manifest directly from the official BiDO+ CelebA file lists.

The classifier lists already contain the final 0..999 class IDs. This avoids
reconstructing a potentially different identity-to-class permutation.
"""

import argparse
import hashlib
import json
from pathlib import Path

from experiment_utils import list_images


def read_labeled_list(path):
    rows = []
    for line_number, raw_line in enumerate(Path(path).expanduser().read_text(encoding="utf-8").splitlines(), 1):
        fields = raw_line.strip().split()
        if not fields:
            continue
        if len(fields) < 2:
            raise ValueError(f"Expected filename and label at {path}:{line_number}")
        try:
            rows.append((fields[0].replace("\\", "/"), int(fields[-1])))
        except ValueError as exc:
            raise ValueError(f"Invalid label at {path}:{line_number}: {raw_line}") from exc
    return rows


def read_unlabeled_list(path):
    return [line.strip().replace("\\", "/") for line in Path(path).expanduser().read_text(encoding="utf-8").splitlines() if line.strip()]


def make_image_lookup(paths, root):
    lookup = {}
    for path in paths:
        rel = path.relative_to(root).as_posix()
        keys = (rel, path.name, path.stem)
        for key in keys:
            if key in lookup and lookup[key] != rel:
                raise ValueError(f"Ambiguous image key {key}: {lookup[key]} and {rel}")
            lookup[key] = rel
    return lookup


def resolve_name(name, lookup, source):
    path = Path(name)
    for key in (name, path.name, path.stem):
        if key in lookup:
            return lookup[key]
    raise RuntimeError(f"{source} references missing image: {name}")


def build_manifest(data_root, train_list, test_list, aux_list, output, drop_aux_overlap=False):
    root, paths = list_images(data_root)
    lookup = make_image_lookup(paths, root)
    train_rows = read_labeled_list(train_list)
    test_rows = read_labeled_list(test_list)
    aux_names = read_unlabeled_list(aux_list)

    assignments = {path.relative_to(root).as_posix(): "excluded" for path in paths}
    labels = {path: -1 for path in assignments}
    seen = {}

    def add(name, label, partition, source):
        rel = resolve_name(name, lookup, source)
        previous = seen.get(rel)
        if previous is not None and previous != (partition, label):
            raise ValueError(f"Image appears in conflicting lists: {rel}: {previous} vs {(partition, label)}")
        seen[rel] = (partition, label)
        assignments[rel] = partition
        labels[rel] = int(label)

    for name, label in train_rows:
        if not 0 <= label < 1000:
            raise ValueError(f"Target train label outside 0..999: {name} {label}")
        add(name, label, "validation", str(train_list))
    for name, label in test_rows:
        if not 0 <= label < 1000:
            raise ValueError(f"Target test label outside 0..999: {name} {label}")
        add(name, label, "test", str(test_list))
    target_rel = set(seen)
    dropped_aux_overlap = []
    for name in aux_names:
        rel = resolve_name(name, lookup, str(aux_list))
        if rel in target_rel:
            if drop_aux_overlap:
                dropped_aux_overlap.append(rel)
                continue
            add(name, 1000, "auxiliary", str(aux_list))
        else:
            add(name, 1000, "auxiliary", str(aux_list))

    relative_paths = [path.relative_to(root).as_posix() for path in paths]
    manifest = {
        "version": 3,
        "protocol": "bido_plus_official_filelists",
        "root": str(root),
        "train_list": str(Path(train_list).expanduser().resolve()),
        "test_list": str(Path(test_list).expanduser().resolve()),
        "auxiliary_list": str(Path(aux_list).expanduser().resolve()),
        "image_count": len(paths),
        "path_list_sha256": hashlib.sha256("\n".join(relative_paths).encode("utf-8")).hexdigest(),
        "assignments": assignments,
        "labels": labels,
        "labeled_image_count": len(labels),
        "split_unit": "official_bido_filelists",
        "counts": {
            partition: sum(value == partition for value in assignments.values())
            for partition in ("auxiliary", "validation", "test", "excluded")
        },
        "dropped_aux_overlap_count": len(dropped_aux_overlap),
        "dropped_aux_overlap_preview": dropped_aux_overlap[:20],
        "target_class_count": 1000,
    }
    output = Path(output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--train_list", required=True)
    parser.add_argument("--test_list", required=True)
    parser.add_argument("--aux_list", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--drop_aux_overlap",
        action="store_true",
        help="Drop auxiliary entries that resolve to target train/test images; default is to fail.",
    )
    args = parser.parse_args()
    manifest = build_manifest(
        args.data_root,
        args.train_list,
        args.test_list,
        args.aux_list,
        args.output,
        drop_aux_overlap=args.drop_aux_overlap,
    )
    print(f"Saved manifest: {args.output}")
    print(f"images: {manifest['image_count']}")
    for partition in ("auxiliary", "validation", "test", "excluded"):
        print(f"{partition}: {manifest['counts'][partition]}")
    print(f"dropped_aux_overlap: {manifest['dropped_aux_overlap_count']}")
    print(f"path_list_sha256: {manifest['path_list_sha256']}")


if __name__ == "__main__":
    main()
