"""Build a manifest matching the BiDO+ CelebA1000/OOD protocol.

The reference implementation ranks identities by frequency, keeps the first
1000 identities as the target classes, and uses the next 2000 identities as
auxiliary OOD data. Target images are shuffled with seed 42 and split 90/10.
"""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from experiment_utils import list_images


def read_identity_rows(path):
    rows = []
    for line_number, raw_line in enumerate(Path(path).expanduser().read_text(encoding="utf-8").splitlines(), 1):
        fields = raw_line.strip().split()
        if not fields:
            continue
        if len(fields) < 2:
            raise ValueError(f"Invalid identity line {line_number}: {raw_line}")
        try:
            rows.append((fields[0].replace("\\", "/"), int(fields[-1])))
        except ValueError as exc:
            raise ValueError(f"Invalid identity on line {line_number}: {raw_line}") from exc
    if not rows:
        raise RuntimeError(f"No identity rows found in {path}")
    return rows


def make_identity_lookup(rows):
    lookup = {}
    for filename, identity in rows:
        path = Path(filename)
        keys = {filename, path.name, path.stem}
        for key in keys:
            previous = lookup.get(key)
            if previous is not None and previous != identity:
                raise ValueError(f"Conflicting identities for image key {key}")
            lookup[key] = identity
    return lookup


def resolve_identity(path, root, lookup):
    rel = path.relative_to(root).as_posix()
    for key in (rel, path.name, path.stem):
        if key in lookup:
            return lookup[key]
    raise RuntimeError(f"No identity metadata for image: {rel}")


def generate_manifest(
    data_root,
    identity_file,
    output,
    target_identity_count=1000,
    auxiliary_start=1000,
    auxiliary_end=3000,
    split_seed=42,
):
    root, paths = list_images(data_root)
    rows = read_identity_rows(identity_file)
    lookup = make_identity_lookup(rows)
    identities = [resolve_identity(path, root, lookup) for path in paths]

    counts = Counter()
    for identity in identities:
        counts[identity] += 1
    ranked = sorted(counts, key=lambda identity: counts[identity], reverse=True)
    if len(ranked) < auxiliary_end:
        raise RuntimeError(
            f"Only {len(ranked)} identities found; need at least {auxiliary_end}"
        )
    target_ids = ranked[:target_identity_count]
    auxiliary_ids = ranked[auxiliary_start:auxiliary_end]
    target_mapping = {identity: index for index, identity in enumerate(target_ids)}

    target_indices = np.asarray(
        [index for index, identity in enumerate(identities) if identity in target_mapping],
        dtype=np.int64,
    )
    rng = np.random.RandomState(split_seed)
    rng.shuffle(target_indices)
    train_size = int(0.9 * len(target_indices))
    target_train = set(int(index) for index in target_indices[:train_size])
    target_test = set(int(index) for index in target_indices[train_size:])
    auxiliary_set = set(auxiliary_ids)

    assignments = {}
    labels = {}
    for index, (path, identity) in enumerate(zip(paths, identities)):
        rel = path.relative_to(root).as_posix()
        if identity in auxiliary_set:
            assignments[rel] = "auxiliary"
            labels[rel] = target_identity_count
        elif index in target_train:
            assignments[rel] = "validation"
            labels[rel] = target_mapping[identity]
        elif index in target_test:
            assignments[rel] = "test"
            labels[rel] = target_mapping[identity]
        else:
            assignments[rel] = "excluded"
            labels[rel] = -1

    relative_paths = [path.relative_to(root).as_posix() for path in paths]
    manifest = {
        "version": 2,
        "protocol": "bido_plus_celeba1000_ood",
        "root": str(root),
        "identity_file": str(Path(identity_file).expanduser().resolve()),
        "image_count": len(paths),
        "path_list_sha256": hashlib.sha256("\n".join(relative_paths).encode("utf-8")).hexdigest(),
        "target_identity_count": target_identity_count,
        "auxiliary_identity_range": [auxiliary_start, auxiliary_end],
        "target_identities_ranked": target_ids,
        "auxiliary_identities_ranked": auxiliary_ids,
        "split_seed": split_seed,
        "target_train_fraction": 0.9,
        "assignments": assignments,
        "labels": labels,
        "labeled_image_count": len(labels),
        "split_unit": "bido_identity_frequency_and_target_image_split",
        "counts": {
            "auxiliary": sum(value == "auxiliary" for value in assignments.values()),
            "validation": sum(value == "validation" for value in assignments.values()),
            "test": sum(value == "test" for value in assignments.values()),
            "excluded": sum(value == "excluded" for value in assignments.values()),
            "target_images": len(target_indices),
            "target_identities": len(target_ids),
            "auxiliary_identities": len(auxiliary_ids),
        },
    }
    output = Path(output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--identity_file", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--target_identity_count", type=int, default=1000)
    parser.add_argument("--auxiliary_start", type=int, default=1000)
    parser.add_argument("--auxiliary_end", type=int, default=3000)
    parser.add_argument("--split_seed", type=int, default=42)
    args = parser.parse_args()
    manifest = generate_manifest(
        args.data_root,
        args.identity_file,
        args.output,
        args.target_identity_count,
        args.auxiliary_start,
        args.auxiliary_end,
        args.split_seed,
    )
    print(f"Saved manifest: {args.output}")
    print(f"images: {manifest['image_count']}")
    print(f"target identities: {manifest['counts']['target_identities']}")
    print(f"auxiliary identities: {manifest['counts']['auxiliary_identities']}")
    for partition in ("auxiliary", "validation", "test", "excluded"):
        print(f"{partition}: {manifest['counts'][partition]}")
    print(f"target test fraction: {manifest['counts']['test'] / manifest['counts']['target_images']:.6f}")
    print(f"path_list_sha256: {manifest['path_list_sha256']}")


if __name__ == "__main__":
    main()
