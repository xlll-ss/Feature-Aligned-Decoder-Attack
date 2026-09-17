import hashlib
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image

try:
    import torch
    from torch.utils.data import Dataset
except ModuleNotFoundError:
    torch = None

    class Dataset:
        pass


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
PARTITIONS = ("auxiliary", "validation", "test")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    if torch is None:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def list_images(root):
    root = Path(root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Image root not found: {root}")
    paths = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    if not paths:
        raise RuntimeError(f"No images found under: {root}")
    return root, paths


def read_label_file(path, label_offset=0):
    labels = {}
    if not path:
        return labels
    for line_number, raw_line in enumerate(Path(path).expanduser().read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        fields = line.split()
        if len(fields) < 2:
            raise ValueError(f"Invalid label line {line_number}: {raw_line}")
        try:
            labels[fields[0].replace("\\", "/")] = int(fields[-1]) + label_offset
        except ValueError as exc:
            raise ValueError(f"Invalid integer label on line {line_number}: {raw_line}") from exc
    return labels


def build_split_manifest(
    root,
    output_path,
    seed=2027,
    auxiliary_fraction=0.70,
    validation_fraction=0.10,
    label_file="",
    label_offset=0,
    group_by_label=False,
):
    if auxiliary_fraction <= 0 or validation_fraction < 0:
        raise ValueError("Split fractions must be non-negative and auxiliary_fraction must be positive")
    if auxiliary_fraction + validation_fraction >= 1:
        raise ValueError("auxiliary_fraction + validation_fraction must be < 1")

    root, paths = list_images(root)
    relative_paths = [p.relative_to(root).as_posix() for p in paths]
    raw_labels = read_label_file(label_file, label_offset=label_offset)
    labels = {}
    if raw_labels:
        for rel_path in relative_paths:
            if rel_path in raw_labels:
                labels[rel_path] = raw_labels[rel_path]
            elif Path(rel_path).name in raw_labels:
                labels[rel_path] = raw_labels[Path(rel_path).name]

    rng = random.Random(seed)
    n = len(relative_paths)
    assignments = {}
    if group_by_label:
        if len(labels) != n:
            raise ValueError("--group_by_label requires complete label coverage")
        unique_labels = sorted(set(labels.values()))
        rng.shuffle(unique_labels)
        n_aux_labels = int(len(unique_labels) * auxiliary_fraction)
        n_val_labels = int(len(unique_labels) * validation_fraction)
        label_partitions = {}
        for idx, label in enumerate(unique_labels):
            if idx < n_aux_labels:
                partition = "auxiliary"
            elif idx < n_aux_labels + n_val_labels:
                partition = "validation"
            else:
                partition = "test"
            label_partitions[label] = partition
        assignments = {rel_path: label_partitions[labels[rel_path]] for rel_path in relative_paths}
    else:
        shuffled_paths = list(relative_paths)
        rng.shuffle(shuffled_paths)
        n_aux = int(n * auxiliary_fraction)
        n_val = int(n * validation_fraction)
        for idx, rel_path in enumerate(shuffled_paths):
            if idx < n_aux:
                partition = "auxiliary"
            elif idx < n_aux + n_val:
                partition = "validation"
            else:
                partition = "test"
            assignments[rel_path] = partition

    digest = hashlib.sha256("\n".join(sorted(relative_paths)).encode("utf-8")).hexdigest()
    manifest = {
        "version": 1,
        "root": str(root),
        "seed": seed,
        "fractions": {
            "auxiliary": auxiliary_fraction,
            "validation": validation_fraction,
            "test": 1.0 - auxiliary_fraction - validation_fraction,
        },
        "image_count": n,
        "path_list_sha256": digest,
        "assignments": assignments,
        "label_file": str(Path(label_file).expanduser().resolve()) if label_file else None,
        "label_offset": label_offset,
        "labels": labels,
        "labeled_image_count": len(labels),
        "split_unit": "label" if group_by_label else "image",
    }
    output_path = Path(output_path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def load_manifest(path, expected_root=None):
    path = Path(path).expanduser()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    assignments = manifest.get("assignments")
    if not isinstance(assignments, dict) or not assignments:
        raise RuntimeError(f"Manifest has no assignments: {path}")
    root = Path(expected_root or manifest["root"]).expanduser().resolve()
    missing = [rel for rel in assignments if not (root / rel).is_file()]
    if missing:
        preview = ", ".join(missing[:3])
        raise FileNotFoundError(f"Manifest references {len(missing)} missing images under {root}: {preview}")
    return manifest, root


class ManifestImageDataset(Dataset):
    def __init__(self, manifest_path, partition, root_override=None, transform=None, max_samples=None):
        if partition not in PARTITIONS:
            raise ValueError(f"partition must be one of {PARTITIONS}")
        manifest, root = load_manifest(manifest_path, root_override)
        self.root = root
        self.partition = partition
        self.transform = transform
        self.paths = [
            root / rel
            for rel, assigned_partition in sorted(manifest["assignments"].items())
            if assigned_partition == partition
        ]
        if max_samples and max_samples < len(self.paths):
            self.paths = self.paths[:max_samples]
        if not self.paths:
            raise RuntimeError(f"No images assigned to partition={partition}")
        self.labels = manifest.get("labels", {})
        self.missing_label_count = sum(
            1 for path in self.paths if path.relative_to(self.root).as_posix() not in self.labels
        )
        self.has_complete_labels = self.missing_label_count == 0

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        image = Image.open(self.paths[index]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        relative_path = self.paths[index].relative_to(self.root).as_posix()
        return image, int(self.labels.get(relative_path, 0))


def bootstrap_mean_ci(values, confidence=0.95, resamples=2000, seed=2027):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {"mean": None, "lower": None, "upper": None, "n": 0}
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(resamples, values.size), replace=True).mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return {
        "mean": float(values.mean()),
        "lower": float(np.quantile(samples, alpha)),
        "upper": float(np.quantile(samples, 1.0 - alpha)),
        "n": int(values.size),
    }


def summarize_values(values, seed=2027):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {"n": 0, "mean": None, "std": None, "median": None, "min": None, "max": None, "ci95": None}
    return {
        "n": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        "median": float(np.median(values)),
        "min": float(values.min()),
        "max": float(values.max()),
        "ci95": bootstrap_mean_ci(values, seed=seed),
    }
