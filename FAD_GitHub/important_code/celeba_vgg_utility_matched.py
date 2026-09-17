"""Train and select a rigorously utility-matched CelebA VGG target pair.

The script creates a fixed, stratified validation split from the official
BiDO+ training list. Both targets start from the same initialization and see
the same epoch-wise data order. The defended checkpoint is selected by best
validation accuracy; the undefended checkpoint is selected only by proximity
to that validation accuracy. The attack/test partition is never used for
checkpoint selection.
"""

import argparse
import hashlib
import json
import math
import random
import shutil
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from experiment_utils import seed_everything, summarize_values
from train_decoder_multidataset import FlexibleVGG, extract_state_dict


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PARTITIONS = ("target_train", "selection_validation", "attack_test", "auxiliary")


def read_json(path):
    return json.loads(Path(path).expanduser().read_text(encoding="utf-8"))


def write_json(path, data):
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def state_sha256(state_dict):
    digest = hashlib.sha256()
    for key in sorted(state_dict):
        value = state_dict[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def create_protocol_manifest(base_manifest_path, output_path, val_fraction, split_seed):
    base = read_json(base_manifest_path)
    assignments = base.get("assignments", {})
    labels = base.get("labels", {})
    if not assignments or not labels:
        raise RuntimeError("Base manifest must contain assignments and labels")

    official_train = defaultdict(list)
    for rel_path, partition in assignments.items():
        if partition == "validation":
            if rel_path not in labels:
                raise RuntimeError(f"Missing target label for {rel_path}")
            official_train[int(labels[rel_path])].append(rel_path)
    if len(official_train) != 1000:
        raise RuntimeError(
            f"Expected 1000 target identities in official train split, got {len(official_train)}"
        )

    target_train, selection_validation = set(), set()
    rng = random.Random(split_seed)
    for label in sorted(official_train):
        paths = sorted(official_train[label])
        rng.shuffle(paths)
        if len(paths) < 2:
            raise RuntimeError(f"Identity {label} has fewer than two training images")
        n_val = max(1, int(round(len(paths) * val_fraction)))
        n_val = min(n_val, len(paths) - 1)
        selection_validation.update(paths[:n_val])
        target_train.update(paths[n_val:])

    protocol_assignments = {}
    for rel_path, partition in assignments.items():
        if rel_path in target_train:
            protocol_assignments[rel_path] = "target_train"
        elif rel_path in selection_validation:
            protocol_assignments[rel_path] = "selection_validation"
        elif partition == "test":
            protocol_assignments[rel_path] = "attack_test"
        elif partition == "auxiliary":
            protocol_assignments[rel_path] = "auxiliary"
        else:
            protocol_assignments[rel_path] = "excluded"

    overlap = target_train & selection_validation
    attack_test = {
        path for path, partition in protocol_assignments.items() if partition == "attack_test"
    }
    if overlap or target_train & attack_test or selection_validation & attack_test:
        raise RuntimeError("Protocol partitions overlap")

    counts = {
        partition: sum(value == partition for value in protocol_assignments.values())
        for partition in (*PARTITIONS, "excluded")
    }
    partition_hashes = {}
    for partition in PARTITIONS:
        paths = sorted(
            path for path, value in protocol_assignments.items() if value == partition
        )
        partition_hashes[partition] = hashlib.sha256(
            "\n".join(paths).encode("utf-8")
        ).hexdigest()

    manifest = {
        "version": 1,
        "protocol": "celeba_vgg_true_utility_matched",
        "root": base["root"],
        "base_manifest": str(Path(base_manifest_path).expanduser()),
        "base_manifest_path_list_sha256": base.get("path_list_sha256"),
        "split_seed": split_seed,
        "selection_validation_fraction_per_identity": val_fraction,
        "selection_rule": "stratified within each identity from official training list",
        "test_policy": "sealed; never used for checkpoint selection",
        "assignments": protocol_assignments,
        "labels": labels,
        "counts": counts,
        "partition_sha256": partition_hashes,
    }
    write_json(output_path, manifest)
    print(f"[OK] protocol manifest: {output_path}")
    print(json.dumps(counts, indent=2))
    return manifest


class ProtocolDataset(Dataset):
    def __init__(self, manifest_path, data_root, partition, train=False):
        if partition not in PARTITIONS:
            raise ValueError(f"Unknown partition: {partition}")
        manifest = read_json(manifest_path)
        self.root = Path(data_root or manifest["root"]).expanduser()
        self.rows = []
        for rel_path, assigned in sorted(manifest["assignments"].items()):
            if assigned == partition:
                self.rows.append((rel_path, int(manifest["labels"].get(rel_path, 1000))))
        if not self.rows:
            raise RuntimeError(f"No samples in partition={partition}")
        missing = [rel for rel, _ in self.rows if not (self.root / rel).is_file()]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} images are missing under {self.root}; first={missing[0]}"
            )
        ops = [transforms.CenterCrop(108), transforms.Resize((64, 64))]
        if train:
            ops.append(transforms.RandomHorizontalFlip(p=0.5))
        ops.append(transforms.ToTensor())
        self.transform = transforms.Compose(ops)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        rel_path, label = self.rows[index]
        image = Image.open(self.root / rel_path).convert("RGB")
        return self.transform(image), label


def remap_pretrained_key(key):
    while key.startswith(("module.", "model.")):
        key = key.split(".", 1)[1]
    if not key.startswith("features."):
        return None
    parts = key.split(".", 2)
    index = int(parts[1])
    for layer_index, (start, stop) in enumerate(
        ((0, 7), (7, 14), (14, 24), (24, 34), (34, 44)), start=1
    ):
        if start <= index < stop:
            return f"layer{layer_index}.{index - start}.{parts[2]}"
    return None


def build_initialized_model(pretrained_path, seed):
    seed_everything(seed)
    model = FlexibleVGG(in_channels=3, num_classes=1000, feature_dim=2048)
    checkpoint = torch.load(Path(pretrained_path).expanduser(), map_location="cpu")
    raw_state = extract_state_dict(checkpoint)
    target_state = model.state_dict()
    mapped = {}
    for key, value in raw_state.items():
        new_key = remap_pretrained_key(key)
        if new_key in target_state and target_state[new_key].shape == value.shape:
            mapped[new_key] = value
    missing, unexpected = model.load_state_dict(mapped, strict=False)
    feature_keys = [key for key in target_state if key.startswith("layer")]
    required_feature_keys = [
        key for key in feature_keys if not key.endswith("num_batches_tracked")
    ]
    missing_required = [key for key in required_feature_keys if key not in mapped]
    missing_counters = [
        key
        for key in feature_keys
        if key.endswith("num_batches_tracked") and key not in mapped
    ]
    if missing_required:
        raise RuntimeError(
            "Pretrained VGG feature extractor is incomplete: "
            f"missing {len(missing_required)} required tensors; "
            f"first={missing_required[0]}"
        )
    # Older torchvision checkpoints omit this BN bookkeeping buffer. Its
    # correct initial value is zero, which is already present in the model.
    for key in missing_counters:
        target_state[key].zero_()
    random_head_keys = [
        key for key in missing if not key.endswith("num_batches_tracked")
    ]
    print(
        f"[OK] pretrained feature tensors={len(mapped)}, "
        f"initialized BN counters={len(missing_counters)}, "
        f"random task-head tensors={len(random_head_keys)}, "
        f"unexpected={len(unexpected)}"
    )
    return model.to(DEVICE)


def forward_full(model, inputs):
    hiddens = []
    x = model.layer1(inputs)
    hiddens.append(x)
    x = model.layer2(x)
    hiddens.append(x)
    x = model.layer3(x)
    hiddens.append(x)
    x = model.layer4(x)
    hiddens.append(x)
    x = model.layer5(x)
    x = model.avgpool(x).flatten(1)
    feature = model.bn(x)
    hiddens.append(feature)
    return hiddens, model.fc_layer(feature), feature


def squared_distance_matrix(x):
    norms = (x * x).sum(dim=1, keepdim=True)
    return (norms - 2.0 * (x @ x.t()) + norms.t()).abs()


def centered_kernel(x, sigma=5.0, linear=False):
    n = x.size(0)
    if linear:
        kernel = x @ x.t()
    else:
        variance = 2.0 * sigma * sigma * x.size(1)
        kernel = torch.exp(-squared_distance_matrix(x) / variance)
    center = torch.eye(n, device=x.device, dtype=x.dtype) - torch.full(
        (n, n), 1.0 / n, device=x.device, dtype=x.dtype
    )
    return kernel @ center


def regularized_projection(kernel):
    """Compute K(K + eps*n*I)^-1 without unstable float32 inversion."""
    matrix = kernel.to(torch.float64)
    n = matrix.size(0)
    eye = torch.eye(n, device=matrix.device, dtype=matrix.dtype)
    base_ridge = 1e-5 * n
    last_error = None
    for multiplier in (1.0, 10.0, 100.0, 1000.0):
        try:
            regularized = matrix + (base_ridge * multiplier) * eye
            # Solving the transposed system is equivalent to K @ inv(K+ridge*I).
            projection = torch.linalg.solve(regularized.t(), matrix.t()).t()
            if torch.isfinite(projection).all():
                return projection
        except RuntimeError as exc:
            last_error = exc
    raise RuntimeError(
        "HSIC regularized solve remained singular after adaptive ridge"
    ) from last_error


def normalized_hsic(hidden, other):
    kx = centered_kernel(hidden, sigma=5.0, linear=False)
    ky = centered_kernel(other, sigma=5.0, linear=True)
    rx = regularized_projection(kx)
    ry = regularized_projection(ky)
    return (rx * ry.t()).sum()


def bido_dependency_loss(inputs, hiddens, labels, alpha, beta):
    batch_size = inputs.size(0)
    data = inputs.reshape(batch_size, -1)
    targets = F.one_hot(labels, num_classes=1000).float()
    loss = torch.zeros((), device=inputs.device)
    for hidden in hiddens:
        hidden = hidden.reshape(batch_size, -1)
        loss = loss + alpha * normalized_hsic(hidden, data)
        loss = loss - beta * normalized_hsic(hidden, targets)
    return loss


def make_loader(dataset, batch_size, workers, shuffle, epoch_seed):
    seed_everything(epoch_seed)
    generator = torch.Generator()
    generator.manual_seed(epoch_seed)

    def seed_worker(worker_id):
        worker_seed = (epoch_seed + worker_id) % (2**32)
        random.seed(worker_seed)
        np.random.seed(worker_seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=shuffle,
        generator=generator,
        worker_init_fn=seed_worker,
    )


def evaluate(model, dataset, args):
    loader = make_loader(dataset, args.eval_batch_size, args.num_workers, False, args.seed)
    model.eval()
    losses, correct, confidences = [], [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            _, logits, _ = forward_full(model, images)
            losses.extend(F.cross_entropy(logits, labels, reduction="none").cpu().tolist())
            correct.extend((logits.argmax(1) == labels).float().cpu().tolist())
            confidences.extend(F.softmax(logits, 1).amax(1).cpu().tolist())
    return {
        "samples": len(correct),
        "accuracy": summarize_values(correct, seed=args.seed),
        "cross_entropy": summarize_values(losses, seed=args.seed),
        "max_confidence": summarize_values(confidences, seed=args.seed),
    }


def checkpoint_payload(model, optimizer, scheduler, epoch, variant, row, args, init_hash):
    return {
        "state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": epoch,
        "dataset": "celeba",
        "arch": "vgg",
        "img_size": 64,
        "in_channels": 3,
        "num_classes": 1000,
        "feature_dim": 2048,
        "variant": variant,
        "defense": (
            {
                "name": "BiDO-HSIC",
                "alpha": args.alpha,
                "beta": args.beta,
                "normalized_hsic_solver": "float64 regularized solve",
                "base_ridge": "1e-5 * batch_size",
                "adaptive_ridge_multipliers": [1.0, 10.0, 100.0, 1000.0],
            }
            if variant == "bido"
            else {"name": "none", "alpha": 0.0, "beta": 0.0}
        ),
        "selection_validation": row["validation"],
        "initialization_sha256": init_hash,
        "protocol_manifest": str(Path(args.protocol_manifest).expanduser()),
        "args": vars(args),
    }


def train_variant(variant, args):
    output_dir = Path(args.output_root).expanduser() / "targets" / variant
    if args.restart and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    latest_path = output_dir / "latest.pth"
    selected_name = "best_validation.pth" if variant == "bido" else "matched_validation.pth"
    selected_path = output_dir / selected_name
    selection_path = output_dir / "selection.json"
    history_path = output_dir / "history.json"
    epochs = args.bido_epochs if variant == "bido" else args.control_epochs

    train_dataset = ProtocolDataset(
        args.protocol_manifest, args.data_root, "target_train", train=True
    )
    validation_dataset = ProtocolDataset(
        args.protocol_manifest, args.data_root, "selection_validation", train=False
    )
    model = build_initialized_model(args.pretrained_vgg, args.seed)
    init_hash = state_sha256(model.state_dict())
    optimizer = optim.Adam(
        model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=list(args.lr_milestones), gamma=args.lr_gamma
    )
    history, start_epoch = [], 1
    if latest_path.is_file() and not args.restart:
        latest = torch.load(latest_path, map_location=DEVICE)
        if latest.get("variant") != variant or latest.get("initialization_sha256") != init_hash:
            raise RuntimeError(f"Refusing incompatible resume checkpoint: {latest_path}")
        model.load_state_dict(latest["state_dict"], strict=True)
        optimizer.load_state_dict(latest["optimizer_state_dict"])
        scheduler.load_state_dict(latest["scheduler_state_dict"])
        start_epoch = int(latest["epoch"]) + 1
        history = read_json(history_path) if history_path.is_file() else []
        print(f"[RESUME] {variant} from epoch {start_epoch}")

    if variant == "control":
        bido_selection = read_json(
            Path(args.output_root).expanduser() / "targets" / "bido" / "selection.json"
        )
        target_accuracy = float(bido_selection["validation_accuracy"])
    else:
        target_accuracy = None

    selection = read_json(selection_path) if selection_path.is_file() else None
    best_score = (
        float(selection["selection_score"])
        if selection is not None
        else -math.inf
    )

    start = time.time()
    for epoch in range(start_epoch, epochs + 1):
        loader = make_loader(
            train_dataset,
            args.batch_size,
            args.num_workers,
            True,
            args.seed * 100000 + epoch,
        )
        model.train()
        totals = {"loss": 0.0, "ce": 0.0, "dependency": 0.0, "correct": 0, "n": 0}
        for images, labels in loader:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            hiddens, logits, _ = forward_full(model, images)
            ce = F.cross_entropy(logits, labels)
            dependency = torch.zeros((), device=DEVICE)
            if variant == "bido":
                dependency = bido_dependency_loss(
                    images, hiddens, labels, args.alpha, args.beta
                )
            loss = ce + dependency
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            n = images.size(0)
            totals["loss"] += float(loss.detach()) * n
            totals["ce"] += float(ce.detach()) * n
            totals["dependency"] += float(dependency.detach()) * n
            totals["correct"] += int((logits.argmax(1) == labels).sum())
            totals["n"] += n
        scheduler.step()
        validation = evaluate(model, validation_dataset, args)
        val_accuracy = float(validation["accuracy"]["mean"])
        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": totals["loss"] / totals["n"],
            "train_ce": totals["ce"] / totals["n"],
            "train_dependency": totals["dependency"] / totals["n"],
            "train_accuracy": totals["correct"] / totals["n"],
            "validation": validation,
        }
        history.append(row)
        write_json(history_path, history)

        if variant == "bido":
            score = val_accuracy
        else:
            score = -abs(val_accuracy - target_accuracy)
        payload = checkpoint_payload(
            model, optimizer, scheduler, epoch, variant, row, args, init_hash
        )
        torch.save(payload, latest_path)
        if selection is None or score > best_score:
            best_score = score
            torch.save(payload, selected_path)
            selection = {
                "variant": variant,
                "checkpoint": str(selected_path),
                "epoch": epoch,
                "validation_accuracy": val_accuracy,
                "target_validation_accuracy": target_accuracy,
                "absolute_gap": (
                    abs(val_accuracy - target_accuracy) if target_accuracy is not None else None
                ),
                "selection_score": score,
                "selection_partition": "selection_validation",
                "test_used_for_selection": False,
                "initialization_sha256": init_hash,
            }
            write_json(selection_path, selection)
        print(
            f"{variant:7s} epoch={epoch:03d}/{epochs} "
            f"train_acc={row['train_accuracy']:.4f} val_acc={val_accuracy:.4f} "
            f"ce={row['train_ce']:.4f} dep={row['train_dependency']:.5f}"
        )

    if not selected_path.is_file():
        raise RuntimeError(f"No selected checkpoint produced for {variant}")
    print(f"[OK] {variant} selected checkpoint: {selected_path}")
    print(f"[TIME] {(time.time() - start) / 60:.1f} minutes")


def load_selected_model(path):
    checkpoint = torch.load(path, map_location=DEVICE)
    model = FlexibleVGG(in_channels=3, num_classes=1000, feature_dim=2048).to(DEVICE)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model


def finalize(args):
    root = Path(args.output_root).expanduser()
    protocol = read_json(args.protocol_manifest)
    results = {
        "protocol": protocol["protocol"],
        "selection_partition": "selection_validation",
        "test_partition": "attack_test",
        "matching_tolerance": args.match_tolerance,
        "test_used_for_selection": False,
        "variants": {},
    }
    initialization_hashes = {}
    for variant, filename in (("bido", "best_validation.pth"), ("control", "matched_validation.pth")):
        checkpoint_path = root / "targets" / variant / filename
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        initialization_hashes[variant] = checkpoint.get("initialization_sha256")
        model = load_selected_model(checkpoint_path)
        results["variants"][variant] = {
            "checkpoint": str(checkpoint_path),
            "selection": read_json(root / "targets" / variant / "selection.json"),
            "selection_validation": evaluate(
                model,
                ProtocolDataset(args.protocol_manifest, args.data_root, "selection_validation"),
                args,
            ),
            "sealed_test": evaluate(
                model,
                ProtocolDataset(args.protocol_manifest, args.data_root, "attack_test"),
                args,
            ),
        }
    if initialization_hashes["bido"] != initialization_hashes["control"]:
        raise RuntimeError("BiDO and control targets did not start from identical initialization")
    results["shared_initialization_sha256"] = initialization_hashes["bido"]
    bido_val = results["variants"]["bido"]["selection_validation"]["accuracy"]["mean"]
    control_val = results["variants"]["control"]["selection_validation"]["accuracy"]["mean"]
    gap = abs(float(bido_val) - float(control_val))
    results["validation_accuracy_absolute_gap"] = gap
    results["match_achieved"] = gap <= args.match_tolerance
    output = root / "target_pair_summary.json"
    write_json(output, results)
    print(f"BiDO validation accuracy:    {bido_val:.4%}")
    print(f"Control validation accuracy: {control_val:.4%}")
    print(f"Absolute validation gap:     {gap:.4%}")
    print(f"Summary: {output}")
    if not results["match_achieved"]:
        raise RuntimeError(
            f"Utility match failed: gap={gap:.4%} exceeds tolerance={args.match_tolerance:.4%}. "
            "Increase --control_epochs or change only a prespecified control-training hyperparameter."
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=["prepare", "train_bido", "train_control", "finalize", "all"],
        default="all",
    )
    parser.add_argument("--base_manifest", required=True)
    parser.add_argument("--protocol_manifest", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--pretrained_vgg", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--validation_fraction", type=float, default=0.10)
    parser.add_argument("--split_seed", type=int, default=314159)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--bido_epochs", type=int, default=20)
    parser.add_argument("--control_epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--lr_milestones", type=int, nargs="+", default=[40])
    parser.add_argument("--lr_gamma", type=float, default=0.2)
    parser.add_argument("--alpha", type=float, default=0.001)
    parser.add_argument("--beta", type=float, default=0.005)
    parser.add_argument("--match_tolerance", type=float, default=0.003)
    parser.add_argument("--restart", action="store_true")
    args = parser.parse_args()
    if not 0 < args.validation_fraction < 0.5:
        parser.error("--validation_fraction must be in (0, 0.5)")
    if not 0 < args.match_tolerance <= 0.02:
        parser.error("--match_tolerance must be in (0, 0.02]")
    if args.batch_size < 3:
        parser.error("BiDO normalized HSIC requires --batch_size >= 3")
    return args


def main():
    args = parse_args()
    seed_everything(args.seed)
    print(f"device: {DEVICE}")
    if args.stage in ("prepare", "all"):
        create_protocol_manifest(
            args.base_manifest,
            args.protocol_manifest,
            args.validation_fraction,
            args.split_seed,
        )
    elif not Path(args.protocol_manifest).expanduser().is_file():
        raise FileNotFoundError(f"Protocol manifest not found: {args.protocol_manifest}")
    if args.stage in ("train_bido", "all"):
        train_variant("bido", args)
    if args.stage in ("train_control", "all"):
        train_variant("control", args)
    if args.stage in ("finalize", "all"):
        finalize(args)


if __name__ == "__main__":
    main()
