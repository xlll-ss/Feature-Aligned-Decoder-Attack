"""Train and evaluate one independently seeded BiDO CelebA/VGG target."""

import argparse
import hashlib
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Subset

from celeba_vgg_utility_matched import (
    DEVICE,
    ProtocolDataset,
    bido_dependency_loss,
    build_initialized_model,
    checkpoint_payload,
    evaluate,
    forward_full,
    make_loader,
    read_json,
    state_sha256,
    write_json,
)
from train_decoder_multidataset import FlexibleVGG


def load_checkpoint(path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def limited(dataset, maximum):
    if maximum and maximum < len(dataset):
        return Subset(dataset, range(maximum))
    return dataset


def partition_hash(dataset, maximum):
    rows = dataset.rows[:maximum] if maximum and maximum < len(dataset.rows) else dataset.rows
    return hashlib.sha256(
        "\n".join(relative_path for relative_path, _ in rows).encode("utf-8")
    ).hexdigest()


def training_config(args):
    return {
        "experiment": "celeba_vgg_bido_target_seed",
        "target_seed": args.target_seed,
        "protocol_manifest": str(Path(args.protocol_manifest).expanduser()),
        "pretrained_vgg": str(Path(args.pretrained_vgg).expanduser()),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "lr_milestones": args.lr_milestones,
        "lr_gamma": args.lr_gamma,
        "alpha": args.alpha,
        "beta": args.beta,
        "max_train_samples": args.max_train_samples,
        "max_validation_samples": args.max_validation_samples,
        "selection_rule": "best independent selection-validation accuracy",
        "test_used_for_selection": False,
    }


def train_target(args):
    output_root = Path(args.output_root).expanduser()
    target_dir = output_root / "target"
    target_dir.mkdir(parents=True, exist_ok=True)
    latest_path = target_dir / "latest.pth"
    selected_path = target_dir / "best_validation.pth"
    selection_path = target_dir / "selection.json"
    history_path = target_dir / "history.json"
    config_path = target_dir / "config.json"

    config = training_config(args)
    if args.restart:
        for path in (
            latest_path,
            selected_path,
            selection_path,
            history_path,
            config_path,
            output_root / "target_summary.json",
        ):
            path.unlink(missing_ok=True)
    if config_path.is_file():
        previous = read_json(config_path)
        if previous != config:
            raise RuntimeError(
                f"Existing target configuration differs: {config_path}. "
                "Use --restart or another output root."
            )
    write_json(config_path, config)

    train_base = ProtocolDataset(
        args.protocol_manifest, args.data_root, "target_train", train=True
    )
    validation_base = ProtocolDataset(
        args.protocol_manifest,
        args.data_root,
        "selection_validation",
        train=False,
    )
    train_dataset = limited(train_base, args.max_train_samples)
    validation_dataset = limited(validation_base, args.max_validation_samples)

    model = build_initialized_model(args.pretrained_vgg, args.target_seed)
    initialization_hash = state_sha256(model.state_dict())
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
    )
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=args.lr_milestones, gamma=args.lr_gamma
    )
    history = []
    start_epoch = 1
    if latest_path.is_file() and not args.restart:
        checkpoint = load_checkpoint(latest_path, DEVICE)
        if checkpoint.get("initialization_sha256") != initialization_hash:
            raise RuntimeError(f"Initialization hash differs: {latest_path}")
        checkpoint_args = checkpoint.get("args", {})
        checkpoint_seed = checkpoint_args.get(
            "target_seed", checkpoint_args.get("seed")
        )
        if checkpoint_seed != args.target_seed:
            raise RuntimeError(f"Target seed differs: {latest_path}")
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        history = read_json(history_path) if history_path.is_file() else []
        print(f"[resume] target seed {args.target_seed} from epoch {start_epoch}")

    selection = read_json(selection_path) if selection_path.is_file() else None
    best_score = (
        float(selection["selection_score"]) if selection is not None else -math.inf
    )
    started = time.time()
    for epoch in range(start_epoch, args.epochs + 1):
        loader = make_loader(
            train_dataset,
            args.batch_size,
            args.num_workers,
            True,
            args.target_seed * 100000 + epoch,
        )
        model.train()
        totals = {"loss": 0.0, "ce": 0.0, "dependency": 0.0, "correct": 0, "n": 0}
        for images, labels in loader:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            hiddens, logits, _ = forward_full(model, images)
            cross_entropy = F.cross_entropy(logits, labels)
            if args.alpha == 0.0 and args.beta == 0.0:
                dependency = torch.zeros((), device=DEVICE)
            else:
                dependency = bido_dependency_loss(
                    images, hiddens, labels, args.alpha, args.beta
                )
            loss = cross_entropy + dependency
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            batch_size = images.size(0)
            totals["loss"] += float(loss.detach()) * batch_size
            totals["ce"] += float(cross_entropy.detach()) * batch_size
            totals["dependency"] += float(dependency.detach()) * batch_size
            totals["correct"] += int((logits.argmax(1) == labels).sum())
            totals["n"] += batch_size
        scheduler.step()

        validation = evaluate(model, validation_dataset, args)
        validation_accuracy = float(validation["accuracy"]["mean"])
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
        payload = checkpoint_payload(
            model,
            optimizer,
            scheduler,
            epoch,
            "bido",
            row,
            args,
            initialization_hash,
        )
        torch.save(payload, latest_path)
        if selection is None or validation_accuracy > best_score:
            best_score = validation_accuracy
            torch.save(payload, selected_path)
            selection = {
                "target_seed": args.target_seed,
                "epoch": epoch,
                "validation_accuracy": validation_accuracy,
                "selection_score": validation_accuracy,
                "selection_partition": "selection_validation",
                "test_used_for_selection": False,
                "initialization_sha256": initialization_hash,
                "checkpoint": str(selected_path),
            }
            write_json(selection_path, selection)
        print(
            f"seed={args.target_seed} epoch={epoch:03d}/{args.epochs} "
            f"train_acc={row['train_accuracy']:.4f} "
            f"val_acc={validation_accuracy:.4f} ce={row['train_ce']:.4f} "
            f"dep={row['train_dependency']:.5f}"
        )

    if not selected_path.is_file():
        raise RuntimeError(f"No selected target checkpoint: {selected_path}")
    print(f"[OK] target: {selected_path}")
    print(f"[TIME] {(time.time() - started) / 60:.1f} minutes")
    return selected_path


def evaluate_target(args, checkpoint_path):
    checkpoint_path = Path(checkpoint_path).expanduser()
    checkpoint = load_checkpoint(checkpoint_path, DEVICE)
    checkpoint_args = checkpoint.get("args", {})
    checkpoint_seed = checkpoint_args.get("target_seed", checkpoint_args.get("seed"))
    if checkpoint_seed is not None and int(checkpoint_seed) != args.target_seed:
        raise RuntimeError(
            f"Checkpoint seed={checkpoint_seed}, requested seed={args.target_seed}: "
            f"{checkpoint_path}"
        )
    model = FlexibleVGG(in_channels=3, num_classes=1000, feature_dim=2048).to(DEVICE)
    model.load_state_dict(checkpoint["state_dict"], strict=True)

    validation_base = ProtocolDataset(
        args.protocol_manifest,
        args.data_root,
        "selection_validation",
        train=False,
    )
    test_base = ProtocolDataset(
        args.protocol_manifest, args.data_root, "attack_test", train=False
    )
    validation_dataset = limited(validation_base, args.max_validation_samples)
    test_dataset = limited(test_base, args.max_test_samples)
    validation = evaluate(model, validation_dataset, args)
    sealed_test = evaluate(model, test_dataset, args)
    protocol = read_json(args.protocol_manifest)
    normalized_hyperparameters = {
        "epochs": checkpoint_args.get("bido_epochs", checkpoint_args.get("epochs")),
        "batch_size": checkpoint_args.get("batch_size"),
        "lr": checkpoint_args.get("lr"),
        "weight_decay": checkpoint_args.get("weight_decay"),
        "lr_milestones": checkpoint_args.get("lr_milestones"),
        "lr_gamma": checkpoint_args.get("lr_gamma"),
        "alpha": checkpoint_args.get("alpha"),
        "beta": checkpoint_args.get("beta"),
    }
    summary = {
        "experiment": "CelebA VGG BiDO target-model seed robustness",
        "target_seed": args.target_seed,
        "target_checkpoint": str(checkpoint_path),
        "selected_epoch": int(checkpoint["epoch"]),
        "initialization_sha256": checkpoint.get("initialization_sha256"),
        "protocol_manifest": str(Path(args.protocol_manifest).expanduser()),
        "protocol": protocol.get("protocol"),
        "partition_sha256": protocol.get("partition_sha256"),
        "training_hyperparameters": normalized_hyperparameters,
        "selection_rule": "best independent selection-validation accuracy",
        "test_used_for_selection": False,
        "sample_counts": {
            "target_train": min(
                len(ProtocolDataset(args.protocol_manifest, args.data_root, "target_train", train=False)),
                args.max_train_samples or 10**18,
            ),
            "selection_validation": len(validation_dataset),
            "attack_test": len(test_dataset),
        },
        "sample_sha256": {
            "selection_validation": partition_hash(
                validation_base, args.max_validation_samples
            ),
            "attack_test": partition_hash(test_base, args.max_test_samples),
        },
        "selection_validation": validation,
        "sealed_test": sealed_test,
    }
    output_path = Path(args.output_root).expanduser() / "target_summary.json"
    write_json(output_path, summary)
    print(
        f"target seed={args.target_seed} selected_epoch={summary['selected_epoch']} "
        f"validation={validation['accuracy']['mean']:.4f} "
        f"sealed_test={sealed_test['accuracy']['mean']:.4f}"
    )
    print(f"summary_json: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["train", "evaluate", "all"], default="all")
    parser.add_argument("--protocol_manifest", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--pretrained_vgg", default="")
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--target_checkpoint", default="")
    parser.add_argument("--target_seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--lr_milestones", type=int, nargs="+", default=[40])
    parser.add_argument("--lr_gamma", type=float, default=0.2)
    parser.add_argument("--alpha", type=float, default=0.001)
    parser.add_argument("--beta", type=float, default=0.005)
    parser.add_argument("--max_train_samples", type=int, default=0)
    parser.add_argument("--max_validation_samples", type=int, default=0)
    parser.add_argument("--max_test_samples", type=int, default=0)
    parser.add_argument("--restart", action="store_true")
    args = parser.parse_args()
    if args.stage in {"train", "all"} and not args.pretrained_vgg:
        parser.error("--pretrained_vgg is required for training")
    if args.batch_size < 3:
        parser.error("BiDO normalized HSIC requires --batch_size >= 3")
    if args.epochs <= 0:
        parser.error("--epochs must be positive")
    if args.max_train_samples and args.max_train_samples < args.batch_size:
        parser.error("--max_train_samples must be zero or at least --batch_size")
    for name in ("max_train_samples", "max_validation_samples", "max_test_samples"):
        if getattr(args, name) < 0:
            parser.error(f"--{name} must be non-negative")
    args.seed = args.target_seed
    args.bido_epochs = args.epochs
    args.control_epochs = 0
    return args


def main():
    args = parse_args()
    output_root = Path(args.output_root).expanduser()
    output_root.mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(args.target_checkpoint).expanduser() if args.target_checkpoint else None
    if args.stage in {"train", "all"}:
        checkpoint_path = train_target(args)
    if args.stage in {"evaluate", "all"}:
        if checkpoint_path is None:
            checkpoint_path = output_root / "target" / "best_validation.pth"
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Target checkpoint not found: {checkpoint_path}")
        evaluate_target(args, checkpoint_path)


if __name__ == "__main__":
    main()
