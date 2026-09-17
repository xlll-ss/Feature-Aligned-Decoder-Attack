"""Fashion-MNIST + ResNet18 cross-dataset generalization experiment for FAD.

The Fashion-MNIST training set is split deterministically into disjoint target
training (18k), target validation (2k), and decoder auxiliary (30k) subsets;
the remaining 10k images are recorded as unused. The official test set is
reserved for target testing and the final attack. MNIST supplies
out-of-distribution images for the BiDO+ OE term.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.utils import save_image

from attack_decoder_multidataset import (
    compute_psnr,
    compute_ssim,
    recover_feat_from_fc_grad,
    save_pair,
)
from celeba_resnet18_ablation import (
    ProjectedResNet18,
    bido_loss_exact,
)
from experiment_utils import seed_everything, summarize_values
from train_decoder_multidataset import DEVICE, FeatureDecoder, build_transform


DEFAULT_SEEDS = (2027, 2028, 2029)


class FashionProjectedResNet18(ProjectedResNet18):
    """The same projected ResNet18 leakage interface with a one-channel stem."""

    def __init__(self, num_classes=10, feature_dim=2048, pretrained=False):
        super().__init__(num_classes, feature_dim, pretrained=pretrained)
        old_conv = self.backbone.conv1
        self.backbone.conv1 = nn.Conv2d(
            1,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )
        if pretrained:
            with torch.no_grad():
                self.backbone.conv1.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))


def output_paths(args):
    root = Path(args.output_root).expanduser()
    return {
        "root": root,
        "split": root / f"fashionmnist_split_seed{args.split_seed}.json",
        "target": root / "target",
        "target_weight": root / "target" / "target_resnet18_bido_plus.pth",
        "decoder": root / "decoder",
        "attack": root / "attack",
        "summary": root / "generalization_summary.json",
        "summary_csv": root / "generalization_summary.csv",
    }


def classifier_input(images):
    return images


def fashionmnist_transform(args, augment=False):
    if not augment:
        return build_transform("fashionmnist", args.img_size, args.in_channels)
    return transforms.Compose(
        [
            transforms.Resize((args.img_size, args.img_size)),
            transforms.RandomCrop(args.img_size, padding=max(args.img_size // 16, 1)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5,) * args.in_channels, (0.5,) * args.in_channels),
        ]
    )


def load_fashionmnist(args, train, augment=False):
    return datasets.FashionMNIST(
        root=str(Path(args.data_root).expanduser()),
        train=train,
        transform=fashionmnist_transform(args, augment=augment),
        download=args.download,
    )


def split_payload(args):
    paths = output_paths(args)
    used_total = args.target_train_samples + args.target_val_samples + args.decoder_samples
    if used_total > 60000:
        raise ValueError(
            "Fashion-MNIST training partitions cannot exceed 60000 images"
        )
    if paths["split"].is_file():
        payload = json.loads(paths["split"].read_text(encoding="utf-8"))
        expected = {
            "dataset": "fashionmnist",
            "split_seed": args.split_seed,
            "target_train_samples": args.target_train_samples,
            "target_val_samples": args.target_val_samples,
            "decoder_samples": args.decoder_samples,
        }
        for key, value in expected.items():
            if payload.get(key) != value:
                raise RuntimeError(
                    f"Existing split mismatch for {key}: {payload.get(key)} != {value}"
                )
        return payload

    rng = np.random.default_rng(args.split_seed)
    indices = rng.permutation(60000).tolist()
    train_end = args.target_train_samples
    val_end = train_end + args.target_val_samples
    payload = {
        "version": 1,
        "dataset": "fashionmnist",
        "official_train_size": 60000,
        "official_test_size": 10000,
        "split_seed": args.split_seed,
        "target_train_samples": args.target_train_samples,
        "target_val_samples": args.target_val_samples,
        "decoder_samples": args.decoder_samples,
        "target_train_indices": indices[:train_end],
        "target_val_indices": indices[train_end:val_end],
        "decoder_indices": indices[val_end:used_total],
        "unused_indices": indices[used_total:],
        "unused_samples": 60000 - used_total,
        "permutation_sha256": hashlib.sha256(
            np.asarray(indices, dtype=np.int64).tobytes()
        ).hexdigest(),
    }
    paths["split"].parent.mkdir(parents=True, exist_ok=True)
    paths["split"].write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def make_subsets(args):
    full_train = load_fashionmnist(args, train=True)
    if len(full_train) != 60000:
        raise RuntimeError(
            f"Expected 60000 Fashion-MNIST training images, found {len(full_train)}"
        )
    payload = split_payload(args)
    return {
        "target_train": Subset(full_train, payload["target_train_indices"]),
        "target_val": Subset(full_train, payload["target_val_indices"]),
        "decoder": Subset(full_train, payload["decoder_indices"]),
    }


def limited_subset(dataset, limit):
    if limit and limit < len(dataset):
        return Subset(dataset, list(range(limit)))
    return dataset


def make_loader(dataset, args, shuffle, drop_last=False):
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def projected_id_loss(target, real, fake):
    with torch.no_grad():
        _, real_feature, _ = target(classifier_input(real))
        _, fake_feature, _ = target(classifier_input(fake))
    return F.mse_loss(real_feature.float(), fake_feature.float())


def load_checked_target(path, args):
    checkpoint = torch.load(path, map_location=DEVICE)
    if not isinstance(checkpoint, dict):
        raise RuntimeError("Target checkpoint has no experiment metadata")
    for key, expected in (
        ("dataset", "fashionmnist"),
        ("arch", "resnet18_projected"),
        ("img_size", args.img_size),
        ("in_channels", args.in_channels),
        ("num_classes", args.num_classes),
        ("feature_dim", args.feature_dim),
    ):
        if checkpoint.get(key) != expected:
            raise RuntimeError(
                f"Target checkpoint {key}={checkpoint.get(key)!r}, expected {expected!r}"
            )
    model = FashionProjectedResNet18(args.num_classes, args.feature_dim, pretrained=False).to(DEVICE)
    state = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def train_target(args):
    paths = output_paths(args)
    paths["target"].mkdir(parents=True, exist_ok=True)
    done = paths["target"] / "training_summary.json"
    if done.is_file() and paths["target_weight"].is_file() and not args.force:
        print(f"[skip] target already complete: {paths['target_weight']}")
        return

    seed_everything(args.target_seed)
    payload = split_payload(args)
    augmented_train = load_fashionmnist(
        args, train=True, augment=args.target_augmentation
    )
    evaluation_train = load_fashionmnist(args, train=True, augment=False)
    target_train = Subset(augmented_train, payload["target_train_indices"])
    target_val = Subset(evaluation_train, payload["target_val_indices"])
    target_train = limited_subset(target_train, args.max_target_train)
    target_val = limited_subset(target_val, args.max_target_val)
    train_loader = make_loader(target_train, args, shuffle=True, drop_last=True)
    val_loader = make_loader(target_val, args, shuffle=False)
    official_test = limited_subset(
        load_fashionmnist(args, train=False), args.max_target_test
    )
    test_loader = make_loader(official_test, args, shuffle=False)
    oe_loader = None
    if args.enable_oe:
        oe_transform = fashionmnist_transform(args, augment=args.target_augmentation)
        oe_dataset = datasets.MNIST(
            root=str(Path(args.oe_data_root or args.data_root).expanduser()),
            train=True,
            transform=oe_transform,
            download=args.download,
        )
        oe_dataset = limited_subset(oe_dataset, args.oe_samples)
        oe_dataset = limited_subset(oe_dataset, args.max_oe_samples)
        oe_loader = make_loader(oe_dataset, args, shuffle=True, drop_last=True)

    model = FashionProjectedResNet18(
        num_classes=args.num_classes,
        feature_dim=args.feature_dim,
        pretrained=args.imagenet_pretrained,
    ).to(DEVICE)
    optimizer = optim.AdamW(
        model.parameters(), lr=args.target_lr, weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.target_epochs, 1)
    )
    history = []
    best_acc = -1.0
    best_epoch = -1
    oe_iter = iter(oe_loader) if oe_loader is not None else None
    start = time.time()

    for epoch in range(1, args.target_epochs + 1):
        model.train()
        totals = {"loss": 0.0, "ce": 0.0, "bido": 0.0, "acc": 0.0, "n": 0}
        for images, labels in train_loader:
            images = classifier_input(images.to(DEVICE, non_blocking=True))
            labels = labels.to(DEVICE, non_blocking=True).long()
            logits, _, hiddens = model(images)
            ce = F.cross_entropy(logits, labels)
            dependency = bido_loss_exact(
                images, hiddens, labels, args.num_classes, args.alpha, args.beta
            )
            loss = ce + dependency

            if args.enable_oe and epoch > args.oe_warmup_epochs:
                try:
                    oe_images, _ = next(oe_iter)
                except StopIteration:
                    oe_iter = iter(oe_loader)
                    oe_images, _ = next(oe_iter)
                oe_images = classifier_input(oe_images.to(DEVICE, non_blocking=True))
                oe_logits, _, _ = model(oe_images)
                entropy = -(
                    F.softmax(oe_logits, dim=1) * F.log_softmax(oe_logits, dim=1)
                ).sum(dim=1).mean()
                loss = loss - args.oe_weight * entropy

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            n = labels.numel()
            totals["loss"] += loss.item() * n
            totals["ce"] += ce.item() * n
            totals["bido"] += dependency.item() * n
            totals["acc"] += (logits.argmax(1) == labels).float().sum().item()
            totals["n"] += n

        model.eval()
        val_correct = val_seen = 0
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images = classifier_input(images.to(DEVICE, non_blocking=True))
                labels = labels.to(DEVICE, non_blocking=True).long()
                logits, _, _ = model(images)
                val_loss += F.cross_entropy(logits, labels, reduction="sum").item()
                val_correct += (logits.argmax(1) == labels).sum().item()
                val_seen += labels.numel()
        val_acc = val_correct / max(val_seen, 1)
        row = {
            "epoch": epoch,
            "train_loss": totals["loss"] / max(totals["n"], 1),
            "train_ce": totals["ce"] / max(totals["n"], 1),
            "train_bido": totals["bido"] / max(totals["n"], 1),
            "train_acc": totals["acc"] / max(totals["n"], 1),
            "val_loss": val_loss / max(val_seen, 1),
            "val_acc": val_acc,
        }
        history.append(row)
        scheduler.step()
        print(
            f"Epoch {epoch:03d}/{args.target_epochs} | "
            f"train_acc={row['train_acc']:.4f} val_acc={val_acc:.4f} "
            f"loss={row['train_loss']:.4f} bido={row['train_bido']:.5f}"
        )
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "dataset": "fashionmnist",
                    "arch": "resnet18_projected",
                    "img_size": args.img_size,
                    "in_channels": args.in_channels,
                    "num_classes": args.num_classes,
                    "feature_dim": args.feature_dim,
                    "val_acc": val_acc,
                    "split_file": str(paths["split"]),
                    "args": vars(args),
                },
                paths["target_weight"],
            )
        (paths["target"] / "training_history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )

    best_model = load_checked_target(paths["target_weight"], args)
    test_correct = test_seen = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = classifier_input(images.to(DEVICE, non_blocking=True))
            labels = labels.to(DEVICE, non_blocking=True).long()
            logits, _, _ = best_model(images)
            test_correct += (logits.argmax(1) == labels).sum().item()
            test_seen += labels.numel()
    summary = {
        "status": "complete",
        "dataset": "fashionmnist",
        "arch": "resnet18_projected",
        "best_epoch": best_epoch,
        "best_val_acc": best_acc,
        "official_test_acc": test_correct / max(test_seen, 1),
        "official_test_samples": test_seen,
        "target_weight": str(paths["target_weight"]),
        "split_file": str(paths["split"]),
        "elapsed_minutes": (time.time() - start) / 60.0,
        "args": vars(args),
    }
    done.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved target: {paths['target_weight']}")


def train_decoder(args, seed):
    paths = output_paths(args)
    seed_dir = paths["decoder"] / f"seed{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    model_path = seed_dir / "decoder.pth"
    done = seed_dir / "training_summary.json"
    if done.is_file() and model_path.is_file() and not args.force:
        print(f"[skip] decoder seed {seed} already complete")
        return

    seed_everything(seed)
    decoder_dataset = make_subsets(args)["decoder"]
    decoder_dataset = limited_subset(decoder_dataset, args.max_decoder_samples)
    loader = make_loader(decoder_dataset, args, shuffle=True, drop_last=True)
    target = load_checked_target(paths["target_weight"], args)
    for parameter in target.parameters():
        parameter.requires_grad = False
    decoder = FeatureDecoder(args.feature_dim, args.img_size, args.in_channels).to(DEVICE)
    optimizer = optim.AdamW(
        decoder.parameters(), lr=args.decoder_lr, betas=(0.5, 0.999), weight_decay=1e-4
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.decoder_epochs, 1)
    )
    history = []
    start = time.time()

    for epoch in range(1, args.decoder_epochs + 1):
        decoder.train()
        total_loss = total_l1 = total_mse = 0.0
        for real, _ in loader:
            real = real.to(DEVICE, non_blocking=True)
            with torch.no_grad():
                _, feature, _ = target(classifier_input(real))
            recon = decoder(feature.detach())
            l1 = F.l1_loss(recon, real)
            mse = F.mse_loss(recon, real)
            loss = args.l1_weight * l1 + args.mse_weight * mse
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item()
            total_l1 += l1.item()
            total_mse += mse.item()
        scheduler.step()
        steps = max(len(loader), 1)
        row = {
            "epoch": epoch,
            "loss": total_loss / steps,
            "l1": total_l1 / steps,
            "mse": total_mse / steps,
        }
        history.append(row)
        print(
            f"Decoder seed {seed} epoch {epoch:03d}/{args.decoder_epochs} | "
            f"l1={row['l1']:.5f}"
        )
        if epoch == 1 or epoch % args.save_every == 0 or epoch == args.decoder_epochs:
            decoder.eval()
            with torch.no_grad():
                real_vis, _ = next(iter(loader))
                real_vis = real_vis.to(DEVICE)
                _, feature_vis, _ = target(classifier_input(real_vis))
                recon_vis = decoder(feature_vis)
            save_image(
                (torch.cat([real_vis[:8], recon_vis[:8]], dim=0) + 1) / 2,
                seed_dir / f"epoch_{epoch:03d}.png",
                nrow=8,
            )
            torch.save(
                {
                    "state_dict": decoder.state_dict(),
                    "dataset": "fashionmnist",
                    "arch": "resnet18_projected",
                    "feature_dim": args.feature_dim,
                    "img_size": args.img_size,
                    "in_channels": args.in_channels,
                    "num_classes": args.num_classes,
                    "seed": seed,
                    "split_file": str(paths["split"]),
                    "args": vars(args),
                },
                model_path,
            )
        (seed_dir / "training_history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )

    done.write_text(
        json.dumps(
            {
                "status": "complete",
                "dataset": "fashionmnist",
                "arch": "resnet18_projected",
                "seed": seed,
                "decoder_path": str(model_path),
                "split_file": str(paths["split"]),
                "elapsed_minutes": (time.time() - start) / 60.0,
                "args": vars(args),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def load_decoder(path, args):
    checkpoint = torch.load(path, map_location=DEVICE)
    if isinstance(checkpoint, dict):
        for key, expected in (
            ("dataset", "fashionmnist"),
            ("arch", "resnet18_projected"),
            ("feature_dim", args.feature_dim),
            ("img_size", args.img_size),
            ("in_channels", args.in_channels),
            ("num_classes", args.num_classes),
        ):
            if checkpoint.get(key) != expected:
                raise RuntimeError(
                    f"Decoder checkpoint {key}={checkpoint.get(key)!r}, expected {expected!r}"
                )
    decoder = FeatureDecoder(args.feature_dim, args.img_size, args.in_channels).to(DEVICE)
    state = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    decoder.load_state_dict(state, strict=True)
    decoder.eval()
    for parameter in decoder.parameters():
        parameter.requires_grad = False
    return decoder


def attack(args, seed):
    paths = output_paths(args)
    seed_dir = paths["attack"] / f"seed{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    summary_path = seed_dir / "summary.json"
    if summary_path.is_file() and not args.force:
        print(f"[skip] attack seed {seed} already complete")
        return

    seed_everything(seed)
    test_dataset = load_fashionmnist(args, train=False)
    scan_limit = min(args.attack_scan, len(test_dataset))
    test_dataset = Subset(test_dataset, list(range(scan_limit)))
    loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)
    target = load_checked_target(paths["target_weight"], args)
    for parameter in target.parameters():
        parameter.requires_grad = True
    decoder = load_decoder(paths["decoder"] / f"seed{seed}" / "decoder.pth", args)
    values = {
        key: []
        for key in (
            "psnr",
            "ssim",
            "id_loss",
            "leakage_feature_mse",
            "relative_feature_error",
            "online_decoder_ms",
        )
    }
    records = []
    skipped = []
    start = time.time()

    for index, (image, label) in enumerate(loader):
        if len(records) >= args.test_samples:
            break
        image = image.to(DEVICE)
        label = label.to(DEVICE).long()
        with torch.no_grad():
            eval_logits, true_feature, _ = target(classifier_input(image))
            predicted = eval_logits.argmax(1)
        grad_logits, _, _ = target(classifier_input(image))
        loss = F.cross_entropy(grad_logits, label)
        grad_w, grad_b = torch.autograd.grad(
            loss, (target.fc_layer.weight, target.fc_layer.bias)
        )
        try:
            leaked, recovered_class, denominator = recover_feat_from_fc_grad(
                grad_w.detach(), grad_b.detach(), label
            )
        except RuntimeError as exc:
            skipped.append({"idx": index, "reason": str(exc)})
            continue
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        online_start = time.perf_counter()
        with torch.no_grad():
            fake = decoder(leaked.to(DEVICE))
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        online_ms = (time.perf_counter() - online_start) * 1000.0
        relative_error = (
            torch.linalg.vector_norm(leaked - true_feature)
            / torch.clamp(torch.linalg.vector_norm(true_feature), min=1e-12)
        ).item()
        leakage_feature_mse = F.mse_loss(leaked, true_feature).item()
        metrics = {
            "idx": index,
            "psnr": float(compute_psnr(fake, image).item()),
            "ssim": float(compute_ssim(fake, image)),
            "id_loss": float(projected_id_loss(target, image, fake).item()),
            "leakage_feature_mse": float(leakage_feature_mse),
            "relative_feature_error": float(relative_error),
            "target_label": int(label.item()),
            "pred_label": int(predicted.item()),
            "recovered_class": int(recovered_class),
            "recovered_denom": denominator,
            "classification_correct": bool(predicted.item() == label.item()),
            "online_decoder_ms": online_ms,
        }
        records.append(metrics)
        for key in values:
            values[key].append(metrics[key])
        if len(records) <= args.save_pairs:
            save_pair(image, fake, seed_dir, len(records) - 1, metrics)
        if args.save_raw_pairs:
            raw_dir = seed_dir / "raw_pairs"
            raw_dir.mkdir(parents=True, exist_ok=True)
            save_image(torch.clamp((image.cpu() + 1) / 2, 0, 1), raw_dir / f"{index:04d}_real.png")
            save_image(torch.clamp((fake.cpu() + 1) / 2, 0, 1), raw_dir / f"{index:04d}_recon.png")
        print(
            f"Attack seed {seed} sample {len(records):03d}/{args.test_samples} | "
            f"PSNR {metrics['psnr']:.2f} | SSIM {metrics['ssim']:.4f}"
        )

    if not records:
        raise RuntimeError("No valid samples were recovered")
    summary = {
        "dataset": "fashionmnist",
        "arch": "resnet18_projected",
        "method": "feature-aligned decoder",
        "seed": seed,
        "test_samples": len(records),
        "requested_test_samples": args.test_samples,
        "scanned_samples": len(records) + len(skipped),
        "saturated_samples": len(skipped),
        "attack_steps": 0,
        "n_restarts": 0,
        "label_source": "dataset",
        "target_accuracy": float(np.mean([r["classification_correct"] for r in records])),
        "target_weight": str(paths["target_weight"]),
        "decoder_path": str(paths["decoder"] / f"seed{seed}" / "decoder.pth"),
        "split_file": str(paths["split"]),
        "official_test_indices": [r["idx"] for r in records],
        "args": vars(args),
        "per_sample": records,
        "skipped_saturated": skipped,
        "elapsed_minutes": (time.time() - start) / 60.0,
    }
    for key, metric_values in values.items():
        summary[key] = summarize_values(metric_values, seed=seed)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved attack summary: {summary_path}")


def evaluate_lpips(args, seed):
    paths = output_paths(args)
    seed_dir = paths["attack"] / f"seed{seed}"
    output = seed_dir / "lpips.json"
    if output.is_file() and not args.force:
        print(f"[skip] LPIPS seed {seed} already complete")
        return
    try:
        import lpips
    except ModuleNotFoundError as exc:
        raise RuntimeError("Install LPIPS with: pip install lpips") from exc
    real_paths = sorted((seed_dir / "raw_pairs").glob("*_real.png"))
    if not real_paths:
        raise RuntimeError(f"No raw pairs found under {seed_dir / 'raw_pairs'}")
    metric = lpips.LPIPS(net="alex").to(DEVICE).eval()
    to_tensor = transforms.ToTensor()
    values = []
    with torch.no_grad():
        for real_path in real_paths:
            recon_path = real_path.with_name(
                real_path.name.replace("_real.png", "_recon.png")
            )
            real = to_tensor(Image.open(real_path).convert("RGB")).mul(2).sub(1).unsqueeze(0).to(DEVICE)
            recon = to_tensor(Image.open(recon_path).convert("RGB")).mul(2).sub(1).unsqueeze(0).to(DEVICE)
            values.append(float(metric(real, recon).mean().item()))
    output.write_text(
        json.dumps(
            {
                "metric": "LPIPS",
                "net": "alex",
                "evaluated_samples": len(values),
                "lpips": summarize_values(values, seed=seed),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved LPIPS: {output}")


def summarize(args):
    paths = output_paths(args)
    target_summary_path = paths["target"] / "training_summary.json"
    if not target_summary_path.is_file():
        raise FileNotFoundError(f"Missing target summary: {target_summary_path}")
    target_summary = json.loads(target_summary_path.read_text(encoding="utf-8"))
    runs = []
    for seed in args.seeds:
        summary_path = paths["attack"] / f"seed{seed}" / "summary.json"
        if not summary_path.is_file():
            raise FileNotFoundError(f"Missing attack summary: {summary_path}")
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        if payload.get("test_samples") != args.test_samples:
            raise RuntimeError(
                f"{summary_path} has {payload.get('test_samples')} samples, expected {args.test_samples}"
            )
        if payload.get("dataset") != "fashionmnist" or payload.get("arch") != "resnet18_projected":
            raise RuntimeError(f"Unexpected experiment metadata in {summary_path}")
        if payload.get("seed") != seed:
            raise RuntimeError(f"Seed mismatch in {summary_path}")
        runs.append(payload)
    reference = runs[0]["official_test_indices"]
    for payload in runs[1:]:
        if payload["official_test_indices"] != reference:
            raise RuntimeError("Decoder seeds were evaluated on different test images")

    metrics = [
        "psnr",
        "ssim",
        "id_loss",
        "leakage_feature_mse",
        "relative_feature_error",
        "online_decoder_ms",
    ]
    per_seed = []
    for payload in runs:
        row = {"seed": payload["seed"], "target_accuracy": payload["target_accuracy"]}
        for metric in metrics:
            row[f"{metric}_mean"] = payload[metric]["mean"]
            row[f"{metric}_sample_std"] = payload[metric]["std"]
        lpips_path = paths["attack"] / f"seed{payload['seed']}" / "lpips.json"
        if lpips_path.is_file():
            lpips_payload = json.loads(lpips_path.read_text(encoding="utf-8"))
            if lpips_payload.get("evaluated_samples") != args.test_samples:
                raise RuntimeError(f"LPIPS sample count mismatch in {lpips_path}")
            row["lpips_mean"] = lpips_payload["lpips"]["mean"]
        per_seed.append(row)

    if args.require_lpips and not all("lpips_mean" in row for row in per_seed):
        raise RuntimeError("LPIPS is required, but one or more decoder seeds have no LPIPS result")
    if any("lpips_mean" in row for row in per_seed) and not all(
        "lpips_mean" in row for row in per_seed
    ):
        raise RuntimeError("LPIPS is present for only some decoder seeds")
    attack_subset_target_accuracy = float(
        np.mean([row["target_accuracy"] for row in per_seed])
    )
    aggregate = {
        "dataset": "fashionmnist",
        "arch": "resnet18_projected",
        "decoder_seeds": list(args.seeds),
        "test_samples": args.test_samples,
        "target_accuracy": attack_subset_target_accuracy,
        "attack_subset_target_accuracy": attack_subset_target_accuracy,
        "official_test_target_accuracy": target_summary["official_test_acc"],
        "split_file": str(paths["split"]),
        "metrics": {},
        "per_seed": per_seed,
    }
    aggregate_metrics = metrics + (["lpips"] if "lpips_mean" in per_seed[0] else [])
    for metric in aggregate_metrics:
        metric_values = [row[f"{metric}_mean"] for row in per_seed]
        aggregate["metrics"][metric] = {
            "mean_across_decoder_seeds": float(np.mean(metric_values)),
            "std_across_decoder_seeds": (
                float(np.std(metric_values, ddof=1)) if len(metric_values) > 1 else 0.0
            ),
        }
    paths["summary"].write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    with paths["summary_csv"].open("w", newline="", encoding="utf-8") as handle:
        fieldnames = sorted({key for row in per_seed for key in row})
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_seed)
    print(f"Saved summary: {paths['summary']}")
    print(f"Saved CSV: {paths['summary_csv']}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=["target", "decoder", "attack", "lpips", "summarize", "all"],
        required=True,
    )
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--oe_data_root", default="")
    parser.add_argument("--output_root", default="fashionmnist_resnet18_generalization")
    parser.add_argument("--download", dest="download", action="store_true", default=True)
    parser.add_argument("--no_download", dest="download", action="store_false")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--decoder_seed", type=int, default=2027)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--split_seed", type=int, default=2027)
    parser.add_argument("--target_seed", type=int, default=2027)
    parser.add_argument("--target_train_samples", type=int, default=18000)
    parser.add_argument("--target_val_samples", type=int, default=2000)
    parser.add_argument("--decoder_samples", type=int, default=30000)
    parser.add_argument("--max_target_train", type=int, default=0)
    parser.add_argument("--max_target_val", type=int, default=0)
    parser.add_argument("--max_target_test", type=int, default=0)
    parser.add_argument("--max_decoder_samples", type=int, default=0)
    parser.add_argument("--oe_samples", type=int, default=30000)
    parser.add_argument("--max_oe_samples", type=int, default=0)
    parser.add_argument("--target_epochs", type=int, default=80)
    parser.add_argument("--decoder_epochs", type=int, default=120)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--img_size", type=int, default=64)
    parser.add_argument("--in_channels", type=int, default=1)
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--feature_dim", type=int, default=2048)
    parser.add_argument("--target_lr", type=float, default=2e-4)
    parser.add_argument("--decoder_lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--alpha", type=float, default=0.001)
    parser.add_argument("--beta", type=float, default=0.005)
    parser.add_argument("--enable_oe", dest="enable_oe", action="store_true", default=True)
    parser.add_argument("--disable_oe", dest="enable_oe", action="store_false")
    parser.add_argument("--oe_weight", type=float, default=1e-3)
    parser.add_argument("--oe_warmup_epochs", type=int, default=60)
    parser.add_argument("--l1_weight", type=float, default=10.0)
    parser.add_argument("--mse_weight", type=float, default=0.0)
    parser.add_argument("--test_samples", type=int, default=500)
    parser.add_argument("--attack_scan", type=int, default=1000)
    parser.add_argument("--save_pairs", type=int, default=40)
    parser.add_argument("--save_raw_pairs", dest="save_raw_pairs", action="store_true", default=True)
    parser.add_argument("--no_save_raw_pairs", dest="save_raw_pairs", action="store_false")
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--imagenet_pretrained", action="store_true")
    parser.add_argument(
        "--target_augmentation", dest="target_augmentation", action="store_true", default=True
    )
    parser.add_argument(
        "--no_target_augmentation", dest="target_augmentation", action="store_false"
    )
    parser.add_argument("--with_lpips", action="store_true")
    parser.add_argument("--require_lpips", action="store_true")
    return parser.parse_args()


def validate_args(args):
    used_total = args.target_train_samples + args.target_val_samples + args.decoder_samples
    if used_total > 60000:
        raise ValueError("The three Fashion-MNIST training partitions cannot exceed 60000")
    if min(args.target_train_samples, args.target_val_samples, args.decoder_samples) <= 0:
        raise ValueError("All three Fashion-MNIST training partitions must be non-empty")
    if args.in_channels != 1 or args.num_classes != 10:
        raise ValueError("Fashion-MNIST requires --in_channels 1 and --num_classes 10")
    if args.target_epochs < 1 or args.decoder_epochs < 1:
        raise ValueError("Training epochs must be positive")
    if args.batch_size < 2:
        raise ValueError("--batch_size must be at least 2 because the models use BatchNorm")
    if args.target_train_samples < args.batch_size or args.decoder_samples < args.batch_size:
        raise ValueError("Training partitions must contain at least one full batch")
    if args.enable_oe and 0 < args.oe_samples < args.batch_size:
        raise ValueError("--oe_samples must be 0 or at least --batch_size")
    for name in (
        "max_target_train",
        "max_target_val",
        "max_target_test",
        "max_decoder_samples",
        "max_oe_samples",
    ):
        value = getattr(args, name)
        if value < 0:
            raise ValueError(f"--{name} cannot be negative")
        if value and value < args.batch_size and name in {
            "max_target_train",
            "max_decoder_samples",
            "max_oe_samples",
        }:
            raise ValueError(f"--{name} must be 0 or at least --batch_size")
    if args.test_samples < 1 or args.attack_scan < args.test_samples:
        raise ValueError("--attack_scan must be at least --test_samples")
    if args.save_pairs < 0 or args.save_pairs > args.test_samples:
        raise ValueError("--save_pairs must be between 0 and --test_samples")


def main():
    args = parse_args()
    validate_args(args)
    paths = output_paths(args)
    paths["root"].mkdir(parents=True, exist_ok=True)
    if args.stage in {"target", "all"}:
        train_target(args)
    if args.stage in {"decoder", "all"}:
        seeds = args.seeds if args.stage == "all" else [args.decoder_seed]
        for seed in seeds:
            train_decoder(args, seed)
    if args.stage in {"attack", "all"}:
        seeds = args.seeds if args.stage == "all" else [args.decoder_seed]
        for seed in seeds:
            attack(args, seed)
            if args.with_lpips:
                evaluate_lpips(args, seed)
    if args.stage == "lpips":
        for seed in args.seeds:
            evaluate_lpips(args, seed)
    if args.stage in {"summarize", "all"}:
        summarize(args)


if __name__ == "__main__":
    main()
