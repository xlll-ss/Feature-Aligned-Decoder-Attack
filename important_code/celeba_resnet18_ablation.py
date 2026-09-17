"""CelebA + ResNet18 architecture-generalization experiment for FAD.

This file is intentionally standalone. It reuses the existing FAD modules but
does not modify the original training, decoder, or attack scripts.

Stages:
  target     train a BiDO+-style ResNet18 target on the manifest validation split
  decoder    train one feature decoder for a requested seed
  attack     run the exact single-sample final-layer gradient attack
  lpips      evaluate saved raw pairs with LPIPS-Alex
  summarize  aggregate decoder seeds into one JSON and CSV table
  all        run target, three decoder seeds, attacks, LPIPS, and summary
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models
from torchvision import transforms
from PIL import Image

from experiment_utils import seed_everything, summarize_values
from train_decoder_multidataset import (
    DEVICE,
    FeatureDecoder,
    build_dataset,
    build_transform,
    target_model_input,
)
from attack_decoder_multidataset import (
    compute_psnr,
    compute_ssim,
    recover_feat_from_fc_grad,
    save_pair,
)


DEFAULT_SEEDS = (2027, 2028, 2029)


class ProjectedResNet18(nn.Module):
    """Torchvision ResNet18 with a fixed 2048-D leakage feature interface."""

    def __init__(self, num_classes=1000, feature_dim=2048, pretrained=False):
        super().__init__()
        if pretrained:
            try:
                backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            except AttributeError:
                backbone = models.resnet18(pretrained=True)
        else:
            try:
                backbone = models.resnet18(weights=None)
            except TypeError:
                backbone = models.resnet18(pretrained=False)
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.feat_proj = nn.Linear(512, feature_dim)
        self.bn = nn.BatchNorm1d(feature_dim)
        self.fc_layer = nn.Linear(feature_dim, num_classes)
        self.num_classes = num_classes
        self.feature_dim = feature_dim

    def forward_features(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        h1 = x
        x = self.backbone.layer1(x)
        h2 = x
        x = self.backbone.layer2(x)
        h3 = x
        x = self.backbone.layer3(x)
        h4 = x
        x = self.backbone.layer4(x)
        h5 = x
        x = self.backbone.avgpool(x).flatten(1)
        feat = self.bn(self.feat_proj(x))
        return feat, [h1, h2, h3, h4, h5, feat]

    def forward(self, x):
        feat, hiddens = self.forward_features(x)
        return self.fc_layer(feat), feat, hiddens


def load_target(path, num_classes, feature_dim, pretrained=False):
    model = ProjectedResNet18(num_classes, feature_dim, pretrained=pretrained).to(DEVICE)
    checkpoint = torch.load(path, map_location=DEVICE)
    state = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def hsic_normalized_cca(x, y, eps=1e-5):
    """Device-safe equivalent of the repository's normalized linear HSIC."""
    x = x.flatten(1).float()
    y = y.flatten(1).float()
    n = x.size(0)
    h = torch.eye(n, device=x.device) - torch.ones((n, n), device=x.device) / n
    kx = (x @ x.t()) @ h
    ky = (y @ y.t()) @ h
    eye = torch.eye(n, device=x.device)
    rx = kx @ torch.linalg.solve(kx + eps * n * eye, eye)
    ry = ky @ torch.linalg.solve(ky + eps * n * eye, eye)
    return (rx * ry.t()).sum()


def bido_loss_exact(inputs, hiddens, labels, num_classes, alpha, beta):
    target = F.one_hot(labels, num_classes=num_classes).float()
    total = torch.zeros((), device=inputs.device)
    for hidden in hiddens:
        total = total + alpha * hsic_normalized_cca(hidden, inputs) - beta * hsic_normalized_cca(hidden, target)
    return total


def projected_id_loss(target, real, fake):
    with torch.no_grad():
        _, real_feature, _ = target(target_model_input(real, "celeba"))
        _, fake_feature, _ = target(target_model_input(fake, "celeba"))
    return F.mse_loss(real_feature.float(), fake_feature.float())


def dataset_for(args, partition, max_samples):
    transform = build_transform("celeba", args.img_size, args.in_channels)
    dataset = build_dataset(
        "celeba",
        args.data_root,
        "test" if partition == "test" else "train",
        transform,
        max_samples,
        False,
        manifest_path=args.manifest_path,
        manifest_partition=partition,
    )
    if hasattr(dataset, "has_complete_labels") and not dataset.has_complete_labels:
        raise RuntimeError(f"Manifest partition {partition} has missing labels")
    return dataset


def output_paths(args):
    root = Path(args.output_root).expanduser()
    return {
        "root": root,
        "target": root / "target",
        "target_weight": root / "target" / "target_resnet18_bido_plus.pth",
        "decoder": root / "decoder",
        "attack": root / "attack",
        "summary": root / "generalization_summary.json",
        "summary_csv": root / "generalization_summary.csv",
    }


def train_target(args):
    paths = output_paths(args)
    paths["target"].mkdir(parents=True, exist_ok=True)
    done = paths["target"] / "training_summary.json"
    if done.is_file() and paths["target_weight"].is_file() and not args.force:
        print(f"[skip] target already complete: {paths['target_weight']}")
        return

    seed_everything(args.target_seed)
    train_ds = dataset_for(args, "validation", args.max_target_train)
    test_ds = dataset_for(args, "test", args.max_target_test)
    aux_ds = dataset_for(args, "auxiliary", args.max_aux_samples)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              drop_last=True, num_workers=args.num_workers,
                              pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers,
                             pin_memory=torch.cuda.is_available())
    aux_loader = DataLoader(aux_ds, batch_size=args.batch_size, shuffle=True,
                            drop_last=True, num_workers=args.num_workers,
                            pin_memory=torch.cuda.is_available())

    model = ProjectedResNet18(args.num_classes, args.feature_dim, args.imagenet_pretrained).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=args.target_lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.target_epochs, 1))
    best_acc = -1.0
    best_epoch = -1
    history = []
    aux_iter = iter(aux_loader)
    start = time.time()

    for epoch in range(1, args.target_epochs + 1):
        model.train()
        total_loss = total_ce = total_dep = total_acc = 0.0
        total_n = 0
        for images, labels in train_loader:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True).long()
            model_images = target_model_input(images, "celeba")
            logits, _, hiddens = model(model_images)
            ce = F.cross_entropy(logits, labels)
            dep = bido_loss_exact(model_images, hiddens, labels, args.num_classes, args.alpha, args.beta)
            loss = ce + dep

            if args.enable_oe and epoch > args.oe_warmup_epochs:
                try:
                    aux_images, _ = next(aux_iter)
                except StopIteration:
                    aux_iter = iter(aux_loader)
                    aux_images, _ = next(aux_iter)
                aux_images = target_model_input(aux_images.to(DEVICE, non_blocking=True), "celeba")
                aux_logits, _, _ = model(aux_images)
                entropy_sum = -(F.softmax(aux_logits, dim=1) * F.log_softmax(aux_logits, dim=1)).sum()
                loss = loss - args.oe_weight * entropy_sum

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            n = images.size(0)
            total_loss += loss.item() * n
            total_ce += ce.item() * n
            total_dep += dep.item() * n
            total_acc += (logits.argmax(1) == labels).float().sum().item()
            total_n += n

        model.eval()
        correct = seen = val_loss = 0.0
        with torch.no_grad():
            for images, labels in test_loader:
                images = target_model_input(images.to(DEVICE, non_blocking=True), "celeba")
                labels = labels.to(DEVICE, non_blocking=True).long()
                logits, _, _ = model(images)
                val_loss += F.cross_entropy(logits, labels, reduction="sum").item()
                correct += (logits.argmax(1) == labels).float().sum().item()
                seen += labels.numel()
        val_acc = correct / max(seen, 1)
        row = {
            "epoch": epoch,
            "train_loss": total_loss / max(total_n, 1),
            "train_ce": total_ce / max(total_n, 1),
            "train_bido": total_dep / max(total_n, 1),
            "train_acc": total_acc / max(total_n, 1),
            "val_loss": val_loss / max(seen, 1),
            "val_acc": val_acc,
        }
        history.append(row)
        scheduler.step()
        print(
            f"Epoch {epoch:03d}/{args.target_epochs} | train_acc={row['train_acc']:.4f} "
            f"val_acc={val_acc:.4f} loss={row['train_loss']:.4f} bido={row['train_bido']:.5f}"
        )
        if val_acc > best_acc or epoch == args.target_epochs:
            best_acc = val_acc
            best_epoch = epoch
            torch.save({
                "state_dict": model.state_dict(),
                "dataset": "celeba",
                "arch": "resnet18_projected",
                "img_size": args.img_size,
                "in_channels": args.in_channels,
                "num_classes": args.num_classes,
                "feature_dim": args.feature_dim,
                "val_acc": val_acc,
                "args": vars(args),
            }, paths["target_weight"])

    summary = {
        "status": "complete",
        "dataset": "celeba",
        "arch": "resnet18_projected",
        "best_epoch": best_epoch,
        "best_val_acc": best_acc,
        "target_weight": str(paths["target_weight"]),
        "elapsed_minutes": (time.time() - start) / 60.0,
        "args": vars(args),
        "history": history,
    }
    (paths["target"] / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
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
        return model_path

    seed_everything(seed)
    aux_ds = dataset_for(args, "auxiliary", args.decoder_samples)
    loader = DataLoader(aux_ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
                        num_workers=args.num_workers, pin_memory=torch.cuda.is_available())
    target = load_target(paths["target_weight"], args.num_classes, args.feature_dim, pretrained=False)
    for parameter in target.parameters():
        parameter.requires_grad = False
    decoder = FeatureDecoder(args.feature_dim, args.img_size, args.in_channels).to(DEVICE)
    optimizer = optim.AdamW(decoder.parameters(), lr=args.decoder_lr, betas=(0.5, 0.999), weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.decoder_epochs, 1))
    history = []
    start = time.time()

    for epoch in range(1, args.decoder_epochs + 1):
        decoder.train()
        totals = {"loss": 0.0, "l1": 0.0, "mse": 0.0}
        for real, _ in loader:
            real = real.to(DEVICE, non_blocking=True)
            with torch.no_grad():
                _, feat, _ = target(target_model_input(real, "celeba"))
            recon = decoder(feat.detach())
            l1 = F.l1_loss(recon, real)
            mse = F.mse_loss(recon, real)
            loss = args.l1_weight * l1 + args.mse_weight * mse
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), 5.0)
            optimizer.step()
            totals["loss"] += loss.item()
            totals["l1"] += l1.item()
            totals["mse"] += mse.item()
        scheduler.step()
        steps = max(len(loader), 1)
        row = {"epoch": epoch, **{key: value / steps for key, value in totals.items()}}
        history.append(row)
        print(f"Decoder seed {seed} epoch {epoch:03d}/{args.decoder_epochs} | l1={row['l1']:.5f}")
        if epoch == 1 or epoch % args.save_every == 0 or epoch == args.decoder_epochs:
            decoder.eval()
            with torch.no_grad():
                real_vis, _ = next(iter(loader))
                real_vis = real_vis.to(DEVICE)
                _, feat_vis, _ = target(target_model_input(real_vis, "celeba"))
                recon_vis = decoder(feat_vis)
            from torchvision.utils import save_image
            save_image((torch.cat([real_vis[:8], recon_vis[:8]], dim=0) + 1) / 2,
                       seed_dir / f"epoch_{epoch:03d}.png", nrow=8)
            torch.save({
                "state_dict": decoder.state_dict(),
                "dataset": "celeba",
                "arch": "resnet18_projected",
                "feature_dim": args.feature_dim,
                "img_size": args.img_size,
                "in_channels": args.in_channels,
                "num_classes": args.num_classes,
                "seed": seed,
                "args": vars(args),
            }, model_path)

    summary = {
        "status": "complete",
        "dataset": "celeba",
        "arch": "resnet18_projected",
        "seed": seed,
        "decoder_path": str(model_path),
        "elapsed_minutes": (time.time() - start) / 60.0,
        "args": vars(args),
        "history": history,
    }
    (seed_dir / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    done.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return model_path


def load_decoder(path, args):
    checkpoint = torch.load(path, map_location=DEVICE)
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
    dataset = dataset_for(args, "test", args.attack_scan)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)
    target = load_target(paths["target_weight"], args.num_classes, args.feature_dim, pretrained=False)
    for parameter in target.parameters():
        parameter.requires_grad = True
    decoder = load_decoder(paths["decoder"] / f"seed{seed}" / "decoder.pth", args)
    psnr_values, ssim_values, id_values, feat_values, times = [], [], [], [], []
    records, skipped = [], []
    start = time.time()

    for index, (image, label) in enumerate(loader):
        if len(records) >= args.test_samples:
            break
        image = image.to(DEVICE)
        label = label.to(DEVICE).long()
        model_image = target_model_input(image, "celeba")
        with torch.no_grad():
            eval_logits, true_feature, _ = target(model_image)
            predicted = eval_logits.argmax(1)
        grad_logits, _, _ = target(model_image)
        loss = F.cross_entropy(grad_logits, label)
        grad_w, grad_b = torch.autograd.grad(loss, (target.fc_layer.weight, target.fc_layer.bias))
        try:
            leaked, recovered_class, denominator = recover_feat_from_fc_grad(grad_w.detach(), grad_b.detach(), label)
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
        relative_error = (torch.linalg.vector_norm(leaked - true_feature) /
                          torch.clamp(torch.linalg.vector_norm(true_feature), min=1e-12)).item()
        metrics = {
            "idx": index,
            "psnr": float(compute_psnr(fake, image).item()),
            "ssim": float(compute_ssim(fake, image)),
            "id_loss": float(projected_id_loss(target, image, fake).item()),
            "relative_feature_error": float(relative_error),
            "target_label": int(label.item()),
            "pred_label": int(predicted.item()),
            "recovered_class": int(recovered_class),
            "recovered_denom": denominator,
            "classification_correct": bool(predicted.item() == label.item()),
            "online_decoder_ms": online_ms,
        }
        records.append(metrics)
        psnr_values.append(metrics["psnr"])
        ssim_values.append(metrics["ssim"])
        id_values.append(metrics["id_loss"])
        feat_values.append(metrics["relative_feature_error"])
        times.append(online_ms)
        if len(records) <= args.save_pairs:
            save_pair(image, fake, seed_dir, len(records) - 1, metrics)
        if args.save_raw_pairs:
            raw_dir = seed_dir / "raw_pairs"
            raw_dir.mkdir(parents=True, exist_ok=True)
            from torchvision.utils import save_image
            save_image(torch.clamp((image.cpu() + 1) / 2, 0, 1), raw_dir / f"{index:04d}_real.png")
            save_image(torch.clamp((fake.cpu() + 1) / 2, 0, 1), raw_dir / f"{index:04d}_recon.png")
        print(f"Attack seed {seed} sample {len(records):03d}/{args.test_samples} | PSNR {metrics['psnr']:.2f} | SSIM {metrics['ssim']:.4f}")

    if not records:
        raise RuntimeError("No valid samples were recovered")
    summary = {
        "dataset": "celeba",
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
        "psnr": summarize_values(psnr_values, seed=seed),
        "ssim": summarize_values(ssim_values, seed=seed),
        "id_loss": summarize_values(id_values, seed=seed),
        "relative_feature_error": summarize_values(feat_values, seed=seed),
        "online_decoder_ms": summarize_values(times, seed=seed),
        "target_accuracy": float(np.mean([r["classification_correct"] for r in records])),
        "target_weight": str(paths["target_weight"]),
        "decoder_path": str(paths["decoder"] / f"seed{seed}" / "decoder.pth"),
        "manifest_path": str(args.manifest_path),
        "manifest_partition": "test",
        "args": vars(args),
        "per_sample": records,
        "skipped_saturated": skipped,
        "elapsed_minutes": (time.time() - start) / 60.0,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved attack summary: {summary_path}")


def evaluate_lpips(args, seed):
    paths = output_paths(args)
    seed_dir = paths["attack"] / f"seed{seed}"
    output = seed_dir / "lpips.json"
    if output.is_file() and not args.force:
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
            recon_path = real_path.with_name(real_path.name.replace("_real.png", "_recon.png"))
            real = to_tensor(Image.open(real_path).convert("RGB")).mul(2).sub(1).unsqueeze(0).to(DEVICE)
            recon = to_tensor(Image.open(recon_path).convert("RGB")).mul(2).sub(1).unsqueeze(0).to(DEVICE)
            values.append(float(metric(real, recon).mean().item()))
    payload = {"metric": "LPIPS", "net": "alex", "evaluated_samples": len(values),
               "lpips": summarize_values(values, seed=seed)}
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved LPIPS: {output}")


def summarize(args):
    paths = output_paths(args)
    rows = []
    for seed in args.seeds:
        path = paths["attack"] / f"seed{seed}" / "summary.json"
        if not path.is_file():
            raise FileNotFoundError(f"Missing attack summary: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("test_samples") != args.test_samples:
            raise RuntimeError(f"{path} contains {payload.get('test_samples')} samples, expected {args.test_samples}")
        rows.append(payload)
    reference_indices = [item["idx"] for item in rows[0]["per_sample"]]
    for payload in rows[1:]:
        if [item["idx"] for item in payload["per_sample"]] != reference_indices:
            raise RuntimeError("Decoder seeds were evaluated on different test images")
    metrics = ["psnr", "ssim", "id_loss", "relative_feature_error", "online_decoder_ms"]
    aggregate = {
        "dataset": "celeba",
        "arch": "resnet18_projected",
        "decoder_seeds": list(args.seeds),
        "test_samples": args.test_samples,
        "target_accuracy": float(np.mean([r["target_accuracy"] for r in rows])),
        "metrics": {},
        "per_seed": [],
    }
    for payload in rows:
        seed_row = {"seed": payload["seed"]}
        for metric in metrics:
            seed_row[f"{metric}_mean"] = payload[metric]["mean"]
            seed_row[f"{metric}_sample_std"] = payload[metric]["std"]
        lpips_path = paths["attack"] / f"seed{payload['seed']}" / "lpips.json"
        if lpips_path.is_file():
            lpips_payload = json.loads(lpips_path.read_text(encoding="utf-8"))
            seed_row["lpips_mean"] = lpips_payload["lpips"]["mean"]
        aggregate["per_seed"].append(seed_row)
    for metric in metrics + (["lpips"] if any("lpips_mean" in row for row in aggregate["per_seed"]) else []):
        values = [row[f"{metric}_mean"] for row in aggregate["per_seed"] if f"{metric}_mean" in row]
        aggregate["metrics"][metric] = {
            "mean_across_decoder_seeds": float(np.mean(values)),
            "std_across_decoder_seeds": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        }
    paths["summary"].write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    with paths["summary_csv"].open("w", newline="", encoding="utf-8") as handle:
        fieldnames = sorted({key for row in aggregate["per_seed"] for key in row})
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(aggregate["per_seed"])
    print(f"Saved summary: {paths['summary']}")
    print(f"Saved CSV: {paths['summary_csv']}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["target", "decoder", "attack", "lpips", "summarize", "all"], required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--manifest_path", required=True)
    parser.add_argument("--output_root", default="celeba_resnet18_generalization")
    parser.add_argument("--decoder_seed", type=int, default=2027)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--imagenet_pretrained", action="store_true")
    parser.add_argument("--target_seed", type=int, default=2027)
    parser.add_argument("--target_epochs", type=int, default=80)
    parser.add_argument("--decoder_epochs", type=int, default=120)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--img_size", type=int, default=64)
    parser.add_argument("--in_channels", type=int, default=3)
    parser.add_argument("--num_classes", type=int, default=1000)
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
    parser.add_argument("--max_target_train", type=int, default=0)
    parser.add_argument("--max_target_test", type=int, default=0)
    parser.add_argument("--max_aux_samples", type=int, default=30000)
    parser.add_argument("--decoder_samples", type=int, default=30000)
    parser.add_argument("--test_samples", type=int, default=500)
    parser.add_argument("--attack_scan", type=int, default=500)
    parser.add_argument("--save_pairs", type=int, default=40)
    parser.add_argument("--save_raw_pairs", dest="save_raw_pairs", action="store_true", default=True)
    parser.add_argument("--no_save_raw_pairs", dest="save_raw_pairs", action="store_false")
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--with_lpips", action="store_true",
                        help="Run LPIPS-Alex immediately after each attack stage.")
    return parser.parse_args()


def main():
    args = parse_args()
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
    if args.stage in {"lpips"}:
        seeds = args.seeds if args.stage == "lpips" else [args.decoder_seed]
        for seed in seeds:
            evaluate_lpips(args, seed)
    if args.stage in {"summarize", "all"}:
        summarize(args)


if __name__ == "__main__":
    main()
