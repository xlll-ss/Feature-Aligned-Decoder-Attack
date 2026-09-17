import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from train_decoder_multidataset import (
    DEVICE,
    build_target_model,
    build_dataset,
    build_transform,
    infer_dataset_num_classes,
)


def linear_hsic(x, y):
    """Linear-kernel HSIC used as a lightweight BiDO dependency measure."""
    x = x.flatten(1).float()
    y = y.flatten(1).float()
    x = F.normalize(x - x.mean(dim=0, keepdim=True), dim=1)
    y = F.normalize(y - y.mean(dim=0, keepdim=True), dim=1)
    kx = x @ x.t()
    ky = y @ y.t()
    n = x.size(0)
    h = torch.eye(n, device=x.device) - torch.full((n, n), 1.0 / n, device=x.device)
    return torch.trace(kx @ h @ ky @ h) / max((n - 1) ** 2, 1)


def forward_with_hiddens(model, x):
    if hasattr(model, "forward_features"):
        feat, hiddens = model.forward_features(x)
        logits = model.fc_layer(feat)
        return hiddens, logits, feat

    hiddens = []
    x = model.layer1(x)
    hiddens.append(x)
    x = model.layer2(x)
    hiddens.append(x)
    x = model.layer3(x)
    hiddens.append(x)
    x = model.layer4(x)
    hiddens.append(x)
    x = model.layer5(x)
    hiddens.append(x)
    x = model.avgpool(x)
    x = x.flatten(1)
    feat = model.bn(x)
    hiddens.append(feat)
    logits = model.fc_layer(feat)
    return hiddens, logits, feat


def bido_loss(inputs, hiddens, labels, num_classes, alpha, beta, max_hidden_dim=8192):
    y = F.one_hot(labels, num_classes=num_classes).float()
    x_flat = inputs.flatten(1)
    loss = torch.zeros((), device=inputs.device)

    for hidden in hiddens:
        h = hidden.flatten(1)
        if h.size(1) > max_hidden_dim:
            # Deterministic strided projection keeps HSIC affordable.
            step = math.ceil(h.size(1) / max_hidden_dim)
            h = h[:, ::step]
        dxz = linear_hsic(x_flat, h)
        dyz = linear_hsic(y, h)
        loss = loss + alpha * dxz - beta * dyz
    return loss


def accuracy(logits, labels):
    return (logits.argmax(dim=1) == labels).float().mean().item()


def build_loader(args, split, dataset_name=None, data_root=None, max_samples=None, shuffle=False):
    dataset_name = dataset_name or args.dataset
    data_root = data_root or args.data_root
    transform = build_transform(dataset_name, args.img_size, args.in_channels)
    dataset = build_dataset(
        dataset_name,
        data_root,
        split,
        transform,
        max_samples if max_samples is not None else args.max_samples,
        args.download,
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=shuffle,
    )


def evaluate(model, loader):
    model.eval()
    total_loss, total_acc, total_n = 0.0, 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)
            _, logits, _ = forward_with_hiddens(model, x)
            ce = F.cross_entropy(logits, y)
            n = x.size(0)
            total_loss += ce.item() * n
            total_acc += accuracy(logits, y) * n
            total_n += n
    return total_loss / max(total_n, 1), total_acc / max(total_n, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["celeba", "cifar10", "fashionmnist", "folder", "imagefolder"], required=True)
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--output_root", type=str, default=".")
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--arch", choices=["vgg", "resnet18"], default="vgg")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--img_size", type=int, default=128)
    parser.add_argument("--in_channels", type=int, default=3)
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--feature_dim", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--max_samples", type=int, default=50000)
    parser.add_argument("--test_samples", type=int, default=10000)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=0.001)
    parser.add_argument("--beta", type=float, default=0.005)
    parser.add_argument("--disable_bido", action="store_true")
    parser.add_argument("--enable_oe", action="store_true")
    parser.add_argument("--oe_dataset", choices=["cifar10", "fashionmnist", "folder", "imagefolder"], default="cifar10")
    parser.add_argument("--oe_data_root", type=str, default="")
    parser.add_argument("--oe_weight", type=float, default=1e-3)
    parser.add_argument("--save_every", type=int, default=10)
    args = parser.parse_args()

    run = args.run_name or f"{args.dataset}_{args.arch}_bido_plus"
    output_root = Path(args.output_root).expanduser()
    model_dir = output_root / "bido_target_models"
    log_dir = output_root / "bido_target_logs" / run
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"target_{run}.pth"

    train_loader = build_loader(args, "train", max_samples=args.max_samples, shuffle=True)
    test_loader = build_loader(args, "test", max_samples=args.test_samples, shuffle=False)
    inferred_num_classes = infer_dataset_num_classes(train_loader.dataset)
    if args.num_classes <= 0:
        if inferred_num_classes is None:
            raise RuntimeError("Cannot infer num_classes from this dataset. Please set --num_classes explicitly.")
        args.num_classes = inferred_num_classes
        print(f"[INFO] inferred num_classes={args.num_classes} from training dataset")
    elif inferred_num_classes is not None and args.num_classes != inferred_num_classes:
        print(
            f"[WARN] --num_classes={args.num_classes}, but dataset appears to have "
            f"{inferred_num_classes} classes. Keeping explicit --num_classes."
        )
    oe_loader = None
    if args.enable_oe:
        oe_root = args.oe_data_root or args.data_root
        oe_loader = build_loader(
            args,
            "train",
            dataset_name=args.oe_dataset,
            data_root=oe_root,
            max_samples=args.max_samples,
            shuffle=True,
        )
        oe_iter = iter(oe_loader)
    else:
        oe_iter = None

    model = build_target_model(
        args.arch,
        in_channels=args.in_channels,
        num_classes=args.num_classes,
        feature_dim=args.feature_dim,
    ).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))

    config = vars(args)
    (log_dir / "training_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    history = []
    best = {"acc": -1.0, "epoch": -1, "path": str(model_path)}
    start = time.time()

    print("=" * 78)
    print("BiDO+ target classifier training")
    print("=" * 78)
    print(f"dataset: {args.dataset}")
    print(f"arch: {args.arch}")
    print(f"device: {DEVICE}")
    print(f"output_model: {model_path}")
    print(f"bido: {not args.disable_bido}, alpha={args.alpha}, beta={args.beta}")
    print(f"oe: {args.enable_oe}, oe_dataset={args.oe_dataset}, oe_weight={args.oe_weight}")
    print("=" * 78)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = {"loss": 0.0, "ce": 0.0, "bido": 0.0, "oe": 0.0, "acc": 0.0, "n": 0}
        for x, y in train_loader:
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)
            hiddens, logits, _ = forward_with_hiddens(model, x)
            ce = F.cross_entropy(logits, y)
            dep = torch.zeros((), device=DEVICE)
            if not args.disable_bido:
                dep = bido_loss(x, hiddens, y, args.num_classes, args.alpha, args.beta)

            oe_term = torch.zeros((), device=DEVICE)
            if args.enable_oe:
                try:
                    oe_x, _ = next(oe_iter)
                except StopIteration:
                    oe_iter = iter(oe_loader)
                    oe_x, _ = next(oe_iter)
                oe_x = oe_x.to(DEVICE, non_blocking=True)
                _, oe_logits, _ = forward_with_hiddens(model, oe_x)
                entropy = -(F.softmax(oe_logits, dim=1) * F.log_softmax(oe_logits, dim=1)).sum(dim=1).mean()
                oe_term = -args.oe_weight * entropy

            loss = ce + dep + oe_term
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            n = x.size(0)
            total["loss"] += loss.item() * n
            total["ce"] += ce.item() * n
            total["bido"] += dep.item() * n
            total["oe"] += oe_term.item() * n
            total["acc"] += accuracy(logits, y) * n
            total["n"] += n

        scheduler.step()
        val_loss, val_acc = evaluate(model, test_loader)
        row = {
            "epoch": epoch,
            "train_loss": total["loss"] / max(total["n"], 1),
            "train_ce": total["ce"] / max(total["n"], 1),
            "train_bido": total["bido"] / max(total["n"], 1),
            "train_oe": total["oe"] / max(total["n"], 1),
            "train_acc": total["acc"] / max(total["n"], 1),
            "val_loss": val_loss,
            "val_acc": val_acc,
        }
        history.append(row)
        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train_acc={row['train_acc']:.4f} val_acc={val_acc:.4f} "
            f"loss={row['train_loss']:.4f} ce={row['train_ce']:.4f} "
            f"bido={row['train_bido']:.5f} oe={row['train_oe']:.5f}"
        )

        if val_acc > best["acc"] or epoch == args.epochs or epoch % args.save_every == 0:
            ckpt = {
                "state_dict": model.state_dict(),
                "epoch": epoch,
                "dataset": args.dataset,
                "arch": args.arch,
                "img_size": args.img_size,
                "in_channels": args.in_channels,
                "num_classes": args.num_classes,
                "feature_dim": args.feature_dim,
                "val_acc": val_acc,
                "args": vars(args),
            }
            if val_acc > best["acc"]:
                best.update({"acc": val_acc, "epoch": epoch})
                torch.save(ckpt, model_path)
            torch.save(ckpt, model_dir / f"target_{run}_latest.pth")
        (log_dir / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    print("=" * 78)
    print(f"Training complete in {(time.time() - start) / 60:.1f} min")
    print(f"Best epoch: {best['epoch']}, best val acc: {best['acc']:.4f}")
    print(f"Best checkpoint: {model_path}")
    print("=" * 78)


if __name__ == "__main__":
    main()
