import argparse
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from torchvision.utils import save_image

from experiment_utils import ManifestImageDataset, seed_everything


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FlexibleVGG(nn.Module):
    """VGG-style BiDO+ classifier with configurable input/classes.

    The feature dimension is fixed to 2048 by adaptive pooling to 2x2 with
    512 channels, matching the CelebA BiDO+ setting used in the main paper.
    """

    def __init__(self, in_channels=3, num_classes=1000, feature_dim=2048):
        super().__init__()
        self.num_classes = num_classes
        if feature_dim != 2048:
            raise ValueError("FlexibleVGG currently expects feature_dim=2048")
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((2, 2))
        self.bn = nn.BatchNorm1d(feature_dim)
        self.fc_layer = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        feat = self.bn(x)
        logits = self.fc_layer(feat)
        return logits, feat


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out, inplace=True)


class FlexibleResNet18(nn.Module):
    """ResNet-18 target with the same final leakage interface as FlexibleVGG.

    The pooled ResNet feature is projected to ``feature_dim`` before the final
    ``fc_layer`` so decoder checkpoints can keep using the 2048-dim feature
    interface used by the main CelebA experiments.
    """

    def __init__(self, in_channels=3, num_classes=10, feature_dim=2048):
        super().__init__()
        self.num_classes = num_classes
        self.in_planes = 64
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.feat_proj = nn.Linear(512, feature_dim)
        self.bn = nn.BatchNorm1d(feature_dim)
        self.fc_layer = nn.Linear(feature_dim, num_classes)

    def _make_layer(self, planes, blocks, stride):
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward_features(self, x):
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        h1 = self.layer1(x)
        h2 = self.layer2(h1)
        h3 = self.layer3(h2)
        h4 = self.layer4(h3)
        pooled = self.avgpool(h4).flatten(1)
        feat = self.bn(self.feat_proj(pooled))
        return feat, [h1, h2, h3, h4, feat]

    def forward(self, x):
        feat, _ = self.forward_features(x)
        logits = self.fc_layer(feat)
        return logits, feat


def build_target_model(arch, in_channels=3, num_classes=10, feature_dim=2048):
    arch = arch.lower()
    if arch == "vgg":
        return FlexibleVGG(in_channels=in_channels, num_classes=num_classes, feature_dim=feature_dim)
    if arch == "resnet18":
        return FlexibleResNet18(in_channels=in_channels, num_classes=num_classes, feature_dim=feature_dim)
    raise ValueError(f"Unsupported arch: {arch}")


class FeatureDecoder(nn.Module):
    def __init__(self, feature_dim=2048, img_size=128, out_channels=3):
        super().__init__()
        if img_size < 16 or (img_size & (img_size - 1)) != 0:
            raise ValueError("img_size must be a power of two and >= 16")
        self.img_size = img_size
        self.out_channels = out_channels
        self.start_size = 4
        start_channels = 512
        output_dim = start_channels * self.start_size * self.start_size

        self.fc = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, 4096),
            nn.GELU(),
            nn.Linear(4096, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(True),
        )

        num_upsample = int(math.log2(img_size)) - 2
        layers = []
        in_channels = start_channels
        out_channels_mid = 256
        for _ in range(num_upsample):
            layers.extend([
                nn.ConvTranspose2d(in_channels, out_channels_mid, 4, 2, 1),
                nn.BatchNorm2d(out_channels_mid),
                nn.ReLU(True),
            ])
            in_channels = out_channels_mid
            out_channels_mid = max(out_channels_mid // 2, 32)
        layers.extend([nn.Conv2d(in_channels, out_channels, 3, 1, 1), nn.Tanh()])
        self.deconv = nn.Sequential(*layers)

    def forward(self, feat):
        x = self.fc(feat)
        x = x.view(x.size(0), 512, self.start_size, self.start_size)
        return self.deconv(x)


class ImageFolderFlat(Dataset):
    def __init__(self, root, transform=None, max_samples=None):
        self.root = Path(root).expanduser()
        self.transform = transform
        if not self.root.exists():
            raise FileNotFoundError(f"Folder not found: {self.root}")
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        self.paths = [p for p in sorted(self.root.rglob("*")) if p.suffix.lower() in exts]
        if max_samples:
            self.paths = self.paths[:max_samples]
        if not self.paths:
            raise RuntimeError(f"No image files found in: {self.root}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        img = Image.open(self.paths[index]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, 0


def extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model", "net", "classifier"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
    return checkpoint


def strip_state_prefix(key):
    for prefix in ("module.", "model.", "classifier."):
        if key.startswith(prefix):
            key = key[len(prefix):]
    return key


def remap_vgg_feature_key(key):
    """Map the official undefended VGG feature.* layout to layer1..layer5."""
    parts = key.split(".", 2)
    if len(parts) != 3 or parts[0] != "feature":
        return key
    try:
        feature_index = int(parts[1])
    except ValueError:
        return key

    # torchvision VGG16-BN feature blocks end at MaxPool indices 6, 13,
    # 23, 33, and 43. FlexibleVGG stores the same modules per block.
    for layer_index, (start, stop) in enumerate(
        ((0, 7), (7, 14), (14, 24), (24, 34), (34, 44)),
        start=1,
    ):
        if start <= feature_index < stop:
            local_index = feature_index - start
            return f"layer{layer_index}.{local_index}.{parts[2]}"
    return key


def infer_checkpoint_num_classes(raw_state):
    for key, value in raw_state.items():
        if strip_state_prefix(key) == "fc_layer.weight" and hasattr(value, "shape"):
            return int(value.shape[0])
    return None


def infer_dataset_num_classes(dataset):
    base = dataset
    while isinstance(base, Subset):
        base = base.dataset
    if hasattr(base, "classes"):
        return len(base.classes)
    if hasattr(base, "targets"):
        targets = base.targets
        if isinstance(targets, torch.Tensor):
            return int(targets.max().item()) + 1
        if targets:
            return int(max(targets)) + 1
    return None


def load_target_model(weight_path, in_channels, num_classes, feature_dim, arch="vgg"):
    if not weight_path:
        raise ValueError("--target_weight is required")
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"Target checkpoint not found: {weight_path}")

    checkpoint = torch.load(weight_path, map_location=DEVICE)
    raw_state = extract_state_dict(checkpoint)
    checkpoint_num_classes = infer_checkpoint_num_classes(raw_state)
    if checkpoint_num_classes is not None:
        if num_classes <= 0:
            print(f"[INFO] inferred num_classes={checkpoint_num_classes} from checkpoint fc_layer.weight")
            num_classes = checkpoint_num_classes
        elif num_classes != checkpoint_num_classes:
            print(
                f"[WARN] --num_classes={num_classes} does not match checkpoint "
                f"fc_layer.weight={checkpoint_num_classes}; using checkpoint value."
            )
            num_classes = checkpoint_num_classes
    elif num_classes <= 0:
        raise RuntimeError("Cannot infer num_classes from checkpoint. Please set --num_classes explicitly.")

    model = build_target_model(
        arch=arch,
        in_channels=in_channels,
        num_classes=num_classes,
        feature_dim=feature_dim,
    ).to(DEVICE)
    target_state = model.state_dict()
    state_dict = {}
    skipped = []
    for key, value in raw_state.items():
        new_key = strip_state_prefix(key)
        if arch == "vgg":
            new_key = remap_vgg_feature_key(new_key)
        if new_key in target_state and target_state[new_key].shape == value.shape:
            state_dict[new_key] = value
        else:
            skipped.append(new_key)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    critical = ["bn.weight", "bn.bias", "fc_layer.weight", "fc_layer.bias"]
    if arch == "resnet18":
        critical.extend(["feat_proj.weight", "feat_proj.bias"])
    critical_missing = [k for k in critical if k not in state_dict]
    if critical_missing:
        raise RuntimeError("Critical target weights missing: " + ", ".join(critical_missing))
    coverage = len(state_dict) / max(1, len(target_state))
    if coverage < 0.90:
        raise RuntimeError(
            "Target checkpoint architecture mismatch: "
            f"loaded only {len(state_dict)}/{len(target_state)} tensors "
            f"({coverage:.1%}); missing={len(missing)}, skipped={len(skipped)}"
        )
    model.eval()
    model.num_classes = num_classes
    for param in model.parameters():
        param.requires_grad = False
    print(f"[OK] loaded target: {weight_path}")
    print(
        f"[INFO] arch={arch}, loaded={len(state_dict)}, coverage={coverage:.1%}, "
        f"missing={len(missing)}, skipped={len(skipped)}, unexpected={len(unexpected)}"
    )
    return model


def build_transform(dataset, img_size, in_channels):
    if dataset == "celeba":
        # BiDO+ low-resolution CelebA preprocessing: center crop 108x108,
        # resize to 64x64, then tensor conversion. Keep decoder tensors in
        # [-1, 1]; target_model_input() converts them back to the classifier's
        # original [0, 1] range.
        return transforms.Compose([
            transforms.CenterCrop(108),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,) * in_channels, (0.5,) * in_channels),
        ])
    ops = []
    if dataset == "fashionmnist":
        ops.append(transforms.Grayscale(num_output_channels=in_channels))
    elif in_channels == 1:
        ops.append(transforms.Grayscale(num_output_channels=1))
    ops.extend([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,) * in_channels, (0.5,) * in_channels),
    ])
    return transforms.Compose(ops)


def target_model_input(images, dataset):
    """Convert experiment tensors to the target model's input convention."""
    if dataset == "celeba":
        return (images + 1.0) / 2.0
    return images


def build_dataset(
    dataset,
    data_root,
    split,
    transform,
    max_samples,
    download,
    manifest_path="",
    manifest_partition="",
):
    root = Path(data_root).expanduser()
    if manifest_path:
        if dataset not in {"celeba", "folder"}:
            raise ValueError("--manifest_path is only used with celeba/folder datasets")
        if not manifest_partition:
            raise ValueError("--manifest_partition is required with --manifest_path")
        return ManifestImageDataset(
            manifest_path,
            manifest_partition,
            root_override=root,
            transform=transform,
            max_samples=max_samples,
        )
    if dataset == "celeba":
        return ImageFolderFlat(root, transform=transform, max_samples=max_samples)
    if dataset == "folder":
        return ImageFolderFlat(root, transform=transform, max_samples=max_samples)
    if dataset == "imagefolder":
        split_root = root / split
        if split_root.exists():
            root = split_root
        ds = datasets.ImageFolder(str(root), transform=transform)
        if max_samples and max_samples < len(ds):
            ds = Subset(ds, list(range(max_samples)))
        return ds
    if dataset == "cifar10":
        train = split == "train"
        ds = datasets.CIFAR10(str(root), train=train, transform=transform, download=download)
    elif dataset == "fashionmnist":
        train = split == "train"
        ds = datasets.FashionMNIST(str(root), train=train, transform=transform, download=download)
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    if max_samples and max_samples < len(ds):
        ds = Subset(ds, list(range(max_samples)))
    return ds


def total_variation_loss(x):
    tv_h = torch.mean(torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]))
    tv_w = torch.mean(torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]))
    return tv_h + tv_w


def save_training_samples(real, recon, path, n=8):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    real = real[:n].detach().cpu()
    recon = recon[:n].detach().cpu()
    save_image((torch.cat([real, recon], dim=0) + 1) / 2, path, nrow=n)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["celeba", "cifar10", "fashionmnist", "folder", "imagefolder"], required=True)
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--aux_dataset", choices=["celeba", "cifar10", "fashionmnist", "folder", "imagefolder"], default="")
    parser.add_argument("--aux_data_root", type=str, default="")
    parser.add_argument("--aux_split", choices=["train", "test"], default="train")
    parser.add_argument("--aux_max_samples", type=int, default=0)
    parser.add_argument("--target_weight", type=str, required=True)
    parser.add_argument("--output_root", type=str, default=".")
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--arch", choices=["vgg", "resnet18"], default="vgg")
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--img_size", type=int, default=128)
    parser.add_argument("--in_channels", type=int, default=3)
    parser.add_argument("--num_classes", type=int, default=10, help="Set to 0 to infer from target checkpoint when loading.")
    parser.add_argument("--feature_dim", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max_samples", type=int, default=50000)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--l1_weight", type=float, default=10.0)
    parser.add_argument("--mse_weight", type=float, default=0.0)
    parser.add_argument("--feat_cycle_weight", type=float, default=0.0)
    parser.add_argument("--tv_weight", type=float, default=0.0)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--manifest_path", type=str, default="")
    parser.add_argument(
        "--manifest_partition",
        choices=["", "auxiliary", "validation", "test"],
        default="",
        help="For flat CelebA/folder data, decoder training should use auxiliary.",
    )
    args = parser.parse_args()

    seed_everything(args.seed)

    aux_dataset_name = args.aux_dataset or args.dataset
    aux_data_root = args.aux_data_root or args.data_root
    aux_split = args.aux_split if args.aux_dataset else args.split
    aux_max_samples = args.aux_max_samples if args.aux_max_samples > 0 else args.max_samples
    run = args.run_name or f"{args.dataset}_{args.arch}_aux_{aux_dataset_name}_l1"
    output_root = Path(args.output_root).expanduser()
    model_dir = output_root / "models"
    sample_dir = output_root / "decoder_multidataset_samples" / run
    model_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"decoder_{run}.pth"

    print("=" * 78)
    print("Feature decoder training for cross-dataset main comparison")
    print("=" * 78)
    print(f"dataset: {args.dataset}")
    print(f"arch: {args.arch}")
    print(f"device: {DEVICE}")
    print(f"data_root: {args.data_root}")
    print(f"aux_dataset: {aux_dataset_name}")
    print(f"aux_data_root: {aux_data_root}")
    print(f"target_weight: {args.target_weight}")
    print(f"output_model: {model_path}")
    print(f"img_size={args.img_size}, in_channels={args.in_channels}, num_classes={args.num_classes}")
    print(f"epochs={args.epochs}, batch_size={args.batch_size}, max_samples={args.max_samples}")
    print(f"loss: l1={args.l1_weight}, mse={args.mse_weight}, feat={args.feat_cycle_weight}, tv={args.tv_weight}")
    print("=" * 78)

    transform = build_transform(aux_dataset_name, args.img_size, args.in_channels)
    dataset = build_dataset(
        aux_dataset_name,
        aux_data_root,
        aux_split,
        transform,
        aux_max_samples,
        args.download,
        manifest_path=args.manifest_path,
        manifest_partition=args.manifest_partition,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )

    target_model = load_target_model(args.target_weight, args.in_channels, args.num_classes, args.feature_dim, args.arch)
    decoder = FeatureDecoder(args.feature_dim, args.img_size, args.in_channels).to(DEVICE)
    optimizer = optim.AdamW(decoder.parameters(), lr=args.lr, betas=(0.5, 0.999), weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))

    history = []
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        decoder.train()
        totals = {"loss": 0.0, "l1": 0.0, "mse": 0.0, "feat": 0.0, "tv": 0.0}
        for real, _ in loader:
            real = real.to(DEVICE, non_blocking=True)
            with torch.no_grad():
                _, feat = target_model(target_model_input(real, aux_dataset_name))
                feat = feat.detach()
            recon = decoder(feat)
            l1 = F.l1_loss(recon, real)
            mse = F.mse_loss(recon, real)
            tv = total_variation_loss(recon)

            if args.feat_cycle_weight > 0:
                _, recon_feat = target_model(target_model_input(recon, aux_dataset_name))
                feat_cycle = F.mse_loss(recon_feat.float(), feat.float())
            else:
                feat_cycle = torch.zeros((), device=DEVICE)

            loss = (
                args.l1_weight * l1
                + args.mse_weight * mse
                + args.feat_cycle_weight * feat_cycle
                + args.tv_weight * tv
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), 5.0)
            optimizer.step()

            totals["loss"] += loss.item()
            totals["l1"] += l1.item()
            totals["mse"] += mse.item()
            totals["feat"] += feat_cycle.item()
            totals["tv"] += tv.item()

        scheduler.step()
        steps = max(len(loader), 1)
        row = {"epoch": epoch, **{k: v / steps for k, v in totals.items()}}
        history.append(row)
        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"loss={row['loss']:.4f} l1={row['l1']:.4f} mse={row['mse']:.4f} "
            f"feat={row['feat']:.5f} tv={row['tv']:.4f}"
        )

        if epoch == 1 or epoch % args.save_every == 0 or epoch == args.epochs:
            decoder.eval()
            with torch.no_grad():
                real_vis, _ = next(iter(loader))
                real_vis = real_vis.to(DEVICE)
                _, feat_vis = target_model(real_vis)
                recon_vis = decoder(feat_vis)
            save_training_samples(real_vis, recon_vis, sample_dir / f"epoch_{epoch:03d}.png")
            torch.save(
                {
                    "state_dict": decoder.state_dict(),
                    "dataset": args.dataset,
                    "aux_dataset": aux_dataset_name,
                    "arch": args.arch,
                    "feature_dim": args.feature_dim,
                    "img_size": args.img_size,
                    "in_channels": args.in_channels,
                    "num_classes": int(getattr(target_model, "num_classes", args.num_classes)),
                    "epoch": epoch,
                    "args": vars(args),
                },
                model_path,
            )

    (sample_dir / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (sample_dir / "training_config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    print("=" * 78)
    print(f"Training complete in {(time.time() - start) / 60:.1f} min")
    print(f"Decoder saved to: {model_path}")
    print(f"Samples saved to: {sample_dir}")
    print("=" * 78)


if __name__ == "__main__":
    main()
