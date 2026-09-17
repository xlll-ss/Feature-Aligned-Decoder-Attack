"""Train controlled FAD decoder ablations without modifying the main trainer."""

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from experiment_utils import seed_everything
from residual_decoder import ResidualFeatureDecoder
from train_decoder_multidataset import (
    DEVICE,
    FeatureDecoder,
    build_dataset,
    build_transform,
    load_target_model,
    target_model_input,
)


VARIANTS = {
    "full": {
        "family": "base",
        "l1_weight": 10.0,
        "mse_weight": 1.0,
        "feat_cycle_weight": 0.0,
        "description": "Main decoder: 10*L1 + MSE",
    },
    "l1_only": {
        "family": "base",
        "l1_weight": 10.0,
        "mse_weight": 0.0,
        "feat_cycle_weight": 0.0,
        "description": "Main architecture trained with L1 only",
    },
    "mse_only": {
        "family": "base",
        "l1_weight": 0.0,
        "mse_weight": 1.0,
        "feat_cycle_weight": 0.0,
        "description": "Main architecture trained with MSE only",
    },
    "pixel_feature": {
        "family": "base",
        "l1_weight": 10.0,
        "mse_weight": 1.0,
        "feat_cycle_weight": 0.01,
        "description": "Main pixel objective plus target-feature cycle loss",
    },
    "no_feature_norm": {
        "family": "no_feature_norm",
        "l1_weight": 10.0,
        "mse_weight": 1.0,
        "feat_cycle_weight": 0.0,
        "description": "Main decoder without input LayerNorm",
    },
    "compact": {
        "family": "compact",
        "l1_weight": 10.0,
        "mse_weight": 1.0,
        "feat_cycle_weight": 0.0,
        "description": "Reduced-width decoder capacity",
    },
    "residual": {
        "family": "residual",
        "l1_weight": 10.0,
        "mse_weight": 1.0,
        "feat_cycle_weight": 0.0,
        "description": "BatchNorm-free residual decoder with the same pixel loss",
    },
}


class ConfigurableFeatureDecoder(nn.Module):
    decoder_type = "core_ablation_configurable_v1"

    def __init__(
        self,
        feature_dim=2048,
        img_size=64,
        out_channels=3,
        feature_norm=True,
        hidden_dim=4096,
        start_channels=512,
    ):
        super().__init__()
        if img_size < 16 or (img_size & (img_size - 1)) != 0:
            raise ValueError("img_size must be a power of two and >= 16")
        self.img_size = img_size
        self.start_size = 4
        self.start_channels = start_channels
        output_dim = start_channels * self.start_size * self.start_size
        self.fc = nn.Sequential(
            nn.LayerNorm(feature_dim) if feature_norm else nn.Identity(),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(True),
        )
        layers = []
        in_channels = start_channels
        out_channels_mid = start_channels // 2
        for _ in range(int(math.log2(img_size)) - 2):
            layers.extend(
                [
                    nn.ConvTranspose2d(in_channels, out_channels_mid, 4, 2, 1),
                    nn.BatchNorm2d(out_channels_mid),
                    nn.ReLU(True),
                ]
            )
            in_channels = out_channels_mid
            out_channels_mid = max(out_channels_mid // 2, 32)
        layers.extend([nn.Conv2d(in_channels, out_channels, 3, 1, 1), nn.Tanh()])
        self.deconv = nn.Sequential(*layers)

    def forward(self, feature):
        image = self.fc(feature)
        image = image.view(
            image.size(0), self.start_channels, self.start_size, self.start_size
        )
        return self.deconv(image)


def variant_model_config(variant):
    family = VARIANTS[variant]["family"]
    if family == "compact":
        return {
            "feature_norm": True,
            "hidden_dim": 2048,
            "start_channels": 256,
        }
    if family == "no_feature_norm":
        return {
            "feature_norm": False,
            "hidden_dim": 4096,
            "start_channels": 512,
        }
    if family == "residual":
        return {"residual_scale": 0.1}
    return {}


def build_decoder(variant, feature_dim, img_size, in_channels):
    family = VARIANTS[variant]["family"]
    config = variant_model_config(variant)
    if family == "base":
        return FeatureDecoder(feature_dim, img_size, in_channels)
    if family == "residual":
        return ResidualFeatureDecoder(
            feature_dim,
            img_size,
            in_channels,
            residual_scale=config["residual_scale"],
        )
    return ConfigurableFeatureDecoder(
        feature_dim,
        img_size,
        in_channels,
        **config,
    )


def capture_rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2), encoding="utf-8")


def load_training_checkpoint(path, map_location):
    """Load trusted local checkpoints that include Python and NumPy RNG state."""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--target_weight", required=True)
    parser.add_argument("--manifest_path", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max_samples", type=int, default=30000)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--restart", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.max_samples <= 0:
        raise ValueError("epochs, batch_size, and max_samples must be positive")
    variant = VARIANTS[args.variant]
    output_root = Path(args.output_root).expanduser()
    model_dir = output_root / "models"
    run_dir = output_root / "training" / args.variant / f"seed{args.seed}"
    model_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"decoder_{args.variant}_seed{args.seed}.pth"
    latest_path = run_dir / "latest.pth"
    history_path = run_dir / "history.json"
    config_path = run_dir / "config.json"

    protocol = {
        "variant": args.variant,
        "variant_spec": variant,
        "model_config": variant_model_config(args.variant),
        "data_root": str(Path(args.data_root).expanduser()),
        "target_weight": str(Path(args.target_weight).expanduser()),
        "manifest_path": str(Path(args.manifest_path).expanduser()),
        "manifest_partition": "auxiliary",
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "max_samples": args.max_samples,
        "seed": args.seed,
        "data_order": "epoch-seeded permutation derived from decoder seed",
        "optimizer": "AdamW(betas=(0.5,0.999),weight_decay=1e-4)",
        "scheduler": "CosineAnnealingLR",
    }
    if args.restart:
        model_path.unlink(missing_ok=True)
        latest_path.unlink(missing_ok=True)
        history_path.unlink(missing_ok=True)
    if config_path.is_file() and not args.restart:
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        if previous != protocol:
            raise RuntimeError(
                f"Existing configuration differs: {config_path}. Use --restart or another output root."
            )
    write_json(config_path, protocol)
    if model_path.is_file() and not args.restart:
        checkpoint = load_training_checkpoint(model_path, map_location="cpu")
        if checkpoint.get("epoch") == args.epochs and checkpoint.get("protocol") == protocol:
            print(f"[skip] decoder already complete: {model_path}")
            return

    seed_everything(args.seed)
    transform = build_transform("celeba", 64, 3)
    dataset = build_dataset(
        "celeba",
        args.data_root,
        "train",
        transform,
        args.max_samples,
        False,
        manifest_path=args.manifest_path,
        manifest_partition="auxiliary",
    )
    loader_generator = torch.Generator()
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        generator=loader_generator,
    )
    target = load_target_model(args.target_weight, 3, 0, 2048, "vgg")
    target.eval()
    for parameter in target.parameters():
        parameter.requires_grad_(False)
    decoder = build_decoder(args.variant, 2048, 64, 3).to(DEVICE)
    optimizer = optim.AdamW(
        decoder.parameters(), lr=args.lr, betas=(0.5, 0.999), weight_decay=1e-4
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs, 1)
    )
    history = []
    start_epoch = 1
    elapsed_before = 0.0
    if latest_path.is_file() and not args.restart:
        checkpoint = load_training_checkpoint(latest_path, map_location=DEVICE)
        if checkpoint.get("protocol") != protocol:
            raise RuntimeError(f"Resume checkpoint protocol differs: {latest_path}")
        decoder.load_state_dict(checkpoint["state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        history = checkpoint["history"]
        start_epoch = int(checkpoint["epoch"]) + 1
        elapsed_before = float(checkpoint.get("elapsed_seconds", 0.0))
        restore_rng_state(checkpoint["rng_state"])
        print(f"[resume] epoch {start_epoch}/{args.epochs} from {latest_path}")
        if start_epoch > args.epochs:
            torch.save(checkpoint, model_path)
            print(f"[recover] restored completed decoder: {model_path}")
            return

    parameter_count = sum(parameter.numel() for parameter in decoder.parameters())
    started = time.time()
    print("=" * 88)
    print(f"variant={args.variant} family={variant['family']} device={DEVICE}")
    print(f"samples={len(dataset)} epochs={args.epochs} batch={args.batch_size} seed={args.seed}")
    print(
        f"loss: l1={variant['l1_weight']} mse={variant['mse_weight']} "
        f"feature_cycle={variant['feat_cycle_weight']} parameters={parameter_count}"
    )
    print("=" * 88)

    for epoch in range(start_epoch, args.epochs + 1):
        loader_generator.manual_seed(args.seed * 1_000_003 + epoch)
        decoder.train()
        totals = {"loss": 0.0, "l1": 0.0, "mse": 0.0, "feature_cycle": 0.0}
        steps = 0
        for real, _ in loader:
            real = real.to(DEVICE, non_blocking=True)
            with torch.no_grad():
                _, feature = target(target_model_input(real, "celeba"))
            reconstruction = decoder(feature.detach())
            l1 = F.l1_loss(reconstruction, real)
            mse = F.mse_loss(reconstruction, real)
            if variant["feat_cycle_weight"] > 0:
                _, reconstruction_feature = target(
                    target_model_input(reconstruction, "celeba")
                )
                feature_cycle = F.mse_loss(
                    reconstruction_feature.float(), feature.float()
                )
            else:
                feature_cycle = torch.zeros((), device=DEVICE)
            loss = (
                variant["l1_weight"] * l1
                + variant["mse_weight"] * mse
                + variant["feat_cycle_weight"] * feature_cycle
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), 5.0)
            optimizer.step()
            totals["loss"] += float(loss.item())
            totals["l1"] += float(l1.item())
            totals["mse"] += float(mse.item())
            totals["feature_cycle"] += float(feature_cycle.item())
            steps += 1
        scheduler.step()
        row = {key: value / max(steps, 1) for key, value in totals.items()}
        row.update({"epoch": epoch, "lr": float(scheduler.get_last_lr()[0])})
        history.append(row)
        total_elapsed = elapsed_before + time.time() - started
        print(
            f"epoch={epoch:03d}/{args.epochs} loss={row['loss']:.5f} "
            f"l1={row['l1']:.5f} mse={row['mse']:.5f} "
            f"feature={row['feature_cycle']:.5f}"
        )
        if epoch % args.save_every == 0 or epoch == args.epochs:
            checkpoint = {
                "state_dict": decoder.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "rng_state": capture_rng_state(),
                "history": history,
                "epoch": epoch,
                "elapsed_seconds": total_elapsed,
                "decoder_type": (
                    ResidualFeatureDecoder.decoder_type
                    if variant["family"] == "residual"
                    else (
                        ConfigurableFeatureDecoder.decoder_type
                        if variant["family"] in {"compact", "no_feature_norm"}
                        else "base"
                    )
                ),
                "decoder_ablation_variant": args.variant,
                "variant_spec": variant,
                "model_config": variant_model_config(args.variant),
                "parameter_count": parameter_count,
                "feature_dim": 2048,
                "img_size": 64,
                "in_channels": 3,
                "protocol": protocol,
            }
            torch.save(checkpoint, latest_path)
            write_json(history_path, history)
            if epoch == args.epochs:
                torch.save(checkpoint, model_path)
    print(f"decoder: {model_path}")
    print(f"training_seconds: {elapsed_before + time.time() - started:.1f}")


if __name__ == "__main__":
    main()
