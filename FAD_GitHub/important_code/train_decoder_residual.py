"""Train the BatchNorm-free residual feature decoder."""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from experiment_utils import seed_everything
from residual_decoder import ResidualFeatureDecoder
from train_decoder_multidataset import (
    DEVICE,
    build_dataset,
    build_transform,
    load_target_model,
    target_model_input,
    total_variation_loss,
)


def main():
    parser = argparse.ArgumentParser(description="Train a BatchNorm-free residual FAD decoder")
    parser.add_argument("--dataset", choices=["celeba", "cifar10", "fashionmnist", "folder", "imagefolder"], required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--target_weight", required=True)
    parser.add_argument("--output_root", default=".")
    parser.add_argument("--run_name", required=True)
    parser.add_argument("--arch", choices=["vgg", "resnet18"], default="vgg")
    parser.add_argument("--img_size", type=int, default=64)
    parser.add_argument("--in_channels", type=int, default=3)
    parser.add_argument("--num_classes", type=int, default=0)
    parser.add_argument("--feature_dim", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=240)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_samples", type=int, default=60000)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--l1_weight", type=float, default=5.0)
    parser.add_argument("--mse_weight", type=float, default=1.0)
    parser.add_argument("--feat_cycle_weight", type=float, default=0.01)
    parser.add_argument("--tv_weight", type=float, default=0.001)
    parser.add_argument("--residual_scale", type=float, default=0.1)
    parser.add_argument("--save_every", type=int, default=20)
    parser.add_argument("--manifest_path", default="")
    parser.add_argument("--manifest_partition", choices=["", "auxiliary", "validation", "test"], default="auxiliary")
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args()
    seed_everything(args.seed)

    output_root = Path(args.output_root).expanduser()
    model_dir = output_root / "models"
    sample_dir = output_root / "decoder_residual_samples" / args.run_name
    model_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"decoder_{args.run_name}.pth"
    history_path = sample_dir / "history.json"

    transform = build_transform(args.dataset, args.img_size, args.in_channels)
    dataset = build_dataset(
        args.dataset,
        args.data_root,
        "train",
        transform,
        args.max_samples,
        False,
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

    target_model = load_target_model(
        args.target_weight,
        args.in_channels,
        args.num_classes,
        args.feature_dim,
        args.arch,
    )
    target_model.eval()
    for parameter in target_model.parameters():
        parameter.requires_grad_(False)

    decoder = ResidualFeatureDecoder(
        args.feature_dim,
        args.img_size,
        args.in_channels,
        residual_scale=args.residual_scale,
    ).to(DEVICE)
    optimizer = optim.AdamW(decoder.parameters(), lr=args.lr, betas=(0.5, 0.999), weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    history = []
    start = time.time()

    print("=" * 78)
    print("Residual feature decoder training")
    print(f"run={args.run_name} device={DEVICE} dataset={args.dataset} samples={len(dataset)}")
    print(f"epochs={args.epochs} batch_size={args.batch_size} lr={args.lr} residual_scale={args.residual_scale}")
    print(f"loss: l1={args.l1_weight} mse={args.mse_weight} feat={args.feat_cycle_weight} tv={args.tv_weight}")
    print("=" * 78)

    for epoch in range(1, args.epochs + 1):
        decoder.train()
        totals = {"loss": 0.0, "l1": 0.0, "mse": 0.0, "feat": 0.0, "tv": 0.0, "residual": 0.0}
        steps = 0
        for real, _ in loader:
            real = real.to(DEVICE, non_blocking=True)
            with torch.no_grad():
                _, feature = target_model(target_model_input(real, args.dataset))
            recon, _, residual = decoder.forward_with_parts(feature.detach())
            l1 = F.l1_loss(recon, real)
            mse = F.mse_loss(recon, real)
            tv = total_variation_loss(residual)
            if args.feat_cycle_weight > 0:
                # Target parameters are frozen, but gradients must flow from
                # the feature-cycle term back into the decoder output.
                _, recon_feature = target_model(target_model_input(recon, args.dataset))
                feat_loss = F.mse_loss(recon_feature.float(), feature.float())
            else:
                feat_loss = torch.zeros((), device=DEVICE)
            loss = (
                args.l1_weight * l1
                + args.mse_weight * mse
                + args.feat_cycle_weight * feat_loss
                + args.tv_weight * tv
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), 5.0)
            optimizer.step()
            totals["loss"] += float(loss.item())
            totals["l1"] += float(l1.item())
            totals["mse"] += float(mse.item())
            totals["feat"] += float(feat_loss.item())
            totals["tv"] += float(tv.item())
            totals["residual"] += float(residual.abs().mean().item())
            steps += 1
        scheduler.step()
        row = {key: value / max(steps, 1) for key, value in totals.items()}
        row["epoch"] = epoch
        row["lr"] = float(scheduler.get_last_lr()[0])
        history.append(row)
        if epoch == 1 or epoch % args.save_every == 0 or epoch == args.epochs:
            torch.save(
                {
                    "state_dict": decoder.state_dict(),
                    "decoder_type": ResidualFeatureDecoder.decoder_type,
                    "feature_dim": args.feature_dim,
                    "img_size": args.img_size,
                    "in_channels": args.in_channels,
                    "residual_scale": args.residual_scale,
                    "args": vars(args),
                    "epoch": epoch,
                },
                model_path,
            )
        print(
            f"Epoch {epoch:03d}/{args.epochs} | loss={row['loss']:.5f} "
            f"l1={row['l1']:.5f} mse={row['mse']:.5f} feat={row['feat']:.5f} "
            f"tv={row['tv']:.5f} residual={row['residual']:.5f}"
        )

    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"Training complete in {(time.time() - start) / 60:.1f} min")
    print(f"Decoder saved to: {model_path}")
    print(f"History saved to: {history_path}")


if __name__ == "__main__":
    main()
