"""Evaluate target-model utility on an immutable experiment manifest."""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from experiment_utils import seed_everything, summarize_values
from train_decoder_multidataset import (
    DEVICE,
    build_dataset,
    build_transform,
    load_target_model,
    target_model_input,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["celeba", "cifar10", "fashionmnist", "folder", "imagefolder"], required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--target_weight", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--arch", choices=["vgg", "resnet18"], default="vgg")
    parser.add_argument("--img_size", type=int, default=64)
    parser.add_argument("--in_channels", type=int, default=3)
    parser.add_argument("--num_classes", type=int, default=0)
    parser.add_argument("--feature_dim", type=int, default=2048)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--manifest_path", default="")
    parser.add_argument("--manifest_partition", choices=["", "auxiliary", "validation", "test"], default="test")
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args()

    seed_everything(args.seed)
    transform = build_transform(args.dataset, args.img_size, args.in_channels)
    dataset = build_dataset(
        args.dataset,
        args.data_root,
        "test",
        transform,
        args.max_samples,
        False,
        manifest_path=args.manifest_path,
        manifest_partition=args.manifest_partition if args.manifest_path else "",
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    model = load_target_model(
        args.target_weight,
        args.in_channels,
        args.num_classes,
        args.feature_dim,
        args.arch,
    )
    model.eval()
    correct, losses, confidences = [], [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True, dtype=torch.long)
            logits, _ = model(target_model_input(images, args.dataset))
            batch_losses = F.cross_entropy(logits, labels, reduction="none")
            probabilities = F.softmax(logits, dim=1)
            predictions = logits.argmax(dim=1)
            correct.extend((predictions == labels).float().cpu().tolist())
            losses.extend(batch_losses.cpu().tolist())
            confidences.extend(probabilities.max(dim=1).values.cpu().tolist())

    summary = {
        "dataset": args.dataset,
        "target_weight": str(Path(args.target_weight).expanduser()),
        "samples": len(correct),
        "accuracy": summarize_values(correct, seed=args.seed),
        "cross_entropy": summarize_values(losses, seed=args.seed),
        "max_confidence": summarize_values(confidences, seed=args.seed),
        "args": vars(args),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"samples: {summary['samples']}")
    print(f"accuracy: {summary['accuracy']['mean']:.4f}")
    print(f"cross_entropy: {summary['cross_entropy']['mean']:.4f}")
    print(f"max_confidence: {summary['max_confidence']['mean']:.4f}")
    print(f"summary_json: {output}")


if __name__ == "__main__":
    main()
