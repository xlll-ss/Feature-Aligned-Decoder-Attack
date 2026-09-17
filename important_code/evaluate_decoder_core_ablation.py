"""Evaluate decoder ablations under one fixed unknown-label W+b interface."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from attack_decoder_multidataset import compute_id_loss, compute_psnr, compute_ssim
from experiment_utils import seed_everything, summarize_values
from train_decoder_core_ablation import VARIANTS, build_decoder
from train_decoder_multidataset import (
    DEVICE,
    FeatureDecoder,
    build_dataset,
    build_transform,
    load_target_model,
    target_model_input,
)


def synchronize():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def load_checkpoint(path):
    try:
        return torch.load(path, map_location=DEVICE, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=DEVICE)


def load_decoder(path, requested_variant):
    checkpoint = load_checkpoint(path)
    state_dict = (
        checkpoint["state_dict"]
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint
        else checkpoint
    )
    checkpoint_variant = (
        checkpoint.get("decoder_ablation_variant")
        if isinstance(checkpoint, dict)
        else None
    )
    if checkpoint_variant and checkpoint_variant != requested_variant:
        raise RuntimeError(
            f"Checkpoint variant={checkpoint_variant}, requested={requested_variant}: {path}"
        )
    if checkpoint_variant:
        decoder = build_decoder(requested_variant, 2048, 64, 3).to(DEVICE)
    elif requested_variant == "full":
        decoder = FeatureDecoder(2048, 64, 3).to(DEVICE)
    else:
        raise RuntimeError(f"Ablation metadata is missing from checkpoint: {path}")
    decoder.load_state_dict(state_dict, strict=True)
    decoder.eval()
    for parameter in decoder.parameters():
        parameter.requires_grad_(False)
    metadata = {
        "checkpoint_variant": checkpoint_variant or "full_main_checkpoint",
        "variant_spec": (
            checkpoint.get("variant_spec")
            if isinstance(checkpoint, dict) and checkpoint.get("variant_spec")
            else VARIANTS["full"]
        ),
        "model_config": (
            checkpoint.get("model_config", {}) if isinstance(checkpoint, dict) else {}
        ),
        "training_protocol": (
            checkpoint.get("protocol") or checkpoint.get("args")
            if isinstance(checkpoint, dict)
            else None
        ),
        "training_epoch": (
            checkpoint.get("epoch") if isinstance(checkpoint, dict) else None
        ),
        "parameter_count": sum(parameter.numel() for parameter in decoder.parameters()),
    }
    return decoder, metadata


def final_layer_gradients(target, feature, label):
    weight = target.fc_layer.weight.detach().requires_grad_(True)
    bias = target.fc_layer.bias.detach().requires_grad_(True)
    logits = F.linear(feature.detach(), weight, bias)
    loss = F.cross_entropy(logits, label)
    grad_weight, grad_bias = torch.autograd.grad(loss, (weight, bias))
    return grad_weight.detach(), grad_bias.detach()


def recover_feature(grad_weight, grad_bias, epsilon):
    row = int(grad_bias.abs().argmax().item())
    denominator = grad_bias[row]
    if denominator.abs().item() <= epsilon:
        raise RuntimeError("all final-layer bias-gradient rows are numerically zero")
    return (grad_weight[row] / denominator).view(1, -1), row


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--target_weight", required=True)
    parser.add_argument("--decoder_path", required=True)
    parser.add_argument("--manifest_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--test_samples", type=int, default=500)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--epsilon", type=float, default=1e-12)
    parser.add_argument("--seed", type=int, default=2027)
    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)
    output_dir = Path(args.output_dir).expanduser()
    raw_dir = output_dir / "raw_pairs"
    raw_dir.mkdir(parents=True, exist_ok=True)
    for pattern in ("*_real.png", "*_recon.png"):
        for old_pair in raw_dir.glob(pattern):
            old_pair.unlink()
    for old_metric in ("summary.json", "lpips.json", "arcface.json"):
        (output_dir / old_metric).unlink(missing_ok=True)
    target = load_target_model(args.target_weight, 3, 0, 2048, "vgg")
    target.eval()
    decoder, decoder_metadata = load_decoder(args.decoder_path, args.variant)
    if args.warmup < 0:
        raise ValueError("--warmup must be non-negative")
    with torch.no_grad():
        warmup_feature = torch.zeros(1, 2048, device=DEVICE)
        for _ in range(args.warmup):
            decoder(warmup_feature)
    synchronize()
    dataset = build_dataset(
        "celeba",
        args.data_root,
        "test",
        build_transform("celeba", 64, 3),
        args.test_samples,
        False,
        manifest_path=args.manifest_path,
        manifest_partition="test",
    )
    if not getattr(dataset, "has_complete_labels", False):
        raise RuntimeError("Decoder ablation requires complete manifest test labels")
    sample_paths = [str(path) for path in getattr(dataset, "paths", [])]
    sample_hash = hashlib.sha256("\n".join(sample_paths).encode("utf-8")).hexdigest()
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
    )
    records, invalid = [], []
    for index, (image, label) in enumerate(loader):
        image = image.to(DEVICE)
        label = label.to(DEVICE, dtype=torch.long)
        with torch.no_grad():
            logits, true_feature = target(target_model_input(image, "celeba"))
        grad_weight, grad_bias = final_layer_gradients(target, true_feature, label)
        try:
            recovered_feature, row = recover_feature(
                grad_weight, grad_bias, args.epsilon
            )
            synchronize()
            started = time.perf_counter()
            with torch.no_grad():
                reconstruction = decoder(recovered_feature)
            synchronize()
            elapsed = time.perf_counter() - started
            with torch.no_grad():
                clean_reconstruction = decoder(true_feature)
            true_norm = torch.linalg.vector_norm(true_feature).clamp_min(1e-12)
            record = {
                "idx": index,
                "dataset_label": int(label.item()),
                "target_prediction": int(logits.argmax(1).item()),
                "target_correct": bool(logits.argmax(1).item() == label.item()),
                "recovered_row": row,
                "inferred_label": int(grad_bias.argmin().item()),
                "inferred_label_correct": bool(grad_bias.argmin().item() == label.item()),
                "relative_feature_error": float(
                    torch.linalg.vector_norm(recovered_feature - true_feature)
                    .div(true_norm)
                    .item()
                ),
                "clean_leaked_output_mse": float(
                    F.mse_loss(clean_reconstruction, reconstruction).item()
                ),
                "psnr": float(compute_psnr(reconstruction, image).item()),
                "ssim": float(compute_ssim(reconstruction, image)),
                "id_loss": float(
                    compute_id_loss(target, image, reconstruction, "celeba").item()
                ),
                "decoder_seconds": elapsed,
            }
            records.append(record)
            save_image(
                (image.detach().cpu() + 1.0).div(2).clamp(0, 1),
                raw_dir / f"{index:04d}_real.png",
            )
            save_image(
                (reconstruction.detach().cpu() + 1.0).div(2).clamp(0, 1),
                raw_dir / f"{index:04d}_recon.png",
            )
        except RuntimeError as exc:
            invalid.append(
                {"idx": index, "dataset_label": int(label.item()), "reason": str(exc)}
            )
        if (index + 1) % 25 == 0 or index + 1 == len(dataset):
            print(f"variant={args.variant} processed={index + 1}/{len(dataset)}")

    metrics = {}
    for metric in (
        "relative_feature_error",
        "clean_leaked_output_mse",
        "psnr",
        "ssim",
        "id_loss",
        "decoder_seconds",
    ):
        metrics[metric] = summarize_values(
            [record[metric] for record in records], seed=args.seed
        )
    training_protocol = decoder_metadata.get("training_protocol") or {}
    auxiliary_samples = training_protocol.get(
        "max_samples", training_protocol.get("offline_auxiliary_images")
    )
    auxiliary_partition = training_protocol.get(
        "manifest_partition", training_protocol.get("auxiliary_partition")
    )
    summary = {
        "experiment": "FAD decoder core ablation",
        "variant": args.variant,
        "seed": args.seed,
        "target_weight": str(Path(args.target_weight).expanduser()),
        "decoder_path": str(Path(args.decoder_path).expanduser()),
        "manifest_path": str(Path(args.manifest_path).expanduser()),
        "protocol": {
            "dataset": "celeba",
            "architecture": "vgg",
            "image_size": 64,
            "batch_size": 1,
            "decoder_timing_warmup_steps": args.warmup,
            "gradient_scope": ["fc_layer.weight", "fc_layer.bias"],
            "label_access": "unknown_not_required_for_feature_recovery",
            "test_selection": "first N sorted paths from immutable manifest test partition",
            "sample_set_sha256": sample_hash,
            "auxiliary_samples": auxiliary_samples,
            "auxiliary_partition": auxiliary_partition,
        },
        "decoder": decoder_metadata,
        "requested_samples": len(dataset),
        "evaluated_samples": len(records),
        "invalid_samples": len(invalid),
        "valid_rate": len(records) / len(dataset),
        "label_inference_accuracy": summarize_values(
            [float(record["inferred_label_correct"]) for record in records],
            seed=args.seed,
        ),
        "metrics": metrics,
        "per_sample": records,
        "invalid": invalid,
        "args": vars(args),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(
        f"{args.variant}: valid={summary['valid_rate']:.4f} "
        f"PSNR={metrics['psnr']['mean']:.4f} SSIM={metrics['ssim']['mean']:.6f}"
    )


if __name__ == "__main__":
    main()
