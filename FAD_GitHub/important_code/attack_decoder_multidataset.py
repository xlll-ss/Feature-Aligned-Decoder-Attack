import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from skimage.metrics import structural_similarity as calc_ssim
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from train_decoder_multidataset import (
    DEVICE,
    FeatureDecoder,
    build_dataset,
    build_transform,
    load_target_model,
    target_model_input,
)
from residual_decoder import ResidualFeatureDecoder
from experiment_utils import seed_everything, summarize_values


def recover_feat_from_fc_grad(grad_w, grad_b, target_label, eps=1e-12):
    """Recover the penultimate feature from the final FC-layer gradient.

    For a linear classifier, every class row satisfies:
        grad_w[j] = grad_b[j] * feature
    The target-class bias gradient can become numerically zero when the
    classifier is extremely confident, so we fall back to the row with the
    largest absolute bias gradient.
    """
    y = int(target_label.view(-1)[0].item())
    abs_grad_b = grad_b.abs()
    recovered_class = y
    if abs_grad_b[y].item() < eps:
        recovered_class = int(abs_grad_b.argmax().item())

    denom = grad_b[recovered_class]
    if torch.abs(denom).item() < eps:
        raise RuntimeError(
            "All final-layer bias gradients are almost zero. "
            "The CE gradient is numerically saturated, so the final-layer "
            "linear leakage signal is unavailable for this sample."
        )
    feat = (grad_w[recovered_class] / denom).view(1, -1)
    return feat, recovered_class, float(denom.detach().cpu().item())


def recover_feat_from_fc_grad_robust(grad_w, grad_b, target_label, ridge=1e-12):
    """Estimate the feature by regressing every FC row jointly.

    For an unnoised single-example CE gradient, ``grad_w[j] = grad_b[j] * h``
    for every class row ``j``.  The original estimator uses one row and is
    therefore fragile when quantization or noise perturbs that row.  The
    least-squares estimator below uses all rows:

        h = sum_j grad_b[j] grad_w[j] / sum_j grad_b[j]^2.

    This is a controlled robustification, not an assumption that removes the
    need for a full-rank coded observation matrix.
    """
    bias = grad_b.reshape(-1).float()
    weight = grad_w.reshape(bias.numel(), -1).float()
    denom = bias.square().sum().clamp_min(ridge)
    feat = (bias.unsqueeze(1) * weight).sum(dim=0) / denom
    y = int(target_label.reshape(-1)[0].item())
    recovered_class = y if 0 <= y < bias.numel() else int(bias.abs().argmax().item())
    return feat.view(1, -1), recovered_class, float(denom.detach().cpu().item())


def recover_feat_from_fc_grad_tls(grad_w, grad_b, target_label, eps=1e-12):
    """Total-least-squares recovery using the rank-one FC-gradient model.

    For one example, ``[grad_W | grad_b] = delta[:, None] @ [h, 1]``.
    Quantization/noise affects both columns, so ordinary regression is biased.
    The dominant right singular vector of the concatenated matrix gives the
    least-squares rank-one direction without treating ``grad_b`` as exact.
    """
    weight = grad_w.reshape(grad_w.shape[0], -1).float()
    bias = grad_b.reshape(-1).float().unsqueeze(1)
    matrix = torch.cat([weight, bias], dim=1)
    _, _, vh = torch.linalg.svd(matrix, full_matrices=False)
    direction = vh[0]
    denom = direction[-1]
    if torch.abs(denom).item() < eps:
        raise RuntimeError("TLS bias component is numerically zero")
    feat = direction[:-1] / denom
    y = int(target_label.reshape(-1)[0].item())
    recovered_class = y if 0 <= y < matrix.shape[0] else int(bias[:, 0].abs().argmax().item())
    return feat.view(1, -1), recovered_class, float(denom.detach().cpu().item())


def compute_psnr(fake, real):
    fake = torch.clamp((fake + 1) / 2, 0, 1)
    real = torch.clamp((real + 1) / 2, 0, 1)
    mse = F.mse_loss(fake, real)
    return 10 * torch.log10(1.0 / (mse + 1e-8))


def compute_ssim(fake, real):
    fake = torch.clamp((fake + 1) / 2, 0, 1)
    real = torch.clamp((real + 1) / 2, 0, 1)
    fake_np = fake.squeeze(0).detach().cpu().permute(1, 2, 0).numpy()
    real_np = real.squeeze(0).detach().cpu().permute(1, 2, 0).numpy()
    if fake_np.shape[2] == 1:
        fake_np = fake_np[:, :, 0]
        real_np = real_np[:, :, 0]
        return calc_ssim(real_np, fake_np, data_range=1)
    try:
        return calc_ssim(real_np, fake_np, data_range=1, channel_axis=2)
    except TypeError:
        return calc_ssim(real_np, fake_np, data_range=1, multichannel=True)


def compute_id_loss(target_model, real, fake, dataset):
    with torch.no_grad():
        _, real_feat = target_model(target_model_input(real, dataset))
        _, fake_feat = target_model(target_model_input(fake, dataset))
    return F.mse_loss(real_feat.float(), fake_feat.float())


def tensor_to_pil(x):
    x = torch.clamp((x.detach().cpu()[0] + 1) / 2, 0, 1)
    if x.shape[0] == 1:
        arr = (x.squeeze(0).numpy() * 255).astype(np.uint8)
        return Image.fromarray(arr, mode="L").convert("RGB")
    arr = (x.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)


def save_pair(real, fake, output_dir, idx, metrics):
    pair_dir = Path(output_dir) / "comparison_pairs"
    pair_dir.mkdir(parents=True, exist_ok=True)
    real_img = tensor_to_pil(real)
    fake_img = tensor_to_pil(fake)
    w, h = real_img.size
    canvas = Image.new("RGB", (w * 2, h + 56), "white")
    canvas.paste(real_img, (0, 0))
    canvas.paste(fake_img, (w, 0))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except OSError:
        font = ImageFont.load_default()
    draw.text((8, h + 8), "Original", fill=(20, 20, 20), font=font)
    draw.text((w + 8, h + 8), "Ours Decoder", fill=(20, 20, 20), font=font)
    draw.text(
        (8, h + 30),
        f"PSNR {metrics['psnr']:.2f} | SSIM {metrics['ssim']:.4f} | ID {metrics['id_loss']:.3f}",
        fill=(20, 20, 20),
        font=font,
    )
    canvas.save(pair_dir / f"comparison_{idx:04d}.png")


def load_decoder(path, feature_dim, img_size, in_channels):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Decoder checkpoint not found: {path}")
    ckpt = torch.load(path, map_location=DEVICE)
    decoder_type = ckpt.get("decoder_type", "base") if isinstance(ckpt, dict) else "base"
    if decoder_type == ResidualFeatureDecoder.decoder_type:
        decoder = ResidualFeatureDecoder(
            feature_dim,
            img_size,
            in_channels,
            residual_scale=float(ckpt.get("residual_scale", 0.1)),
        ).to(DEVICE)
    else:
        decoder = FeatureDecoder(feature_dim, img_size, in_channels).to(DEVICE)
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    decoder.load_state_dict(state, strict=True)
    decoder.eval()
    for param in decoder.parameters():
        param.requires_grad = False
    return decoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["celeba", "cifar10", "fashionmnist", "folder", "imagefolder"], required=True)
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--target_weight", type=str, required=True)
    parser.add_argument("--decoder_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--arch", choices=["vgg", "resnet18"], default="vgg")
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--img_size", type=int, default=128)
    parser.add_argument("--in_channels", type=int, default=3)
    parser.add_argument("--num_classes", type=int, default=10, help="Set to 0 to infer from target checkpoint.")
    parser.add_argument("--feature_dim", type=int, default=2048)
    parser.add_argument("--test_samples", type=int, default=100)
    parser.add_argument("--max_eval_scan", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--label_source", choices=["pred", "dataset"], default="pred")
    parser.add_argument("--grad_label_smoothing", type=float, default=0.0)
    parser.add_argument("--grad_temperature", type=float, default=1.0)
    parser.add_argument("--save_pairs", type=int, default=20)
    parser.add_argument("--save_raw_pairs", action="store_true",
                        help="Save separate real/reconstruction PNGs for independent metrics.")
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--manifest_path", type=str, default="")
    parser.add_argument(
        "--manifest_partition",
        choices=["", "auxiliary", "validation", "test"],
        default="",
        help="For flat CelebA/folder evaluation, use test.",
    )
    args = parser.parse_args()

    seed_everything(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("Feature decoder attack for cross-dataset main comparison")
    print("=" * 78)
    print(f"dataset: {args.dataset}")
    print(f"arch: {args.arch}")
    print(f"device: {DEVICE}")
    print(f"test_samples: {args.test_samples}")
    print(f"target_weight: {args.target_weight}")
    print(f"decoder_path: {args.decoder_path}")
    print(f"output_dir: {args.output_dir}")
    print("attack_steps: 0")
    print("n_restarts: 0")
    print(f"label_source: {args.label_source}")
    print(f"grad_label_smoothing: {args.grad_label_smoothing}")
    print(f"grad_temperature: {args.grad_temperature}")
    print("=" * 78)

    transform = build_transform(args.dataset, args.img_size, args.in_channels)
    max_eval_scan = args.max_eval_scan if args.max_eval_scan > 0 else max(args.test_samples * 10, args.test_samples)
    dataset = build_dataset(
        args.dataset,
        args.data_root,
        args.split,
        transform,
        max_eval_scan,
        args.download,
        manifest_path=args.manifest_path,
        manifest_partition=args.manifest_partition,
    )
    if args.label_source == "dataset" and hasattr(dataset, "has_complete_labels") and not dataset.has_complete_labels:
        raise RuntimeError(
            f"Dataset-label mode requires complete manifest labels; missing={dataset.missing_label_count}"
        )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)
    target_model = load_target_model(args.target_weight, args.in_channels, args.num_classes, args.feature_dim, args.arch)
    for param in target_model.parameters():
        param.requires_grad = True
    decoder = load_decoder(args.decoder_path, args.feature_dim, args.img_size, args.in_channels)

    all_psnr, all_ssim, all_id, all_feature_error = [], [], [], []
    per_sample = []
    saturated_samples = []
    for idx, (img, label) in enumerate(loader):
        if len(all_psnr) >= args.test_samples:
            break
        img = img.to(DEVICE)
        label = label.to(DEVICE)
        model_img = target_model_input(img, args.dataset)

        with torch.no_grad():
            logits, true_feature = target_model(model_img)
            pred_label = logits.argmax(dim=1)
        target_label = label if args.label_source == "dataset" else pred_label

        logits, _ = target_model(model_img)
        logits_for_grad = logits / max(args.grad_temperature, 1e-6)
        loss_cls = CrossEntropyLoss(label_smoothing=args.grad_label_smoothing)(logits_for_grad, target_label)
        grad_w, grad_b = torch.autograd.grad(
            loss_cls,
            (target_model.fc_layer.weight, target_model.fc_layer.bias),
            retain_graph=False,
            create_graph=False,
        )
        try:
            leaked_feat, recovered_class, recovered_denom = recover_feat_from_fc_grad(
                grad_w.detach(),
                grad_b.detach(),
                target_label,
            )
        except RuntimeError as exc:
            saturated_samples.append({
                "idx": idx,
                "target_label": int(target_label.item()),
                "pred_label": int(pred_label.item()),
                "reason": str(exc),
            })
            if len(saturated_samples) <= 20:
                print(
                    f"Skip sample {idx:04d}: saturated final-layer gradient "
                    f"(target={int(target_label.item())}, pred={int(pred_label.item())})"
                )
            continue
        leaked_feat = leaked_feat.to(DEVICE)
        relative_feature_error = (
            torch.linalg.vector_norm(leaked_feat - true_feature)
            / torch.clamp(torch.linalg.vector_norm(true_feature), min=1e-12)
        ).item()
        with torch.no_grad():
            fake = decoder(leaked_feat)

        if args.save_raw_pairs:
            raw_dir = output_dir / "raw_pairs"
            raw_dir.mkdir(parents=True, exist_ok=True)
            save_image(torch.clamp((img.detach().cpu() + 1) / 2, 0, 1), raw_dir / f"{idx:04d}_real.png")
            save_image(torch.clamp((fake.detach().cpu() + 1) / 2, 0, 1), raw_dir / f"{idx:04d}_recon.png")

        metrics = {
            "idx": idx,
            "psnr": float(compute_psnr(fake, img).item()),
            "ssim": float(compute_ssim(fake, img)),
            "id_loss": float(compute_id_loss(target_model, img, fake, args.dataset).item()),
            "target_label": int(target_label.item()),
            "pred_label": int(pred_label.item()),
            "recovered_class": int(recovered_class),
            "recovered_denom": recovered_denom,
            "relative_feature_error": float(relative_feature_error),
            "classification_correct": (
                bool(pred_label.item() == label.item()) if args.label_source == "dataset" else None
            ),
        }
        all_psnr.append(metrics["psnr"])
        all_ssim.append(metrics["ssim"])
        all_id.append(metrics["id_loss"])
        all_feature_error.append(metrics["relative_feature_error"])
        per_sample.append(metrics)

        print(
            f"Sample {len(all_psnr):03d} src_idx={idx:04d} | PSNR {metrics['psnr']:.2f} | "
            f"SSIM {metrics['ssim']:.4f} | ID {metrics['id_loss']:.3f}"
        )
        if len(all_psnr) <= args.save_pairs:
            save_pair(img, fake, output_dir, len(all_psnr) - 1, metrics)

    if not all_psnr:
        raise RuntimeError(
            "No valid samples were recovered. Try --grad_label_smoothing 0.001 "
            "or --grad_temperature 2.0 to reduce numerical saturation."
        )
    summary = {
        "dataset": args.dataset,
        "arch": args.arch,
        "test_samples": len(all_psnr),
        "requested_test_samples": args.test_samples,
        "scanned_samples": len(all_psnr) + len(saturated_samples),
        "saturated_samples": len(saturated_samples),
        "attack_steps": 0,
        "n_restarts": 0,
        "method": "feature-aligned decoder",
        "avg_psnr": float(np.mean(all_psnr)),
        "max_psnr": float(np.max(all_psnr)),
        "min_psnr": float(np.min(all_psnr)),
        "avg_ssim": float(np.mean(all_ssim)),
        "avg_id_loss": float(np.mean(all_id)),
        "relative_feature_error": summarize_values(all_feature_error, seed=args.seed),
        "psnr": summarize_values(all_psnr, seed=args.seed),
        "ssim": summarize_values(all_ssim, seed=args.seed),
        "id_loss": summarize_values(all_id, seed=args.seed),
        "target_accuracy": (
            float(np.mean([m["classification_correct"] for m in per_sample]))
            if args.label_source == "dataset"
            else None
        ),
        "args": vars(args),
        "per_sample": per_sample,
        "skipped_saturated": saturated_samples,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n" + "=" * 78)
    print("Attack Summary")
    print("=" * 78)
    print(f"dataset: {summary['dataset']}")
    print(f"test_samples: {summary['test_samples']} / requested {summary['requested_test_samples']}")
    print(f"scanned_samples: {summary['scanned_samples']}")
    print(f"saturated_samples: {summary['saturated_samples']}")
    print("attack_steps: 0")
    print("n_restarts: 0")
    print(f"avg PSNR: {summary['avg_psnr']:.2f} dB")
    print(f"max PSNR: {summary['max_psnr']:.2f} dB")
    print(f"min PSNR: {summary['min_psnr']:.2f} dB")
    print(f"avg SSIM: {summary['avg_ssim']:.4f}")
    print(f"avg ID-Loss: {summary['avg_id_loss']:.3f}")
    if summary.get("target_accuracy") is not None:
        print(f"target accuracy: {summary['target_accuracy']:.4f}")
    print(
        f"PSNR 95% CI: [{summary['psnr']['ci95']['lower']:.2f}, "
        f"{summary['psnr']['ci95']['upper']:.2f}]"
    )
    print(f"relative feature error: {summary['relative_feature_error']['mean']:.3e}")
    print(f"summary_json: {output_dir / 'summary.json'}")
    print("=" * 78)


if __name__ == "__main__":
    main()
