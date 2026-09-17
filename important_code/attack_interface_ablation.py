"""Label-knowledge and released-interface ablation for FAD.

The loss always uses the dataset label, as it would during target training.
`label_known` controls only what the attacker receives. This distinction is
essential for a valid unknown-label experiment.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from attack_decoder_multidataset import (
    compute_id_loss,
    compute_psnr,
    compute_ssim,
    load_decoder,
)
from experiment_utils import seed_everything, summarize_values
from train_decoder_multidataset import (
    DEVICE,
    build_dataset,
    build_transform,
    load_target_model,
    target_model_input,
)


CONDITIONS = {
    "known_wb_ce": {
        "label_known": True,
        "observed": "weight_and_bias_gradients",
        "label_smoothing": 0.0,
        "temperature": 1.0,
        "recovery": "exact_ratio_known_label_with_stable_fallback",
    },
    "unknown_wb_ce": {
        "label_known": False,
        "observed": "weight_and_bias_gradients",
        "label_smoothing": 0.0,
        "temperature": 1.0,
        "recovery": "exact_ratio_largest_absolute_bias_row",
    },
    "unknown_wb_label_smoothing_01": {
        "label_known": False,
        "observed": "weight_and_bias_gradients",
        "label_smoothing": 0.1,
        "temperature": 1.0,
        "recovery": "exact_ratio_largest_absolute_bias_row",
    },
    "unknown_wb_temperature_2": {
        "label_known": False,
        "observed": "weight_and_bias_gradients",
        "label_smoothing": 0.0,
        "temperature": 2.0,
        "recovery": "exact_ratio_largest_absolute_bias_row",
    },
    "known_weight_only": {
        "label_known": True,
        "observed": "weight_gradient_only",
        "label_smoothing": 0.0,
        "temperature": 1.0,
        "recovery": "target_row_direction_with_auxiliary_median_norm",
    },
    "unknown_weight_only": {
        "label_known": False,
        "observed": "weight_gradient_only",
        "label_smoothing": 0.0,
        "temperature": 1.0,
        "recovery": "largest_row_direction_with_auxiliary_median_norm",
    },
    "bias_only": {
        "label_known": False,
        "observed": "bias_gradient_only",
        "label_smoothing": 0.0,
        "temperature": 1.0,
        "recovery": "auxiliary_mean_feature",
    },
}


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def final_layer_gradients(target_model, feature, label, smoothing, temperature):
    weight = target_model.fc_layer.weight.detach().requires_grad_(True)
    bias = target_model.fc_layer.bias.detach().requires_grad_(True)
    logits = F.linear(feature.detach(), weight, bias) / temperature
    loss = F.cross_entropy(logits, label, label_smoothing=smoothing)
    grad_weight, grad_bias = torch.autograd.grad(loss, (weight, bias))
    return grad_weight.detach(), grad_bias.detach()


def exact_ratio(grad_weight, grad_bias, attacker_label, eps):
    abs_bias = grad_bias.abs()
    if attacker_label is not None:
        row = int(attacker_label)
        used_fallback = bool(abs_bias[row].item() <= eps)
        if used_fallback:
            row = int(abs_bias.argmax().item())
    else:
        row = int(abs_bias.argmax().item())
        used_fallback = False
    denominator = grad_bias[row]
    if denominator.abs().item() <= eps:
        raise RuntimeError("all observed bias-gradient rows are numerically zero")
    feature = (grad_weight[row] / denominator).view(1, -1)
    inferred_label = int(grad_bias.argmin().item())
    return feature, row, inferred_label, used_fallback


def weight_only(grad_weight, attacker_label, median_norm, eps):
    row_norms = torch.linalg.vector_norm(grad_weight, dim=1)
    row = (
        int(attacker_label)
        if attacker_label is not None
        else int(row_norms.argmax().item())
    )
    norm = row_norms[row]
    if norm.item() <= eps:
        raise RuntimeError("all observed weight-gradient rows are numerically zero")
    # Under ordinary CE, the target row coefficient is non-positive. For an
    # unknown label, the largest row norm identifies that target row because
    # |p_y-1| equals the sum of all non-target probabilities.
    direction = -grad_weight[row] / norm
    feature = direction.view(1, -1) * median_norm
    inferred_label = int(row_norms.argmax().item())
    return feature, row, inferred_label, False


def load_or_create_calibration(args, target_model):
    path = Path(args.calibration_path)
    expected = {
        "target_weight": str(Path(args.target_weight).expanduser()),
        "manifest_path": str(Path(args.manifest_path).expanduser()),
        "manifest_partition": args.calibration_partition,
        "samples": args.calibration_samples,
    }
    if path.is_file() and not args.recompute_calibration:
        cached = torch.load(path, map_location=DEVICE)
        if cached.get("metadata") != expected:
            raise RuntimeError(
                f"Calibration cache metadata differs; remove it or use --recompute_calibration: {path}"
            )
        print(f"[OK] loaded auxiliary calibration: {path}")
        return cached["mean_feature"].to(DEVICE), float(cached["median_norm"])

    transform = build_transform("celeba", args.img_size, args.in_channels)
    dataset = build_dataset(
        "celeba",
        args.data_root,
        "train",
        transform,
        args.calibration_samples,
        False,
        manifest_path=args.manifest_path,
        manifest_partition=args.calibration_partition,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.calibration_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    feature_sum = torch.zeros(args.feature_dim, dtype=torch.float64, device=DEVICE)
    norms, count = [], 0
    target_model.eval()
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(DEVICE, non_blocking=True)
            _, feature = target_model(target_model_input(images, "celeba"))
            feature_sum += feature.double().sum(dim=0)
            norms.extend(torch.linalg.vector_norm(feature.float(), dim=1).cpu().tolist())
            count += feature.size(0)
    mean_feature = (feature_sum / count).float().view(1, -1)
    median_norm = float(np.median(norms))
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "mean_feature": mean_feature.cpu(),
            "median_norm": median_norm,
            "feature_norm": summarize_values(norms, seed=args.seed),
            "metadata": expected,
        },
        path,
    )
    write_json(
        path.with_suffix(".json"),
        {
            "median_norm": median_norm,
            "feature_norm": summarize_values(norms, seed=args.seed),
            "metadata": expected,
        },
    )
    print(f"[OK] created auxiliary calibration from {count} images: {path}")
    return mean_feature.to(DEVICE), median_norm


def recover_for_condition(config, grad_weight, grad_bias, label, calibration, eps):
    mean_feature, median_norm = calibration
    attacker_label = label if config["label_known"] else None
    if config["observed"] == "weight_and_bias_gradients":
        return exact_ratio(grad_weight, grad_bias, attacker_label, eps)
    if config["observed"] == "weight_gradient_only":
        return weight_only(grad_weight, attacker_label, median_norm, eps)
    inferred_label = int(grad_bias.argmin().item())
    return mean_feature.clone(), None, inferred_label, False


def summarize_condition(config, records, invalid, requested, seed):
    metrics = {}
    for metric in (
        "psnr",
        "ssim",
        "id_loss",
        "relative_feature_error",
        "feature_cosine",
        "feature_norm_ratio",
    ):
        metrics[metric] = summarize_values([row[metric] for row in records], seed=seed)
    inferred = [float(row["inferred_label_correct"]) for row in records]
    return {
        "protocol": config,
        "requested_samples": requested,
        "evaluated_samples": len(records),
        "invalid_samples": len(invalid),
        "valid_rate": len(records) / requested,
        "label_inference_accuracy": (
            summarize_values(inferred, seed=seed) if not config["label_known"] else None
        ),
        "metrics": metrics,
        "per_sample": records,
        "invalid": invalid,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--target_weight", required=True)
    parser.add_argument("--decoder_path", required=True)
    parser.add_argument("--manifest_path", required=True)
    parser.add_argument("--manifest_partition", default="test", choices=["test"])
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--calibration_path", required=True)
    parser.add_argument("--calibration_partition", default="auxiliary", choices=["auxiliary"])
    parser.add_argument("--calibration_samples", type=int, default=30000)
    parser.add_argument("--calibration_batch_size", type=int, default=64)
    parser.add_argument("--recompute_calibration", action="store_true")
    parser.add_argument("--conditions", nargs="+", default=list(CONDITIONS))
    parser.add_argument("--test_samples", type=int, default=500)
    parser.add_argument("--img_size", type=int, default=64)
    parser.add_argument("--in_channels", type=int, default=3)
    parser.add_argument("--feature_dim", type=int, default=2048)
    parser.add_argument("--num_classes", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--epsilon", type=float, default=1e-12)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--save_raw_pairs", action="store_true")
    args = parser.parse_args()
    unknown = sorted(set(args.conditions) - set(CONDITIONS))
    if unknown:
        parser.error(f"Unknown conditions: {unknown}")

    seed_everything(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    target_model = load_target_model(
        args.target_weight, args.in_channels, args.num_classes, args.feature_dim, "vgg"
    )
    decoder = load_decoder(args.decoder_path, args.feature_dim, args.img_size, args.in_channels)
    calibration = load_or_create_calibration(args, target_model)

    transform = build_transform("celeba", args.img_size, args.in_channels)
    dataset = build_dataset(
        "celeba",
        args.data_root,
        "test",
        transform,
        args.test_samples,
        False,
        manifest_path=args.manifest_path,
        manifest_partition=args.manifest_partition,
    )
    if not getattr(dataset, "has_complete_labels", False):
        raise RuntimeError("Interface ablation requires complete manifest labels")
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)
    records = {name: [] for name in args.conditions}
    invalid = {name: [] for name in args.conditions}
    target_correct = []

    for index, (image, label_tensor) in enumerate(loader):
        image = image.to(DEVICE)
        label_tensor = label_tensor.to(DEVICE, dtype=torch.long)
        label = int(label_tensor.item())
        with torch.no_grad():
            logits, true_feature = target_model(target_model_input(image, "celeba"))
        target_correct.append(float(logits.argmax(1).item() == label))

        gradient_cache = {}
        for name in args.conditions:
            config = CONDITIONS[name]
            loss_key = (config["label_smoothing"], config["temperature"])
            if loss_key not in gradient_cache:
                gradient_cache[loss_key] = final_layer_gradients(
                    target_model,
                    true_feature,
                    label_tensor,
                    config["label_smoothing"],
                    config["temperature"],
                )
            grad_weight, grad_bias = gradient_cache[loss_key]
            try:
                recovered, row, inferred_label, used_fallback = recover_for_condition(
                    config,
                    grad_weight,
                    grad_bias,
                    label,
                    calibration,
                    args.epsilon,
                )
                recovered = recovered.to(DEVICE)
                with torch.no_grad():
                    reconstruction = decoder(recovered)
                true_norm = torch.linalg.vector_norm(true_feature).clamp_min(1e-12)
                recovered_norm = torch.linalg.vector_norm(recovered)
                record = {
                    "idx": index,
                    "dataset_label": label,
                    "target_prediction": int(logits.argmax(1).item()),
                    "target_correct": bool(logits.argmax(1).item() == label),
                    "recovered_row": row,
                    "inferred_label": inferred_label,
                    "inferred_label_correct": bool(inferred_label == label),
                    "used_stable_fallback": used_fallback,
                    "psnr": float(compute_psnr(reconstruction, image).item()),
                    "ssim": float(compute_ssim(reconstruction, image)),
                    "id_loss": float(
                        compute_id_loss(target_model, image, reconstruction, "celeba").item()
                    ),
                    "relative_feature_error": float(
                        torch.linalg.vector_norm(recovered - true_feature).div(true_norm).item()
                    ),
                    "feature_cosine": float(
                        F.cosine_similarity(recovered.float(), true_feature.float()).item()
                    ),
                    "feature_norm_ratio": float(recovered_norm.div(true_norm).item()),
                }
                records[name].append(record)
                if args.save_raw_pairs:
                    pair_dir = output_dir / name / "raw_pairs"
                    pair_dir.mkdir(parents=True, exist_ok=True)
                    save_image(
                        torch.clamp((image.detach().cpu() + 1.0) / 2.0, 0, 1),
                        pair_dir / f"{index:04d}_real.png",
                    )
                    save_image(
                        torch.clamp((reconstruction.detach().cpu() + 1.0) / 2.0, 0, 1),
                        pair_dir / f"{index:04d}_recon.png",
                    )
            except RuntimeError as exc:
                invalid[name].append({"idx": index, "dataset_label": label, "reason": str(exc)})
        if (index + 1) % 25 == 0 or index + 1 == len(dataset):
            print(f"processed {index + 1}/{len(dataset)}")

    summary = {
        "experiment": "FAD label-knowledge and released-interface ablation",
        "dataset": "celeba",
        "arch": "vgg",
        "target_weight": str(Path(args.target_weight).expanduser()),
        "decoder_path": str(Path(args.decoder_path).expanduser()),
        "manifest_path": str(Path(args.manifest_path).expanduser()),
        "manifest_partition": args.manifest_partition,
        "seed": args.seed,
        "target_accuracy": summarize_values(target_correct, seed=args.seed),
        "auxiliary_calibration": {
            "path": str(Path(args.calibration_path).expanduser()),
            "samples": args.calibration_samples,
            "partition": args.calibration_partition,
            "private_test_information_used": False,
        },
        "conditions": {
            name: summarize_condition(
                CONDITIONS[name], records[name], invalid[name], len(dataset), args.seed
            )
            for name in args.conditions
        },
        "args": vars(args),
    }
    write_json(output_dir / "summary.json", summary)
    print("=" * 88)
    for name, result in summary["conditions"].items():
        feature_error = result["metrics"]["relative_feature_error"]["mean"]
        psnr = result["metrics"]["psnr"]["mean"]
        label_acc = result["label_inference_accuracy"]
        label_text = "known" if label_acc is None else f"{label_acc['mean']:.4f}"
        print(
            f"{name:34s} valid={result['valid_rate']:.4f} "
            f"label={label_text:>6s} feat_err={feature_error:.3e} PSNR={psnr:.2f}"
        )
    print(f"summary_json: {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
