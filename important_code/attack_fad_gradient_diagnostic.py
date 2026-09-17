"""Diagnostic FAD refinement with constrained residuals and layerwise losses.

This script is deliberately separate from the primary zero-step FAD attack.
It tests whether additional observed gradients can improve an FAD
initialization without allowing the optimizer to move arbitrarily far from it.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from attack_decoder_multidataset import (
    compute_id_loss,
    compute_psnr,
    compute_ssim,
    load_decoder,
    recover_feat_from_fc_grad,
    save_pair,
)
from experiment_utils import seed_everything, summarize_values
from train_decoder_multidataset import (
    DEVICE,
    build_dataset,
    build_transform,
    load_target_model,
    target_model_input,
)


def total_variation(x):
    return (
        (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean()
        + (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()
    )


def select_parameters(model, scope):
    named = list(model.named_parameters())
    if scope == "classifier":
        selected = [(n, p) for n, p in named if n in {"fc_layer.weight", "fc_layer.bias"}]
    elif scope == "last_conv":
        selected = [(n, p) for n, p in named if n.startswith("layer5.")]
    elif scope == "last_two_conv":
        selected = [(n, p) for n, p in named if n.startswith(("layer4.", "layer5."))]
    elif scope == "full":
        selected = [(n, p) for n, p in named if p.requires_grad]
    else:
        raise ValueError(f"Unsupported gradient scope: {scope}")
    if not selected:
        raise RuntimeError(f"No parameters matched gradient scope {scope}")
    return selected


def gradients_for_image(model, image, label, parameters, dataset_name, create_graph):
    logits, _ = model(target_model_input(image, dataset_name))
    loss = CrossEntropyLoss()(logits, label)
    return torch.autograd.grad(
        loss,
        [p for _, p in parameters],
        retain_graph=create_graph,
        create_graph=create_graph,
        allow_unused=False,
    )


def backward_gradients_for_image(model, image, label, parameters, dataset_name):
    """Independent reference path used for the x_gt gradient consistency check."""
    model.zero_grad(set_to_none=True)
    logits, _ = model(target_model_input(image, dataset_name))
    loss = CrossEntropyLoss()(logits, label)
    loss.backward()
    values = []
    for _, parameter in parameters:
        if parameter.grad is None:
            raise RuntimeError("Missing gradient in backward reference path")
        values.append(parameter.grad.detach().clone())
    model.zero_grad(set_to_none=True)
    return tuple(values)


def layerwise_distance(candidate, reference, names, mode, eps=1e-12):
    terms = []
    used = []
    for name, current, target in zip(names, candidate, reference):
        current_flat = current.reshape(current.shape[0], -1)
        target_flat = target.detach().reshape(target.shape[0], -1)
        target_norm = target_flat.norm(dim=1)
        valid = target_norm > eps
        if not bool(valid.any()):
            continue
        if mode == "cosine":
            cos = F.cosine_similarity(current_flat[valid], target_flat[valid], dim=1)
            term = (1.0 - cos).mean()
        elif mode == "relative_mse":
            scale = target_norm[valid].reshape(-1, 1).clamp_min(eps)
            term = ((current_flat[valid] - target_flat[valid]) / scale).pow(2).mean()
        else:
            raise ValueError(f"Unsupported distance mode: {mode}")
        terms.append(term)
        used.append(name)
    if not terms:
        return torch.zeros((), device=DEVICE), used
    return torch.stack(terms).mean(), used


def constrained_candidate(initialization, delta, residual_eps):
    return torch.clamp(initialization + residual_eps * torch.tanh(delta), -1.0, 1.0)


def refine_image(
    target_model,
    image,
    label,
    initialization,
    parameters,
    dataset_name,
    steps,
    lr,
    prior_weight,
    tv_weight,
    residual_eps,
    distance_mode,
    save_steps,
):
    names = [name for name, _ in parameters]
    observed = tuple(
        grad.detach()
        for grad in gradients_for_image(
            target_model, image, label, parameters, dataset_name, create_graph=False
        )
    )
    gt_grads = backward_gradients_for_image(
        target_model, image, label, parameters, dataset_name
    )
    gt_loss, gt_used = layerwise_distance(gt_grads, observed, names, distance_mode)
    delta = torch.zeros_like(initialization, requires_grad=True)
    optimizer = torch.optim.Adam([delta], lr=lr)
    checkpoints = {0: initialization.detach().clone()}
    trace = [{
        "step": 0,
        "gradient_loss": 0.0,
        "prior_loss": 0.0,
        "tv_loss": 0.0,
        "used_layers": len(gt_used),
    }]
    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        target_model.zero_grad(set_to_none=True)
        candidate = constrained_candidate(initialization, delta, residual_eps)
        current = gradients_for_image(
            target_model, candidate, label, parameters, dataset_name, create_graph=True
        )
        grad_loss, used_layers = layerwise_distance(
            current, observed, names, distance_mode
        )
        prior_loss = F.mse_loss(candidate, initialization)
        tv_loss = total_variation(candidate)
        objective = grad_loss + prior_weight * prior_loss + tv_weight * tv_loss
        objective.backward()
        torch.nn.utils.clip_grad_norm_([delta], max_norm=5.0)
        optimizer.step()
        target_model.zero_grad(set_to_none=True)
        with torch.no_grad():
            updated = constrained_candidate(initialization, delta, residual_eps)
        if step in save_steps:
            checkpoints[step] = updated.detach().clone()
        trace.append({
            "step": step,
            "gradient_loss": float(grad_loss.detach().cpu().item()),
            "prior_loss": float(prior_loss.detach().cpu().item()),
            "tv_loss": float(tv_loss.detach().cpu().item()),
            "used_layers": len(used_layers),
        })
    if steps not in checkpoints:
        with torch.no_grad():
            checkpoints[steps] = constrained_candidate(
                initialization, delta, residual_eps
            ).detach().clone()
    return checkpoints, trace, float(gt_loss.detach().cpu().item())


def save_raw_pair(real, recon, output_dir, name):
    raw_dir = Path(output_dir) / "raw_pairs"
    raw_dir.mkdir(parents=True, exist_ok=True)
    save_image(torch.clamp((real.detach().cpu() + 1) / 2, 0, 1), raw_dir / f"{name}_real.png")
    save_image(torch.clamp((recon.detach().cpu() + 1) / 2, 0, 1), raw_dir / f"{name}_recon.png")


def parse_steps(value):
    steps = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    if not steps or steps[0] != 0:
        raise ValueError("--save_steps must include 0")
    if any(step < 0 for step in steps):
        raise ValueError("--save_steps cannot contain negative values")
    return steps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["celeba", "cifar10", "fashionmnist", "folder", "imagefolder"], required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--target_weight", required=True)
    parser.add_argument("--decoder_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--arch", choices=["vgg", "resnet18"], default="vgg")
    parser.add_argument("--img_size", type=int, default=64)
    parser.add_argument("--in_channels", type=int, default=3)
    parser.add_argument("--num_classes", type=int, default=0)
    parser.add_argument("--feature_dim", type=int, default=2048)
    parser.add_argument("--test_samples", type=int, default=20)
    parser.add_argument("--max_eval_scan", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--label_source", choices=["pred", "dataset"], default="dataset")
    parser.add_argument("--gradient_scope", choices=["classifier", "last_conv", "last_two_conv", "full"], default="last_conv")
    parser.add_argument("--distance", choices=["cosine", "relative_mse"], default="cosine")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--save_steps", default="0,1,5,10,25,50,100")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--prior_weight", type=float, default=0.1)
    parser.add_argument("--tv_weight", type=float, default=0.001)
    parser.add_argument("--residual_eps", type=float, default=0.1)
    parser.add_argument("--save_pairs", type=int, default=20)
    parser.add_argument("--save_raw_pairs", action="store_true")
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--manifest_path", default="")
    parser.add_argument("--manifest_partition", choices=["", "auxiliary", "validation", "test"], default="")
    args = parser.parse_args()
    save_steps = parse_steps(args.save_steps)
    if max(save_steps) > args.steps:
        raise ValueError("--save_steps cannot exceed --steps")
    seed_everything(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    transform = build_transform(args.dataset, args.img_size, args.in_channels)
    scan_limit = args.max_eval_scan if args.max_eval_scan > 0 else max(args.test_samples * 10, args.test_samples)
    dataset = build_dataset(
        args.dataset,
        args.data_root,
        "test",
        transform,
        scan_limit,
        False,
        manifest_path=args.manifest_path,
        manifest_partition=args.manifest_partition,
    )
    if args.label_source == "dataset" and hasattr(dataset, "has_complete_labels") and not dataset.has_complete_labels:
        raise RuntimeError("Dataset-label mode requires complete manifest labels")
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)
    target_model = load_target_model(
        args.target_weight, args.in_channels, args.num_classes, args.feature_dim, args.arch
    )
    target_model.eval()
    for parameter in target_model.parameters():
        parameter.requires_grad_(True)
    fc_parameters = [
        (name, parameter)
        for name, parameter in target_model.named_parameters()
        if name in {"fc_layer.weight", "fc_layer.bias"}
    ]
    parameters = select_parameters(target_model, args.gradient_scope)
    decoder = load_decoder(args.decoder_path, args.feature_dim, args.img_size, args.in_channels)
    decoder.eval()

    checkpoint_metrics = {step: {"psnr": [], "ssim": [], "id_loss": [], "gradient_loss": []} for step in save_steps}
    per_sample = []
    saturated = []
    gt_losses = []
    for source_idx, (image, label) in enumerate(loader):
        if len(per_sample) >= args.test_samples:
            break
        image, label = image.to(DEVICE), label.to(DEVICE)
        with torch.no_grad():
            logits, _ = target_model(target_model_input(image, args.dataset))
            predicted = logits.argmax(dim=1)
        target_label = label if args.label_source == "dataset" else predicted
        try:
            fc_grads = gradients_for_image(
                target_model, image, target_label, fc_parameters, args.dataset, create_graph=False
            )
            leaked_feature, recovered_class, denominator = recover_feat_from_fc_grad(
                fc_grads[0].detach(), fc_grads[1].detach(), target_label
            )
        except RuntimeError as exc:
            saturated.append({"source_idx": source_idx, "reason": str(exc)})
            continue
        with torch.no_grad():
            initialization = decoder(leaked_feature.to(DEVICE))
        checkpoints, trace, gt_loss = refine_image(
            target_model,
            image,
            target_label,
            initialization,
            parameters,
            args.dataset,
            args.steps,
            args.lr,
            args.prior_weight,
            args.tv_weight,
            args.residual_eps,
            args.distance,
            save_steps,
        )
        gt_losses.append(gt_loss)
        sample_metrics = {}
        for step, reconstruction in sorted(checkpoints.items()):
            with torch.no_grad():
                psnr = float(compute_psnr(reconstruction, image).item())
                ssim = float(compute_ssim(reconstruction, image))
                id_loss = float(compute_id_loss(target_model, image, reconstruction, args.dataset).item())
            sample_metrics[str(step)] = {"psnr": psnr, "ssim": ssim, "id_loss": id_loss}
            if step in checkpoint_metrics:
                checkpoint_metrics[step]["psnr"].append(psnr)
                checkpoint_metrics[step]["ssim"].append(ssim)
                checkpoint_metrics[step]["id_loss"].append(id_loss)
            if len(per_sample) < args.save_pairs:
                step_dir = output_dir / f"step_{step:03d}"
                save_pair(
                    image,
                    reconstruction,
                    step_dir,
                    len(per_sample),
                    {"psnr": psnr, "ssim": ssim, "id_loss": id_loss},
                )
                if args.save_raw_pairs:
                    save_raw_pair(image, reconstruction, step_dir, f"{source_idx:04d}")
        for item in trace:
            if item["step"] in checkpoint_metrics:
                checkpoint_metrics[item["step"]]["gradient_loss"].append(item["gradient_loss"])
        per_sample.append({
            "source_idx": source_idx,
            "target_label": int(target_label.item()),
            "pred_label": int(predicted.item()),
            "recovered_class": int(recovered_class),
            "recovered_denom": denominator,
            "ground_truth_gradient_loss": gt_loss,
            "metrics_by_step": sample_metrics,
            "trace": trace,
        })
        initial = sample_metrics["0"]["psnr"]
        final = sample_metrics[str(args.steps)]["psnr"]
        print(
            f"Sample {len(per_sample):03d} src_idx={source_idx:04d} | "
            f"step0 {initial:.2f} -> step{args.steps} {final:.2f} dB"
        )

    if not per_sample:
        raise RuntimeError("No valid samples were recovered")
    summary = {
        "dataset": args.dataset,
        "method": "FAD constrained residual gradient diagnostic",
        "gradient_scope": args.gradient_scope,
        "distance": args.distance,
        "residual_eps": args.residual_eps,
        "test_samples": len(per_sample),
        "requested_test_samples": args.test_samples,
        "scanned_samples": len(per_sample) + len(saturated),
        "saturated_samples": len(saturated),
        "steps": args.steps,
        "save_steps": save_steps,
        "ground_truth_gradient_loss": summarize_values(gt_losses, seed=args.seed),
        "checkpoints": {
            str(step): {
                key: summarize_values(values, seed=args.seed)
                for key, values in metrics.items()
                if values
            }
            for step, metrics in checkpoint_metrics.items()
        },
        "args": vars(args),
        "per_sample": per_sample,
        "saturated": saturated,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("=" * 78)
    print("FAD constrained residual diagnostic")
    print("=" * 78)
    print(f"gradient_scope: {args.gradient_scope}")
    print(f"distance: {args.distance}")
    print(f"residual_eps: {args.residual_eps}")
    print(f"ground-truth gradient loss: {summary['ground_truth_gradient_loss']['mean']:.3e}")
    for step in save_steps:
        row = summary["checkpoints"].get(str(step), {})
        if row:
            print(f"step {step:3d}: PSNR={row['psnr']['mean']:.2f} SSIM={row['ssim']['mean']:.4f}")
    print(f"summary_json: {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
