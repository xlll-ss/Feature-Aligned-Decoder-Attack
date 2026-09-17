import argparse
import copy
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from attack_decoder_multidataset import (
    compute_id_loss,
    compute_psnr,
    compute_ssim,
    load_decoder,
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


def final_layer_gradient(model, images, labels):
    logits, features = model(images)
    loss = F.cross_entropy(logits, labels)
    grad_w, grad_b = torch.autograd.grad(
        loss,
        (model.fc_layer.weight, model.fc_layer.bias),
        retain_graph=False,
        create_graph=False,
    )
    return grad_w.detach(), grad_b.detach(), features.detach(), float(loss.item())


def final_layer_model_delta(model, images, labels, steps, lr, momentum, weight_decay, bn_mode="train"):
    local_model = copy.deepcopy(model)
    if bn_mode == "train":
        local_model.train()
    elif bn_mode == "eval":
        local_model.eval()
    else:
        raise ValueError(f"Unsupported bn_mode: {bn_mode}")
    for parameter in local_model.parameters():
        parameter.requires_grad = True
    optimizer = torch.optim.SGD(
        local_model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
    )
    losses = []
    for _ in range(steps):
        logits, _ = local_model(images)
        loss = F.cross_entropy(logits, labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))

    # global - local has the same sign as the accumulated gradient for plain SGD.
    grad_w_equivalent = (model.fc_layer.weight.detach() - local_model.fc_layer.weight.detach()).clone()
    grad_b_equivalent = (model.fc_layer.bias.detach() - local_model.fc_layer.bias.detach()).clone()
    with torch.no_grad():
        _, features = model(images)
    del local_model
    return grad_w_equivalent, grad_b_equivalent, features.detach(), float(np.mean(losses))


def perturb_observation(
    grad_w,
    grad_b,
    clip_norm=0.0,
    noise_std=0.0,
    noise_multiplier=0.0,
    quantization_bits=0,
):
    """Apply server-observable clipping, noise, and symmetric quantization."""
    grad_w = grad_w.clone()
    grad_b = grad_b.clone()
    if clip_norm > 0:
        total_norm = torch.sqrt(torch.sum(grad_w.square()) + torch.sum(grad_b.square()))
        scale = min(1.0, float(clip_norm) / max(float(total_norm.item()), 1e-12))
        grad_w.mul_(scale)
        grad_b.mul_(scale)

    effective_noise_std = float(noise_std)
    if noise_multiplier > 0:
        if clip_norm <= 0:
            raise ValueError("--noise_multiplier requires --clip_grad_norm > 0")
        effective_noise_std += float(noise_multiplier) * float(clip_norm)
    if effective_noise_std > 0:
        grad_w.add_(torch.randn_like(grad_w) * effective_noise_std)
        grad_b.add_(torch.randn_like(grad_b) * effective_noise_std)

    if quantization_bits > 0:
        if quantization_bits < 2:
            raise ValueError("--quantization_bits must be >= 2")
        qmax = float(2 ** (quantization_bits - 1) - 1)
        for tensor in (grad_w, grad_b):
            max_abs = float(tensor.abs().max().item())
            if max_abs > 0:
                step = max_abs / qmax
                tensor.copy_(torch.round(tensor / step).clamp(-qmax, qmax) * step)
    return grad_w, grad_b


def candidate_rows(grad_b, absolute_threshold, relative_threshold):
    scale = max(float(grad_b.abs().max().item()), 1e-30)
    threshold = max(absolute_threshold, relative_threshold * scale)
    rows = torch.nonzero(grad_b < -threshold, as_tuple=False).flatten().tolist()
    return rows, threshold


def relative_error(estimate, target):
    return float(
        (
            torch.linalg.vector_norm(estimate - target)
            / torch.clamp(torch.linalg.vector_norm(target), min=1e-12)
        ).item()
    )


def main():
    parser = argparse.ArgumentParser(description="Recover batch candidates from negative final-layer bias-gradient rows")
    parser.add_argument("--dataset", choices=["celeba", "cifar10", "fashionmnist", "folder", "imagefolder"], required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--target_weight", required=True)
    parser.add_argument("--decoder_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--arch", choices=["vgg", "resnet18"], default="vgg")
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--img_size", type=int, default=128)
    parser.add_argument("--in_channels", type=int, default=3)
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--feature_dim", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_batches", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--label_source", choices=["pred", "dataset"], default="pred")
    parser.add_argument("--observation", choices=["gradient", "model_delta"], default="gradient")
    parser.add_argument("--fedavg_clients", type=int, default=1,
                        help="Number of equal-size client updates averaged per observation.")
    parser.add_argument("--client_batch_size", type=int, default=0,
                        help="Per-client batch size when --fedavg_clients > 1.")
    parser.add_argument("--local_steps", type=int, default=1)
    parser.add_argument("--local_lr", type=float, default=0.01)
    parser.add_argument("--local_momentum", type=float, default=0.0)
    parser.add_argument("--local_weight_decay", type=float, default=0.0)
    parser.add_argument("--bn_mode", choices=["train", "eval"], default="train",
                        help="BN behavior inside each simulated local model update.")
    parser.add_argument("--clip_grad_norm", type=float, default=0.0)
    parser.add_argument("--noise_std", type=float, default=0.0,
                        help="Absolute Gaussian noise std in observed gradient units.")
    parser.add_argument("--noise_multiplier", type=float, default=0.0,
                        help="Noise std multiplier times clip_grad_norm.")
    parser.add_argument("--quantization_bits", type=int, default=0,
                        help="Symmetric per-tensor quantization bits; 0 disables quantization.")
    parser.add_argument("--absolute_bias_threshold", type=float, default=1e-10)
    parser.add_argument("--relative_bias_threshold", type=float, default=1e-4)
    parser.add_argument("--manifest_path", default="")
    parser.add_argument("--manifest_partition", choices=["", "auxiliary", "validation", "test"], default="")
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--save_pairs", type=int, default=40)
    args = parser.parse_args()

    if args.batch_size < 2:
        raise ValueError("Use attack_decoder_multidataset.py for batch_size=1")
    if args.local_steps < 1:
        raise ValueError("--local_steps must be >= 1")
    if args.fedavg_clients < 1:
        raise ValueError("--fedavg_clients must be >= 1")
    if args.fedavg_clients > 1 and args.client_batch_size < 1:
        raise ValueError("--client_batch_size is required when --fedavg_clients > 1")
    seed_everything(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    transform = build_transform(args.dataset, args.img_size, args.in_channels)
    aggregate_batch_size = (
        args.batch_size
        if args.fedavg_clients == 1
        else args.fedavg_clients * args.client_batch_size
    )
    max_samples = aggregate_batch_size * args.num_batches
    dataset = build_dataset(
        args.dataset,
        args.data_root,
        args.split,
        transform,
        max_samples,
        args.download,
        manifest_path=args.manifest_path,
        manifest_partition=args.manifest_partition,
    )
    if args.label_source == "dataset" and hasattr(dataset, "has_complete_labels") and not dataset.has_complete_labels:
        raise RuntimeError(
            f"Dataset-label mode requires complete manifest labels; missing={dataset.missing_label_count}"
        )
    loader = DataLoader(
        dataset,
        batch_size=aggregate_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
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
    decoder = load_decoder(args.decoder_path, args.feature_dim, args.img_size, args.in_channels)

    per_batch = []
    unique_records = []
    collision_records = []
    total_unique_targets = 0
    total_candidate_rows = 0
    total_false_rows = 0
    saved_pairs = 0

    for batch_index, (images, dataset_labels) in enumerate(loader):
        if batch_index >= args.num_batches:
            break
        images = images.to(DEVICE)
        dataset_labels = dataset_labels.to(DEVICE)
        model_images = target_model_input(images, args.dataset)
        with torch.no_grad():
            global_logits, _ = target_model(model_images)
            predicted_labels = global_logits.argmax(dim=1)
        labels = dataset_labels if args.label_source == "dataset" else predicted_labels

        for parameter in target_model.parameters():
            parameter.requires_grad = args.observation == "gradient"
        if args.fedavg_clients == 1:
            client_images = [model_images]
            client_labels = [labels]
        else:
            client_images = list(torch.chunk(model_images, args.fedavg_clients, dim=0))
            client_labels = list(torch.chunk(labels, args.fedavg_clients, dim=0))

        client_results = []
        for client_x, client_y in zip(client_images, client_labels):
            if args.observation == "gradient":
                result = final_layer_gradient(target_model, client_x, client_y)
            else:
                result = final_layer_model_delta(
                    target_model,
                    client_x,
                    client_y,
                    args.local_steps,
                    args.local_lr,
                    args.local_momentum,
                    args.local_weight_decay,
                    args.bn_mode,
                )
            client_grad_w, client_grad_b = perturb_observation(
                result[0],
                result[1],
                clip_norm=args.clip_grad_norm,
                noise_std=args.noise_std,
                noise_multiplier=args.noise_multiplier,
                quantization_bits=args.quantization_bits,
            )
            client_results.append((client_grad_w, client_grad_b, result[2], result[3]))
        grad_w = torch.stack([result[0] for result in client_results]).mean(dim=0)
        grad_b = torch.stack([result[1] for result in client_results]).mean(dim=0)
        true_features = torch.cat([result[2] for result in client_results], dim=0)
        loss = float(np.mean([result[3] for result in client_results]))

        rows, threshold = candidate_rows(
            grad_b,
            args.absolute_bias_threshold,
            args.relative_bias_threshold,
        )
        label_counts = Counter(int(v) for v in labels.detach().cpu().tolist())
        total_unique_targets += sum(1 for count in label_counts.values() if count == 1)
        total_candidate_rows += len(rows)
        batch_false_rows = 0
        batch_unique_recovered = 0
        batch_collision_rows = 0

        for class_row in rows:
            member_indices = torch.nonzero(labels == class_row, as_tuple=False).flatten().tolist()
            recovered_feature = (grad_w[class_row] / grad_b[class_row]).view(1, -1).to(DEVICE)
            with torch.no_grad():
                reconstruction = decoder(recovered_feature)

            if not member_indices:
                batch_false_rows += 1
                total_false_rows += 1
                continue

            if len(member_indices) == 1:
                source_index = member_indices[0]
                real = images[source_index : source_index + 1]
                true_feature = true_features[source_index : source_index + 1]
                record = {
                    "batch_index": batch_index,
                    "class_row": class_row,
                    "source_index_in_batch": source_index,
                    "psnr": float(compute_psnr(reconstruction, real).item()),
                    "ssim": float(compute_ssim(reconstruction, real)),
                    "id_loss": float(compute_id_loss(target_model, real, reconstruction, args.dataset).item()),
                    "relative_feature_error": relative_error(recovered_feature, true_feature),
                    "bias_gradient": float(grad_b[class_row].item()),
                }
                unique_records.append(record)
                batch_unique_recovered += 1
                if saved_pairs < args.save_pairs:
                    save_pair(real, reconstruction, output_dir / "unique_candidates", saved_pairs, record)
                    saved_pairs += 1
            else:
                member_tensor = torch.tensor(member_indices, device=DEVICE)
                class_mean_feature = true_features.index_select(0, member_tensor).mean(dim=0, keepdim=True)
                collision_records.append({
                    "batch_index": batch_index,
                    "class_row": class_row,
                    "members": len(member_indices),
                    "relative_feature_error_to_class_mean": relative_error(recovered_feature, class_mean_feature),
                    "bias_gradient": float(grad_b[class_row].item()),
                })
                batch_collision_rows += 1

        per_batch.append({
            "batch_index": batch_index,
            "loss": loss,
            "candidate_rows": len(rows),
            "unique_label_targets": sum(1 for count in label_counts.values() if count == 1),
            "unique_recovered": batch_unique_recovered,
            "collision_rows": batch_collision_rows,
            "false_candidate_rows": batch_false_rows,
            "row_threshold": threshold,
            "label_histogram": {str(k): v for k, v in sorted(label_counts.items())},
        })
        print(
            f"batch={batch_index:03d} candidates={len(rows)} "
            f"unique={batch_unique_recovered} collisions={batch_collision_rows} false={batch_false_rows}"
        )

    if not per_batch:
        raise RuntimeError("No complete batches were evaluated")

    summary = {
        "schema_version": 2,
        "method": "negative-row batch candidate recovery",
        "args": vars(args),
        "evaluated_batches": len(per_batch),
        "evaluated_samples": len(per_batch) * aggregate_batch_size,
        "candidate_rows": total_candidate_rows,
        "false_candidate_rows": total_false_rows,
        "false_candidate_rate": total_false_rows / max(total_candidate_rows, 1),
        "unique_label_targets": total_unique_targets,
        "unique_candidates_recovered": len(unique_records),
        "unique_candidate_coverage": len(unique_records) / max(total_unique_targets, 1),
        "collision_candidates": len(collision_records),
        "unique_metrics": {
            "psnr": summarize_values([r["psnr"] for r in unique_records], seed=args.seed),
            "ssim": summarize_values([r["ssim"] for r in unique_records], seed=args.seed),
            "id_loss": summarize_values([r["id_loss"] for r in unique_records], seed=args.seed),
            "relative_feature_error": summarize_values(
                [r["relative_feature_error"] for r in unique_records], seed=args.seed
            ),
        },
        "collision_feature_error": summarize_values(
            [r["relative_feature_error_to_class_mean"] for r in collision_records], seed=args.seed
        ),
        "per_batch": per_batch,
        "unique_records": unique_records,
        "collision_records": collision_records,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("=" * 78)
    print(f"observation: {args.observation}")
    print(f"evaluated_batches: {summary['evaluated_batches']}")
    print(f"unique_candidate_coverage: {summary['unique_candidate_coverage']:.4f}")
    print(f"false_candidate_rate: {summary['false_candidate_rate']:.4f}")
    if unique_records:
        print(f"unique PSNR: {summary['unique_metrics']['psnr']['mean']:.2f} dB")
        print(f"unique SSIM: {summary['unique_metrics']['ssim']['mean']:.4f}")
    print(f"summary_json: {summary_path}")
    print("=" * 78)


if __name__ == "__main__":
    main()
