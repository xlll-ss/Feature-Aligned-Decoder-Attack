"""Audit label and feature leakage from single-sample final-layer gradients."""

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
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--manifest_path", required=True)
    parser.add_argument("--target_weight", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--test_samples", type=int, default=500)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--eps", type=float, default=1e-12)
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args()

    seed_everything(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    transform = build_transform("celeba", 64, 3)
    dataset = build_dataset(
        "celeba",
        args.data_root,
        "test",
        transform,
        args.test_samples,
        False,
        manifest_path=args.manifest_path,
        manifest_partition="test",
    )
    if not getattr(dataset, "has_complete_labels", False):
        raise RuntimeError("Dataset-label audit requires complete manifest labels.")

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
    )
    model = load_target_model(args.target_weight, 3, 0, 2048, "vgg")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(True)

    records = []
    label_correct = []
    stable_row_correct = []
    feature_errors = []
    abs_denominators = []
    saturated = []

    print("=" * 78)
    print("Single-sample final-layer label leakage audit")
    print("=" * 78)
    print(f"target_weight: {args.target_weight}")
    print("label estimator: argmin(final-layer bias gradient)")
    print("stable row: argmax(abs(final-layer bias gradient))")
    print(f"requested samples: {args.test_samples}")

    for source_idx, (image, label) in enumerate(loader):
        if len(records) >= args.test_samples:
            break

        image = image.to(DEVICE)
        label = label.to(DEVICE)

        with torch.no_grad():
            logits, true_feature = model(target_model_input(image, "celeba"))
            predicted_label = logits.argmax(dim=1)

        loss = F.cross_entropy(
            model(target_model_input(image, "celeba"))[0],
            label,
        )
        grad_w, grad_b = torch.autograd.grad(
            loss,
            (model.fc_layer.weight, model.fc_layer.bias),
            create_graph=False,
            retain_graph=False,
        )
        grad_w = grad_w.detach()
        grad_b = grad_b.detach()

        inferred_label = int(grad_b.argmin().item())
        stable_row = int(grad_b.abs().argmax().item())
        denominator = float(grad_b[stable_row].item())
        is_saturated = abs(denominator) <= args.eps

        record = {
            "source_idx": source_idx,
            "true_label": int(label.item()),
            "pred_label": int(predicted_label.item()),
            "inferred_label_from_bias_argmin": inferred_label,
            "stable_row_from_bias_absmax": stable_row,
            "stable_denominator": denominator,
            "saturated": is_saturated,
        }

        if is_saturated:
            saturated.append(record)
        else:
            recovered_feature = (
                grad_w[stable_row] / grad_b[stable_row]
            ).view(1, -1)
            relative_error = float(
                (
                    torch.linalg.vector_norm(recovered_feature - true_feature)
                    / torch.linalg.vector_norm(true_feature).clamp_min(1e-12)
                ).item()
            )
            record["relative_feature_error"] = relative_error
            feature_errors.append(relative_error)
            abs_denominators.append(abs(denominator))

        label_correct.append(
            float(inferred_label == int(label.item()))
        )
        stable_row_correct.append(
            float(stable_row == int(label.item()))
        )
        records.append(record)

        if len(records) <= 10 or len(records) % 100 == 0:
            print(
                f"sample={len(records):03d} src_idx={source_idx:04d} "
                f"true={int(label.item())} inferred={inferred_label} "
                f"stable={stable_row} saturated={is_saturated}"
            )

    if not records:
        raise RuntimeError("No samples were audited.")

    summary = {
        "method": "single-sample final-layer label leakage audit",
        "label_estimator": "argmin(final-layer bias gradient)",
        "stable_row_rule": "argmax(abs(final-layer bias gradient))",
        "args": vars(args),
        "evaluated_samples": len(records),
        "saturated_samples": len(saturated),
        "label_inference_accuracy": summarize_values(label_correct, args.seed),
        "stable_row_matches_true_label": summarize_values(
            stable_row_correct,
            args.seed,
        ),
        "relative_feature_error": summarize_values(feature_errors, args.seed),
        "stable_denominator_abs": summarize_values(abs_denominators, args.seed),
        "records": records,
        "saturated": saturated,
    }

    path = output_dir / "summary.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("=" * 78)
    print(f"evaluated samples: {summary['evaluated_samples']}")
    print(f"saturated samples: {summary['saturated_samples']}")
    print(
        "label inference accuracy: "
        f"{summary['label_inference_accuracy']['mean']:.6f}"
    )
    print(
        "stable-row true-label rate: "
        f"{summary['stable_row_matches_true_label']['mean']:.6f}"
    )
    print(
        "feature relative error: "
        f"{summary['relative_feature_error']['mean']:.3e}"
    )
    print(f"summary_json: {path}")
    print("=" * 78)


if __name__ == "__main__":
    main()