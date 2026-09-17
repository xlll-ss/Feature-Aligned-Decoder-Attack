"""Fair same-interface comparison between FAD and iterative baselines."""

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
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


METHODS = {
    "fad": {
        "display_name": "FAD",
        "family": "decoder",
        "label_access": "unknown_not_required_for_feature_recovery",
        "optimizer": "none",
        "default_steps": 0,
        "default_restarts": 0,
        "offline_auxiliary_images": 30000,
    },
    "dlg_joint": {
        "display_name": "DLG (joint label)",
        "family": "dlg",
        "label_access": "unknown_jointly_optimized",
        "optimizer": "LBFGS",
        "default_steps": 300,
        "default_restarts": 1,
        "offline_auxiliary_images": 0,
    },
    "idlg": {
        "display_name": "iDLG (W+b label inference)",
        "family": "dlg",
        "label_access": "unknown_inferred_from_bias_gradient",
        "optimizer": "LBFGS",
        "default_steps": 300,
        "default_restarts": 1,
        "offline_auxiliary_images": 0,
    },
    "ig": {
        "display_name": "Inverting Gradients (W+b label inference)",
        "family": "ig",
        "label_access": "unknown_inferred_from_bias_gradient",
        "optimizer": "Adam",
        "default_steps": 4800,
        "default_restarts": 3,
        "offline_auxiliary_images": 0,
    },
    "dlg_known": {
        "display_name": "DLG (known-label diagnostic)",
        "family": "dlg",
        "label_access": "known_dataset_label",
        "optimizer": "LBFGS",
        "default_steps": 300,
        "default_restarts": 1,
        "offline_auxiliary_images": 0,
    },
}


class FinalLayerGradientView(nn.Module):
    """Expose only the final classifier weight and bias to inversefed."""

    def __init__(self, target):
        super().__init__()
        self.target = target
        for parameter in target.parameters():
            parameter.requires_grad_(False)
        self.selected_parameters = (target.fc_layer.weight, target.fc_layer.bias)
        for parameter in self.selected_parameters:
            parameter.requires_grad_(True)
        self.forward_evaluations = 0

    def forward(self, normalized_image):
        self.forward_evaluations += 1
        logits, _ = self.target(target_model_input(normalized_image, "celeba"))
        return logits

    def parameters(self, recurse=True):
        del recurse
        return iter(self.selected_parameters)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def read_progress(path):
    records = {}
    path = Path(path)
    if not path.is_file():
        return records
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Invalid progress JSON at line {line_number}: {path}") from exc
        records[int(event["source_idx"])] = event
    return records


def append_progress(path, event):
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event) + "\n")
        handle.flush()


def synchronize():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def iterative_config(family, steps, restarts):
    if family == "ig":
        return {
            "signed": False,
            "boxed": True,
            "cost_fn": "sim",
            "indices": "def",
            "weights": "equal",
            "lr": 0.1,
            "optim": "adam",
            "restarts": restarts,
            "max_iterations": steps,
            "total_variation": 1e-1,
            "init": "randn",
            "filter": "none",
            "lr_decay": True,
            "scoring_choice": "loss",
        }
    return {
        "signed": False,
        "boxed": False,
        "cost_fn": "l2",
        "indices": "def",
        "weights": "equal",
        "lr": 1e-4,
        "optim": "LBFGS",
        "restarts": restarts,
        "max_iterations": steps,
        "total_variation": 0.0,
        "init": "randn",
        "filter": "none",
        "lr_decay": False,
        "scoring_choice": "loss",
    }


def infer_label_from_bias(grad_bias):
    return grad_bias.argmin().view(1).detach()


def reconstruct_fad(grad_weight, grad_bias, decoder, epsilon):
    row = int(grad_bias.abs().argmax().item())
    denominator = grad_bias[row]
    if denominator.abs().item() <= epsilon:
        raise RuntimeError("all final-layer bias-gradient rows are numerically zero")
    feature = (grad_weight[row] / denominator).view(1, -1)
    with torch.no_grad():
        reconstruction = decoder(feature)
    return reconstruction, {"recovered_row": row, "objective": 0.0, "forward_evaluations": 1}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=sorted(METHODS), required=True)
    parser.add_argument("--ig_repo", default="third_party/invertinggradients")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--manifest_path", required=True)
    parser.add_argument("--target_weight", required=True)
    parser.add_argument("--decoder_path", default="")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--test_samples", type=int, default=50)
    parser.add_argument("--steps", type=int, default=-1)
    parser.add_argument("--restarts", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--epsilon", type=float, default=1e-12)
    parser.add_argument("--restart", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    method = METHODS[args.method]
    steps = method["default_steps"] if args.steps < 0 else args.steps
    restarts = method["default_restarts"] if args.restarts < 0 else args.restarts
    if args.test_samples <= 0:
        raise ValueError("--test_samples must be positive")
    if steps < 0 or restarts < 0:
        raise ValueError("steps and restarts must be non-negative")

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "progress.jsonl"
    summary_path = output_dir / "summary.json"
    config_path = output_dir / "run_config.json"
    if args.restart:
        progress_path.unlink(missing_ok=True)
        summary_path.unlink(missing_ok=True)
        (output_dir / "lpips.json").unlink(missing_ok=True)
        (output_dir / "arcface.json").unlink(missing_ok=True)
        raw_dir = output_dir / "raw_pairs"
        if raw_dir.is_dir():
            shutil.rmtree(raw_dir)

    run_config = {
        "method": args.method,
        "target_weight": str(Path(args.target_weight).expanduser()),
        "decoder_path": str(Path(args.decoder_path).expanduser()) if args.decoder_path else None,
        "manifest_path": str(Path(args.manifest_path).expanduser()),
        "test_samples": args.test_samples,
        "steps": steps,
        "restarts": restarts,
        "seed": args.seed,
        "gradient_scope": ["fc_layer.weight", "fc_layer.bias"],
    }
    if config_path.is_file() and not args.restart:
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        if previous != run_config:
            raise RuntimeError(
                f"Existing run configuration differs: {config_path}. Use --restart or a new output directory."
            )
    write_json(config_path, run_config)

    seed_everything(args.seed)
    target = load_target_model(args.target_weight, 3, 0, 2048, "vgg")
    target.eval()
    model = FinalLayerGradientView(target).to(DEVICE).eval()
    decoder = None
    if args.method == "fad":
        if not args.decoder_path:
            raise ValueError("--decoder_path is required for FAD")
        decoder = load_decoder(args.decoder_path, 2048, 64, 3)

    inversefed = None
    attack_config = None
    if method["family"] != "decoder":
        repo = Path(args.ig_repo).expanduser().resolve()
        if not (repo / "inversefed").is_dir():
            raise FileNotFoundError(f"Cannot find inversefed under: {repo}")
        sys.path.insert(0, str(repo))
        import inversefed as inversefed_module

        inversefed = inversefed_module
        attack_config = iterative_config(method["family"], steps, restarts)

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
        raise RuntimeError("Fair baseline comparison requires complete test labels")
    paths = [str(path) for path in getattr(dataset, "paths", [])]
    sample_set_sha256 = hashlib.sha256("\n".join(paths).encode("utf-8")).hexdigest()
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)
    completed = read_progress(progress_path)

    print("=" * 88)
    print(f"method: {method['display_name']}")
    print(f"label_access: {method['label_access']}")
    print("gradient_scope: fc_layer.weight + fc_layer.bias")
    print(f"samples: {len(dataset)}, steps: {steps}, restarts: {restarts}, seed: {args.seed}")
    print(f"resume_completed: {len(completed)}")
    print("=" * 88)

    for source_idx, (image, label) in enumerate(loader):
        if source_idx in completed:
            continue
        sample_seed = args.seed * 1000003 + source_idx
        seed_everything(sample_seed)
        image = image.to(DEVICE)
        label = label.to(DEVICE, dtype=torch.long)
        with torch.no_grad():
            prediction = model(image).argmax(dim=1)

        observed_loss = F.cross_entropy(model(image), label)
        grad_weight, grad_bias = [
            gradient.detach()
            for gradient in torch.autograd.grad(
                observed_loss,
                list(model.parameters()),
                create_graph=False,
                retain_graph=False,
            )
        ]
        inferred_label = infer_label_from_bias(grad_bias)
        base_event = {
            "source_idx": source_idx,
            "sample_seed": sample_seed,
            "dataset_label": int(label.item()),
            "target_prediction": int(prediction.item()),
            "target_correct": bool(prediction.item() == label.item()),
            "inferred_label": int(inferred_label.item()),
            "inferred_label_correct": bool(inferred_label.item() == label.item()),
            "observed_loss": float(observed_loss.item()),
        }

        synchronize()
        started = time.perf_counter()
        try:
            if args.method == "fad":
                reconstruction, details = reconstruct_fad(
                    grad_weight, grad_bias, decoder, args.epsilon
                )
            else:
                normalization_mean = torch.full((3, 1, 1), 0.5, device=DEVICE)
                normalization_std = torch.full((3, 1, 1), 0.5, device=DEVICE)
                reconstructor = inversefed.GradientReconstructor(
                    model,
                    mean_std=(normalization_mean, normalization_std),
                    config=dict(attack_config),
                    num_images=1,
                )
                if args.method == "dlg_joint":
                    reconstructor.iDLG = False
                    attack_label = None
                elif args.method == "dlg_known":
                    attack_label = label
                else:
                    attack_label = inferred_label
                model.forward_evaluations = 0
                reconstruction, stats = reconstructor.reconstruct(
                    [grad_weight, grad_bias],
                    attack_label,
                    img_shape=(3, 64, 64),
                )
                details = {
                    "recovered_row": None,
                    "objective": float(stats.get("opt", float("nan"))),
                    "forward_evaluations": model.forward_evaluations,
                }
            synchronize()
            elapsed = time.perf_counter() - started
            if not torch.isfinite(reconstruction).all():
                raise RuntimeError("reconstruction contains NaN or infinity")
            reconstruction = reconstruction.detach().clamp(-1.0, 1.0)
            with torch.no_grad():
                metrics = {
                    "psnr": float(compute_psnr(reconstruction, image).item()),
                    "ssim": float(compute_ssim(reconstruction, image)),
                    "id_loss": float(
                        compute_id_loss(target, image, reconstruction, "celeba").item()
                    ),
                }
            raw_dir = output_dir / "raw_pairs"
            raw_dir.mkdir(parents=True, exist_ok=True)
            save_image((image.detach().cpu() + 1.0).div(2).clamp(0, 1), raw_dir / f"{source_idx:04d}_real.png")
            save_image(
                (reconstruction.cpu() + 1.0).div(2).clamp(0, 1),
                raw_dir / f"{source_idx:04d}_recon.png",
            )
            event = {
                **base_event,
                "status": "success",
                **metrics,
                **details,
                "elapsed_seconds": elapsed,
            }
            print(
                f"sample={source_idx:04d} PSNR={metrics['psnr']:.2f} "
                f"SSIM={metrics['ssim']:.4f} time={elapsed:.2f}s"
            )
        except (RuntimeError, ValueError, IndexError) as exc:
            synchronize()
            elapsed = time.perf_counter() - started
            event = {
                **base_event,
                "status": "failure",
                "reason": f"{type(exc).__name__}: {exc}",
                "elapsed_seconds": elapsed,
                "forward_evaluations": (
                    model.forward_evaluations if args.method != "fad" else 0
                ),
            }
            print(f"sample={source_idx:04d} FAILED time={elapsed:.2f}s: {event['reason']}")
        append_progress(progress_path, event)
        completed[source_idx] = event
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    events = [completed[index] for index in sorted(completed) if index < len(dataset)]
    if len(events) != len(dataset):
        raise RuntimeError(f"Progress contains {len(events)}/{len(dataset)} requested samples")
    successes = [event for event in events if event["status"] == "success"]
    failures = [event for event in events if event["status"] == "failure"]
    inferred_events = events if args.method in {"fad", "idlg", "ig"} else []
    summary = {
        "experiment": "fair same-interface reconstruction baseline comparison",
        "method": args.method,
        "display_name": method["display_name"],
        "protocol": {
            "dataset": "celeba",
            "architecture": "vgg",
            "image_size": 64,
            "batch_size": 1,
            "gradient_scope": ["fc_layer.weight", "fc_layer.bias"],
            "loss": "cross_entropy",
            "label_access": method["label_access"],
            "private_label_used_by_attacker": args.method == "dlg_known",
            "test_selection": "first N paths in sorted immutable manifest test partition",
            "sample_set_sha256": sample_set_sha256,
            "output_projection": "clamp to normalized image range [-1, 1] before evaluation",
            "offline_auxiliary_images": method["offline_auxiliary_images"],
            "metric_population": "successful reconstructions; failures reported separately",
        },
        "target_weight": str(Path(args.target_weight).expanduser()),
        "decoder_path": str(Path(args.decoder_path).expanduser()) if args.decoder_path else None,
        "attack_config": attack_config,
        "steps": steps,
        "restarts": restarts,
        "seed": args.seed,
        "requested_samples": len(dataset),
        "evaluated_samples": len(successes),
        "failed_samples": len(failures),
        "valid_rate": len(successes) / len(dataset),
        "target_accuracy": summarize_values(
            [float(event["target_correct"]) for event in events], seed=args.seed
        ),
        "label_inference_accuracy": (
            summarize_values(
                [float(event["inferred_label_correct"]) for event in inferred_events],
                seed=args.seed,
            )
            if inferred_events
            else None
        ),
        "metrics": {
            **{
                metric: summarize_values(
                    [event[metric] for event in successes], seed=args.seed
                )
                for metric in ("psnr", "ssim", "id_loss")
            },
            "elapsed_seconds": summarize_values(
                [event["elapsed_seconds"] for event in events], seed=args.seed
            ),
            "forward_evaluations": summarize_values(
                [event["forward_evaluations"] for event in events], seed=args.seed
            ),
        },
        "per_sample": successes,
        "failures": failures,
        "args": vars(args),
    }
    write_json(summary_path, summary)
    print("=" * 88)
    print(f"completed: {len(successes)}/{len(dataset)}, failures: {len(failures)}")
    if successes:
        print(f"PSNR={summary['metrics']['psnr']['mean']:.4f}")
        print(f"SSIM={summary['metrics']['ssim']['mean']:.6f}")
    print(f"summary_json: {summary_path}")


if __name__ == "__main__":
    main()
