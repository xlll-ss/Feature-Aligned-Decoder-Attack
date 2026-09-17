import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from attack_decoder_multidataset import (
    compute_id_loss,
    compute_psnr,
    compute_ssim,
    load_decoder,
    tensor_to_pil,
)
from experiment_utils import seed_everything, summarize_values
from train_decoder_multidataset import (
    DEVICE,
    build_dataset,
    build_transform,
    load_target_model,
    target_model_input,
)


def metric_record(target_model, reconstruction, real, dataset):
    return {
        "psnr": float(compute_psnr(reconstruction, real).item()),
        "ssim": float(compute_ssim(reconstruction, real)),
        "id_loss": float(compute_id_loss(target_model, real, reconstruction, dataset).item()),
    }


def summarize_records(records, seed):
    return {
        metric: summarize_values([record[metric] for record in records], seed=seed)
        for metric in ("psnr", "ssim", "id_loss")
    }


def save_raw_pair(real, fake, output_dir, index):
    """Save a raw pair in the naming convention consumed by evaluate_lpips.py."""
    pair_dir = Path(output_dir)
    pair_dir.mkdir(parents=True, exist_ok=True)
    tensor_to_pil(real).save(pair_dir / f"{index:06d}_real.png")
    tensor_to_pil(fake).save(pair_dir / f"{index:06d}_recon.png")


def main():
    parser = argparse.ArgumentParser(description="Evaluate mean-image, auxiliary nearest-neighbor, and FAD decoder baselines")
    parser.add_argument("--dataset", choices=["celeba", "cifar10", "fashionmnist", "folder", "imagefolder"], required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--target_weight", required=True)
    parser.add_argument("--decoder_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--arch", choices=["vgg", "resnet18"], default="vgg")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--img_size", type=int, default=128)
    parser.add_argument("--in_channels", type=int, default=3)
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--feature_dim", type=int, default=2048)
    parser.add_argument("--aux_samples", type=int, default=5000)
    parser.add_argument("--test_samples", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--distance", choices=["cosine", "l2"], default="cosine")
    parser.add_argument("--manifest_path", default="")
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument(
        "--save_raw_pairs",
        action="store_true",
        help="Save per-method *_real.png and *_recon.png pairs for independent LPIPS evaluation.",
    )
    parser.add_argument(
        "--save_pairs",
        type=int,
        default=0,
        help="Number of pairs per method to save; 0 saves all evaluated samples.",
    )
    args = parser.parse_args()

    seed_everything(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    transform = build_transform(args.dataset, args.img_size, args.in_channels)

    aux_dataset = build_dataset(
        args.dataset,
        args.data_root,
        "train",
        transform,
        args.aux_samples,
        args.download,
        manifest_path=args.manifest_path,
        manifest_partition="auxiliary" if args.manifest_path else "",
    )
    test_dataset = build_dataset(
        args.dataset,
        args.data_root,
        "test",
        transform,
        args.test_samples,
        args.download,
        manifest_path=args.manifest_path,
        manifest_partition="test" if args.manifest_path else "",
    )
    aux_loader = DataLoader(aux_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)
    target_model = load_target_model(
        args.target_weight,
        args.in_channels,
        args.num_classes,
        args.feature_dim,
        args.arch,
    )
    target_model.eval()
    decoder = load_decoder(args.decoder_path, args.feature_dim, args.img_size, args.in_channels)

    feature_chunks = []
    image_sum = None
    image_count = 0
    with torch.no_grad():
        for images, _ in aux_loader:
            images = images.to(DEVICE)
            _, features = target_model(target_model_input(images, args.dataset))
            feature_chunks.append(features.detach().cpu())
            batch_sum = images.detach().cpu().sum(dim=0, keepdim=True)
            image_sum = batch_sum if image_sum is None else image_sum + batch_sum
            image_count += images.size(0)
    feature_bank = torch.cat(feature_chunks, dim=0)
    mean_image = image_sum / image_count
    normalized_bank = F.normalize(feature_bank, dim=1) if args.distance == "cosine" else None

    records = {"mean_image": [], "nearest_auxiliary": [], "fad_decoder": []}
    per_sample = []
    for index, (real, _) in enumerate(test_loader):
        if index >= args.test_samples:
            break
        real = real.to(DEVICE)
        with torch.no_grad():
            _, feature = target_model(target_model_input(real, args.dataset))
            decoded = decoder(feature)
        feature_cpu = feature.detach().cpu()
        if args.distance == "cosine":
            scores = normalized_bank @ F.normalize(feature_cpu, dim=1).t()
            nearest_index = int(scores.argmax().item())
        else:
            distances = torch.cdist(feature_cpu, feature_bank).squeeze(0)
            nearest_index = int(distances.argmin().item())
        nearest_image, _ = aux_dataset[nearest_index]
        nearest_image = nearest_image.unsqueeze(0).to(DEVICE)
        mean_reconstruction = mean_image.to(DEVICE)

        sample = {"index": index, "nearest_auxiliary_index": nearest_index}
        for method, reconstruction in (
            ("mean_image", mean_reconstruction),
            ("nearest_auxiliary", nearest_image),
            ("fad_decoder", decoded),
        ):
            metrics = metric_record(target_model, reconstruction, real, args.dataset)
            records[method].append(metrics)
            sample[method] = metrics
            if args.save_raw_pairs and (args.save_pairs <= 0 or index < args.save_pairs):
                save_raw_pair(real, reconstruction, output_dir / "raw_pairs" / method, index)
        per_sample.append(sample)
        print(
            f"sample={index:04d} mean={sample['mean_image']['psnr']:.2f} "
            f"nn={sample['nearest_auxiliary']['psnr']:.2f} fad={sample['fad_decoder']['psnr']:.2f}"
        )

    summary = {
        "schema_version": 2,
        "args": vars(args),
        "auxiliary_samples": len(feature_bank),
        "evaluated_samples": len(per_sample),
        "methods": {method: summarize_records(method_records, args.seed) for method, method_records in records.items()},
        "per_sample": per_sample,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("=" * 78)
    for method, metrics in summary["methods"].items():
        print(f"{method}: PSNR={metrics['psnr']['mean']:.2f}, SSIM={metrics['ssim']['mean']:.4f}")
    print(f"summary_json: {summary_path}")
    print("=" * 78)


if __name__ == "__main__":
    main()
