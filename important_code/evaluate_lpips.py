"""Evaluate reconstructions with an independent LPIPS perceptual metric."""

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from experiment_utils import summarize_values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs_dir", required=True, help="Directory containing *_real.png and *_recon.png files.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--net", choices=["alex", "vgg", "squeeze"], default="alex")
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args()

    try:
        import lpips
    except ModuleNotFoundError as exc:
        raise RuntimeError("Install LPIPS first: pip install lpips") from exc

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metric = lpips.LPIPS(net=args.net).to(device).eval()
    to_tensor = transforms.ToTensor()
    pairs_dir = Path(args.pairs_dir).expanduser()
    real_paths = sorted(pairs_dir.glob("*_real.png"))
    if args.max_samples > 0:
        real_paths = real_paths[:args.max_samples]
    if not real_paths:
        raise RuntimeError(f"No *_real.png files found in {pairs_dir}")

    values = []
    records = []
    with torch.no_grad():
        for real_path in real_paths:
            recon_path = real_path.with_name(real_path.name.replace("_real.png", "_recon.png"))
            if not recon_path.is_file():
                raise RuntimeError(f"Missing reconstruction for {real_path.name}")
            real = to_tensor(Image.open(real_path).convert("RGB")).mul(2).sub(1).unsqueeze(0).to(device)
            recon = to_tensor(Image.open(recon_path).convert("RGB")).mul(2).sub(1).unsqueeze(0).to(device)
            value = float(metric(real, recon).mean().item())
            values.append(value)
            records.append({"real": real_path.name, "recon": recon_path.name, "lpips": value})

    summary = {
        "metric": "LPIPS",
        "net": args.net,
        "device": str(device),
        "evaluated_samples": len(values),
        "lpips": summarize_values(values, seed=args.seed),
        "records": records,
    }
    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"LPIPS ({args.net}): {summary['lpips']['mean']:.6f}")
    print(f"95% CI: [{summary['lpips']['ci95']['lower']:.6f}, {summary['lpips']['ci95']['upper']:.6f}]")
    print(f"summary_json: {output}")


if __name__ == "__main__":
    main()
