"""Compute LPIPS and fixed-crop ArcFace once for all ablation conditions."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from experiment_utils import summarize_values


def fixed_crop_embedding(recognizer, path, size):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Unable to read {path}")
    image = cv2.resize(image, (size, size), interpolation=cv2.INTER_CUBIC)
    feature = np.asarray(recognizer.get_feat(image), dtype=np.float64).reshape(-1)
    norm = np.linalg.norm(feature)
    if norm <= 0:
        raise RuntimeError(f"ArcFace returned zero embedding for {path}")
    return feature / norm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed_dir", required=True)
    parser.add_argument("--conditions", nargs="+", required=True)
    parser.add_argument("--lpips_net", default="alex", choices=["alex", "vgg", "squeeze"])
    parser.add_argument("--arcface_model", default="buffalo_l")
    parser.add_argument("--arcface_ctx_id", type=int, default=-1)
    parser.add_argument("--embedding_size", type=int, default=112)
    parser.add_argument("--threshold", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args()

    try:
        import lpips
        from insightface.app import FaceAnalysis
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Install lpips, insightface, onnxruntime/onnxruntime-gpu, and opencv-python"
        ) from exc

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lpips_model = lpips.LPIPS(net=args.lpips_net).to(device).eval()
    app = FaceAnalysis(name=args.arcface_model)
    app.prepare(ctx_id=args.arcface_ctx_id)
    recognizer = app.models.get("recognition")
    if recognizer is None:
        raise RuntimeError("ArcFace recognition model is unavailable")
    to_tensor = transforms.ToTensor()
    seed_dir = Path(args.seed_dir)

    for condition in args.conditions:
        pairs_dir = seed_dir / condition / "raw_pairs"
        real_paths = sorted(pairs_dir.glob("*_real.png"))
        if not real_paths:
            raise RuntimeError(f"No raw pairs for {condition}: {pairs_dir}")
        lpips_values, arcface_values, threshold_values = [], [], []
        lpips_records, arcface_records = [], []
        for real_path in real_paths:
            recon_path = real_path.with_name(real_path.name.replace("_real.png", "_recon.png"))
            if not recon_path.is_file():
                raise RuntimeError(f"Missing reconstruction: {recon_path}")
            sample_id = int(real_path.name.split("_", 1)[0])
            real_tensor = to_tensor(Image.open(real_path).convert("RGB")).mul(2).sub(1)
            recon_tensor = to_tensor(Image.open(recon_path).convert("RGB")).mul(2).sub(1)
            with torch.no_grad():
                lpips_value = float(
                    lpips_model(
                        real_tensor.unsqueeze(0).to(device),
                        recon_tensor.unsqueeze(0).to(device),
                    ).mean().item()
                )
            real_embedding = fixed_crop_embedding(recognizer, real_path, args.embedding_size)
            recon_embedding = fixed_crop_embedding(recognizer, recon_path, args.embedding_size)
            cosine = float(np.dot(real_embedding, recon_embedding))
            passed = bool(cosine >= args.threshold)
            lpips_values.append(lpips_value)
            arcface_values.append(cosine)
            threshold_values.append(float(passed))
            lpips_records.append(
                {"sample_id": sample_id, "real": real_path.name, "recon": recon_path.name, "lpips": lpips_value}
            )
            arcface_records.append(
                {
                    "sample_id": sample_id,
                    "real": real_path.name,
                    "recon": recon_path.name,
                    "arcface_cosine": cosine,
                    "similarity_above_threshold": passed,
                }
            )
        lpips_output = {
            "metric": "LPIPS",
            "net": args.lpips_net,
            "evaluated_samples": len(lpips_values),
            "lpips": summarize_values(lpips_values, seed=args.seed),
            "records": lpips_records,
        }
        arcface_output = {
            "metric": "fixed-crop ArcFace cosine similarity",
            "backend": "insightface",
            "model": args.arcface_model,
            "embedding_size": args.embedding_size,
            "threshold": args.threshold,
            "preprocessing": "full saved crop resized with bicubic interpolation; no detector alignment",
            "total_pairs": len(arcface_values),
            "valid_pairs": len(arcface_values),
            "arcface_cosine": summarize_values(arcface_values, seed=args.seed),
            "similarity_above_threshold": summarize_values(threshold_values, seed=args.seed),
            "records": arcface_records,
        }
        (seed_dir / condition / "lpips.json").write_text(
            json.dumps(lpips_output, indent=2), encoding="utf-8"
        )
        (seed_dir / condition / "arcface.json").write_text(
            json.dumps(arcface_output, indent=2), encoding="utf-8"
        )
        print(
            f"{condition}: LPIPS={np.mean(lpips_values):.6f}, "
            f"ArcFace={np.mean(arcface_values):.6f}, pass={np.mean(threshold_values):.4f}"
        )


if __name__ == "__main__":
    main()
