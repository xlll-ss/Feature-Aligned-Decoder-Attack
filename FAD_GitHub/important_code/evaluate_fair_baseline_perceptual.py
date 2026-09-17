"""Evaluate LPIPS and fixed-crop ArcFace for one fair-baseline run."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from experiment_utils import summarize_values


def sample_id(path):
    prefix = path.name[: -len("_real.png")]
    if not prefix.isdigit():
        raise ValueError(f"Expected numeric '*_real.png' name, got {path.name}")
    return int(prefix)


def fixed_crop_embedding(recognizer, path, size):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Unable to read image: {path}")
    image = cv2.resize(image, (size, size), interpolation=cv2.INTER_CUBIC)
    feature = np.asarray(recognizer.get_feat(image), dtype=np.float64).reshape(-1)
    norm = np.linalg.norm(feature)
    if norm <= 0:
        raise RuntimeError(f"ArcFace returned a zero embedding for {path}")
    return feature / norm


def write_empty(output_dir, seed, lpips_net, arcface_model, threshold):
    empty = summarize_values([], seed=seed)
    write_json(
        output_dir / "lpips.json",
        {
            "metric": "LPIPS",
            "net": lpips_net,
            "evaluated_samples": 0,
            "lpips": empty,
            "records": [],
        },
    )
    write_json(
        output_dir / "arcface.json",
        {
            "metric": "fixed-crop ArcFace cosine similarity",
            "model": arcface_model,
            "threshold": threshold,
            "total_pairs": 0,
            "valid_pairs": 0,
            "arcface_cosine": empty,
            "similarity_above_threshold": empty,
            "records": [],
        },
    )


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--lpips_net", default="alex", choices=["alex", "vgg", "squeeze"])
    parser.add_argument("--arcface_model", default="buffalo_l")
    parser.add_argument("--arcface_ctx_id", type=int, default=-1)
    parser.add_argument("--embedding_size", type=int, default=112)
    parser.add_argument("--threshold", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args()
    run_dir = Path(args.run_dir).expanduser()
    pairs_dir = run_dir / "raw_pairs"
    real_paths = sorted(pairs_dir.glob("*_real.png"))
    if not real_paths:
        write_empty(
            run_dir,
            args.seed,
            args.lpips_net,
            args.arcface_model,
            args.threshold,
        )
        print(f"No successful pairs under {pairs_dir}; wrote empty metric files")
        return

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
    lpips_values, arcface_values, pass_values = [], [], []
    lpips_records, arcface_records = [], []

    for real_path in real_paths:
        reconstruction_path = real_path.with_name(
            real_path.name.replace("_real.png", "_recon.png")
        )
        if not reconstruction_path.is_file():
            raise RuntimeError(f"Missing reconstruction: {reconstruction_path}")
        index = sample_id(real_path)
        real = to_tensor(Image.open(real_path).convert("RGB")).mul(2).sub(1)
        reconstruction = to_tensor(Image.open(reconstruction_path).convert("RGB")).mul(2).sub(1)
        with torch.no_grad():
            lpips_value = float(
                lpips_model(
                    real.unsqueeze(0).to(device),
                    reconstruction.unsqueeze(0).to(device),
                ).mean().item()
            )
        real_embedding = fixed_crop_embedding(recognizer, real_path, args.embedding_size)
        reconstruction_embedding = fixed_crop_embedding(
            recognizer, reconstruction_path, args.embedding_size
        )
        cosine = float(np.dot(real_embedding, reconstruction_embedding))
        passed = bool(cosine >= args.threshold)
        lpips_values.append(lpips_value)
        arcface_values.append(cosine)
        pass_values.append(float(passed))
        lpips_records.append(
            {"sample_id": index, "real": real_path.name, "recon": reconstruction_path.name, "lpips": lpips_value}
        )
        arcface_records.append(
            {
                "sample_id": index,
                "real": real_path.name,
                "recon": reconstruction_path.name,
                "arcface_cosine": cosine,
                "similarity_above_threshold": passed,
            }
        )

    write_json(
        run_dir / "lpips.json",
        {
            "metric": "LPIPS",
            "net": args.lpips_net,
            "evaluated_samples": len(lpips_values),
            "lpips": summarize_values(lpips_values, seed=args.seed),
            "records": lpips_records,
        },
    )
    write_json(
        run_dir / "arcface.json",
        {
            "metric": "fixed-crop ArcFace cosine similarity",
            "backend": "insightface",
            "model": args.arcface_model,
            "embedding_size": args.embedding_size,
            "threshold": args.threshold,
            "preprocessing": "full saved crop resized with bicubic interpolation; no detector alignment",
            "total_pairs": len(arcface_values),
            "valid_pairs": len(arcface_values),
            "arcface_cosine": summarize_values(arcface_values, seed=args.seed),
            "similarity_above_threshold": summarize_values(pass_values, seed=args.seed),
            "records": arcface_records,
        },
    )
    print(
        f"pairs={len(real_paths)} LPIPS={np.mean(lpips_values):.6f} "
        f"ArcFace={np.mean(arcface_values):.6f} pass={np.mean(pass_values):.4f}"
    )


if __name__ == "__main__":
    main()
