"""Independent fixed-crop ArcFace similarity for low-resolution face pairs."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from experiment_utils import summarize_values


def sample_id_from_name(path):
    prefix = path.name[: -len("_real.png")]
    if not prefix.isdigit():
        raise ValueError(f"Expected numeric '*_real.png' name, got: {path.name}")
    return int(prefix)


def load_image(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Unable to read image: {path}")
    return image


def fixed_crop_embedding(recognizer, image, embedding_size):
    """Embed the full center-cropped CelebA image without face detection."""
    aligned = cv2.resize(
        image,
        (embedding_size, embedding_size),
        interpolation=cv2.INTER_CUBIC,
    )
    feature = np.asarray(recognizer.get_feat(aligned), dtype=np.float64).reshape(-1)
    norm = np.linalg.norm(feature)
    if norm <= 0:
        raise RuntimeError("ArcFace returned a zero embedding")
    return feature / norm


def main():
    parser = argparse.ArgumentParser(
        description="Fixed-crop ArcFace similarity for *_real.png / *_recon.png pairs."
    )
    parser.add_argument("--pairs_dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="buffalo_l")
    parser.add_argument("--ctx_id", type=int, default=-1)
    parser.add_argument("--embedding_size", type=int, default=112)
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.30,
        help="Report fraction of pairs with cosine similarity >= threshold.",
    )
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args()

    if args.embedding_size < 64:
        raise ValueError("--embedding_size must be at least 64")
    if not -1.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be in [-1, 1]")

    try:
        from insightface.app import FaceAnalysis
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Install dependencies: pip install insightface onnxruntime-gpu opencv-python"
        ) from exc

    pairs_dir = Path(args.pairs_dir).expanduser()
    real_paths = sorted(pairs_dir.glob("*_real.png"))
    if args.max_samples > 0:
        real_paths = real_paths[:args.max_samples]
    if not real_paths:
        raise RuntimeError(f"No '*_real.png' files found under {pairs_dir}")

    app = FaceAnalysis(name=args.model)
    app.prepare(ctx_id=args.ctx_id)
    recognizer = app.models.get("recognition")
    if recognizer is None:
        raise RuntimeError(f"Recognition model unavailable in InsightFace pack: {args.model}")

    scores = []
    threshold_passes = []
    records = []

    for real_path in real_paths:
        recon_path = real_path.with_name(
            real_path.name.replace("_real.png", "_recon.png")
        )
        if not recon_path.is_file():
            raise RuntimeError(f"Missing reconstruction: {recon_path}")

        real_embedding = fixed_crop_embedding(
            recognizer, load_image(real_path), args.embedding_size
        )
        recon_embedding = fixed_crop_embedding(
            recognizer, load_image(recon_path), args.embedding_size
        )
        cosine = float(np.dot(real_embedding, recon_embedding))
        threshold_pass = bool(cosine >= args.threshold)

        record = {
            "sample_id": sample_id_from_name(real_path),
            "real": real_path.name,
            "recon": recon_path.name,
            "arcface_cosine": cosine,
            "similarity_above_threshold": threshold_pass,
        }
        records.append(record)
        scores.append(cosine)
        threshold_passes.append(float(threshold_pass))
        print(
            f"sample={record['sample_id']:04d} "
            f"arcface_cosine={cosine:.4f} "
            f"pass={int(threshold_pass)}"
        )

    summary = {
        "metric": "fixed-crop ArcFace cosine similarity",
        "backend": "insightface",
        "model": args.model,
        "ctx_id": args.ctx_id,
        "embedding_size": args.embedding_size,
        "threshold": args.threshold,
        "total_pairs": len(records),
        "valid_pairs": len(records),
        "preprocessing": (
            "full saved CelebA crop resized to 112x112 with bicubic interpolation; "
            "no detector or landmark alignment"
        ),
        "arcface_cosine": summarize_values(scores, seed=args.seed),
        "similarity_above_threshold": summarize_values(
            threshold_passes, seed=args.seed
        ),
        "records": records,
    }

    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    cosine = summary["arcface_cosine"]
    passed = summary["similarity_above_threshold"]
    print("=" * 78)
    print(f"pairs: {summary['total_pairs']}")
    print(f"ArcFace cosine: {cosine['mean']:.6f}")
    print(
        f"ArcFace cosine 95% CI: "
        f"[{cosine['ci95']['lower']:.6f}, {cosine['ci95']['upper']:.6f}]"
    )
    print(
        f"fraction >= {args.threshold:.2f}: {passed['mean']:.4f}"
    )
    print(f"summary_json: {output}")
    print("=" * 78)


if __name__ == "__main__":
    main()