"""Create three separate 8x8 ground-truth/reconstruction comparison grids."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageOps, ImageStat


PAIR_PATTERN = re.compile(r"^(\d+)_real\.png$")
DATASETS = (
    ("celeba", "celeba_root", "celeba_resnet18_batch64"),
    ("cifar10", "cifar10_root", "cifar10_resnet18_batch64"),
    ("fashionmnist", "fashionmnist_root", "fashionmnist_resnet18_batch64"),
)
WHITE = (255, 255, 255)
RED = (225, 36, 31)


def find_raw_pair_dir(experiment_root, seed):
    root = Path(experiment_root).expanduser()
    candidates = (
        root / "attack" / f"seed{seed}" / "raw_pairs",
        root / f"seed{seed}" / "raw_pairs",
        root / "raw_pairs",
        root,
    )
    for candidate in candidates:
        if candidate.is_dir() and any(candidate.glob("*_real.png")):
            return candidate
    raise FileNotFoundError(f"No raw image pairs for seed {seed} under {root}")


def discover_pairs(directory):
    pairs = {}
    for real_path in sorted(directory.glob("*_real.png")):
        match = PAIR_PATTERN.match(real_path.name)
        if not match:
            continue
        sample_id = int(match.group(1))
        recon_path = real_path.with_name(
            real_path.name.replace("_real.png", "_recon.png")
        )
        if recon_path.is_file():
            pairs[sample_id] = (real_path, recon_path)
    if not pairs:
        raise RuntimeError(f"No complete real/reconstruction pairs in {directory}")
    return pairs


def find_summary(experiment_root, seed, raw_pair_directory):
    root = Path(experiment_root).expanduser()
    candidates = (
        root / "attack" / f"seed{seed}" / "summary.json",
        root / f"seed{seed}" / "summary.json",
        raw_pair_directory.parent / "summary.json",
        root / "summary.json",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def load_ssim_scores(summary_path):
    if summary_path is None:
        return {}
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    scores = {}
    for record in payload.get("per_sample", []):
        if "idx" in record and "ssim" in record:
            scores[int(record["idx"])] = float(record["ssim"])
    return scores


def pixel_mae(real_path, recon_path):
    with Image.open(real_path) as real_source, Image.open(recon_path) as recon_source:
        real = real_source.convert("RGB").resize((64, 64), Image.Resampling.BILINEAR)
        recon = recon_source.convert("RGB").resize((64, 64), Image.Resampling.BILINEAR)
    difference = ImageChops.difference(real, recon)
    return sum(ImageStat.Stat(difference).mean) / 3.0


def select_highlights(pairs, sample_ids, ssim_scores, count):
    if count <= 0:
        return [], "none", {}
    available_ssim = {
        sample_id: ssim_scores[sample_id]
        for sample_id in sample_ids
        if sample_id in ssim_scores
    }
    if len(available_ssim) == len(sample_ids):
        ranked = sorted(sample_ids, key=lambda sample_id: available_ssim[sample_id])
        return ranked[:count], "lowest_ssim", available_ssim

    mae_scores = {
        sample_id: pixel_mae(*pairs[sample_id]) for sample_id in sample_ids
    }
    ranked = sorted(sample_ids, key=lambda sample_id: mae_scores[sample_id], reverse=True)
    return ranked[:count], "highest_pixel_mae", mae_scores


def choose_ids(common_ids, count, selection):
    if len(common_ids) < count:
        raise RuntimeError(
            f"Need {count} indices shared by all datasets, found {len(common_ids)}"
        )
    if selection == "first":
        return common_ids[:count]
    if count == 1:
        return [common_ids[len(common_ids) // 2]]
    positions = [round(i * (len(common_ids) - 1) / (count - 1)) for i in range(count)]
    return [common_ids[position] for position in positions]


def contained_image(path, tile_size):
    with Image.open(path) as source:
        image = source.convert("RGB")
        return ImageOps.contain(
            image, (tile_size, tile_size), Image.Resampling.LANCZOS
        )


def paste_without_crop(canvas, path, x, y, tile_size):
    image = contained_image(path, tile_size)
    paste_x = x + (tile_size - image.width) // 2
    paste_y = y + (tile_size - image.height) // 2
    canvas.paste(image, (paste_x, paste_y))


def draw_batch64(pairs, sample_ids, highlighted_ids, output_stem, grid_size, tile_size):
    gutter = max(round(tile_size * 0.12), 6)
    center_gap = max(round(tile_size * 0.72), 36)
    margin = max(round(tile_size * 0.14), 8)
    panel_size = grid_size * tile_size + (grid_size - 1) * gutter
    width = margin * 2 + panel_size * 2 + center_gap
    height = margin * 2 + panel_size
    canvas = Image.new("RGB", (width, height), WHITE)
    draw = ImageDraw.Draw(canvas)

    right_start = margin + panel_size + center_gap
    for position, sample_id in enumerate(sample_ids):
        row, column = divmod(position, grid_size)
        y = margin + row * (tile_size + gutter)
        left_x = margin + column * (tile_size + gutter)
        right_x = right_start + column * (tile_size + gutter)
        real_path, recon_path = pairs[sample_id]
        paste_without_crop(canvas, real_path, left_x, y, tile_size)
        paste_without_crop(canvas, recon_path, right_x, y, tile_size)
        if sample_id in highlighted_ids:
            border = max(round(tile_size * 0.045), 3)
            draw.rectangle(
                (
                    right_x - border,
                    y - border,
                    right_x + tile_size + border - 1,
                    y + tile_size + border - 1,
                ),
                outline=RED,
                width=border,
            )

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_stem.with_suffix(".png")
    pdf_path = output_stem.with_suffix(".pdf")
    canvas.save(png_path, dpi=(300, 300), optimize=True)
    canvas.save(pdf_path, "PDF", resolution=300.0)
    return png_path, pdf_path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--celeba_root", required=True)
    parser.add_argument("--cifar10_root", required=True)
    parser.add_argument("--fashionmnist_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--grid_size", type=int, default=8)
    parser.add_argument("--tile_size", type=int, default=96)
    parser.add_argument("--selection", choices=("first", "even"), default="first")
    parser.add_argument("--indices", nargs="+", type=int, default=None)
    parser.add_argument("--highlight_count", type=int, default=2)
    parser.add_argument("--celeba_highlight", nargs="*", type=int, default=None)
    parser.add_argument("--cifar10_highlight", nargs="*", type=int, default=None)
    parser.add_argument("--fashionmnist_highlight", nargs="*", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.grid_size < 1 or args.tile_size < 32:
        raise ValueError("--grid_size must be positive and --tile_size must be at least 32")
    if args.highlight_count < 0:
        raise ValueError("--highlight_count cannot be negative")

    pair_sets = {}
    raw_directories = {}
    roots = {}
    for dataset, root_key, _ in DATASETS:
        root = getattr(args, root_key)
        directory = find_raw_pair_dir(root, args.seed)
        roots[dataset] = root
        raw_directories[dataset] = directory
        pair_sets[dataset] = discover_pairs(directory)

    count = args.grid_size * args.grid_size
    common_ids = sorted(set.intersection(*(set(pairs) for pairs in pair_sets.values())))
    if args.indices is not None:
        if len(args.indices) != count:
            raise ValueError(f"--indices must contain exactly {count} values")
        unavailable = [sample_id for sample_id in args.indices if sample_id not in common_ids]
        if unavailable:
            raise ValueError(f"Indices unavailable in all three datasets: {unavailable}")
        sample_ids = args.indices
        selection_method = "explicit"
    else:
        sample_ids = choose_ids(common_ids, count, args.selection)
        selection_method = f"{args.selection}_{count}_shared_indices"

    output_dir = Path(args.output_dir).expanduser()
    record = {
        "seed": args.seed,
        "grid": [args.grid_size, args.grid_size],
        "sample_ids": sample_ids,
        "selection": selection_method,
        "red_box_meaning": "See each dataset's highlight_criterion and scores",
        "datasets": {},
    }

    for dataset, _, output_name in DATASETS:
        manual_highlights = getattr(args, f"{dataset}_highlight")
        if manual_highlights is not None:
            unavailable = [value for value in manual_highlights if value not in sample_ids]
            if unavailable:
                raise ValueError(
                    f"Manual {dataset} highlights are not displayed: {unavailable}"
                )
            highlighted_ids = manual_highlights
            criterion = "manual"
            scores = {}
        else:
            summary_path = find_summary(
                roots[dataset], args.seed, raw_directories[dataset]
            )
            highlighted_ids, criterion, scores = select_highlights(
                pair_sets[dataset],
                sample_ids,
                load_ssim_scores(summary_path),
                min(args.highlight_count, len(sample_ids)),
            )

        png_path, pdf_path = draw_batch64(
            pair_sets[dataset],
            sample_ids,
            set(highlighted_ids),
            output_dir / output_name,
            args.grid_size,
            args.tile_size,
        )
        record["datasets"][dataset] = {
            "raw_pair_directory": str(raw_directories[dataset]),
            "highlighted_ids": highlighted_ids,
            "highlight_criterion": criterion,
            "highlight_scores": {
                str(sample_id): scores[sample_id]
                for sample_id in highlighted_ids
                if sample_id in scores
            },
            "png": str(png_path),
            "pdf": str(pdf_path),
        }
        print(f"Saved {dataset}: {png_path}")
        print(f"Red boxes ({criterion}): {highlighted_ids}")

    record_path = output_dir / "batch64_selection.json"
    record_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"Saved selection record: {record_path}")


if __name__ == "__main__":
    main()
