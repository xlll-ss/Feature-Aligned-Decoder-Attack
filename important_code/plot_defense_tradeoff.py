"""Plot the defense-strength/privacy/utility sweep as a publication figure."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def metric(points, name):
    means = np.asarray([point["metrics"][name]["mean"] for point in points])
    intervals = np.asarray(
        [point["metrics"][name]["image_bootstrap_ci95"] for point in points]
    )
    errors = np.vstack((means - intervals[:, 0], intervals[:, 1] - means))
    return means, errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True)
    parser.add_argument("--output_png", required=True)
    parser.add_argument("--output_pdf", required=True)
    args = parser.parse_args()

    summary = json.loads(Path(args.summary).read_text(encoding="utf-8"))
    points = sorted(summary["points"], key=lambda point: point["strength"])
    strengths = np.asarray([point["strength"] for point in points])
    labels = [f"{value:g}" for value in strengths]
    utility = 100.0 * np.asarray([point["sealed_test_accuracy"] for point in points])
    selection = 100.0 * np.asarray(
        [point["selection_validation_accuracy"] for point in points]
    )
    utility_intervals = 100.0 * np.asarray(
        [point["sealed_test_accuracy_ci95"] for point in points]
    )
    utility_errors = np.vstack(
        (utility - utility_intervals[:, 0], utility_intervals[:, 1] - utility)
    )

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.linewidth": 0.8,
        }
    )
    colors = {
        "blue": "#2463A7",
        "red": "#C23B3B",
        "green": "#2D7F5E",
        "gold": "#B47B12",
        "gray": "#5F6670",
    }
    figure, axes = plt.subplots(2, 2, figsize=(11.2, 7.3), constrained_layout=True)

    ax = axes[0, 0]
    ax.errorbar(
        strengths,
        utility,
        yerr=utility_errors,
        color=colors["blue"],
        marker="o",
        linewidth=1.8,
        capsize=3,
        label="Sealed test",
    )
    ax.plot(
        strengths,
        selection,
        color=colors["gray"],
        marker="s",
        linestyle="--",
        linewidth=1.4,
        label="Selection validation",
    )
    ax.set_title("(a) Target-model utility")
    ax.set_ylabel("Top-1 accuracy (%)")
    ax.legend(frameon=False)

    psnr, psnr_error = metric(points, "psnr")
    ssim, ssim_error = metric(points, "ssim")
    ax = axes[0, 1]
    psnr_line = ax.errorbar(
        strengths,
        psnr,
        yerr=psnr_error,
        color=colors["red"],
        marker="o",
        linewidth=1.8,
        capsize=3,
        label="PSNR",
    )
    ax.set_ylabel("PSNR (dB)", color=colors["red"])
    ax.tick_params(axis="y", colors=colors["red"])
    twin = ax.twinx()
    ssim_line = twin.errorbar(
        strengths,
        ssim,
        yerr=ssim_error,
        color=colors["green"],
        marker="s",
        linestyle="--",
        linewidth=1.5,
        capsize=3,
        label="SSIM",
    )
    twin.set_ylabel("SSIM", color=colors["green"])
    twin.tick_params(axis="y", colors=colors["green"])
    ax.set_title("(b) Pixel and structural leakage")
    ax.legend(
        [psnr_line, ssim_line],
        ["PSNR", "SSIM"],
        frameon=False,
        loc="best",
    )

    lpips, lpips_error = metric(points, "lpips")
    arcface, arcface_error = metric(points, "arcface_cosine")
    ax = axes[1, 0]
    lpips_line = ax.errorbar(
        strengths,
        lpips,
        yerr=lpips_error,
        color=colors["gold"],
        marker="o",
        linewidth=1.8,
        capsize=3,
        label="LPIPS",
    )
    ax.set_ylabel("LPIPS", color=colors["gold"])
    ax.tick_params(axis="y", colors=colors["gold"])
    twin = ax.twinx()
    arcface_line = twin.errorbar(
        strengths,
        arcface,
        yerr=arcface_error,
        color=colors["blue"],
        marker="s",
        linestyle="--",
        linewidth=1.5,
        capsize=3,
        label="ArcFace cosine",
    )
    twin.set_ylabel("ArcFace cosine", color=colors["blue"])
    twin.tick_params(axis="y", colors=colors["blue"])
    ax.set_title("(c) Perceptual and identity leakage")
    ax.legend(
        [lpips_line, arcface_line],
        ["LPIPS", "ArcFace cosine"],
        frameon=False,
        loc="best",
    )

    ax = axes[1, 1]
    ax.plot(utility, psnr, color=colors["gray"], linewidth=1.3, zorder=1)
    scatter = ax.scatter(
        utility,
        psnr,
        c=strengths,
        cmap="viridis",
        s=46,
        edgecolors="white",
        linewidths=0.7,
        zorder=2,
    )
    for x_value, y_value, label in zip(utility, psnr, labels):
        ax.annotate(
            f"lambda={label}",
            (x_value, y_value),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=7.5,
        )
    ax.invert_yaxis()
    ax.set_xlabel("Sealed-test accuracy (%)")
    ax.set_ylabel("PSNR (dB, lower is more private)")
    ax.set_title("(d) Privacy-utility frontier")
    colorbar = figure.colorbar(scatter, ax=ax, pad=0.02)
    colorbar.set_label("Defense strength")

    for ax in (axes[0, 0], axes[0, 1], axes[1, 0]):
        ax.set_xticks(strengths, labels)
        ax.set_xlabel("Defense strength")
        ax.grid(True, color="#D9DDE3", linewidth=0.6, alpha=0.75)
        ax.set_axisbelow(True)
    axes[1, 1].grid(True, color="#D9DDE3", linewidth=0.6, alpha=0.75)
    axes[1, 1].set_axisbelow(True)
    axes[1, 1].margins(x=0.12, y=0.08)
    figure.suptitle("BiDO Defense Strength: Privacy-Utility Trade-off", fontsize=12)

    output_png = Path(args.output_png).expanduser()
    output_pdf = Path(args.output_pdf).expanduser()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_png, dpi=320, bbox_inches="tight", facecolor="white")
    figure.savefig(output_pdf, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    print(f"figure_png: {output_png}")
    print(f"figure_pdf: {output_pdf}")


if __name__ == "__main__":
    main()
