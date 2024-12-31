"""
Visualization utilities for the Scene Augmentation pipeline.

Generates:
  - Training loss curves
  - Depth comparison grids (Original / ViT-S / ViT-B / ViT-L)
  - Generation comparison grids (Original / Depth / Generated per variant)
  - Custom prompt showcase grids
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def plot_training_loss(log_file: str, output_path: str = "figures/training_loss.png",
                       title: str = "Training Loss", window: int = 5):
    """
    Plot training loss from a log file. Expects one loss value per line,
    or lines containing 'avg loss: X.XXXX'.
    """
    losses = []
    with open(log_file) as f:
        for line in f:
            if "avg loss:" in line:
                val = float(line.strip().split("avg loss:")[-1].strip())
                losses.append(val)

    if not losses:
        print(f"No loss values found in {log_file}")
        return

    epochs = range(1, len(losses) + 1)

    # Smoothed curve
    smoothed = np.convolve(losses, np.ones(window) / window, mode="valid")
    smoothed_x = range(window, len(losses) + 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, losses, alpha=0.3, color="steelblue", label="Per-epoch loss")
    ax.plot(smoothed_x, smoothed, color="darkblue", linewidth=2, label=f"Smoothed (window={window})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_depth_comparison(
    img_dir: str,
    depth_dirs: dict,
    sample_ids: list,
    output_path: str = "figures/depth_comparison.png",
):
    """
    Plot original images alongside depth maps from multiple variants.

    Args:
        depth_dirs: dict like {"ViT-S": "outputs/depth_maps_vits", "ViT-B": ...}
    """
    n = len(sample_ids)
    cols = 1 + len(depth_dirs)
    fig, axes = plt.subplots(n, cols, figsize=(5 * cols, 4 * n))

    col_titles = ["Original"] + list(depth_dirs.keys())
    for j, title in enumerate(col_titles):
        axes[0][j].set_title(title, fontsize=14, fontweight="bold", pad=10)

    for i, sid in enumerate(sample_ids):
        fname = f"{sid}.png"

        # Original
        orig = Image.open(os.path.join(img_dir, fname))
        axes[i][0].imshow(orig)
        axes[i][0].axis("off")

        # Depth maps
        for j, (label, ddir) in enumerate(depth_dirs.items(), start=1):
            depth_path = os.path.join(ddir, fname)
            if os.path.exists(depth_path):
                axes[i][j].imshow(Image.open(depth_path), cmap="inferno")
            else:
                axes[i][j].text(0.5, 0.5, "N/A", ha="center", va="center")
            axes[i][j].axis("off")

    plt.tight_layout(pad=1.0)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_generation_comparison(
    img_dir: str,
    depth_dir: str,
    gen_dirs: dict,
    sample_ids: list,
    output_path: str = "figures/generation_grid.png",
):
    """
    Plot original / depth / generated images across variants.

    Args:
        gen_dirs: dict like {"ViT-S": "outputs/generated_images_vits", ...}
    """
    n = len(sample_ids)
    cols = 2 + len(gen_dirs)
    fig, axes = plt.subplots(n, cols, figsize=(5 * cols, 4 * n))

    col_titles = ["Original", "Depth (ViT-L)"] + [f"Gen ({k})" for k in gen_dirs.keys()]
    for j, title in enumerate(col_titles):
        axes[0][j].set_title(title, fontsize=14, fontweight="bold", pad=10)

    original_size = Image.open(
        os.path.join(img_dir, f"{sample_ids[0]}.png")
    ).size

    for i, sid in enumerate(sample_ids):
        fname = f"{sid}.png"

        orig = Image.open(os.path.join(img_dir, fname))
        axes[i][0].imshow(orig)
        axes[i][0].axis("off")

        depth = Image.open(os.path.join(depth_dir, fname)).resize(original_size, Image.LANCZOS)
        axes[i][1].imshow(depth, cmap="inferno")
        axes[i][1].axis("off")

        for j, (label, gdir) in enumerate(gen_dirs.items(), start=2):
            gen_path = os.path.join(gdir, fname)
            if os.path.exists(gen_path):
                gen = Image.open(gen_path).resize(original_size, Image.LANCZOS)
                axes[i][j].imshow(gen)
            else:
                axes[i][j].text(0.5, 0.5, "N/A", ha="center", va="center")
            axes[i][j].axis("off")

    plt.tight_layout(pad=1.0)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")
