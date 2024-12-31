"""
Stage 4: Evaluation — FID and Depth Consistency Metrics.

Computes:
  - FID (Frechet Inception Distance) between real and generated images
  - Depth consistency (AbsRel, RMSE) by re-running Depth Anything V2 on
    generated images and comparing with original conditioning depth maps

Usage:
    python scripts/evaluate.py \
        --real_dir data/images \
        --gen_dir outputs/generated_images_vitl \
        --depth_dir outputs/depth_maps_vitl \
        --checkpoint_dir checkpoints \
        --depth_anything_path Depth-Anything-V2
"""

import argparse
import glob
import os
import sys
import tempfile
import shutil
import subprocess

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


def compute_fid(real_dir: str, gen_dir: str, resolution: int = 512):
    """Compute FID between real and generated images using pytorch-fid."""
    # Prepare resized copies for fair comparison
    with tempfile.TemporaryDirectory() as tmpdir:
        real_resized = os.path.join(tmpdir, "real")
        fake_resized = os.path.join(tmpdir, "fake")
        os.makedirs(real_resized)
        os.makedirs(fake_resized)

        for fname in os.listdir(real_dir):
            if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            Image.open(os.path.join(real_dir, fname)).convert("RGB").resize(
                (resolution, resolution)
            ).save(os.path.join(real_resized, fname))

            gen_path = os.path.join(gen_dir, fname)
            if os.path.exists(gen_path):
                Image.open(gen_path).convert("RGB").resize(
                    (resolution, resolution)
                ).save(os.path.join(fake_resized, fname))

        print(f"Real: {len(os.listdir(real_resized))} images")
        print(f"Generated: {len(os.listdir(fake_resized))} images")

        # Run pytorch-fid
        result = subprocess.run(
            ["python", "-m", "pytorch_fid", real_resized, fake_resized],
            capture_output=True, text=True,
        )
        print(result.stdout.strip())
        if result.returncode != 0:
            print(f"FID error: {result.stderr}")
        return result.stdout.strip()


def compute_depth_consistency(
    gen_dir: str,
    depth_dir: str,
    checkpoint_dir: str,
    depth_anything_path: str = None,
    device: str = "cuda",
):
    """Compute AbsRel and RMSE between original and re-estimated depth maps."""
    if depth_anything_path:
        sys.path.insert(0, depth_anything_path)
    from depth_anything_v2.dpt import DepthAnythingV2

    # Load ViT-S for evaluation (lightweight, consistent across ablation)
    model = DepthAnythingV2(
        encoder="vits", features=64, out_channels=[48, 96, 192, 384]
    )
    ckpt_path = os.path.join(checkpoint_dir, "depth_anything_v2_vits.pth")
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=True))
    model = model.to(device).eval()

    gen_files = sorted(glob.glob(os.path.join(gen_dir, "*.png")))
    abs_rels, rmses = [], []

    print(f"Computing depth consistency for {len(gen_files)} images...")
    for gen_path in tqdm(gen_files, desc="Depth consistency"):
        fname = os.path.basename(gen_path)

        # Re-estimate depth on generated image
        gen_img = cv2.imread(gen_path)
        gen_depth = model.infer_image(gen_img)

        # Load original conditioning depth map
        orig_depth = cv2.imread(
            os.path.join(depth_dir, fname), cv2.IMREAD_GRAYSCALE
        ).astype(np.float32)

        # Resize to match
        gen_depth_resized = cv2.resize(
            gen_depth, (orig_depth.shape[1], orig_depth.shape[0])
        )

        # Normalize to [0, 1]
        gen_norm = (gen_depth_resized - gen_depth_resized.min()) / (
            gen_depth_resized.max() - gen_depth_resized.min() + 1e-8
        )
        orig_norm = (orig_depth - orig_depth.min()) / (
            orig_depth.max() - orig_depth.min() + 1e-8
        )

        # AbsRel with valid mask
        valid_mask = orig_norm > 0.05
        if valid_mask.sum() > 0:
            abs_rel = np.mean(
                np.abs(gen_norm[valid_mask] - orig_norm[valid_mask]) / orig_norm[valid_mask]
            )
        else:
            abs_rel = 0.0
        abs_rels.append(abs_rel)

        # RMSE over full image
        rmse = np.sqrt(np.mean((gen_norm - orig_norm) ** 2))
        rmses.append(rmse)

    del model
    torch.cuda.empty_cache()

    mean_absrel = np.mean(abs_rels)
    std_absrel = np.std(abs_rels)
    mean_rmse = np.mean(rmses)
    std_rmse = np.std(rmses)

    print(f"\nAbsRel: {mean_absrel:.4f} ± {std_absrel:.4f}")
    print(f"RMSE:   {mean_rmse:.4f} ± {std_rmse:.4f}")

    return {
        "absrel_mean": mean_absrel,
        "absrel_std": std_absrel,
        "rmse_mean": mean_rmse,
        "rmse_std": std_rmse,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate generated images")
    parser.add_argument("--real_dir", type=str, required=True, help="Directory with real images")
    parser.add_argument("--gen_dir", type=str, required=True, help="Directory with generated images")
    parser.add_argument("--depth_dir", type=str, required=True,
                        help="Directory with original conditioning depth maps")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints",
                        help="Depth Anything V2 checkpoint directory")
    parser.add_argument("--depth_anything_path", type=str, default=None,
                        help="Path to Depth-Anything-V2 repo")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--skip_fid", action="store_true")
    parser.add_argument("--skip_depth", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    print("=" * 60)
    print("EVALUATION")
    print("=" * 60)

    if not args.skip_fid:
        print("\n--- FID ---")
        compute_fid(args.real_dir, args.gen_dir, args.resolution)

    if not args.skip_depth:
        print("\n--- Depth Consistency ---")
        compute_depth_consistency(
            args.gen_dir, args.depth_dir,
            args.checkpoint_dir, args.depth_anything_path, args.device,
        )


if __name__ == "__main__":
    main()
