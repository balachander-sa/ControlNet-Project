"""
Stage 1: Depth Map Extraction using Depth Anything V2.

Extracts monocular depth maps from RGB driving images using Depth Anything V2
with configurable encoder variants (ViT-S, ViT-B, ViT-L).

Usage:
    # Single variant:
    python scripts/extract_depth.py --encoder vits --img_dir data/images --output_dir outputs/depth_maps_vits

    # All variants (for ablation):
    python scripts/extract_depth.py --encoder all --img_dir data/images --output_dir outputs/depth_maps
"""

import argparse
import os
import sys
import glob
import cv2
import numpy as np
import torch
from tqdm import tqdm


MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
}

CHECKPOINT_URLS = {
    "vits": "https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth",
    "vitb": "https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth",
    "vitl": "https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth",
}


def load_model(encoder: str, checkpoint_dir: str, device: str = "cuda"):
    """Load a Depth Anything V2 model."""
    from depth_anything_v2.dpt import DepthAnythingV2

    ckpt_path = os.path.join(checkpoint_dir, f"depth_anything_v2_{encoder}.pth")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            f"Download it with:\n"
            f"  wget -O {ckpt_path} {CHECKPOINT_URLS[encoder]}"
        )

    model = DepthAnythingV2(**MODEL_CONFIGS[encoder])
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=True))
    model = model.to(device).eval()
    return model


def extract_depth_maps(
    model, img_files: list, output_dir: str, desc: str = "Extracting depth"
):
    """Run depth extraction on a list of image files."""
    os.makedirs(output_dir, exist_ok=True)

    for img_path in tqdm(img_files, desc=desc):
        raw_img = cv2.imread(img_path)
        depth = model.infer_image(raw_img)

        # Normalize to 0-255 for saving as grayscale PNG
        depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8) * 255.0
        cv2.imwrite(
            os.path.join(output_dir, os.path.basename(img_path)),
            depth_norm.astype(np.uint8),
        )


def main():
    parser = argparse.ArgumentParser(description="Extract depth maps using Depth Anything V2")
    parser.add_argument("--encoder", type=str, default="vitl", choices=["vits", "vitb", "vitl", "all"],
                        help="Encoder variant (default: vitl). Use 'all' for ablation.")
    parser.add_argument("--img_dir", type=str, required=True, help="Directory containing input images")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for depth maps")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints",
                        help="Directory containing model checkpoints")
    parser.add_argument("--depth_anything_path", type=str, default=None,
                        help="Path to Depth-Anything-V2 repo (added to sys.path)")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    # Add Depth Anything V2 to path if specified
    if args.depth_anything_path:
        sys.path.insert(0, args.depth_anything_path)

    # Gather images
    img_files = sorted(
        glob.glob(os.path.join(args.img_dir, "*.png"))
        + glob.glob(os.path.join(args.img_dir, "*.jpg"))
    )
    if not img_files:
        raise FileNotFoundError(f"No images found in {args.img_dir}")
    print(f"Found {len(img_files)} images in {args.img_dir}")

    # Determine which encoders to run
    encoders = ["vits", "vitb", "vitl"] if args.encoder == "all" else [args.encoder]

    for encoder in encoders:
        if args.encoder == "all":
            output_dir = f"{args.output_dir}_{encoder}"
        else:
            output_dir = args.output_dir

        print(f"\nLoading {encoder}...")
        model = load_model(encoder, args.checkpoint_dir, args.device)

        extract_depth_maps(model, img_files, output_dir, desc=f"{encoder}")

        # Free GPU memory before loading next model
        del model
        torch.cuda.empty_cache()

        print(f"Done! {len(img_files)} depth maps saved to {output_dir}")


if __name__ == "__main__":
    main()
