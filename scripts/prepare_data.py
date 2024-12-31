"""
Stage 2: Prepare Training Data.

Aligns images, captions, and depth maps into a HuggingFace-compatible dataset
with metadata.jsonl for LoRA training.

Usage:
    # Single variant:
    python scripts/prepare_data.py --variant vitl --img_dir data/images --txt_dir data/captions \
        --depth_dir outputs/depth_maps_vitl --output_dir outputs/train_data_vitl

    # All variants (for ablation):
    python scripts/prepare_data.py --variant all --img_dir data/images --txt_dir data/captions \
        --depth_base_dir outputs/depth_maps --output_base_dir outputs/train_data
"""

import argparse
import json
import os
import shutil


def prepare_variant(img_dir: str, txt_dir: str, depth_dir: str, output_dir: str):
    """Prepare training data for a single depth variant."""
    os.makedirs(output_dir, exist_ok=True)

    metadata = []
    for f in sorted(os.listdir(img_dir)):
        if not f.endswith(".png"):
            continue

        stem = os.path.splitext(f)[0]
        txt_path = os.path.join(txt_dir, stem + ".txt")
        depth_path = os.path.join(depth_dir, f)

        if not os.path.exists(txt_path):
            continue
        if not os.path.exists(depth_path):
            continue

        with open(txt_path) as fh:
            caption = fh.read().strip()

        # Copy image and depth map into training directory
        shutil.copy2(os.path.join(img_dir, f), os.path.join(output_dir, f))
        shutil.copy2(depth_path, os.path.join(output_dir, f"depth_{f}"))

        metadata.append({
            "file_name": f,
            "conditioning_image": f"depth_{f}",
            "text": caption,
        })

    # Write metadata.jsonl
    with open(os.path.join(output_dir, "metadata.jsonl"), "w") as fh:
        for m in metadata:
            fh.write(json.dumps(m) + "\n")

    print(f"Created {len(metadata)} training samples in {output_dir}")
    return len(metadata)


def main():
    parser = argparse.ArgumentParser(description="Prepare training data for LoRA fine-tuning")
    parser.add_argument("--variant", type=str, default="vitl",
                        choices=["vits", "vitb", "vitl", "all"])
    parser.add_argument("--img_dir", type=str, required=True, help="Directory with source images")
    parser.add_argument("--txt_dir", type=str, required=True, help="Directory with caption .txt files")
    parser.add_argument("--depth_dir", type=str, default=None,
                        help="Depth map directory (for single variant)")
    parser.add_argument("--depth_base_dir", type=str, default=None,
                        help="Base path for depth maps (for 'all' mode, appends _vits/_vitb/_vitl)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (for single variant)")
    parser.add_argument("--output_base_dir", type=str, default=None,
                        help="Base output path (for 'all' mode)")
    args = parser.parse_args()

    variants = ["vits", "vitb", "vitl"] if args.variant == "all" else [args.variant]

    for variant in variants:
        if args.variant == "all":
            depth_dir = f"{args.depth_base_dir}_{variant}"
            output_dir = f"{args.output_base_dir}_{variant}"
        else:
            depth_dir = args.depth_dir
            output_dir = args.output_dir

        print(f"\nPreparing data for {variant}...")
        prepare_variant(args.img_dir, args.txt_dir, depth_dir, output_dir)


if __name__ == "__main__":
    main()
