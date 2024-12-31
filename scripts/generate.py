"""
Stage 3B: Generate Augmented Images with ControlNet + LoRA.

Generates depth-conditioned driving scene images using a trained LoRA model
with ControlNet-depth spatial conditioning.

Usage:
    # Generate from all depth maps with their original captions:
    python scripts/generate.py \
        --depth_dir outputs/depth_maps_vitl \
        --caption_dir data/captions \
        --lora_dir outputs/lora_output_vitl/checkpoint-epoch-50 \
        --output_dir outputs/generated_images_vitl

    # Generate with a custom prompt (for qualitative demos):
    python scripts/generate.py \
        --depth_dir outputs/depth_maps_vitl \
        --lora_dir outputs/lora_output_vitl/checkpoint-epoch-50 \
        --output_dir outputs/custom_generated \
        --prompt "a rainy night urban street with heavy traffic, wet road, reflections"
"""

import argparse
import glob
import os

import torch
from PIL import Image
from tqdm import tqdm
from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    UniPCMultistepScheduler,
)
from peft import LoraConfig


def load_pipeline(lora_dir: str, model_id: str, device: str = "cuda"):
    """Load SD 1.5 + ControlNet-depth + trained LoRA weights."""
    print("Loading ControlNet-depth...")
    controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-depth")

    print("Loading SD 1.5 pipeline...")
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        model_id,
        controlnet=controlnet,
        torch_dtype=torch.float32,
    )

    # Add LoRA adapter (must match training config)
    lora_config = LoraConfig(
        r=4,
        lora_alpha=4,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    pipe.unet.add_adapter(lora_config)

    # Load trained LoRA weights
    lora_weights = torch.load(
        os.path.join(lora_dir, "lora_weights.pt"), map_location="cpu"
    )
    pipe.unet.load_state_dict(lora_weights, strict=False)

    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)

    print("Pipeline loaded with LoRA weights!")
    return pipe


def main():
    parser = argparse.ArgumentParser(description="Generate augmented driving scenes")
    parser.add_argument("--depth_dir", type=str, required=True, help="Directory with depth map PNGs")
    parser.add_argument("--caption_dir", type=str, default=None, help="Directory with caption .txt files")
    parser.add_argument("--lora_dir", type=str, required=True, help="Path to LoRA checkpoint directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for generated images")
    parser.add_argument("--model_id", type=str, default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    parser.add_argument("--prompt", type=str, default=None, help="Override all captions with a single prompt")
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load pipeline
    pipe = load_pipeline(args.lora_dir, args.model_id, args.device)

    if args.seed is not None:
        generator = torch.Generator(device=args.device).manual_seed(args.seed)
    else:
        generator = None

    # Gather depth maps
    depth_files = sorted(glob.glob(os.path.join(args.depth_dir, "*.png")))
    if not depth_files:
        raise FileNotFoundError(f"No depth maps found in {args.depth_dir}")

    print(f"Generating {len(depth_files)} images...")
    for depth_path in tqdm(depth_files):
        fname = os.path.basename(depth_path)
        stem = os.path.splitext(fname)[0]

        depth_img = Image.open(depth_path).convert("RGB").resize(
            (args.resolution, args.resolution)
        )

        # Get caption
        if args.prompt:
            caption = args.prompt
        elif args.caption_dir:
            txt_path = os.path.join(args.caption_dir, stem + ".txt")
            with open(txt_path) as f:
                caption = f.read().strip()
        else:
            raise ValueError("Must provide either --caption_dir or --prompt")

        with torch.no_grad():
            result = pipe(
                caption,
                image=depth_img,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                generator=generator,
            ).images[0]

        result.save(os.path.join(args.output_dir, fname))

    print(f"Done! Generated images saved to {args.output_dir}")


if __name__ == "__main__":
    main()
