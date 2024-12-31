"""
Stage 3A: LoRA Fine-Tuning with ControlNet Depth Conditioning.

Trains LoRA adapters on the UNet of Stable Diffusion 1.5, with a frozen
ControlNet-depth model providing spatial conditioning from depth maps.

Usage:
    python scripts/train_lora.py \
        --data_dir outputs/train_data_vitl \
        --output_dir outputs/lora_output_vitl \
        --batch_size 4 \
        --num_epochs 50 \
        --learning_rate 1e-4 \
        --lora_rank 4 \
        --save_every 10
"""

import argparse
import json
import os

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from torchvision import transforms

from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    DDPMScheduler,
    UNet2DConditionModel,
)
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from transformers import CLIPTextModel, CLIPTokenizer


class ControlNetLoRADataset(Dataset):
    """Dataset that loads image/depth/caption triplets from metadata.jsonl."""

    def __init__(self, data_dir: str, tokenizer, resolution: int = 512):
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.resolution = resolution

        metadata_path = os.path.join(data_dir, "metadata.jsonl")
        self.samples = []
        with open(metadata_path) as f:
            for line in f:
                self.samples.append(json.loads(line))

        self.image_transforms = transforms.Compose([
            transforms.Resize((resolution, resolution)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

        self.conditioning_transforms = transforms.Compose([
            transforms.Resize((resolution, resolution)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        image = Image.open(os.path.join(self.data_dir, sample["file_name"])).convert("RGB")
        image = self.image_transforms(image)

        cond = Image.open(os.path.join(self.data_dir, sample["conditioning_image"])).convert("RGB")
        cond = self.conditioning_transforms(cond)

        tokens = self.tokenizer(
            sample["text"],
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )

        return {
            "pixel_values": image,
            "conditioning_pixel_values": cond,
            "input_ids": tokens.input_ids.squeeze(0),
        }


def main():
    parser = argparse.ArgumentParser(description="Train LoRA adapters with ControlNet conditioning")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs/lora_output")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lora_rank", type=int, default=4)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--model_id", type=str, default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda")

    # --- Load models ---
    print("Loading models...")
    tokenizer = CLIPTokenizer.from_pretrained(args.model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.model_id, subfolder="text_encoder").to(device)
    vae = AutoencoderKL.from_pretrained(args.model_id, subfolder="vae").to(device)
    unet = UNet2DConditionModel.from_pretrained(args.model_id, subfolder="unet")
    controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-depth").to(device)
    noise_scheduler = DDPMScheduler.from_pretrained(args.model_id, subfolder="scheduler")

    # Freeze everything except LoRA
    text_encoder.requires_grad_(False)
    vae.requires_grad_(False)
    controlnet.requires_grad_(False)

    # Add LoRA adapters to UNet
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    unet.add_adapter(lora_config)
    unet = unet.to(device)

    trainable = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    total = sum(p.numel() for p in unet.parameters())
    print(f"LoRA trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    # --- Dataset ---
    dataset = ControlNetLoRADataset(args.data_dir, tokenizer, args.resolution)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    print(f"Dataset: {len(dataset)} samples, {len(dataloader)} batches per epoch")

    # --- Optimizer ---
    lora_params = [p for p in unet.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(lora_params, lr=args.learning_rate, weight_decay=1e-2)

    # --- Training loop ---
    print(f"\nStarting training: {args.num_epochs} epochs, batch_size={args.batch_size}, lr={args.learning_rate}")

    for epoch in range(args.num_epochs):
        epoch_loss = 0.0
        progress = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{args.num_epochs}")

        for batch in progress:
            pixel_values = batch["pixel_values"].to(device, dtype=torch.float32)
            cond_values = batch["conditioning_pixel_values"].to(device, dtype=torch.float32)
            input_ids = batch["input_ids"].to(device)

            # Encode image to latent space & get text embeddings
            with torch.no_grad():
                latents = vae.encode(pixel_values).latent_dist.sample() * vae.config.scaling_factor
                encoder_hidden_states = text_encoder(input_ids)[0]

            # Forward diffusion
            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=device
            ).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # ControlNet conditioning (frozen)
            with torch.no_grad():
                down_block_res_samples, mid_block_res_sample = controlnet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    controlnet_cond=cond_values,
                    return_dict=False,
                )

            # UNet prediction (LoRA layers are trainable)
            model_pred = unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=encoder_hidden_states,
                down_block_additional_residuals=down_block_res_samples,
                mid_block_additional_residual=mid_block_res_sample,
            ).sample

            loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            progress.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch + 1} avg loss: {avg_loss:.4f}")

        # Checkpoint
        if (epoch + 1) % args.save_every == 0 or (epoch + 1) == args.num_epochs:
            save_path = os.path.join(args.output_dir, f"checkpoint-epoch-{epoch + 1}")
            os.makedirs(save_path, exist_ok=True)

            lora_state_dict = get_peft_model_state_dict(unet)
            torch.save(lora_state_dict, os.path.join(save_path, "lora_weights.pt"))
            unet.save_pretrained(save_path)
            print(f"Saved LoRA weights to {save_path}")

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
