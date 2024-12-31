# Depth-Conditioned Scene Augmentation for Autonomous Driving

A pipeline for generating realistic augmented driving scenes using monocular depth estimation and diffusion models. Built as an ablation study comparing Depth Anything V2 encoder variants (ViT-S, ViT-B, ViT-L) as conditioning inputs for ControlNet-guided image generation with LoRA fine-tuning.

## Pipeline Overview

```
RGB Image → Depth Anything V2 → Depth Map
                                    ↓
                            ControlNet-Depth
                                    ↓
Caption (.txt) → SD 1.5 + LoRA → Augmented Image
                                    ↓
                          FID + Depth Consistency
```

**Four stages:**

1. **Depth Extraction** — Monocular depth estimation with [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) (ViT-S/B/L)
2. **Data Preparation** — Align images, captions, and depth maps into training format
3. **LoRA Training + Generation** — Fine-tune SD 1.5 UNet with LoRA adapters, conditioned on depth via frozen [ControlNet-depth](https://huggingface.co/lllyasviel/sd-controlnet-depth)
4. **Evaluation** — FID (perceptual quality) and depth consistency metrics (AbsRel, RMSE)

## Results

| Depth Model | FID ↓ | AbsRel ↓ | RMSE ↓ |
|---|---|---|---|
| ViT-S (25M) | 119.85 | **0.1656 ± 0.054** | **0.0534 ± 0.020** |
| ViT-B (97M) | **117.15** | 0.1706 ± 0.063 | 0.0549 ± 0.024 |
| ViT-L (335M) | 120.34 | 0.1706 ± 0.056 | 0.0554 ± 0.020 |

**Key finding:** Larger depth models do not monotonically improve generation quality. ViT-B achieves the best FID, while ViT-S achieves the best depth consistency — suggesting that the ControlNet's internal depth representation (trained on MiDaS) creates a distribution mismatch with higher-capacity encoders.

## Quick Start

```bash
# 1. Setup environment + download data & checkpoints
bash setup.sh

# 2. Run the full pipeline (single variant)
bash run_pipeline.sh vitl

# 3. Or run the full ablation study
bash run_ablation.sh
```

## Repository Structure

```
├── paper/
│   └── Report.pdf            # Project Report
├── scripts/
│   ├── extract_depth.py      # Stage 1: Depth map extraction
│   ├── prepare_data.py       # Stage 2: Align data into training format
│   ├── train_lora.py         # Stage 3A: LoRA fine-tuning with ControlNet
│   ├── generate.py           # Stage 3B: Depth-conditioned image generation
│   └── evaluate.py           # Stage 4: FID + depth consistency metrics

├── utils/
│   └── visualize.py          # Plotting utilities
├── setup.sh                  # Environment setup
├── run_pipeline.sh           # Run full pipeline (single variant)
├── run_ablation.sh           # Run ablation study (all variants)
├── requirements.txt
└── README.md
```

## Training Details

- **Base model:** Stable Diffusion 1.5
- **ControlNet:** `lllyasviel/sd-controlnet-depth` (frozen during training)
- **LoRA config:** rank=4, alpha=4, targeting `to_q`, `to_k`, `to_v`, `to_out.0` (797K trainable / 860M total params)
- **Training:** 50 epochs, batch size 4, AdamW (lr=1e-4, wd=0.01), 512×512 resolution
- **Dataset:** 181 KITTI driving images with auto-generated captions
- **Hardware:** NVIDIA A100 40GB (~40 min per variant)

## Custom Generation

Generate images with custom prompts while preserving scene geometry:

```bash
python scripts/generate.py \
    --depth_dir outputs/depth_maps_vitl \
    --lora_dir outputs/lora_output_vitl/checkpoint-epoch-50 \
    --output_dir outputs/custom \
    --prompt "a rainy night urban street with heavy traffic, wet road, reflections"
```

## Requirements

- Python 3.10+
- NVIDIA GPU with ≥16GB VRAM (24GB+ recommended)
- ~15GB disk for model weights
- ~5GB disk for dataset + outputs

## Acknowledgments

- [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) — Yang et al.
- [Stable Diffusion](https://github.com/CompVis/stable-diffusion) — Rombach et al.
- [ControlNet](https://github.com/lllyasviel/ControlNet) — Zhang et al.
- [HuggingFace Diffusers](https://github.com/huggingface/diffusers)
- [KITTI Dataset](http://www.cvlibs.net/datasets/kitti/)
