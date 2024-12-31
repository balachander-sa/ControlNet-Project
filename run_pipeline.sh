#!/bin/bash
# Run the full pipeline end-to-end with a single depth variant.
# Default: ViT-L (best quality depth maps).
#
# Usage:
#   bash run_pipeline.sh              # default (vitl)
#   bash run_pipeline.sh vits         # use ViT-S depth

set -e

VARIANT="${1:-vitl}"
IMG_DIR="data/stablediffusion/imgtrain/1_imgtrain"
TXT_DIR="data/stablediffusion/imgtxt"
DEPTH_DIR="outputs/depth_maps_${VARIANT}"
TRAIN_DIR="outputs/train_data_${VARIANT}"
LORA_DIR="outputs/lora_output_${VARIANT}"
GEN_DIR="outputs/generated_images_${VARIANT}"

echo "============================================"
echo "  Smart Traffic Scene Augmentation Pipeline"
echo "  Variant: ${VARIANT}"
echo "============================================"

# Stage 1: Depth extraction
echo ""
echo ">>> Stage 1: Extracting depth maps (${VARIANT})..."
python scripts/extract_depth.py \
    --encoder "${VARIANT}" \
    --img_dir "${IMG_DIR}" \
    --output_dir "${DEPTH_DIR}" \
    --checkpoint_dir checkpoints \
    --depth_anything_path Depth-Anything-V2

# Stage 2: Prepare training data
echo ""
echo ">>> Stage 2: Preparing training data..."
python scripts/prepare_data.py \
    --variant "${VARIANT}" \
    --img_dir "${IMG_DIR}" \
    --txt_dir "${TXT_DIR}" \
    --depth_dir "${DEPTH_DIR}" \
    --output_dir "${TRAIN_DIR}"

# Stage 3A: LoRA training
echo ""
echo ">>> Stage 3A: Training LoRA (this takes ~20 min on A100)..."
python scripts/train_lora.py \
    --data_dir "${TRAIN_DIR}" \
    --output_dir "${LORA_DIR}" \
    --batch_size 4 \
    --num_epochs 50 \
    --learning_rate 1e-4 \
    --lora_rank 4 \
    --save_every 10

# Stage 3B: Generate images
echo ""
echo ">>> Stage 3B: Generating augmented images..."
python scripts/generate.py \
    --depth_dir "${DEPTH_DIR}" \
    --caption_dir "${TXT_DIR}" \
    --lora_dir "${LORA_DIR}/checkpoint-epoch-50" \
    --output_dir "${GEN_DIR}"

# Stage 4: Evaluation
echo ""
echo ">>> Stage 4: Evaluating..."
python scripts/evaluate.py \
    --real_dir "${IMG_DIR}" \
    --gen_dir "${GEN_DIR}" \
    --depth_dir "${DEPTH_DIR}" \
    --checkpoint_dir checkpoints \
    --depth_anything_path Depth-Anything-V2

echo ""
echo "============================================"
echo "  Pipeline complete!"
echo "  Results in: outputs/"
echo "============================================"
