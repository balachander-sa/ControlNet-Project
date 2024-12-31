#!/bin/bash
# Run the full ablation study across all three depth variants.
# This trains and evaluates ViT-S, ViT-B, and ViT-L independently.
#
# Estimated time on A100 80GB: ~2 hours total
# Estimated cost on RunPod A100: ~$4-5

set -e

IMG_DIR="data/stablediffusion/imgtrain/1_imgtrain"
TXT_DIR="data/stablediffusion/imgtxt"

echo "============================================"
echo "  Ablation Study: Depth Variant Comparison"
echo "  Variants: ViT-S (25M), ViT-B (97M), ViT-L (335M)"
echo "============================================"

# Stage 1: Extract depth maps for all variants
echo ""
echo ">>> Stage 1: Extracting depth maps for all variants..."
python scripts/extract_depth.py \
    --encoder all \
    --img_dir "${IMG_DIR}" \
    --output_dir outputs/depth_maps \
    --checkpoint_dir checkpoints \
    --depth_anything_path Depth-Anything-V2

# Stage 2: Prepare training data for all variants
echo ""
echo ">>> Stage 2: Preparing training data..."
python scripts/prepare_data.py \
    --variant all \
    --img_dir "${IMG_DIR}" \
    --txt_dir "${TXT_DIR}" \
    --depth_base_dir outputs/depth_maps \
    --output_base_dir outputs/train_data

# Stages 3-4: Train, generate, and evaluate each variant
for VARIANT in vits vitb vitl; do
    echo ""
    echo "============================================"
    echo "  Processing: ${VARIANT}"
    echo "============================================"

    TRAIN_DIR="outputs/train_data_${VARIANT}"
    LORA_DIR="outputs/lora_output_${VARIANT}"
    GEN_DIR="outputs/generated_images_${VARIANT}"
    DEPTH_DIR="outputs/depth_maps_${VARIANT}"

    # Train
    echo ">>> Training LoRA (${VARIANT})..."
    python scripts/train_lora.py \
        --data_dir "${TRAIN_DIR}" \
        --output_dir "${LORA_DIR}" \
        --batch_size 4 \
        --num_epochs 50 \
        --learning_rate 1e-4 \
        --lora_rank 4 \
        --save_every 50

    # Generate
    echo ">>> Generating images (${VARIANT})..."
    python scripts/generate.py \
        --depth_dir "${DEPTH_DIR}" \
        --caption_dir "${TXT_DIR}" \
        --lora_dir "${LORA_DIR}/checkpoint-epoch-50" \
        --output_dir "${GEN_DIR}"

    # Evaluate
    echo ">>> Evaluating (${VARIANT})..."
    python scripts/evaluate.py \
        --real_dir "${IMG_DIR}" \
        --gen_dir "${GEN_DIR}" \
        --depth_dir "${DEPTH_DIR}" \
        --checkpoint_dir checkpoints \
        --depth_anything_path Depth-Anything-V2

    echo ""
done

echo "============================================"
echo "  Ablation study complete!"
echo "  Compare results across variants above."
echo "============================================"
