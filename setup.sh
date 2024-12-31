#!/bin/bash
# Environment setup for Smart Traffic Scene Augmentation pipeline.
# Run on a fresh RunPod/GCP instance with GPU.

set -e

echo "=== Installing Python dependencies ==="
pip install diffusers transformers accelerate datasets \
    opencv-python-headless tqdm matplotlib pytorch-fid \
    gdown peft torchvision

echo ""
echo "=== Cloning Depth Anything V2 ==="
if [ ! -d "Depth-Anything-V2" ]; then
    git clone https://github.com/DepthAnything/Depth-Anything-V2.git
    cd Depth-Anything-V2
    pip install -r requirements.txt
    cd ..
else
    echo "Depth-Anything-V2 already exists, skipping."
fi

echo ""
echo "=== Downloading checkpoints ==="
mkdir -p checkpoints

declare -A URLS=(
    ["vits"]="https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth"
    ["vitb"]="https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth"
    ["vitl"]="https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth"
)

for variant in vits vitb vitl; do
    CKPT="checkpoints/depth_anything_v2_${variant}.pth"
    if [ ! -f "$CKPT" ]; then
        echo "Downloading ${variant}..."
        wget -q -O "$CKPT" "${URLS[$variant]}"
    else
        echo "${variant} checkpoint already exists."
    fi
done

echo ""
echo "=== Downloading dataset from Google Drive ==="
if [ ! -d "data" ]; then
    mkdir -p data
    gdown "https://drive.usercontent.google.com/download?id=1fpu678kLDu-9cUMceNz2SajyPYBCQ_Sy" -O data/stablediffusion.zip
    cd data
    python -c "import zipfile; zipfile.ZipFile('stablediffusion.zip').extractall('.')"
    cd ..
    echo "Dataset extracted to data/"
else
    echo "data/ directory already exists, skipping download."
fi

echo ""
echo "=== Setup complete! ==="
echo ""
echo "Directory structure:"
echo "  checkpoints/       — Depth Anything V2 model weights"
echo "  Depth-Anything-V2/ — Depth estimation code"
echo "  data/              — KITTI images and captions"
echo "  scripts/           — Pipeline scripts"
echo ""
echo "Next steps:"
echo "  1. bash run_pipeline.sh          # Run full pipeline"
echo "  2. bash run_ablation.sh          # Run ablation study"
