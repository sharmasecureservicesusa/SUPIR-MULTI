#!/bin/bash
set -e

CHECKPOINT_DIR="/opt/ComfyUI/models/checkpoints"
SUPIR_DIR="/opt/ComfyUI/models/SUPIR"
CLIP_DIR="/opt/ComfyUI/models/clip"
VAE_DIR="/opt/ComfyUI/models/vae"

mkdir -p "$CHECKPOINT_DIR" "$SUPIR_DIR" "$CLIP_DIR" "$VAE_DIR"

# Download helper using multi-connection aria2c (if available) or resilient curl
download_file() {
    local url="$1"
    local dest_dir="$2"
    local file_name="$3"
    local target="$dest_dir/$file_name"

    if [ -f "$target" ]; then
        echo "✓ $file_name already exists. Skipping."
        return 0
    fi

    echo "Downloading $file_name..."
    if command -v aria2c &> /dev/null; then
        aria2c -x 16 -s 16 -k 1M --console-log-level=warn -d "$dest_dir" -o "$file_name" "$url"
    else
        curl -fL -C - --retry 5 --retry-delay 2 --retry-connrefused "$url" -o "$target"
    fi
}

# 1. Download SDXL Base Checkpoint
download_file \
    "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors" \
    "$CHECKPOINT_DIR" \
    "sd_xl_base_1.0.safetensors"

# 2. Download SUPIR-v0F Model Weights (Fidelity variant for fast restoration)
download_file \
    "https://huggingface.co/Fanghua-Yu/SUPIR/resolve/main/SUPIR-v0F.ckpt" \
    "$SUPIR_DIR" \
    "SUPIR-v0F.ckpt"

echo "✓ All SUPIR and SDXL models verified successfully!"