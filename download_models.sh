#!/bin/bash
set -e

CHECKPOINT_DIR="/opt/ComfyUI/models/checkpoints"
SUPIR_DIR="/opt/ComfyUI/models/SUPIR"

mkdir -p "$CHECKPOINT_DIR" "$SUPIR_DIR"

SDXL_PATH="$CHECKPOINT_DIR/sd_xl_base_1.0.safetensors"
SUPIR_PATH="$SUPIR_DIR/SUPIR-v0F.ckpt"

# Build authorization header if HF_TOKEN is provided
AUTH_HEADER=()
if [ -n "$HF_TOKEN" ]; then
    AUTH_HEADER=(-H "Authorization: Bearer $HF_TOKEN")
fi

echo "=== Verifying Base Checkpoints ==="

# 1. Download SDXL Base 1.0 via single-stream curl to prevent CDN 403 chunk blocks
if [ ! -f "$SDXL_PATH" ] || [ ! -s "$SDXL_PATH" ]; then
    echo "Downloading SDXL Base 1.0 (6.9 GB)..."
    curl -fL "${AUTH_HEADER[@]}" --retry 5 --retry-delay 3 \
        "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors" \
        -o "$SDXL_PATH"
    echo "✓ SDXL Base 1.0 downloaded successfully."
else
    echo "✓ SDXL Base 1.0 checkpoint present."
fi

# 2. Download SUPIR-v0F Model
if [ ! -f "$SUPIR_PATH" ] || [ ! -s "$SUPIR_PATH" ]; then
    echo "Downloading SUPIR-v0F Weights (10.3 GB)..."
    curl -fL "${AUTH_HEADER[@]}" --retry 5 --retry-delay 3 \
        "https://huggingface.co/Fanghua-Yu/SUPIR/resolve/main/SUPIR-v0F.ckpt" \
        -o "$SUPIR_PATH"
    echo "✓ SUPIR-v0F weights downloaded successfully."
else
    echo "✓ SUPIR-v0F weights present."
fi