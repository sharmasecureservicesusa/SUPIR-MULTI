#!/bin/bash
set -e

CHECKPOINT_DIR="/opt/ComfyUI/models/checkpoints"
SUPIR_DIR="/opt/ComfyUI/models/SUPIR"

mkdir -p "$CHECKPOINT_DIR" "$SUPIR_DIR"

SDXL_PATH="$CHECKPOINT_DIR/sd_xl_base_1.0.safetensors"
SUPIR_PATH="$SUPIR_DIR/SUPIR-v0F.ckpt"

echo "=== Verifying Base Checkpoints ==="

# 1. Download SDXL Base 1.0 (Uses HF_TOKEN if present, falls back to public mirror)
if [ ! -f "$SDXL_PATH" ] || [ ! -s "$SDXL_PATH" ]; then
    echo "Downloading SDXL Base 1.0 (6.9 GB)..."
    if [ -n "$HF_TOKEN" ]; then
        curl -fL -H "Authorization: Bearer $HF_TOKEN" --retry 5 --retry-delay 3 \
            "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors" \
            -o "$SDXL_PATH"
    else
        # Un-gated public mirror URL requiring no authentication
        curl -fL --retry 5 --retry-delay 3 \
            "https://huggingface.co/bdsqlsz/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors" \
            -o "$SDXL_PATH"
    fi
    echo "✓ SDXL Base 1.0 downloaded successfully."
else
    echo "✓ SDXL Base 1.0 checkpoint present."
fi

# 2. Download SUPIR-v0F Model Weights
if [ ! -f "$SUPIR_PATH" ] || [ ! -s "$SUPIR_PATH" ]; then
    echo "Downloading SUPIR-v0F Weights (10.3 GB)..."
    curl -fL --retry 5 --retry-delay 3 \
        "https://huggingface.co/Fanghua-Yu/SUPIR/resolve/main/SUPIR-v0F.ckpt" \
        -o "$SUPIR_PATH"
    echo "✓ SUPIR-v0F weights downloaded successfully."
else
    echo "✓ SUPIR-v0F weights present."
fi