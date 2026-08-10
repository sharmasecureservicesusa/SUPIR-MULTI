#!/bin/bash
set -e

CHECKPOINT_DIR="/opt/ComfyUI/models/checkpoints"
SUPIR_DIR="/opt/ComfyUI/models/SUPIR"

mkdir -p "$CHECKPOINT_DIR" "$SUPIR_DIR"

SDXL_PATH="$CHECKPOINT_DIR/sd_xl_base_1.0.safetensors"
SUPIR_PATH="$SUPIR_DIR/SUPIR-v0F.ckpt"

PYTHON_BIN="/opt/environments/python/comfyui/bin/python3"
if [ ! -f "$PYTHON_BIN" ]; then
    PYTHON_BIN="python3"
fi

echo "=== Verifying Base Checkpoints ==="

# 1. Download SDXL Base 1.0 using huggingface_hub
if [ ! -f "$SDXL_PATH" ] || [ ! -s "$SDXL_PATH" ]; then
    echo "Downloading SDXL Base 1.0 via huggingface_hub..."
    "$PYTHON_BIN" -c "
import os
from huggingface_hub import hf_hub_download

token = os.environ.get('HF_TOKEN') or None
print(f'Downloading SDXL Base 1.0 (Token present: {bool(token)})...')

try:
    hf_hub_download(
        repo_id='stabilityai/stable-diffusion-xl-base-1.0',
        filename='sd_xl_base_1.0.safetensors',
        local_dir='$CHECKPOINT_DIR',
        token=token
    )
except Exception as e:
    print(f'Official repo download failed ({e}), trying public un-gated mirror...')
    hf_hub_download(
        repo_id='benjamin-paine/sd-xl-alternative-bases',
        filename='sd_xl_base_1.0_fp16_vae.safetensors',
        local_dir='$CHECKPOINT_DIR'
    )
    os.rename('$CHECKPOINT_DIR/sd_xl_base_1.0_fp16_vae.safetensors', '$SDXL_PATH')
"
    echo "✓ SDXL Base 1.0 verified."
else
    echo "✓ SDXL Base 1.0 checkpoint present."
fi

# 2. Download SUPIR-v0F Weights
if [ ! -f "$SUPIR_PATH" ] || [ ! -s "$SUPIR_PATH" ]; then
    echo "Downloading SUPIR-v0F Weights..."
    "$PYTHON_BIN" -c "
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id='Fanghua-Yu/SUPIR',
    filename='SUPIR-v0F.ckpt',
    local_dir='$SUPIR_DIR'
)
"
    echo "✓ SUPIR-v0F weights verified."
else
    echo "✓ SUPIR-v0F weights present."
fi