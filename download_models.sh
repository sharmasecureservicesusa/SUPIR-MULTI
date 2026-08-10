#!/bin/bash
set -e

CHECKPOINT_DIR="/opt/ComfyUI/models/checkpoints"
SUPIR_DIR="/opt/ComfyUI/models/SUPIR"

mkdir -p "$CHECKPOINT_DIR" "$SUPIR_DIR"

SDXL_PATH="$CHECKPOINT_DIR/sd_xl_base_1.0.safetensors"
SUPIR_PATH="$SUPIR_DIR/SUPIR-v0F.ckpt"
SUPIR_CKPT_LINK="$CHECKPOINT_DIR/SUPIR-v0F.ckpt"

PYTHON_BIN="/opt/environments/python/comfyui/bin/python3"
if [ ! -f "$PYTHON_BIN" ]; then
    PYTHON_BIN="python3"
fi

echo "=== Verifying Model Weights & Checkpoints ==="

# 1. Download SDXL Base 1.0
if [ ! -f "$SDXL_PATH" ] || [ ! -s "$SDXL_PATH" ]; then
    echo "Downloading SDXL Base 1.0 (6.9 GB)..."
    "$PYTHON_BIN" -c "
import os
from huggingface_hub import hf_hub_download

token = os.environ.get('HF_TOKEN')
if token and not token.startswith('hf_'):
    token = None

download_success = False

if token:
    try:
        print('Attempting download from official stabilityai/stable-diffusion-xl-base-1.0...')
        hf_hub_download(
            repo_id='stabilityai/stable-diffusion-xl-base-1.0',
            filename='sd_xl_base_1.0.safetensors',
            local_dir='$CHECKPOINT_DIR',
            token=token
        )
        download_success = True
    except Exception as e:
        print(f'Official repository download failed ({e}).')

if not download_success:
    print('Downloading from un-gated public mirror (Pie31415/stable-diffusion-xl-base-1.0)...')
    hf_hub_download(
        repo_id='Pie31415/stable-diffusion-xl-base-1.0',
        filename='sd_xl_base_1.0.safetensors',
        local_dir='$CHECKPOINT_DIR',
        token=False
    )
"
    echo "✓ SDXL Base 1.0 verified."
else
    echo "✓ SDXL Base 1.0 checkpoint present."
fi

# 2. Download SUPIR-v0F Weights & Create Symlink in checkpoints
if [ ! -f "$SUPIR_PATH" ] || [ ! -s "$SUPIR_PATH" ]; then
    echo "Downloading SUPIR-v0F Weights (10.3 GB)..."
    "$PYTHON_BIN" -c "
import os
from huggingface_hub import hf_hub_download

print('Downloading SUPIR-v0F from public repository (camenduru/SUPIR)...')
try:
    hf_hub_download(
        repo_id='camenduru/SUPIR',
        filename='SUPIR-v0F.ckpt',
        local_dir='$SUPIR_DIR',
        token=False
    )
except Exception as e:
    print(f'camenduru download failed ({e}), trying ashleykleynhans/SUPIR...')
    hf_hub_download(
        repo_id='ashleykleynhans/SUPIR',
        filename='SUPIR-v0F.ckpt',
        local_dir='$SUPIR_DIR',
        token=False
    )
"
    echo "✓ SUPIR-v0F weights verified."
else
    echo "✓ SUPIR-v0F weights present."
fi

# Symlink to checkpoints directory so all ComfyUI node variants resolve it
if [ ! -f "$SUPIR_CKPT_LINK" ]; then
    ln -sf "$SUPIR_PATH" "$SUPIR_CKPT_LINK"
fi