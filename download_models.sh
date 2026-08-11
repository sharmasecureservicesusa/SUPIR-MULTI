#!/bin/bash
set -e

CHECKPOINT_DIR="/opt/ComfyUI/models/checkpoints"
SUPIR_DIR="/opt/ComfyUI/models/SUPIR"
S3_MOUNT="/mnt/s3bucket"

mkdir -p "$CHECKPOINT_DIR" "$SUPIR_DIR"

SDXL_DEST="$CHECKPOINT_DIR/sd_xl_base_1.0.safetensors"
SUPIR_DEST="$SUPIR_DIR/SUPIR-v0F.ckpt"
SUPIR_LINK="$CHECKPOINT_DIR/SUPIR-v0F.ckpt"

MIN_SDXL_SIZE=6000000000    # ~6.0 GB
MIN_SUPIR_SIZE=10000000000   # ~10.0 GB

echo "=== Syncing Model Checkpoints from S3 Mount ==="

# 1. Sync SDXL Base 1.0
if [ ! -f "$SDXL_DEST" ] || [ $(stat -c%s "$SDXL_DEST" 2>/dev/null || echo 0) -lt $MIN_SDXL_SIZE ]; then
    SDXL_SRC=""
    if [ -f "$S3_MOUNT/models/checkpoints/sd_xl_base_1.0.safetensors" ]; then
        SDXL_SRC="$S3_MOUNT/models/checkpoints/sd_xl_base_1.0.safetensors"
    elif [ -f "$S3_MOUNT/models/sd_xl_base_1.0.safetensors" ]; then
        SDXL_SRC="$S3_MOUNT/models/sd_xl_base_1.0.safetensors"
    fi

    if [ -n "$SDXL_SRC" ]; then
        SRC_SIZE=$(stat -c%s "$SDXL_SRC" 2>/dev/null || echo 0)
        if [ "$SRC_SIZE" -ge $MIN_SDXL_SIZE ]; then
            echo "Copying SDXL Base 1.0 from $SDXL_SRC..."
            cp "$SDXL_SRC" "$SDXL_DEST"
            echo "✓ SDXL Base 1.0 synced successfully."
        else
            echo "❌ Found $SDXL_SRC but size ($SRC_SIZE bytes) is below minimum required (~6 GB)."
            exit 1
        fi
    else
        echo "❌ SDXL Base 1.0 not found in $S3_MOUNT/models/checkpoints/ or $S3_MOUNT/models/"
        exit 1
    fi
else
    echo "✓ SDXL Base 1.0 present and valid."
fi

# 2. Sync SUPIR-v0F
if [ ! -f "$SUPIR_DEST" ] || [ $(stat -c%s "$SUPIR_DEST" 2>/dev/null || echo 0) -lt $MIN_SUPIR_SIZE ]; then
    SUPIR_SRC=""
    if [ -f "$S3_MOUNT/models/SUPIR-v0F.ckpt" ]; then
        SUPIR_SRC="$S3_MOUNT/models/SUPIR-v0F.ckpt"
    elif [ -f "$S3_MOUNT/models/checkpoints/SUPIR-v0F.ckpt" ]; then
        SUPIR_SRC="$S3_MOUNT/models/checkpoints/SUPIR-v0F.ckpt"
    fi

    if [ -n "$SUPIR_SRC" ]; then
        SRC_SIZE=$(stat -c%s "$SUPIR_SRC" 2>/dev/null || echo 0)
        if [ "$SRC_SIZE" -ge $MIN_SUPIR_SIZE ]; then
            echo "Copying SUPIR-v0F from $SUPIR_SRC..."
            cp "$SUPIR_SRC" "$SUPIR_DEST"
            echo "✓ SUPIR-v0F synced successfully."
        else
            echo "❌ Found $SUPIR_SRC but file size ($SRC_SIZE bytes) is incomplete. Requires ~10.3 GB."
            exit 1
        fi
    else
        echo "❌ SUPIR-v0F not found in $S3_MOUNT/models/ or $S3_MOUNT/models/checkpoints/"
        exit 1
    fi
else
    echo "✓ SUPIR-v0F present and valid."
fi

# Create symlink for nodes resolving SUPIR inside checkpoints folder
ln -sf "$SUPIR_DEST" "$SUPIR_LINK"
        