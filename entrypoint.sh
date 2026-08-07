#!/bin/bash
set -e

# --- PyTorch & CUDA Tuning for L40S Performance ---
# expandable_segments prevents CUDA Out-Of-Memory errors caused by dynamic memory fragmentation
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,garbage_collection_threshold:0.8,max_split_size_mb:512"
export CUDA_MODULE_LOADING="LAZY"
export CUDNN_BENCHMARK="1"

# Dynamically bind thread pools to actual allocated CPU core counts
CPU_CORES=$(nproc)
export OMP_NUM_THREADS=$CPU_CORES
export MKL_NUM_THREADS=$CPU_CORES

echo "=== Mounting Nebius Object Storage ==="
mkdir -p /mnt/s3bucket /tmp/s3cache

if [ -n "$S3_ACCESS_KEY" ] && [ -n "$S3_SECRET_KEY" ]; then
    echo "${S3_ACCESS_KEY}:${S3_SECRET_KEY}" > /tmp/passwd-s3fs
    chmod 600 /tmp/passwd-s3fs

    # Tuned s3fs options: added caching, stat caching, kernel cache, and parallel chunk fetching for maximum I/O throughput
    s3fs "${S3_BUCKET_NAME:-ai-upscale-bucket}" /mnt/s3bucket \
        -o passwd_file=/tmp/passwd-s3fs \
        -o url=https://storage.eu-north1.nebius.cloud \
        -o use_path_request_style \
        -o allow_other \
        -o max_stat_cache_size=100000 \
        -o use_cache=/tmp/s3cache \
        -o multipart_size=64 \
        -o parallel_count=16 \
        -o kernel_cache \
        -o drop_attr_cache
    echo "✓ Storage mounted successfully with high-throughput I/O parameters."
fi

echo "=== Creating Required Directory Paths ==="
mkdir -p /opt/ComfyUI/models/checkpoints \
         /opt/ComfyUI/models/SUPIR

echo "=== Checking Required Model Files ==="

CHECKPOINT_PATH="/opt/ComfyUI/models/checkpoints/sd_xl_base_1.0.safetensors"
if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "Downloading SDXL Base Model..."
    curl -fL -C - --retry 5 --retry-delay 2 "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors" \
         -o "$CHECKPOINT_PATH"
fi

SUPIR_PATH="/opt/ComfyUI/models/SUPIR/SUPIR-v0F.ckpt"
if [ ! -f "$SUPIR_PATH" ]; then
    echo "Downloading SUPIR-v0F Model Weights..."
    curl -fL -C - --retry 5 --retry-delay 2 "https://huggingface.co/Fanghua-Yu/SUPIR/resolve/main/SUPIR-v0F.ckpt" \
         -o "$SUPIR_PATH"
fi

echo "✓ All required SUPIR models verified."

# Automatically select environment python executable
PYTHON_BIN="/opt/environments/python/comfyui/bin/python3"
if [ ! -f "$PYTHON_BIN" ]; then
    PYTHON_BIN="python3"
fi

if [ "$MODE" = "endpoint" ]; then
    echo "=== Starting Endpoint Service (Port 8000) ==="
    exec uvicorn server:app --host 0.0.0.0 --port 8000
else
    echo "=== Starting SUPIR Batch Job ==="
    exec "$PYTHON_BIN" /app/run_usdu_batch.py
fi