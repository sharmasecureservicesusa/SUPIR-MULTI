#!/bin/bash
set -e

# --- PyTorch & CUDA Tuning for Dual L40S Performance ---
# expandable_segments prevents CUDA Out-Of-Memory errors caused by dynamic memory fragmentation
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,garbage_collection_threshold:0.8,max_split_size_mb:512"
export CUDA_MODULE_LOADING="LAZY"
export CUDNN_BENCHMARK="1"

# Dynamically bind thread pools to match your 64 vCPUs
CPU_CORES=$(nproc)
export OMP_NUM_THREADS=$CPU_CORES
export MKL_NUM_THREADS=$CPU_CORES

echo "=== Mounting Nebius Object Storage & Preparing RAM Disk ==="
mkdir -p /mnt/s3bucket /tmp/s3cache /dev/shm/batch_input /dev/shm/batch_output

if [ -n "$S3_ACCESS_KEY" ] && [ -n "$S3_SECRET_KEY" ]; then
    echo "${S3_ACCESS_KEY}:${S3_SECRET_KEY}" > /tmp/passwd-s3fs
    chmod 600 /tmp/passwd-s3fs

    # Tuned s3fs options: caching, stat caching, kernel cache, and parallel chunk fetching
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
         /opt/ComfyUI/models/SUPIR \
         /opt/ComfyUI/models/clip \
         /opt/ComfyUI/models/vae

echo "=== Checking Required Model Files ==="

# Trigger parallel model downloading via aria2 if script exists
if [ -f "/app/download_models.sh" ]; then
    /app/download_models.sh
fi

echo "✓ All required SUPIR models verified."

# Automatically select environment python executable
PYTHON_BIN="/opt/environments/python/comfyui/bin/python3"
if [ ! -f "$PYTHON_BIN" ]; then
    PYTHON_BIN="python3"
fi

if [ "$MODE" = "endpoint" ]; then
    echo "=== Starting Multi-GPU Endpoint Service (Port 8000) ==="
    exec uvicorn server:app --host 0.0.0.0 --port 8000
else
    echo "=== Starting High-Throughput SUPIR Dual-L40S Batch Job ==="
    exec "$PYTHON_BIN" /app/run_usdu_batch.py
fi