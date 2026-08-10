#!/bin/bash
set -e

# --- PyTorch & CUDA Tuning ---
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,garbage_collection_threshold:0.8,max_split_size_mb:512"
export CUDA_MODULE_LOADING="LAZY"
export CUDNN_BENCHMARK="1"

CPU_CORES=$(nproc)
export OMP_NUM_THREADS=$CPU_CORES
export MKL_NUM_THREADS=$CPU_CORES

echo "=== Mounting Object Storage & Preparing RAM Disk ==="
mkdir -p /mnt/s3bucket /tmp/s3cache /dev/shm/batch_input /dev/shm/batch_output

if [ -n "$S3_ACCESS_KEY" ] && [ -n "$S3_SECRET_KEY" ]; then
    echo "${S3_ACCESS_KEY}:${S3_SECRET_KEY}" > /tmp/passwd-s3fs
    chmod 600 /tmp/passwd-s3fs

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
        -o attr_timeout=0 \
        -o entry_timeout=0
    echo "✓ Storage mounted successfully."
fi

echo "=== Checking Model Files ==="
if [ -f "/app/download_models.sh" ]; then
    chmod +x /app/download_models.sh
    /app/download_models.sh
fi

PYTHON_BIN="/opt/environments/python/comfyui/bin/python3"
if [ ! -f "$PYTHON_BIN" ]; then
    PYTHON_BIN="python3"
fi

if [ "$MODE" = "endpoint" ]; then
    echo "=== Starting Multi-GPU Endpoint Service (Port 8000) ==="
    exec "$PYTHON_BIN" -m uvicorn server:app --host 0.0.0.0 --port 8000
else
    echo "=== Starting High-Throughput SUPIR Batch Job ==="
    exec "$PYTHON_BIN" /app/run_usdu_batch.py
fi