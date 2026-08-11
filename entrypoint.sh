#!/bin/bash
set -e

# Memory allocation and CUDA runtime options
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,garbage_collection_threshold:0.8,max_split_size_mb:512"
export CUDA_MODULE_LOADING="LAZY"
export CUDNN_BENCHMARK="1"
export HF_HUB_DISABLE_IMPLICIT_TOKEN_WARNING="1"

CPU_CORES=$(nproc)
export OMP_NUM_THREADS=$CPU_CORES
export MKL_NUM_THREADS=$CPU_CORES

PYTHON_BIN="/opt/environments/python/comfyui/bin/python3"
if [ ! -f "$PYTHON_BIN" ]; then
    PYTHON_BIN="python3"
fi

echo "=== Applying PyTorch / comfy_kitchen Compatibility Patches ==="
PYTHON_SITE=$("$PYTHON_BIN" -c "import site; print(site.getsitepackages()[0])" 2>/dev/null || true)
NA_FILE="$PYTHON_SITE/comfy_kitchen/backends/eager/na.py"

if [ -f "$NA_FILE" ]; then
    sed -i 's/kernel_size: list\[int\]/kernel_size: typing.List[int]/g' "$NA_FILE" || true
    sed -i 's/is_causal: list\[bool\]/is_causal: typing.List[bool]/g' "$NA_FILE" || true
    sed -i 's/list\[int\]/typing.List[int]/g' "$NA_FILE" || true
    sed -i 's/list\[bool\]/typing.List[bool]/g' "$NA_FILE" || true
    if ! grep -q "import typing" "$NA_FILE"; then
        sed -i '1s/^/import typing\n/' "$NA_FILE" || true
    fi
    echo "✓ Successfully patched comfy_kitchen type annotations for PyTorch infer_schema."
fi

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

if [ "$MODE" = "endpoint" ]; then
    echo "=== Starting Multi-GPU Endpoint Service (Port 8000) ==="
    exec "$PYTHON_BIN" -m uvicorn server:app --host 0.0.0.0 --port 8000
else
    echo "=== Starting High-Throughput SUPIR Batch Job ==="
    exec "$PYTHON_BIN" /app/run_usdu_batch.py
fi
    