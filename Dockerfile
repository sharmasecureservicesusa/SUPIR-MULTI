FROM ghcr.io/ai-dock/comfyui:latest-cuda

LABEL org.opencontainers.image.source="https://github.com/adminsharmasecureservicescausa/nebiusupscale"

WORKDIR /app

ENV PIPX_HOME=/opt/pipx
ENV PIPX_BIN_DIR=/usr/local/bin
ENV PYTHONUNBUFFERED=1

# 1. Install system utilities and aria2 for accelerated multi-connection downloads
RUN apt-get update && apt-get install -y \
    s3fs \
    dos2unix \
    wget \
    git \
    aria2 \
    python3 \
    python3-pip \
    pipx \
    && rm -rf /var/lib/apt/lists/*

# 2. Install global endpoint dependencies via pipx
RUN pipx install uvicorn && \
    pipx inject uvicorn fastapi python-multipart

# 3. Clone SUPIR custom node
RUN git config --global --add safe.directory '*' && \
    mkdir -p /opt/ComfyUI/custom_nodes && \
    rm -rf /opt/ComfyUI/custom_nodes/ComfyUI-SUPIR && \
    git clone --depth 1 https://github.com/kijai/ComfyUI-SUPIR /opt/ComfyUI/custom_nodes/ComfyUI-SUPIR

# 4. Install SUPIR dependencies directly into ComfyUI's primary Python runtime
RUN PYTHON_BIN=$(which python3); \
    if [ -f "/opt/environments/python/comfyui/bin/python3" ]; then PYTHON_BIN="/opt/environments/python/comfyui/bin/python3"; fi; \
    $PYTHON_BIN -m pip install --no-cache-dir \
        einops \
        open_clip_torch \
        spandex \
        scipy \
        -r /opt/ComfyUI/custom_nodes/ComfyUI-SUPIR/requirements.txt

# 5. Pre-create required model directories
RUN mkdir -p /opt/ComfyUI/models/checkpoints \
             /opt/ComfyUI/models/SUPIR \
             /opt/ComfyUI/models/clip \
             /opt/ComfyUI/models/vae

# 6. Copy application scripts
COPY . /app

RUN dos2unix /app/entrypoint.sh /app/download_models.sh 2>/dev/null || true && \
    chmod +x /app/entrypoint.sh /app/download_models.sh 2>/dev/null || true

ENTRYPOINT ["/app/entrypoint.sh"]