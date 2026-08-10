FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    s3fs \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    jq \
    python3 \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Set up Python Virtual Environment
RUN python3 -m venv /opt/environments/python/comfyui
ENV PATH="/opt/environments/python/comfyui/bin:$PATH"

# Upgrade pip & install PyTorch 2.x
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Clone ComfyUI core
RUN git clone https://github.com/comfyanonymous/ComfyUI.git /opt/ComfyUI && \
    pip install --no-cache-dir -r /opt/ComfyUI/requirements.txt

# Clone valid custom node repositories
RUN git clone https://github.com/kijai/ComfyUI-SUPIR.git /opt/ComfyUI/custom_nodes/ComfyUI-SUPIR && \
    git clone https://github.com/ssitu/ComfyUI_UltimateSDUpscale --recursive /opt/ComfyUI/custom_nodes/ComfyUI_UltimateSDUpscale && \
    git clone https://github.com/cubiq/ComfyUI_essentials.git /opt/ComfyUI/custom_nodes/ComfyUI_essentials

# Install custom node dependencies & comfy-kitchen
RUN pip install --no-cache-dir -r /opt/ComfyUI/custom_nodes/ComfyUI-SUPIR/requirements.txt || true

RUN pip install --no-cache-dir --upgrade \
    comfy-kitchen \
    huggingface_hub \
    uvicorn \
    fastapi \
    einops \
    open_clip_torch \
    spandrel \
    scipy \
    pillow \
    accelerate \
    transformers \
    diffusers \
    sentry-sdk

WORKDIR /app
COPY download_models.sh entrypoint.sh run_usdu_batch.py /app/

RUN chmod +x /app/download_models.sh /app/entrypoint.sh

EXPOSE 8000
ENTRYPOINT ["/app/entrypoint.sh"]