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

RUN python3 -m venv /opt/environments/python/comfyui
ENV PATH="/opt/environments/python/comfyui/bin:$PATH"

RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

RUN git clone https://github.com/comfyanonymous/ComfyUI.git /opt/ComfyUI && \
    pip install --no-cache-dir -r /opt/ComfyUI/requirements.txt

RUN git clone https://github.com/f25252525252/ComfyUI-SUPIR.git /opt/ComfyUI/custom_nodes/ComfyUI-SUPIR && \
    git clone https://github.com/ssitu/ComfyUI_UltimateSDUpscale --recursive /opt/ComfyUI/custom_nodes/ComfyUI_UltimateSDUpscale && \
    git clone https://github.com/cubiq/ComfyUI_ESSENTIALS.git /opt/ComfyUI/custom_nodes/ComfyUI_ESSENTIALS

RUN pip install --no-cache-dir \
    huggingface_hub \
    uvicorn \
    fastapi \
    einops \
    open_clip_torch \
    spandrel \
    scipy \
    pillow

WORKDIR /app
COPY download_models.sh entrypoint.sh run_usdu_batch.py /app/

RUN chmod +x /app/download_models.sh /app/entrypoint.sh

EXPOSE 8000
ENTRYPOINT ["/app/entrypoint.sh"]