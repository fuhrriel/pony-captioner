FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_BREAK_SYSTEM_PACKAGES=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.12 \
    python3-pip \
    git \
    wget \
    libgl1 \
    libglx-mesa0 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch with CUDA 12.6 support first
RUN pip3 install --no-cache-dir \
    torch==2.8.0 torchvision --index-url https://download.pytorch.org/whl/cu128

# Install other Python dependencies
RUN pip3 install --no-cache-dir \
    ninja \
    packaging \
    numpy==2.0.2 \
    transformers==4.44.2 \
    rich \
    pandas \
    onnxruntime-gpu==1.22 \
    git+https://github.com/openai/CLIP.git \
    huggingface-hub \
    Pillow \
    timm

RUN pip3 install --no-cache-dir \
    https://github.com/InternLM/lmdeploy/releases/download/v0.10.1/lmdeploy-0.10.1+cu128-cp312-cp312-manylinux2014_x86_64.whl

RUN pip3 install --no-cache-dir \
    https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl

# Create working directory
WORKDIR /workspace

# Copy the scripts
COPY pony_captioner.py /workspace/
COPY test_gpu.py /workspace/

# Set HuggingFace cache directory
ENV HF_HOME=/models/huggingface_cache
ENV TRANSFORMERS_CACHE=/models/huggingface_cache

# Create directories
RUN mkdir -p /workspace/images /models

# Default command
ENTRYPOINT ["python3", "/workspace/pony_captioner.py"]
CMD ["--help"]
