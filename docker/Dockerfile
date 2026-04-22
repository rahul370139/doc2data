# Use the LATEST NVIDIA PyTorch container (25.01+ supports Blackwell/GB10)
FROM nvcr.io/nvidia/pytorch:25.01-py3

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    tesseract-ocr \
    poppler-utils \
    git \
    cmake \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements from docker folder (build context is project root)
COPY docker/requirements_docker.txt .

# Fix NumPy version FIRST to avoid compatibility issues
RUN pip install "numpy<2" --force-reinstall

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements_docker.txt || \
    pip install --no-cache-dir --ignore-installed -r requirements_docker.txt

# Install Detectron2 from source
RUN pip install 'git+https://github.com/facebookresearch/detectron2.git'

# Install PaddlePaddle
RUN pip install paddlepaddle || echo "PaddlePaddle install failed, continuing..."

# Re-enforce NumPy 1.x
RUN pip install "numpy<2" --force-reinstall

# Install Ollama binary with robust fallback.
# We intentionally FAIL the build if ``ollama`` is unavailable so we
# never ship a container where VLM/SLM silently disable themselves.
RUN apt-get update -qq && apt-get install -y -qq zstd && rm -rf /var/lib/apt/lists/* && \
    ARCH=$(uname -m) && \
    if [ "$ARCH" = "aarch64" ] || [ "$ARCH" = "arm64" ]; then OLLAMA_ARCH="arm64"; else OLLAMA_ARCH="amd64"; fi && \
    echo "Downloading Ollama for ${OLLAMA_ARCH}..." && \
    (curl -fsSL -L -o /tmp/ollama.tar.zst "https://github.com/ollama/ollama/releases/download/v0.17.4/ollama-linux-${OLLAMA_ARCH}.tar.zst" && \
     cd /tmp && zstd -d ollama.tar.zst -o ollama.tar && tar -xf ollama.tar && \
     cp bin/ollama /usr/local/bin/ollama && chmod +x /usr/local/bin/ollama) || \
    (echo "Tarball install failed; trying official install.sh fallback..." && \
     curl -fsSL https://ollama.com/install.sh | sh) && \
    rm -f /tmp/ollama.tar.zst /tmp/ollama.tar /tmp/bin/ollama && \
    test -x /usr/local/bin/ollama && /usr/local/bin/ollama --version

# Copy source code
COPY . .

# TrOCR download DISABLED - models are loaded from mounted HuggingFace cache volume
# The cache is mounted at runtime: -v /home/radiant-dgx2/.cache/huggingface:/root/.cache/huggingface
# Models download on first use if not in cache. To re-enable build-time download, uncomment below:
# RUN USE_TF=0 python3 -c "\
# from transformers import TrOCRProcessor, VisionEncoderDecoderModel; \
# print('Downloading TrOCR-large-handwritten (safetensors)...'); \
# TrOCRProcessor.from_pretrained('microsoft/trocr-large-handwritten'); \
# VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-large-handwritten', use_safetensors=True); \
# print('Downloading TrOCR-base-handwritten (safetensors)...'); \
# TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten'); \
# VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten', use_safetensors=True); \
# print('TrOCR models cached successfully')"

# Florence-2-large is PRE-DOWNLOADED on DGX host (not during Docker build).
# Run `python3 scripts/download_florence2.py` on DGX first, or deploy_and_run.sh does it automatically.
# The model is loaded from mounted HuggingFace cache: /root/.cache/huggingface

# Optional model pre-download at build time (disabled by default)
ARG PRELOAD_MODELS=false
RUN if [ "$PRELOAD_MODELS" = "true" ]; then \
    python3 -c "import layoutparser as lp; print('Downloading Detectron2 model...'); lp.Detectron2LayoutModel('lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config')" || echo "Detectron2 model pre-download skipped"; \
    python3 -c "import layoutparser as lp; print('Downloading PaddleDetection model...'); lp.PaddleDetectionLayoutModel('lp://PubLayNet/ppyolov2_r50vd_dcn_365e/config')" || echo "PaddleDetection model pre-download skipped"; \
    mkdir -p /root/.detectron2/models && \
    curl -L -o /root/.detectron2/models/publaynet_faster_rcnn_R_50_FPN_3x.pth \
    "https://www.dropbox.com/s/dgy9c10wykk4lq4/model_final.pth?dl=1" || echo "Detectron2 model download skipped"; \
    USE_TF=0 python3 -c "from transformers import AutoImageProcessor, TableTransformerForObjectDetection; \
        AutoImageProcessor.from_pretrained('microsoft/table-transformer-structure-recognition'); \
        TableTransformerForObjectDetection.from_pretrained('microsoft/table-transformer-structure-recognition')" || echo "TATR model pre-download skipped"; \
else \
    echo "Skipping optional model pre-download (PRELOAD_MODELS=false)"; \
fi

# Create data directories
RUN mkdir -p /app/data/schemas /app/data/corrections

# Expose ports
EXPOSE 8501 8000 11434

# Environment variables
ENV USE_GPU=true
ENV CUDA_VISIBLE_DEVICES=0
ENV OLLAMA_HOST=localhost:11434
ENV ENABLE_SLM=true
ENV ENABLE_VLM=true
ENV TATR_MODEL_PATH=microsoft/table-transformer-structure-recognition
ENV DISABLE_MODEL_SOURCE_CHECK=true
ENV OLLAMA_PULL_ON_START=false
# CRITICAL: Prevent broken TF/ml_dtypes from crashing transformers TrOCR import
ENV USE_TF=0
ENV TF_CPP_MIN_LOG_LEVEL=3
# HuggingFace cache (mount volume in docker run for persistence)
ENV HF_HOME=/root/.cache/huggingface

# Copy and prepare startup script
COPY docker/start_services.sh /app/start_services.sh
RUN chmod +x /app/start_services.sh

# Health check to ensure services are running
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/ || curl -f http://localhost:8501/ || exit 1

# Default command
CMD ["/bin/bash", "/app/start_services.sh"]
