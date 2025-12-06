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

# Copy requirements
COPY requirements_docker.txt .

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

# Install Ollama for SLM
RUN curl -fsSL https://ollama.ai/install.sh | sh || echo "Ollama install failed"

# Copy source code
COPY . .

# Pre-download models at build time
RUN python3 -c "import layoutparser as lp; lp.PaddleDetectionLayoutModel('lp://PubLayNet/ppyolov2_r50vd_dcn_365e/config')" || echo "Model pre-download skipped"

# Download Detectron2 PubLayNet model to avoid iopath caching issues
RUN mkdir -p /root/.detectron2/models && \
    curl -L -o /root/.detectron2/models/publaynet_faster_rcnn_R_50_FPN_3x.pth \
    "https://www.dropbox.com/s/dgy9c10wykk4lq4/model_final.pth?dl=1" || \
    echo "Detectron2 model download skipped"

# Pre-download Table Transformer (TATR) model from HuggingFace
RUN python3 -c "from transformers import AutoImageProcessor, TableTransformerForObjectDetection; \
    AutoImageProcessor.from_pretrained('microsoft/table-transformer-structure-recognition'); \
    TableTransformerForObjectDetection.from_pretrained('microsoft/table-transformer-structure-recognition')" || \
    echo "TATR model pre-download skipped"

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

# Create startup script
RUN printf '#!/bin/bash\n\
set -e\n\
\n\
# Start Ollama in background if available\n\
if command -v ollama > /dev/null 2>&1; then\n\
    echo "Starting Ollama server..."\n\
    nohup ollama serve > /tmp/ollama.log 2>&1 &\n\
    sleep 5\n\
    echo "Pulling qwen2.5:7b-instruct (SLM) model..."\n\
    ollama pull qwen2.5:7b-instruct 2>&1 || echo "SLM model pull failed"\n\
    if [ "$ENABLE_VLM" = "true" ]; then\n\
        echo "Pulling qwen2-vl:7b (VLM) model..."\n\
        ollama pull qwen2-vl:7b 2>&1 || echo "VLM model pull skipped"\n\
    fi\n\
    echo "Ollama ready"\n\
else\n\
    echo "Ollama not installed, SLM/VLM features disabled"\n\
fi\n\
\n\
echo "Starting Streamlit on port 8501..."\n\
exec streamlit run app/streamlit_main.py --server.address 0.0.0.0 --server.port 8501\n\
' > /app/start.sh && chmod +x /app/start.sh

# Default command
CMD ["/bin/bash", "/app/start.sh"]
